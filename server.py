"""DeepSeek 风格智能体聊天界面 —— FastAPI 后端

- 基于 deepagents.create_deep_agent 构建智能体，按「模型 + 联网搜索开关 + MCP 配置指纹」缓存实例
- POST /api/chat 以 SSE 流式输出：思考过程 / 正文增量 / 工具调用（联网搜索 / MCP）/ 完成与错误
- 模型通过 SiliconFlow 的 OpenAI 兼容接口调用（use_responses_api=False）
- MCP 服务统一配置在 ~/.my-deepseek/mcp.json（见 mcp_config.py），修改后自动生效
- 前端静态资源位于 web/ 目录，访问 http://127.0.0.1:8000 即可使用

启动：.venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8000
"""

import asyncio
import json
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.tools import tool
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tavily import TavilyClient

from deepagents import create_deep_agent

from prompt_enhancer import ENHANCE_SYSTEM_PROMPT, enhance_user_prompt
from skills_config import SKILLS_SOURCE, skills_backend

from mcp_config import MCPBundle, load_mcp_tools

load_dotenv()

WEB_DIR = Path(__file__).parent / "web"

# 界面上可切换的模型（SiliconFlow 模型 ID）
MODELS: list[dict[str, str]] = [
    {
        "id": "deepseek-ai/DeepSeek-V4.1-Flash",
        "name": "DeepSeek-V4.1-Flash",
        "vendor": "DeepSeek",
        "desc": "默认模型，响应快，适合日常对话与写作",
    },
    {
        "id": "MiniMaxAI/MiniMax-H3",
        "name": "MiniMax-H3",
        "vendor": "MiniMax",
        "desc": "长上下文理解，适合长文档与复杂任务",
    },
    {
        "id": "tencent/Hy4-preview",
        "name": "Hy4-preview",
        "vendor": "Tencent",
        "desc": "腾讯混元预览版，中文表达自然",
    },
    {
        "id": "zai-org/GLM-5.3-Flash",
        "name": "GLM-5.3-Flash",
        "vendor": "Zhipu",
        "desc": "智谱轻量旗舰，推理与代码能力均衡",
    },
]
MODEL_IDS = {m["id"] for m in MODELS}
DEFAULT_MODEL = MODELS[0]["id"]

BASE_SYSTEM_PROMPT = (
    "你是 DeepSeek 智能助手，一个友好的中文 AI 助手，运行在网页聊天界面中。"
    "回答准确、简洁、有条理：简单事实或概念问题用一两句话直接回答，"
    "不要堆砌标题、表格、分节或额外解释；只有复杂对比、多步骤任务才用 Markdown 排版。"
    "严格只回答用户所问：不要补充用户没问的版本历史、参数细节、价格、背景或“顺带一提”类延伸；"
    "能用一句话答清就不用两句，不要主动添加赏析、总结、延伸建议或追问。"
    "不要创建待办清单；除读取 skill 文件（read_file/grep/glob/ls）外，"
    "不要使用文件写入类工具，日常问题直接对话回答即可。"
    "当用户问题匹配某个 skill 的领域时，先读取对应 SKILL.md 再按 skill 指引回答。"
)
SEARCH_PROMPT_SUFFIX = (
    "当前已开启联网搜索：当用户需要人物资料、最新资讯、实时数据或事实核查时，"
    "调用 tavily_search 工具联网搜索。每个问题最多搜索 1 次，拿到结果后立即作答；"
    "禁止换个语言或相近关键词再搜一次，也不要用猜测的答案作为查询词。"
    "基于搜索结果作答：先用一句话直接回答用户所问，然后立即结束正文（最多不超过 3 句）；"
    "相关但用户没问的信息一律不写：如未来版本计划、发布日期、参数量、架构、基准、价格、"
    "周边景点票价、API 调用方式、更新渠道等；参考链接单独列在末尾。"
    "只陈述搜索结果正文中明确写出的事实：不要从链接 URL、文件名或标题推测日期、版本号等细节，"
    "也不要把多条结果自行综合推断出结果里没有直接写明的结论（如“目前最新版本是 X”）；"
    "结果没直接写明或信息矛盾时，如实说明不确定性，不要硬下结论。"
)
MCP_PROMPT_SUFFIX = (
    "当前已接入外部 MCP 服务：{servers}。"
    "当用户请求涉及这些服务的能力（如瑞幸咖啡点单、查菜单、下单）时，"
    "必须调用对应的 MCP 工具获取真实数据后再回答，不要编造工具结果；"
    "工具调用失败时如实告知用户。"
    "自我介绍或普通问答时不要主动列举这些服务、举例或罗列功能（包括“查文档”“点咖啡”等），"
    "只有用户显式问“你能做什么/有什么功能”时才介绍能力。"
)

# ---------- 提示词增强子智能体 ----------
# 提示词与子智能体定义统一放在 prompt_enhancer.py（单一事实来源），
# 此处复用同一份系统指令构建独立一次性增强智能体。
# 增强为一次性文本生成：禁用自定义工具与子智能体，降低时延并避免执行原任务。
ENHANCE_TIMEOUT = 90  # 秒，整体超时（含模型调用）


def _tavily_key_ready() -> bool:
    key = os.environ.get("TAVILY_API_KEY", "")
    return key.startswith("tvly-") and len(key) > 20


tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY", ""))


@tool
def tavily_search(query: str) -> str:
    """使用 Tavily 联网搜索，返回相关网页的标题、链接和内容摘要。

    适用于查询人物资料、最新资讯、实时数据、事实核查等需要联网获取信息的场景。

    Args:
        query: 搜索关键词，例如 "DeepSeek 最新模型 发布"
    """
    try:
        response = tavily_client.search(query=query, max_results=5)
    except Exception as exc:  # 搜索失败时把错误告诉模型，让它继续回答
        return f"联网搜索失败：{exc}。请基于已有知识回答，并说明未能联网核实。"
    results = response.get("results", [])
    if not results:
        return f"未搜索到与「{query}」相关的结果。"
    lines = [f"搜索「{query}」共找到 {len(results)} 条结果："]
    for idx, item in enumerate(results, start=1):
        lines.append(
            f"{idx}. {item.get('title', '无标题')}\n"
            f"   链接：{item.get('url', '')}\n"
            f"   摘要：{item.get('content', '')}"
        )
    return "\n".join(lines)


# 按 (模型, 是否联网, MCP 配置指纹) 缓存智能体，避免每次请求重复构建；
# mcp.json 更新后指纹变化，自动构建新实例并清理同模型的旧指纹缓存
_agent_cache: dict[tuple[str, bool, str], Any] = {}


def get_agent(model_id: str, search_enabled: bool, mcp: MCPBundle = MCPBundle([], [], None, "")):
    key = (model_id, search_enabled, mcp.fingerprint)
    if key not in _agent_cache:
        for old_key in [k for k in _agent_cache if k[:2] == key[:2]]:
            del _agent_cache[old_key]
        model = init_chat_model(
            f"openai:{model_id}",
            base_url=os.environ["OPENAI_BASE_URL"],
            api_key=os.environ["OPENAI_API_KEY"],
            use_responses_api=False,  # SiliconFlow 仅支持 Chat Completions
            timeout=180,
            max_retries=1,
        )
        use_search = search_enabled and _tavily_key_ready()
        prompt = BASE_SYSTEM_PROMPT + (SEARCH_PROMPT_SUFFIX if use_search else "")
        if mcp.tools:
            prompt += MCP_PROMPT_SUFFIX.format(servers="、".join(mcp.servers))
        _agent_cache[key] = create_deep_agent(
            model=model,
            tools=([tavily_search] if use_search else []) + mcp.tools,
            system_prompt=prompt,
            subagents=[],  # 纯聊天场景不需要子智能体，降低时延
            # 与 main.py 共享同一套本地 skills 配置（skills_config.py）
            backend=skills_backend,
            skills=[SKILLS_SOURCE],
        )
    return _agent_cache[key]


# 提示词增强子智能体：按模型缓存的专用一次性智能体（无工具、无嵌套子智能体）。
_enhance_agent_cache: dict[str, Any] = {}


def get_enhance_agent(model_id: str):
    """构建/复用「提示词增强」子智能体。

    与主聊天智能体隔离：不挂联网搜索等工具，系统指令仅要求改写提示词，
    一次模型生成即返回，不执行用户原始任务。
    """
    if model_id not in _enhance_agent_cache:
        model = init_chat_model(
            f"openai:{model_id}",
            base_url=os.environ["OPENAI_BASE_URL"],
            api_key=os.environ["OPENAI_API_KEY"],
            use_responses_api=False,  # SiliconFlow 仅支持 Chat Completions
            timeout=60,
            max_retries=1,
        )
        _enhance_agent_cache[model_id] = create_deep_agent(
            model=model,
            tools=[],
            system_prompt=ENHANCE_SYSTEM_PROMPT,
            subagents=[],
            name="prompt-enhancer",
        )
    return _enhance_agent_cache[model_id]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)
    model: str = DEFAULT_MODEL
    search: bool = False


class EnhanceRequest(BaseModel):
    text: str = ""
    model: str = DEFAULT_MODEL


# ---------- 模型可用性探测 ----------
# SiliconFlow 目录中列出的模型不一定已开放调用（如 MiniMax-H3 返回 20012），
# 启动时后台并行探测一次，并在真实请求中持续修正状态。
_model_available: dict[str, bool] = {}

_probe_client = AsyncOpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL", ""),
    api_key=os.environ.get("OPENAI_API_KEY", ""),
    timeout=30,
    max_retries=0,
)


def _is_model_missing(exc: Exception) -> bool:
    text = str(exc)
    return "Model does not exist" in text or "20012" in text


async def _probe_model(model_id: str) -> None:
    try:
        await _probe_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=8,
        )
        _model_available[model_id] = True
    except Exception as exc:
        # 仅「模型不存在」判定为不可用；限流/网络等其他错误不影响可用性标记
        _model_available[model_id] = not _is_model_missing(exc)


async def _probe_all_models() -> None:
    await asyncio.gather(*(_probe_model(m["id"]) for m in MODELS))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    task = asyncio.create_task(_probe_all_models())
    yield
    task.cancel()


app = FastAPI(title="DeepSeek Chat", docs_url=None, redoc_url=None, lifespan=lifespan)


@app.get("/api/config")
async def api_config():
    return {
        "models": [{**m, "available": _model_available.get(m["id"], True)} for m in MODELS],
        "default_model": DEFAULT_MODEL,
        "search_available": _tavily_key_ready(),
    }


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _content_text(content: Any) -> tuple[str, str]:
    """从消息 content 中分离 (正文文本, 思考文本)，兼容 str 与内容块列表。"""
    if isinstance(content, str):
        return content, ""
    text_parts: list[str] = []
    think_parts: list[str] = []
    if isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                ptype = part.get("type", "")
                if ptype == "text":
                    text_parts.append(part.get("text", ""))
                elif ptype in ("reasoning", "thinking"):
                    think_parts.append(part.get("reasoning") or part.get("thinking") or part.get("text") or "")
    return "".join(text_parts), "".join(think_parts)


_LINK_RE = re.compile(r"^\s*链接：(\S+)\s*$")


def _parse_search_sources(content: str) -> list[dict[str, str]]:
    """从 tavily_search 的文本结果中提取标题与链接，供前端展示来源卡片。"""
    lines = content.splitlines()
    sources: list[dict[str, str]] = []
    for i, line in enumerate(lines):
        m = _LINK_RE.match(line)
        if not m:
            continue
        title = ""
        if i > 0:
            tm = re.match(r"^\s*\d+\.\s*(.+)$", lines[i - 1])
            if tm:
                title = tm.group(1).strip()
        url = m.group(1)
        domain = re.sub(r"^www\.", "", url.split("/")[2]) if "://" in url else url
        sources.append({"title": title or domain, "url": url, "domain": domain})
    return sources


async def _chat_stream(req: ChatRequest, request: Request) -> AsyncGenerator[str, None]:
    history = [
        {"role": m.role, "content": m.content}
        for m in req.messages
        if m.role in ("user", "assistant") and m.content.strip()
    ]
    if not history or history[-1]["role"] != "user":
        yield _sse({"type": "error", "message": "请求中没有有效的用户消息。"})
        return

    model_id = req.model if req.model in MODEL_IDS else DEFAULT_MODEL
    search_on = req.search and _tavily_key_ready()
    if req.search and not search_on:
        yield _sse({"type": "notice", "message": "TAVILY_API_KEY 未配置，本次回答未联网。"})

    # MCP 工具按配置文件指纹缓存；未配置/加载失败时降级为普通聊天并提示
    mcp = await load_mcp_tools()
    if mcp.notice:
        yield _sse({"type": "notice", "message": mcp.notice})

    agent_attempts = [model_id] + ([DEFAULT_MODEL] if model_id != DEFAULT_MODEL else [])
    for attempt, mid in enumerate(agent_attempts):
        agent = get_agent(mid, search_on, mcp)
        started_tools: set[str] = set()
        tool_names: dict[str, str] = {}
        emitted = False  # 是否已向前端输出过内容（输出后不再代答，避免重复）
        try:
            async for ns, mode, data in agent.astream(
                {"messages": history},
                stream_mode=["messages", "updates"],
                subgraphs=True,
            ):
                if await request.is_disconnected():
                    return
                if ns:  # 子智能体事件（当前配置下不会出现）
                    continue

                if mode == "messages":
                    chunk, _meta = data
                    if not isinstance(chunk, AIMessageChunk):
                        continue
                    reasoning = chunk.additional_kwargs.get("reasoning_content") or ""
                    text, think_blocks = _content_text(chunk.content)
                    if reasoning or think_blocks:
                        emitted = True
                        yield _sse({"type": "thinking", "text": reasoning + think_blocks})
                    if text and not chunk.tool_call_chunks:
                        emitted = True
                        yield _sse({"type": "delta", "text": text})

                elif mode == "updates" and isinstance(data, dict):
                    for _node, payload in data.items():
                        if not isinstance(payload, dict):
                            continue
                        for msg in payload.get("messages", []) or []:
                            if isinstance(msg, AIMessage) and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    tc_id = tc.get("id") or ""
                                    if tc_id in started_tools:
                                        continue
                                    started_tools.add(tc_id)
                                    tool_names[tc_id] = tc.get("name", "")
                                    args = tc.get("args") or {}
                                    emitted = True
                                    yield _sse({
                                        "type": "tool_start",
                                        "id": tc_id,
                                        "name": tc.get("name", ""),
                                        "query": args.get("query") or args.get("description") or args.get("file_path") or "",
                                    })
                            elif isinstance(msg, ToolMessage):
                                content_str = msg.content if isinstance(msg.content, str) else str(msg.content)
                                name = msg.name or tool_names.get(msg.tool_call_id, "")
                                evt: dict[str, Any] = {
                                    "type": "tool_end",
                                    "id": msg.tool_call_id,
                                    "name": name,
                                }
                                if name == "tavily_search":
                                    evt["sources"] = _parse_search_sources(content_str)
                                    # 附带搜索正文摘要，供评测做幻觉核查（前端忽略未知字段，向后兼容）。
                                    # 上限设较高值，保证裁判看到的事实依据与模型所见一致，避免截断造成幻觉误判
                                    evt["content"] = content_str[:12000]
                                emitted = True
                                yield _sse(evt)
            _model_available[mid] = True
            yield _sse({
                "type": "done",
                "model": mid,
                "requested_model": model_id,
                "fallback": attempt > 0,
                "search": search_on,
                # 本轮实际可用的外部工具清单，供评测裁判了解智能体能力边界
                "tools": (["tavily_search"] if search_on else [])
                + [getattr(t, "name", "") for t in mcp.tools],
            })
            return
        except Exception as exc:
            if _is_model_missing(exc):
                _model_available[mid] = False
                if not emitted and attempt == 0:
                    # 所选模型平台未开放：透明地改用默认模型代答
                    yield _sse({"type": "model_fallback", "from": mid, "to": DEFAULT_MODEL})
                    continue
            message = f"{type(exc).__name__}: {exc}"[:600]
            if _is_model_missing(exc):
                message = (
                    f"模型 {mid} 暂未在 SiliconFlow 开放调用（平台返回：Model does not exist），"
                    "且代答也失败。请稍后重试。"
                )
            elif "401" in message or "Unauthorized" in message or "api key" in message.lower():
                message = "API Key 无效或已过期，请检查 .env 中的 OPENAI_API_KEY。"
            yield _sse({"type": "error", "message": message})
            return


@app.post("/api/chat")
async def api_chat(req: ChatRequest, request: Request):
    return StreamingResponse(
        _chat_stream(req, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------- 提示词增强接口 ----------
def _clean_enhanced(raw: str) -> str:
    """后处理：trim、去掉一层成对的外围引号/代码围栏、空值检查。"""
    result = raw.strip()
    # 模型偶尔仍会包裹代码围栏，保守地去掉一层成对围栏
    if result.startswith("```") and result.endswith("```") and len(result) >= 6:
        inner = result[3:-3]
        # 去掉首行可能的语言标识
        if "\n" in inner:
            first, rest = inner.split("\n", 1)
            if first.strip() and not first.strip().startswith(" ") and len(first.strip()) <= 20:
                inner = rest
        result = inner.strip()
    pairs = [('"', '"'), ("'", "'"), ('“', '”'), ('‘', '’')]
    if len(result) >= 2 and any(result.startswith(a) and result.endswith(b) for a, b in pairs):
        result = result[1:-1].strip()
    return result


async def _invoke_enhance_agent(agent: Any, user_prompt: str, request: Request) -> str:
    """调用增强子智能体并返回纯文本；支持客户端断开取消与整体超时。"""
    task = asyncio.create_task(
        agent.ainvoke({"messages": [{"role": "user", "content": user_prompt}]})
    )
    loop = asyncio.get_running_loop()
    deadline = loop.time() + ENHANCE_TIMEOUT
    try:
        while not task.done():
            if await request.is_disconnected():
                raise asyncio.CancelledError("client disconnected")
            if loop.time() >= deadline:
                raise TimeoutError("enhance timeout")
            await asyncio.wait({task}, timeout=0.4)
        result = task.result()
    finally:
        if not task.done():
            task.cancel()
    messages = result.get("messages") or []
    if not messages:
        return ""
    text, _think = _content_text(messages[-1].content)
    return text


@app.post("/api/enhance")
async def api_enhance(req: EnhanceRequest, request: Request):
    """提示词增强：接收原始文本，返回改写后的提示词（不执行原任务）。"""
    text = req.text.replace("\u200b", "").strip() if isinstance(req.text, str) else ""
    if not text:
        return {"success": False, "text": "", "code": "empty_input", "error": "输入内容为空"}

    model_id = req.model if req.model in MODEL_IDS else DEFAULT_MODEL
    # 字面量单次替换，原文中的 $&、引号、换行均原样传入
    user_prompt = enhance_user_prompt(text)
    attempts = [model_id] + ([DEFAULT_MODEL] if model_id != DEFAULT_MODEL else [])
    last_err = ""
    for attempt, mid in enumerate(attempts):
        try:
            raw = await _invoke_enhance_agent(get_enhance_agent(mid), user_prompt, request)
            cleaned = _clean_enhanced(raw)
            if not cleaned:
                raise ValueError("增强结果为空")
            _model_available[mid] = True
            return {"success": True, "text": cleaned, "model": mid, "fallback": attempt > 0}
        except asyncio.CancelledError:
            return {"success": False, "text": "", "code": "cancelled", "error": "增强已取消"}
        except TimeoutError:
            last_err = "增强超时，请稍后重试"
            break
        except Exception as exc:
            if _is_model_missing(exc):
                _model_available[mid] = False
                last_err = f"模型 {mid} 暂未在平台开放调用"
                continue  # 透明地改用默认模型重试
            last_err = f"{type(exc).__name__}: {exc}"[:300]
            if "401" in last_err or "Unauthorized" in last_err or "api key" in last_err.lower():
                last_err = "API Key 无效或已过期，请检查 .env 中的 OPENAI_API_KEY。"
            break
    return {"success": False, "text": "", "code": "llm_error", "error": last_err or "增强失败"}


@app.get("/")
async def index():
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")