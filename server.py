"""DeepSeek 风格智能体聊天界面 —— FastAPI 后端

- 基于 deepagents.create_deep_agent 构建智能体，按「模型 + 联网搜索开关」缓存实例
- POST /api/chat 以 SSE 流式输出：思考过程 / 正文增量 / 工具调用（联网搜索）/ 完成与错误
- 模型通过 SiliconFlow 的 OpenAI 兼容接口调用（use_responses_api=False）
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
    "回答准确、简洁、有条理，善用 Markdown 排版（标题、列表、表格、代码块）。"
    "不要使用文件读写类工具，也不要创建待办清单，直接对话回答即可。"
)
SEARCH_PROMPT_SUFFIX = (
    "当前已开启联网搜索：当用户需要人物资料、最新资讯、实时数据或事实核查时，"
    "必须先调用 tavily_search 工具联网搜索，再基于搜索结果用简洁的中文回答，"
    "并在结尾列出参考链接。不要编造搜索结果中没有的信息。"
)


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


# 按 (模型, 是否联网) 缓存智能体，避免每次请求重复构建
_agent_cache: dict[tuple[str, bool], Any] = {}


def get_agent(model_id: str, search_enabled: bool):
    key = (model_id, search_enabled)
    if key not in _agent_cache:
        model = init_chat_model(
            f"openai:{model_id}",
            base_url=os.environ["OPENAI_BASE_URL"],
            api_key=os.environ["OPENAI_API_KEY"],
            use_responses_api=False,  # SiliconFlow 仅支持 Chat Completions
            timeout=180,
            max_retries=1,
        )
        use_search = search_enabled and _tavily_key_ready()
        _agent_cache[key] = create_deep_agent(
            model=model,
            tools=[tavily_search] if use_search else [],
            system_prompt=BASE_SYSTEM_PROMPT + (SEARCH_PROMPT_SUFFIX if use_search else ""),
            subagents=[],  # 纯聊天场景不需要子智能体，降低时延
        )
    return _agent_cache[key]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)
    model: str = DEFAULT_MODEL
    search: bool = False


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

    agent_attempts = [model_id] + ([DEFAULT_MODEL] if model_id != DEFAULT_MODEL else [])
    for attempt, mid in enumerate(agent_attempts):
        agent = get_agent(mid, search_on)
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
                                        "query": args.get("query") or args.get("description") or "",
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
                                emitted = True
                                yield _sse(evt)
            _model_available[mid] = True
            yield _sse({
                "type": "done",
                "model": mid,
                "requested_model": model_id,
                "fallback": attempt > 0,
                "search": search_on,
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


@app.get("/")
async def index():
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")