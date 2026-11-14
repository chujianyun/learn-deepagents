"""网页端智能体评测 —— openevals + LangSmith evaluate

评测链路（与网页端完全一致）：
    评测数据集 → POST /api/chat（SSE）→ 采集回答 + 工具轨迹 → openevals 评分 → LangSmith 报告

评测维度：
    1. correctness        回答正确性（LLM-as-judge，需参考答案）
    2. conciseness        简洁性（LLM-as-judge，无参考）
    3. no_hallucination   幻觉检查（LLM-as-judge，以联网搜索结果为事实依据，仅搜索用例）
    4. tool_selection     工具选择合理性（LLM-as-judge，仅涉及工具的用例）
    5. trajectory_match   工具调用匹配（规则：superset 模式，只比工具名不比参数）
    6. search_used        联网开关是否真正触发 tavily_search（规则，零成本）
    7. knowledge_retention 多轮对话知识保留（LLM-as-judge，仅多轮用例）

前置条件：
    - 服务已启动：.venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8000
    - .env 中已配置 OPENAI_*（裁判模型复用 SiliconFlow）与 LANGSMITH_*

运行：.venv/bin/python eval_agent.py
结果自动上报 LangSmith（实验名前缀 web-agent），并在终端打印汇总表。
"""

import json
import os
import time
from typing import Any

import httpx
import re
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langsmith import Client, traceable
from langsmith.evaluation import evaluate
from openevals.llm import create_llm_as_judge
from openevals.prompts import (
    CONCISENESS_PROMPT,
    CORRECTNESS_PROMPT,
    HALLUCINATION_PROMPT,
    TOOL_SELECTION_PROMPT,
)
from openevals.trajectory import (
    create_trajectory_llm_as_judge,
    create_trajectory_match_evaluator,
)

load_dotenv()

BASE_URL = os.environ.get("EVAL_BASE_URL", "http://127.0.0.1:8000")
DATASET_NAME = "web-agent-eval-v1"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
CHAT_TIMEOUT = 300  # 单轮请求超时（秒），与 server.py 的模型 timeout 对齐

# 裁判模型：复用 SiliconFlow OpenAI 兼容端点。
# 必须通过 judge= 显式传 LangChain 模型实例（model= 只接受字符串且会走 OpenAI 官方端点），
# 且 SiliconFlow 仅支持 Chat Completions，需 use_responses_api=False。
# SiliconFlow 兼容要点（实测结论）：
# 1. 模型实例必须走 judge= 参数（model= 只接受字符串，会走 OpenAI 官方端点）；
# 2. 默认 json_schema 结构化输出在该端点不稳定（返回裸数字），function_calling 返回 None；
#    改用 json_mode + 系统指令显式约束键名/类型后稳定可用。
JUDGE_SYSTEM = (
    "你必须严格只输出一个 JSON 对象，且只包含两个键："
    "reasoning（字符串，中文评分理由）和 score（布尔值 true 或 false，禁止输出数字）。"
    "不要输出任何其他内容。"
)


def _normalize_judge_output(raw):
    """归一化裁判输出，兼容 SiliconFlow json_mode 偶发的错键/裸值/数字分数。

    openevals 内部读 response["score"]/["reasoning"]，模型偶尔返回
    {"reason": ...}、裸数字或缺键，这里统一补齐并规范化类型，避免 KeyError。
    """
    if not isinstance(raw, dict):
        score = bool(raw) if isinstance(raw, (int, float, bool)) else False
        return {"score": score, "reasoning": str(raw)}
    out = dict(raw)
    if "score" not in out:
        for k in ("value", "result", "label", "answer", "correct"):
            if k in out:
                out["score"] = out[k]
                break
    if "reasoning" not in out:
        for k in ("reason", "comment", "explanation", "rationale"):
            if k in out:
                out["reasoning"] = out[k]
                break
    out.setdefault("score", False)
    out.setdefault("reasoning", "")
    s = out["score"]
    if isinstance(s, str):
        out["score"] = s.strip().lower() in ("true", "yes", "1", "正确", "是", "pass")
    elif isinstance(s, (int, float)):
        out["score"] = bool(s)
    if not isinstance(out["reasoning"], str):
        out["reasoning"] = str(out["reasoning"])
    return out


def _salvage_judge_text(text: str):
    """JSON 解析失败时的兜底：从原始文本提取 score/reasoning。

    SiliconFlow json_mode 偶发在 reasoning 里嵌入未转义引号导致 JSON 非法，
    此时 score 字段通常仍完好，用正则提取，避免整条评测因格式问题作废。
    """
    m = re.search(r'"score"\s*:\s*(true|false|"[^"]*"|\d+(?:\.\d+)?)', text, re.I)
    score = False
    if m:
        v = m.group(1).strip('"').lower()
        score = v in ("true", "yes", "1", "正确", "是", "pass") or (
            v.replace(".", "", 1).isdigit() and float(v) > 0
        )
    rm = re.search(r'"reasoning"\s*:\s*"(.*?)"\s*,\s*"score"', text, re.S)
    reasoning = rm.group(1) if rm else text[:500]
    return {"score": score, "reasoning": reasoning}


class JsonModeJudge(BaseChatModel):
    """包装裁判模型：强制 with_structured_output 走 json_mode 并归一化输出。

    openevals 0.2.5 的 create_trajectory_llm_as_judge 不透传
    structured_output_kwargs，通过包装层统一兼容，避免各端点默认
    json_schema 模式在 SiliconFlow 上返回裸数字/None 的问题；
    输出再经 _normalize_judge_output 补齐键名/类型，消除偶发 KeyError；
    JSON 本身非法（如 reasoning 内嵌未转义引号）时用 _salvage_judge_text 兜底。
    """

    model: BaseChatModel

    @property
    def _llm_type(self) -> str:
        return "json-mode-judge"

    def _generate(self, *args, **kwargs):
        return self.model._generate(*args, **kwargs)

    def with_structured_output(self, schema, **kwargs):
        kwargs.setdefault("method", "json_mode")
        inner = self.model.with_structured_output(schema, **kwargs)

        def _invoke_with_salvage(inp, config=None, **kw):
            try:
                return inner.invoke(inp, config, **kw)
            except OutputParserException as e:
                raw = getattr(e, "llm_output", None) or ""
                if isinstance(raw, AIMessage):
                    raw = raw.content
                return _salvage_judge_text(str(raw))

        return RunnableLambda(_invoke_with_salvage) | RunnableLambda(
            _normalize_judge_output
        )


judge_model = JsonModeJudge(model=init_chat_model(
    f"openai:{DEFAULT_MODEL}",
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
    use_responses_api=False,
    temperature=0,
    timeout=120,
    max_retries=1,
))

# deepagents 内置文件/任务工具不算外部工具（skills 读取属于正常行为）
INTERNAL_TOOLS = {
    "write_todos", "ls", "read_file", "write_file", "edit_file",
    "glob", "grep", "task", "think_tool", "execute",
}


# ---------- 1. target：驱动真实网页端链路，采集回答 + 工具轨迹 ----------

def _chat_once(history: list[dict], model: str, search: bool) -> dict:
    """调用一次 /api/chat，解析 SSE 事件流，返回本轮回答与工具调用。"""
    answer_parts: list[str] = []
    tools: dict[str, dict] = {}  # tool_call_id -> {name, query, sources, content}
    notices: list[str] = []
    done: dict[str, Any] = {}
    error = ""
    with httpx.Client(timeout=CHAT_TIMEOUT) as client:
        with client.stream(
            "POST",
            f"{BASE_URL}/api/chat",
            json={"messages": history, "model": model, "search": search},
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                evt = json.loads(line[6:])
                etype = evt.get("type")
                if etype == "delta":
                    answer_parts.append(evt.get("text", ""))
                elif etype == "tool_start":
                    tools[evt["id"]] = {
                        "name": evt.get("name", ""),
                        "query": evt.get("query", ""),
                        "sources": [],
                        "content": "",
                    }
                elif etype == "tool_end":
                    if evt.get("id") in tools:
                        tools[evt["id"]]["sources"] = evt.get("sources", [])
                        tools[evt["id"]]["content"] = evt.get("content", "")
                elif etype == "notice":
                    notices.append(evt.get("message", ""))
                elif etype == "done":
                    done = evt
                elif etype == "error":
                    error = evt.get("message", "unknown error")
    return {
        "answer": "".join(answer_parts),
        "tools": list(tools.values()),
        "notices": notices,
        "done": done,
        "error": error,
    }


def run_agent(inputs: dict) -> dict:
    """LangSmith evaluate 的 target 函数。

    inputs 支持两种形式：
      - 单轮：{"question": "...", "search": bool, "model": str}
      - 多轮：{"turns": ["...", "..."], ...}（逐轮携带历史调用，模拟网页端连续对话）
    """
    turns = inputs.get("turns") or [inputs["question"]]
    model = inputs.get("model", DEFAULT_MODEL)
    search = bool(inputs.get("search", False))

    history: list[dict] = []
    trajectory: list[dict] = []
    tools_used: list[str] = []
    tools_available: list[str] = []
    search_context: list[str] = []
    notices: list[str] = []
    latency = 0.0
    fallback = False
    answer = ""

    for turn in turns:
        history.append({"role": "user", "content": turn})
        trajectory.append({"role": "user", "content": turn})
        started = time.perf_counter()
        result = _chat_once(history, model, search)
        latency += time.perf_counter() - started
        notices.extend(result["notices"])
        if result["error"]:
            # 出错不中断整个实验：记录错误，评分器会给出 0 分
            return {
                "answer": "", "trajectory": trajectory, "tools_used": tools_used,
                "tools_available": tools_available,
                "search_context": "", "error": result["error"],
                "latency_s": round(latency, 2), "fallback": fallback,
                "notices": notices, "model": model,
            }
        answer = result["answer"]
        if result["tools"]:
            trajectory.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": f"call_{i}",
                        "function": {
                            "name": t["name"],
                            "arguments": json.dumps({"query": t["query"]}, ensure_ascii=False),
                        },
                    }
                    for i, t in enumerate(result["tools"])
                ],
            })
            for i, t in enumerate(result["tools"]):
                tools_used.append(t["name"])
                # 优先用搜索正文摘要作为事实依据（含标题/链接/摘要），
                # 没有正文时退回来源标题+链接
                if t.get("content"):
                    search_context.append(t["content"])
                else:
                    for s in t["sources"]:
                        search_context.append(f"{s.get('title', '')}: {s.get('url', '')}")
                trajectory.append({
                    "role": "tool",
                    "tool_call_id": f"call_{i}",
                    "content": f"[{t['name']}] 调用完成",
                })
        trajectory.append({"role": "assistant", "content": answer})
        history.append({"role": "assistant", "content": answer})
        fallback = fallback or bool(result["done"].get("fallback"))
        tools_available = result["done"].get("tools") or tools_available

    return {
        "answer": answer,
        "trajectory": trajectory,
        "tools_used": tools_used,
        "tools_available": tools_available,
        "search_context": "\n".join(search_context),
        "error": "",
        "latency_s": round(latency, 2),
        "fallback": fallback,
        "notices": notices,
        "model": model,
    }


# ---------- 2. openevals 评分器 ----------

correctness_judge = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT, judge=judge_model, feedback_key="correctness",
    system=JUDGE_SYSTEM)
conciseness_judge = create_llm_as_judge(
    prompt=CONCISENESS_PROMPT, judge=judge_model, feedback_key="conciseness",
    system=JUDGE_SYSTEM)
hallucination_judge = create_llm_as_judge(
    prompt=HALLUCINATION_PROMPT, judge=judge_model, feedback_key="no_hallucination",
    system=JUDGE_SYSTEM)
tool_selection_judge = create_trajectory_llm_as_judge(
    prompt=TOOL_SELECTION_PROMPT, judge=judge_model, feedback_key="tool_selection",
    system=JUDGE_SYSTEM)
retention_judge = create_llm_as_judge(
    prompt=(
        "你是对话质量评审员。下面是一段多轮对话的完整记录，以及助手对最后一轮用户问题的回答。\n"
        "请判断回答是否正确保留并使用了此前对话中用户提供的信息（如名称、时间、偏好等），"
        "没有遗忘、张冠李戴或编造。\n\n"
        "<conversation>\n{inputs}\n</conversation>\n\n"
        "<final_answer>\n{outputs}\n</final_answer>"
    ),
    judge=judge_model,
    feedback_key="knowledge_retention",
    system=JUDGE_SYSTEM,
)
# 工具序列匹配：superset=实际调用需包含期望工具；ignore=只比工具名不比参数
trajectory_matcher = create_trajectory_match_evaluator(
    trajectory_match_mode="superset", tool_args_match_mode="ignore")


def _question(inputs: dict) -> str:
    turns = inputs.get("turns")
    return turns[-1] if turns else inputs.get("question", "")


def _judge_context(inputs: dict, outputs: dict) -> str:
    """为裁判补充智能体能力边界与搜索事实依据，减少因信息不足导致的误判。

    - 可用工具清单：避免裁判把真实 MCP 能力（如点瑞幸咖啡）误判为吹牛；
    - 搜索正文：让裁判基于搜索事实核实回答，而非仅凭自身内置知识否定新信息。
    """
    parts = [_question(inputs)]
    tools = outputs.get("tools_available") or []
    if tools:
        parts.append(
            f"[智能体本轮可用的外部工具：{', '.join(tools)}。"
            "请据此判断其声称的能力是否真实，不要否定已配置工具对应的能力]"
        )
    if outputs.get("search_context"):
        parts.append(
            "[联网搜索获取的事实依据如下，请优先据此核实回答中的事实，"
            "不要仅凭你的内置知识否定搜索结果中的新信息]\n" + outputs["search_context"]
        )
    return "\n\n".join(parts)


# 参考链接/来源小节是搜索任务的硬性要求（提示词明确要求“结尾列参考链接”），
# 属于必要的引用凭证而非正文冗余。评简洁性时公平剥离该 footer，只衡量正文本身。
# 模型可能用“参考链接：”独占行，也可能用“参考：url...”同行内联，两者都需覆盖。
_REF_MARKERS = ("参考链接", "参考资料", "参考来源", "参考文献", "参考", "references")


def _strip_references(text: str) -> str:
    """去掉末尾的参考链接/来源小节（“参考链接：”独占行或“参考：url...”同行）。"""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        s = line.strip().lstrip("*#").strip()
        for m in _REF_MARKERS:
            if s.startswith(m):
                rest = s[len(m):].lstrip()
                # 标记后紧跟冒号（同行内联）或独占一行，均视为引用 footer 起点
                if rest == "" or rest.startswith(("：", ":")):
                    return "\n".join(lines[:i]).rstrip()
    return text.rstrip()


def _skip(key: str) -> dict:
    return {"key": key, "score": None, "comment": "本用例不适用该维度"}


def _failed(outputs: dict, key: str) -> dict | None:
    if outputs.get("error"):
        return {"key": key, "score": False, "comment": f"请求失败：{outputs['error']}"}
    return None


@traceable(name="eval_correctness")
def eval_correctness(inputs: dict, outputs: dict, reference_outputs: dict):
    if (err := _failed(outputs, "correctness")) is not None:
        return err
    expected = (reference_outputs or {}).get("expected_answer", "")
    if not expected:
        return _skip("correctness")
    return correctness_judge(
        inputs=_judge_context(inputs, outputs), outputs=outputs["answer"], reference_outputs=expected)


@traceable(name="eval_conciseness")
def eval_conciseness(inputs: dict, outputs: dict):
    if (err := _failed(outputs, "conciseness")) is not None:
        return err
    # 剥离硬性要求的参考链接 footer，只评正文简洁性（避免“要求附链接”与“简洁”互相矛盾）
    body = _strip_references(outputs["answer"])
    return conciseness_judge(inputs=_question(inputs), outputs=body)


@traceable(name="eval_no_hallucination")
def eval_no_hallucination(inputs: dict, outputs: dict):
    if (err := _failed(outputs, "no_hallucination")) is not None:
        return err
    if not outputs.get("search_context"):
        return _skip("no_hallucination")  # 无联网事实依据时不评幻觉
    # 只核正文事实是否被搜索内容支持；参考链接 footer 是硬性引用凭证，
    # 其 URL 是否与搜索来源逐字一致属于“引用准确性”另一维度，不在幻觉评测范围
    return hallucination_judge(
        context=outputs["search_context"],
        inputs=_question(inputs),
        outputs=_strip_references(outputs["answer"]),
        reference_outputs="",
    )


@traceable(name="eval_tool_selection")
def eval_tool_selection(inputs: dict, outputs: dict):
    if (err := _failed(outputs, "tool_selection")) is not None:
        return err
    # 仅在「开启联网」或「实际调用了外部工具」时评工具选择，
    # 避免裁判在不知道工具可用性的情况下误判纯聊天用例
    external = [t for t in outputs["tools_used"] if t not in INTERNAL_TOOLS]
    if not inputs.get("search") and not external:
        return _skip("tool_selection")
    return tool_selection_judge(outputs=outputs["trajectory"])


@traceable(name="eval_trajectory_match")
def eval_trajectory_match(inputs: dict, outputs: dict, reference_outputs: dict):
    if (err := _failed(outputs, "trajectory_match")) is not None:
        return err
    expected_tools = (reference_outputs or {}).get("expected_tools")
    if not expected_tools:
        return _skip("trajectory_match")
    reference = [{
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"type": "function", "id": f"ref_{i}",
             "function": {"name": name, "arguments": "{}"}}
            for i, name in enumerate(expected_tools)
        ],
    }]
    result = trajectory_matcher(outputs=outputs["trajectory"], reference_outputs=reference)
    return {**result, "key": "trajectory_match"}


@traceable(name="eval_search_used")
def eval_search_used(inputs: dict, outputs: dict):
    """规则评分：要求联网的用例必须真的调用了 tavily_search（零模型成本）。"""
    if (err := _failed(outputs, "search_used")) is not None:
        return err
    if not inputs.get("search"):
        return _skip("search_used")
    used = "tavily_search" in outputs["tools_used"]
    return {
        "key": "search_used",
        "score": used,
        "comment": f"实际工具调用：{outputs['tools_used'] or '无'}",
    }


@traceable(name="eval_knowledge_retention")
def eval_knowledge_retention(inputs: dict, outputs: dict):
    if (err := _failed(outputs, "knowledge_retention")) is not None:
        return err
    if not inputs.get("turns") or len(inputs["turns"]) < 2:
        return _skip("knowledge_retention")
    lines = []
    for msg in outputs["trajectory"]:
        role = msg["role"]
        if role == "tool":
            continue
        content = msg.get("content") or ""
        if msg.get("tool_calls"):
            names = [tc["function"]["name"] for tc in msg["tool_calls"]]
            content = f"[调用工具：{', '.join(names)}]"
        lines.append(f"{role}: {content}")
    return retention_judge(inputs="\n".join(lines), outputs=outputs["answer"])


# ---------- 3. 评测数据集（9 条，覆盖六大维度） ----------

EXAMPLES: list[dict] = [
    {
        "inputs": {"question": "用一句话介绍你自己。"},
        "outputs": {"expected_answer": "说明自己是一个中文 AI 助手，可以回答问题、提供帮助。"},
    },
    {
        "inputs": {"question": "中国的首都是哪座城市？"},
        "outputs": {"expected_answer": "北京。"},
    },
    {
        "inputs": {"question": "17 乘以 23 等于多少？只回答数字。"},
        "outputs": {"expected_answer": "391"},
    },
    {
        "inputs": {"question": "DeepSeek 最近发布的新模型是什么？", "search": True},
        "outputs": {
            "expected_answer": "基于联网搜索结果回答，给出具体模型名称，并在结尾附参考链接。",
            "expected_tools": ["tavily_search"],
        },
    },
    {
        "inputs": {"question": "Python 目前最新的稳定版本号是多少？", "search": True},
        "outputs": {
            "expected_answer": "基于联网搜索结果给出 Python 最新稳定版本号，并附参考链接。",
            "expected_tools": ["tavily_search"],
        },
    },
    {
        "inputs": {"question": "杭州西湖需要门票吗？景区开放时间是什么？", "search": True},
        "outputs": {
            "expected_answer": "基于联网搜索结果说明西湖景区门票与开放情况，并附参考链接。",
            "expected_tools": ["tavily_search"],
        },
    },
    {
        "inputs": {"turns": [
            "记住：我的项目代号叫「星尘」，计划下周三上线。",
            "帮我写一句适合周报的项目进展开场白。",
            "我的项目代号是什么？计划什么时候上线？",
        ]},
        "outputs": {"expected_answer": "项目代号是「星尘」，计划下周三上线。"},
    },
    {
        "inputs": {"question": "用不超过三句话解释机器学习、深度学习和大模型三者的关系。"},
        "outputs": {
            "expected_answer": "深度学习是机器学习的子集，大模型是基于深度学习（Transformer）的超大参数模型。"},
    },
    {
        "inputs": {"question": "写一首关于秋天的五言绝句。"},
        "outputs": {"expected_answer": "一首四句、每句五字的原创古诗，内容与秋天相关。"},
    },
]


def ensure_dataset(client: Client) -> None:
    """数据集不存在时创建并写入用例（已存在则复用，便于对比多次实验）。"""
    if client.has_dataset(dataset_name=DATASET_NAME):
        print(f"复用已有数据集：{DATASET_NAME}")
        return
    ds = client.create_dataset(
        DATASET_NAME, description="网页端智能体评测集 v1：正确性/简洁/幻觉/工具轨迹/联网/多轮")
    client.create_examples(dataset_id=ds.id, examples=EXAMPLES)
    print(f"已创建数据集：{DATASET_NAME}（{len(EXAMPLES)} 条用例）")


# ---------- 4. 执行评测并汇总 ----------

def print_summary(scores: dict[str, list[float]], errors: list[str], experiment: str) -> None:
    print("\n========== 评测汇总 ==========")
    print(f"{'维度':<22}{'平均分':>8}{'通过/适用':>12}")
    for key, vals in scores.items():
        passed = sum(1 for v in vals if v >= 1.0)
        print(f"{key:<22}{sum(vals) / len(vals):>8.2f}{passed:>6}/{len(vals):<5}")
    if errors:
        print("\n失败用例：")
        for e in errors:
            print(f"  - {e[:200]}")
    print(f"\n实验名：{experiment}")
    print("逐条得分与裁判评语可在 LangSmith 对应实验页查看（trace 落库约有 10 秒延迟）。")


def report_experiment(experiment: str) -> None:
    """复盘已有实验：从 LangSmith 拉取 feedback 并打印汇总（不重新调用智能体）。"""
    client = Client()
    scores: dict[str, list[float]] = {}
    for f in client.list_feedback(project_name=experiment):
        if f.score is not None:
            scores.setdefault(f.key, []).append(float(f.score))
    print_summary(scores, [], experiment)


def main() -> None:
    try:
        httpx.get(f"{BASE_URL}/api/config", timeout=5).raise_for_status()
    except Exception:
        raise SystemExit(
            f"服务未启动或不可达：{BASE_URL}\n"
            "请先运行：.venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8000"
        )

    client = Client()
    ensure_dataset(client)

    print(f"开始评测（{len(EXAMPLES)} 条用例，max_concurrency=2）…")
    results = evaluate(
        run_agent,
        data=DATASET_NAME,
        evaluators=[
            eval_correctness,
            eval_conciseness,
            eval_no_hallucination,
            eval_tool_selection,
            eval_trajectory_match,
            eval_search_used,
            eval_knowledge_retention,
        ],
        experiment_prefix="web-agent",
        max_concurrency=2,  # 控制 SiliconFlow 并发，避免限流
    )

    # 汇总各维度得分（score=None 表示该用例不适用，不计入）
    # 注意：ExperimentResultRow / EvaluationResults 均为 TypedDict，需用下标访问
    scores: dict[str, list[float]] = {}
    errors: list[str] = []
    for item in results:
        run = item["run"]
        if getattr(run, "error", None):
            errors.append(str(run.error))
        for r in item["evaluation_results"].get("results", []):
            if r.score is None:
                continue
            scores.setdefault(r.key, []).append(float(r.score))

    print_summary(scores, errors, results.experiment_name)


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3 and sys.argv[1] == "--report":
        report_experiment(sys.argv[2])
    else:
        main()