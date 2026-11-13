"""Deep Agents + Tavily 联网搜索示例

模型：deepseek-ai/DeepSeek-V4.1-Flash（SiliconFlow，OpenAI 兼容协议）
搜索：Tavily（deepagents 官方示例推荐的联网搜索工具）
Skills：主智能体挂载 /Users/wuming/Downloads/test/skills 下的本地 skill
配置：从 .env 读取 OPENAI_API_KEY、OPENAI_BASE_URL、TAVILY_API_KEY
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from tavily import TavilyClient

from deepagents import create_deep_agent

from prompt_enhancer import ENHANCE_SUBAGENT_NAME, build_enhance_subagent
from skills_config import SKILLS_DIR, SKILLS_SOURCE, skills_backend

# 加载 .env 中的 OPENAI_API_KEY / OPENAI_BASE_URL
load_dotenv()

# SiliconFlow 只支持 Chat Completions 接口，不支持 OpenAI Responses API，
# 因此必须显式构造模型实例并设置 use_responses_api=False，
# 不能直接传 "openai:xxx" 字符串（默认会走 Responses API 导致报错）。
model = init_chat_model(
    "openai:deepseek-ai/DeepSeek-V4.1-Flash",
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
    use_responses_api=False,
)


# Tavily 搜索客户端：Key 从 .env 的 TAVILY_API_KEY 读取（稍后自行填写真实 Key）
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))


@tool
def tavily_search(query: str) -> str:
    """使用 Tavily 联网搜索，返回相关网页的标题、链接和内容摘要。

    适用于查询人物资料、最新资讯、事实核查等需要联网获取信息的场景。

    Args:
        query: 搜索关键词，例如 "讲师 悟鸣 简介"
    """
    response = tavily_client.search(query=query, max_results=5)
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


agent = create_deep_agent(
    model=model,
    tools=[tavily_search],
    system_prompt=(
        "你是一个友好的中文助手。当用户需要人物资料、最新资讯或事实核查时，"
        "必须先调用 tavily_search 工具联网搜索，再基于搜索结果用简洁的中文回答，"
        "并在结尾列出参考链接。不要编造搜索结果中没有的信息。"
        f"你挂载了一个名为 {ENHANCE_SUBAGENT_NAME} 的子智能体，专门做提示词增强。"
        "当用户要求增强/优化/改写提示词，或希望把模糊需求变得更清晰可执行时，"
        f"必须调用 task 工具并把 subagent_type 设为 {ENHANCE_SUBAGENT_NAME}，"
        "在 description 中原样传入用户的原始提示词文本（不要自行改写或执行该任务）；"
        "拿到子智能体返回的增强提示词后，直接把它作为最终答复返回，"
        "不要添加额外解释，也不要执行原始任务。"
    ),
    # 把「提示词增强」作为声明式子智能体挂到主智能体，主智能体通过内置 task 工具按需委派
    subagents=[build_enhance_subagent()],
    # 仅主智能体加载本地 skills（子智能体不继承）；启动时解析各 SKILL.md 的
    # name/description 注入系统提示词，模型按需 read_file 读取完整指令
    # 配置与 server.py 共享，见 skills_config.py
    backend=skills_backend,
    skills=[SKILLS_SOURCE],
)


def _tavily_key_ready() -> bool:
    """判断 TAVILY_API_KEY 是否已填写为有效格式（tvly- 开头且长度足够）。"""
    key = os.environ.get("TAVILY_API_KEY", "")
    return key.startswith("tvly-") and len(key) > 20


def _demo_enhance(original: str) -> None:
    """演示主智能体通过 task 工具委派给 prompt-enhancer 子智能体。"""
    print(f"原始提示词：{original}\n")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": f"请帮我增强这段提示词：{original}"}]}
    )
    delegated = False
    for m in result["messages"]:
        for tc in getattr(m, "tool_calls", None) or []:
            if tc.get("name") == "task":
                args = tc.get("args") or {}
                delegated = True
                print(f"[委派] task -> subagent_type={args.get('subagent_type')}")
    print(f"\n是否委派给子智能体：{'是' if delegated else '否'}\n")
    print("增强后的提示词：")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "enhance":
        # 提示词增强演示（无需 TAVILY_API_KEY）：python main.py enhance "帮我写一份项目周报"
        _demo_enhance(sys.argv[2] if len(sys.argv) > 2 else "帮我写一份项目周报")
    elif not _tavily_key_ready():
        print(
            "尚未检测到有效的 TAVILY_API_KEY。\n"
            "请打开 .env，把 TAVILY_API_KEY 替换为你在 https://app.tavily.com "
            "申请的真实 Key（tvly- 开头），再重新运行：uv run python main.py\n"
            "提示：无需联网 Key 也可体验子智能体委派：uv run python main.py enhance"
        )
    else:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "搜索一下讲师：悟鸣AI的信息"}]}
        )
        print(result["messages"][-1].content)