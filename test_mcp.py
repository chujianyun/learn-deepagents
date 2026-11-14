"""MCP 接入自动化测试（~/.my-deepseek/mcp.json -> 网页智能体）

1. 离线降级：占位 API Key 自动跳过 / 指纹缓存命中 / 配置文件缺失静默降级
2. 加载链路：本地 FastMCP HTTP 服务 + 临时 mcp.json，验证工具发现与真实调用
3. 在线端到端：真实模型 + 本地 MCP 服务，验证智能体在需要时自动调用 MCP 工具

运行：.venv/bin/python test_mcp.py
"""

import asyncio
import json
import os
import socket
import sys
import tempfile
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

warnings.simplefilter("ignore")  # 屏蔽 langchain.mcp 的 beta 警告

import mcp_config
from mcp_config import MCPBundle, load_mcp_tools

# 与用户 ~/.my-deepseek/mcp.json 中瑞幸配置同构（占位 Key）
PLACEHOLDER_CONFIG = {
    "mcpServers": {
        "my-coffee": {
            "type": "streamablehttp",
            "url": "https://gwmcp.lkcoffee.com/order/user/mcp",
            "headers": {"Authorization": "Bearer 这里换成你的API KEY"},
        }
    }
}


def _reset_state() -> None:
    mcp_config._cached = None
    mcp_config._connect_failed = None


def _write_config(tmp: Path, data: dict) -> Path:
    path = tmp / "mcp.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


async def test_offline_degrade(tmp: Path) -> None:
    """离线：占位 Key 跳过（不发起网络）、指纹缓存、文件缺失降级。"""
    _reset_state()
    mcp_config.MCP_CONFIG_PATH = _write_config(tmp, PLACEHOLDER_CONFIG)
    bundle = await load_mcp_tools()
    assert bundle.tools == [] and bundle.servers == [], bundle
    assert bundle.notice and "尚未配置" in bundle.notice, bundle.notice
    assert bundle.fingerprint, "占位配置也应记录指纹"

    again = await load_mcp_tools()
    assert again is bundle, "同指纹应命中缓存返回同一快照"

    mcp_config.MCP_CONFIG_PATH = tmp / "absent.json"
    empty = await load_mcp_tools()
    assert empty == MCPBundle([], [], None, ""), empty
    print("[1/3] 离线降级通过：占位跳过 / 指纹缓存 / 缺失降级")


@asynccontextmanager
async def _local_mcp_server():
    """起一个本地 streamable-http MCP 服务，模拟瑞幸下单工具。"""
    import uvicorn
    from fastmcp import FastMCP

    mcp = FastMCP("coffee-test")

    @mcp.tool
    def order_coffee(sku: str, count: int = 1) -> str:
        """瑞幸咖啡下单（测试模拟）：按饮品名称与数量创建订单，返回订单号。"""
        return f"已下单 {count} 杯 {sku}，订单号 TEST-123，请凭取餐码取餐。"

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(mcp.http_app(), host="127.0.0.1", port=port, log_level="warning")
    )
    task = asyncio.create_task(server.serve())
    try:
        while not server.started:
            await asyncio.sleep(0.05)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        server.should_exit = True
        await task


def _coffee_config(url: str) -> dict:
    return {"mcpServers": {"coffee-test": {"type": "streamablehttp", "url": url}}}


async def test_load_and_call(tmp: Path, url: str) -> None:
    """链路：从 mcp.json 发现工具并真实调用（不经过模型）。"""
    _reset_state()
    mcp_config.MCP_CONFIG_PATH = _write_config(tmp, _coffee_config(url))
    bundle = await load_mcp_tools()
    assert bundle.servers == ["coffee-test"], bundle
    names = [t.name for t in bundle.tools]
    assert "order_coffee" in names, names
    tool = next(t for t in bundle.tools if t.name == "order_coffee")
    result = await tool.ainvoke({"sku": "生椰拿铁", "count": 2})
    assert "TEST-123" in str(result), result
    print(f"[2/3] 加载链路通过：发现 {len(bundle.tools)} 个工具，真实调用返回：{result}")


async def test_agent_end_to_end(tmp: Path, url: str) -> None:
    """在线：真实模型构建网页智能体，验证需要时自动调用 MCP 工具。"""
    from server import DEFAULT_MODEL, get_agent  # 导入即 load_dotenv

    if not os.environ.get("OPENAI_API_KEY"):
        print("[3/3] 跳过：未配置 OPENAI_API_KEY")
        return
    _reset_state()
    mcp_config.MCP_CONFIG_PATH = _write_config(tmp, _coffee_config(url))
    bundle = await load_mcp_tools()
    agent = get_agent(DEFAULT_MODEL, False, bundle)
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "帮我点一杯生椰拿铁，要 1 杯，直接下单。"}]}
    )
    calls = [
        tc
        for m in result["messages"]
        for tc in (getattr(m, "tool_calls", None) or [])
    ]
    coffee_calls = [tc for tc in calls if tc["name"] == "order_coffee"]
    assert coffee_calls, f"模型未调用 MCP 工具，实际调用：{[tc['name'] for tc in calls]}"
    answer = result["messages"][-1].content
    if not isinstance(answer, str):
        answer = "".join(p.get("text", "") for p in answer if isinstance(p, dict))
    print(f"[3/3] 端到端通过：模型自动调用 order_coffee({coffee_calls[0]['args']})")
    print(f"  模型回答：{answer[:200]}")


async def _main() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        await test_offline_degrade(tmp)
        async with _local_mcp_server() as url:
            await test_load_and_call(tmp, url)
            await test_agent_end_to_end(tmp, url)
    print("\n全部测试通过")


if __name__ == "__main__":
    asyncio.run(_main())
    sys.exit(0)