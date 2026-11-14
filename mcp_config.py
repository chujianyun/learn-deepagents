"""MCP 服务接入 —— 读取 ~/.my-deepseek/mcp.json，把 MCP 工具适配给智能体

- 配置格式兼容常见 MCP 客户端写法（mcpServers + type/url/headers；
  stdio 服务用 command/args/env），由 FastMCP 的 MCPConfig 原生解析
- API Key 仍是占位符的服务自动跳过；填入真实 Key 保存后，
  下一次聊天请求按文件指纹自动重新加载，无需重启服务
- 任何加载失败都不阻塞聊天：返回空工具集和一条提示，由调用方转告用户

用法（async 上下文）：
    bundle = await load_mcp_tools()
    agent = create_deep_agent(model=..., tools=bundle.tools, ...)
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any, NamedTuple

from langchain_core.tools import BaseTool

MCP_CONFIG_PATH = Path.home() / ".my-deepseek" / "mcp.json"
LOAD_TIMEOUT = 20  # 秒：发现工具的总超时，避免远程服务不可达时拖慢聊天
FAIL_RETRY_INTERVAL = 30  # 秒：连接失败后的冷却期，期内不重复连接

# 常见 API Key 占位写法（不区分大小写），命中即认为该服务尚未配置
_PLACEHOLDER_MARKERS = ("这里换成", "换成你的", "your_api_key", "your-api-key", "placeholder", "xxxx")


class MCPBundle(NamedTuple):
    """一次 MCP 加载的结果快照；fingerprint 用于智能体缓存失效。"""

    tools: list[BaseTool]
    servers: list[str]  # 成功接入的服务名
    notice: str | None  # 需要转告用户的提示（未配置/加载失败），无则为 None
    fingerprint: str  # 配置文件指纹；文件不存在或无服务时为 ""


_EMPTY = MCPBundle([], [], None, "")

_lock = asyncio.Lock()
_cached: MCPBundle | None = None
_connect_failed: tuple[str, float] | None = None  # (指纹, 失败时刻)，用于失败冷却


def _is_placeholder(value: Any) -> bool:
    text = str(value).strip().lower()
    return not text or text == "bearer" or any(m in text for m in _PLACEHOLDER_MARKERS)


def _split_ready_servers(raw: Any) -> tuple[dict, list[str]]:
    """把配置拆成 (可加载的 mcpServers, 因占位 Key 跳过的服务名)。"""
    servers = raw.get("mcpServers") or {} if isinstance(raw, dict) else {}
    ready: dict[str, Any] = {}
    skipped: list[str] = []
    for name, spec in servers.items():
        if not isinstance(spec, dict):
            skipped.append(name)
            continue
        secrets = list((spec.get("headers") or {}).values()) + list((spec.get("env") or {}).values())
        if any(_is_placeholder(v) for v in secrets):
            skipped.append(name)
        else:
            ready[name] = spec
    return {"mcpServers": ready}, skipped


def _fingerprint(path: Path) -> str:
    st = path.stat()
    return f"{st.st_mtime_ns}:{st.st_size}"


async def _discover(config: dict) -> list[BaseTool]:
    # 延迟导入：未安装 langchain[mcp] 时服务仍能启动，仅在真正加载时报错降级
    from langchain.mcp import MCPAdapter

    return await asyncio.wait_for(MCPAdapter(config).list_tools(), timeout=LOAD_TIMEOUT)


async def load_mcp_tools() -> MCPBundle:
    """读取配置并加载 MCP 工具；带指纹缓存与失败冷却，可被并发请求安全调用。"""
    global _cached, _connect_failed
    async with _lock:
        if not MCP_CONFIG_PATH.is_file():
            _cached, _connect_failed = None, None
            return _EMPTY

        fingerprint = _fingerprint(MCP_CONFIG_PATH)
        if _cached is not None and _cached.fingerprint == fingerprint:
            failed_recently = (
                _connect_failed is not None
                and _connect_failed[0] == fingerprint
                and time.monotonic() - _connect_failed[1] < FAIL_RETRY_INTERVAL
            )
            if not failed_recently:
                return _cached
            # 连接失败且仍在冷却期：直接复用失败结果，不重复连接

        try:
            raw = json.loads(MCP_CONFIG_PATH.read_text(encoding="utf-8"))
            ready_config, skipped = _split_ready_servers(raw)
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            bundle = MCPBundle([], [], f"mcp.json 解析失败：{exc}，本次未接入 MCP 工具。", fingerprint)
            _cached = bundle
            return bundle

        names = list(ready_config["mcpServers"])
        if not names:
            notice = (
                f"MCP 服务 {'、'.join(skipped)} 的 API Key 尚未配置（{MCP_CONFIG_PATH}），本次未接入。"
                if skipped
                else None
            )
            bundle = MCPBundle([], [], notice, fingerprint)
            _cached, _connect_failed = bundle, None
            return bundle

        try:
            tools = await _discover(ready_config)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"[:200]
            bundle = MCPBundle([], [], f"MCP 服务加载失败（{reason}），本次回答未使用 MCP 工具。", fingerprint)
            _cached = bundle
            _connect_failed = (fingerprint, time.monotonic())
            return bundle

        notice = f"MCP 服务 {'、'.join(skipped)} 的 API Key 尚未配置，已跳过。" if skipped else None
        bundle = MCPBundle(tools, names, notice, fingerprint)
        _cached, _connect_failed = bundle, None
        return bundle