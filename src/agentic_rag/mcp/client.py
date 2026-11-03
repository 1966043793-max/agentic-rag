"""MCP 客户端：连接外部 MCP 服务，把其工具注入 Agent。

用「后台线程跑 asyncio 事件循环」的方式桥接异步 MCP SDK 与同步 Agent 循环，
连接（含 stdio 子进程）在整个会话内保持存活。
"""
from __future__ import annotations

import asyncio
import sys
import threading
from typing import Any

from ..agent.tools import register_mcp_client


def mcp_tool_to_openai(tool: Any) -> dict:
    """把 MCP Tool 转成 OpenAI function-calling 格式。"""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
        },
    }


class MCPClient:
    """同步可用的 MCP 客户端（内部维护后台事件循环）。"""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._session: Any = None
        self._cleanup: Any = None
        self.tool_schemas: list[dict] = []
        self.tool_names: set[str] = set()

    # ---- stdio 传输 ----
    def start_stdio(self, command: str, args: list[str], env: dict | None = None) -> "MCPClient":
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_stdio, args=(command, args, env), daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout=30):
            raise TimeoutError("MCP 服务连接超时")
        register_mcp_client(self)
        return self

    def _run_stdio(self, command: str, args: list[str], env: dict | None) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect_stdio(command, args, env))
        self._ready.set()
        self._loop.run_forever()

    async def _connect_stdio(self, command: str, args: list[str], env: dict | None) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(command=command, args=args, env=env)
        cm = stdio_client(params)
        read, write = await cm.__aenter__()
        self._cleanup = cm
        session = ClientSession(read, write)
        await session.__aenter__()
        await session.initialize()
        self._session = session
        await self._load_tools(session)

    # ---- streamable-http 传输 ----
    def start_streamable_http(self, url: str) -> "MCPClient":
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_http, args=(url,), daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=30):
            raise TimeoutError("MCP 服务连接超时")
        register_mcp_client(self)
        return self

    def _run_http(self, url: str) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect_http(url))
        self._ready.set()
        self._loop.run_forever()

    async def _connect_http(self, url: str) -> None:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        cm = streamablehttp_client(url)
        read, write, _ = await cm.__aenter__()
        self._cleanup = cm
        session = ClientSession(read, write)
        await session.__aenter__()
        await session.initialize()
        self._session = session
        await self._load_tools(session)

    async def _load_tools(self, session: Any) -> None:
        tools = await session.list_tools()
        self.tool_schemas = [mcp_tool_to_openai(t) for t in tools.tools]
        self.tool_names = {t.name for t in tools.tools}

    # ---- 调用（同步桥接） ----
    def call(self, name: str, arguments: dict[str, Any]) -> str:
        future = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(name, arguments), self._loop
        )
        result = future.result(timeout=60)
        content = getattr(result, "content", None)
        if content:
            texts = [c.text for c in content if hasattr(c, "text")]
            if texts:
                return "\n".join(texts)
        return str(result)

    def close(self) -> None:
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:
            pass


def connect_demo_server() -> MCPClient:
    """启动并连接内置的物流演示 MCP 服务（stdio 子进程）。"""
    client = MCPClient()
    client.start_stdio(sys.executable, ["-m", "agentic_rag.mcp.demo_server"])
    return client
