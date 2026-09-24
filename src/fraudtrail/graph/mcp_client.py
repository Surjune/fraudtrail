"""The agent's graph tools, reached through the TigerGraph MCP server.

The round asks for TigerGraph MCP to expose the graph to the agent, and this is that
path: the same installed queries, called as MCP tools over stdio instead of over REST.
`run_agent.py --mcp` switches the evidence layer onto it, and the answers come out the
same, because the queries are the same. What changes is who is driving them — an MCP
client, which is what an external agent framework would use.

MCP is asynchronous and the investigation is not, so the session lives on its own event
loop in a background thread and each call is handed to it and waited on. One process, one
server, one session for the whole run.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from concurrent.futures import Future
from pathlib import Path
from types import TracebackType
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

log = logging.getLogger(__name__)

# The MCP server ships as a console script inside the environment.
SERVER_COMMAND = ".venv/Scripts/tigergraph-mcp.exe"
SERVER_COMMAND_POSIX = ".venv/bin/tigergraph-mcp"

RUN_QUERY_TOOL = "tigergraph__run_installed_query"

# Answers come back as text holding a fenced JSON document.
FENCED = re.compile(r"```json\s*(.*?)\s*```", re.S)

# Starting the server and waking a stopped workspace both take time.
STARTUP_TIMEOUT_S = 120.0
CALL_TIMEOUT_S = 180.0


class McpError(RuntimeError):
    """The MCP server could not be reached, or refused a call."""


def _jsonable(value: Any) -> Any:
    """Tuples are how a vertex parameter is written in Python; JSON has only lists."""
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _server_command() -> str:
    windows = Path(SERVER_COMMAND)
    return str(windows) if windows.exists() else SERVER_COMMAND_POSIX


class McpGraphClient:
    """Installed queries over MCP, with the same shape `GraphClient` returns."""

    def __init__(self, graph: str, env_file: Path | None = None) -> None:
        self._graph = graph
        self._env_file = env_file or Path(".env")
        self._loop = asyncio.new_event_loop()
        self._ready: Future[None] = Future()
        self._stop = asyncio.Event()
        self._session: ClientSession | None = None
        self._thread = threading.Thread(target=self._run_loop, name="mcp", daemon=True)
        self._thread.start()
        try:
            self._ready.result(timeout=STARTUP_TIMEOUT_S)
        except Exception as exc:
            raise McpError(f"the TigerGraph MCP server did not start: {exc}") from exc
        log.info("connected to the TigerGraph MCP server for graph %s", graph)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        params = StdioServerParameters(
            command=_server_command(),
            args=["--transport", "stdio", "--env-file", str(self._env_file)],
        )
        try:
            async with (
                stdio_client(params) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                self._session = session
                self._ready.set_result(None)
                await self._stop.wait()
        except Exception as exc:  # pragma: no cover - startup failure path
            if not self._ready.done():
                self._ready.set_exception(exc)

    @staticmethod
    def _payload(text: str) -> dict[str, Any]:
        fenced = FENCED.search(text)
        body = fenced.group(1) if fenced else text
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            raise McpError(f"unreadable answer from the MCP server: {text[:200]}") from exc
        if not isinstance(decoded, dict):
            raise McpError(f"expected a JSON object, got {type(decoded).__name__}")
        return decoded

    async def _call(self, name: str, arguments: dict[str, Any]) -> list[Any]:
        if self._session is None:  # pragma: no cover - guarded by the ready future
            raise McpError("the MCP session is not open")
        result = await self._session.call_tool(name, arguments)
        texts = [item.text for item in result.content if isinstance(item, TextContent)]
        if not texts:
            raise McpError(f"{name} returned no content")
        payload = self._payload(texts[0])
        if not payload.get("success"):
            raise McpError(f"{name} failed: {payload.get('summary') or payload.get('error')}")
        data = payload.get("data")
        rows = data.get("result") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            raise McpError(f"{name} returned no result list")
        return rows

    def run_query(self, name: str, params: dict[str, Any] | None = None) -> list[Any]:
        """Run an installed query through MCP and return its result list."""
        arguments = {
            "graph_name": self._graph,
            "query_name": name,
            "params": _jsonable(params or {}),
        }
        future = asyncio.run_coroutine_threadsafe(self._call(RUN_QUERY_TOOL, arguments), self._loop)
        try:
            return future.result(timeout=CALL_TIMEOUT_S)
        except McpError:
            raise
        except Exception as exc:
            raise McpError(f"query {name} failed over MCP: {exc}") from exc

    def close(self) -> None:
        if self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(timeout=30)

    def __enter__(self) -> McpGraphClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
