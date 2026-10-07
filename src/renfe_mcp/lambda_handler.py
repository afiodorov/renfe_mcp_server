"""Remote MCP server as an AWS Lambda (Function URL, payload v2).

Transport: Streamable HTTP, stateless, JSON responses only (no SSE, no
sessions). That is a small JSON-RPC surface, so it is implemented here directly
rather than through FastMCP's HTTP app, whose session manager wants a
long-lived task group a Lambda invocation doesn't have. The tools themselves,
their schemas and argument validation still come from the FastMCP server in
server.py, so the stdio and remote servers can't drift apart.

The server expects a writable working directory holding the GTFS feed in
renfe_schedule/ (and writes logs/ next to it), but /var/task is read-only. So
the cold start moves to /tmp, seeds renfe_schedule/ from the copy bundled in the
package at deploy time, and only then imports server.py, whose import loads the
feed after downloading a newer one if data.renfe.com has it.

The request path is ignored, so the function can sit behind any CloudFront
path (e.g. /renfe/mcp).

Handler: renfe_mcp.lambda_handler.lambda_handler
"""

import asyncio
import base64
import json
import logging
import os
import shutil
from pathlib import Path

_bundled = Path(os.environ.get("LAMBDA_TASK_ROOT", ".")) / "renfe_schedule"
os.chdir(os.environ.get("RENFE_WORKDIR", "/tmp"))
if not Path("renfe_schedule").exists() and _bundled.exists():
    shutil.copytree(_bundled, "renfe_schedule")

from fastmcp.exceptions import NotFoundError  # noqa: E402

from renfe_mcp.server import mcp  # noqa: E402

logger = logging.getLogger(__name__)

PROTOCOL_VERSIONS = ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"]
SERVER_INFO = {"name": "renfe-mcp", "title": "Renfe", "version": "0.4.0"}
INSTRUCTIONS = """\
Spanish long-distance and mid-distance train (Renfe) timetables from the \
official GTFS feed, plus live ticket prices scraped from renfe.com. Use \
find_station to see which stations a city has, search_trains for the timetable \
on a date, and get_train_prices for fares (slower: it scrapes)."""

_loop = asyncio.new_event_loop()  # reused across warm invocations


def _run(coro):
    return _loop.run_until_complete(coro)


def _dump(model) -> dict:
    return model.model_dump(by_alias=True, exclude_none=True, mode="json")


_tools_cache: list[dict] | None = None


def list_tools() -> list[dict]:
    global _tools_cache
    if _tools_cache is None:
        _tools_cache = [_dump(t.to_mcp_tool()) for t in _run(mcp.list_tools())]
    return _tools_cache


def call_tool(name: str, args: dict) -> dict:
    """Run a tool; tool failures are reported in the result, per the MCP spec."""
    try:
        result = _run(mcp.call_tool(name, args))
    except NotFoundError:
        raise LookupError(name) from None
    except Exception as e:  # validation errors and anything the tool raised
        logger.warning("tool %s failed: %s", name, e, exc_info=True)
        return {"content": [{"type": "text", "text": str(e)}], "isError": True}
    out = {"content": [_dump(c) for c in result.content], "isError": False}
    if result.structured_content is not None:
        out["structuredContent"] = result.structured_content
    return out


def _ok(id_, result):
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def _err(id_, code, message):
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}


def handle_message(msg: dict) -> dict | None:
    """One JSON-RPC message in, one response out (None for notifications)."""
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or "method" not in msg:
        return _err(msg.get("id") if isinstance(msg, dict) else None, -32600, "Invalid Request")
    if "id" not in msg:
        return None  # notification, e.g. notifications/initialized
    id_, method, params = msg["id"], msg["method"], msg.get("params") or {}
    if method == "initialize":
        requested = params.get("protocolVersion")
        version = requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        return _ok(
            id_,
            {
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": SERVER_INFO,
                "instructions": INSTRUCTIONS,
            },
        )
    if method == "ping":
        return _ok(id_, {})
    if method == "tools/list":
        return _ok(id_, {"tools": list_tools()})
    if method == "tools/call":
        try:
            return _ok(id_, call_tool(params.get("name"), params.get("arguments") or {}))
        except LookupError:
            return _err(id_, -32602, f"Unknown tool: {params.get('name')}")
    return _err(id_, -32601, f"Method not found: {method}")


CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
    "Access-Control-Allow-Headers": (
        "Content-Type, Accept, Authorization, Mcp-Session-Id, Mcp-Protocol-Version, Last-Event-ID"
    ),
    "Access-Control-Expose-Headers": "Mcp-Session-Id, Mcp-Protocol-Version",
}


def _http(status: int, body=None, content_type="application/json") -> dict:
    headers = dict(CORS)
    if body is not None:
        headers["Content-Type"] = content_type
        body = body if isinstance(body, str) else json.dumps(body, default=str)
    return {"statusCode": status, "headers": headers, "body": body or ""}


def lambda_handler(event, context):
    """Lambda Function URL (payload v2) entry point."""
    method = event.get("requestContext", {}).get("http", {}).get("method", "POST")
    if method == "OPTIONS":
        return _http(204)
    if method == "GET":
        # No server-initiated SSE stream in stateless mode; a plain browser visit gets a pointer.
        accept = (event.get("headers") or {}).get("accept", "")
        if "text/event-stream" in accept:
            return _http(405, _err(None, -32000, "SSE stream not supported; POST JSON-RPC instead"))
        return _http(
            200,
            f"Renfe MCP server (Streamable HTTP). POST JSON-RPC here.\n\n{INSTRUCTIONS}\n",
            "text/plain; charset=utf-8",
        )
    if method != "POST":
        return _http(405, _err(None, -32000, "Method not allowed"))

    body = event.get("body") or ""
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode()
    try:
        payload = json.loads(body)
    except ValueError:
        return _http(400, _err(None, -32700, "Parse error"))

    if isinstance(payload, list):  # JSON-RPC batch (pre-2025-06-18 clients)
        responses = [r for r in (handle_message(m) for m in payload) if r is not None]
        return _http(200, responses) if responses else _http(202)
    response = handle_message(payload)
    return _http(200, response) if response is not None else _http(202)
