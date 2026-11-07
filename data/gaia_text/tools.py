"""GAIA web and calculator tools built from the existing serving sandboxes."""

from __future__ import annotations

from typing import Any

from dualpd.data.bfcl_web_search.tools import WebBackendError, WebTools, validate_call as validate_web_call
from dualpd.data.retool.tools import tool_registry


TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "search_engine_query",
            "description": "Search the public web for sources relevant to a factual question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {"type": "string"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                    "region": {"type": "string"},
                },
                "required": ["keywords"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url_content",
            "description": "Fetch a public HTTP(S) webpage returned by search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "mode": {"type": "string", "enum": ["raw", "markdown", "truncate"]},
                },
                "required": ["url"],
            },
        },
    },
    tool_registry.tools["code_interpreter"],
]


def validate_call(call: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(call, dict) or not isinstance(call.get("arguments", {}), dict):
        raise ValueError("Tool call and arguments must be JSON objects")
    name = call.get("name")
    arguments = dict(call.get("arguments", {}))
    if name in {"search_engine_query", "fetch_url_content"}:
        return validate_web_call(call)
    if name == "code_interpreter":
        if set(arguments) != {"code"} or not isinstance(arguments["code"], str):
            raise ValueError("code_interpreter requires exactly one string argument: code")
        if not arguments["code"].strip():
            raise ValueError("code must be nonempty")
        return name, arguments
    raise ValueError(f"Unknown tool: {name}")


async def execute_call(web: WebTools, call: dict[str, Any]) -> Any:
    name, arguments = validate_call(call)
    if name == "code_interpreter":
        return await tool_registry.execute_tool(name, arguments)
    try:
        return await web.call({"name": name, "arguments": arguments})
    except WebBackendError:
        raise
