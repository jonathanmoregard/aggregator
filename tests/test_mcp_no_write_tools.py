"""Contract tests for the MCP surface — CI security gate.

Fails LOUDLY if any write-shaped tool is registered on ``build_server()``.
Spec §Security: "MCP has no write tools in v1. Not now, not 'just in case.'"
Adding any later requires a documented human-approve gate + separate credential
AND removing the corresponding pattern from ``WRITE_TOOL_RE``.

Enforcement: exact allow-list assertion (``test_only_three_tools_registered``)
PLUS a regex sweep (``test_no_write_tool_names``). The allow-list catches
additions; the regex catches renames — belt and braces.
"""
from __future__ import annotations

import asyncio
import re

from aggregator.mcp import build_server

# Regex covers the common write-verb shapes across MCP servers we've seen in
# the wild (Gmail, Docs, Calendar, Router, etc.). Prefix OR contains match.
WRITE_TOOL_RE = re.compile(
    r"^(write_|send_|post_|delete_|create_|update_|patch_|put_|manage_|"
    r"resolve_|insert_|modify_)|(mutate|sudo)",
    re.IGNORECASE,
)


def _list_tools_sync(server) -> list:
    """FastMCP 3.x exposes ``list_tools()`` as async. Bridge to sync for tests."""
    return asyncio.run(server.list_tools())


def _tool_names(server) -> list[str]:
    return [t.name for t in _list_tools_sync(server)]


def _tool_docs(server) -> dict[str, str]:
    return {t.name: (t.description or "") for t in _list_tools_sync(server)}


def test_only_three_tools_registered():
    server = build_server()
    tools = _tool_names(server)
    assert set(tools) == {
        "aggregator_query",
        "aggregator_capabilities",
        "aggregator_ingest",
    }, f"unexpected tool set: {sorted(tools)}"


def test_no_write_tool_names():
    server = build_server()
    for name in _tool_names(server):
        assert not WRITE_TOOL_RE.search(name), (
            f"Tool {name!r} matches write-tool pattern. MCP v1 must have NO write tools."
        )


def test_aggregator_query_docstring_mentions_external_content():
    """Docstring is the model's contract for how to treat returned content."""
    docs = _tool_docs(build_server())
    doc = docs["aggregator_query"]
    assert "ExternalContent" in doc, (
        "aggregator_query docstring must reference the ExternalContent wrapper "
        "so downstream models know the body is untrusted data, not instructions."
    )
    assert "untrusted" in doc.lower(), (
        "aggregator_query docstring must explicitly call the body 'untrusted'."
    )


def test_aggregator_ingest_is_human_approve_gate():
    """Ingest MUST be documented as a human-approve gate (not auto-run)."""
    docs = _tool_docs(build_server())
    doc = docs["aggregator_ingest"].lower()
    assert "human" in doc or "approve" in doc or "does not" in doc, (
        "aggregator_ingest docstring must state that it does not auto-run ingest."
    )
