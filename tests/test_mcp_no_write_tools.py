"""Contract tests for the MCP surface — CI security gate.

Fails LOUDLY if any write-shaped tool is registered on ``build_server()``.
Spec §Security: "MCP has no write tools in v1. Not now, not 'just in case.'"
Adding any later requires a documented human-approve gate + separate credential
AND removing the corresponding pattern from ``WRITE_TOOL_RE``.

Enforcement: exact allow-list assertion (``test_only_three_tools_registered``)
PLUS a regex sweep (``test_no_write_tool_names``). The allow-list catches
additions; the regex catches renames — belt and braces.

AND ONE THAT CHECKS THE BEHAVIOUR RATHER THAN THE NAME. Everything above reads
metadata: the tool list, the tool names, the docstrings. All three passed while
``aggregator_search_memory`` — annotated ``readOnlyHint: True``, on a server
whose client instructions say "Nothing on this surface writes" — appended the
user's raw query text to a second SQLite database on every zero-result call. A
gate that only ever reads the label cannot see that, so
``test_the_search_tool_writes_nothing_to_disk`` runs the tool and looks at the
filesystem.
"""
from __future__ import annotations

import asyncio
import re

from aggregator.core.store import Store
from aggregator.mcp import (
    CAPABILITIES_TOOL_NAME,
    INGEST_TOOL_NAME,
    SEARCH_TOOL_NAME,
    _tool_aggregator_query,
    build_server,
)

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
        SEARCH_TOOL_NAME,
        CAPABILITIES_TOOL_NAME,
        INGEST_TOOL_NAME,
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
    doc = docs[SEARCH_TOOL_NAME]
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
    doc = docs[INGEST_TOOL_NAME].lower()
    assert "human" in doc or "approve" in doc or "does not" in doc, (
        "aggregator_ingest docstring must state that it does not auto-run ingest."
    )


def test_the_search_tool_writes_nothing_to_disk(tmp_path, monkeypatch):
    """THE ANNOTATION IS A CLAIM ABOUT BEHAVIOUR, AND IT WAS FALSE.

    Driven through ``_tool_aggregator_query`` — the coroutine FastMCP actually
    registers and annotates — rather than through the Python function, because
    the whole question is what the ANNOTATED surface does. A zero-result query
    is the case that used to write: it appended the raw query text to
    ``retrieval_eval.db`` beside the cache, append-only and never deduplicated,
    from a tool the client is told writes nothing. Clients use
    ``readOnlyHint`` to decide what may run without asking, so the annotation
    is load-bearing rather than decorative, and the payload was the user's own
    words.

    The sweep is over the whole directory rather than over the one filename
    that regressed: a second write under a different name is the same defect,
    and naming only ``retrieval_eval.db`` would let it through. ``cache.db``
    and its WAL/SHM siblings are excluded because opening a SQLite file for
    reading creates them; the cache is what the tool is FOR.
    """
    store = Store(db_path=tmp_path / "cache.db")
    store.migrate()
    monkeypatch.setattr("aggregator.mcp._default_store", lambda: store)
    monkeypatch.setattr("aggregator.mcp._miss_log", None)
    monkeypatch.setattr("aggregator.mcp._miss_log_path", None)

    result = asyncio.run(_tool_aggregator_query(dsl="beef wellington recipe"))

    # Vacuous otherwise: a refusal writes nothing either, and would pass this
    # test while telling us nothing about the path that does write.
    assert result["ok"] is True and result["total"] == 0, result
    strays = sorted(
        p.name for p in tmp_path.iterdir() if not p.name.startswith("cache.db")
    )
    assert strays == [], (
        f"{SEARCH_TOOL_NAME} is annotated readOnlyHint=True and the client "
        f"instructions say nothing on this surface writes, but calling it "
        f"created {strays} beside the cache. Spec §Security: MCP has no write "
        f"tools. The zero-result log belongs on the CLI path."
    )
