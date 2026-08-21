"""Contract tests for the usage-assurance surface.

The aggregator lost every recall query to the client's built-in memory
directory and to raw grep over ~/.claude/projects, for one reason: nothing
named it anywhere the model could see. Under deferred tool loading the client
sees tool NAMES first and fetches schemas only after a tool-search, so
discoverability is a hard contract, not cosmetics.

These tests pin the four levers that fix it:

1. ``instructions`` is populated (the MCP ``initialize`` field the client
   injects into its system prompt).
2. The search tool's NAME carries recall vocabulary.
3. Descriptions say when to use the tool AND what it supersedes.
4. Neither the instructions nor the descriptions hardcode a source list —
   sources are read live from the cache, so the text can never go stale.

(4) is the one most likely to rot silently: a hardcoded "github, research,
substack" reads as authoritative long after a source is dropped.
"""
from __future__ import annotations

import asyncio

import pytest

from aggregator.mcp import (
    _INSTRUCTIONS_CORE,
    SEARCH_TOOL_NAME,
    _live_inventory,
    build_server,
)

# Source names must never appear in the STATIC strings. Keep this list in
# sync with aggregator/sources/ — a new source added here that shows up
# hardcoded in a description will fail the test, which is the point.
KNOWN_SOURCE_NAMES = [
    "github",
    "research",
    "sota-watch",
    "substack",
    "chatgpt",
    "claude-web",
    "exportdrops",
]


def _tools(server) -> dict:
    return {t.name: t for t in asyncio.run(server.list_tools())}


def test_server_instructions_are_populated():
    """The initialize-result ``instructions`` field is the only lever that
    names this server in the client's system prompt without the user editing
    their own config. Empty instructions = invisible server."""
    server = build_server()
    instructions = server.instructions or ""
    assert instructions.strip(), "server instructions must not be empty"
    assert SEARCH_TOOL_NAME in instructions, (
        "instructions must name the search tool explicitly — the client has "
        "to know what string to tool-search for."
    )


def test_instructions_state_what_they_supersede():
    """Naming the tool is not enough; the model already has a default path.
    The instructions have to say which path this replaces."""
    instructions = (build_server().instructions or "").lower()
    assert "~/.claude/projects" in instructions
    assert "memory" in instructions


def test_search_tool_name_carries_recall_vocabulary():
    """Deferred loading matches on names. A name without recall words is
    never surfaced for a "do you remember…" prompt."""
    assert "search" in SEARCH_TOOL_NAME
    assert "memory" in SEARCH_TOOL_NAME
    assert SEARCH_TOOL_NAME.startswith("aggregator_"), (
        "keep the service prefix for namespacing"
    )


def test_search_description_has_use_and_dont_use_guidance():
    tool = _tools(build_server())[SEARCH_TOOL_NAME]
    desc = (tool.description or "").lower()
    assert "instead of" in desc, (
        "description must state what it supersedes, not just what it does"
    )
    assert "do not use" in desc or "don't use" in desc, (
        "description must carve out the negative cases (repo code, web)"
    )
    assert desc.count(".") >= 4, "description should be several sentences"


def test_search_description_states_the_measured_cost_of_rerank():
    """A parameter whose real cost is 400x its documented cost is a trap.

    ``rerank=True`` was first documented as "roughly 300 ms per hit", so a
    caller reading the tool schema would reasonably set it inside an
    interactive turn. That was replaced by Task M's "47 s median", and THAT
    did not reproduce either: re-measured 2026-08-21 over 12 real pages of 20
    hits from a read-only snapshot of the live cache, the shipped stage costs
    **273 s median / 304 s p95** per call on this CPU, versus 0.65 s without.

    The caller reads THIS string and nothing else before deciding, so the
    number in it has to be the current measured one. Twice now the figure here
    has been optimistic by an order of magnitude, which is why this test pins
    it rather than trusting review — but it pins the ORDER, not the digits, so
    honest re-measurement does not have to fight it.
    """
    desc = _tools(build_server())[SEARCH_TOOL_NAME].description or ""
    _, _, rerank_doc = desc.partition("rerank:")
    assert rerank_doc, "the rerank parameter is not documented at all"
    assert "300 ms" not in rerank_doc, (
        "the rerank docs still quote the falsified 300 ms/hit estimate"
    )
    assert "47 s" not in rerank_doc, (
        "the rerank docs still quote the falsified 47 s median; it was "
        "re-measured at 273 s on the real index"
    )
    assert "273 s" in rerank_doc, (
        "the rerank docs must quote the measured per-call cost so a caller "
        "can tell an interactive call from a batch one. If this was "
        "re-measured, update the number here and at _RERANK_WINDOW."
    )
    assert "minute" in rerank_doc, (
        "seconds-with-three-digits reads as small at a glance; the docs must "
        "also say it in minutes, which is the unit that stops an LLM caller "
        "setting this in an interactive turn"
    )


@pytest.mark.parametrize("source_name", KNOWN_SOURCE_NAMES)
def test_static_strings_do_not_hardcode_sources(source_name):
    """Source lists come from the cache at build time, never from a literal.

    A stale enumeration in an always-in-context string is worse than no
    enumeration: it asserts coverage that may no longer exist.
    """
    from aggregator import mcp as mcp_mod

    static = f"{_INSTRUCTIONS_CORE}\n{mcp_mod.aggregator_query.__doc__ or ''}"
    assert source_name not in static, (
        f"{source_name!r} is hardcoded in a static description string. "
        "Let _live_inventory() read it from the cache instead."
    )


def test_live_inventory_degrades_to_empty_when_store_unreadable():
    """A broken cache must not break server startup or emit a half-truth."""

    class Boom:
        def capabilities(self):
            raise RuntimeError("cache unavailable")

    assert _live_inventory(Boom()) == ""


def test_live_inventory_renders_sources_from_the_store():
    class Fake:
        def capabilities(self):
            return {
                "sources": ["alpha", "beta"],
                "counts": {"alpha": 3},
                "date_range": ["2020-01-01", "2026-01-01"],
            }

    out = _live_inventory(Fake())
    assert "alpha (3)" in out
    assert "beta" in out
    assert "2020-01-01 .. 2026-01-01" in out


def test_build_server_survives_unreadable_store():
    class Boom:
        def capabilities(self):
            raise RuntimeError("cache unavailable")

    server = build_server(Boom())
    instructions = server.instructions or ""
    assert instructions.strip()
    assert "Cached sources" not in instructions, (
        "with no readable cache the text must claim no coverage at all"
    )


def test_every_tool_is_annotated_read_only():
    for name, tool in _tools(build_server()).items():
        ann = tool.annotations
        assert ann is not None, f"{name} has no annotations"
        assert ann.readOnlyHint is True, f"{name} must be readOnlyHint=True"
        assert ann.destructiveHint is False, f"{name} must not be destructive"
