"""An agent must be able to ask which sources are searchable SEMANTICALLY.

THE SITUATION THIS EXISTS FOR IS THE NORMAL ONE, NOT AN EDGE CASE. The vector
arm is filled incrementally over a measured 25-30 days of CPU, in the order the
user fixed: dropbox -> substack -> claude-web -> sessions/subagents ->
everything else. For most of that window the honest answer to "is X searchable
semantically" is "not yet", per source and not per corpus. A global percentage
cannot answer it: one "62% embedded" is compatible with dropbox untouched and
with dropbox finished, which are opposite answers to "can I search my notes
yet". ``aggregator status`` has said this since the priority walk landed; over
MCP there was no way to ask, so an agent had to present a partially built index
as a complete one.

ON AN EXPLICIT CALL, NEVER AT CONNECT. The scan is two grouped queries per
ontology and costs a measured 0.27 s warm / 4.3 s cold against the live cache's
1.3 GB. ``capabilities()`` runs on every client handshake, including the cold
one, and a previous fix specifically stopped it pulling the model stack onto
that path. So the tally is behind a parameter that defaults to off, and
``test_the_default_call_does_not_pay_for_the_scan`` is what keeps it there.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from aggregator.core.store import Store
from aggregator.mcp import aggregator_capabilities
from aggregator.sources.base import ObservationRow, Record, SessionRow

_TS = datetime(2026, 7, 25, tzinfo=UTC)


@pytest.fixture
def store(tmp_path):
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert(
        [
            Record(
                stable_id="github:acme/api:1", source="github", subject="pr",
                body="body body", tags=["pr"], created_at=_TS, updated_at=_TS,
            )
        ]
    )
    s.upsert_entities(
        [
            SessionRow(
                session_id="sess-a", root_session_id="sess-a",
                parent_session_id=None, kind="session", agent_id=None,
                agent_type=None, spawned_by_tool_use_id=None, cwd="/x",
                git_branch="main", first_ts=_TS, last_ts=_TS,
                jsonl_path="/tmp/x.jsonl",
            ),
            ObservationRow(
                obs_id="o1", session_id="sess-a", root_session_id="sess-a",
                parent_obs_id=None, type="user", ts=_TS, model=None,
                input_tokens=None, output_tokens=None, tool_name=None,
                tool_use_id=None, body="hello",
            ),
        ]
    )
    return s


def test_the_default_call_does_not_pay_for_the_scan(store, monkeypatch):
    """MCP connect must stay cheap. 4.3 s on a cold page cache, once per
    handshake, is the shape of regression this guards."""
    calls: list[int] = []
    monkeypatch.setattr(
        store,
        "embed_progress_by_source",
        lambda *a, **kw: (calls.append(1), [])[1],
    )
    result = aggregator_capabilities(_store=store)
    assert result["ok"] is True, result
    assert calls == []
    assert "embedding_coverage" not in result


def test_asking_for_coverage_returns_one_row_per_source(store):
    result = aggregator_capabilities(embedding_coverage=True, _store=store)
    assert result["ok"] is True, result
    rows = result["embedding_coverage"]
    assert rows, "coverage was requested and came back empty"
    for row in rows:
        assert {"kind", "source", "state", "total", "embedded", "pending"} <= set(row)


def test_a_source_with_nothing_in_it_is_empty_and_never_complete(store):
    """The failure the constraint names: a partially embedded corpus must never
    look like a fully embedded one that simply found nothing. A source holding
    no rows and a source fully embedded return the same zero vector hits."""
    rows = aggregator_capabilities(embedding_coverage=True, _store=store)[
        "embedding_coverage"
    ]
    states = {r["source"]: r["state"] for r in rows}
    assert states["dropbox"] == "empty"
    assert "sessions" in states


def test_an_unembedded_source_reads_not_started_not_complete(store):
    rows = aggregator_capabilities(embedding_coverage=True, _store=store)[
        "embedding_coverage"
    ]
    sessions = next(r for r in rows if r["source"] == "sessions")
    assert sessions["state"] == "not_started"
    assert sessions["embedded"] == 0
    assert sessions["total"] >= 1


def test_the_rows_come_back_in_the_backfill_order_the_user_chose(store):
    """The order is not a display detail. It is what decides when the index
    becomes useful, and the user overrode the proposed one to get it."""
    rows = aggregator_capabilities(embedding_coverage=True, _store=store)[
        "embedding_coverage"
    ]
    order = [r["source"] for r in rows]
    for earlier, later in (
        ("dropbox", "substack"),
        ("substack", "claude-web"),
        ("claude-web", "sessions"),
        ("sessions", "subagents"),
    ):
        assert order.index(earlier) < order.index(later), order


def test_the_response_says_what_the_coverage_means_for_a_query(store):
    """A tally an agent cannot act on is a tally it will ignore. The note has
    to name the consequence — a not-yet-embedded source answers by keyword
    only — because that is the sentence the agent needs to pass on."""
    result = aggregator_capabilities(embedding_coverage=True, _store=store)
    note = result["embedding_coverage_note"]
    assert "keyword" in note.lower()


def test_a_broken_coverage_scan_is_reported_not_swallowed(store, monkeypatch):
    """Fail loudly. A capabilities call that quietly dropped the key the caller
    asked for reads exactly like a server that predates the feature."""
    def boom(*a, **kw):
        raise RuntimeError("no such table: chunk_embeddings")

    monkeypatch.setattr(store, "embed_progress_by_source", boom)
    result = aggregator_capabilities(embedding_coverage=True, _store=store)
    assert result["ok"] is False, result
    assert "chunk_embeddings" in result["reason"]


def test_the_mcp_tool_adapter_exposes_the_parameter():
    """The tool the model calls is ``_tool_aggregator_capabilities``, not the
    Python function under it — a parameter added to only one of them is
    invisible to every caller that matters."""
    import inspect

    from aggregator.mcp import _tool_aggregator_capabilities

    params = inspect.signature(_tool_aggregator_capabilities).parameters
    assert "embedding_coverage" in params
    assert params["embedding_coverage"].default is False


def test_the_tool_docstring_tells_an_agent_the_parameter_exists():
    """A capability nothing in context mentions is one no agent will use."""
    from aggregator.mcp import aggregator_capabilities as fn

    assert "embedding_coverage" in (fn.__doc__ or "")
