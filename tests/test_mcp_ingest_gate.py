"""Tests for aggregator_ingest MCP tool — HUMAN-APPROVE GATE (M3).

Spec §Security: "Adding any [write tool] later requires a documented
human-approve gate + a separate credential."

The MCP-surface ``aggregator_ingest`` is deliberately a stub that returns
instructions telling the caller (usually a model) to run the CLI command
in the terminal. It NEVER triggers ingest itself, and it NEVER writes to
the store.
"""
from __future__ import annotations

from datetime import UTC, datetime

from aggregator.core.store import Store
from aggregator.mcp import aggregator_ingest
from aggregator.sources.base import Record


def _seed_and_count(store) -> int:
    store.migrate()
    store.upsert(
        [
            Record(
                stable_id="sessions:pre",
                source="sessions",
                subject="hi",
                body="body",
                tags=[],
                created_at=datetime(2026, 7, 25, tzinfo=UTC),
                updated_at=datetime(2026, 7, 25, tzinfo=UTC),
            ),
        ]
    )
    return store._c().execute("SELECT COUNT(*) AS n FROM records").fetchone()["n"]


def test_ingest_returns_ok_with_instructions(tmp_data_home):
    store = Store()
    store.migrate()
    result = aggregator_ingest(source="sessions", _store=store)
    assert result["ok"] is True
    assert "message" in result
    # Instructions must reference the CLI command shape (M5).
    assert "aggregator ingest sessions" in result["message"]


def test_ingest_mentions_terminal_or_manual_step(tmp_data_home):
    store = Store()
    store.migrate()
    result = aggregator_ingest(source="sessions", _store=store)
    msg = result["message"].lower()
    assert (
        "terminal" in msg or "manually" in msg or "does not" in msg
    ), "ingest response must communicate that it does not auto-run"


def test_ingest_does_not_write_to_store(tmp_data_home):
    """Contract: no store writes, ever, from this tool."""
    store = Store()
    baseline = _seed_and_count(store)
    aggregator_ingest(source="sessions", _store=store)
    aggregator_ingest(source="github", _store=store)
    after = store._c().execute("SELECT COUNT(*) AS n FROM records").fetchone()["n"]
    assert after == baseline, "aggregator_ingest MUST NOT write to the store"


def test_ingest_accepts_unknown_source_without_side_effects(tmp_data_home):
    """Even a bogus source name must not raise or write — return instructions."""
    store = Store()
    baseline = _seed_and_count(store)
    result = aggregator_ingest(source="does-not-exist", _store=store)
    assert result["ok"] is True
    assert "does-not-exist" in result["message"]
    after = store._c().execute("SELECT COUNT(*) AS n FROM records").fetchone()["n"]
    assert after == baseline
