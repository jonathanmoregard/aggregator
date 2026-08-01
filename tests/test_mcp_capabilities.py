"""Tests for aggregator_capabilities MCP tool (M3).

Read-only inventory: sources present, freshness per source, cache path,
schema version, tool tier. Side-effect-free — should not touch the store's
write path.
"""
from __future__ import annotations

from datetime import UTC, datetime

from aggregator.core.store import Store
from aggregator.mcp import aggregator_capabilities
from aggregator.sources.base import Record


def _seed(store) -> None:
    store.migrate()
    store.upsert(
        [
            Record(
                stable_id="sessions:cap1",
                source="sessions",
                subject="hi",
                body="body body",
                tags=["proj-alpha"],
                created_at=datetime(2026, 7, 25, tzinfo=UTC),
                updated_at=datetime(2026, 7, 25, tzinfo=UTC),
            ),
        ]
    )


def test_capabilities_ok_and_shape(tmp_data_home):
    store = Store()
    _seed(store)
    caps = aggregator_capabilities(_store=store)
    assert caps["ok"] is True
    for key in (
        "sources",
        "freshness",
        "cache_path",
        "schema_version",
        "tool_tier",
        "help",
    ):
        assert key in caps, f"missing capabilities key {key!r}"


def test_capabilities_lists_seeded_source(tmp_data_home):
    store = Store()
    _seed(store)
    caps = aggregator_capabilities(_store=store)
    assert "sessions" in caps["sources"]


def test_capabilities_reports_freshness_for_source(tmp_data_home):
    store = Store()
    _seed(store)
    caps = aggregator_capabilities(_store=store)
    assert caps["freshness"]["sessions"] is not None


def test_capabilities_tool_tier_is_read_only(tmp_data_home):
    """Contract: this tool never mutates; tier must document that."""
    store = Store()
    _seed(store)
    caps = aggregator_capabilities(_store=store)
    assert caps["tool_tier"] == "read-only"


def test_capabilities_cache_path_points_at_store_db(tmp_data_home):
    store = Store()
    _seed(store)
    caps = aggregator_capabilities(_store=store)
    assert caps["cache_path"] == str(store.db_path)


def test_capabilities_side_effect_free(tmp_data_home):
    """Two consecutive calls must return the same shape + not add records."""
    store = Store()
    _seed(store)
    caps1 = aggregator_capabilities(_store=store)
    caps2 = aggregator_capabilities(_store=store)
    assert caps1["sources"] == caps2["sources"]
    # No records added by capabilities()
    row = store._c().execute("SELECT COUNT(*) AS n FROM records").fetchone()
    assert row["n"] == 1


def test_capabilities_on_empty_store(tmp_data_home):
    """Fresh store with no ingested data: still returns ok with empty lists."""
    store = Store()
    store.migrate()
    caps = aggregator_capabilities(_store=store)
    assert caps["ok"] is True
    assert caps["sources"] == []


def test_capabilities_help_includes_dsl_syntax(tmp_data_home):
    store = Store()
    _seed(store)
    caps = aggregator_capabilities(_store=store)
    assert "source:" in caps["help"]
