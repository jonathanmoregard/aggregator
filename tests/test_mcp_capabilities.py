"""Tests for aggregator_capabilities MCP tool (v2, Schema B)."""
from __future__ import annotations

from datetime import UTC, datetime

from aggregator.core.store import Store
from aggregator.mcp import aggregator_capabilities
from aggregator.sources.base import ObservationRow, Record, SessionRow


def _seed(store) -> None:
    store.migrate()
    store.upsert(
        [
            Record(
                stable_id="github:acme/api:1",
                source="github",
                subject="pr",
                body="body body",
                tags=["pr"],
                created_at=datetime(2026, 7, 25, tzinfo=UTC),
                updated_at=datetime(2026, 7, 25, tzinfo=UTC),
            ),
        ]
    )
    store.upsert_entities(
        [
            SessionRow(
                session_id="sess-a",
                root_session_id="sess-a",
                parent_session_id=None,
                kind="session",
                agent_id=None,
                agent_type=None,
                spawned_by_tool_use_id=None,
                cwd="/x",
                git_branch="main",
                first_ts=datetime(2026, 7, 25, tzinfo=UTC),
                last_ts=datetime(2026, 7, 25, tzinfo=UTC),
                jsonl_path="/tmp/x.jsonl",
            ),
            ObservationRow(
                obs_id="o1",
                session_id="sess-a",
                root_session_id="sess-a",
                parent_obs_id=None,
                type="user",
                ts=datetime(2026, 7, 25, tzinfo=UTC),
                model=None,
                input_tokens=None,
                output_tokens=None,
                tool_name=None,
                tool_use_id=None,
                body="hello",
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
        "counts",
    ):
        assert key in caps, f"missing capabilities key {key!r}"


def test_capabilities_lists_both_records_and_sessions_sources(tmp_data_home):
    store = Store()
    _seed(store)
    caps = aggregator_capabilities(_store=store)
    assert "github" in caps["sources"]
    assert "sessions" in caps["sources"]


def test_capabilities_reports_v2_counts(tmp_data_home):
    store = Store()
    _seed(store)
    caps = aggregator_capabilities(_store=store)
    assert caps["counts"]["sessions"] == 1
    assert caps["counts"]["observations"] == 1
    assert caps["counts"]["records"] == 1


def test_capabilities_reports_freshness_for_source(tmp_data_home):
    store = Store()
    _seed(store)
    caps = aggregator_capabilities(_store=store)
    assert caps["freshness"]["github"] is not None
    assert caps["freshness"]["sessions"] is not None


def test_capabilities_tool_tier_is_read_only(tmp_data_home):
    store = Store()
    _seed(store)
    caps = aggregator_capabilities(_store=store)
    assert caps["tool_tier"] == "read-only"


def test_capabilities_default_store_reports_missing_cache(tmp_data_home):
    caps = aggregator_capabilities()
    assert caps["ok"] is False
    assert "cache unavailable" in caps["reason"]
    assert "MCP recall is read-only" in caps["remediation"]


def test_capabilities_cache_path_points_at_store_db(tmp_data_home):
    store = Store()
    _seed(store)
    caps = aggregator_capabilities(_store=store)
    assert caps["cache_path"] == str(store.db_path)


def test_capabilities_side_effect_free(tmp_data_home):
    store = Store()
    _seed(store)
    caps1 = aggregator_capabilities(_store=store)
    caps2 = aggregator_capabilities(_store=store)
    assert caps1["sources"] == caps2["sources"]


def test_capabilities_on_empty_store(tmp_data_home):
    store = Store()
    store.migrate()
    caps = aggregator_capabilities(_store=store)
    assert caps["ok"] is True
    assert caps["sources"] == []


def test_capabilities_registers_research_like_github(tmp_data_home):
    """Chunk 4: research records register in sources / freshness /
    tags_by_source / counts — same automatic path as github, plus a
    per-record-source entry under ``counts``."""
    store = Store()
    store.migrate()
    store.upsert(
        [
            Record(
                stable_id=f"research:rep{i}",
                source="research",
                subject=f"report {i}",
                body="body",
                tags=["research"],
                created_at=datetime(2026, 7, 30, tzinfo=UTC),
                updated_at=datetime(2026, 7, 30, tzinfo=UTC),
            )
            for i in range(2)
        ]
    )
    caps = aggregator_capabilities(_store=store)
    assert "research" in caps["sources"]
    assert caps["freshness"]["research"] is not None
    assert caps["tags_by_source"]["research"] == ["research"]
    assert caps["counts"]["research"] == 2


def test_capabilities_registers_sota_watch_like_github(tmp_data_home):
    """Chunk 7: sota-watch records register in sources / freshness /
    tags_by_source / counts — same automatic records-source path."""
    store = Store()
    store.migrate()
    store.upsert(
        [
            Record(
                stable_id=f"sota-watch:2026-07-3{i}-topic",
                source="sota-watch",
                subject=f"proposal {i}",
                body="proposal body",
                tags=["sota-watch"],
                created_at=datetime(2026, 7, 30, tzinfo=UTC),
                updated_at=datetime(2026, 7, 30, tzinfo=UTC),
            )
            for i in range(2)
        ]
    )
    caps = aggregator_capabilities(_store=store)
    assert "sota-watch" in caps["sources"]
    assert caps["freshness"]["sota-watch"] is not None
    assert caps["tags_by_source"]["sota-watch"] == ["sota-watch"]
    assert caps["counts"]["sota-watch"] == 2


# --- v5: vector-index state reaches the MCP surface -----------------------


def test_capabilities_exposes_vector_index(tmp_data_home):
    """Task L: a caller must be able to see, from the tool surface alone,
    whether hybrid retrieval is warm on this cache."""
    store = Store()
    _seed(store)
    caps = aggregator_capabilities(_store=store)
    assert "vector_index" in caps
    vi = caps["vector_index"]
    assert vi["available"] is True
    assert vi["state"] == "not_started"
    assert vi["observations"]["pending"] == 1
    assert vi["observations"]["vectors"] == 0


def test_capabilities_vector_index_distinguishes_unavailable_from_unembedded(
    tmp_data_home, monkeypatch
):
    """The two situations that would otherwise both render as "0 embedded"."""
    import sqlite3

    from aggregator.core import store as store_mod

    unembedded = aggregator_capabilities(_store=_seeded_store())["vector_index"]

    def _boom(conn):
        raise sqlite3.OperationalError("simulated sqlite-vec ABI mismatch")

    monkeypatch.setattr(store_mod, "_load_sqlite_vec", _boom)
    monkeypatch.setattr(store_mod, "_VEC_LOAD_WARNED", False)
    broken = aggregator_capabilities(_store=_seeded_store())["vector_index"]

    assert unembedded["state"] == "not_started"
    assert broken["state"] == "unavailable"
    assert unembedded != broken


def _seeded_store():
    store = Store()
    _seed(store)
    return store


# --- the enumeration a caller reads must match what the code emits --------
#
# Two docstrings enumerate the ``vector_index`` states: ``aggregator_capabilities``
# (which IS the MCP tool description a caller reads before acting) and
# ``Store.vector_index_state`` (which the first one restates). Chunk N added
# ``degraded`` to the code and neither enumeration noticed, so a caller reading
# the tool description would conclude the state did not exist. Asserted together
# and in one place, because they are one invariant with two copies.


def _states_the_store_can_report() -> set[str]:
    """Every literal ``state`` value ``vector_index_state`` can return.

    Read out of the production source, NOT listed here: a hand-maintained list
    in a test is the same artefact as the hand-maintained list in a docstring.
    It would go stale in the same way, at the same moment, and for the same
    reason — which is exactly what happened.
    """
    import inspect
    import re

    src = inspect.getsource(Store.vector_index_state)
    return set(re.findall(r'^\s*state = "([a-z_]+)"$', src, re.MULTILINE))


def test_the_state_extraction_itself_has_teeth():
    """Without this, a refactor that breaks the regex makes the two tests
    below pass vacuously — a doc guard that guards nothing is worse than none,
    because it is also believed."""
    states = _states_the_store_can_report()
    assert states >= {
        "unavailable",
        "empty",
        "complete",
        "degraded",
        "not_started",
        "backfilling",
    }, f"state extraction found only {sorted(states)}"


def test_capabilities_docstring_documents_every_state_it_can_report():
    documented = aggregator_capabilities.__doc__ or ""
    missing = sorted(s for s in _states_the_store_can_report() if s not in documented)
    assert not missing, (
        f"aggregator_capabilities' docstring — the MCP tool description a "
        f"caller reads — does not name these states it can return: {missing}"
    )


def test_vector_index_state_docstring_documents_every_state_it_can_report():
    documented = Store.vector_index_state.__doc__ or ""
    missing = sorted(s for s in _states_the_store_can_report() if s not in documented)
    assert not missing, (
        f"Store.vector_index_state's docstring does not name these states it "
        f"returns: {missing}"
    )


def test_capabilities_help_includes_dsl_syntax(tmp_data_home):
    store = Store()
    _seed(store)
    caps = aggregator_capabilities(_store=store)
    assert "source:" in caps["help"]
    assert "session:" in caps["help"]
