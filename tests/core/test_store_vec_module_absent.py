"""FTS5 recall must survive a vec table whose MODULE will not load.

Round 3's H3. The store guarded every vector-table write with a probe that
asked ``sqlite_master`` whether the table exists. For an ordinary table that is
the whole question. For a VIRTUAL table it is half of it: ``sqlite_master``
records that the table was created, and says nothing about whether the module
implementing it is loadable right now.

Those two come apart in a completely ordinary way. Fill a cache on an
interpreter where ``sqlite-vec`` works, then open it on one where the wheel is
missing or ABI-mismatched — a different python, a partial ``uv sync``, a NixOS
closure where interpreter, extension and SQLite come from three store paths.
``vector_available`` is then False, but the ``vec_observations`` row in
``sqlite_master`` is still there, so the existence probe said yes and the
``DELETE`` behind it raised ``no such module: vec0``.

WHERE THAT LANDED IS THE POINT. The delete is the per-row vector invalidation
inside ``upsert_entities`` — it runs in the same transaction as the row write,
for every row whose body changed. So the raise did not cost the vector arm,
which was already gone; it aborted the INGEST WRITE and took FTS5 keyword
recall down with it. An optional feature became a single point of failure for
the mandatory one, which is exactly the inversion the module docstring's v5
note says must never happen.

The asymmetry that gave it away: ``rebuild_all`` guarded its vec DROPs on
``self._vector_available`` and was fine. The ingest write path guarded on
existence alone.
"""

import sqlite3
from datetime import UTC, datetime

import pytest

from aggregator.core import store as store_mod
from aggregator.core.store import Store
from aggregator.sources.base import ObservationRow, QueryAST, Record, SessionRow


@pytest.fixture
def broken_vec(monkeypatch):
    """The extension stops loading, exactly as a real ABI mismatch would."""

    def _boom(conn):
        raise sqlite3.OperationalError("simulated sqlite-vec ABI mismatch")

    monkeypatch.setattr(store_mod, "_load_sqlite_vec", _boom)
    monkeypatch.setattr(store_mod, "_VEC_LOAD_WARNED", False)
    return _boom


def _obs(obs_id, body):
    return ObservationRow(
        obs_id=obs_id,
        session_id="sid",
        root_session_id="sid",
        parent_obs_id=None,
        type="user",
        ts=datetime(2026, 1, 1, tzinfo=UTC),
        model=None,
        input_tokens=None,
        output_tokens=None,
        tool_name=None,
        tool_use_id=None,
        body=body,
    )


def _session():
    return SessionRow(
        session_id="sid",
        root_session_id="sid",
        parent_session_id=None,
        kind="session",
        agent_id=None,
        agent_type=None,
        spawned_by_tool_use_id=None,
        cwd=None,
        git_branch=None,
        first_ts=datetime(2026, 1, 1, tzinfo=UTC),
        last_ts=datetime(2026, 1, 1, tzinfo=UTC),
        jsonl_path="/tmp/x.jsonl",
    )


def _record(stable_id, body):
    return Record(
        stable_id=stable_id,
        source="github",
        subject="subject line",
        body=body,
        tags=["t"],
        created_at=None,
        updated_at=None,
        extra={},
    )


@pytest.fixture
def cache_with_vec_tables(tmp_path):
    """A cache carrying real vec tables, written while the extension worked."""
    db = tmp_path / "cache.db"
    store = Store(db_path=db)
    store.migrate()
    assert store.vector_available, "fixture needs a working sqlite-vec"
    store.upsert_entities([_session(), _obs("o1", "the original body")])
    store.upsert([_record("github:1", "the original record body")])
    c = store._c()
    assert c.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'vec_observations'"
    ).fetchone()
    store.close()
    return db


# --- the observations write path -------------------------------------------


def test_editing_an_observation_survives_a_module_that_will_not_load(
    cache_with_vec_tables, broken_vec
):
    """THE H3 REGRESSION: the DELETE aborted the row write that owned it."""
    store = Store(db_path=cache_with_vec_tables)
    assert store.vector_available is False

    store.upsert_entities([_session(), _obs("o1", "an EDITED body")])

    row = store._c().execute(
        "SELECT body FROM observations WHERE obs_id = 'o1'"
    ).fetchone()
    assert row["body"] == "an EDITED body"


def test_keyword_recall_still_serves_the_edited_observation(
    cache_with_vec_tables, broken_vec
):
    """The founding requirement: FTS5 keeps working with the arm gone."""
    store = Store(db_path=cache_with_vec_tables)
    store.upsert_entities([_session(), _obs("o1", "quadratic voting rewritten")])

    hits = store.query_observations(QueryAST(source="sessions", text="quadratic"))
    assert [h.obs_id for h in hits] == ["o1"], (
        "the edited row is not in the keyword index"
    )


# --- the records write path, which had the identical guard ------------------


def test_editing_a_record_survives_a_module_that_will_not_load(
    cache_with_vec_tables, broken_vec
):
    store = Store(db_path=cache_with_vec_tables)
    store.upsert([_record("github:1", "an EDITED record body")])

    hits = store.query(QueryAST(source="github", text="EDITED"))
    assert [r.stable_id for r in hits] == ["github:1"]


def test_a_scoped_rebuild_survives_a_module_that_will_not_load(
    cache_with_vec_tables, broken_vec
):
    """``_purge_orphan_vectors`` carried the same existence-only guard."""
    store = Store(db_path=cache_with_vec_tables)
    store.rebuild_and_upsert("github", [_record("github:1", "rebuilt body")])

    hits = store.query(QueryAST(source="github", text="rebuilt"))
    assert [r.stable_id for r in hits] == ["github:1"]


def test_rebuild_of_entities_survives_a_module_that_will_not_load(
    cache_with_vec_tables, broken_vec
):
    store = Store(db_path=cache_with_vec_tables)
    store.rebuild_and_upsert_entities(
        [_session(), _obs("o1", "rebuilt observation body")]
    )

    hits = store.query_observations(QueryAST(source="sessions", text="rebuilt"))
    assert [h.obs_id for h in hits] == ["o1"]


# --- and the probe itself ---------------------------------------------------


def test_the_probe_separates_existence_from_usability(
    cache_with_vec_tables, broken_vec
):
    store = Store(db_path=cache_with_vec_tables)
    c = store._c()
    assert store_mod._table_present(c, "vec_observations") is True
    assert store_mod._vec_table_usable(c, "vec_observations") is False


def test_the_probe_says_yes_when_the_module_is_loaded(cache_with_vec_tables):
    store = Store(db_path=cache_with_vec_tables)
    c = store._c()
    assert store_mod._vec_table_usable(c, "vec_observations") is True
    assert store_mod._vec_table_usable(c, "vec_nonexistent") is False
