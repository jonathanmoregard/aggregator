"""Deleting a row must delete its vectors too.

``rebuild_all`` drops the vec tables outright, so it was already safe. The
SCOPED delete paths were not: ``rebuild(source)`` and
``rebuild_and_upsert_entities(origins=…)`` remove rows from ``records`` /
``observations`` and leave the corresponding ``vec_*`` rows behind forever.

Nothing ever notices, and that is the problem. An orphaned vector still
competes for one of the ``_VECTOR_ARM_K`` = 50 KNN slots the vector arm gets
per query, and then matches no row when the fused id set hits SQL — so it is
not a wrong result, it is a silently smaller one. Enough orphans and the
vector arm returns 50 neighbours and contributes nothing at all, while every
count still says the index is complete.

A rebuild is exactly where this piles up: the chat-export and github sources
rebuild by design, so the orphan count grows with every rebuild and never
shrinks.
"""

import numpy as np
import pytest

from aggregator.core.store import _VEC_DIM, Store
from aggregator.sources.base import QueryAST
from tests.core.test_store_id_scope import _obs, _rec, _session


@pytest.fixture
def store(tmp_path):
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    return s


def _unit(seed=0):
    rng = np.random.default_rng(seed)
    v = rng.random(_VEC_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def test_rebuilding_a_record_source_drops_its_vectors(store):
    store.upsert([_rec("github:x:1"), _rec("github:x:2")])
    store.upsert_vec_records([("github:x:1", _unit(1)), ("github:x:2", _unit(2))])
    assert store.count_vec_rows("records") == 2

    store.rebuild("github")

    assert store.count_vec_rows("records") == 0


def test_rebuilding_one_source_leaves_another_source_vectors_alone(store):
    store.upsert([_rec("github:x:1")])
    store.upsert_vec_records([("github:x:1", _unit(1))])
    store._c().execute(
        "INSERT INTO records(stable_id, source, subject, body, tags, "
        "created_at, updated_at) VALUES ('dropbox:f:1','dropbox','s','b','[]',"
        "'2026-01-01','2026-01-01')"
    )
    store._c().commit()
    store.upsert_vec_records([("dropbox:f:1", _unit(2))])
    assert store.count_vec_rows("records") == 2

    store.rebuild("github")

    assert store.count_vec_rows("records") == 1


def test_a_scoped_entity_rebuild_drops_the_vectors_it_deleted(store):
    store.upsert_entities([_session("s1"), _obs("o1", "s1")])
    store.upsert_vec_observations([("o1", _unit(1))])
    assert store.count_vec_rows("observations") == 1

    store.rebuild_and_upsert_entities([], origins=["claude-code"])

    assert store.count_vec_rows("observations") == 0


def test_an_unscoped_entity_rebuild_drops_every_observation_vector(store):
    store.upsert_entities([_session("s1"), _obs("o1", "s1"), _obs("o2", "s1")])
    store.upsert_vec_observations([("o1", _unit(1)), ("o2", _unit(2))])

    store.rebuild_and_upsert_entities([], origins=None, min_sessions=0)

    assert store.count_vec_rows("observations") == 0


def test_chunk_vectors_of_a_deleted_row_go_too(store):
    """A long body is stored as ``<id>:N``, not under the bare id."""
    store.upsert([_rec("github:x:1")])
    store.upsert_vec_records([(f"github:x:1:{i}", _unit(i)) for i in range(3)])
    assert store.count_vec_rows("records") == 3

    store.rebuild("github")

    assert store.count_vec_rows("records") == 0


def test_rebuild_paths_still_work_without_sqlite_vec(tmp_path, monkeypatch):
    """No extension means no vec tables to clean; the rebuild must not raise."""
    import sqlite3

    from aggregator.core import store as store_mod

    def _boom(conn):
        raise sqlite3.OperationalError("simulated sqlite-vec ABI mismatch")

    monkeypatch.setattr(store_mod, "_load_sqlite_vec", _boom)
    monkeypatch.setattr(store_mod, "_VEC_LOAD_WARNED", False)
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert([_rec("github:x:1")])

    s.rebuild("github")
    s.rebuild_and_upsert_entities([], origins=["claude-code"])

    assert s.vector_available is False
    assert s.query(QueryAST()) == []


def test_a_row_that_comes_back_from_a_rebuild_keeps_its_vector(store):
    """The purge runs AFTER the re-write, and that ordering is the point.

    A rebuild re-inserts most of what it deleted. Purging before the write
    would drop every vector in the scope and leave recall dark until a
    25-30 day backfill caught up; purging after it costs only the rows that
    genuinely did not return.
    """
    store.upsert_entities([_session("s1"), _obs("o1", "s1"), _obs("o2", "s1")])
    store.upsert_vec_observations([("o1", _unit(1)), ("o2", _unit(2))])

    # o1 comes back, o2 does not.
    store.rebuild_and_upsert_entities(
        [_session("s1"), _obs("o1", "s1")], origins=["claude-code"]
    )

    assert store.count_vec_rows("observations") == 1
    assert store._vec_obs_ids(_unit(1), k=5) == ["o1"]


def test_a_record_that_comes_back_keeps_its_vector(store):
    store.upsert([_rec("github:x:1"), _rec("github:x:2")])
    store.upsert_vec_records([("github:x:1", _unit(1)), ("github:x:2", _unit(2))])

    store.rebuild_and_upsert("github", [_rec("github:x:1")])

    assert store.count_vec_rows("records") == 1
