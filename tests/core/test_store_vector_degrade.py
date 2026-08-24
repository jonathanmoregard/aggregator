"""The vector arm is OPTIONAL. Losing it must not take the store with it.

``sqlite-vec`` is a loadable native extension. It can be absent, built for a
different SQLite ABI, or blocked outright by a python built without
``enable_load_extension`` — and none of those are hypothetical on a NixOS box
where the interpreter, the extension and SQLite itself come from three
different store paths.

Every read in this product goes through ``Store``, including
``aggregator_search_memory``, which today has no vector dependency at all.
Loading the extension unconditionally in ``_c()`` would therefore convert an
optional feature into a single point of failure for recall. The rule instead:
attempt the load, complain loudly exactly once, set ``vector_available``, and
keep FTS5 serving.
"""

import logging
import sqlite3

import pytest

from aggregator.core import store as store_mod
from aggregator.core.store import Store, VectorIndexUnavailableError
from aggregator.sources.base import QueryAST, Record


@pytest.fixture
def broken_vec(monkeypatch):
    """Simulate the extension failing to load, the way a real one would."""

    def _boom(conn):
        raise sqlite3.OperationalError("simulated sqlite-vec ABI mismatch")

    monkeypatch.setattr(store_mod, "_load_sqlite_vec", _boom)
    monkeypatch.setattr(store_mod, "_VEC_LOAD_WARNED", False)
    return _boom


def _record(stable_id: str, body: str) -> Record:
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


def test_vector_available_true_when_extension_loads(tmp_path):
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    assert s.vector_available is True


def test_vector_available_false_when_extension_fails(tmp_path, broken_vec):
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    assert s.vector_available is False


def test_migrate_succeeds_without_the_extension(tmp_path, broken_vec):
    """A failed load must not take the whole schema down with it."""
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    c = s._c()
    tables = {
        row[0] for row in c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"observations", "records", "sessions", "ingest_state"} <= tables
    assert "vec_observations" not in tables
    obs_cols = {row[1] for row in c.execute("PRAGMA table_info(observations)")}
    assert "embedding_state" in obs_cols


def test_fts5_still_serves_when_extension_missing(tmp_path, broken_vec):
    """The whole point: recall keeps working with the vector arm gone."""
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert([_record("github:1", "quadratic voting is a governance mechanism")])
    hits = s.query(QueryAST(source="github", text="quadratic"))
    assert [r.stable_id for r in hits] == ["github:1"]
    assert s.count(QueryAST(source="github", text="quadratic")) == 1


def test_read_only_store_opens_when_extension_missing(tmp_path, broken_vec):
    """The recall path is read-only and must degrade the same way."""
    writable = Store(db_path=tmp_path / "cache.db")
    writable.migrate()
    writable.upsert([_record("github:1", "hello world")])
    writable.close()

    ro = Store(db_path=tmp_path / "cache.db", read_only=True)
    assert ro.vector_available is False
    assert [r.stable_id for r in ro.query(QueryAST(text="hello"))] == ["github:1"]


def test_failure_is_logged_once_not_per_connection(tmp_path, broken_vec, caplog):
    caplog.set_level(logging.WARNING, logger=store_mod.__name__)
    for i in range(3):
        s = Store(db_path=tmp_path / f"cache{i}.db")
        s.migrate()
        s.close()
    warnings = [
        r for r in caplog.records if "sqlite-vec" in r.getMessage() and r.levelno >= logging.WARNING
    ]
    assert len(warnings) == 1, (
        f"expected exactly one loud warning across three connections, got "
        f"{[r.getMessage() for r in warnings]}"
    )
    assert "simulated sqlite-vec ABI mismatch" in warnings[0].getMessage()


def test_rebuild_all_survives_missing_extension(tmp_path, broken_vec):
    """``DROP TABLE`` on a vec0 vtab needs the module; guard it the same way."""
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.rebuild_all()
    assert s.schema_version() == store_mod.SCHEMA_VERSION


def test_vector_unavailable_error_is_a_named_type():
    assert issubclass(VectorIndexUnavailableError, RuntimeError)
