"""v7 → v8: the retired vector index is dropped, everything else is kept.

A v7 cache carries sqlite-vec ``vec0`` virtual tables. This build no longer
ships sqlite-vec, and ``DROP TABLE`` on a virtual table whose module is not
loaded raises ``no such module: vec0`` — so the fixture below builds exactly
that state: a vtab row in ``sqlite_master`` with no module to back it, plus its
ordinary shadow tables, the way a real v7 cache looks to this build.
"""

import sqlite3
from datetime import UTC, datetime

import pytest

from aggregator.core.store import SCHEMA_VERSION, Store
from aggregator.sources.base import ObservationRow, QueryAST, Record, SessionRow

_TS = datetime(2026, 7, 1, 8, 0, tzinfo=UTC)

_VTABS = {
    "vec_observations": "model TEXT PARTITION KEY, obs_id TEXT, embedding float[768]",
    "vec_records": "model TEXT PARTITION KEY, stable_id TEXT, embedding float[768]",
}
_SHADOW_SUFFIXES = (
    "chunks",
    "info",
    "rowids",
    "vector_chunks00",
    "metadatachunks00",
    "metadatatext00",
)


def _seed(store: Store) -> None:
    store.upsert_entities(
        [
            SessionRow(
                session_id="s1", root_session_id="s1", parent_session_id=None,
                kind="session", agent_id=None, agent_type=None,
                spawned_by_tool_use_id=None, cwd="/x", git_branch="main",
                first_ts=_TS, last_ts=_TS, jsonl_path="/tmp/s1.jsonl",
            ),
            ObservationRow(
                obs_id="o1", session_id="s1", root_session_id="s1",
                parent_obs_id=None, type="user", ts=_TS, model=None,
                input_tokens=None, output_tokens=None, tool_name=None,
                tool_use_id=None, body="quadratic voting is the mechanism",
            ),
        ]
    )
    store.upsert(
        [
            Record(
                stable_id="github:acme/api:1", source="github", subject="pr",
                body="quadratic funding notes", tags=["pr"],
                created_at=_TS, updated_at=_TS,
            )
        ]
    )


def _make_v7(path) -> None:
    """A populated cache in the v7 shape, vec0 tables and all."""
    store = Store(db_path=path)
    store.migrate()
    _seed(store)
    store.close()

    c = sqlite3.connect(path)
    for table in ("observations", "records"):
        c.execute(f"ALTER TABLE {table} ADD COLUMN embedding_state TEXT")
    c.execute("UPDATE observations SET embedding_state = 'ok'")
    c.execute("CREATE INDEX obs_embedding_state ON observations(embedding_state)")
    c.execute("CREATE INDEX rec_embedding_state ON records(embedding_state)")
    c.execute(
        "CREATE TABLE chunk_embeddings (chunk_id TEXT NOT NULL, model TEXT NOT "
        "NULL, kind TEXT NOT NULL, owner_id TEXT NOT NULL, dim INTEGER NOT NULL, "
        "content_sha256 TEXT, created_at TEXT NOT NULL, PRIMARY KEY (chunk_id, model))"
    )
    c.execute("INSERT INTO chunk_embeddings VALUES ('o1', 'm', 'observations', 'o1', 768, NULL, 't')")
    c.execute(
        "CREATE TABLE embedding_versions (model TEXT PRIMARY KEY, dim INTEGER "
        "NOT NULL, started_at TEXT NOT NULL, completed_at TEXT)"
    )
    for vtab in _VTABS:
        for suffix in _SHADOW_SUFFIXES:
            c.execute(f"CREATE TABLE {vtab}_{suffix} (id INTEGER PRIMARY KEY, data BLOB)")
    c.execute("PRAGMA writable_schema = ON")
    for vtab, columns in _VTABS.items():
        c.execute(
            "INSERT INTO sqlite_master(type, name, tbl_name, rootpage, sql) "
            "VALUES ('table', ?, ?, 0, ?)",
            (vtab, vtab, f"CREATE VIRTUAL TABLE {vtab} USING vec0({columns})"),
        )
    c.execute("PRAGMA writable_schema = OFF")
    c.execute(
        "INSERT INTO meta(key, value) VALUES ('vector_provenance', '{}'), "
        "('embed_inflight', 'observations:o1')"
    )
    c.execute("PRAGMA user_version = 7")
    c.commit()
    c.close()


@pytest.fixture
def v7_cache(tmp_path):
    path = tmp_path / "cache.db"
    _make_v7(path)
    return path


def _names(c: sqlite3.Connection) -> set[str]:
    return {row[0] for row in c.execute("SELECT name FROM sqlite_master")}


def test_the_fixture_is_a_cache_this_build_cannot_drop_the_plain_way(v7_cache):
    """The premise: without sqlite-vec a vec0 table refuses ``DROP TABLE``."""
    c = sqlite3.connect(v7_cache)
    with pytest.raises(sqlite3.OperationalError, match="no such module: vec0"):
        c.execute("DROP TABLE vec_observations")
    c.close()


def test_migrate_drops_every_vector_artifact(v7_cache):
    store = Store(db_path=v7_cache)
    store.migrate()
    c = store._c()
    leftovers = {
        n for n in _names(c)
        if n.startswith("vec_") or "embedding" in n or n == "chunk_embeddings"
    }
    assert leftovers == set(), leftovers
    for table in ("observations", "records"):
        columns = {row[1] for row in c.execute(f"PRAGMA table_info({table})")}
        assert "embedding_state" not in columns, (table, columns)
    keys = {row[0] for row in c.execute("SELECT key FROM meta")}
    assert not keys & {"vector_provenance", "embed_inflight"}, keys
    assert store.schema_version() == SCHEMA_VERSION == 8
    store.close()


def test_migrate_keeps_the_rows_and_the_keyword_index(v7_cache):
    store = Store(db_path=v7_cache)
    store.migrate()
    assert [r.obs_id for r in store.query_observations(QueryAST(text="voting"))] == ["o1"]
    assert [r.stable_id for r in store.query(QueryAST(text="funding"))] == [
        "github:acme/api:1"
    ]
    assert store._c().execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    store.close()


def test_the_migrated_file_opens_read_only(v7_cache):
    Store(db_path=v7_cache).migrate()
    reader = Store(db_path=v7_cache, read_only=True)
    assert reader.schema_version() == SCHEMA_VERSION
    assert reader.count_observations(QueryAST()) == 1


def test_a_second_migrate_is_a_no_op(v7_cache):
    Store(db_path=v7_cache).migrate()
    before = sqlite3.connect(v7_cache).execute(
        "SELECT type, name, sql FROM sqlite_master ORDER BY name"
    ).fetchall()
    Store(db_path=v7_cache).migrate()
    after = sqlite3.connect(v7_cache).execute(
        "SELECT type, name, sql FROM sqlite_master ORDER BY name"
    ).fetchall()
    assert after == before


def test_a_fresh_cache_has_no_vector_artifacts(tmp_path):
    store = Store(db_path=tmp_path / "cache.db")
    store.migrate()
    c = store._c()
    assert not {n for n in _names(c) if n.startswith("vec_") or "embedding" in n}
    for table in ("observations", "records"):
        columns = {row[1] for row in c.execute(f"PRAGMA table_info({table})")}
        assert "embedding_state" not in columns
