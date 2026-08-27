"""v4 → v5: additive ALTER TABLE + vec virtual tables, no rebuild.

v5 is the RAG schema. It adds a nullable ``embedding_state`` column to
``observations`` and ``records`` and two sqlite-vec ``vec0`` virtual tables.

The property that matters most here is what it does NOT do. v4 is the
incremental-ingest schema — ``ingest_state`` (the per-source high-water mark),
``quarantine`` and ``poison_faults``. Those tables hold the only record of what
has already been ingested and what is permanently broken; a migration that
disturbed them would either re-ingest 372k observations from scratch or lose
the ledger of quiet faults. So every test below runs against a store that
already carries live v4 artefacts and asserts they come out the far side
untouched.
"""

import sqlite3

from aggregator.core.store import SCHEMA_VERSION, Store


def _drop_column_if_present(c: sqlite3.Connection, table: str, column: str) -> None:
    cols = {row[1] for row in c.execute(f"PRAGMA table_info({table})")}
    if column in cols:
        c.execute(f"ALTER TABLE {table} DROP COLUMN {column}")


def _make_v4_store(tmp_path, *, seed_v4_state: bool = True) -> Store:
    """Return a handle on a store in the v4 shape.

    Built by migrating to current and then removing exactly the v5 artefacts,
    so the fixture cannot drift away from what a real v4 database looks like:
    ``ingest_state`` / ``quarantine`` / ``poison_faults`` present, schema
    stamped 4, no ``embedding_state``, no vec tables.
    """
    db = tmp_path / "cache.db"
    s = Store(db_path=db)
    s.migrate()
    c = s._c()
    if seed_v4_state:
        # A live watermark and a live permanent fault — the two artefacts the
        # v5 migration must not disturb.
        s.advance_ingest_cursor(
            "sessions",
            cursor_kind="timestamp",
            cursor_value="2026-08-01T00:00:00+00:00",
            rows=372_000,
            at="2026-08-01T00:00:00+00:00",
        )
        s.record_fault(
            "sessions",
            "fault-1",
            scope="/tmp/broken.jsonl",
            scope_stamp="1-2",
            reason="malformed-json",
            detail="two unparseable lines",
            record_count=2,
            line="17",
            at="2026-08-01T00:00:00+00:00",
        )
    # v5 puts indexes ON embedding_state; SQLite DROP COLUMN does not cascade,
    # so the indexes go first or the DROP COLUMN errors out.
    c.execute("DROP INDEX IF EXISTS obs_embedding_state")
    c.execute("DROP INDEX IF EXISTS rec_embedding_state")
    # v6's column comes off too, or the fixture is not the v4 shape it claims
    # to be — and the migration under test would then never run the ALTER.
    c.execute("DROP INDEX IF EXISTS obs_provenance")
    _drop_column_if_present(c, "observations", "provenance")
    _drop_column_if_present(c, "observations", "embedding_state")
    _drop_column_if_present(c, "records", "embedding_state")
    c.execute("DROP TABLE IF EXISTS vec_observations")
    c.execute("DROP TABLE IF EXISTS vec_records")
    c.execute("PRAGMA user_version = 4")
    c.commit()
    s.close()
    return Store(db_path=db)


def _seed_observation(c: sqlite3.Connection, obs_id: str) -> None:
    c.execute(
        "INSERT OR IGNORE INTO sessions(session_id, root_session_id, kind, "
        "first_ts, last_ts, jsonl_path) VALUES ('sid', 'sid', 'session', "
        "'2026-01-01T00:00:00', '2026-01-01T00:00:00', '/tmp/x.jsonl')"
    )
    c.execute(
        "INSERT INTO observations(obs_id, session_id, root_session_id, "
        "type, ts, body) VALUES (?, 'sid', 'sid', 'user', "
        "'2026-01-01T00:00:00', 'some body text')",
        (obs_id,),
    )


def test_v5_migration_is_idempotent(tmp_path):
    s = _make_v4_store(tmp_path)
    s.migrate()
    s.migrate()  # second call must not fail
    assert s.schema_version() == SCHEMA_VERSION


def test_v5_migration_adds_columns(tmp_path):
    s = _make_v4_store(tmp_path)
    s.migrate()
    c = s._c()
    obs_cols = {row[1] for row in c.execute("PRAGMA table_info(observations)")}
    rec_cols = {row[1] for row in c.execute("PRAGMA table_info(records)")}
    assert "embedding_state" in obs_cols
    assert "embedding_state" in rec_cols


def test_v5_migration_creates_vec_tables(tmp_path):
    s = _make_v4_store(tmp_path)
    s.migrate()
    assert s.vector_available, "sqlite-vec must load for this test to mean anything"
    c = s._c()
    tables = {
        row[0]
        for row in c.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        )
    }
    assert "vec_observations" in tables
    assert "vec_records" in tables


def test_v5_migration_leaves_existing_rows_null(tmp_path):
    """Every pre-v5 row comes out QUEUED FOR BACKFILL, never silently done.

    ``embedding_state IS NULL`` is what ``select_unembedded`` looks for. A
    migration that backfilled a non-NULL default would mark 372k un-embedded
    observations as embedded, and the vector arm would then quietly return
    nothing for almost the whole corpus while looking healthy.
    """
    s = _make_v4_store(tmp_path)
    c = s._c()
    _seed_observation(c, "existing_obs")
    c.commit()
    s.migrate()
    row = c.execute(
        "SELECT embedding_state FROM observations WHERE obs_id = ?",
        ("existing_obs",),
    ).fetchone()
    assert row["embedding_state"] is None


def test_v5_bumps_pragma_user_version(tmp_path):
    s = _make_v4_store(tmp_path)
    assert s.schema_version() == 4
    s.migrate()
    assert s.schema_version() == SCHEMA_VERSION


def test_v5_migration_preserves_v4_watermark_and_faults(tmp_path):
    s = _make_v4_store(tmp_path)
    s.migrate()
    state = s.read_ingest_state("sessions")
    assert state["cursor_value"] == "2026-08-01T00:00:00+00:00"
    assert state["rows_seen"] == 372_000
    faults = s.read_faults("sessions")
    assert [f["fault_key"] for f in faults] == ["fault-1"]
    assert faults[0]["record_count"] == 2


def test_v5_migration_twice_on_populated_db_changes_nothing(tmp_path):
    """Second migrate() on a DB with rows must be a genuine no-op."""
    s = _make_v4_store(tmp_path)
    c = s._c()
    for i in range(5):
        _seed_observation(c, f"o{i}")
    c.commit()
    s.migrate()

    def snapshot():
        return {
            "obs": c.execute("SELECT COUNT(*) AS n FROM observations").fetchone()["n"],
            "unembedded": c.execute(
                "SELECT COUNT(*) AS n FROM observations WHERE embedding_state IS NULL"
            ).fetchone()["n"],
            "ingest_state": [dict(r) for r in c.execute("SELECT * FROM ingest_state")],
            "faults": [dict(r) for r in c.execute("SELECT * FROM poison_faults")],
            "quarantine": [dict(r) for r in c.execute("SELECT * FROM quarantine")],
            "version": s.schema_version(),
        }

    before = snapshot()
    s.migrate()
    assert snapshot() == before
    assert before["obs"] == before["unembedded"] == 5
