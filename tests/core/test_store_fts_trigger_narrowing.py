"""``observations_au`` fires on a body edit, and on nothing else.

WHY THIS IS ITS OWN CHANGE. ``observations_au`` was ``AFTER UPDATE ON
observations`` with no column list, so *any* write to *any* column — a
provenance stamp, an ``embedding_state`` flip — did a full FTS5 ``'delete'``
plus re-insert of the whole body. With ``auto_vacuum=0`` the freed pages are
never handed back (the mechanism already documented at ``store.py`` around the
ingest UPSERT), so the cost is permanent file growth on a 1.4 GB database.

Measured on a throwaway database at 1/5 of live scale: a chunked column-only
UPDATE ran 87.5 s and grew the file 128 MB under the wide trigger, and 7.4 s
with no growth at all under the narrowed one — 12x, and several hundred
megabytes when extrapolated to the live 549,952 rows.

THE SAFETY ARGUMENT, PINNED BY :func:`test_upsert_keeps_the_fts_index_correct`.
SQLite fires ``UPDATE OF body`` whenever ``body`` appears in the SET clause,
whether or not the value actually moved, and the ingest UPSERT always names
``body = excluded.body``. So the index stays exactly as correct as it was. That
sentence is the whole justification for the narrowing, which is why a test
matches a sentinel token through the real upsert path rather than trusting it.
"""

import sqlite3
from datetime import UTC, datetime

from aggregator.core.store import Store
from aggregator.sources.base import ObservationRow, SessionRow

_WIDE_TRIGGER = """
CREATE TRIGGER observations_au AFTER UPDATE ON observations
BEGIN
    INSERT INTO obs_fts(obs_fts, rowid, body) VALUES ('delete', old.rowid, old.body);
    INSERT INTO obs_fts(rowid, body) VALUES (new.rowid, new.body);
END;
"""


def _session(session_id: str = "s1") -> SessionRow:
    return SessionRow(
        session_id=session_id,
        root_session_id=session_id,
        parent_session_id=None,
        kind="session",
        agent_id=None,
        agent_type=None,
        spawned_by_tool_use_id=None,
        cwd="/x",
        git_branch="main",
        first_ts=datetime(2026, 7, 25, 10, 0, tzinfo=UTC),
        last_ts=datetime(2026, 7, 25, 10, 5, tzinfo=UTC),
        jsonl_path="/tmp/x.jsonl",
    )


def _obs(obs_id: str, body: str, session_id: str = "s1") -> ObservationRow:
    return ObservationRow(
        obs_id=obs_id,
        session_id=session_id,
        root_session_id=session_id,
        parent_obs_id=None,
        type="user",
        ts=datetime(2026, 7, 25, 10, 0, tzinfo=UTC),
        model=None,
        input_tokens=None,
        output_tokens=None,
        tool_name=None,
        tool_use_id=None,
        body=body,
    )


def _trigger_sql(c: sqlite3.Connection) -> str:
    row = c.execute(
        "SELECT sql FROM sqlite_master WHERE type='trigger' AND name='observations_au'"
    ).fetchone()
    return row[0] if row else ""


def _fts_segments(c: sqlite3.Connection) -> int:
    """Rows in the FTS5 storage table — an FTS write appends to it."""
    return c.execute("SELECT count(*) FROM obs_fts_data").fetchone()[0]


def test_trigger_is_scoped_to_the_body_column(tmp_path):
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    assert "UPDATE OF body" in _trigger_sql(s._c())
    s.close()


def test_migrate_replaces_an_existing_wide_trigger(tmp_path):
    """``CREATE TRIGGER IF NOT EXISTS`` will NOT replace it — so DROP first.

    Every database that already exists carries the wide trigger, and they are
    the ones the narrowing is FOR. A migration that only ships the narrow form
    to fresh files fixes nothing on the 1.4 GB cache this was measured against.
    """
    db = tmp_path / "cache.db"
    s = Store(db_path=db)
    s.migrate()
    c = s._c()
    c.execute("DROP TRIGGER observations_au")
    c.executescript(_WIDE_TRIGGER)
    c.commit()
    assert "UPDATE OF body" not in _trigger_sql(c)
    s.close()

    s2 = Store(db_path=db)
    s2.migrate()
    assert "UPDATE OF body" in _trigger_sql(s2._c())
    s2.close()


def test_a_non_body_update_does_not_rewrite_the_fts_index(tmp_path):
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert_entities(
        [_session(), *[_obs(f"o{i}", f"row {i} sentinelalpha") for i in range(200)]]
    )
    c = s._c()
    before = _fts_segments(c)
    for i in range(200):
        c.execute(
            "UPDATE observations SET embedding_state = 'ok' WHERE obs_id = ?",
            (f"o{i}",),
        )
    c.commit()
    assert _fts_segments(c) == before
    # And the index still answers, i.e. nothing was skipped INTO correctness.
    assert (
        c.execute(
            "SELECT count(*) FROM obs_fts WHERE obs_fts MATCH 'sentinelalpha'"
        ).fetchone()[0]
        == 200
    )
    s.close()


def test_a_body_update_still_reindexes(tmp_path):
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert_entities([_session(), _obs("o1", "sentinelalpha here")])
    c = s._c()
    c.execute("UPDATE observations SET body = 'sentinelbeta here' WHERE obs_id = 'o1'")
    c.commit()
    assert (
        c.execute(
            "SELECT count(*) FROM obs_fts WHERE obs_fts MATCH 'sentinelbeta'"
        ).fetchone()[0]
        == 1
    )
    assert (
        c.execute(
            "SELECT count(*) FROM obs_fts WHERE obs_fts MATCH 'sentinelalpha'"
        ).fetchone()[0]
        == 0
    )
    s.close()


def test_upsert_keeps_the_fts_index_correct(tmp_path):
    """The §3.3 claim, matched through the REAL ingest path, not asserted.

    The ingest UPSERT names ``body = excluded.body`` unconditionally, so an
    edited row's new text has to be findable and its old text gone — under the
    narrowed trigger exactly as under the wide one.
    """
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert_entities([_session(), _obs("o1", "sentinelalpha in the first body")])
    s.upsert_entities([_session(), _obs("o1", "sentinelbeta in the edited body")])
    c = s._c()
    assert [
        r[0]
        for r in c.execute(
            "SELECT o.obs_id FROM obs_fts f JOIN observations o ON o.rowid = f.rowid "
            "WHERE obs_fts MATCH 'sentinelbeta'"
        )
    ] == ["o1"]
    assert (
        c.execute(
            "SELECT count(*) FROM obs_fts WHERE obs_fts MATCH 'sentinelalpha'"
        ).fetchone()[0]
        == 0
    )
    s.close()
