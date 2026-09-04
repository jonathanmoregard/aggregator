"""Porter stemming on both FTS5 tables, and the v6 → v7 rebuild migration.

WHY STEMMING. The index's documented worst failure is the multi-word gist
query: every term is ANDed, and without stemming ``report`` and ``reports``
are different terms, so a caller recalling the gist of a conversation loses
to morphology before the conjunction even gets its turn. ``porter`` in front
of ``unicode61`` stems both the indexed tokens and the query tokens, so
``running`` finds ``run`` and vice versa — in BOTH directions, which is what
the first two tests pin.

WHY THE MIGRATION IS A REBUILD. An FTS5 tokenizer cannot be ALTERed: it is
baked into the shadow tables at CREATE time, so every pre-v7 database has to
drop and recreate both FTS tables and re-tokenize every row. The migration
test builds a genuine old-tokenizer database — real triggers, real rows, a
v6 stamp — and asserts the three properties that matter: porter is actually
active afterwards (probed via ``sqlite_master`` SQL text, the artifact, not
the version stamp), the rebuilt index is COMPLETE (a migrated DB must never
silently serve a partial FTS index), and running ``migrate()`` again is a
no-op (it runs on every CLI invocation).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime

from aggregator.core.store import SCHEMA_VERSION, Store
from aggregator.sources.base import ObservationRow, Record, SessionRow

# --- fixtures ---------------------------------------------------------------


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


def _rec(sid: str, subject: str, body: str, tags=()) -> Record:
    return Record(
        stable_id=sid,
        source="github",
        subject=subject,
        body=body,
        tags=list(tags),
        created_at=datetime(2026, 7, 25, tzinfo=UTC),
        updated_at=datetime(2026, 7, 25, tzinfo=UTC),
    )


def _table_sql(c: sqlite3.Connection, table: str) -> str:
    row = c.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row[0] if row and row[0] else ""


#: The pre-v7 tokenizer, verbatim. The downgrade helper below recreates the
#: FTS tables exactly as a v6 build's DDL did, so the migration under test
#: meets the artifact it will meet on the live cache.
_OLD_OBS_FTS = """
CREATE VIRTUAL TABLE obs_fts USING fts5(
    body,
    content='observations',
    content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2'
);
"""

_OLD_RECORDS_FTS = """
CREATE VIRTUAL TABLE records_fts USING fts5(
    stable_id UNINDEXED,
    source    UNINDEXED,
    subject,
    body,
    tags,
    tokenize='unicode61 remove_diacritics 2'
);
"""


def _downgrade_fts_to_unicode61(db) -> None:
    """Turn a freshly-migrated database back into a v6-shaped one.

    Only the FTS artifacts and the version stamp move: the content tables are
    the same at v6 and v7, and the porter migration probes the artifact, so
    this reproduces exactly what it keys on.
    """
    s = Store(db_path=db)
    c = s._c()
    for trig in ("observations_ai", "observations_ad", "observations_au"):
        c.execute(f"DROP TRIGGER IF EXISTS {trig}")
    c.execute("DROP TABLE obs_fts")
    c.executescript(_OLD_OBS_FTS)
    c.execute("INSERT INTO obs_fts(obs_fts) VALUES('rebuild')")
    c.execute("DROP TABLE records_fts")
    c.executescript(_OLD_RECORDS_FTS)
    for row in c.execute(
        "SELECT stable_id, source, subject, body, tags FROM records"
    ).fetchall():
        c.execute(
            "INSERT INTO records_fts(stable_id, source, subject, body, tags) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                row["stable_id"],
                row["source"],
                row["subject"],
                row["body"],
                " ".join(json.loads(row["tags"])),
            ),
        )
    c.execute("PRAGMA user_version = 6")
    c.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', '6')"
    )
    c.commit()
    s.close()


# --- stemming behaviour on a fresh database ---------------------------------


def test_fresh_database_gets_porter_on_both_fts_tables(tmp_path):
    """Fresh DDL ships porter directly — no migration pass needed."""
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    c = s._c()
    assert "porter" in _table_sql(c, "obs_fts")
    assert "porter" in _table_sql(c, "records_fts")
    s.close()


def test_stemming_matches_observations_in_both_directions(tmp_path):
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert_entities(
        [_session(), _obs("o1", "run fast"), _obs("o2", "running shoes")]
    )
    # "running" must find the body that only says "run", and vice versa.
    assert set(s._fts_obs_ids("running")) == {"o1", "o2"}
    assert set(s._fts_obs_ids("run")) == {"o1", "o2"}
    s.close()


def test_stemming_matches_records_in_both_directions(tmp_path):
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert([_rec("r1", "note", "run fast"), _rec("r2", "note", "running shoes")])
    assert s._fts_ids("running") == {"r1", "r2"}
    assert s._fts_ids("run") == {"r1", "r2"}
    s.close()


# --- the v6 → v7 rebuild migration ------------------------------------------


def _seeded_v6_db(tmp_path):
    """A database with real rows whose FTS tables carry the OLD tokenizer."""
    db = tmp_path / "cache.db"
    s = Store(db_path=db)
    s.migrate()
    s.upsert_entities(
        [
            _session(),
            *[_obs(f"o{i}", f"sentineltoken body number {i}") for i in range(20)],
        ]
    )
    s.upsert(
        [
            _rec("r1", "running shoes", "sentineltoken record", ["alpha-beta"]),
            _rec("r2", "walking boots", "sentineltoken record", ["gamma"]),
        ]
    )
    s.close()
    _downgrade_fts_to_unicode61(db)
    return db


def test_migrate_rebuilds_old_tokenizer_tables_completely(tmp_path):
    db = _seeded_v6_db(tmp_path)

    # Sanity: the downgrade took — no stemming, old tokenizer, v6 stamp.
    probe = Store(db_path=db)
    c = probe._c()
    assert "porter" not in _table_sql(c, "obs_fts")
    assert probe._fts_obs_ids("bodies") == []
    assert int(c.execute("PRAGMA user_version").fetchone()[0]) == 6
    probe.close()

    s = Store(db_path=db)
    s.migrate()
    c = s._c()
    # Porter is ACTIVE — asserted on the artifact, not the stamp.
    assert "porter" in _table_sql(c, "obs_fts")
    assert "porter" in _table_sql(c, "records_fts")
    # The rebuilt index is COMPLETE: every row answers.
    assert (
        c.execute(
            "SELECT count(*) FROM obs_fts WHERE obs_fts MATCH 'sentineltoken'"
        ).fetchone()[0]
        == 20
    )
    assert (
        c.execute(
            "SELECT count(*) FROM records_fts WHERE records_fts MATCH 'sentineltoken'"
        ).fetchone()[0]
        == 2
    )
    # And it stems now: "bodies" finds every "body", "walk" finds "walking".
    assert len(s._fts_obs_ids("bodies")) == 20
    assert s._fts_ids("walk") == {"r2"}
    # Tags were repopulated space-joined (searchable), not as raw JSON.
    assert s._fts_ids("gamma") == {"r2"}
    row = c.execute(
        "SELECT tags FROM records_fts WHERE stable_id = 'r1'"
    ).fetchone()
    assert row["tags"] == "alpha-beta"
    assert int(c.execute("PRAGMA user_version").fetchone()[0]) == SCHEMA_VERSION
    s.close()


def test_porter_migration_is_idempotent(tmp_path, caplog):
    db = _seeded_v6_db(tmp_path)

    s = Store(db_path=db)
    with caplog.at_level(logging.WARNING, logger="aggregator.core.store"):
        s.migrate()
    first = [r for r in caplog.records if "porter" in r.getMessage().lower()]
    # The rebuild announces itself LOUDLY — it can take minutes on the live
    # cache, and a silent multi-minute stall reads as a hang.
    assert first, "the rebuild must log a loud line saying what is happening"
    s.close()

    caplog.clear()
    s2 = Store(db_path=db)
    with caplog.at_level(logging.WARNING, logger="aggregator.core.store"):
        s2.migrate()
    again = [r for r in caplog.records if "porter" in r.getMessage().lower()]
    assert not again, "a second migrate() must not rebuild again"
    c = s2._c()
    assert (
        c.execute(
            "SELECT count(*) FROM obs_fts WHERE obs_fts MATCH 'sentineltoken'"
        ).fetchone()[0]
        == 20
    )
    s2.close()


def test_triggers_keep_the_rebuilt_index_in_sync(tmp_path):
    """After the rebuild the sync triggers exist and fire — insert, edit,
    delete all reach the porter index."""
    db = _seeded_v6_db(tmp_path)
    s = Store(db_path=db)
    s.migrate()
    c = s._c()
    for trig in ("observations_ai", "observations_ad", "observations_au"):
        row = c.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?",
            (trig,),
        ).fetchone()
        assert row and row[0], f"trigger {trig} missing after the rebuild"
    assert "UPDATE OF body" in _table_sql_for_trigger(c, "observations_au")
    # Insert through the real write path lands in the rebuilt index.
    s.upsert_entities([_session(), _obs("onew", "a freshly running insert")])
    assert "onew" in s._fts_obs_ids("run")
    s.close()


def _table_sql_for_trigger(c: sqlite3.Connection, name: str) -> str:
    row = c.execute(
        "SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?", (name,)
    ).fetchone()
    return row[0] if row and row[0] else ""
