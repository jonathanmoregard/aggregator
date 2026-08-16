"""Round-2 LOW-1: a write that raised mid-loop left its rows pending.

``sqlite3`` opens an implicit transaction before the first DML statement and
holds it until someone commits or rolls back. ``upsert_entities`` (and
``upsert``) executed a loop and committed at the end, with no rollback on the
way out — so a batch that raised part-way through left its already-executed
INSERTs sitting in an open transaction on the SHARED connection.

The next writer's ``COMMIT`` then committed them. The rows land, and nothing
counts them: the sink establishes counts before the write and the runner
discards them when the write raises. Rows in the store that no report ever
mentioned is the exact shape this repo keeps ruling out.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from aggregator.core.store import Store
from aggregator.sources.base import Record, SessionRow


def _session(sid: str) -> SessionRow:
    now = datetime(2026, 8, 11, tzinfo=UTC)
    return SessionRow(
        session_id=sid,
        root_session_id=sid,
        parent_session_id=None,
        kind="session",
        agent_id=None,
        agent_type=None,
        spawned_by_tool_use_id=None,
        cwd="/tmp",
        git_branch=None,
        first_ts=now,
        last_ts=now,
        jsonl_path=f"/tmp/{sid}.jsonl",
    )


class _ExplodingRecord(Record):
    """Raises from inside the write loop, after earlier rows are executed."""

    @property
    def body(self):  # type: ignore[override]
        raise RuntimeError("scrub blew up")

    @body.setter
    def body(self, value):
        pass


def test_a_failed_entity_batch_is_not_committed_by_the_next_one(tmp_path):
    """THE finding. The second write's COMMIT used to publish the first
    write's partial rows, which no report ever counted."""
    db = tmp_path / "cache.db"
    store = Store(db_path=db)
    store.migrate()

    with pytest.raises(TypeError):
        store.upsert_entities([_session("s1"), object()])
    store.upsert_entities([_session("s2")])

    # A SEPARATE connection: only committed rows are visible here, which is
    # the question — the writing connection sees its own open transaction.
    reader = Store(db_path=db)
    assert reader.existing_ids("sessions", ["s1", "s2"]) == {"s2"}


def test_a_failed_record_batch_is_not_committed_by_the_next_one(tmp_path):
    """Same hole one function away, on the Record path."""
    db = tmp_path / "cache.db"
    store = Store(db_path=db)
    store.migrate()

    good = Record(stable_id="t:1", source="t", subject="one", body="b")
    boom = _ExplodingRecord(stable_id="t:2", source="t", subject="two", body="b")
    with pytest.raises(RuntimeError):
        store.upsert([good, boom])
    store.upsert([Record(stable_id="t:3", source="t", subject="three", body="b")])

    reader = Store(db_path=db)
    assert reader.existing_ids("records", ["t:1", "t:2", "t:3"]) == {"t:3"}


def test_a_clean_batch_still_commits(tmp_path):
    """The other half — the rollback must not cost a good write its commit."""
    db = tmp_path / "cache.db"
    store = Store(db_path=db)
    store.migrate()

    store.upsert_entities([_session("s1")])
    store.upsert([Record(stable_id="t:1", source="t", subject="one", body="b")])

    reader = Store(db_path=db)
    assert reader.existing_ids("sessions", ["s1"]) == {"s1"}
    assert reader.existing_ids("records", ["t:1"]) == {"t:1"}


def test_the_nested_rebuild_path_is_unaffected(tmp_path):
    """``rebuild_and_upsert_entities`` calls ``upsert_entities(_commit=False)``
    inside a SAVEPOINT and does its own ROLLBACK TO SAVEPOINT. A connection-wide
    rollback there would destroy the savepoint the caller is about to release,
    so the nested call must not roll back."""
    db = tmp_path / "cache.db"
    store = Store(db_path=db)
    store.migrate()
    store.upsert_entities([_session("keep")])

    with pytest.raises(TypeError):
        store.rebuild_and_upsert_entities([_session("s1"), object()])

    reader = Store(db_path=db)
    assert reader.existing_ids("sessions", ["keep", "s1"]) == {"keep"}
