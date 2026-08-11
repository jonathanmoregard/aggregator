"""Round-2 LOW-2: the sink wrote before it validated.

``StoreSink.write`` wrote the batch's ``Record`` items, THEN reached the
entity path, where ``_refuse_orphan_observations`` raises. The raise discards
the return value, so the counts for the rows that DID land are lost: the
runner sees an exception, not a ``WriteCounts``, and the report never mentions
them. Rows in the store that no report counted is the shape this repo keeps
ruling out.

Validation now runs over the whole batch before anything is written, so a
batch that will be refused writes nothing at all.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from aggregator.core.store import Store
from aggregator.imports.store_sink import StoreSink
from aggregator.sources.base import ObservationRow, Record, SessionRow


def _store(tmp_path) -> Store:
    store = Store(db_path=tmp_path / "cache.db")
    store.migrate()
    return store


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


def _orphan(obs_id: str) -> ObservationRow:
    return ObservationRow(
        obs_id=obs_id,
        session_id="never-yielded",
        root_session_id="never-yielded",
        parent_obs_id=None,
        type="user",
        ts=datetime(2026, 8, 11, tzinfo=UTC),
        model=None,
        input_tokens=None,
        output_tokens=None,
        tool_name=None,
        tool_use_id=None,
        body="hi",
    )


def test_a_refused_batch_writes_nothing_at_all(tmp_path):
    """THE finding. The record landed before the orphan check ran, and the
    exception took its counts with it."""
    store = _store(tmp_path)
    sink = StoreSink(store)

    with pytest.raises(ValueError, match="which is neither in this batch"):
        sink.write(
            [
                Record(stable_id="t:1", source="t", subject="one", body="b"),
                _orphan("o1"),
            ]
        )

    assert store.count_by_source("t") == 0, "an uncounted row landed"


def test_a_valid_mixed_batch_still_writes_everything(tmp_path):
    """The other half: reordering the check must not cost a good batch."""
    store = _store(tmp_path)
    sink = StoreSink(store)

    counts = sink.write(
        [
            Record(stable_id="t:1", source="t", subject="one", body="b"),
            _session("s1"),
        ]
    )

    assert counts.added == 2
    assert store.count_by_source("t") == 1
    assert store.existing_ids("sessions", ["s1"]) == {"s1"}


def test_an_observation_whose_session_is_already_stored_is_accepted(tmp_path):
    """The check's real subject: a session from an EARLIER batch."""
    store = _store(tmp_path)
    sink = StoreSink(store)
    sink.write([_session("never-yielded")])

    counts = sink.write([_orphan("o1")])

    assert counts.added == 1
