"""Tests for aggregator.imports.store_sink — the real write target.

Proves the counts in a run report are the store's truth, not the batch
length, and that one sink handles BOTH existing write paths without the two
ontologies being collapsed into one shape.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest

from aggregator.core.store import Store
from aggregator.imports.port import ImportItem, ImportSink
from aggregator.imports.runner import run_imports
from aggregator.imports.store_sink import StoreSink
from aggregator.sources.base import ObservationRow, Record, SessionRow

NOW = datetime(2026, 8, 11, tzinfo=UTC)


def _rec(sid: str, body: str = "b") -> Record:
    return Record(
        stable_id=sid,
        source="research",
        subject="s",
        body=body,
        created_at=NOW,
        updated_at=NOW,
    )


def _sess(sid: str) -> SessionRow:
    return SessionRow(
        session_id=sid,
        root_session_id=sid,
        parent_session_id=None,
        kind="session",
        agent_id=None,
        agent_type=None,
        spawned_by_tool_use_id=None,
        cwd=None,
        git_branch=None,
        first_ts=NOW,
        last_ts=NOW,
        jsonl_path=f"/tmp/{sid}.jsonl",
    )


def _obs(oid: str, sid: str) -> ObservationRow:
    return ObservationRow(
        obs_id=oid,
        session_id=sid,
        root_session_id=sid,
        parent_obs_id=None,
        type="user",
        ts=NOW,
        model=None,
        input_tokens=None,
        output_tokens=None,
        tool_name=None,
        tool_use_id=None,
        body="hello",
    )


def test_StoreSink_satisfies_the_sink_port(tmp_data_home):
    store = Store()
    store.migrate()
    assert isinstance(StoreSink(store), ImportSink)


def test_records_are_written_and_counted_as_added_then_updated(tmp_data_home):
    store = Store()
    store.migrate()
    sink = StoreSink(store)

    first = sink.write([_rec("research:a"), _rec("research:b")])
    assert (first.added, first.updated, first.skipped) == (2, 0, 0)

    # Same ids again -> updates, not adds. This is the number cli.py gets
    # wrong today (it prints added=len(records) updated=0 every run).
    second = sink.write([_rec("research:a", body="changed"), _rec("research:c")])
    assert (second.added, second.updated, second.skipped) == (1, 1, 0)

    assert store.count_by_source("research") == 3


def test_entities_are_written_through_the_entity_path(tmp_data_home):
    store = Store()
    store.migrate()
    sink = StoreSink(store)

    first = sink.write([_sess("s1"), _obs("o1", "s1")])
    assert (first.added, first.updated) == (2, 0)

    second = sink.write([_sess("s1"), _obs("o2", "s1")])
    assert (second.added, second.updated) == (1, 1)

    assert store.count_by_source("sessions") == 1


def test_a_mixed_batch_lands_in_both_ontologies(tmp_data_home):
    """The two shapes are not collapsed — one batch can carry both and each
    item goes to the write path that fits it."""
    store = Store()
    store.migrate()
    sink = StoreSink(store)

    counts = sink.write([_rec("research:a"), _sess("s1"), _obs("o1", "s1")])

    assert (counts.added, counts.updated) == (3, 0)
    assert store.count_by_source("research") == 1
    assert store.count_by_source("sessions") == 1


def test_session_rows_are_written_before_their_observations(tmp_data_home):
    """The observations FK points at sessions; a batch that lists the
    observation first must not blow up on ordering."""
    store = Store()
    store.migrate()
    sink = StoreSink(store)

    counts = sink.write([_obs("o1", "s1"), _sess("s1")])

    assert counts.added == 2


def test_an_unknown_item_type_fails_loudly(tmp_data_home):
    store = Store()
    store.migrate()
    with pytest.raises(TypeError, match="unsupported import item"):
        StoreSink(store).write(["not an item"])


def test_end_to_end_runner_over_the_real_store(tmp_data_home):
    """The seam, wired: two adapters -> runner -> StoreSink -> SQLite, with
    one of them failing and the other's rows still landing."""
    store = Store()
    store.migrate()

    class Good:
        name = "good"

        async def get_data(self) -> AsyncIterator[ImportItem]:
            yield _rec("research:x")
            yield _rec("research:y")

    class Bad:
        name = "bad"

        async def get_data(self) -> AsyncIterator[ImportItem]:
            yield _rec("research:z")
            raise RuntimeError("upstream 500")

    report = asyncio.run(run_imports([Good(), Bad()], StoreSink(store)))

    assert store.count_by_source("research") == 3
    assert report.adapters["good"].added == 2
    assert report.adapters["bad"].added == 1
    assert report.ok is False
    assert report.failed_adapters == ["bad"]
