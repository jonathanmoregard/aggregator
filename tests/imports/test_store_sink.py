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


# -- the reordering is per-batch, and the contract has to say so -----------
#
# Round-1 MEDIUM: the sink's own comment read "an adapter is allowed to yield
# an observation before its session row (batch boundaries can split a stream
# anywhere)". That is true inside one batch and exactly backwards about batch
# boundaries: the runner flushes every 500 items and the sink has no memory
# between calls, so an observation landing in an earlier batch than its session
# hits the FK. Green in a small test, aborts the adapter on real volume.


def test_an_observation_ahead_of_its_session_across_batches_is_refused(
    tmp_data_home
):
    store = Store()
    store.migrate()
    sink = StoreSink(store)

    with pytest.raises(ValueError) as excinfo:
        sink.write([_obs("o1", "s1")])

    message = str(excinfo.value)
    assert "o1" in message, "must name the observation"
    assert "s1" in message, "must name the session it wanted"
    assert "before" in message.lower(), "must state the ordering rule"


def test_the_refusal_replaces_a_bare_foreign_key_error(tmp_data_home):
    """``FOREIGN KEY constraint failed`` names no row and no rule, so the
    adapter author has nothing to act on."""
    import sqlite3

    store = Store()
    store.migrate()

    with pytest.raises(ValueError):
        StoreSink(store).write([_obs("o1", "s1")])
    # And the store is untouched, not half-written.
    assert store.existing_ids("observations", ["o1"]) == set()
    assert not isinstance(sqlite3.IntegrityError("x"), ValueError)


def test_a_session_split_across_batches_is_legal(tmp_data_home):
    """The legal direction: a session's observations may land in later batches
    than the session row. That is what batch boundaries actually do to the
    shipped sources' streams, and it must keep working."""
    store = Store()
    store.migrate()
    sink = StoreSink(store)

    first = sink.write([_sess("s1"), _obs("o1", "s1")])
    second = sink.write([_obs("o2", "s1"), _obs("o3", "s1")])

    assert (first.added, second.added) == (2, 2)
    assert store.existing_ids("observations", ["o1", "o2", "o3"]) == {
        "o1",
        "o2",
        "o3",
    }


def test_the_shipped_stream_shape_survives_a_batch_boundary(tmp_data_home):
    """End to end through the runner at a batch size that splits every
    session — the volume case the per-batch reordering hid."""
    store = Store()
    store.migrate()

    class Sessions:
        name = "sessions"

        async def get_data(self) -> AsyncIterator[ImportItem]:
            for s in range(3):
                yield _sess(f"s{s}")
                for o in range(3):
                    yield _obs(f"s{s}-o{o}", f"s{s}")

    report = asyncio.run(run_imports([Sessions()], StoreSink(store), batch_size=2))

    assert report.ok is True, report.errors
    assert store.count_by_source("sessions") == 3
    assert report.adapters["sessions"].added == 12


def test_an_orphan_observation_is_isolated_to_its_own_adapter(tmp_data_home):
    """It is still just an adapter error: the runner contains it and the other
    sources complete."""
    store = Store()
    store.migrate()

    class Orphan:
        name = "orphan"

        async def get_data(self) -> AsyncIterator[ImportItem]:
            yield _obs("o1", "never-yielded")

    class Fine:
        name = "fine"

        async def get_data(self) -> AsyncIterator[ImportItem]:
            yield _rec("research:ok")

    report = asyncio.run(run_imports([Orphan(), Fine()], StoreSink(store)))

    assert report.failed_adapters == ["orphan"]
    assert store.count_by_source("research") == 1


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


# -- round 3 L1: the two paths counted differently ------------------------


def test_both_paths_satisfy_added_plus_updated_equals_items(tmp_data_home):
    """The identity a caller reads the summary with. It held on the records
    path and not on the entity path, which de-duplicated within a batch."""
    store = Store()
    store.migrate()
    sink = StoreSink(store)

    records = [_rec("research:a"), _rec("research:a"), _rec("research:b")]
    entities = [_sess("s1"), _sess("s1"), _obs("o1", "s1")]

    r = sink.write(records)
    e = sink.write(entities)

    assert r.added + r.updated == len(records)
    assert e.added + e.updated == len(entities)
    assert (r.skipped, e.skipped) == (0, 0)


def test_entity_totals_do_not_move_with_batch_size(tmp_data_home, tmp_path):
    """A repeat inside one batch used to be de-duplicated and a repeat across
    a flush boundary could not be, so the SAME stream reported
    added=2 updated=0 at batch_size=10 and added=2 updated=2 at 2. A number
    that changes with an unrelated tuning knob is not a report."""
    stream = [_sess("s1"), _obs("o1", "s1"), _sess("s1"), _obs("o1", "s1")]

    class _Adapter:
        name = "sessions"

        async def get_data(self) -> AsyncIterator[ImportItem]:
            for item in stream:
                yield item

    totals = []
    for size in (10, 2):
        store = Store(db_path=tmp_path / f"batch{size}.db")
        store.migrate()
        report = asyncio.run(
            run_imports([_Adapter()], StoreSink(store), batch_size=size)
        )
        totals.append((report.added, report.updated, report.skipped))

    assert totals[0] == totals[1] == (2, 2, 0)
