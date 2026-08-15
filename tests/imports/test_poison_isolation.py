"""One bad record must not cost the run, and must not be retried forever.

TWO FAILURES AVOIDED AT THE SAME TIME, and it is easy to fix one by causing
the other. Aborting a 372k-row stream because one row will not write is
absurd; so is re-scrubbing and re-attempting that row on every 30-minute tick
until somebody notices. The answer is the local form of a dead-letter queue: a
table in the same database, an attempt count, a jittered next-attempt time, and
a terminal state that is still COUNTED rather than deleted — because a failure
nobody can count is a gap in the index that reads as full coverage, which is
the one failure mode this repo keeps ruling out.

The third case is the one that makes the first two safe: if EVERY record in a
chunk fails, the sink is broken rather than the data, and the run must say so
instead of quietly setting a chunk's worth of good rows aside.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest

from aggregator.core.store import Store
from aggregator.imports.ingest_state import POISON_MAX_ATTEMPTS, PoisonLedger, Watermarks
from aggregator.imports.port import ImportItem, WriteCounts
from aggregator.imports.runner import run_imports
from aggregator.imports.store_sink import StoreSink
from aggregator.sources.base import Record

NOW = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "cache.db")
    s.migrate()
    yield s
    s.close()


def _rec(n: int) -> Record:
    return Record(
        stable_id=f"dropbox:{n}",
        source="dropbox",
        subject=f"s{n}",
        body=f"b{n}",
        updated_at=NOW - timedelta(days=n),
    )


class ListAdapter:
    name = "dropbox"

    def __init__(self, records: list[Record]) -> None:
        self._records = records

    async def get_data(self) -> AsyncIterator[ImportItem]:
        for r in self._records:
            yield r


class PoisonSink(StoreSink):
    """Refuses exactly one record, however it is batched."""

    def __init__(self, store: Store, bad: str) -> None:
        super().__init__(store)
        self.bad = bad
        self.calls = 0

    def write(self, items):
        self.calls += 1
        if any(getattr(i, "stable_id", "") == self.bad for i in items):
            raise ValueError(f"{self.bad} is malformed")
        return super().write(items)

    def write_checkpoint(self, items, checkpoint):
        self.calls += 1
        if any(getattr(i, "stable_id", "") == self.bad for i in items):
            raise ValueError(f"{self.bad} is malformed")
        return super().write_checkpoint(items, checkpoint)


def drive(store: Store, adapter, sink, marks: Watermarks, ledger: PoisonLedger):
    return asyncio.run(
        run_imports(
            [adapter],
            sink,
            batch_size=5,
            watermarks=marks,
            poison=ledger,
        )
    )


def test_one_bad_record_does_not_abort_the_stream(store):
    """The other nine records in the chunk land, and the run keeps going."""
    marks, ledger = Watermarks(store), PoisonLedger(store)
    sink = PoisonSink(store, "dropbox:3")
    report = drive(store, ListAdapter([_rec(i) for i in range(10)]), sink, marks, ledger)

    assert store.count_by_source("dropbox") == 9
    assert report.adapters["dropbox"].quarantined == 1
    # LOUD: a set-aside record is data that never reached the index.
    assert not report.adapters["dropbox"].ok
    assert any("dropbox:3" in e for e in report.adapters["dropbox"].errors)


def test_a_set_aside_record_is_not_attempted_again_next_run(store):
    """The "must not be retried forever" half.

    Without this a poison record is re-read, re-scrubbed and re-attempted on
    every single tick — which is the doom loop again, at one row instead of
    372k, and just as permanent.
    """
    marks, ledger = Watermarks(store), PoisonLedger(store)
    records = [_rec(i) for i in range(10)]
    first = PoisonSink(store, "dropbox:3")
    drive(store, ListAdapter(records), first, marks, ledger)

    second = PoisonSink(store, "dropbox:3")
    report = drive(store, ListAdapter(records), second, marks, ledger)

    # It never reached the sink at all this time.
    assert "dropbox:3" in ledger.held("dropbox")
    assert report.adapters["dropbox"].quarantined == 1
    assert any("earlier run" in e for e in report.adapters["dropbox"].errors)


def test_it_goes_terminal_rather_than_retrying_indefinitely(store):
    """After the attempt cap the record is never retried — and never deleted."""
    ledger = PoisonLedger(store)
    entry = None
    for _ in range(POISON_MAX_ATTEMPTS):
        entry = ledger.hold("dropbox", "dropbox:3", ValueError("bad"), previous=entry)
    assert entry is not None
    assert entry.terminal
    summary = ledger.summary()
    assert summary == [
        {
            "source": "dropbox",
            "error_type": "ValueError",
            "terminal": True,
            "count": 1,
        }
    ]


def test_a_record_that_starts_working_is_released(store):
    """A transient fault must not condemn a record for good."""
    marks, ledger = Watermarks(store), PoisonLedger(store)
    records = [_rec(i) for i in range(4)]
    drive(store, ListAdapter(records), PoisonSink(store, "dropbox:2"), marks, ledger)
    assert "dropbox:2" in ledger.held("dropbox")

    # Pretend the next attempt is due, then run with a healthy sink.
    store._c().execute(
        "UPDATE quarantine SET next_retry_at = ? WHERE record_key = 'dropbox:2'",
        ["2000-01-01T00:00:00+00:00"],
    )
    store._c().commit()
    drive(store, ListAdapter(records), StoreSink(store), marks, ledger)

    assert ledger.held("dropbox") == {}
    assert store.count_by_source("dropbox") == 4


def test_a_chunk_where_everything_fails_is_a_broken_sink_not_bad_data(store):
    """The distinction that keeps per-record isolation from hiding an outage.

    If a disk is full or a schema is wrong, every row in the chunk fails
    individually — and setting all of them aside would turn a total outage into
    a quiet "some records were odd" line. So a chunk with no survivors re-raises,
    the adapter reports the real fault, and nothing is set aside.
    """

    class DeadSink(StoreSink):
        def write(self, items):
            raise OSError("no space left on device")

        def write_checkpoint(self, items, checkpoint):
            raise OSError("no space left on device")

    marks, ledger = Watermarks(store), PoisonLedger(store)
    report = drive(
        store, ListAdapter([_rec(i) for i in range(4)]), DeadSink(store), marks, ledger
    )

    entry = report.adapters["dropbox"]
    assert not entry.ok
    assert entry.quarantined == 0
    assert ledger.held("dropbox") == {}
    assert any("no space left on device" in e for e in entry.errors)


def test_a_transient_failure_is_retried_before_anything_is_set_aside(store):
    """``database is locked`` is a wait, not a poison record."""
    attempts = {"n": 0}

    class FlakySink(StoreSink):
        def write_checkpoint(self, items, checkpoint):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("database is locked")
            return super().write_checkpoint(items, checkpoint)

    marks, ledger = Watermarks(store), PoisonLedger(store)
    report = drive(
        store, ListAdapter([_rec(i) for i in range(3)]), FlakySink(store), marks, ledger
    )

    assert attempts["n"] == 2
    assert report.adapters["dropbox"].ok
    assert report.adapters["dropbox"].quarantined == 0
    assert store.count_by_source("dropbox") == 3


def test_an_isolated_chunk_does_not_advance_the_mark(store):
    """A pass that set something aside has not fully stored its window.

    Advancing past a record that was never written is how it becomes
    permanently invisible; standing still costs one re-read.
    """
    marks, ledger = Watermarks(store), PoisonLedger(store)
    drive(
        store,
        ListAdapter([_rec(i) for i in range(3)]),
        PoisonSink(store, "dropbox:1"),
        marks,
        ledger,
    )
    assert marks.plan("dropbox").cursor_value is None


def test_a_sink_that_cannot_checkpoint_still_records_the_mark(store):
    """The mechanism must work with ANY sink, not only the one that ships.

    A sink without the optional protocol cannot make the chunk and the mark one
    transaction, so the runner falls back to the safe ORDER instead: the chunk
    is already committed, then the mark is written. A crash between the two
    costs a re-read rather than the records — which is the whole reason that
    order, and not the other one, is the acceptable fallback.
    """

    class PlainSink:
        def __init__(self) -> None:
            self.items: list[ImportItem] = []

        def write(self, items):
            self.items.extend(items)
            return WriteCounts(added=len(items))

    marks = Watermarks(store)
    sink = PlainSink()
    report = asyncio.run(
        run_imports([ListAdapter([_rec(0)])], sink, watermarks=marks)
    )
    assert report.adapters["dropbox"].ok
    assert len(sink.items) == 1
    assert marks.plan("dropbox").cursor_value == NOW
