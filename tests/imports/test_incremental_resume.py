"""The two guards that would have caught the doom loop, and the shutdown fix.

WHAT HAPPENED. ``aggregator ingest --all`` never computed a watermark:
``cli.py`` set ``since`` only from an explicit ``--since``, so
``default_adapters(since=None)`` handed every one of nine sources a full scan
on every 30-minute tick. 372k observations at 827 rows/min is an ETA of 04:55
against a ``TimeoutStartSec=4h`` that fired at 01:58:58 — SIGTERM at ~44%, the
work discarded, the timer refiring half an hour later. Forever.

Every test that existed at the time passed. They asserted that a run imported
its records, isolated its failures and reported its counts, and all of that was
true on every doomed tick. What none of them asserted is the property that was
actually broken:

    THE SECOND RUN MUST DO SUBSTANTIALLY LESS WORK THAN THE FIRST.

and its companion, which is what makes a 4-hour timeout survivable:

    A KILL MID-RUN MUST LEAVE STATE THE NEXT RUN CAN RESUME FROM.

Both are pinned here against a real ``Store``, a real ``StoreSink`` and real
watermarks — not against doubles — because every link in that chain
(watermark -> since -> the source's own filter -> chunk commit -> watermark)
had to work for the bug to be fixed, and a double anywhere in it would have
been equally green on 2026-08-15.
"""
from __future__ import annotations

import asyncio
import signal
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest

from aggregator.core.store import Store
from aggregator.imports.ingest_state import (
    MODIFIED_TIME_OVERLAP,
    PoisonLedger,
    Watermarks,
)
from aggregator.imports.port import Checkpoint, ImportItem
from aggregator.imports.runner import graceful_shutdown, run_imports
from aggregator.imports.store_sink import StoreSink
from aggregator.sources.base import Record

NOW = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "cache.db")
    s.migrate()
    yield s
    s.close()


class WindowedAdapter:
    """An adapter that honours ``since`` exactly as the shipped sources do.

    Every records-shaped source in this repo skips an item whose
    ``updated_at`` is at or below ``since`` — dropbox and research and
    sota-watch on file mtime, substack on a zip member's mtime, github via the
    API's own ``updated:>=`` qualifier. Reproducing that rule rather than
    stubbing it is the point: the bug was never in a source's filter, it was
    that nothing ever gave the filter a value.
    """

    def __init__(self, name: str, records: list[Record], since: datetime | None):
        self.name = name
        self._records = records
        self._since = since
        self.emitted = 0
        self.filtered = 0

    async def get_data(self) -> AsyncIterator[ImportItem]:
        for r in self._records:
            if self._since is not None and r.updated_at <= self._since:
                self.filtered += 1
                continue
            self.emitted += 1
            yield r


def corpus(source: str, count: int, *, newest: datetime = NOW) -> list[Record]:
    """``count`` records, one per day going back from ``newest``."""
    return [
        Record(
            stable_id=f"{source}:{i}",
            source=source,
            subject=f"item {i}",
            body=f"body {i}",
            created_at=newest - timedelta(days=i),
            updated_at=newest - timedelta(days=i),
        )
        for i in range(count)
    ]


def drive(
    store: Store,
    adapter,
    *,
    marks: Watermarks,
    stop=None,
    batch_size: int = 10,
):
    return asyncio.run(
        run_imports(
            [adapter],
            StoreSink(store),
            batch_size=batch_size,
            watermarks=marks,
            poison=PoisonLedger(store),
            stop=stop,
        )
    )


# --- guard 1: the second run does substantially less work -----------------


def test_a_second_consecutive_run_does_substantially_less_work(store):
    """THE REGRESSION GUARD FOR THE DOOM LOOP.

    Fifty records spread over fifty days. The first run has no mark and reads
    all fifty. The second must read only what the one-hour overlap window
    reaches back over, which is the single newest record — not fifty, and
    emphatically not fifty every half hour forever.
    """
    marks = Watermarks(store)
    records = corpus("dropbox", 50)

    first = WindowedAdapter("dropbox", records, since=marks.plan("dropbox").since)
    report_one = drive(store, first, marks=marks)
    assert first.emitted == 50
    assert report_one.adapters["dropbox"].added == 50

    second = WindowedAdapter("dropbox", records, since=marks.plan("dropbox").since)
    report_two = drive(store, second, marks=marks)

    # The window did the work: 49 of 50 never left the source.
    assert second.filtered == 49
    assert second.emitted == 1
    # And the one that did come through cost nothing to re-apply.
    entry = report_two.adapters["dropbox"]
    assert entry.added == 0
    assert entry.unchanged == 1
    assert store.count_by_source("dropbox") == 50


def test_the_window_is_the_mark_minus_the_overlap(store):
    """The margin is the late-data mitigation, so assert it is really applied."""
    marks = Watermarks(store)
    records = corpus("dropbox", 3)
    drive(store, WindowedAdapter("dropbox", records, since=None), marks=marks)
    assert marks.plan("dropbox").cursor_value == NOW
    assert marks.plan("dropbox").since == NOW - MODIFIED_TIME_OVERLAP


def test_a_full_scan_source_keeps_full_scanning_and_says_so(store):
    """TickTick cannot be windowed, and must not pretend otherwise.

    Its ``since`` filters the CSV BACKUP FILES by mtime while its records carry
    the TASK's completion time, and the Open API leg reads a completion as a
    task's DISAPPEARANCE between two FULL polls. A window would drop rows in
    the first case and invent completions in the second. So it re-reads
    everything every run — and the report says FULL SCAN rather than leaving an
    empty mark that reads as "nothing new yet".
    """
    marks = Watermarks(store)
    records = corpus("ticktick", 5)

    for _ in range(2):
        plan = marks.plan("ticktick")
        assert plan.since is None
        adapter = WindowedAdapter("ticktick", records, since=plan.since)
        report = drive(store, adapter, marks=marks)
        assert adapter.emitted == 5
        assert report.adapters["ticktick"].window.startswith("FULL SCAN")

    # The second pass still cost nothing, because re-applying an unchanged row
    # is free even when re-READING it is not.
    assert report.adapters["ticktick"].unchanged == 5


def test_a_failed_run_does_not_advance_the_mark(store):
    """A window that moved on a failed pass would skip whatever it missed."""

    class Exploding(WindowedAdapter):
        async def get_data(self):
            yield self._records[0]
            raise RuntimeError("connection reset")

    marks = Watermarks(store)
    report = drive(
        store, Exploding("dropbox", corpus("dropbox", 3), since=None), marks=marks
    )
    assert not report.adapters["dropbox"].ok
    assert marks.plan("dropbox").cursor_value is None
    # The row that DID arrive is still stored: partial ingest beats total loss.
    assert store.count_by_source("dropbox") == 1
    # ...and the failure is counted, so a source that keeps dying gets rested.
    assert marks.plan("dropbox").consecutive_failures == 1


def test_an_item_with_no_cursor_timestamp_refuses_to_advance_the_mark(store):
    """Undated rows are why the refusal exists rather than a "best effort" max.

    A record with no ``updated_at`` cannot be placed on the cursor, so there is
    no value that means "past it". Advancing to the maximum of the rows that DO
    have one would skip the undated row forever. Standing still costs one full
    re-read; the alternative costs the record.
    """
    marks = Watermarks(store)
    records = corpus("dropbox", 2)
    records[1].updated_at = None
    report = drive(store, WindowedAdapter("dropbox", records, since=None), marks=marks)

    assert marks.plan("dropbox").cursor_value is None
    assert any("no timestamp" in n for n in report.adapters["dropbox"].notes)
    assert store.count_by_source("dropbox") == 2


# --- guard 2: a kill mid-run leaves a resumable state ---------------------


def test_a_shutdown_mid_run_commits_what_it_had_and_resumes_next_run(store):
    """THE REGRESSION GUARD FOR THE SIGTERM HALF.

    Before this branch a SIGTERM at 44% threw away 44% of a five-hour run and
    the next tick started from zero — which is why the loop could never
    terminate. Now: every chunk before the stop is committed, the mark is
    deliberately NOT advanced (the stream is incomplete, so the maximum seen is
    not a high-water mark), and the next run walks the same window again at
    almost no cost and finishes it.
    """
    marks = Watermarks(store)
    records = corpus("dropbox", 50)

    chunks = {"n": 0}

    def stop_after_first_chunk() -> bool:
        chunks["n"] += 1
        return chunks["n"] >= 10  # batch_size, so: at the first chunk boundary

    interrupted = WindowedAdapter("dropbox", records, since=None)
    report = drive(store, interrupted, marks=marks, stop=stop_after_first_chunk)

    assert report.interrupted is True
    assert report.adapters["dropbox"].interrupted is True
    # Not an error: a clean stop is the designed behaviour, not a failure.
    assert report.ok is True
    # The first chunk is durable...
    stored_after_kill = store.count_by_source("dropbox")
    assert 0 < stored_after_kill < 50
    # ...and the mark did NOT move, so nothing beyond it can be skipped.
    assert marks.plan("dropbox").cursor_value is None
    assert any("shutdown" in n for n in report.adapters["dropbox"].notes)

    # The next run resumes: same window, everything lands, mark advances.
    resumed = WindowedAdapter("dropbox", records, since=marks.plan("dropbox").since)
    second = drive(store, resumed, marks=marks)
    assert store.count_by_source("dropbox") == 50
    assert marks.plan("dropbox").cursor_value == NOW
    # And the work already done was not redone: the rows from the killed run
    # were re-read but cost no scrub and no page write.
    assert second.adapters["dropbox"].unchanged == stored_after_kill


def test_a_source_that_finished_before_the_stop_keeps_its_mark(store):
    """Per-source marks are independent, including under shutdown.

    A SIGTERM that arrived while source B was still streaming must not undo
    source A's completed pass — otherwise every interrupted run would rewind
    every source that had already finished, and a busy source would keep the
    whole run's progress permanently at zero.
    """
    marks = Watermarks(store)
    finished = WindowedAdapter("research", corpus("research", 2), since=None)
    unfinished = WindowedAdapter("dropbox", corpus("dropbox", 50), since=None)

    seen = {"n": 0}

    def stop_soon() -> bool:
        seen["n"] += 1
        return seen["n"] > 12

    asyncio.run(
        run_imports(
            [finished, unfinished],
            StoreSink(store),
            batch_size=10,
            watermarks=marks,
            poison=PoisonLedger(store),
            stop=stop_soon,
        )
    )
    assert marks.plan("research").cursor_value == NOW


def test_the_signal_handler_really_flips_the_flag():
    """``graceful_shutdown`` has to install something, not merely return a lambda.

    The installation is asserted BEFORE the signal is raised: if the handler is
    not there this fails as a test rather than terminating the test runner.
    """

    async def scenario():
        with graceful_shutdown() as stop:
            assert signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL
            assert stop() is False
            signal.raise_signal(signal.SIGTERM)
            await asyncio.sleep(0.05)
            return stop()

    assert asyncio.run(scenario()) is True


# --- the chunk transaction ------------------------------------------------


def test_the_chunk_and_its_mark_commit_together(store, monkeypatch):
    """Neither may land without the other. That is the whole reason the mark
    lives in cache.db rather than in a JSON file beside it."""
    marks = Watermarks(store)
    sink = StoreSink(store)
    original = store.advance_ingest_cursor

    def explode(*a, **kw):
        raise RuntimeError("crash between the two writes")

    monkeypatch.setattr(store, "advance_ingest_cursor", explode)
    with pytest.raises(RuntimeError):
        sink.write_checkpoint(
            corpus("dropbox", 3),
            Checkpoint(source="dropbox", cursor_value=NOW, rows=3),
        )
    monkeypatch.setattr(store, "advance_ingest_cursor", original)

    # The rows rolled back with the mark: no half-applied chunk.
    assert store.count_by_source("dropbox") == 0
    assert marks.plan("dropbox").cursor_value is None


def test_every_chunk_is_committed_as_it_goes(store):
    """Streaming, not one giant transaction: a kill costs one chunk, not a run."""
    marks = Watermarks(store)
    seen: list[int] = []

    class Watching(WindowedAdapter):
        async def get_data(self):
            for i, r in enumerate(self._records):
                if i and i % 10 == 0:
                    # Read through a SEPARATE connection, so only committed
                    # rows are visible. An uncommitted chunk would read as 0.
                    other = Store(store.db_path)
                    try:
                        seen.append(other.count_by_source("dropbox"))
                    finally:
                        other.close()
                yield r

    drive(store, Watching("dropbox", corpus("dropbox", 35), since=None), marks=marks)
    assert seen == [10, 20, 30]
