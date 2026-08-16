"""Silence must be impossible, and a quiet source must say that it is quiet.

WHAT THIS IS FOR, precisely. On 2026-08-15 the run printed 102 lines of pypdf
warnings and then nothing for two hours. The sessions/observations leg — which
was doing all of the work and doing it wrong — logged NOTHING AT ALL, so the
only voice in the journal belonged to a library, and its absence was read as a
hang. Two hours and a disproven hypothesis went into that gap.

Two separate rules, and passing one does not give you the other:

* every committed chunk logs, which is a pulse while work is flowing;
* a source that has produced no chunk logs anyway, on a timer, which is the
  only thing that can distinguish "working slowly" from "gone".

The second rule is the one that matters, because the failure it describes is
exactly the state in which the first rule produces nothing.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest

from aggregator.core.store import Store
from aggregator.imports.ingest_state import Watermarks
from aggregator.imports.port import ImportItem
from aggregator.imports.progress import RunProgress
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
        body="b",
        updated_at=NOW - timedelta(days=n),
    )


class ListAdapter:
    name = "dropbox"

    def __init__(self, records) -> None:
        self._records = records

    async def get_data(self) -> AsyncIterator[ImportItem]:
        for r in self._records:
            yield r


class FakeClock:
    """Advance time by hand. A cadence rule verified by sleeping is a rule
    that gets marked flaky and then deleted."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_every_chunk_logs_a_line(store, caplog):
    caplog.set_level(logging.INFO, logger="aggregator.ingest")
    asyncio.run(
        run_imports(
            [ListAdapter([_rec(i) for i in range(25)])],
            StoreSink(store),
            batch_size=10,
            watermarks=Watermarks(store),
        )
    )
    chunks = [r for r in caplog.records if "phase=chunk" in r.getMessage()]
    assert len(chunks) == 3


def test_the_line_carries_what_an_operator_needs(store, caplog):
    """rows, rate and the unchanged count — the numbers that tell a healthy
    re-read from the doom loop, which reported neither."""
    caplog.set_level(logging.INFO, logger="aggregator.ingest")
    asyncio.run(
        run_imports(
            [ListAdapter([_rec(i) for i in range(5)])],
            StoreSink(store),
            batch_size=5,
            watermarks=Watermarks(store),
        )
    )
    text = "\n".join(r.getMessage() for r in caplog.records)
    for field in ("run=", "source=dropbox", "rows=", "unchanged=", "rate="):
        assert field in text


def test_the_run_says_which_window_each_source_got(store, caplog):
    """"Was this run incremental?" has to be answerable from the journal alone."""
    caplog.set_level(logging.INFO, logger="aggregator.ingest")
    marks = Watermarks(store)
    asyncio.run(
        run_imports(
            [ListAdapter([_rec(0)])],
            StoreSink(store),
            watermarks=marks,
        )
    )
    begin = [r.getMessage() for r in caplog.records if "phase=begin" in r.getMessage()]
    assert begin and "window=FULL SCAN (first run" in begin[0]

    caplog.clear()
    asyncio.run(
        run_imports([ListAdapter([_rec(0)])], StoreSink(store), watermarks=marks)
    )
    begin = [r.getMessage() for r in caplog.records if "phase=begin" in r.getMessage()]
    assert begin and "window=since 2026-08-1" in begin[0]


def test_the_end_line_says_where_the_mark_went(store, caplog):
    caplog.set_level(logging.INFO, logger="aggregator.ingest")
    asyncio.run(
        run_imports(
            [ListAdapter([_rec(0)])],
            StoreSink(store),
            watermarks=Watermarks(store),
        )
    )
    end = [r.getMessage() for r in caplog.records if "phase=end" in r.getMessage()]
    assert end and "status=finished" in end[0]
    assert "mark=2026-08-16T12:00:00+00:00" in end[0]


# --- the heartbeat --------------------------------------------------------


def test_a_source_that_produces_no_chunk_is_still_heard_from(caplog):
    """THE RULE THE INCIDENT PRODUCED.

    A leg that is parsing one enormous file, blocked on a socket, or genuinely
    stuck produces no chunk boundaries at all — which is precisely the state
    that looked like a hang for two hours. The heartbeat runs on the clock
    rather than on the work, so silence itself becomes a logged fact.
    """
    caplog.set_level(logging.INFO, logger="aggregator.ingest")
    clock = FakeClock()
    progress = RunProgress(run_id="test", heartbeat_seconds=30, clock=clock)
    progress.begin("sessions", "since 2026-08-16T09:00:00+00:00")

    assert progress.emit_heartbeat() == []  # nothing to say yet
    clock.t = 31.0
    said = progress.emit_heartbeat()

    assert len(said) == 1
    assert "source=sessions" in said[0]
    assert "quiet_for=31s" in said[0]


def test_the_heartbeat_stops_once_the_source_has_finished(caplog):
    """A finished source is not a silent one, and must not keep nagging."""
    caplog.set_level(logging.INFO, logger="aggregator.ingest")
    clock = FakeClock()
    progress = RunProgress(run_id="test", heartbeat_seconds=30, clock=clock)
    entry = progress.begin("sessions", "full scan")
    progress.end(entry, status="finished", mark="unchanged")
    clock.t = 500.0
    assert progress.emit_heartbeat() == []


def test_a_chunk_resets_the_quiet_timer(caplog):
    caplog.set_level(logging.INFO, logger="aggregator.ingest")
    clock = FakeClock()
    progress = RunProgress(run_id="test", heartbeat_seconds=30, clock=clock)
    entry = progress.begin("sessions", "full scan")
    clock.t = 25.0
    progress.chunk(entry, rows=500, added=500, updated=0, unchanged=0)
    clock.t = 50.0
    assert progress.emit_heartbeat() == []
    clock.t = 90.0
    assert len(progress.emit_heartbeat()) == 1


def test_the_heartbeat_runs_beside_a_real_run(store, caplog):
    """It has to be a TASK, not a check in the chunk loop — the condition it
    reports is the one in which the chunk loop is not running."""
    caplog.set_level(logging.INFO, logger="aggregator.ingest")

    class SlowAdapter:
        name = "dropbox"

        async def get_data(self) -> AsyncIterator[ImportItem]:
            await asyncio.sleep(0.15)
            yield _rec(0)

    asyncio.run(
        run_imports(
            [SlowAdapter()],
            StoreSink(store),
            watermarks=Watermarks(store),
            progress=RunProgress(heartbeat_seconds=0.04),
        )
    )
    assert any("phase=heartbeat" in r.getMessage() for r in caplog.records)


def test_a_rested_source_says_so_rather_than_vanishing(store, caplog, monkeypatch):
    """A source missing from the output reads exactly like a source with
    nothing new — and this is the case where that difference matters most."""
    import aggregator.imports.ingest_state as state_mod

    monkeypatch.setattr(
        state_mod,
        "full_jitter",
        lambda attempt, *, base, cap: timedelta(
            seconds=min(cap.total_seconds(), base.total_seconds() * 2**attempt)
        ),
    )
    caplog.set_level(logging.INFO, logger="aggregator.ingest")
    marks = Watermarks(store)
    for _ in range(6):
        marks.record_failure("dropbox", "ConnectTimeout")

    report = asyncio.run(
        run_imports([ListAdapter([_rec(0)])], StoreSink(store), watermarks=marks)
    )
    assert report.adapters["dropbox"].skipped_for_backoff
    assert any("phase=skipped" in r.getMessage() for r in caplog.records)
    assert report.adapters["dropbox"].notes
    # ...and it did not run: nothing was read, so nothing was written.
    assert store.count_by_source("dropbox") == 0
