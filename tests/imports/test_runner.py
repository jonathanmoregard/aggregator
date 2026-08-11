"""Tests for the unified import runner (aggregator.imports.runner).

Each requirement in the task gets a test:
- N adapters run CONCURRENTLY (proved by a rendezvous that deadlocks if not),
- per-adapter failure isolation (one raising adapter cannot stop the others,
  nor lose what it had already yielded),
- a report with REAL per-adapter added/updated/skipped/errors + run totals,
- ``ok`` False when any adapter errored (the CLI's exit code is a later task),
- an injected failure-notification hook, defaulting to a no-op so library
  code never shells out to notify-send itself.

Async is driven with ``asyncio.run()`` in sync tests — see test_port.py for
why no pytest plugin is used.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest

from aggregator.imports.port import ImportItem, WriteCounts
from aggregator.imports.runner import RunReport, run_imports
from aggregator.sources.base import Record


def _rec(n: str) -> Record:
    return Record(stable_id=f"t:{n}", source="t", subject=n, body=n)


class RecordingSink:
    """Spy sink. Reports counts that do NOT equal len(items), so a runner
    that fakes its numbers from the batch size fails the assertion."""

    def __init__(self) -> None:
        self.written: list[ImportItem] = []

    def write(self, items) -> WriteCounts:
        items = list(items)
        self.written.extend(items)
        # First item of every batch counts as an update, rest as adds.
        return WriteCounts(
            added=max(0, len(items) - 1),
            updated=1 if items else 0,
            skipped=0,
        )


class ListAdapter:
    def __init__(self, name: str, items: list[ImportItem]) -> None:
        self.name = name
        self._items = items

    async def get_data(self) -> AsyncIterator[ImportItem]:
        for item in self._items:
            yield item


def test_runs_adapters_and_reports_real_counts_from_the_sink():
    sink = RecordingSink()
    adapters = [
        ListAdapter("alpha", [_rec("a1"), _rec("a2"), _rec("a3")]),
        ListAdapter("beta", [_rec("b1")]),
    ]

    report = asyncio.run(run_imports(adapters, sink, batch_size=100))

    assert isinstance(report, RunReport)
    assert {r.stable_id for r in sink.written} == {"t:a1", "t:a2", "t:a3", "t:b1"}
    alpha = report.adapters["alpha"]
    beta = report.adapters["beta"]
    # 3 items in one batch -> 2 added + 1 updated (NOT added=3).
    assert (alpha.added, alpha.updated, alpha.skipped) == (2, 1, 0)
    assert (beta.added, beta.updated, beta.skipped) == (0, 1, 0)
    assert (report.added, report.updated, report.skipped) == (2, 2, 0)
    assert report.ok is True
    assert report.failed_adapters == []


def test_adapters_run_concurrently_not_one_after_another():
    """Rendezvous, not a stopwatch: each adapter waits for the other to start.

    Sequential execution can never satisfy both waits, so the first adapter
    trips its ``wait_for`` timeout and the assertion on written items fails.
    No sleeps, no timing flake.
    """
    started_a = asyncio.Event()
    started_b = asyncio.Event()

    class Rendezvous:
        def __init__(self, name, mine, theirs):
            self.name = name
            self._mine = mine
            self._theirs = theirs

        async def get_data(self) -> AsyncIterator[ImportItem]:
            self._mine.set()
            await asyncio.wait_for(self._theirs.wait(), timeout=2.0)
            yield _rec(self.name)

    sink = RecordingSink()
    report = asyncio.run(
        run_imports(
            [
                Rendezvous("alpha", started_a, started_b),
                Rendezvous("beta", started_b, started_a),
            ],
            sink,
        )
    )

    assert report.ok is True, report.errors
    assert {r.stable_id for r in sink.written} == {"t:alpha", "t:beta"}


class ExplodingAdapter:
    """Yields ``before`` items, then raises. Models a source whose acquisition
    dies part-way (API 500, disk error) after some data already arrived."""

    def __init__(self, name: str, before: list[ImportItem], exc: Exception) -> None:
        self.name = name
        self._before = before
        self._exc = exc

    async def get_data(self) -> AsyncIterator[ImportItem]:
        for item in self._before:
            yield item
        raise self._exc


def test_one_adapter_raising_does_not_stop_the_others():
    sink = RecordingSink()
    adapters = [
        ExplodingAdapter("boom", [], RuntimeError("token expired")),
        ListAdapter("alpha", [_rec("a1"), _rec("a2")]),
        ListAdapter("beta", [_rec("b1")]),
    ]

    report = asyncio.run(run_imports(adapters, sink, batch_size=100))

    # The healthy adapters completed AND their items were written.
    assert {r.stable_id for r in sink.written} == {"t:a1", "t:a2", "t:b1"}
    assert report.adapters["alpha"].ok is True
    assert report.adapters["beta"].ok is True
    # The failure is recorded, named, and loud — never swallowed.
    boom = report.adapters["boom"]
    assert boom.ok is False
    assert any("token expired" in e for e in boom.errors)
    assert any("RuntimeError" in e for e in boom.errors)
    assert report.ok is False
    assert report.failed_adapters == ["boom"]


def test_items_yielded_before_a_crash_are_still_written():
    """Partial ingest beats total loss — the policy the record sources already
    follow for per-file errors, applied to a mid-stream crash."""
    sink = RecordingSink()
    adapters = [
        ExplodingAdapter("half", [_rec("h1"), _rec("h2")], OSError("disk gone")),
    ]

    report = asyncio.run(run_imports(adapters, sink, batch_size=100))

    assert {r.stable_id for r in sink.written} == {"t:h1", "t:h2"}
    half = report.adapters["half"]
    assert (half.added, half.updated) == (1, 1)
    assert half.ok is False


def test_a_sink_write_failure_is_isolated_to_its_adapter():
    """Isolation has to hold for the write leg too, not just acquisition."""

    class PickySink(RecordingSink):
        def write(self, items):
            items = list(items)
            if any(getattr(i, "source", "") == "poison" for i in items):
                raise sqlite_ish_error("database is locked")
            return super().write(items)

    def sqlite_ish_error(msg: str) -> Exception:
        return RuntimeError(msg)

    poison = Record(stable_id="poison:1", source="poison", subject="p", body="p")
    sink = PickySink()
    report = asyncio.run(
        run_imports(
            [ListAdapter("bad", [poison]), ListAdapter("good", [_rec("g1")])],
            sink,
            batch_size=100,
        )
    )

    assert {r.stable_id for r in sink.written} == {"t:g1"}
    assert report.adapters["good"].ok is True
    assert report.adapters["bad"].ok is False
    assert any("database is locked" in e for e in report.adapters["bad"].errors)


def test_notify_hook_fires_once_with_the_report_when_an_adapter_fails():
    """Fail loudly. The hook is injected — library code never shells out to
    notify-send itself, so the desktop dependency lives in the CLI/systemd
    layer and the failure path stays testable."""
    seen: list[RunReport] = []
    report = asyncio.run(
        run_imports(
            [
                ExplodingAdapter("boom", [], RuntimeError("nope")),
                ListAdapter("alpha", [_rec("a1")]),
            ],
            RecordingSink(),
            notify=seen.append,
        )
    )

    assert len(seen) == 1
    assert seen[0] is report
    assert seen[0].failed_adapters == ["boom"]


def test_notify_hook_is_not_called_on_a_clean_run():
    seen: list[RunReport] = []
    asyncio.run(
        run_imports(
            [ListAdapter("alpha", [_rec("a1")])], RecordingSink(), notify=seen.append
        )
    )
    assert seen == []


def test_notify_defaults_to_a_no_op_so_library_code_stays_silent():
    report = asyncio.run(
        run_imports(
            [ExplodingAdapter("boom", [], RuntimeError("nope"))], RecordingSink()
        )
    )
    assert report.ok is False


def test_non_fatal_errors_reach_the_report_via_the_optional_protocol():
    """A source that survives per-file failures must not lose them here — a
    run ending with a non-empty errors list has to be able to notify."""

    class PartlyBrokenAdapter:
        name = "partly"

        async def get_data(self) -> AsyncIterator[ImportItem]:
            yield _rec("p1")

        def drain_errors(self) -> list[str]:
            return ["/x/a.md: read failed", "/x/b.md: stat failed"]

    sink = RecordingSink()
    report = asyncio.run(run_imports([PartlyBrokenAdapter()], sink))

    partly = report.adapters["partly"]
    assert [r.stable_id for r in sink.written] == ["t:p1"]
    assert partly.errors == ["/x/a.md: read failed", "/x/b.md: stat failed"]
    assert partly.ok is False
    assert report.ok is False


def test_non_fatal_errors_are_drained_even_when_the_adapter_also_crashed():
    class BothWaysBroken:
        name = "both"

        async def get_data(self) -> AsyncIterator[ImportItem]:
            yield _rec("x1")
            raise RuntimeError("stream died")

        def drain_errors(self) -> list[str]:
            return ["/x/a.md: read failed"]

    report = asyncio.run(run_imports([BothWaysBroken()], RecordingSink()))

    errors = report.adapters["both"].errors
    assert any("stream died" in e for e in errors)
    assert "/x/a.md: read failed" in errors


def test_input_freshness_is_recorded_when_the_adapter_offers_it():
    """Sources differ on acquisition: a stale local export re-imports the same
    zip forever and looks like success. The runner records the input's age so a
    later task can report it; adapters that can't answer stay silent."""

    class FreshnessAware:
        name = "aware"

        async def get_data(self) -> AsyncIterator[ImportItem]:
            yield _rec("f1")

        def input_freshness(self) -> datetime | None:
            return datetime(2026, 7, 11, 9, 0, tzinfo=UTC)

    report = asyncio.run(
        run_imports([FreshnessAware(), ListAdapter("plain", [])], RecordingSink())
    )

    assert report.adapters["aware"].input_newest_at == datetime(
        2026, 7, 11, 9, 0, tzinfo=UTC
    )
    assert report.adapters["plain"].input_newest_at is None


def test_a_failing_notify_hook_does_not_lose_the_report():
    """notify-send may be missing under systemd. Losing the report to that
    would hide WHICH adapter broke — the one thing the run has to say."""

    def explode(report: RunReport) -> None:
        raise FileNotFoundError("notify-send: not found")

    report = asyncio.run(
        run_imports(
            [ExplodingAdapter("boom", [], RuntimeError("nope"))],
            RecordingSink(),
            notify=explode,
        )
    )

    assert report.failed_adapters == ["boom"]
    assert any("notify-send: not found" in e for e in report.errors)
    assert report.ok is False


def test_duplicate_adapter_names_are_refused_before_any_work_happens():
    """The report is keyed by name; two adapters sharing one would silently
    drop a whole source's outcome."""
    sink = RecordingSink()
    with pytest.raises(ValueError, match="duplicate adapter name"):
        asyncio.run(
            run_imports(
                [ListAdapter("dupe", [_rec("a")]), ListAdapter("dupe", [_rec("b")])],
                sink,
            )
        )
    assert sink.written == []


def test_items_are_flushed_in_batches_while_the_stream_is_still_running():
    """The port yields instead of returning a list precisely so a 359k-row
    source doesn't sit in memory. That only pays off if the runner writes as
    it goes, so pin it: writes must interleave with production."""
    produced = 0
    produced_at_write: list[int] = []
    batch_sizes: list[int] = []

    class Counting:
        name = "counting"

        async def get_data(self) -> AsyncIterator[ImportItem]:
            nonlocal produced
            for i in range(5):
                produced += 1
                yield _rec(f"c{i}")

    class SizeSpy(RecordingSink):
        def write(self, items):
            items = list(items)
            batch_sizes.append(len(items))
            produced_at_write.append(produced)
            return super().write(items)

    asyncio.run(run_imports([Counting()], SizeSpy(), batch_size=2))

    assert batch_sizes == [2, 2, 1]
    # First flush happened while only 2 of the 5 items existed — i.e. the
    # runner never held the whole stream.
    assert produced_at_write == [2, 4, 5]
