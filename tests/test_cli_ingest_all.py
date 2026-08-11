"""Tests for ``aggregator ingest --all`` — every source, one runner, one pass.

Why this command exists: exactly one source auto-imports today (github, via a
systemd user timer) and the rest are hand-run and days to weeks stale. One
entry point that drives every adapter is what lets ONE timer replace eight,
with the notify-on-failure wiring written once.

No test here touches the user's real cache or the real sources — the store is
a tmp XDG dir and the adapters are injected through ``main(_adapters=...)``.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest

from aggregator.cli import DEFAULT_STALE_AFTER_DAYS, main
from aggregator.core.store import Store
from aggregator.imports.port import ImportItem
from aggregator.sources.base import Record


class _FakeAdapter:
    """An ``ImportAdapter`` with no source behind it.

    Deliberately not a stub of any real source: these tests are about the
    CLI's fan-out, isolation and reporting, and driving them from real sources
    would make them a filesystem test as well.
    """

    def __init__(
        self,
        name: str,
        *,
        records: list[Record] | None = None,
        errors: list[str] | None = None,
        raises: Exception | None = None,
    ) -> None:
        self.name = name
        self._records = records or []
        self._errors = errors or []
        self._raises = raises

    async def get_data(self) -> AsyncIterator[ImportItem]:
        for r in self._records:
            yield r
        if self._raises is not None:
            raise self._raises

    def drain_errors(self) -> list[str]:
        errors, self._errors = self._errors, []
        return errors


class _ExportAdapter(_FakeAdapter):
    """An adapter whose input is a hand-downloaded archive.

    ``SupportsInputFreshness`` is structural, so simply having the method is
    what opts this class in.
    """

    def __init__(self, name: str, age_days: float | None, **kw) -> None:
        super().__init__(name, **kw)
        self._freshness = (
            None
            if age_days is None
            else datetime.now(UTC) - timedelta(days=age_days)
        )

    def input_freshness(self) -> datetime | None:
        return self._freshness


class _RecordSource:
    """A single-source (non-adapter) stub for the `ingest <name>` path."""

    def __init__(self, name: str) -> None:
        self.name = name

    def iter_records(self, since, errors=None):
        yield _record(self.name)


def _record(source: str, n: int = 1) -> Record:
    return Record(
        stable_id=f"{source}:{n}",
        source=source,
        subject=f"{source} {n}",
        body="body",
    )


def _store(tmp_path) -> Store:
    store = Store(db_path=tmp_path / "cache.db")
    store.migrate()
    return store


def test_ingest_all_runs_every_adapter(tmp_path, capsys):
    store = _store(tmp_path)
    adapters = [
        _FakeAdapter("alpha", records=[_record("alpha")]),
        _FakeAdapter("beta", records=[_record("beta"), _record("beta", 2)]),
    ]

    rc = main(["ingest", "--all"], _store=store, _adapters=adapters)

    assert rc == 0
    assert store.count_by_source("alpha") == 1
    assert store.count_by_source("beta") == 2
    out = capsys.readouterr().out
    assert "alpha: added=1 updated=0 skipped=0 errors=0" in out
    assert "beta: added=2 updated=0 skipped=0 errors=0" in out


def test_summary_reports_real_counts_not_hardcoded_zeros(tmp_path, capsys):
    """The single-source path used to print ``added=len(records) updated=0``
    on every run, so an import of 313 new PRs and a re-write of the same 313
    rows read identically. That bug is not repeated here: a second pass over
    unchanged input must report updates, not adds."""
    store = _store(tmp_path)

    def adapters():
        return [_FakeAdapter("alpha", records=[_record("alpha")])]

    main(["ingest", "--all"], _store=store, _adapters=adapters())
    capsys.readouterr()

    rc = main(["ingest", "--all"], _store=store, _adapters=adapters())

    assert rc == 0
    out = capsys.readouterr().out
    assert "alpha: added=0 updated=1 skipped=0 errors=0" in out
    assert store.count_by_source("alpha") == 1


def test_one_source_failing_does_not_stop_the_others(tmp_path, capsys):
    """REGRESSION GUARD (passes on arrival — the runner already isolates
    per adapter). Pinned at the CLI level because this is the property the
    whole run-all command is for: an expired TickTick token must not cost
    sessions, github and dropbox their nightly import.
    """
    store = _store(tmp_path)
    adapters = [
        _FakeAdapter("alpha", records=[_record("alpha")]),
        _FakeAdapter("boom", raises=RuntimeError("token expired")),
        _FakeAdapter("beta", records=[_record("beta")]),
    ]

    rc = main(["ingest", "--all"], _store=store, _adapters=adapters)

    assert rc == 3
    assert store.count_by_source("alpha") == 1
    assert store.count_by_source("beta") == 1
    captured = capsys.readouterr()
    assert "boom: added=0 updated=0 skipped=0 errors=1" in captured.out
    assert "token expired" in captured.err


def test_partial_output_from_a_failing_source_is_still_written(tmp_path):
    """REGRESSION GUARD. Partial ingest beats total loss: rows that arrived
    before the crash are flushed, and the run still reports the failure."""
    store = _store(tmp_path)
    adapters = [
        _FakeAdapter(
            "half",
            records=[_record("half")],
            raises=RuntimeError("connection reset"),
        )
    ]

    rc = main(["ingest", "--all"], _store=store, _adapters=adapters)

    assert rc == 3
    assert store.count_by_source("half") == 1


def test_non_fatal_errors_exit_three(tmp_path, capsys):
    """REGRESSION GUARD. A source that dropped three PDFs and imported the
    rest completed — but it is not a success. Exit 3 matches the single-source
    path so a timer wrapper handles both identically."""
    store = _store(tmp_path)
    adapters = [
        _FakeAdapter(
            "alpha",
            records=[_record("alpha")],
            errors=["a/file.pdf: parse failed"],
        )
    ]

    rc = main(["ingest", "--all"], _store=store, _adapters=adapters)

    assert rc == 3
    captured = capsys.readouterr()
    assert "alpha: added=1 updated=0 skipped=0 errors=1" in captured.out
    assert "a/file.pdf" in captured.err


def test_run_total_is_printed(tmp_path, capsys):
    store = _store(tmp_path)
    adapters = [
        _FakeAdapter("alpha", records=[_record("alpha")]),
        _FakeAdapter("beta", records=[_record("beta"), _record("beta", 2)]),
    ]

    main(["ingest", "--all"], _store=store, _adapters=adapters)

    out = capsys.readouterr().out
    assert "total: added=3 updated=0 skipped=0 errors=0" in out


# -- input staleness -------------------------------------------------------


def test_stale_input_warns_and_names_the_age(tmp_path, capsys):
    """The whole point of the freshness seam. Without this the run reports a
    cheerful ``added=0 errors=0`` on a zip from July and the operator has no
    way to tell that apart from "nothing new to import"."""
    store = _store(tmp_path)
    adapters = [_ExportAdapter("substack", 31, records=[_record("substack")])]

    rc = main(["ingest", "--all"], _store=store, _adapters=adapters)

    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "substack" in err
    assert "31 days" in err
    # Staleness is a warning, not an error: nothing failed, the input is just
    # old. Exit code stays clean so `errors` keeps meaning exactly one thing.
    assert rc == 0


def test_fresh_input_does_not_warn(tmp_path, capsys):
    store = _store(tmp_path)
    adapters = [_ExportAdapter("substack", 2, records=[_record("substack")])]

    rc = main(["ingest", "--all"], _store=store, _adapters=adapters)

    assert rc == 0
    assert "WARNING" not in capsys.readouterr().err


def test_the_threshold_is_overridable(tmp_path, capsys):
    store = _store(tmp_path)

    rc = main(
        ["ingest", "--all", "--stale-after-days", "60"],
        _store=store,
        _adapters=[_ExportAdapter("substack", 31, records=[_record("substack")])],
    )

    assert rc == 0
    assert "WARNING" not in capsys.readouterr().err


def test_the_default_threshold_is_two_weeks(tmp_path, capsys):
    """A monthly export ritual should nag once it is overdue, not the morning
    after it was run. Two weeks is half that cadence."""
    assert DEFAULT_STALE_AFTER_DAYS == 14
    store = _store(tmp_path)
    adapters = [
        _ExportAdapter("just-under", 13, records=[_record("just-under")]),
        _ExportAdapter("just-over", 15, records=[_record("just-over")]),
    ]

    main(["ingest", "--all"], _store=store, _adapters=adapters)

    err = capsys.readouterr().err
    assert "just-over" in err
    assert "just-under" not in err


def test_a_missing_export_warns_rather_than_reading_as_fresh(tmp_path, capsys):
    """No archive at all is the loudest version of this problem: the source
    imported nothing and every count is zero, which is exactly what a healthy
    no-op looks like."""
    store = _store(tmp_path)

    rc = main(
        ["ingest", "--all"],
        _store=store,
        _adapters=[_ExportAdapter("chatgpt", None)],
    )

    assert rc == 0
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "chatgpt" in err
    assert "no input" in err.lower()


def test_a_source_without_a_freshness_signal_never_warns(tmp_path, capsys):
    """github reads a live API and sessions a directory Claude Code appends
    to. Neither offers ``input_freshness``, and neither may be reported as
    stale — a warning nobody can act on trains the operator to ignore the one
    that matters."""
    store = _store(tmp_path)

    main(
        ["ingest", "--all"],
        _store=store,
        _adapters=[_FakeAdapter("github")],
    )

    assert "WARNING" not in capsys.readouterr().err


# -- staleness has to reach something other than stderr --------------------
#
# Round-1 MEDIUM: the warnings were printed to stderr on an exit-0 run and
# nothing else. `notify` fired only when `!ok`, so a 31-day-stale input that
# imported 0 rows was indistinguishable from a healthy no-op to any timer.


def test_a_clean_but_stale_run_still_reaches_the_notifier(tmp_path):
    """THE finding. Nothing about this run is an error, and that is exactly
    why it was invisible: no notifier ran, the exit code was 0, and every
    count was 0 — the same shape as a healthy import with nothing new."""
    store = _store(tmp_path)
    seen = []

    rc = main(
        ["ingest", "--all"],
        _store=store,
        _adapters=[_ExportAdapter("substack", 31)],
        _notify=seen.append,
    )

    assert rc == 0
    assert len(seen) == 1, "the notifier must run on a clean run too"
    report = seen[0]
    assert report.ok is True
    assert len(report.warnings) == 1
    assert "31 days" in report.warnings[0]


def test_a_healthy_no_op_carries_no_warnings(tmp_path):
    """The other half of the same contract: the notifier must be able to tell
    these two apart, so a fresh input has to produce an EMPTY warnings list,
    not merely a differently-worded one."""
    store = _store(tmp_path)
    seen = []

    main(
        ["ingest", "--all"],
        _store=store,
        _adapters=[_ExportAdapter("substack", 2)],
        _notify=seen.append,
    )

    assert seen[0].warnings == []


def test_a_missing_export_reaches_the_notifier_too(tmp_path):
    store = _store(tmp_path)
    seen = []

    main(
        ["ingest", "--all"],
        _store=store,
        _adapters=[_ExportAdapter("chatgpt", None)],
        _notify=seen.append,
    )

    assert len(seen[0].warnings) == 1
    assert "no input" in seen[0].warnings[0].lower()


def test_a_source_with_no_freshness_signal_produces_no_warning(tmp_path):
    """github reads a live API; sessions reads a directory Claude Code appends
    to. Neither has an export ritual to forget. A warning nobody can act on
    trains the operator to ignore the one that matters."""
    store = _store(tmp_path)
    seen = []

    main(
        ["ingest", "--all"],
        _store=store,
        _adapters=[_FakeAdapter("github")],
        _notify=seen.append,
    )

    assert seen[0].warnings == []


def test_warnings_never_change_the_exit_code(tmp_path):
    """Staleness is not a failure — nothing broke, a human just has not
    exported lately — so ``errors`` keeps meaning exactly one thing."""
    store = _store(tmp_path)

    rc = main(
        ["ingest", "--all"],
        _store=store,
        _adapters=[_ExportAdapter("substack", 400)],
    )

    assert rc == 0


def test_the_printed_total_counts_the_warnings(tmp_path, capsys):
    """The journal is where an operator reads this after the fact, and there
    a stale no-op and a healthy no-op printed byte-identical totals."""
    store = _store(tmp_path)

    main(
        ["ingest", "--all"],
        _store=store,
        _adapters=[_ExportAdapter("substack", 31), _FakeAdapter("github")],
    )

    out = capsys.readouterr().out
    assert "total: added=0 updated=0 skipped=0 errors=0 warnings=1" in out


def test_the_threshold_reaches_the_report_not_just_stderr(tmp_path):
    store = _store(tmp_path)
    seen = []

    main(
        ["ingest", "--all", "--stale-after-days", "60"],
        _store=store,
        _adapters=[_ExportAdapter("substack", 31)],
        _notify=seen.append,
    )

    assert seen[0].warnings == []


def test_a_notifier_that_blows_up_is_reported_on_an_otherwise_clean_run(
    tmp_path, capsys
):
    """A notifier that cannot notify is a fault in its own right, and on a
    clean run this is the only thing that says so."""
    store = _store(tmp_path)

    def explode(report):
        raise FileNotFoundError("notify-send: not found")

    rc = main(
        ["ingest", "--all"],
        _store=store,
        _adapters=[_FakeAdapter("alpha", records=[_record("alpha")])],
        _notify=explode,
    )

    assert rc == 3
    assert "notify-send: not found" in capsys.readouterr().err


# -- usage errors ----------------------------------------------------------


def test_a_source_name_together_with_all_is_a_usage_error(tmp_path, capsys):
    """Silently ignoring the name would run nine sources when the operator
    asked for one — and print a report that looks like they got what they
    typed."""
    store = _store(tmp_path)

    rc = main(
        ["ingest", "alpha", "--all"],
        _store=store,
        _adapters=[_FakeAdapter("alpha", records=[_record("alpha")])],
    )

    assert rc == 2
    assert store.count_by_source("alpha") == 0
    assert "--all" in capsys.readouterr().err


def test_neither_a_source_nor_all_is_a_usage_error(tmp_path, capsys):
    store = _store(tmp_path)

    rc = main(["ingest"], _store=store, _sources={})

    assert rc == 2
    err = capsys.readouterr().err
    assert "--all" in err


def test_stale_after_days_without_all_is_a_usage_error(tmp_path, capsys):
    """Round-1 LOW: it parsed on every ingest invocation and did nothing
    without --all. An operator who typed a threshold believes they changed
    when the run nags them."""
    store = _store(tmp_path)

    rc = main(
        ["ingest", "alpha", "--stale-after-days", "60"],
        _store=store,
        _sources={"alpha": _RecordSource("alpha")},
    )

    assert rc == 2
    err = capsys.readouterr().err
    assert "--stale-after-days" in err
    assert "--all" in err
    assert store.count_by_source("alpha") == 0


@pytest.mark.parametrize("flag", ["--force", "--yes"])
def test_force_and_yes_without_rebuild_are_usage_errors(tmp_path, capsys, flag):
    """Somebody passing --force believes they have authorised a destructive
    run. Accepting it silently on a run that destroys nothing is the widest
    gap between what was typed and what happened."""
    store = _store(tmp_path)

    rc = main(
        ["ingest", "alpha", flag],
        _store=store,
        _sources={"alpha": _RecordSource("alpha")},
    )

    assert rc == 2
    err = capsys.readouterr().err
    assert flag in err
    assert "--rebuild" in err
    assert store.count_by_source("alpha") == 0


def test_the_flags_are_accepted_where_they_do_something(tmp_path):
    """The other half: they must still work on the runs they apply to."""
    store = _store(tmp_path)

    assert (
        main(
            ["ingest", "--all", "--stale-after-days", "60"],
            _store=store,
            _adapters=[_ExportAdapter("substack", 31)],
        )
        == 0
    )
    assert (
        main(
            ["ingest", "alpha", "--rebuild", "--force", "--yes"],
            _store=store,
            _sources={"alpha": _RecordSource("alpha")},
        )
        == 0
    )


def test_rebuild_is_refused_with_all(tmp_path, capsys):
    """--rebuild is destructive and per-source; the runner path is upsert-only
    across nine sources at once. Refuse rather than silently ignore the flag —
    an operator who typed it is expecting rows to be replaced."""
    store = _store(tmp_path)

    rc = main(
        ["ingest", "--all", "--rebuild"],
        _store=store,
        _adapters=[_FakeAdapter("alpha", records=[_record("alpha")])],
    )

    assert rc == 2
    assert store.count_by_source("alpha") == 0
    assert "--rebuild" in capsys.readouterr().err


# -- wiring ----------------------------------------------------------------


def test_the_real_registry_drives_the_run_when_nothing_is_injected(
    tmp_path, monkeypatch
):
    """REGRESSION GUARD. Without it the command could be green against
    injected fakes and import nothing in production. ``--since`` has to reach
    the registry too: the port has no ``since`` parameter, so an adapter that
    was not built with one silently ignores the window."""
    seen: list[datetime | None] = []

    def fake_registry(since=None):
        seen.append(since)
        return [_FakeAdapter("alpha", records=[_record("alpha")])]

    monkeypatch.setattr("aggregator.cli.default_adapters", fake_registry)
    store = _store(tmp_path)

    rc = main(["ingest", "--all", "--since", "2026-08-01"], _store=store)

    assert rc == 0
    assert seen == [datetime(2026, 8, 1, tzinfo=UTC)]
    assert store.count_by_source("alpha") == 1


def test_the_single_source_path_still_works(tmp_path, capsys):
    """REGRESSION GUARD. Both paths coexist until a later cleanup; --all must
    not have broken `aggregator ingest <name>`."""

    class _Source:
        name = "solo"

        def iter_records(self, since, errors=None):
            yield _record("solo")

    store = _store(tmp_path)
    rc = main(["ingest", "solo"], _store=store, _sources={"solo": _Source()})

    assert rc == 0
    assert store.count_by_source("solo") == 1
    assert "ingest solo: added=1 updated=0 skipped=0 errors=0" in (
        capsys.readouterr().out
    )


def test_bad_since_is_a_usage_error(tmp_path, capsys):
    """REGRESSION GUARD (passes on arrival — both ingest paths share
    ``_parse_since``). Pinned so the run-all window can never start meaning
    something different from the single-source one."""
    store = _store(tmp_path)

    rc = main(
        ["ingest", "--all", "--since", "not-a-date"],
        _store=store,
        _adapters=[_FakeAdapter("alpha", records=[_record("alpha")])],
    )

    assert rc == 2
    assert store.count_by_source("alpha") == 0
    assert "not-a-date" in capsys.readouterr().err
