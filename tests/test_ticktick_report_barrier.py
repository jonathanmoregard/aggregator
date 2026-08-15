"""Round-5 HIGH 1: a receipt must not outlive a notification that never arrived.

The uncovered-project receipt is what lets a poll stay quiet about a
disappearance an earlier poll already reported — the round-4 fix, and a
necessary one: deleting a project in TickTick is routine, and an error on every
30-minute tick is an alarm an operator learns to ignore.

It was written by the WRITE barrier, which fires while the run's report is still
inside the process. The report leaves later — after every adapter has finished,
through ``notify``. So a notify hook that could not run (missing program,
non-zero exit, an exception) left behind a mark saying the operator had been
told, and every later poll read the disappearance as already-reported and said
nothing. The single alert the mechanism exists to deliver was suppressed by a
record of its own delivery.

The receipt now waits for a second barrier, ``commit_after_report``, which only
the delivered path reaches. The two barriers fail in opposite directions on
purpose: skipping the write barrier loses a completion permanently, skipping
this one costs one more copy of a report that was already made.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime

import pytest

from aggregator import cli
from aggregator.core.store import Store
from aggregator.imports.port import Delivery, SupportsReportBarrier, WriteCounts
from aggregator.imports.runner import run_imports
from aggregator.imports.sync_bridge import SyncSourceAdapter
from aggregator.imports.ticktick import TickTickAdapter
from aggregator.sources import ticktick_api
from aggregator.sources.ticktick import TickTickSource

UNCOVERED = "never covered"


@pytest.fixture(autouse=True)
def _no_network_or_real_credentials(monkeypatch, tmp_path):
    """Same guard as ``tests/sources/test_ticktick_api.py``. The token is a
    write-scoped credential and the endpoint is the user's live task list."""

    def _forbidden(*args, **kwargs):
        raise AssertionError("test attempted a real network call")

    monkeypatch.setattr(ticktick_api, "_open", _forbidden)
    monkeypatch.setattr(ticktick_api, "DEFAULT_ENV_FILE", str(tmp_path / "no-such-env"))
    for var in (
        "TICKTICK_ACCESS_TOKEN",
        "TICKTICK_TOKEN_EXPIRES_AT",
        "AGGREGATOR_TICKTICK_TOKEN",
        "AGGREGATOR_TICKTICK_TOKEN_FILE",
        "AGGREGATOR_TICKTICK_DIR",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _live_project_only(monkeypatch):
    """The poll covers ``live`` and serves ``t9`` from it. The baseline's ``t1``
    is in ``gone``, which the poll never looked at — so t1 is retained, never
    inferred completed, and REPORTED."""
    monkeypatch.setattr(
        ticktick_api,
        "poll_open_tasks",
        lambda token, errors=None: ticktick_api.OpenTaskPoll(
            [{"id": "t9", "title": "still open", "projectId": "live"}],
            complete=True,
            covered_project_ids=frozenset({"live"}),
        ),
    )


@pytest.fixture
def state_file(tmp_path):
    path = tmp_path / "open_tasks.json"
    path.write_text(
        json.dumps(
            {"t1": {"task": {"id": "t1", "projectId": "gone"}, "last_seen": "x"}}
        ),
        encoding="utf-8",
    )
    return path


class _Sink:
    def write(self, items) -> WriteCounts:
        return WriteCounts(added=len(items))


class _FailingSink:
    def write(self, items) -> WriteCounts:
        raise sqlite3.OperationalError("database is locked")


def _source(tmp_path, state_file) -> TickTickSource:
    return TickTickSource(
        backup_dir=tmp_path / "no-downloads",
        archive_dir=tmp_path / "no-archive",
        token="fake-token",
        state_file=state_file,
    )


def _broken_notify(report) -> None:
    raise FileNotFoundError("notify-send: command not found")


def _working_notify(heard: list[str]):
    """A hook standing in for a notifier that reached somebody, and SAYS SO.

    Returning ``Delivery.DELIVERED`` is the whole declaration (round 6): a hook
    that merely returns is a hook that might have done nothing, and the default
    one does exactly nothing. See ``imports/port.Delivery``.
    """

    def notify(report) -> Delivery:
        heard.extend(report.errors)
        return Delivery.DELIVERED

    return notify


def _run(tmp_path, state_file, notify, sink=None):
    """One whole run through the runner, on a fresh adapter — what a timer does."""
    adapter = TickTickAdapter(source=_source(tmp_path, state_file))
    return asyncio.run(
        run_imports([adapter], sink or _Sink(), notify=notify)
    )


def _reported(report) -> list[str]:
    return [e for e in report.errors if UNCOVERED in e]


# -- the finding ----------------------------------------------------------


def test_a_notification_that_never_arrived_leaves_the_next_run_loud(
    tmp_path, state_file
):
    """THE repro. Run 1 reports the vanished project and the notify hook blows
    up, so nobody hears it. Run 2 used to be silent: the receipt run 1 wrote
    claimed the operator had already been told."""
    first = _run(tmp_path, state_file, _broken_notify)
    assert _reported(first), first.errors
    assert any("notify hook failed" in e for e in first.errors), first.errors

    second = _run(tmp_path, state_file, _broken_notify)

    assert _reported(second), (
        "permanent silence: run 1's notification reached nobody, but its "
        f"receipt says it did. run 2 reported {second.errors}"
    )


def test_the_alert_survives_until_a_notifier_that_works_delivers_it(
    tmp_path, state_file
):
    """The point of the fix, stated as the outcome a human experiences: the
    single alert is not lost by a broken notifier, only delayed until one
    works. Before, fixing notify-send afterwards was too late — the disappearance
    had already been marked reported and nothing would ever raise it again."""
    _run(tmp_path, state_file, _broken_notify)

    delivered: list[str] = []
    third = _run(tmp_path, state_file, _working_notify(delivered))

    assert [e for e in delivered if UNCOVERED in e], (
        f"the working notifier was told nothing about it: {delivered}"
    )
    assert _reported(third)


# -- and the round-4 fix it must not undo ---------------------------------


def test_a_delivered_report_still_silences_the_next_run(tmp_path, state_file):
    """Round 4's HIGH, unregressed. Reporting a deleted project on every poll
    forever is an alarm an operator learns to ignore, which costs the next real
    ingest failure its audience."""
    heard: list[str] = []
    first = _run(tmp_path, state_file, _working_notify(heard))
    assert _reported(first)
    assert [e for e in heard if UNCOVERED in e], "the hook was told"

    second = _run(tmp_path, state_file, _working_notify(heard))

    assert second.errors == [], f"the same alarm fired twice: {second.errors}"


def test_the_retained_task_is_still_never_inferred_completed(tmp_path, state_file):
    """Quiet is only ever about REPORTING. Whatever the notifier did, a task in
    a project the poll never covered holds no evidence of anything and must
    stay in the baseline."""
    for notify in (_broken_notify, lambda report: None):
        report = _run(tmp_path, state_file, notify)
        assert not any("api-inferred-complete" in e for e in report.errors)
        assert "t1" in json.loads(state_file.read_text())


# -- the two barriers are independent, and fail opposite ways -------------


def test_a_failed_notification_does_not_freeze_the_baseline(tmp_path, state_file):
    """The regression this fix must NOT introduce. Deferring the whole advance
    until the report was delivered would let a permanently-broken notify-send
    freeze the baseline — and a task created after the freeze is never in it, so
    its later disappearance is invisible to every future poll. Unbounded loss,
    to avoid one duplicate alert."""
    _run(tmp_path, state_file, _broken_notify)

    baseline = json.loads(state_file.read_text())
    assert "t9" in baseline, "the poll's own tasks must land whatever notify did"
    assert ticktick_api.uncovered_mark(baseline["t1"]) is None


def test_a_run_whose_write_failed_still_earns_the_receipt_it_reported(
    tmp_path, state_file
):
    """The other direction. The sink died, so no baseline advance — but the
    uncovered project WAS reported and the report WAS delivered, so the human
    heard it and the next poll has nothing new to say. Tying the receipt to the
    write barrier instead would re-alarm forever on every failing run."""
    heard: list[str] = []
    first = _run(
        tmp_path,
        state_file,
        _working_notify(heard),
        sink=_FailingSink(),
    )
    assert first.ok is False
    assert [e for e in heard if UNCOVERED in e]
    assert "t9" not in json.loads(state_file.read_text()), "no advance happened"

    second = _run(tmp_path, state_file, _working_notify([]))

    assert _reported(second) == [], f"already told: {second.errors}"


def test_a_receipt_is_not_stamped_onto_an_entry_that_moved_meanwhile(tmp_path):
    """The receipt is applied by a read-modify-write, so it has to re-check what
    it is marking. A task that came back under another project between the
    advance and the delivery has had a NEW fact happen to it, and a stale mark
    would mute the next poll's report of that new fact."""
    path = tmp_path / "open_tasks.json"
    now = datetime(2026, 8, 15, tzinfo=UTC)
    ticktick_api.save_state(path, [{"id": "t1", "projectId": "gone"}], now)
    _, commit, commit_receipts = ticktick_api.plan_open_task_reconcile(
        ticktick_api.JsonFileState(path),
        ticktick_api.OpenTaskPoll(
            [], complete=True, covered_project_ids=frozenset({"live"})
        ),
        now,
    )
    commit()
    # Another run observed t1 under a different project before the report landed.
    ticktick_api.save_state(path, [{"id": "t1", "projectId": "elsewhere"}], now)

    commit_receipts()

    assert ticktick_api.uncovered_mark(ticktick_api.load_state(path)["t1"]) is None


# -- the single-source CLI path has a channel too -------------------------


def test_the_cli_path_records_the_report_it_printed_to_stderr(
    tmp_path, state_file, capsys
):
    """``aggregator ingest ticktick`` installs no notifier at all — ``--notify``
    is refused without ``--all`` — so stderr IS the delivery and the receipt is
    earned by printing. Without this the fix would have traded permanent silence
    for a permanent alarm on the interactive path."""
    store = Store(db_path=tmp_path / "cache.db")
    store.migrate()

    def _ingest() -> int:
        return cli.main(
            ["ingest", "ticktick"],
            _store=store,
            _sources={"ticktick": _source(tmp_path, state_file)},
        )

    assert _ingest() == cli.EXIT_COMPLETED_WITH_ERRORS
    assert UNCOVERED in capsys.readouterr().err

    assert _ingest() == 0, "the interactive path re-raised a reported alarm"
    assert UNCOVERED not in capsys.readouterr().err


# -- the contract, against a probe rather than TickTick -------------------


class _Probe:
    """A source wearing nothing but the two barriers."""

    name = "probe"

    def __init__(self, *, angry: bool = False) -> None:
        self.events: list[str] = []
        self._angry = angry

    def iter_records(self, since, errors=None):
        return iter(())

    def commit_after_write(self) -> None:
        self.events.append("write")

    def commit_after_report(self) -> None:
        if self._angry:
            raise OSError("read-only file system")
        self.events.append("report")


def test_the_report_barrier_fires_after_the_write_barrier_and_only_once():
    probe = _Probe()

    asyncio.run(
        run_imports([SyncSourceAdapter(probe)], _Sink(), notify=_working_notify([]))
    )

    assert probe.events == ["write", "report"]


def test_a_hook_that_raised_means_the_report_reached_nobody():
    probe = _Probe()

    report = asyncio.run(
        run_imports([SyncSourceAdapter(probe)], _Sink(), notify=_broken_notify)
    )

    assert "report" not in probe.events
    assert report.ok is False


def test_a_report_barrier_that_raises_is_reported_not_swallowed():
    """The loss is small — one duplicate report next run — but a receipt that
    silently never lands turns "reported once" back into "reported forever"."""
    probe = _Probe(angry=True)

    report = asyncio.run(
        run_imports([SyncSourceAdapter(probe)], _Sink(), notify=_working_notify([]))
    )

    assert report.ok is False
    assert any("commit_after_report failed" in e for e in report.errors)


def test_ticktick_is_a_report_barrier_adapter_at_both_seams():
    """The contract above is worth nothing if the one source that needs it does
    not reach either call site: the runner checks structurally, the CLI by
    attribute."""
    adapter = TickTickAdapter(source=TickTickSource())
    assert isinstance(adapter, SupportsReportBarrier)
    assert callable(getattr(TickTickSource(), "commit_after_report", None))
