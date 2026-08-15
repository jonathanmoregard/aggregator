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
import io
import json
import sqlite3
import sys
from datetime import UTC, datetime, timedelta

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


class _WatchedTerminal(io.StringIO):
    """stderr with a person in front of it — the only shape in which printing
    to it is a delivery. See ``cli._stderr_delivery``."""

    def isatty(self) -> bool:
        return True


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


def test_a_failed_write_leaves_the_next_run_deriving_exactly_the_same_diff(tmp_path):
    """Round-6 (a): does the receipt outlive the state that justified it?

    The report barrier fires on a run whose WRITE failed, on the grounds that
    the human heard the report either way. So a run can report a disappearance,
    stamp it, and leave the baseline unadvanced — and the question is whether
    what the next run derives from that baseline is still the same thing that
    was reported. It is: the receipt is applied by a read-modify-write over
    whatever is on disk, and it adds one key to one entry, so a covered project's
    completion is re-inferred exactly as before and only the REPORT is
    suppressed. Retention and reporting are separate decisions.
    """
    path = tmp_path / "open_tasks.json"
    path.write_text(
        json.dumps(
            {
                "t1": {"task": {"id": "t1", "projectId": "gone"}, "last_seen": "x"},
                "t5": {"task": {"id": "t5", "projectId": "live"}, "last_seen": "x"},
            }
        ),
        encoding="utf-8",
    )

    heard: list[str] = []
    first = _run(tmp_path, path, _working_notify(heard), sink=_FailingSink())
    assert first.ok is False
    assert [e for e in heard if UNCOVERED in e], "t1's project was reported"

    written: list = []

    class _Recording:
        def write(self, items) -> WriteCounts:
            written.extend(items)
            return WriteCounts(added=len(items))

    second = _run(tmp_path, path, _working_notify([]), sink=_Recording())

    inferred = [
        r.stable_id
        for r in written
        if r.extra.get("provenance") == ticktick_api.INFERRED_COMPLETE_PROVENANCE
    ]
    assert [s for s in inferred if s.endswith("t5")], (
        f"the completion the failed run inferred was not re-derived: {inferred}"
    )
    assert _reported(second) == [], "only the report was suppressed, not the diff"


def test_a_receipt_is_not_stamped_onto_an_entry_that_moved_meanwhile(tmp_path):
    """The receipt is applied by a read-modify-write, so it has to re-check what
    it is marking. A task that came back under another project between the
    advance and the delivery has had a NEW fact happen to it, and a stale mark
    would mute the next poll's report of that new fact."""
    path = tmp_path / "open_tasks.json"
    now = datetime(2026, 8, 15, tzinfo=UTC)
    ticktick_api.save_state(path, [{"id": "t1", "projectId": "gone"}], now)
    plan = ticktick_api.plan_open_task_reconcile(
        ticktick_api.JsonFileState(path),
        ticktick_api.OpenTaskPoll(
            [], complete=True, covered_project_ids=frozenset({"live"})
        ),
        now,
    )
    plan.commit_baseline()
    # Another run observed t1 under a different project before the report landed.
    ticktick_api.save_state(path, [{"id": "t1", "projectId": "elsewhere"}], now)

    plan.commit_receipts()

    assert ticktick_api.uncovered_mark(ticktick_api.load_state(path)["t1"]) is None


def test_a_receipt_is_not_stamped_onto_an_entry_that_was_seen_again(tmp_path):
    """The same guard, for the case the project check cannot see.

    A task that comes back in the SAME project passes "still there, still that
    project" — but it has been OBSERVED OPEN since the report was written, so the
    next time it vanishes that is a new disappearance, and the stale mark would
    mute the report of it. Reachable whenever two ingests overlap, which is what
    the baseline's compare-and-swap already exists for: `ingest ticktick` under a
    timer firing `ingest --all`.
    """
    path = tmp_path / "open_tasks.json"
    now = datetime(2026, 8, 15, tzinfo=UTC)
    ticktick_api.save_state(path, [{"id": "t1", "projectId": "gone"}], now)
    plan = ticktick_api.plan_open_task_reconcile(
        ticktick_api.JsonFileState(path),
        ticktick_api.OpenTaskPoll(
            [], complete=True, covered_project_ids=frozenset({"live"})
        ),
        now,
    )
    plan.commit_baseline()
    # Another run DID cover "gone" before the report landed, and t1 is open.
    ticktick_api.save_state(
        path, [{"id": "t1", "projectId": "gone"}], now + timedelta(hours=1)
    )

    plan.commit_receipts()

    assert ticktick_api.uncovered_mark(ticktick_api.load_state(path)["t1"]) is None, (
        "the task was observed open after the report, so its next disappearance "
        "is a new fact — and this receipt would silence it"
    )


# -- the single-source CLI path has a channel too -------------------------


def test_the_cli_path_records_the_report_it_printed_to_stderr(
    tmp_path, state_file, monkeypatch
):
    """``aggregator ingest ticktick`` installs no notifier at all — ``--notify``
    is refused without ``--all`` — so stderr IS the delivery and the receipt is
    earned by printing. Without this the fix would have traded permanent silence
    for a permanent alarm on the interactive path.

    ROUND 6 NARROWED THIS to an INTERACTIVE stderr. The claim above is only true
    with a person in front of the terminal; under the timer, stderr is the
    journal. See ``tests/test_delivery_contract.py`` for the unattended half.
    """
    store = Store(db_path=tmp_path / "cache.db")
    store.migrate()
    terminal = _WatchedTerminal()
    monkeypatch.setattr(sys, "stderr", terminal)

    def _ingest() -> int:
        return cli.main(
            ["ingest", "ticktick"],
            _store=store,
            _sources={"ticktick": _source(tmp_path, state_file)},
        )

    assert _ingest() == cli.EXIT_COMPLETED_WITH_ERRORS
    assert UNCOVERED in terminal.getvalue()
    terminal.truncate(0)

    assert _ingest() == 0, "the interactive path re-raised a reported alarm"
    assert UNCOVERED not in terminal.getvalue()


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
