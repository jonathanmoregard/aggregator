"""Permanently-malformed input is reported ONCE, then remembered.

THE FAILURE THIS EXISTS FOR, observed live on 2026-08-16. Four ``.jsonl`` files
under ``~/.claude/projects`` hold lines that will never parse. The sessions
source reported them through ``drain_errors``, the run ended with a non-empty
errors list, ``ingest --all`` exited 3, and the unit's ``OnFailure=`` fired a
CRITICAL desktop toast. Every 30 minutes. Four runs in six hours, the identical
eight errors each time — a permanently-red alarm, which is the one an operator
learns to dismiss unread, which costs the NEXT real failure its audience.

THE LINE THIS WALKS, and it is narrow on purpose. "Fail loudly"
(``tasks/session-constraints.md``, 2026-08-08) is not weakened here: it means
the operator learns of each NEW problem exactly once, not that the same
permanent problem shouts twice an hour. So:

* a fault nobody has seen before is loud — errors, exit 3, notification;
* a fault already IN the ledger is a note, and the run exits clean;
* a fault the source did NOT declare permanent (a lock, a timeout, an expired
  token) is loud on every single run and can never enter the ledger;
* the ledger row is written only once a channel CARRIED the line, so a fault
  reported to nobody buys no silence at all;
* and the quiet set is listed by ``aggregator status``, because quiet is only
  acceptable while it is not the same as forgotten.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from aggregator.cli import _cmd_status, _notification_text
from aggregator.core.store import Store
from aggregator.imports.ingest_state import PoisonLedger, Watermarks
from aggregator.imports.port import Delivery
from aggregator.imports.runner import RunReport, run_imports
from aggregator.imports.sessions import SessionsAdapter
from aggregator.imports.store_sink import StoreSink

_LINE = {
    "parentUuid": None,
    "isSidechain": False,
    "type": "user",
    "message": {"role": "user", "content": "hello"},
    "uuid": "u-1",
    "timestamp": "2026-07-25T10:00:01.000Z",
    "cwd": "/home/u/proj",
    "sessionId": "sess-1",
    "version": "2.1.92",
    "gitBranch": "main",
}


def _backdate(path: Path) -> None:
    """Past the source's 5-minute live-file window, or it is skipped."""
    old = time.time() - 24 * 60 * 60
    os.utime(path, (old, old))


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "cache.db")
    s.migrate()
    yield s
    s.close()


@pytest.fixture()
def projects(tmp_path):
    root = tmp_path / "projects" / "proj-slug"
    root.mkdir(parents=True)
    good = root / "sess-1.jsonl"
    good.write_text(json.dumps(_LINE) + "\n")
    _backdate(good)
    return tmp_path / "projects"


def _poison(projects: Path, *, lines: list[str]) -> Path:
    """A JSONL file whose given lines can never be parsed."""
    path = projects / "proj-slug" / "broken.jsonl"
    path.write_text("".join(lines))
    _backdate(path)
    return path


def _corrupt_file(projects: Path, *, bad: int = 2) -> Path:
    body = [json.dumps({**_LINE, "uuid": f"u-{i}"}) + "\n" for i in range(5)]
    for i in range(bad):
        body[i] = "{not json at all\n"
    return _poison(projects, lines=body)


class Toast:
    """A notify hook that behaves like the shipped desktop one.

    Declares delivery out of the very text it "sent", which is what makes a
    receipt mean something — see ``port.Delivery``.
    """

    def __init__(self) -> None:
        self.sent: list[str] = []

    def __call__(self, report: RunReport) -> Delivery:
        text = _notification_text(report)
        if text is None:
            return Delivery()
        body = text[2]
        self.sent.append(body)
        return Delivery.accepted(body, report.reported)


def _run(store: Store, projects: Path, notify=None):
    """One ``ingest --all``-shaped pass over the sessions source."""
    ledger = PoisonLedger(store)
    report = asyncio.run(
        run_imports(
            [SessionsAdapter(projects_root=projects)],
            StoreSink(store),
            notify=notify or Toast(),
            watermarks=Watermarks(store),
            poison=ledger,
        )
    )
    return report


# --- new is loud ---------------------------------------------------------


def test_a_never_seen_corrupt_line_fails_the_run_loudly(store, projects):
    """First sighting is a failure, unchanged. This is the half that must not move."""
    _corrupt_file(projects)
    report = _run(store, projects)

    assert not report.ok
    assert any("DROPPED 2 corrupt line(s)" in e for e in report.errors)
    # And it reached a human: the toast is what earns the receipt.
    assert _notification_text(report) is not None


# --- known is quiet ------------------------------------------------------


def test_the_same_poison_twice_exits_clean_and_does_not_notify(store, projects):
    """The whole bug, in one test.

    Second run: same file, same lines, same everything. Nothing NEW happened,
    so nothing new is said — no error, no toast, exit 0.
    """
    _corrupt_file(projects)
    toast = Toast()
    first = _run(store, projects, notify=toast)
    assert not first.ok

    second = _run(store, projects, notify=toast)

    assert second.ok, second.errors
    assert second.errors == []
    assert _notification_text(second) is None
    assert len(toast.sent) == 1, "the second run must not have sent anything"


def test_a_quiet_fault_is_still_reported_as_a_note(store, projects):
    """Quiet is not the same as hidden: the run report still names it."""
    _corrupt_file(projects)
    _run(store, projects)
    second = _run(store, projects)

    entry = second.adapters["sessions"]
    assert entry.known_faults, "the fault must survive on the report"
    assert any("broken.jsonl" in note for note in entry.notes)
    assert second.quarantined_records == 2


# --- a NEW fault is loud even when other poison is known -----------------


def test_a_new_corrupt_line_is_loud_even_beside_known_poison(store, projects):
    """Identity, not count. A different bad line in the SAME file is new."""
    _corrupt_file(projects, bad=2)
    _run(store, projects)
    assert _run(store, projects).ok

    # A third corrupt line appears in the same file.
    _corrupt_file(projects, bad=3)
    third = _run(store, projects)

    assert not third.ok
    assert any("DROPPED 3 corrupt line(s)" in e for e in third.errors)


def test_a_corrupt_line_in_a_different_file_is_loud(store, projects):
    """A second poisoned file is a second fault, not the same one."""
    _corrupt_file(projects)
    _run(store, projects)
    assert _run(store, projects).ok

    other = projects / "proj-slug" / "other.jsonl"
    other.write_text("{also not json\n")
    _backdate(other)
    report = _run(store, projects)

    assert not report.ok
    assert any("other.jsonl" in e for e in report.errors)


# --- poison that goes away leaves the ledger -----------------------------


def test_a_fault_that_stops_reproducing_leaves_the_ledger(store, projects):
    """The file was rewritten and now parses. It must be indexed normally."""
    path = _corrupt_file(projects)
    _run(store, projects)
    assert PoisonLedger(store).fault_summary()

    path.write_text(
        "".join(json.dumps({**_LINE, "uuid": f"u-{i}"}) + "\n" for i in range(5))
    )
    _backdate(path)
    report = _run(store, projects)

    assert report.ok
    assert PoisonLedger(store).fault_summary() == []
    assert report.quarantined_records == 0


# --- transient stays loud, forever ---------------------------------------


def test_a_transient_failure_is_never_absorbed_into_the_ledger(store, projects):
    """A lock, a timeout, an expired token. Loud on run one and on run fifty.

    The source never DECLARED this permanent, so nothing may quieten it. This
    is the guard that keeps the ledger from becoming a general mute.
    """

    class Flaky(SessionsAdapter):
        def drain_errors(self) -> list[str]:
            return [*super().drain_errors(), "database is locked"]

    ledger, marks = PoisonLedger(store), Watermarks(store)

    def once():
        return asyncio.run(
            run_imports(
                [Flaky(projects_root=projects)],
                StoreSink(store),
                notify=Toast(),
                watermarks=marks,
                poison=ledger,
            )
        )

    for _ in range(3):
        report = once()
        assert not report.ok
        assert any("database is locked" in e for e in report.errors)
    assert ledger.fault_summary() == []


def test_known_poison_does_not_excuse_a_transient_error_beside_it(store, projects):
    """The mixed run, which is the one a blanket mute gets wrong.

    A file that has been broken for months AND a database that is locked right
    now. The first is a note, the second fails the run — one report, two
    verdicts, decided per line.
    """
    _corrupt_file(projects)
    _run(store, projects)

    class Flaky(SessionsAdapter):
        def drain_errors(self) -> list[str]:
            return [*super().drain_errors(), "database is locked"]

    report = asyncio.run(
        run_imports(
            [Flaky(projects_root=projects)],
            StoreSink(store),
            notify=Toast(),
            watermarks=Watermarks(store),
            poison=PoisonLedger(store),
        )
    )

    assert not report.ok
    assert report.errors == ["sessions: database is locked"]
    assert report.adapters["sessions"].known_faults


def test_an_adapter_that_reports_nonsense_faults_excuses_nothing(store, projects):
    """A ``drain_faults`` of the wrong shape must not buy silence.

    ``SupportsPermanentFaults`` is ``runtime_checkable``, which gates on method
    PRESENCE only — so an adapter returning strings, dicts or None arrives here
    looking perfectly conformant. Refusing it loudly is the only safe answer:
    the alternative is hashing something nobody designed and going quiet about
    a line on the strength of it.
    """
    _corrupt_file(projects)

    class Nonsense(SessionsAdapter):
        def drain_faults(self):
            return "the file is a bit broken"

    report = asyncio.run(
        run_imports(
            [Nonsense(projects_root=projects)],
            StoreSink(store),
            notify=Toast(),
            watermarks=Watermarks(store),
            poison=PoisonLedger(store),
        )
    )

    assert not report.ok
    assert any("DROPPED 2 corrupt line(s)" in e for e in report.errors)
    assert any("could not be reconciled" in e for e in report.errors)
    assert PoisonLedger(store).fault_summary() == []


# --- the receipt discipline ----------------------------------------------


def test_a_fault_nobody_was_told_about_stays_loud(store, projects):
    """No channel, no receipt, no silence. The staleness rule, applied here.

    A ledger row written on the strength of a run whose report reached nobody
    is the silent-degradation failure this repo keeps ruling out.
    """
    _corrupt_file(projects)
    for _ in range(2):
        report = _run(store, projects, notify=lambda r: None)
        assert not report.ok
    assert PoisonLedger(store).fault_summary() == []


# --- visible on demand ---------------------------------------------------


def test_status_reports_what_is_quarantined_and_since_when(store, projects, capsys):
    _corrupt_file(projects)
    _run(store, projects)
    _run(store, projects)

    class Args:
        json = False

    _cmd_status(Args(), store)
    out = capsys.readouterr().out

    assert "sessions" in out
    assert "broken.jsonl" in out
    assert "2 record(s)" in out


def test_status_json_carries_the_fault_ledger(store, projects, capsys):
    _corrupt_file(projects)
    _run(store, projects)

    class Args:
        json = True

    _cmd_status(Args(), store)
    payload = json.loads(capsys.readouterr().out)

    faults = payload["known_faults"]
    assert len(faults) == 1
    assert faults[0]["source"] == "sessions"
    assert faults[0]["count"] == 2
    assert faults[0]["first_seen_at"]
