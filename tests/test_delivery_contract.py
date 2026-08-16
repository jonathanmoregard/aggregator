"""The delivery contract: a receipt costs the delivery of THAT line, to a human.

Four rounds on one bug and the same conceptual error each time — DELIVERY
ASSERTED AT A COARSER GRAIN THAN SUPPRESSION. An uncovered TickTick project must
be reported once and then go quiet, and the receipt that makes it quiet was
stamped:

  R4: whenever the report was EMITTED.                  -> stamp after notify
  R5: when ``notify`` returned without raising.         -> a separate barrier
  R6: ...and the DEFAULT notifier does nothing at all, which returns without
      raising. So every run with no notifier configured stamped receipts with no
      human channel in existence, and the single alert the mechanism exists to
      raise was suppressed by a record of its own delivery.
  R6 again, one surface over: the single-source CLI path stamped after printing
      to stderr — but the timer that runs these ingests sends stderr to the
      journal, which nobody reads unprompted.
  R7: the hook returned a run-wide "delivered" while the toast it sent was
      ``report.errors[:5]``. Five failures from other sources pushed the
      uncovered-project line out of the payload and its receipt was stamped for
      a sentence that never left the process.
  R7 again, the other direction: an interactive ``ingest --all`` prints its
      report to a terminal a person is watching, AFTER ``run_imports`` returned,
      so the runner never heard about the one channel that actually worked and
      the same gap re-reported on every run, forever.
  R8: the set was right and its MEMBERSHIP TEST was ``line in payload``, i.e.
      substring. One adapter reporting ``box: file unreadable`` and another
      reporting ``dropbox: file unreadable``: send only the second and the first
      was marked delivered, receipt and all, for a sentence never sent.

The shape changed rather than the check. ``Delivery`` is now a SET OF LINES
built by ``Delivery.accepted(payload, report.errors)`` — read out of the text a
channel accepted rather than asserted beside it — and a barrier fires only for
an adapter whose EVERY reported line is in that set. Truncate, reorder, reformat
or send nothing, and the set shrinks by itself; there is no constant meaning
"all of it got through" to reach for by mistake.

Stubs first: this file drives a real ``TickTickSource`` and must never reach the
live API, the developer's credential, or the real baseline.
"""
from __future__ import annotations

import asyncio
import io
import json
import shlex
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from aggregator import cli
from aggregator.core.store import Store
from aggregator.imports.port import Delivery, WriteCounts
from aggregator.imports.runner import RunReport, _no_notification, run_imports
from aggregator.imports.sync_bridge import SyncSourceAdapter
from aggregator.imports.ticktick import TickTickAdapter
from aggregator.sources import ticktick_api
from aggregator.sources.ticktick import TickTickSource

UNCOVERED = "never covered"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("test attempted a real network call")

    monkeypatch.setattr(ticktick_api, "_open", _forbidden)


@pytest.fixture(autouse=True)
def _no_real_credentials(monkeypatch, tmp_path):
    monkeypatch.setattr(ticktick_api, "DEFAULT_ENV_FILE", str(tmp_path / "no-such-env"))
    for var in (
        "TICKTICK_ACCESS_TOKEN",
        "TICKTICK_TOKEN_EXPIRES_AT",
        "AGGREGATOR_TICKTICK_TOKEN",
        "AGGREGATOR_TICKTICK_TOKEN_FILE",
        "AGGREGATOR_TICKTICK_DIR",
        # A notifier configured on the developer's machine would install the
        # real one on the tests below that are about having NO channel.
        cli.NOTIFY_COMMAND_ENV_VAR,
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _live_project_only(monkeypatch):
    """The poll covers ``live`` and serves ``t9`` from it. The baseline's ``t1``
    sits in ``gone``, which the poll never looked at — so t1 is retained, never
    inferred completed, and REPORTED once."""
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
        json.dumps({"t1": {"task": {"id": "t1", "projectId": "gone"}, "last_seen": "x"}}),
        encoding="utf-8",
    )
    return path


class _Sink:
    def write(self, items) -> WriteCounts:
        return WriteCounts(added=len(items))


class _WatchedTerminal(io.StringIO):
    """stderr with a person in front of it."""

    def isatty(self) -> bool:
        return True


def _source(tmp_path, state_file) -> TickTickSource:
    return TickTickSource(
        backup_dir=tmp_path / "no-downloads",
        archive_dir=tmp_path / "no-archive",
        token="fake-token",
        state_file=state_file,
    )


def _run(tmp_path, state_file, **kwargs):
    """One whole run through the runner, on a fresh adapter — what a timer does.

    No ``notify=`` means the shipped default, which is the configuration this
    finding is about.
    """
    adapter = TickTickAdapter(source=_source(tmp_path, state_file))
    return asyncio.run(run_imports([adapter], _Sink(), **kwargs))


def _reported(report) -> list[str]:
    return [e for e in report.errors if UNCOVERED in e]


def _delivering(heard: list[str]):
    """A notifier that reached a human AND SAYS WHAT IT CARRIED."""

    def notify(report) -> Delivery:
        payload = "\n".join(report.errors)
        heard.extend(report.errors)
        return Delivery.accepted(payload, report.errors)

    return notify


def _lossy(heard: list[str], keep: int):
    """A channel with a size budget — every real one has. It sends the first
    ``keep`` lines and, because the declaration is read out of what it sent,
    cannot claim the rest."""

    def notify(report) -> Delivery:
        payload = "\n".join(report.errors[:keep])
        heard.extend(report.errors[:keep])
        return Delivery.accepted(payload, report.errors)

    return notify


def _broken(report) -> None:
    raise FileNotFoundError("notify-send: command not found")


def _ingest(tmp_path, state_file, store) -> int:
    """``aggregator ingest ticktick`` — the single-source path, no notifier."""
    return cli.main(
        ["ingest", "ticktick"],
        _store=store,
        _sources={"ticktick": _source(tmp_path, state_file)},
    )


# -- HIGH 1 ---------------------------------------------------------------


def test_the_default_no_op_notifier_is_not_a_delivery_channel(tmp_path, state_file):
    """THE repro. ``run_imports``' default ``notify`` does nothing at all — it
    returns without raising, which was read as "the report reached a human", so
    the receipt was stamped and every later poll went quiet about a
    disappearance nobody was ever told about."""
    first = _run(tmp_path, state_file)
    assert _reported(first), first.errors

    second = _run(tmp_path, state_file)

    assert _reported(second), (
        "permanent silence: nothing was configured that could reach a human, "
        f"yet the receipt says one was told. run 2 reported {second.errors}"
    )


def test_a_hook_that_does_nothing_has_nothing_to_declare_delivery_with():
    """Why a fourth layer cannot be written the same way as the first three.

    Delivery is a value the hook returns, and the do-nothing hooks are annotated
    ``-> None``: neither sends a payload, so neither has anything to build a
    ``Delivery`` out of. The previous three fixes were all conditions at the
    call site, and a condition is a thing the next call site forgets."""
    report = RunReport()

    assert not isinstance(_no_notification(report), Delivery)
    assert not isinstance(cli._silent_notification(report), Delivery)
    assert not Delivery().covers(["anything at all"])


def test_a_truthy_return_value_is_not_a_declaration(tmp_path, state_file):
    """Identity, not truthiness. The obvious hook —
    ``lambda report: subprocess.run(...)`` — returns a CompletedProcess, which
    is truthy even when the notifier exited non-zero."""
    first = _run(tmp_path, state_file, notify=lambda report: "sent, honest")
    assert _reported(first)

    assert _reported(_run(tmp_path, state_file)), "a truthy value passed for delivery"


# -- HIGH 2 ---------------------------------------------------------------


def test_stderr_on_an_unattended_run_is_not_a_delivery_channel(
    tmp_path, state_file, capsys
):
    """THE repro. ``aggregator ingest ticktick`` under the systemd timer writes
    its errors into the journal and stamps the receipt for having done so — but
    nobody reads a journal unprompted, so the one alert was spent on nothing.
    (pytest's captured stderr is not a tty, which is that exact shape.)"""
    store = Store(db_path=tmp_path / "cache.db")
    store.migrate()
    assert not sys.stderr.isatty(), "this fixture is the unattended shape"

    assert _ingest(tmp_path, state_file, store) == cli.EXIT_COMPLETED_WITH_ERRORS
    assert UNCOVERED in capsys.readouterr().err

    assert _ingest(tmp_path, state_file, store) == cli.EXIT_COMPLETED_WITH_ERRORS, (
        "the disappearance was written to a journal nobody reads and then "
        "marked as reported to a human"
    )


def test_stderr_with_somebody_watching_it_is_a_delivery_channel(
    tmp_path, state_file, monkeypatch
):
    """The other half, and the round-4 fix it must not cost: at an interactive
    terminal the print IS the delivery — a person is looking at it — so the
    alarm is raised once and not on every subsequent run."""
    store = Store(db_path=tmp_path / "cache.db")
    store.migrate()
    terminal = _WatchedTerminal()
    monkeypatch.setattr(sys, "stderr", terminal)

    assert _ingest(tmp_path, state_file, store) == cli.EXIT_COMPLETED_WITH_ERRORS
    assert UNCOVERED in terminal.getvalue()

    assert _ingest(tmp_path, state_file, store) == 0, (
        "a human was told and the alarm fired again anyway"
    )


def test_a_report_the_terminal_never_showed_is_not_delivered(
    tmp_path, state_file, monkeypatch
):
    """The other way the print misses its audience. Only the first
    ``ERROR_PRINT_LIMIT`` errors are shown, and the vanished-project line is
    appended mid-run — enough earlier failures push it off the end, and the
    receipt would be stamped for a sentence that was never on screen."""
    terminal = _WatchedTerminal()
    monkeypatch.setattr(sys, "stderr", terminal)
    store = Store(db_path=tmp_path / "cache.db")
    store.migrate()
    noisy = _source(tmp_path, state_file)
    real_iter = noisy.iter_records

    def iter_records(since, errors=None):
        if errors is not None:
            errors.extend(f"an earlier unrelated failure {n}" for n in range(5))
        return real_iter(since, errors=errors)

    noisy.iter_records = iter_records

    first = cli.main(
        ["ingest", "ticktick"], _store=store, _sources={"ticktick": noisy}
    )

    assert first == cli.EXIT_COMPLETED_WITH_ERRORS
    assert UNCOVERED not in terminal.getvalue(), "precondition: it was truncated"
    assert _ingest(tmp_path, state_file, store) == cli.EXIT_COMPLETED_WITH_ERRORS, (
        "a receipt was stamped for a line the operator never saw"
    )


# -- round 7 HIGH: the payload is the declaration -------------------------


class _Noisy:
    """Another source having a bad day. No report barrier: it repeats its own
    errors next run regardless, so nothing of its is gating."""

    name = "noisy"

    def __init__(self, count: int = 5) -> None:
        self._count = count

    async def get_data(self) -> AsyncIterator[object]:
        return
        yield  # pragma: no cover - makes this an async generator

    def drain_errors(self) -> list[str]:
        return [f"unrelated failure {n}" for n in range(self._count)]


def _run_all(adapters, **kwargs):
    return asyncio.run(run_imports(adapters, _Sink(), **kwargs))


def _toast(tmp_path: Path, monkeypatch) -> Path:
    """Install a real notify program that records the payload it was handed.

    A stand-in for ``notify-send`` rather than a patched ``subprocess.run``: the
    whole finding is about what the channel ACTUALLY RECEIVED, so the assertion
    should be made on bytes that crossed a process boundary.
    """
    log = tmp_path / "toast.txt"
    script = tmp_path / "fake-notify-send"
    script.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$@\" >> " + shlex.quote(str(log)) + "\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    monkeypatch.setenv(cli.NOTIFY_COMMAND_ENV_VAR, str(script))
    return log


def test_a_line_the_channel_never_carried_is_not_delivered(tmp_path, state_file):
    """THE round-7 repro, stated against any lossy channel rather than today's
    limit. Five failures from other sources, then TickTick's vanished project;
    the channel has room for five lines, so the sixth reaches nobody — and the
    receipt was stamped anyway, because delivery was declared for the RUN while
    the payload was a slice of it."""
    heard: list[str] = []
    first = _run_all(
        [_Noisy(), TickTickAdapter(source=_source(tmp_path, state_file))],
        notify=_lossy(heard, keep=5),
    )
    assert _reported(first), first.errors
    assert not [e for e in heard if UNCOVERED in e], "precondition: it was cut"

    second = _run_all(
        [TickTickAdapter(source=_source(tmp_path, state_file))],
        notify=_delivering([]),
    )

    assert _reported(second), (
        "a receipt was stamped for a line the notifier was never given: "
        f"run 2 reported {second.errors}"
    )


def test_an_adapter_that_could_only_show_half_its_report_stays_loud(
    tmp_path, state_file
):
    """The rule one level down: coverage is per ADAPTER, not per line, because a
    receipt is the adapter's decision to stop saying something. TickTick reports
    six things — five unreadable backups and the vanished project — and the
    channel carries the first five. The gating line got through; the adapter
    still may not go quiet, because a report it could only half-deliver is not
    one it can reason about which half of."""
    heard: list[str] = []
    noisy = _source(tmp_path, state_file)
    real_iter = noisy.iter_records

    def iter_records(since, errors=None):
        if errors is not None:
            errors.extend(f"unreadable backup {n}" for n in range(5))
        return real_iter(since, errors=errors)

    noisy.iter_records = iter_records

    first = _run_all(
        [TickTickAdapter(source=noisy)], notify=_lossy(heard, keep=5)
    )
    assert _reported(first), first.errors
    assert len(first.errors) == 6, first.errors

    assert _reported(_run_all(
        [TickTickAdapter(source=_source(tmp_path, state_file))],
        notify=_delivering([]),
    )), "the adapter went quiet on a report the channel only partly carried"


def test_the_shipped_toast_spends_its_budget_on_the_gating_line_first(
    tmp_path, state_file, monkeypatch
):
    """The other half of the fix, and the reason it is not merely loud forever.

    Making an elided line undeliverable is correct and, on its own, means a
    chronically noisy run re-alarms about a known gap on every tick — which is
    the round-4 problem back again. So the toast's five slots go to the lines
    that GATE a receipt first: a source with no report barrier repeats its
    errors next run anyway, so its line is the cheap one to drop."""
    log = _toast(tmp_path, monkeypatch)

    first = _run_all(
        [_Noisy(), TickTickAdapter(source=_source(tmp_path, state_file))],
        notify=cli._desktop_notification,
    )
    assert _reported(first), first.errors
    payload = log.read_text()
    assert UNCOVERED in payload, f"pushed out by five unrelated failures:\n{payload}"

    second = _run_all(
        [_Noisy(), TickTickAdapter(source=_source(tmp_path, state_file))],
        notify=cli._desktop_notification,
    )

    assert _reported(second) == [], (
        f"a human was shown the line and it fired again: {second.errors}"
    )


class _NoisySource:
    """The same bad day, as a SOURCE driven through the real bridge.

    Which is how all eight non-TickTick sources actually reach the runner, and
    the difference the round-8 MEDIUM lived in: ``_Noisy`` above wears no
    ``commit_after_report`` at all, while ``SyncSourceAdapter`` defines one for
    every source it wraps.
    """

    name = "noisy-source"

    def iter_records(self, since, errors=None):
        if errors is not None:
            errors.extend(f"unrelated failure {n}" for n in range(5))
        return iter(())


def test_a_bridged_source_cannot_starve_the_toast_of_the_gating_line(
    tmp_path, state_file, monkeypatch
):
    """THE round-8 MEDIUM repro, and it is the test above with the noisy source
    adapted the way real ones are.

    ``SupportsReportBarrier`` was a ``runtime_checkable`` Protocol, so
    ``isinstance`` matched on METHOD PRESENCE — and the bridge defines
    ``commit_after_report`` for every source it wraps, because it merely
    forwards. All nine sources therefore counted as receipt-gating,
    ``gating_errors`` was the whole error list, "spend the budget on the gating
    lines first" reordered nothing, and five chronic errors from anywhere kept
    TickTick's uncovered-project line out of the toast — on every run, forever.
    Exactly the starvation ``gating_errors`` was added to prevent.
    """
    log = _toast(tmp_path, monkeypatch)

    def run():
        return _run_all(
            [
                SyncSourceAdapter(_NoisySource()),
                TickTickAdapter(source=_source(tmp_path, state_file)),
            ],
            notify=cli._desktop_notification,
        )

    first = run()
    assert _reported(first), first.errors
    assert first.gating_errors == first.errors_from("ticktick"), (
        "only genuinely gating lines may claim the budget: "
        f"{first.gating_errors}"
    )
    payload = log.read_text()
    assert UNCOVERED in payload, f"starved by five unrelated failures:\n{payload}"

    assert _reported(run()) == [], "a human was shown the line and it fired again"


# -- round 7 MEDIUM: an interactive `ingest --all` has an audience ---------


def _ingest_all(tmp_path, state_file, store) -> int:
    """``aggregator ingest --all`` with nothing configured — no ``--notify``,
    no ``$AGGREGATOR_NOTIFY_COMMAND``. stderr is the only channel there is."""
    return cli.main(
        ["ingest", "--all"],
        _store=store,
        _adapters=[TickTickAdapter(source=_source(tmp_path, state_file))],
    )


def test_an_interactive_ingest_all_is_a_delivery_channel(
    tmp_path, state_file, monkeypatch
):
    """THE repro. ``--all`` resolves to ``_silent_notification`` and does its
    reporting AFTER ``run_imports`` returns, so the runner never saw the one
    channel that worked: a person sitting at the terminal watching the report
    scroll past got the same TickTick gap re-reported every run, forever."""
    store = Store(db_path=tmp_path / "cache.db")
    store.migrate()
    terminal = _WatchedTerminal()
    monkeypatch.setattr(sys, "stderr", terminal)

    assert _ingest_all(tmp_path, state_file, store) == cli.EXIT_COMPLETED_WITH_ERRORS
    assert UNCOVERED in terminal.getvalue()

    assert _ingest_all(tmp_path, state_file, store) == 0, (
        "the operator read it off their own terminal and it alarmed again"
    )


def test_an_unattended_ingest_all_is_not_a_delivery_channel(
    tmp_path, state_file, capsys
):
    """The half the fix above must not cost. Under the systemd timer stdout and
    stderr go to the journal — written, retained, and read by nobody unprompted.
    (pytest's captured streams are not a tty, which is that exact shape.)"""
    store = Store(db_path=tmp_path / "cache.db")
    store.migrate()
    assert not sys.stderr.isatty(), "this fixture is the unattended shape"

    assert _ingest_all(tmp_path, state_file, store) == cli.EXIT_COMPLETED_WITH_ERRORS
    assert UNCOVERED in capsys.readouterr().err

    assert _ingest_all(tmp_path, state_file, store) == cli.EXIT_COMPLETED_WITH_ERRORS, (
        "the journal was treated as an audience"
    )


# -- round 8 HIGH: a line is delivered by ITSELF, not by one containing it -


class _Probe:
    """An adapter that reports exactly what it is told and records its receipt.

    Deliberately not TickTick: this finding is about the identity test inside
    ``Delivery``, and the cheapest way to state it is two adapters whose report
    lines stand in the offending relation.
    """

    # The explicit opt-in — round 8 MEDIUM. Having ``commit_after_report`` is
    # not a declaration that this adapter's lines gate a receipt; saying so is.
    gates_report = True

    def __init__(self, name: str, errors: list[str]) -> None:
        self.name = name
        self._errors = list(errors)
        self.receipted = False

    async def get_data(self) -> AsyncIterator[object]:
        return
        yield  # pragma: no cover - makes this an async generator

    def drain_errors(self) -> list[str]:
        return list(self._errors)

    def commit_after_report(self) -> None:
        self.receipted = True


def test_a_line_is_not_delivered_by_a_longer_line_that_contains_it():
    """THE round-8 repro. Membership was ``line in payload`` — SUBSTRING — so a
    line was "delivered" by any longer line that happened to contain it.

    ``box`` and ``dropbox`` both report ``file unreadable``; rendered, the first
    is a substring of the second. Send only dropbox's line and box's receipt was
    stamped for a sentence that never left the process — which is the exact
    thing rounds 4 through 7 were each about, arrived at by a fifth route.

    Adapter names are never validated against being suffixes of one another, and
    two lines from ONE adapter need no coincidence at all (``A`` and ``A plus
    detail``), which the next test pins directly.
    """
    box = _Probe("box", ["file unreadable"])
    dropbox = _Probe("dropbox", ["file unreadable"])

    def notify(report) -> Delivery:
        # A channel with room for one source's worth of report — every real one
        # has a budget, and which lines fall inside it is not this class's call.
        return Delivery.accepted("\n".join(report.errors_from("dropbox")), report.errors)

    _run_all([box, dropbox], notify=notify)

    assert dropbox.receipted, "the line that WAS sent must still earn its receipt"
    assert not box.receipted, (
        "a receipt was stamped for a line the channel never carried — it was "
        "merely a substring of one that it did"
    )


def test_one_adapters_own_prefix_line_is_not_delivered_by_its_longer_one():
    """The same defect without the naming coincidence, and the more likely one.

    A source that reports both ``A`` and ``A plus detail`` is ordinary — a
    summary line and a detailed one. Under substring matching, a channel that
    carried only the detailed line covered BOTH, so the adapter went quiet about
    a sentence nobody read. Stated on ``Delivery`` directly because that is
    where the rule lives; the test above proves the runner honours it.
    """
    delivered = Delivery.accepted("A plus detail", ["A", "A plus detail"])

    assert delivered.lines == frozenset({"A plus detail"})
    assert not delivered.covers(["A", "A plus detail"])


def test_the_delivered_set_is_read_off_whole_lines():
    """The neighbours of the identity test, each decided the loud way.

    A trailing newline on the payload and a CRLF channel must both still
    deliver: getting those wrong is safe but permanently loud, and loud-forever
    is the round-4 alarm fatigue this whole mechanism exists to avoid. Padding
    is the other direction and is a MISMATCH — a channel that changed the text
    does not get to have this class guess how much change is still the same
    line.
    """
    assert Delivery.accepted("a\nb\n", ["b"]).lines == frozenset({"b"})
    assert Delivery.accepted("a\r\nb", ["a", "b"]).lines == frozenset({"a", "b"})
    assert Delivery.accepted("A ", ["A"]).lines == frozenset()
    assert Delivery.accepted(" A", ["A"]).lines == frozenset()
    assert Delivery.accepted("", ["A"]).lines == frozenset()
    # A blank report line gates nothing: it carries no sentence to a human, and
    # under substring matching it was delivered by every payload in existence.
    assert Delivery.accepted("anything", ["", "   "]).lines == frozenset()


def test_a_multi_line_report_entry_is_delivered_only_as_one_block():
    """Exception text with an embedded newline is real, and the channel carries
    it verbatim. Refusing it would keep an adapter loud forever over a line the
    human demonstrably read — so it is matched, but only as a CONTIGUOUS run of
    lines, never assembled out of pieces of other reports."""
    entry = "CalledProcessError: notify-send failed\n  stderr: no such display"

    assert Delivery.accepted(f"first line\n{entry}", [entry]).lines == frozenset({entry})
    scattered = "CalledProcessError: notify-send failed\nunrelated\n  stderr: no such display"
    assert Delivery.accepted(scattered, [entry]).lines == frozenset()


def test_two_identical_reported_lines_are_one_sentence():
    """THE DECISION on duplicates, stated so the next reader does not re-open it.

    A line reported twice and carried once counts as delivered for both
    occurrences. This is a set of TEXT and a receipt suppresses a repeat of that
    text; a human who read the sentence has read it, and a verbatim second copy
    cannot be a sentence they missed.

    It is the fail-safe choice because it can never silence anything unread —
    the only thing it over-covers is an exact duplicate of a line that WAS on
    screen. Counting multiplicity instead would leave an adapter that stuttered
    loud forever with nothing an operator could do about it, and adapter-name
    prefixing already means two DIFFERENT adapters can never render the same
    line.
    """
    stutter = _Probe("stutter", ["the same failure", "the same failure"])

    def notify(report) -> Delivery:
        # Carried once, reported twice.
        return Delivery.accepted("stutter: the same failure", report.errors)

    _run_all([stutter], notify=notify)

    assert stutter.receipted, "the operator read that sentence; it may go quiet"


# -- the contract, end to end ---------------------------------------------


def test_the_delivery_contract_end_to_end(tmp_path, state_file):
    """ONE test for the whole rule, because this bug has now grown three layers
    by each fix pinning only the layer in front of it.

    No channel at all: reported on every run, forever, until there is one.
    A channel that raised: the same — nobody heard it, so it stands.
    A channel that delivered: reported exactly once, then quiet.

    Anything that breaks the rule breaks this, whatever route it takes to the
    receipt."""
    heard: list[str] = []

    # 1. Nothing configured. The disappearance is unreported and stays that way.
    assert _reported(_run(tmp_path, state_file)), "run 1 must raise it"
    assert _reported(_run(tmp_path, state_file)), "and keep raising it"

    # 2. A notifier that blew up reached nobody either (round 5, unregressed).
    raised = _run(tmp_path, state_file, notify=_broken)
    assert _reported(raised)
    assert any("notify hook failed" in e for e in raised.errors)
    assert _reported(_run(tmp_path, state_file)), "a raise is not a delivery"

    # 3. One that worked, and said so.
    delivered = _run(tmp_path, state_file, notify=_delivering(heard))
    assert _reported(delivered)
    assert [e for e in heard if UNCOVERED in e], f"the human was told: {heard}"

    # 4. Exactly once (round 4, unregressed): quiet from here, whatever channel
    #    the next run has. An alarm on every 30-minute tick is one an operator
    #    learns to ignore, which costs the next real failure its audience.
    assert _reported(_run(tmp_path, state_file)) == []
    assert _reported(_run(tmp_path, state_file, notify=_delivering(heard))) == []


def test_the_baseline_still_advances_with_no_channel_at_all(tmp_path, state_file):
    """The regression the loud-forever direction must not smuggle in. Only the
    REPORT waits for delivery; the write barrier may never wait for anything. A
    frozen baseline never learns about a task created after the freeze, so that
    task's later disappearance is invisible to every future poll — unbounded
    loss, versus one duplicated line."""
    _run(tmp_path, state_file)

    baseline = json.loads(state_file.read_text())
    assert "t9" in baseline, "the poll's own tasks must land whatever notify did"
    assert "t1" in baseline, "and the uncovered task is still never inferred done"
    assert ticktick_api.uncovered_mark(baseline["t1"]) is None
