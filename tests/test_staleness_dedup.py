"""A stale input is reported ONCE PER EPISODE, then goes quiet, then re-arms.

THE DEFECT. ``staleness_warnings`` is correct and was raised on every single
run. ``substack``'s export is 31 days old right now and ``chatgpt`` has never
been ingested at all, so the moment the 30-minute systemd timer lands with
``AGGREGATOR_NOTIFY_COMMAND`` wired up, that is a desktop notification every
half hour, forever, until a human visits a vendor's export page.

It is the SAME failure this branch spent four review rounds eliminating for a
vanished TickTick project: a permanently-red signal trains an operator to
dismiss toasts unread, which costs the next real failure its audience. Only the
subject changed. So does the fix — the episode marker waits for the same
``port.Delivery`` the report barriers wait for, computed in the same two places
by the same functions, because a second notion of "a human was told" is exactly
how that bug got to eight rounds.

AN EPISODE is one continuous period during which a source's input is older than
the threshold. Its identity is the INPUT — the mtime the adapter reports — not
the run and not the warning text, which changes every day as the age ticks up.
So a refreshed export ends the episode and a later one begins a new, loud one.

Stubs first: nothing here may reach the network, the developer's TickTick
credential, their real open-task baseline, or their real markers.
"""
from __future__ import annotations

import asyncio
import io
import json
import shlex
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from aggregator import cli
from aggregator.core.store import Store
from aggregator.imports.ingest_state import (
    STALE_INPUTS,
    IngestMarkers,
    default_marker_path,
    stale_input_markers,
)
from aggregator.imports.port import Delivery, WriteCounts
from aggregator.imports.registry import default_adapters
from aggregator.imports.runner import (
    AdapterReport,
    RunReport,
    commit_staleness_receipts,
    plan_staleness_report,
    run_imports,
)
from aggregator.sources import ticktick_api

NOW = datetime(2026, 8, 15, 12, 0, tzinfo=UTC)
THRESHOLD = 14


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


def _aged(days: float) -> datetime:
    return NOW - timedelta(days=days)


class _Export:
    """A source whose input is an archive a human downloads by hand.

    ``SupportsInputFreshness`` is structural, so having the method is the opt-in.
    The mtime is a value a test MOVES, because that is what a human refreshing
    an export does and it is the whole re-arm condition.
    """

    def __init__(self, name: str, exported: datetime | None) -> None:
        self.name = name
        self.exported = exported

    async def get_data(self):
        return
        yield  # pragma: no cover - makes this an async generator

    def input_freshness(self) -> datetime | None:
        return self.exported


class _LiveApi:
    """github / dropbox: no export ritual, so no staleness question to ask."""

    name = "github"

    async def get_data(self):
        return
        yield  # pragma: no cover - makes this an async generator


class _Sink:
    def write(self, items) -> WriteCounts:
        return WriteCounts()


class _WatchedTerminal(io.StringIO):
    """stderr with a person in front of it."""

    def isatty(self) -> bool:
        return True


def _heard(report) -> Delivery:
    """A channel that carried the whole report, and says what it carried."""
    return Delivery.accepted("\n".join(report.reported), report.reported)


def _lossy(keep: int):
    """A channel with a size budget. Every real one has."""

    def notify(report) -> Delivery:
        return Delivery.accepted("\n".join(report.reported[:keep]), report.reported)

    return notify


def _broken(report) -> None:
    raise FileNotFoundError("notify-send: command not found")


def _run(adapters, markers, *, notify=_heard, days=THRESHOLD, now=NOW) -> RunReport:
    return asyncio.run(
        run_imports(
            adapters,
            _Sink(),
            notify=notify,
            stale_after_days=days,
            now=now,
            markers=markers,
        )
    )


def _markers(tmp_path) -> IngestMarkers:
    return IngestMarkers(tmp_path / "state" / "aggregator" / "ingest" / "markers.json")


def _stale_lines(report: RunReport, name: str = "substack") -> list[str]:
    return [w for w in report.warnings if w.startswith(f"{name}:")]


# -- the episode: loud once, quiet after, loud again when it recurs -------


def test_the_first_run_past_the_threshold_warns(tmp_path):
    """Requirement 1. Nothing is being traded away: a source that crosses the
    threshold is still reported the first time, in full, with its age."""
    report = _run([_Export("substack", _aged(31))], _markers(tmp_path))

    assert _stale_lines(report) == [
        "substack: input is 31 days stale (threshold 14). Nothing on this "
        "machine refreshes it — export a new one, or raise --stale-after-days "
        "if that is the intended cadence."
    ]


def test_the_same_stale_export_is_never_reported_twice(tmp_path):
    """Requirement 2, and THE defect. Two identical runs, one warning.

    Under the timer this is the difference between one toast and a critical
    toast every 30 minutes until somebody downloads a fresh export.
    """
    markers = _markers(tmp_path)
    exported = _aged(31)

    assert _stale_lines(_run([_Export("substack", exported)], markers))

    second = _run([_Export("substack", exported)], markers)
    assert _stale_lines(second) == []
    # HELD BACK, not merely absent. The two are the same empty warning list and
    # opposite states — the source going fresh would also produce no line — so
    # the plan says which it was, and the marker that did it.
    assert second.stale_episodes.suppressed["substack"]["stale_after_days"] == 14

    assert _stale_lines(_run([_Export("substack", exported)], markers)) == []


def test_a_refreshed_export_is_silent_and_re_arms(tmp_path):
    """Requirement 3. The human did the thing the warning asked for.

    No warning — it is not stale any more — AND the marker is gone, because a
    suppression that is no longer true would make ``aggregator status`` claim
    a source is being held quiet when nothing is being held at all.
    """
    markers = _markers(tmp_path)
    _run([_Export("substack", _aged(31))], markers)
    assert "substack" in markers.load(STALE_INPUTS)

    report = _run([_Export("substack", _aged(0))], markers)

    assert report.warnings == []
    assert markers.load(STALE_INPUTS) == {}, "the marker outlived its episode"


def test_a_refreshed_export_that_goes_stale_again_is_loud_again(tmp_path):
    """Requirement 4, and the reason suppression is keyed on the INPUT rather
    than on the source. Reported, refreshed, and stale once more months later:
    that is a NEW episode and a fact the operator has not been told."""
    markers = _markers(tmp_path)
    _run([_Export("substack", _aged(31))], markers)
    _run([_Export("substack", _aged(0))], markers)

    later = _run(
        [_Export("substack", NOW)], markers, now=NOW + timedelta(days=40)
    )

    assert _stale_lines(later), "a second episode went unreported"
    assert "40 days stale" in later.warnings[0]


def test_a_different_export_that_is_ALSO_stale_is_a_new_episode(tmp_path):
    """The case that makes the episode key the INPUT rather than the source,
    and the one a run of fresh-then-stale never reaches.

    A new archive can arrive already old: ``TickTickSource._copy_private``
    deliberately RESTORES the source mtime onto the archived copy, and a drop
    the user copies out of an old backup keeps its own. The previous warning's
    subject is gone, the operator's action did not fix it, and telling them
    again is the direction that cannot cost an alert.
    """
    markers = _markers(tmp_path)
    _run([_Export("ticktick", _aged(100))], markers)

    replaced = _run([_Export("ticktick", _aged(20))], markers)

    assert _stale_lines(replaced, "ticktick"), "a different stale input went unreported"
    assert "20 days stale" in replaced.warnings[0]


def test_a_second_source_is_never_muted_by_the_first(tmp_path):
    """Requirement 5. Markers are keyed by source: never a global mute.

    A run-scoped "we already warned about staleness" flag would be the cheap
    implementation and would silently cost chatgpt its only alert for as long
    as substack stayed stale.
    """
    markers = _markers(tmp_path)
    _run([_Export("substack", _aged(31))], markers)

    report = _run(
        [_Export("substack", _aged(31)), _Export("chatgpt", _aged(90))], markers
    )

    assert _stale_lines(report, "substack") == [], "the reported one must stay quiet"
    assert _stale_lines(report, "chatgpt"), "a different source was muted by it"


def test_a_source_with_no_export_ritual_is_never_marked(tmp_path):
    """github reads a live API: there is no staleness question to ask, so it
    must not appear in the markers to be re-armed or suppressed either."""
    markers = _markers(tmp_path)

    report = _run([_LiveApi(), _Export("substack", _aged(31))], markers)

    assert report.warnings == _stale_lines(report)
    assert set(markers.load(STALE_INPUTS)) == {"substack"}


# -- nothing goes quiet until a human was told ----------------------------


def test_no_channel_at_all_means_loud_forever(tmp_path):
    """Requirement 6, THE case. The shipped default ``notify`` does nothing —
    it returns without raising, which is the round-6 defect — so no marker may
    be earned and the warning stands on every run until a channel exists.

    Loud forever is bounded by configuring a notifier, which is the very action
    the noise is asking for.
    """
    markers = _markers(tmp_path)
    exported = _aged(31)

    def silent_run() -> RunReport:
        """No ``notify=`` at all: the shipped default, which does nothing."""
        return asyncio.run(
            run_imports(
                [_Export("substack", exported)],
                _Sink(),
                stale_after_days=THRESHOLD,
                now=NOW,
                markers=markers,
            )
        )

    assert _stale_lines(silent_run()), "run 1 must raise it"
    assert _stale_lines(silent_run()), "a run with no channel at all went quiet"
    assert markers.load(STALE_INPUTS) == {}


def test_a_notify_hook_that_raised_delivered_nothing(tmp_path):
    """A missing ``notify-send`` reaches nobody. Same answer as no hook."""
    markers = _markers(tmp_path)
    exported = _aged(31)

    raised = _run([_Export("substack", exported)], markers, notify=_broken)
    assert _stale_lines(raised)
    assert any("notify hook failed" in e for e in raised.errors)

    assert _stale_lines(_run([_Export("substack", exported)], markers)), (
        "a raise was read as a delivery"
    )


def test_a_truthy_return_value_is_not_a_declaration(tmp_path):
    """TYPE, not truthiness — the same rule the report barriers follow. The
    obvious hook, ``lambda report: subprocess.run(...)``, returns a truthy
    CompletedProcess even when the notifier exited non-zero."""
    markers = _markers(tmp_path)
    exported = _aged(31)

    _run([_Export("substack", exported)], markers, notify=lambda r: "sent, honest")

    assert _stale_lines(_run([_Export("substack", exported)], markers))


def test_a_warning_the_channel_truncated_away_is_not_delivered(tmp_path):
    """The round-7 shape, on the new subject. A channel with a budget carries
    the first lines and drops the rest; the declaration is read out of what it
    sent, so the dropped warning cannot buy silence for itself."""
    markers = _markers(tmp_path)
    substack, chatgpt = _Export("substack", _aged(31)), _Export("chatgpt", _aged(90))

    first = _run([substack, chatgpt], markers, notify=_lossy(keep=1))
    assert len(first.warnings) == 2
    carried = first.warnings[0]

    second = _run([substack, chatgpt], markers, notify=_heard)

    assert carried not in second.warnings, "the line that WAS carried must go quiet"
    assert second.warnings, "a line the channel dropped bought silence anyway"


def test_the_delivered_set_is_the_one_the_report_barriers_use(tmp_path):
    """Requirement 6's other half: ONE notion of delivered, not two.

    ``Delivery`` is computed in exactly the places it already was, from exactly
    the payload the channel took. The only change is that warnings are offered
    to it alongside errors (``RunReport.reported``) — a line that is not offered
    can never be in the delivered set, so its marker could never be earned.
    """
    report = RunReport()
    report.warnings.append("substack: input is 31 days stale")

    assert report.reported == ["substack: input is 31 days stale"]
    assert not Delivery().covers(report.reported)
    assert Delivery.accepted(
        "substack: input is 31 days stale", report.reported
    ).covers(report.reported)


# -- the threshold is an operator's to move -------------------------------


def test_lowering_the_threshold_makes_a_suppressed_source_loud_again(tmp_path):
    """THE THRESHOLD RULE. A marker suppresses only a warning raised at a
    threshold AT LEAST AS LOOSE as the one it was earned at.

    Lowering ``--stale-after-days`` is the operator saying the old cadence was
    too generous: a source they were told about at 14 days is one they have NOT
    been told about at 7, so it says so again and re-earns its marker at the
    stricter number.
    """
    markers = _markers(tmp_path)
    exported = _aged(31)
    _run([_Export("substack", exported)], markers, days=14)

    tightened = _run([_Export("substack", exported)], markers, days=7)

    assert _stale_lines(tightened), "the operator tightened the rule and heard nothing"
    assert "threshold 7" in tightened.warnings[0]
    assert markers.load(STALE_INPUTS)["substack"]["stale_after_days"] == 7


def test_raising_the_threshold_does_not_re_report_the_same_export(tmp_path):
    """The other direction is not a new fact. Same input, already heard about,
    judged by a kinder rule — re-reporting it would be noise for nothing."""
    markers = _markers(tmp_path)
    exported = _aged(31)
    _run([_Export("substack", exported)], markers, days=14)

    assert _run([_Export("substack", exported)], markers, days=20).warnings == []


def test_a_threshold_raised_past_the_age_drops_the_marker(tmp_path):
    """"That IS the intended cadence" is one of the two fixes the warning
    offers. Once it is no longer stale it is no longer suppressed either, so
    ``status`` does not go on claiming a source is being held quiet."""
    markers = _markers(tmp_path)
    exported = _aged(31)
    _run([_Export("substack", exported)], markers, days=14)

    assert _run([_Export("substack", exported)], markers, days=60).warnings == []
    assert markers.load(STALE_INPUTS) == {}


# -- the missing export: chatgpt's real state today -----------------------


def test_a_missing_export_is_reported_once_too(tmp_path):
    """chatgpt has never been ingested at all, which is the LOUDER case: every
    count is zero and it looks exactly like a healthy run with nothing new. It
    still may not toast every 30 minutes forever."""
    markers = _markers(tmp_path)

    first = _run([_Export("chatgpt", None)], markers)
    assert "no input found" in first.warnings[0]

    assert _run([_Export("chatgpt", None)], markers).warnings == []


def test_no_threshold_re_arms_a_missing_export(tmp_path):
    """A missing export is not a matter of degree: no threshold makes "there is
    no file" more or less true, so moving it must not re-raise the warning."""
    markers = _markers(tmp_path)
    _run([_Export("chatgpt", None)], markers, days=14)

    assert _run([_Export("chatgpt", None)], markers, days=1).warnings == []


def test_an_export_appearing_at_last_but_already_old_is_a_new_fact(tmp_path):
    """"There is no export" and "there is one and it is 90 days old" are
    different sentences with the same remedy, and the operator was only ever
    told the first."""
    markers = _markers(tmp_path)
    _run([_Export("chatgpt", None)], markers)

    landed = _run([_Export("chatgpt", _aged(90))], markers)

    assert "90 days stale" in landed.warnings[0]


# -- the marker file ------------------------------------------------------


def test_the_marker_file_is_private_and_lands_atomically(tmp_path):
    """0600 and a rename, the same discipline as the open-task baseline. It
    holds the user's source names and export dates — no credential, but not
    something a state file should publish on their behalf either."""
    markers = _markers(tmp_path)

    _run([_Export("substack", _aged(31))], markers)

    assert markers.path.stat().st_mode & 0o777 == 0o600
    assert not markers.path.with_name(markers.path.name + ".tmp").exists()
    assert json.loads(markers.path.read_text())[STALE_INPUTS]["substack"]


def test_the_markers_live_under_the_state_dir_the_baseline_uses(monkeypatch, tmp_path):
    """State, not cache and not data: regenerable, but nothing else can
    reconstruct it. An unset OR EMPTY variable takes the spec default — reading
    an empty one literally yields a relative path, so the markers would land
    wherever the timer happened to start and the next run would find none."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    assert default_marker_path() == tmp_path / "xdg/aggregator/ingest/markers.json"

    monkeypatch.setenv("XDG_STATE_HOME", "")
    assert default_marker_path().is_absolute()


def test_an_unreadable_marker_file_warns_rather_than_suppressing(tmp_path):
    """FAIL SAFE, which is the opposite of the TickTick baseline's rule.

    There a broken file RAISES, because the next act would overwrite
    unrecoverable completions. Here the only thing a marker buys is silence, so
    a file nobody can read must resolve to "nothing is suppressed" — one toast
    too many, never one alert too few.
    """
    markers = _markers(tmp_path)
    _run([_Export("substack", _aged(31))], markers)
    markers.path.write_text("{not json at all", encoding="utf-8")

    report = _run([_Export("substack", _aged(31))], markers)

    assert _stale_lines(report), "an unreadable marker file bought silence"
    assert markers.load(STALE_INPUTS)["substack"], "and the file healed itself"


def test_a_marker_that_cannot_be_written_is_reported_not_raised(tmp_path):
    """The ingest succeeded and the human was told; all that is lost is the
    silence. Still said out loud, because a marker that never lands turns
    "reported once" back into "reported every 30 minutes" and nothing else in
    the run would say so."""
    unwritable = tmp_path / "wall" / "markers.json"
    unwritable.parent.mkdir()
    unwritable.parent.chmod(0o500)
    try:
        report = _run([_Export("substack", _aged(31))], IngestMarkers(unwritable))
    finally:
        unwritable.parent.chmod(0o700)

    assert _stale_lines(report)
    assert any("staleness markers could not be written" in e for e in report.errors)


def test_markers_for_sources_this_run_never_drove_are_left_alone(tmp_path):
    """A run that did not look at a source cannot speak for it. Pruning by
    absence would let one partial run erase every other source's marker and
    re-alarm about all of them."""
    markers = _markers(tmp_path)
    _run([_Export("substack", _aged(31)), _Export("chatgpt", None)], markers)

    _run([_Export("substack", _aged(0))], markers)

    assert set(markers.load(STALE_INPUTS)) == {"chatgpt"}


def test_a_caller_with_no_staleness_policy_touches_no_marker(tmp_path):
    """``stale_after_days=None`` skips the evaluation entirely, so it must
    neither suppress nor re-arm: a run that never asked the question has no
    business answering it."""
    markers = _markers(tmp_path)
    _run([_Export("substack", _aged(31))], markers)
    before = markers.load(STALE_INPUTS)

    report = asyncio.run(
        run_imports(
            [_Export("substack", _aged(0))],
            _Sink(),
            notify=_heard,
            markers=markers,
        )
    )

    assert report.warnings == []
    assert report.stale_episodes is None
    assert markers.load(STALE_INPUTS) == before


def test_the_steady_state_costs_no_write_at_all(tmp_path):
    """Every 30 minutes, forever. A read-modify-write on each tick would be a
    lot of churn for a file that changes about twice a year."""
    markers = _markers(tmp_path)
    _run([_Export("substack", _aged(31))], markers)
    stamped = markers.path.stat().st_mtime_ns

    _run([_Export("substack", _aged(31))], markers)

    assert markers.path.stat().st_mtime_ns == stamped


# -- the CLI: the shipped channels, end to end ----------------------------


def _store(tmp_path) -> Store:
    store = Store(db_path=tmp_path / "cache.db")
    store.migrate()
    return store


def _toast(tmp_path: Path, monkeypatch) -> Path:
    """A real notify program that records the payload it was handed.

    A stand-in for ``notify-send`` rather than a patched ``subprocess.run``:
    the whole rule is about what the channel ACTUALLY RECEIVED, so the
    assertion is made on bytes that crossed a process boundary.
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


def _ingest_all(tmp_path, store, adapters) -> int:
    return cli.main(["ingest", "--all"], _store=store, _adapters=adapters)


def test_the_shipped_toast_pops_once_and_then_stops(tmp_path, monkeypatch):
    """END TO END, through the command the systemd timer runs and the notifier
    ``AGGREGATOR_NOTIFY_COMMAND`` installs. This is the production shape of the
    finding: without the fix, this toast fires on every 30-minute tick until a
    human downloads a fresh Substack export."""
    log = _toast(tmp_path, monkeypatch)
    store = _store(tmp_path)
    exported = datetime.now(UTC) - timedelta(days=31)

    assert _ingest_all(tmp_path, store, [_Export("substack", exported)]) == 0
    assert "31 days stale" in log.read_text()
    log.unlink()

    assert _ingest_all(tmp_path, store, [_Export("substack", exported)]) == 0
    assert not log.exists(), "the same stale export toasted again"


def test_an_interactive_ingest_all_is_a_channel_for_warnings_too(
    tmp_path, monkeypatch
):
    """``ingest --all`` installs no notifier interactively and prints the report
    itself, AFTER ``run_imports`` returned — so the runner never sees the one
    channel that worked. Same second commit the report barriers get, for the
    same reason and off the same ``_stderr_delivery``."""
    store = _store(tmp_path)
    exported = datetime.now(UTC) - timedelta(days=31)
    first, second = _WatchedTerminal(), _WatchedTerminal()

    monkeypatch.setattr(sys, "stderr", first)
    assert _ingest_all(tmp_path, store, [_Export("substack", exported)]) == 0
    assert "31 days stale" in first.getvalue()

    monkeypatch.setattr(sys, "stderr", second)
    assert _ingest_all(tmp_path, store, [_Export("substack", exported)]) == 0
    assert "31 days stale" not in second.getvalue(), (
        "the operator read it off their own terminal and it warned again"
    )


def test_an_unattended_ingest_all_is_not_a_channel(tmp_path, capsys):
    """The half the test above must not cost. Under the timer stderr is the
    journal: written, retained, and read by nobody unprompted. (pytest's
    captured streams are not a tty, which is that exact shape.)"""
    store = _store(tmp_path)
    assert not sys.stderr.isatty(), "this fixture is the unattended shape"
    exported = datetime.now(UTC) - timedelta(days=31)

    assert _ingest_all(tmp_path, store, [_Export("substack", exported)]) == 0
    assert "31 days stale" in capsys.readouterr().err

    assert _ingest_all(tmp_path, store, [_Export("substack", exported)]) == 0
    assert "31 days stale" in capsys.readouterr().err, (
        "the journal was treated as an audience"
    )


def test_staleness_still_never_changes_the_exit_code(tmp_path, monkeypatch):
    """DELIBERATE, and unchanged by any of this: a stale export is not a failed
    run, and exit 3 on every tick for weeks would have systemd mark the unit
    failed the whole time. The notification IS the channel, which is exactly
    why deduplicating it was the thing that needed fixing."""
    _toast(tmp_path, monkeypatch)
    store = _store(tmp_path)

    assert _ingest_all(tmp_path, store, [_Export("chatgpt", None)]) == 0
    assert _ingest_all(tmp_path, store, [_Export("chatgpt", None)]) == 0


def test_status_shows_what_is_being_held_quiet(tmp_path, monkeypatch, capsys):
    """Requirement 7, and the other half of the bargain: quiet is only
    acceptable while it is not the same as forgotten. Listed on demand, beside
    the TickTick uncovered projects, rather than pushed at somebody every 30
    minutes."""
    _toast(tmp_path, monkeypatch)
    store = _store(tmp_path)
    exported = datetime.now(UTC) - timedelta(days=31)
    _ingest_all(tmp_path, store, [_Export("substack", exported)])
    capsys.readouterr()

    assert cli.main(["status"], _store=store) == 0

    out = capsys.readouterr().out
    assert "stale inputs" in out
    assert "substack" in out
    assert "threshold 14 days" in out
    assert str(default_marker_path()) in out


def test_status_json_carries_the_markers_too(tmp_path, monkeypatch, capsys):
    """The machine-readable surface, for the same reason: Raycast and any other
    caller must be able to see a suppression without parsing prose."""
    _toast(tmp_path, monkeypatch)
    store = _store(tmp_path)
    _ingest_all(tmp_path, store, [_Export("chatgpt", None)])
    capsys.readouterr()

    assert cli.main(["status", "--json"], _store=store) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["stale_input_markers"]["chatgpt"]["input_newest_at"] == ""
    assert payload["stale_input_markers"]["chatgpt"]["first_reported"]


def test_status_says_nothing_when_nothing_is_suppressed(tmp_path, capsys):
    """A permanent extra section would be its own kind of noise."""
    assert cli.main(["status"], _store=_store(tmp_path)) == 0

    assert "stale inputs" not in capsys.readouterr().out
    assert stale_input_markers() == {}


# -- the machinery this must not disturb ----------------------------------


def test_the_export_sources_did_not_become_receipt_gating(tmp_path):
    """THE DESIGN CALL, pinned. The staleness marker is the RUNNER's, not an
    adapter's: the runner computes the warning, from a threshold no adapter
    knows, for four sources that have no state file at all.

    Routing it through ``commit_after_report`` instead would have meant
    declaring ``gates_report`` on chatgpt, claude-web, substack and ticktick —
    and ``RunReport.gating_errors`` is what the toast spends its five-line
    budget on FIRST, precisely so a chronically noisy source cannot starve
    TickTick's uncovered-project line out of the payload. Four more gating
    sources is that starvation back again, which is the round-8 MEDIUM.
    """
    report = _run(
        [_Export("substack", _aged(31)), _Export("chatgpt", None)],
        _markers(tmp_path),
    )

    assert report.gating_errors == []
    assert not any(a.holds_report_barrier for a in report.adapters.values())
    # And on the real ones, which is where it would actually have cost the toast.
    assert [a.gates_report for a in default_adapters() if a.name != "ticktick"] == [
        False
    ] * 8
    assert next(a for a in default_adapters() if a.name == "ticktick").gates_report


def test_a_second_commit_cannot_undo_the_first(tmp_path):
    """``commit_staleness_receipts`` runs twice on the CLI path — once on what
    the notify hook declared, once on what a watched terminal showed. Rebuilding
    the section from the plan alone, the second call would erase the marker the
    first one earned."""
    markers = _markers(tmp_path)
    report = _run([_Export("substack", _aged(31))], markers)
    assert markers.load(STALE_INPUTS)["substack"]

    commit_staleness_receipts(report, Delivery())

    assert markers.load(STALE_INPUTS)["substack"], "an empty second delivery undid it"


def test_the_planner_is_side_effect_free_until_it_is_committed(tmp_path):
    """Two-phase, the same shape as ``plan_open_task_reconcile``: the diff needs
    the run, the write needs a human. Planning writes nothing at all."""
    markers = _markers(tmp_path)
    report = RunReport(
        adapters={
            "substack": AdapterReport(
                name="substack",
                offers_input_freshness=True,
                input_newest_at=_aged(31),
            )
        }
    )

    plan = plan_staleness_report(
        report, max_age_days=THRESHOLD, now=NOW, markers=markers
    )

    assert len(plan.warnings) == 1
    assert not markers.path.exists(), "planning wrote a marker"

    plan.commit(Delivery.accepted(plan.warnings[0], plan.warnings))

    assert markers.load(STALE_INPUTS)["substack"]["stale_after_days"] == THRESHOLD
