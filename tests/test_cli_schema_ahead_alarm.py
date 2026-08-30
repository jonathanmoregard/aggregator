"""The writer refuses a newer cache — and a refusal nobody hears is the bug.

``Store.migrate()`` now raises ``SchemaAheadError`` rather than stamping a
newer cache back down to this build's version. That is the correct thing for
the DATA and, on its own, a regression for the OPERATOR: an uncaught exception
out of ``migrate()`` reaches ``aggregator-ingest.service`` as a traceback and a
non-zero exit, every thirty minutes, forever. The unit's ``OnFailure=`` notifier
carries no debounce, so that is a ``notify-send -u critical`` toast forty-eight
times a day about a condition that cannot change until a human deploys a new
build. Alarm fatigue is not a smaller version of silence; it is the same
outcome by a different route, and this repo already says so in as many words —
see the "deliberately no fourth code" paragraph in ``README.md``, which refuses
a distinct non-zero exit for known poison on exactly this reasoning.

SO THE CLI OWNS THE ALARM, and its contract is:

1. THE JOURNAL, EVERY RUN, UNDEBOUNCED. stderr costs nothing and nothing
   suppresses it, so the full diagnosis is there for whoever eventually looks.
2. ONE CRITICAL DESKTOP TOAST PER 24 h, through the notifier the unit already
   installs in ``AGGREGATOR_NOTIFY_COMMAND`` — the proven path, not a second
   mechanism invented beside it. The debounce stamp is armed ONLY after the
   notifier exits 0, copying ``mkFailureNotify`` in ``nix/aggregator.nix``,
   including its rule that a notifier gets its OWN stamp and never shares one.
3. EXIT 0 WHEN, AND ONLY WHEN, A HUMAN HAS BEEN TOLD. A delivered toast is the
   receipt; holding a receipt, the process has no business also failing its
   unit and firing the same toast twice from ``OnFailure=``.
4. EXIT NON-ZERO WHEN NOBODY COULD BE TOLD — notifier missing, unresolvable,
   or exiting non-zero. Then the unit's ``OnFailure=`` is the last channel
   left and it must fire. This is fail-closed on the reporting path: silence
   has to mean "a human was told", never "we could not tell anyone".

Point 4 is also why an interactive run with no notifier configured exits
non-zero: the operator is reading stderr, and a shell that returns 0 after
refusing to do the work would be lying to a script.
"""

from __future__ import annotations

import os
import shlex
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from aggregator.cli import NOTIFY_COMMAND_ENV_VAR, main
from aggregator.core.store import SCHEMA_VERSION, Store
from aggregator.sources.base import ObservationRow, SessionRow

_TS = datetime(2026, 8, 30, tzinfo=UTC)

#: 24 h, spelled out here rather than imported, so a change to the production
#: constant has to be made twice and is noticed the second time.
_DEBOUNCE_SECONDS = 24 * 60 * 60


def _seed(store: Store) -> None:
    store.upsert_entities(
        [
            SessionRow(
                session_id="s0",
                root_session_id="s0",
                parent_session_id=None,
                kind="session",
                agent_id=None,
                agent_type=None,
                spawned_by_tool_use_id=None,
                cwd="/x",
                git_branch="main",
                first_ts=_TS,
                last_ts=_TS,
                jsonl_path="/tmp/s0.jsonl",
            ),
            ObservationRow(
                obs_id="o0",
                session_id="s0",
                root_session_id="s0",
                parent_obs_id=None,
                type="user",
                ts=_TS,
                model=None,
                input_tokens=None,
                output_tokens=None,
                tool_name=None,
                tool_use_id=None,
                body="quadratic voting note",
            ),
        ]
    )


@pytest.fixture
def ahead_db(tmp_path):
    """A cache one schema version ahead of this build, with rows in it."""
    db = tmp_path / "cache.db"
    store = Store(db_path=db)
    store.migrate()
    _seed(store)
    store.close()

    c = sqlite3.connect(db)
    c.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1};")
    c.commit()
    c.close()
    return db


def _recording_notifier(tmp_path: Path, *, exit_code: int = 0):
    """A stand-in for ``notify-send``, recording every invocation's argv.

    A real executable rather than a monkeypatched ``subprocess.run``, because
    the property under test is "the notification program exited 0" and a fake
    that cannot exit non-zero cannot express the case that decides the exit
    code. Appends, so a test can prove a SECOND call did not happen.

    Copied in shape from ``tests/test_cli_notifier.py``; kept local because
    that one truncates on each call and this one must not.
    """
    log = tmp_path / "notify.argv"
    script = tmp_path / "fake-notify-send"
    script.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' '--- call ---' \"$@\" >> " + shlex.quote(str(log)) + "\n"
        f"exit {exit_code}\n"
    )
    script.chmod(0o755)
    return script, log


def _calls(log: Path) -> list[list[str]]:
    if not log.exists():
        return []
    calls: list[list[str]] = []
    for line in log.read_text().splitlines():
        if line == "--- call ---":
            calls.append([])
        else:
            calls[-1].append(line)
    return calls


def _stamp_path() -> Path:
    """Where the debounce receipt lives.

    Resolved the way the production code must resolve it, from
    ``$XDG_STATE_HOME`` (which ``conftest`` points at a tmp dir for every
    test), so a test can never read or arm the developer's real stamp.
    """
    base = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
    return Path(base) / "aggregator" / "schema-ahead-notified"


def _run(db) -> int:
    """Any subcommand at all. ``main`` migrates before it dispatches, so the
    refusal happens upstream of everything ``status`` would otherwise do."""
    return main(["status"], _store=Store(db_path=db))


# --- the alarm fires, once, and is critical ---------------------------------


def test_the_refusal_reaches_a_human_as_a_critical_notification(
    ahead_db, tmp_path, monkeypatch
):
    """THE REPRO for the alarm half. Pre-fix nothing was raised at all, so
    nothing was notified; with the raise but no handler, the toast comes from
    the unit's undebounced ``OnFailure=`` instead and repeats forever."""
    script, log = _recording_notifier(tmp_path)
    monkeypatch.setenv(NOTIFY_COMMAND_ENV_VAR, str(script))

    _run(ahead_db)

    calls = _calls(log)
    assert len(calls) == 1, calls
    assert "critical" in calls[0], calls[0]
    assert any(str(SCHEMA_VERSION + 1) in a for a in calls[0]), calls[0]
    assert any(str(SCHEMA_VERSION) in a for a in calls[0]), calls[0]


def test_a_delivered_alarm_exits_zero_so_onfailure_does_not_toast_again(
    ahead_db, tmp_path, monkeypatch
):
    """The receipt IS the reason not to fail the unit.

    ``aggregator-ingest.service`` fires ``OnFailure=`` on any non-zero exit,
    and that notifier has no debounce. Exiting non-zero after a toast has
    already been accepted buys a duplicate toast now and forty-eight a day
    thereafter, for a condition no amount of retrying can change.
    """
    script, log = _recording_notifier(tmp_path)
    monkeypatch.setenv(NOTIFY_COMMAND_ENV_VAR, str(script))

    assert _run(ahead_db) == 0
    # Both halves, or this test passes vacuously against the pre-fix source,
    # where the run also returned 0 — by migrating the cache downward and
    # doing the incident. Exit 0 is only the right answer once a toast has
    # actually gone out, so the toast is asserted here alongside the code.
    assert len(_calls(log)) == 1, _calls(log)


def test_a_second_run_inside_the_window_notifies_nobody_again(
    ahead_db, tmp_path, monkeypatch
):
    """THE SPAM CONTAINMENT, asserted on the notifier's own log.

    The ingest timer ticks every thirty minutes and this condition persists
    until a human deploys a new build, so the second tick and the ninety-sixth
    must be as quiet on the desktop as they are loud in the journal.
    """
    script, log = _recording_notifier(tmp_path)
    monkeypatch.setenv(NOTIFY_COMMAND_ENV_VAR, str(script))

    assert _run(ahead_db) == 0
    assert _run(ahead_db) == 0
    assert _run(ahead_db) == 0

    assert len(_calls(log)) == 1, _calls(log)


def test_the_journal_still_gets_the_full_diagnosis_on_every_run(
    ahead_db, tmp_path, monkeypatch, capsys
):
    """Debounce is a property of the DESKTOP channel only.

    stderr is free, nothing suppresses it, and it is the one place a person
    investigating three days later can still find the fault. A run that went
    quiet everywhere because a toast was sent on Tuesday would be the incident
    all over again.
    """
    script, _log = _recording_notifier(tmp_path)
    monkeypatch.setenv(NOTIFY_COMMAND_ENV_VAR, str(script))

    _run(ahead_db)
    capsys.readouterr()
    _run(ahead_db)

    err = capsys.readouterr().err
    assert str(SCHEMA_VERSION + 1) in err, err
    assert str(SCHEMA_VERSION) in err, err


def test_the_window_expires_and_the_alarm_speaks_again(
    ahead_db, tmp_path, monkeypatch
):
    """Debounce, not permanent silence.

    A stamp that never expires converts "told once" into "never told again",
    and the operator who dismissed one toast in August has no channel left in
    October. Aged by rewriting the stamp's mtime, which is the same thing the
    passage of a day does.
    """
    script, log = _recording_notifier(tmp_path)
    monkeypatch.setenv(NOTIFY_COMMAND_ENV_VAR, str(script))

    assert _run(ahead_db) == 0
    stamp = _stamp_path()
    assert stamp.exists(), "the receipt was never armed"
    old = time.time() - _DEBOUNCE_SECONDS - 60
    os.utime(stamp, (old, old))

    assert _run(ahead_db) == 0
    assert len(_calls(log)) == 2, _calls(log)


# --- and when nobody can be told, the unit has to fail ----------------------


def test_a_notifier_that_exits_nonzero_arms_no_receipt_and_fails_the_run(
    ahead_db, tmp_path, monkeypatch
):
    """FAIL-CLOSED ON THE REPORTING PATH.

    A swallowed notification is indistinguishable from a delivered one except
    by the exit code, which is exactly why ``mkFailureNotify`` arms its stamp
    inside the success branch and nowhere else. Arming it here would spend the
    day's one toast on a message nobody received, and exiting 0 would spend
    the ``OnFailure=`` fallback on top of it.
    """
    script, _log = _recording_notifier(tmp_path, exit_code=1)
    monkeypatch.setenv(NOTIFY_COMMAND_ENV_VAR, str(script))

    assert _run(ahead_db) != 0
    assert not _stamp_path().exists()


def test_an_unresolvable_notifier_fails_the_run_rather_than_tracebacking(
    ahead_db, monkeypatch
):
    """``notify-sned`` is a real class of configuration fault, and the alarm
    path must report it as a failure to notify rather than dying inside the
    handler that exists to keep the process from dying."""
    monkeypatch.setenv(NOTIFY_COMMAND_ENV_VAR, "definitely-not-a-real-program")

    assert _run(ahead_db) != 0
    assert not _stamp_path().exists()


def test_with_no_notifier_configured_at_all_the_run_fails(ahead_db, monkeypatch):
    """The default is the loud one.

    No desktop channel means nobody was told, and a process that returns 0
    after refusing to do its work would report success to every script and
    every unit that reads its exit code. ``_resolve_notify``'s rule applies:
    the notifier exists when the operator said so, and here nobody did.
    """
    monkeypatch.delenv(NOTIFY_COMMAND_ENV_VAR, raising=False)

    assert _run(ahead_db) != 0


# --- the cache, throughout, is untouched ------------------------------------


def test_no_run_of_any_kind_lowers_the_stamp(ahead_db, tmp_path, monkeypatch):
    """The whole point, restated at the CLI boundary.

    Whatever the alarm path decides about notifications and exit codes, the
    bytes on disk are the thing the incident was about, and they are read back
    here from a connection the CLI never held.
    """
    script, _log = _recording_notifier(tmp_path)
    monkeypatch.setenv(NOTIFY_COMMAND_ENV_VAR, str(script))

    _run(ahead_db)
    _run(ahead_db)

    c = sqlite3.connect(ahead_db)
    try:
        assert int(c.execute("PRAGMA user_version").fetchone()[0]) == (
            SCHEMA_VERSION + 1
        )
        assert c.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 1
    finally:
        c.close()


def test_the_receipt_is_its_own_and_not_the_embed_units(
    ahead_db, tmp_path, monkeypatch
):
    """A SHARED STAMP IS A BUG, and one this repo has already reproduced —
    ``nix/aggregator.nix`` documents running the embed and embed-seed notifiers
    against one state dir in both orders and watching each silence the other.
    Two faults that a human must act on separately need two receipts."""
    script, _log = _recording_notifier(tmp_path)
    monkeypatch.setenv(NOTIFY_COMMAND_ENV_VAR, str(script))

    _run(ahead_db)

    stamp = _stamp_path()
    assert stamp.exists(), "the receipt was never armed"
    assert stamp.name != "embed-failure-notified"
    assert stamp.name != "embed-seed-failure-notified"


def test_a_current_cache_runs_the_command_and_arms_nothing(tmp_path, monkeypatch):
    """The control. The alarm path must be unreachable on a healthy cache —
    no toast, no receipt, and the ordinary exit code for the subcommand."""
    script, log = _recording_notifier(tmp_path)
    monkeypatch.setenv(NOTIFY_COMMAND_ENV_VAR, str(script))

    db = tmp_path / "cache.db"
    store = Store(db_path=db)
    store.migrate()
    _seed(store)
    store.close()

    assert main(["status"], _store=Store(db_path=db)) == 0
    assert _calls(log) == []
    assert not _stamp_path().exists()
