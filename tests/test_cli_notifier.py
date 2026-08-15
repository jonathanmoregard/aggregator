"""Round-2 MEDIUM-1: nothing could install a REAL notifier.

Round 1 made the notify hook fire on every run, including clean-but-stale
ones. That fix was a no-op in production: the console entry point is
``aggregator.cli:main``, ``_notify`` is a Python-only injection seam, and no
argv or env could reach it — so every real invocation got
``_silent_notification``. On a systemd timer the staleness warning was stderr
text on an exit-0 run, which nothing reads. The silence moved; it did not go.

The spelling chosen here: ``--notify`` on ``ingest --all``, plus
``AGGREGATOR_NOTIFY_COMMAND`` (which installs the notifier by itself, so a
unit file needs no argv change). Library code still never shells out — this
lives in the CLI layer and remains an injected callable.
"""
from __future__ import annotations

import json
import shlex
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

from aggregator import cli
from aggregator.cli import NOTIFY_COMMAND_ENV_VAR, main
from aggregator.core.store import Store
from aggregator.imports.port import ImportItem
from aggregator.imports.runner import AdapterReport, RunReport


class _FakeAdapter:
    def __init__(self, name: str, *, raises: Exception | None = None) -> None:
        self.name = name
        self._raises = raises

    async def get_data(self) -> AsyncIterator[ImportItem]:
        if self._raises is not None:
            raise self._raises
        return
        yield  # pragma: no cover - makes this an async generator


class _StaleAdapter(_FakeAdapter):
    def __init__(self, name: str, age_days: int) -> None:
        super().__init__(name)
        self._freshness = datetime.now(UTC) - timedelta(days=age_days)

    def input_freshness(self) -> datetime | None:
        return self._freshness


def _store(tmp_path) -> Store:
    store = Store(db_path=tmp_path / "cache.db")
    store.migrate()
    return store


def _recording_notifier(tmp_path: Path) -> Path:
    """A stand-in for ``notify-send`` that writes its argv to a file."""
    log = tmp_path / "notify.argv"
    script = tmp_path / "fake-notify-send"
    script.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$@\" > " + shlex.quote(str(log)) + "\n"
    )
    script.chmod(0o755)
    return script, log


def test_a_failing_run_reaches_a_real_notifier_from_env(
    tmp_path, monkeypatch
):
    """THE finding. No ``_notify=`` here — this is what a timer invocation
    looks like, and before the fix nothing at all was told."""
    script, log = _recording_notifier(tmp_path)
    monkeypatch.setenv(NOTIFY_COMMAND_ENV_VAR, str(script))

    rc = main(
        ["ingest", "--all"],
        _store=_store(tmp_path),
        _adapters=[_FakeAdapter("ticktick", raises=RuntimeError("token expired"))],
    )

    assert rc == 3
    assert log.exists(), "a failing unattended run notified nobody"
    argv = log.read_text().splitlines()
    assert "critical" in argv, "the 2026-08-08 constraint says CRITICAL on failure"
    assert any("ticktick" in a for a in argv)
    assert any("token expired" in a for a in argv)


def test_the_shipped_notifier_declares_the_delivery_it_made(tmp_path, monkeypatch):
    """Round 6, the production half. A report barrier only fires when the hook
    declares delivery, so the one notifier that ever really reaches a human has
    to say so — otherwise the fix trades permanent silence for an alarm that
    repeats forever on the configuration that works.

    Round 7: it declares WHICH LINES, read out of the body it sent. Three
    answers from one run each — the line that went out, the line that did not
    fit in the toast, and the healthy run with nothing to send.
    """
    script, _log = _recording_notifier(tmp_path)
    monkeypatch.setenv(NOTIFY_COMMAND_ENV_VAR, str(script))
    noisy = RunReport(
        adapters={
            "ticktick": AdapterReport(
                "ticktick", errors=[f"e{n}" for n in range(cli.NOTIFY_ERROR_LIMIT + 1)]
            )
        }
    )

    delivered = cli._desktop_notification(noisy)

    assert delivered.covers(["ticktick: e0"]), "the toast did carry the first line"
    assert not delivered.covers(noisy.errors), (
        "one line did not fit, so the whole report was not delivered"
    )
    assert not cli._desktop_notification(RunReport()).covers(["anything"])


def test_a_clean_but_stale_run_reaches_the_notifier_too(tmp_path, monkeypatch):
    """The run round 1's fix exists for: exit 0, every count 0, and the only
    thing that distinguishes it from a healthy no-op is the warning."""
    script, log = _recording_notifier(tmp_path)
    monkeypatch.setenv(NOTIFY_COMMAND_ENV_VAR, str(script))

    rc = main(
        ["ingest", "--all"],
        _store=_store(tmp_path),
        _adapters=[_StaleAdapter("substack", 31)],
    )

    assert rc == 0
    assert log.exists()
    argv = log.read_text().splitlines()
    assert "normal" in argv, "staleness is not a failure"
    assert any("31 days" in a for a in argv)


def test_a_healthy_run_pops_no_toast(tmp_path, monkeypatch):
    """A notification on every timer tick trains the operator to ignore them."""
    script, log = _recording_notifier(tmp_path)
    monkeypatch.setenv(NOTIFY_COMMAND_ENV_VAR, str(script))

    rc = main(
        ["ingest", "--all"],
        _store=_store(tmp_path),
        _adapters=[_FakeAdapter("github")],
    )

    assert rc == 0
    assert not log.exists()


def test_the_flag_installs_the_notifier_without_any_env(tmp_path, monkeypatch):
    """``--notify`` alone must work — the default program is ``notify-send``."""
    monkeypatch.delenv(NOTIFY_COMMAND_ENV_VAR, raising=False)
    seen: list[list[str]] = []

    def fake_run(argv, **kw):
        seen.append(argv)

    monkeypatch.setattr("aggregator.cli.subprocess.run", fake_run)
    # The notifier is now checked for resolvability before it is used, so a
    # machine without notify-send would otherwise fail this on its config
    # rather than on the thing under test.
    monkeypatch.setattr("aggregator.cli.shutil.which", lambda p: f"/usr/bin/{p}")

    rc = main(
        ["ingest", "--all", "--notify"],
        _store=_store(tmp_path),
        _adapters=[_FakeAdapter("dropbox", raises=RuntimeError("dir missing"))],
    )

    assert rc == 3
    assert seen and seen[0][0] == "notify-send"
    assert "critical" in seen[0]


def test_an_injected_notifier_still_wins(tmp_path, monkeypatch):
    """The Python seam stays authoritative, so tests and library callers are
    unaffected by an env var set on the developer's machine."""
    script, log = _recording_notifier(tmp_path)
    monkeypatch.setenv(NOTIFY_COMMAND_ENV_VAR, str(script))
    seen = []

    main(
        ["ingest", "--all"],
        _store=_store(tmp_path),
        _adapters=[_FakeAdapter("boom", raises=RuntimeError("x"))],
        _notify=seen.append,
    )

    assert len(seen) == 1
    assert not log.exists()


def test_notify_without_all_is_a_usage_error(tmp_path, capsys):
    """Same rule as --force and --stale-after-days: a flag that parses and
    then does nothing is worse than one that stops."""
    rc = main(
        ["ingest", "alpha", "--notify"], _store=_store(tmp_path), _sources={}
    )

    assert rc == 2
    err = capsys.readouterr().err
    assert "--notify" in err
    assert "--all" in err


def test_a_notifier_that_cannot_run_is_itself_reported(tmp_path, monkeypatch):
    """A broken notifier on an otherwise-clean run is the one fault nothing
    else would ever surface."""
    monkeypatch.setenv(
        NOTIFY_COMMAND_ENV_VAR, str(tmp_path / "does-not-exist")
    )

    rc = main(
        ["ingest", "--all"],
        _store=_store(tmp_path),
        _adapters=[_StaleAdapter("substack", 31)],
    )

    assert rc == 3


# -- round 3 N1: the empty value is the broken one ------------------------


def test_an_empty_notify_command_is_loud(tmp_path, monkeypatch, capsys):
    """THE finding. ``Environment=AGGREGATOR_NOTIFY_COMMAND=`` in a unit file
    produces exactly this, and it was falsy at BOTH gates: no notifier was
    installed and nothing said so. An operator who wrote that line believes
    notifications are on; they are off, undetectably."""
    monkeypatch.setenv(NOTIFY_COMMAND_ENV_VAR, "")

    rc = main(
        ["ingest", "--all"],
        _store=_store(tmp_path),
        _adapters=[_FakeAdapter("github")],
    )

    assert rc == 3, "a clean run must still surface the broken config"
    err = capsys.readouterr().err
    assert NOTIFY_COMMAND_ENV_VAR in err


def test_empty_and_whitespace_only_are_treated_the_same(tmp_path, monkeypatch):
    """They are one keystroke and zero intent apart. Whitespace-only used to
    raise "set but empty" while empty said nothing at all — backwards."""
    codes = []
    for value in ("", "   ", "\t"):
        monkeypatch.setenv(NOTIFY_COMMAND_ENV_VAR, value)
        target = tmp_path / f"run{len(codes)}"
        target.mkdir()
        codes.append(
            main(
                ["ingest", "--all"],
                _store=_store(target),
                _adapters=[_FakeAdapter("github")],
            )
        )

    assert codes == [3, 3, 3]


def test_an_unset_variable_is_still_silence_not_a_fault(tmp_path, monkeypatch):
    """The other side: not asking for notifications must stay free. Only a
    PRESENT variable is a statement of intent."""
    monkeypatch.delenv(NOTIFY_COMMAND_ENV_VAR, raising=False)

    rc = main(
        ["ingest", "--all"],
        _store=_store(tmp_path),
        _adapters=[_FakeAdapter("github")],
    )

    assert rc == 0


# -- round 3 N2: summary and body are positional and not ours -------------


def _getopt_notifier(tmp_path: Path):
    """A stand-in that parses its argv the way notify-send and dunstify do."""
    log = tmp_path / "notify.json"
    script = tmp_path / "fake-notify-send"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import getopt, json, sys\n"
        "opts, rest = getopt.gnu_getopt(sys.argv[1:], 'u:a:')\n"
        f"json.dump({{'opts': opts, 'rest': rest}}, open({str(log)!r}, 'w'))\n"
    )
    script.chmod(0o755)
    return script, log


def test_a_body_line_starting_with_a_dash_is_not_parsed_as_an_option(
    tmp_path, monkeypatch
):
    """THE finding. Adapter names are not validated against a leading dash, so
    the body is attacker-of-convenience data as far as GOption is concerned.
    Without ``--`` the notification is lost to "option -x not recognized" —
    measured — which is the one message that had to get through."""
    script, log = _getopt_notifier(tmp_path)
    monkeypatch.setenv(NOTIFY_COMMAND_ENV_VAR, str(script))

    main(
        ["ingest", "--all"],
        _store=_store(tmp_path),
        _adapters=[_FakeAdapter("-x", raises=RuntimeError("token expired"))],
    )

    assert log.exists(), "the notifier died parsing the body as an option"
    payload = json.loads(log.read_text())
    assert any("token expired" in r for r in payload["rest"])
    assert ("-u", "critical") in [tuple(o) for o in payload["opts"]], (
        "the real options must still be parsed as options"
    )


# -- round 3 N3: a typo must not wait for a failing run -------------------


def test_a_typoed_program_is_caught_on_a_clean_run(tmp_path, monkeypatch):
    """THE finding. The notify command was only ever exercised when there was
    something to send, so a typo stayed latent until the first FAILING run —
    and that run's notify failure then reached only the journal, the channel
    the notifier exists to replace."""
    monkeypatch.setenv(NOTIFY_COMMAND_ENV_VAR, str(tmp_path / "notify-sned"))

    rc = main(
        ["ingest", "--all"],
        _store=_store(tmp_path),
        _adapters=[_FakeAdapter("github")],
    )

    assert rc == 3


def test_a_resolvable_program_stays_quiet_on_a_clean_run(tmp_path, monkeypatch):
    """The validation must not become the toast-on-every-tick it replaced."""
    script, log = _getopt_notifier(tmp_path)
    monkeypatch.setenv(NOTIFY_COMMAND_ENV_VAR, str(script))

    rc = main(
        ["ingest", "--all"],
        _store=_store(tmp_path),
        _adapters=[_FakeAdapter("github")],
    )

    assert rc == 0
    assert not log.exists()
