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

import shlex
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

from aggregator.cli import NOTIFY_COMMAND_ENV_VAR, main
from aggregator.core.store import Store
from aggregator.imports.port import ImportItem


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
