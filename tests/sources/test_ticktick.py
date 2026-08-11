"""Tests for the merged TickTick source (CSV backup + Open API poll).

The two legs are tested separately in ``test_ticktick_csv.py`` and
``test_ticktick_api.py``; what is tested here is only what the merge adds —
which leg wins for a given task, and what survives when one leg is broken.

The load-bearing invariant, asserted from several angles below: a broken API
credential degrades the run to CSV-only, never to a dead ingest. The API leg
is optional (many runs have no token at all) while the CSV backup carries all
1302 rows of history, so an exception escaping the API leg would trade the
whole archive for a credential the user may never have configured.
"""
from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta
from urllib.error import HTTPError

import pytest

from aggregator.sources import ticktick, ticktick_api
from aggregator.sources.ticktick import TickTickSource
from tests.sources.test_ticktick_csv import HEADER, _row


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """No test in this file may reach the real TickTick API.

    Same reasoning as the identical fixture in ``test_ticktick_api.py``: the
    token is write-scoped and the endpoint is the user's live task list.
    ``_open`` is the API module's single network seam.
    """

    def _forbidden(*args, **kwargs):
        raise AssertionError("test attempted a real network call")

    monkeypatch.setattr(ticktick_api, "_open", _forbidden)


@pytest.fixture(autouse=True)
def _no_real_credentials(monkeypatch, tmp_path):
    """No test in this file may read the developer's own TickTick token.

    ``resolve_token`` falls back to ``$TICKTICK_ACCESS_TOKEN`` and then to the
    shared ``~/.config/todo/env`` store, which on this machine holds a live
    token. Without this, every "no token configured" test below would pass
    *because* a real token was found, and the suite's result would depend on
    whose machine it ran on.
    """
    monkeypatch.setattr(ticktick_api, "DEFAULT_ENV_FILE", str(tmp_path / "no-such-env"))
    monkeypatch.delenv("TICKTICK_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("TICKTICK_TOKEN_EXPIRES_AT", raising=False)
    # The source's own knobs, cleared for the same reason: a developer who
    # exports one would otherwise turn every "CSV-only" test below green (or
    # red) for reasons that have nothing to do with the code under test.
    monkeypatch.delenv("AGGREGATOR_TICKTICK_TOKEN", raising=False)
    monkeypatch.delenv("AGGREGATOR_TICKTICK_TOKEN_FILE", raising=False)
    monkeypatch.delenv("AGGREGATOR_TICKTICK_DIR", raising=False)


def _backup(path, rows):
    preamble = "\n".join(f'"Date: line {i}"' for i in range(6))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([preamble, HEADER, *rows]) + "\n", encoding="utf-8")
    return path


def _poll(tasks, complete=True):
    """An ``OpenTaskPoll``, the shape ``poll_open_tasks`` now returns.

    Completeness travels with the tasks so no caller can run the completion
    inference over a partial view — see ``ticktick_api.OpenTaskPoll``.
    """
    return ticktick_api.OpenTaskPoll(list(tasks), complete=complete)


def _source(tmp_path, **kw):
    return TickTickSource(
        backup_dir=tmp_path / "downloads",
        archive_dir=tmp_path / "archive",
        state_file=tmp_path / "state.json",
        **kw,
    )


def test_csv_only_when_no_token(tmp_path):
    _backup(tmp_path / "downloads" / "TickTick.csv", [_row()])
    records = list(_source(tmp_path).iter_records(None))
    assert [r.stable_id for r in records] == ["ticktick:abc123"]
    assert records[0].extra["provenance"] == "csv"


def test_non_ticktick_csv_ignored(tmp_path):
    """The scan is over ~/Downloads, which is full of other people's CSVs."""
    downloads = tmp_path / "downloads"
    downloads.mkdir(parents=True)
    (downloads / "bank.csv").write_text("date,amount\n2026-01-01,42\n", encoding="utf-8")
    assert list(_source(tmp_path).iter_records(None)) == []


def test_backup_is_archived(tmp_path):
    """The backup is a manual export a human drops in ~/Downloads and then
    deletes. Copying it out is what keeps the deep history rebuildable."""
    _backup(tmp_path / "downloads" / "TickTick.csv", [_row()])
    list(_source(tmp_path).iter_records(None))
    assert (tmp_path / "archive" / "TickTick.csv").exists()


def test_archived_backup_still_ingested_after_download_cleared(tmp_path):
    """Archiving is pointless if the archive is not also a scan root."""
    _backup(tmp_path / "downloads" / "TickTick.csv", [_row()])
    list(_source(tmp_path).iter_records(None))
    (tmp_path / "downloads" / "TickTick.csv").unlink()
    records = list(_source(tmp_path).iter_records(None))
    assert [r.stable_id for r in records] == ["ticktick:abc123"]


def test_since_skips_old_backups(tmp_path):
    """An incremental run must not re-parse a backup it already ingested."""
    path = _backup(tmp_path / "downloads" / "TickTick.csv", [_row()])
    stale = (datetime.now(UTC) - timedelta(days=30)).timestamp()
    os.utime(path, (stale, stale))
    since = datetime.now(UTC) - timedelta(days=1)
    assert list(_source(tmp_path).iter_records(since)) == []


def test_naive_since_is_treated_as_utc(tmp_path):
    """``aggregator ingest ticktick --since 2026-08-01`` parses to a NAIVE
    datetime. Comparing one against an aware mtime raises TypeError, which
    would take the whole ingest down over a flag the CLI documents."""
    path = _backup(tmp_path / "downloads" / "TickTick.csv", [_row()])
    stale = (datetime.now(UTC) - timedelta(days=30)).timestamp()
    os.utime(path, (stale, stale))
    naive = (datetime.now(UTC) - timedelta(days=1)).replace(tzinfo=None)
    assert list(_source(tmp_path).iter_records(naive)) == []


def test_api_leg_merges_with_csv(tmp_path, monkeypatch):
    """Disjoint task sets union; neither leg replaces the other wholesale."""
    _backup(tmp_path / "downloads" / "TickTick.csv", [_row(task_id="csv1", title="From CSV")])
    monkeypatch.setattr(
        ticktick_api,
        "poll_open_tasks",
        lambda token, errors=None: _poll(
            [{"id": "api1", "title": "From API", "_projectName": "Work"}]
        ),
    )
    records = {r.stable_id: r for r in _source(tmp_path, token="tok").iter_records(None)}
    assert set(records) == {"ticktick:csv1", "ticktick:api1"}
    assert records["ticktick:api1"].extra["provenance"] == "api"


def test_fresher_api_observation_beats_stale_csv(tmp_path, monkeypatch):
    """Un-completing a task is a real thing a human does in TickTick.

    Last month's backup says completed; today's poll serves it, which the API
    only does for OPEN tasks. The fresher observation has to win or the index
    keeps insisting a task the user is working on is finished.
    """
    path = _backup(tmp_path / "downloads" / "TickTick.csv", [_row(task_id="t1", status="2")])
    stale = (datetime.now(UTC) - timedelta(days=30)).timestamp()
    os.utime(path, (stale, stale))
    monkeypatch.setattr(
        ticktick_api,
        "poll_open_tasks",
        lambda token, errors=None: _poll([{"id": "t1", "title": "Reopened"}]),
    )
    (rec,) = list(_source(tmp_path, token="tok").iter_records(None))
    assert rec.extra["provenance"] == "api"
    assert "open" in rec.tags


def test_fresher_csv_beats_api(tmp_path, monkeypatch):
    """The other direction of the same rule, and the reason the CSV leg is
    authoritative for completions: a backup newer than the poll it is merged
    with carries the real Completed Time, which the API can never report."""
    _backup(
        tmp_path / "downloads" / "TickTick.csv",
        [_row(task_id="t1", status="2", completed="2026-08-03T17:30:00+0000")],
    )
    monkeypatch.setattr(
        ticktick_api,
        "poll_open_tasks",
        lambda token, errors=None: _poll([{"id": "t1", "title": "Stale open view"}]),
    )
    src = _source(tmp_path, token="tok")
    src._api_observed_at = datetime(2000, 1, 1, tzinfo=UTC)
    (rec,) = list(src.iter_records(None))
    assert rec.extra["provenance"] == "csv"
    assert "completed" in rec.tags


def test_completion_inferred_across_two_polls(tmp_path, monkeypatch):
    """The Open API cannot report a completion, so disappearance is the signal.

    This only works if the poll is diffed against the baseline BEFORE the
    baseline is overwritten with the poll — ``plan_open_task_reconcile``
    sequences load -> diff -> save for exactly that reason.

    ``commit_after_write`` stands in for the writing caller: since round 2 the
    baseline is advanced by the CLI / runner AFTER the records land, not by
    iteration, so a poll nobody stored is re-offered next time.
    """
    tasks = [{"id": "t1", "title": "Gone"}, {"id": "t2", "title": "Stays"}]
    monkeypatch.setattr(
        ticktick_api, "poll_open_tasks", lambda token, errors=None: _poll(tasks)
    )
    first = _source(tmp_path, token="tok")
    list(first.iter_records(None))
    first.commit_after_write()

    tasks = [{"id": "t2", "title": "Stays"}]
    records = {r.stable_id: r for r in _source(tmp_path, token="tok").iter_records(None)}
    assert records["ticktick:t1"].extra["provenance"] == "api-inferred-complete"
    # task_to_record writes the STRING "true", not the bool True: extra is
    # json.dumps'd, and test_extra_values_are_all_strings_on_an_inferred_completion
    # (tests/sources/test_ticktick_api.py) asserts every value in extra is a str.
    # A bool here would serialise to the JSON literal `true`, which a DSL
    # exact-match filter can't match against text — do not "fix" this back.
    assert records["ticktick:t1"].extra["completed_time_approx"] == "true"
    assert records["ticktick:t2"].extra["provenance"] == "api"


def test_an_unchanged_task_is_not_inferred_completed_every_poll(tmp_path, monkeypatch):
    """The baseline keys ids through ``_task_id``, which STRIPS.

    Diffing a hand-rolled ``str(task["id"])`` against those keys means a padded
    id never matches its own baseline entry: the task reads as disappeared on
    every single poll and is recorded as completed forever, with no error
    anywhere. Two identical polls must produce no completion at all.
    """
    tasks = [{"id": " t1 ", "title": "Padded"}]
    monkeypatch.setattr(
        ticktick_api, "poll_open_tasks", lambda token, errors=None: _poll(tasks)
    )
    list(_source(tmp_path, token="tok").iter_records(None))

    records = list(_source(tmp_path, token="tok").iter_records(None))
    assert [r.extra["provenance"] for r in records] == ["api"]


def test_padded_api_id_does_not_duplicate_the_csv_record(tmp_path, monkeypatch):
    """Same task, two legs, one record — even when the API pads the id.

    The merge key has to be the id as the RECORD minted it (``_task_id``
    strips), not the raw payload value. Keyed raw, a padded id lands under a
    key the CSV leg's clean ``taskId`` never matches and the same task is
    emitted twice: two rows, one stable_id, and whichever the store writes
    last wins at random.
    """
    _backup(tmp_path / "downloads" / "TickTick.csv", [_row(task_id="t1", status="2")])
    monkeypatch.setattr(
        ticktick_api,
        "poll_open_tasks",
        lambda token, errors=None: _poll([{"id": " t1 ", "title": "Padded"}]),
    )
    records = list(_source(tmp_path, token="tok").iter_records(None))
    assert [r.stable_id for r in records] == ["ticktick:t1"]


def test_api_failure_records_error_and_keeps_csv(tmp_path, monkeypatch):
    """TickTick being down costs us the poll, not the 1302-row archive."""
    _backup(tmp_path / "downloads" / "TickTick.csv", [_row()])

    def boom(token, errors=None):
        raise OSError("network down")

    monkeypatch.setattr(ticktick_api, "poll_open_tasks", boom)
    errors: list[str] = []
    records = list(_source(tmp_path, token="tok").iter_records(None, errors=errors))
    assert [r.stable_id for r in records] == ["ticktick:abc123"]
    assert len(errors) == 1
    assert "network down" in errors[0]


def test_token_file_is_read(tmp_path, monkeypatch):
    token_file = tmp_path / "token"
    token_file.write_text("filetoken\n", encoding="utf-8")
    seen = {}

    def fake_fetch(token, errors=None):
        seen["token"] = token
        return _poll([])

    monkeypatch.setattr(ticktick_api, "poll_open_tasks", fake_fetch)
    list(_source(tmp_path, token_file=str(token_file)).iter_records(None))
    assert seen["token"] == "filetoken"


def test_shared_token_store_is_the_fallback(tmp_path, monkeypatch):
    """No token configured here means "read the one that already has an owner".

    ``~/.claude/todo/backends/ticktick.py`` rewrites ``~/.config/todo/env`` on
    every OAuth refresh, so that file is where the live token is by definition
    (2026-08-11 constraint: reuse an existing integration's credential, never
    copy it). A second copy would go stale the moment the first refreshed.
    """
    env_file = tmp_path / "env"
    env_file.write_text('TICKTICK_ACCESS_TOKEN="shared-token"\n', encoding="utf-8")
    monkeypatch.setattr(ticktick_api, "DEFAULT_ENV_FILE", str(env_file))
    seen = {}

    def fake_fetch(token, errors=None):
        seen["token"] = token
        return _poll([])

    monkeypatch.setattr(ticktick_api, "poll_open_tasks", fake_fetch)
    list(_source(tmp_path).iter_records(None))
    assert seen["token"] == "shared-token"


def test_missing_token_file_does_not_take_down_the_csv_leg(tmp_path):
    """A token file the caller names but cannot read must cost the API leg only.

    ``resolve_token`` raises ``TokenUnavailableError`` (an OSError subclass) for
    a configured-but-unreadable token file. Caught inside the API leg's own try
    in ``_api_candidates``, it is recorded and the run degrades to CSV-only —
    the failure never escapes ``iter_records`` and takes the CSV leg's 1302
    backup rows down with it.
    """
    _backup(tmp_path / "downloads" / "TickTick.csv", [_row()])
    errors: list[str] = []
    src = _source(tmp_path, token_file=str(tmp_path / "nope" / "token"))
    records = list(src.iter_records(None, errors=errors))
    assert [r.stable_id for r in records] == ["ticktick:abc123"]
    assert len(errors) == 1
    assert "token" in errors[0]


def test_expired_shared_token_degrades_to_csv_only(tmp_path, monkeypatch):
    """The failure that actually happens in production, and the reason
    ``resolve_token`` is called INSIDE the API leg's try.

    The shared store holds an expiry alongside the token, and an expired one
    raises rather than 401-ing halfway through the project walk. The CSV backup
    needs no token at all, so an expired credential must cost the poll and
    nothing else — and the error has to name the command that fixes it, because
    this leg runs unattended from a timer and cannot re-authorize itself.
    """
    env_file = tmp_path / "env"
    env_file.write_text(
        'TICKTICK_ACCESS_TOKEN="stale-token"\nTICKTICK_TOKEN_EXPIRES_AT="1000000000"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(ticktick_api, "DEFAULT_ENV_FILE", str(env_file))
    _backup(tmp_path / "downloads" / "TickTick.csv", [_row()])
    errors: list[str] = []

    records = list(_source(tmp_path).iter_records(None, errors=errors))

    assert [r.stable_id for r in records] == ["ticktick:abc123"]
    assert len(errors) == 1
    assert ticktick_api.RELOGIN_COMMAND in errors[0]


def test_an_empty_shared_token_is_a_loud_error_not_a_silent_csv_only_run(
    tmp_path, monkeypatch
):
    """Round-1 MEDIUM. A failed OAuth refresh leaves the key present with no
    value, and that read as "the API leg was never configured" — reported at a
    log level nothing prints, on a run that still exited 0. The poll silently
    disabled itself and completed-task inference stopped, permanently."""
    env_file = tmp_path / "env"
    env_file.write_text("TICKTICK_ACCESS_TOKEN=\n", encoding="utf-8")
    monkeypatch.setattr(ticktick_api, "DEFAULT_ENV_FILE", str(env_file))
    _backup(tmp_path / "downloads" / "TickTick.csv", [_row()])
    errors: list[str] = []

    records = list(_source(tmp_path).iter_records(None, errors=errors))

    # Still CSV-only rather than a dead ingest — the load-bearing invariant.
    assert [r.stable_id for r in records] == ["ticktick:abc123"]
    assert len(errors) == 1
    assert "empty" in errors[0]
    assert ticktick_api.RELOGIN_COMMAND in errors[0]


def test_a_genuinely_unconfigured_api_leg_is_not_an_error_but_is_audible(
    tmp_path, monkeypatch, caplog
):
    """The other side of that distinction. Never configuring the API leg is a
    supported setup, so it must not fire a CRITICAL notification every tick —
    but it was logged at INFO, and nothing under ``aggregator/`` configures
    logging, so ``logging.lastResort`` (WARNING) meant it reached nobody."""
    env_file = tmp_path / "env"
    env_file.write_text("# no token here\n", encoding="utf-8")
    monkeypatch.setattr(ticktick_api, "DEFAULT_ENV_FILE", str(env_file))
    _backup(tmp_path / "downloads" / "TickTick.csv", [_row()])
    errors: list[str] = []

    with caplog.at_level(logging.WARNING, logger="aggregator.sources.ticktick"):
        records = list(_source(tmp_path).iter_records(None, errors=errors))

    assert [r.stable_id for r in records] == ["ticktick:abc123"]
    assert errors == []
    assert any(
        "CSV-only" in r.getMessage() and r.levelno >= logging.WARNING
        for r in caplog.records
    ), f"the CSV-only degradation must print, got {caplog.records!r}"


def test_unusable_task_payload_is_recorded_and_skipped(tmp_path, monkeypatch):
    """One surprising task must not cost the other 237, nor the CSV archive.

    ``task_to_record`` raises on a payload with no usable id — deliberately, so
    a null id cannot collapse every such task onto the record ``ticktick:None``.
    Raised from the middle of the merge loop that would abort the whole ingest,
    so it is caught per task, the same rule the API client's project walk
    follows for a malformed payload.
    """
    _backup(tmp_path / "downloads" / "TickTick.csv", [_row()])
    monkeypatch.setattr(
        ticktick_api,
        "poll_open_tasks",
        lambda token, errors=None: _poll(
            [
                {"id": None, "title": "no id"},
                {"id": "ok1", "title": "fine"},
            ]
        ),
    )
    errors: list[str] = []
    records = list(_source(tmp_path, token="tok").iter_records(None, errors=errors))
    assert [r.stable_id for r in records] == ["ticktick:abc123", "ticktick:ok1"]
    assert len(errors) == 1
    assert "id" in errors[0]


def test_unwritable_state_file_does_not_take_down_the_csv_leg(tmp_path, monkeypatch):
    """Same invariant as the token case, one step further into the API leg.

    Persisting the baseline writes to ``$XDG_STATE_HOME``. If that path is not
    a directory the write raises an OSError — which is not a reason to lose the
    archive. The poll's own tasks are still emitted; only the next run's
    completion inference is lost, and the failure says so.

    Round 2 moved WHEN this happens: the save is now the writing caller's
    ``commit_after_write``, so it raises after the records have landed instead
    of aborting the merge. The CLI turns that into an ``errors`` entry and exit
    3; the runner into a report error. Either way it is loud, and — unlike
    before — the poll's inferred completions were written first.
    """
    (tmp_path / "notadir").write_text("i am a file, not a directory", encoding="utf-8")
    _backup(tmp_path / "downloads" / "TickTick.csv", [_row()])
    monkeypatch.setattr(
        ticktick_api,
        "poll_open_tasks",
        lambda token, errors=None: _poll([{"id": "api1", "title": "From API"}]),
    )
    errors: list[str] = []
    src = TickTickSource(
        backup_dir=tmp_path / "downloads",
        archive_dir=tmp_path / "archive",
        state_file=tmp_path / "notadir" / "state.json",
        token="tok",
    )
    records = list(src.iter_records(None, errors=errors))
    assert [r.stable_id for r in records] == ["ticktick:abc123", "ticktick:api1"]
    assert errors == []
    with pytest.raises(OSError, match="state could not be updated"):
        src.commit_after_write()


def test_backup_that_fails_to_parse_is_recorded_and_the_scan_continues(
    tmp_path, monkeypatch
):
    """Detection and parse are two separate reads of a file in ~/Downloads.

    That directory is live — a browser can truncate, replace or remove a file
    between the two — and ``parse_backup`` deliberately does not swallow read
    errors. One unreadable backup must cost that file only, not the archive
    sitting next to it.
    """
    _backup(tmp_path / "downloads" / "TickTick.csv", [_row()])
    _backup(tmp_path / "archive" / "Older.csv", [_row(task_id="old1")])
    real_parse = ticktick.parse_backup

    def flaky(path, errors=None):
        if path.name == "TickTick.csv":
            raise OSError("vanished mid-read")
        return real_parse(path, errors)

    monkeypatch.setattr(ticktick, "parse_backup", flaky)
    errors: list[str] = []
    records = list(_source(tmp_path).iter_records(None, errors=errors))
    assert [r.stable_id for r in records] == ["ticktick:old1"]
    assert len(errors) == 1
    assert "TickTick.csv" in errors[0]


def test_record_shape_documents_every_extra_key_both_legs_write(tmp_path, monkeypatch):
    """``record_shape`` is the DSL's help surface, and the merge means a task
    can arrive from either leg. A key one leg writes and the shape omits is a
    filter a user is never told exists."""
    _backup(tmp_path / "downloads" / "TickTick.csv", [_row(task_id="t1")])
    monkeypatch.setattr(
        ticktick_api,
        "poll_open_tasks",
        lambda token, errors=None: _poll([{"id": "t2", "title": "open"}]),
    )
    src = _source(tmp_path, token="tok")
    list(src.iter_records(None))
    src.commit_after_write()  # the writing caller's barrier; see above
    # Second poll: t2 disappears, so an inferred completion (the third record
    # flavour, and the only one carrying completed_time_approx) shows up too.
    monkeypatch.setattr(
        ticktick_api, "poll_open_tasks", lambda token, errors=None: _poll([])
    )
    records = list(_source(tmp_path, token="tok").iter_records(None))

    written = {key for r in records for key in r.extra}
    assert {"provenance", "source_file", "completed_time_approx"} <= written
    assert written <= set(src.record_shape())


def test_no_arg_construction_resolves_xdg_defaults(tmp_path, monkeypatch):
    """``_default_sources()`` builds this with no arguments, so the defaults
    have to be real paths and construction has to stay side-effect-free — no
    filesystem or network work until an ingest actually runs."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("AGGREGATOR_TICKTICK_DIR", str(tmp_path / "downloads"))

    src = TickTickSource()

    assert src.backup_dir == tmp_path / "downloads"
    assert src.archive_dir == tmp_path / "data" / "aggregator" / "ticktick" / "backups"
    assert src.state_file == ticktick_api.default_state_path()
    # State, not data: regenerable, but regenerating it costs a poll's
    # completions and nothing else can reconstruct them.
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "state").exists()


def test_env_vars_configure_the_api_leg(tmp_path, monkeypatch):
    """The knobs the design doc names, so a systemd unit can point this at a
    credential without a code change."""
    token_file = tmp_path / "token"
    token_file.write_text("from-env-file\n", encoding="utf-8")
    monkeypatch.setenv("AGGREGATOR_TICKTICK_TOKEN_FILE", str(token_file))
    seen = {}

    def fake_fetch(token, errors=None):
        seen["token"] = token
        return _poll([])

    monkeypatch.setattr(ticktick_api, "poll_open_tasks", fake_fetch)
    list(_source(tmp_path).iter_records(None))
    assert seen["token"] == "from-env-file"

    monkeypatch.setenv("AGGREGATOR_TICKTICK_TOKEN", "from-env")
    list(_source(tmp_path).iter_records(None))
    assert seen["token"] == "from-env"


def test_ingest_counts_records_and_carries_the_errors_out(tmp_path, monkeypatch):
    """``Source.ingest`` is the count-only path the CLI falls back to. Errors
    have to ride out on the result — a run that reports added=1 errors=0 while
    the API leg was dead is the silent rot the fail-loudly constraint bans."""
    _backup(tmp_path / "downloads" / "TickTick.csv", [_row()])

    def boom(token, errors=None):
        raise OSError("network down")

    monkeypatch.setattr(ticktick_api, "poll_open_tasks", boom)
    result = _source(tmp_path, token="tok").ingest(None)
    assert (result.added, result.updated, result.skipped) == (1, 0, 0)
    assert len(result.errors) == 1


def test_newest_backup_mtime_reports_a_stale_export(tmp_path):
    """The CSV is a MANUAL export a human drops in ~/Downloads; nothing on this
    machine refreshes it. Reporting its age is what lets a later surface say
    "the ticktick backup is 31 days stale" instead of re-importing the same
    file forever and calling it success.

    Deliberately independent of ``since``: the run that skips a stale backup
    because it is older than the window is precisely the run that must still be
    able to report how stale it is.
    """
    path = _backup(tmp_path / "downloads" / "TickTick.csv", [_row()])
    stale = (datetime.now(UTC) - timedelta(days=31)).timestamp()
    os.utime(path, (stale, stale))
    src = _source(tmp_path)

    assert list(src.iter_records(datetime.now(UTC) - timedelta(days=1))) == []
    newest = src.newest_backup_mtime()
    assert newest is not None
    assert (datetime.now(UTC) - newest).days == 31


def test_newest_backup_mtime_is_none_without_a_backup(tmp_path):
    """Unknown, not "epoch" — a fabricated timestamp would read as fresh."""
    assert _source(tmp_path).newest_backup_mtime() is None


# --- a partial poll must never arm the baseline ---------------------------
#
# Driven through the module's real ``_request`` seam rather than a stubbed
# ``poll_open_tasks``, because the whole defect lived in the join between the
# project walk and the completion inference: a stub that hands over a task list
# is exactly the shape that hid it.


def _two_projects(*, p2_serves):
    """A fake ``_request`` for two projects; ``p2_serves`` may raise instead."""

    def fake_request(method, url, token, timeout=30):
        if url.endswith("/project"):
            return [{"id": "p1", "name": "Work"}, {"id": "p2", "name": "Home"}]
        if url.endswith("/project/p1/data"):
            return {"tasks": [_api_task("t1", "Still open")]}
        return p2_serves()

    return fake_request


def _api_task(task_id, title):
    # Dated, so the "no parseable timestamp anywhere" tripwire stays quiet and
    # the errors these tests count are only the ones they are about.
    return {
        "id": task_id,
        "title": title,
        "status": 0,
        "modifiedTime": "2026-08-09T04:00:00.000+0000",
    }


def _healthy_p2():
    return {"tasks": [_api_task("t2", "Also open")]}


def _dead_p2():
    raise HTTPError(f"{ticktick_api.BASE_URL}/project/p2/data", 500, "boom", {}, None)


def test_a_blip_on_one_project_does_not_complete_that_projects_tasks(
    tmp_path, monkeypatch
):
    """The measured defect. One project 500s; every open task in it disappears
    from the poll, which is precisely the signal the completion inference reads
    as "finished". The Open API serves OPEN tasks only, so nothing it can ever
    return contradicts that — only a manual CSV export could. So an incomplete
    poll infers nothing at all, and leaves the baseline exactly where it was.
    """
    monkeypatch.setattr(ticktick_api, "_request", _two_projects(p2_serves=_healthy_p2))
    first = _source(tmp_path, token="tok")
    list(first.iter_records(None))
    first.commit_after_write()
    assert ticktick_api.load_state(tmp_path / "state.json").keys() == {"t1", "t2"}

    monkeypatch.setattr(ticktick_api, "_request", _two_projects(p2_serves=_dead_p2))
    second = _source(tmp_path, token="tok")
    errors: list[str] = []
    records = list(second.iter_records(None, errors=errors))
    second.commit_after_write()

    assert [r.extra["provenance"] for r in records] == ["api"], (
        "an open task in the project that 500'd was recorded as completed"
    )
    assert ticktick_api.load_state(tmp_path / "state.json").keys() == {"t1", "t2"}, (
        "the partial poll advanced the baseline, so no later poll can undo this"
    )
    # Loud, not merely correct: the failed project AND the consequence.
    assert len(errors) == 2
    assert any("p2" in e for e in errors)
    assert any("SKIPPED" in e for e in errors)


def test_the_next_complete_poll_still_infers_the_real_completion(tmp_path, monkeypatch):
    """Declining to infer costs nothing, which is why declining is safe.

    The baseline was never advanced, so the very next healthy poll diffs against
    the same still-true baseline and picks up the completion that really did
    happen during the outage.
    """
    monkeypatch.setattr(ticktick_api, "_request", _two_projects(p2_serves=_healthy_p2))
    first = _source(tmp_path, token="tok")
    list(first.iter_records(None))
    first.commit_after_write()

    monkeypatch.setattr(ticktick_api, "_request", _two_projects(p2_serves=_dead_p2))
    blipped = _source(tmp_path, token="tok")
    list(blipped.iter_records(None, errors=[]))
    blipped.commit_after_write()

    # p2 is back, and t2 really was completed while it was down.
    monkeypatch.setattr(
        ticktick_api, "_request", _two_projects(p2_serves=lambda: {"tasks": []})
    )
    third = _source(tmp_path, token="tok")
    records = {r.stable_id: r for r in third.iter_records(None, errors=[])}
    third.commit_after_write()

    assert records["ticktick:t2"].extra["provenance"] == "api-inferred-complete"
    assert ticktick_api.load_state(tmp_path / "state.json").keys() == {"t1"}
