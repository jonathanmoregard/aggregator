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


def _poll(tasks, complete=True, covered=("p1",)):
    """An ``OpenTaskPoll``, the shape ``poll_open_tasks`` now returns.

    Completeness AND coverage travel with the tasks so no caller can run the
    completion inference over a partial view, nor over a project this poll
    never enumerated — see ``ticktick_api.OpenTaskPoll``. Every task fixture in
    this file lives in ``p1``, so the default covers exactly them.
    """
    return ticktick_api.OpenTaskPoll(
        list(tasks), complete=complete, covered_project_ids=frozenset(covered)
    )


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
        lambda token, errors=None: _poll([{"id": "t1", "title": "Reopened", "projectId": "p1"}]),
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
        lambda token, errors=None: _poll([{"id": "t1", "title": "Stale open view", "projectId": "p1"}]),
    )
    # An old POLL against a backup exported since. Patching the clock rather
    # than the instance, because the stamp is no longer taken at construction:
    # ``_now`` is read one line before the request goes out. The alternative
    # would be a backup file dated in the future.
    monkeypatch.setattr(ticktick, "_now", lambda: datetime(2000, 1, 1, tzinfo=UTC))
    (rec,) = list(_source(tmp_path, token="tok").iter_records(None))
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
    tasks = [
        {"id": "t1", "title": "Gone", "projectId": "p1"},
        {"id": "t2", "title": "Stays", "projectId": "p1"},
    ]
    monkeypatch.setattr(
        ticktick_api, "poll_open_tasks", lambda token, errors=None: _poll(tasks)
    )
    first = _source(tmp_path, token="tok")
    list(first.iter_records(None))
    first.commit_after_write()

    tasks = [{"id": "t2", "title": "Stays", "projectId": "p1"}]
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
        lambda token, errors=None: _poll([{"id": "t2", "title": "open", "projectId": "p1"}]),
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


def test_vocabulary_drift_in_a_backup_reaches_the_runs_errors_sink(tmp_path):
    """An unrecognised status must not ride out on an exit-0 run.

    End to end, because the sink only matters if it survives the hop from
    ``status_tag`` through ``row_to_record`` to the run report the CLI turns
    into an exit code and a notification.
    """
    _backup(tmp_path / "downloads" / "TickTick.csv", [_row(status="1")])
    errors: list[str] = []
    (rec,) = list(_source(tmp_path).iter_records(None, errors=errors))
    assert "open" not in rec.tags
    assert rec.extra["status"] == "1"
    assert len(errors) == 1
    assert "'1'" in errors[0]


def test_a_status_drift_across_a_whole_backup_stays_bounded(tmp_path):
    """Round-11 LOW 4. One error per ROW starves the notification of ticktick's own line.

    ``status_tag`` appends one entry per unrecognised row and a vendor drift is
    UNIFORM, so across the user's real 1302-row backup this put ~1302 gating
    ticktick lines into ``report.errors``. The notification budget is five lines
    and it spends them on receipt-gating errors first — TickTick is the only
    adapter that gates — so the flood would have crowded out TickTick's own
    uncovered-project line. That line's receipt is only stamped once it reaches
    a human (``commit_after_report``), so the alert would never be earned and
    would repeat, drowned, every 30 minutes. A flood that destroys the alert it
    is raising.
    """
    rows = [_row(task_id=f"t{i}", status="7") for i in range(1302)]
    _backup(tmp_path / "downloads" / "TickTick.csv", rows)
    errors: list[str] = []

    records = list(_source(tmp_path).iter_records(None, errors=errors))

    # Every row is still EMITTED — dropping them would lose tasks from an index
    # whose whole job is to remember them, and nothing regenerates the backup.
    assert len(records) == 1302
    assert all(r.extra["status"] == "7" for r in records)
    # One line, not 1302, and the exact magnitude survives inside it.
    assert len(errors) == 1
    assert "1302 rows share this fault" in errors[0]
    assert "'7'" in errors[0]
    assert "TickTick.csv" in errors[0]


def test_two_drifted_codes_are_two_lines_with_their_own_exact_counts(tmp_path):
    """Distinct faults are distinct diagnoses; only the repeats collapse."""
    rows = [_row(task_id=f"a{i}", status="7") for i in range(40)]
    rows += [_row(task_id=f"b{i}", status="9") for i in range(3)]
    _backup(tmp_path / "downloads" / "TickTick.csv", rows)
    errors: list[str] = []

    list(_source(tmp_path).iter_records(None, errors=errors))

    assert len(errors) == 2
    assert any("40 rows share this fault" in e and "'7'" in e for e in errors)
    assert any("3 rows share this fault" in e and "'9'" in e for e in errors)


def test_many_distinct_drifted_codes_cap_the_kinds_but_not_the_counts(tmp_path):
    """The cap is on the KINDS named. Capping a count would hide the magnitude,
    which is the same trade the sessions and github drop reports refuse."""
    rows = [_row(task_id=f"t{i}", status=str(20 + i)) for i in range(9)]
    _backup(tmp_path / "downloads" / "TickTick.csv", rows)
    errors: list[str] = []

    list(_source(tmp_path).iter_records(None, errors=errors))

    assert len(errors) == 6  # five named kinds + one tail line
    assert "and 4 further DISTINCT row fault(s) not shown" in errors[-1]


def test_a_drifted_status_from_the_api_leg_is_bounded_too(tmp_path, monkeypatch):
    """The two legs share ``status_tag``, so they share the flood."""
    monkeypatch.setattr(
        ticktick_api,
        "poll_open_tasks",
        lambda token, errors=None: _poll(
            [{"id": f"t{i}", "projectId": "p1", "status": 7} for i in range(200)]
        ),
    )
    errors: list[str] = []

    records = list(_source(tmp_path, token="tok").iter_records(None, errors=errors))

    assert len(records) == 200
    assert len(errors) == 1
    assert "200 rows share this fault" in errors[0]
    assert "ticktick api poll" in errors[0]


def test_a_normal_backup_leaves_the_errors_sink_empty(tmp_path):
    """The other half: an alert that fires on every healthy run is not an alert."""
    _backup(tmp_path / "downloads" / "TickTick.csv", [_row(), _row(task_id="t2", status="2")])
    errors: list[str] = []
    assert len(list(_source(tmp_path).iter_records(None, errors=errors))) == 2
    assert errors == []


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


# --- guessed data must not overwrite measured data -------------------------
#
# Round 4 HIGH 2. "Newest observation wins" is right for freshness and wrong
# for provenance. An inferred completion carries the POLL's clock, so it is
# always newer than the backup file it is merged against — and it was
# therefore replacing the export's exact ``Completed Time`` with an
# approximate one on the very run that first read that export. The backup is
# the only place an exact completion timestamp exists, and an incremental run
# never re-reads it, so nothing could put the measurement back.


def _seed_baseline(tmp_path, tasks, when=datetime(2026, 8, 1, tzinfo=UTC)):
    """Make ``tasks`` the previous poll's open set, without running a poll."""
    ticktick_api.save_state(tmp_path / "state.json", tasks, when)


def _aged(path, hours):
    stamp = (datetime.now(UTC) - timedelta(hours=hours)).timestamp()
    os.utime(path, (stamp, stamp))
    return path


def test_an_inferred_completion_never_overwrites_a_measured_one(tmp_path, monkeypatch):
    """Measured-then-inferred, in one run: the exact timestamp survives.

    The ordinary sequence — a task is open at the last poll, the user finishes
    it, the user exports a backup, the next run reads both. The backup's
    ``Completed Time`` is a measurement TickTick made; the poll's inference is
    a guess that the task's disappearance means completion, stamped 'now'.
    Recency alone handed the run to the guess.
    """
    _seed_baseline(tmp_path, [{"id": "t1", "title": "Ship it", "projectId": "p1"}])
    _aged(
        _backup(
            tmp_path / "downloads" / "TickTick.csv",
            [_row(task_id="t1", status="2", completed="2026-08-03T17:30:00+0000")],
        ),
        hours=1,
    )
    monkeypatch.setattr(ticktick_api, "poll_open_tasks", lambda token, errors=None: _poll([]))

    (rec,) = list(_source(tmp_path, token="tok").iter_records(None))

    assert rec.extra["provenance"] == "csv"
    assert "completed_time_approx" not in rec.extra
    assert rec.updated_at == datetime(2026, 8, 3, 17, 30, tzinfo=UTC)


@pytest.mark.parametrize("status", ["2", "-1", "7"])
def test_every_measured_non_open_status_is_protected(tmp_path, monkeypatch, status):
    """``-1`` (abandoned) is as terminal as ``2``, and an unrecognised code is a
    measured value this repo keeps verbatim rather than guessing at — replacing
    either with an approximate completion destroys the same evidence."""
    _seed_baseline(tmp_path, [{"id": "t1", "projectId": "p1"}])
    _aged(
        _backup(
            tmp_path / "downloads" / "TickTick.csv",
            [_row(task_id="t1", status=status, completed="2026-08-03T17:30:00+0000")],
        ),
        hours=1,
    )
    monkeypatch.setattr(ticktick_api, "poll_open_tasks", lambda token, errors=None: _poll([]))

    (rec,) = list(_source(tmp_path, token="tok").iter_records(None, errors=[]))
    assert rec.extra["provenance"] == "csv"
    assert rec.extra["status"] == status


def test_a_measured_completion_still_corrects_an_earlier_inferred_one(tmp_path, monkeypatch):
    """Inferred-then-measured, across two runs: the guess is not permanent.

    Run 1 has no backup at all and infers the completion. Run 2 reads the
    export the user finally took. The measurement must win — the fix is
    "provenance outranks recency", not "whatever is in the index stays".
    """
    _seed_baseline(tmp_path, [{"id": "t1", "title": "Ship it", "projectId": "p1"}])
    monkeypatch.setattr(ticktick_api, "poll_open_tasks", lambda token, errors=None: _poll([]))
    first = _source(tmp_path, token="tok")
    (guess,) = list(first.iter_records(None))
    first.commit_after_write()
    assert guess.extra["provenance"] == "api-inferred-complete"

    _backup(
        tmp_path / "downloads" / "TickTick.csv",
        [_row(task_id="t1", status="2", completed="2026-08-03T17:30:00+0000")],
    )
    (rec,) = list(_source(tmp_path, token="tok").iter_records(None))

    assert rec.extra["provenance"] == "csv"
    assert rec.updated_at == datetime(2026, 8, 3, 17, 30, tzinfo=UTC)


def test_a_reopened_task_still_beats_a_stale_measured_completion(tmp_path, monkeypatch):
    """The rule is about GUESSES, not about completions in general.

    The poll serving a task is a measurement — the Open API only ever serves
    OPEN tasks — so a re-opened task must still overturn last month's backup
    row. A rule that read "a completed CSV row always wins" would freeze every
    re-opened task as finished forever.
    """
    _seed_baseline(tmp_path, [{"id": "t1", "projectId": "p1"}])
    _aged(
        _backup(
            tmp_path / "downloads" / "TickTick.csv",
            [_row(task_id="t1", status="2", completed="2026-07-01T09:00:00+0000")],
        ),
        hours=24 * 30,
    )
    monkeypatch.setattr(
        ticktick_api,
        "poll_open_tasks",
        lambda token, errors=None: _poll([{"id": "t1", "title": "Reopened", "projectId": "p1"}]),
    )

    (rec,) = list(_source(tmp_path, token="tok").iter_records(None))

    assert rec.extra["provenance"] == "api"
    assert "open" in rec.tags


def test_both_legs_one_poll_an_open_csv_row_yields_to_the_inference(tmp_path, monkeypatch):
    """Same task, both legs, one run — and here the inference SHOULD win.

    The backup says the task was open when it was exported; that row holds no
    completion to protect, and superseding it is the whole reason the API leg
    exists (a task finished since the last export is completed in the index
    without waiting for the next manual backup). A guard that fired on any CSV
    row at all would have quietly disabled completion inference for every task
    that has ever appeared in a backup.
    """
    _seed_baseline(tmp_path, [{"id": "t1", "title": "Ship it", "projectId": "p1"}])
    _aged(
        _backup(tmp_path / "downloads" / "TickTick.csv", [_row(task_id="t1", status="0")]),
        hours=24 * 7,
    )
    monkeypatch.setattr(ticktick_api, "poll_open_tasks", lambda token, errors=None: _poll([]))

    (rec,) = list(_source(tmp_path, token="tok").iter_records(None))

    assert rec.extra["provenance"] == "api-inferred-complete"
    assert rec.extra["completed_time_approx"] == "true"


# -- round 5 HIGH 2: an observation timestamp describes the observation ----
#
# The API leg's ``observed_at`` was fixed at ADAPTER CONSTRUCTION. ``run_imports``
# is public and takes adapter INSTANCES, so a long-lived caller reuses one across
# runs — and the merge is newest-observation-wins, so that stale stamp made a
# poll happening right now lose to any backup exported since the adapter was
# built. A reopened task stayed completed, and a stale open row could override
# current inference. The clock is now read one line before the request goes out.


class _Clock:
    """A wall clock a test can move. Installed over ``ticktick._now``.

    The seam exists because the alternative for testing "the poll is older than
    the backup" is a backup file dated in the FUTURE, and because a construction-
    time bug is only visible when construction and the poll are different
    moments — which, with a real clock, they are by microseconds.
    """

    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


def test_a_reused_adapter_stamps_the_poll_not_its_own_construction(
    tmp_path, monkeypatch
):
    """THE round-5 finding, in the shape that costs a user real data.

    A task was completed in the backup exported on the 10th and REOPENED since.
    The poll on the 20th serves it — the Open API serves open tasks only, so
    that is a measurement of "open right now" — and must win. It did not: the
    adapter was constructed on the 1st and every record from its API leg was
    stamped with that, so a nineteen-day-old birthday lost to a ten-day-old
    file, and the index kept insisting a task the user is working on is done.
    """
    clock = _Clock(datetime(2026, 8, 1, tzinfo=UTC))
    monkeypatch.setattr(ticktick, "_now", clock)
    src = _source(tmp_path, token="tok")  # built on the 1st, held ever since

    path = _backup(
        tmp_path / "downloads" / "TickTick.csv",
        [_row(task_id="t1", status="2", completed="2026-08-09T17:30:00+0000")],
    )
    exported = datetime(2026, 8, 10, tzinfo=UTC).timestamp()
    os.utime(path, (exported, exported))
    monkeypatch.setattr(
        ticktick_api,
        "poll_open_tasks",
        lambda token, errors=None: _poll(
            [{"id": "t1", "title": "Reopened", "projectId": "p1"}]
        ),
    )
    clock.value = datetime(2026, 8, 20, tzinfo=UTC)  # ...and polling today

    (rec,) = list(src.iter_records(None))

    assert rec.extra["provenance"] == "api"
    assert "open" in rec.tags


def test_a_poll_time_stamp_still_does_not_let_a_guess_beat_a_measurement(
    tmp_path, monkeypatch
):
    """The interaction the fix must not break, pushed to its limit.

    Poll-time is newer than construction-time by definition, so this moves every
    API record FORWARD in the merge — including the inferred completions, which
    are guesses stamped with the poll's own clock. The provenance rule is what
    keeps that harmless, and it is a rule about provenance rather than about
    dates: even a clock a year ahead may not overwrite a measured Completed Time.
    """
    _seed_baseline(tmp_path, [{"id": "t1", "title": "Ship it", "projectId": "p1"}])
    _aged(
        _backup(
            tmp_path / "downloads" / "TickTick.csv",
            [_row(task_id="t1", status="2", completed="2026-08-03T17:30:00+0000")],
        ),
        hours=1,
    )
    monkeypatch.setattr(ticktick_api, "poll_open_tasks", lambda token, errors=None: _poll([]))
    monkeypatch.setattr(
        ticktick, "_now", lambda: datetime.now(UTC) + timedelta(days=365)
    )

    (rec,) = list(_source(tmp_path, token="tok").iter_records(None))

    assert rec.extra["provenance"] == "csv"
    assert rec.updated_at == datetime(2026, 8, 3, 17, 30, tzinfo=UTC)


def test_a_reused_adapter_sees_a_backup_that_landed_after_it_was_built(tmp_path):
    """The CSV leg's half of the same question, checked rather than assumed.

    Its observation time is the file's mtime, and ``_backup_files`` stats on
    every scan — so an export a human takes while a long-lived caller is holding
    the adapter is picked up on the next run with its own timestamp. Nothing
    about the directory listing is captured at construction.
    """
    src = _source(tmp_path)  # no token: CSV-only, so only the mtime decides
    assert list(src.iter_records(None)) == []

    _backup(tmp_path / "downloads" / "TickTick.csv", [_row()])

    assert [r.stable_id for r in src.iter_records(None)] == ["ticktick:abc123"]
