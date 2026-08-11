"""Round-2 MEDIUM-4: the open-task baseline advanced before anything was written.

``reconcile_open_tasks`` loads the previous poll, diffs it, and saves this poll
as the new baseline — all DURING iteration, before the first record has reached
any sink. A store or sink failure after that point loses the completions the
diff just inferred, and a re-run cannot recover them: the Open API serves OPEN
tasks only, so a task that disappeared between two polls is reported exactly
once.

The fix is two-phase. The diff is computed during iteration (it has to be — it
needs the poll), but the advanced baseline is not persisted until the caller
confirms the records were written, via the ``commit_after_write`` write
barrier. Both write paths call it: the CLI after its upsert, the runner after
the last batch flushes cleanly.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest

from aggregator import cli
from aggregator.core.store import Store
from aggregator.imports.port import WriteCounts
from aggregator.imports.runner import run_imports
from aggregator.imports.ticktick import TickTickAdapter
from aggregator.sources import ticktick_api
from aggregator.sources.ticktick import TickTickSource


@pytest.fixture(autouse=True)
def _no_network_or_real_credentials(monkeypatch, tmp_path):
    def _forbidden(*args, **kwargs):
        raise AssertionError("test attempted a real network call")

    monkeypatch.setattr(ticktick_api, "_open", _forbidden)
    monkeypatch.setattr(
        ticktick_api, "DEFAULT_ENV_FILE", str(tmp_path / "no-such-env")
    )
    for var in (
        "TICKTICK_ACCESS_TOKEN",
        "TICKTICK_TOKEN_EXPIRES_AT",
        "AGGREGATOR_TICKTICK_TOKEN",
        "AGGREGATOR_TICKTICK_TOKEN_FILE",
        "AGGREGATOR_TICKTICK_DIR",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def state_file(tmp_path):
    """A baseline holding two open tasks; this poll will serve only ``t1``,
    so ``t2`` reads as an inferred completion."""
    path = tmp_path / "open_tasks.json"
    path.write_text(
        json.dumps(
            {
                "t1": {"task": {"id": "t1", "title": "open"}, "last_seen": "x"},
                "t2": {"task": {"id": "t2", "title": "done"}, "last_seen": "x"},
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture(autouse=True)
def _one_open_task(monkeypatch):
    monkeypatch.setattr(
        ticktick_api,
        "fetch_open_tasks",
        lambda token, errors=None: [{"id": "t1", "title": "open"}],
    )


def _source(tmp_path, state_file) -> TickTickSource:
    return TickTickSource(
        backup_dir=tmp_path / "no-downloads",
        archive_dir=tmp_path / "no-archive",
        token="fake-token",
        state_file=state_file,
    )


def _baseline(state_file) -> set[str]:
    return set(json.loads(state_file.read_text()))


class _FailingStore(Store):
    """The sink/store failure the finding is about."""

    def upsert(self, records):
        raise sqlite3.OperationalError("database is locked")


class _FailingSink:
    def write(self, items) -> WriteCounts:
        raise sqlite3.OperationalError("database is locked")


class _CountingSink:
    def __init__(self) -> None:
        self.written = []

    def write(self, items) -> WriteCounts:
        self.written.extend(items)
        return WriteCounts(added=len(items))


# -- the finding ----------------------------------------------------------


def test_a_failed_store_write_does_not_consume_the_baseline(
    tmp_path, state_file
):
    """THE finding, on the single-source path. Without the barrier the poll's
    inferred completion for t2 is gone the moment iteration ran, and the API
    will never report it again."""
    store = _FailingStore(db_path=tmp_path / "cache.db")
    store.migrate()

    with pytest.raises(sqlite3.OperationalError):
        cli.main(
            ["ingest", "ticktick"],
            _store=store,
            _sources={"ticktick": _source(tmp_path, state_file)},
        )

    assert _baseline(state_file) == {"t1", "t2"}, (
        "a run that wrote nothing must leave the completion re-inferable"
    )


def test_a_failed_sink_write_does_not_consume_the_baseline(
    tmp_path, state_file
):
    """The same on the run-all path, where the sink failure is contained by
    the runner and the process survives to report it."""
    adapter = TickTickAdapter(source=_source(tmp_path, state_file))

    report = asyncio.run(run_imports([adapter], _FailingSink()))

    assert report.ok is False
    assert _baseline(state_file) == {"t1", "t2"}


# -- the other half: inference must still work ----------------------------


def test_a_successful_run_still_advances_the_baseline(tmp_path, state_file):
    """Without this the fix would silently kill inference forever, which is
    the exact trap ``reconcile_open_tasks``' docstring warns about."""
    store = Store(db_path=tmp_path / "cache.db")
    store.migrate()

    rc = cli.main(
        ["ingest", "ticktick"],
        _store=store,
        _sources={"ticktick": _source(tmp_path, state_file)},
    )

    assert rc == 0
    assert _baseline(state_file) == {"t1"}
    assert store.count_by_source("ticktick") == 2


def test_a_successful_runner_pass_still_advances_the_baseline(
    tmp_path, state_file
):
    sink = _CountingSink()
    adapter = TickTickAdapter(source=_source(tmp_path, state_file))

    report = asyncio.run(run_imports([adapter], sink))

    assert report.ok is True
    assert len(sink.written) == 2
    assert _baseline(state_file) == {"t1"}


def test_the_inferred_completion_is_re_offered_on_the_next_run(
    tmp_path, state_file
):
    """The point of not consuming it: the retry gets a second chance."""
    failing = _FailingStore(db_path=tmp_path / "cache.db")
    failing.migrate()
    with pytest.raises(sqlite3.OperationalError):
        cli.main(
            ["ingest", "ticktick"],
            _store=failing,
            _sources={"ticktick": _source(tmp_path, state_file)},
        )

    store = Store(db_path=tmp_path / "cache.db")
    store.migrate()
    rc = cli.main(
        ["ingest", "ticktick"],
        _store=store,
        _sources={"ticktick": _source(tmp_path, state_file)},
    )

    assert rc == 0
    assert store.count_by_source("ticktick") == 2
    assert _baseline(state_file) == {"t1"}


# -- a barrier that cannot persist has to be loud -------------------------


def test_a_state_write_that_fails_after_the_records_landed_is_reported(
    tmp_path, state_file, monkeypatch
):
    """The records ARE written in this case, so the run is not a failure of
    ingest — but the next poll has lost its baseline, and every completion
    from here on with it. It cannot be silent."""
    store = Store(db_path=tmp_path / "cache.db")
    store.migrate()

    def _boom(path, tasks, now):
        raise OSError("read-only file system")

    monkeypatch.setattr(ticktick_api, "save_state", _boom)

    rc = cli.main(
        ["ingest", "ticktick"],
        _store=store,
        _sources={"ticktick": _source(tmp_path, state_file)},
    )

    assert rc == cli.EXIT_COMPLETED_WITH_ERRORS
    assert store.count_by_source("ticktick") == 2


# -- round 3 W1: a pending commit must not outlive the poll that made it ---
#
# ``run_imports`` is public and takes adapter INSTANCES, so a long-lived caller
# reuses one across runs. The pending advance was only ever cleared by
# ``commit_after_write``; every early return in the poll left the PREVIOUS
# poll's baseline armed, and the next successful write fired it — committing an
# advance whose inferred completions nobody had written.


def test_a_failed_poll_does_not_leave_the_previous_polls_commit_armed(
    tmp_path, state_file, monkeypatch
):
    """THE round-3 finding. Poll 1 infers t2's completion and the write fails,
    so the runner skips the barrier. Poll 2 dies on the API and the run
    degrades to CSV-only. That run's write succeeds — and used to commit poll
    1's baseline, consuming a completion no store ever received."""
    source = _source(tmp_path, state_file)

    list(source.iter_records(None, errors=[]))  # poll 1, then a failed write

    def _dead_api(token, errors=None):
        raise RuntimeError("ticktick 503")

    monkeypatch.setattr(ticktick_api, "fetch_open_tasks", _dead_api)
    errors: list[str] = []
    list(source.iter_records(None, errors=errors))  # poll 2: CSV-only
    assert errors, "the failed poll is still reported"

    source.commit_after_write()  # poll 2's rows landed, so the barrier fires

    assert _baseline(state_file) == {"t1", "t2"}, (
        "a run whose poll failed has no baseline to advance; t2's completion "
        "must stay re-inferable"
    )


def test_a_pollless_run_does_not_leave_the_previous_polls_commit_armed(
    tmp_path, state_file, monkeypatch
):
    """The other early return: the token went away, so there is no poll at
    all. Same rule — nothing to commit."""
    source = _source(tmp_path, state_file)
    list(source.iter_records(None, errors=[]))  # poll 1, then a failed write

    monkeypatch.setattr(source, "_token_arg", None)
    list(source.iter_records(None, errors=[]))  # CSV-only, no credential
    source.commit_after_write()

    assert _baseline(state_file) == {"t1", "t2"}


def test_the_barrier_is_a_no_op_for_sources_that_do_not_have_one(tmp_path):
    """Every other source is unaffected — the barrier is optional."""

    class _Plain:
        name = "plain"

        def iter_records(self, since, errors=None):
            return iter(())

    store = Store(db_path=tmp_path / "cache.db")
    store.migrate()

    assert (
        cli.main(["ingest", "plain"], _store=store, _sources={"plain": _Plain()})
        == 0
    )
