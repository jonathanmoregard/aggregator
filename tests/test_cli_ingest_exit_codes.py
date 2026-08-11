"""Exit codes for `aggregator ingest`. Task 9: fail loudly.

An ingest that ends with a non-empty ``errors`` list used to exit 0, so a
systemd timer saw success while files were being dropped from the index. Per
``tasks/session-constraints.md`` (2026-08-08, "Fail loudly") that is the exact
silent rot this project cannot tolerate: an aggregator that quietly stops
ingesting looks identical to one with nothing new to ingest.

Contract:

* ``0`` — clean run, no errors.
* ``1`` — hard failure (iterator raised, guard refused a wipe). Unchanged.
* ``2`` — usage error (unknown source, bad ``--since``, unknown subcommand).
  Unchanged: the wrapper must tell "you typed a bad source name" apart from
  "the run finished but dropped three PDFs" — different notification text,
  different human response.
* ``3`` — the run completed but reported errors. Partial success included:
  a run that wrote some records and errored on others is NOT a success.
"""
from __future__ import annotations

from datetime import UTC, datetime

from aggregator import cli
from aggregator.core.store import Store
from aggregator.sources.base import ObservationRow, QueryAST, Record, SessionRow

NOW = datetime(2026, 8, 11, tzinfo=UTC)


class _CleanRecordSource:
    name = "clean"

    def iter_records(self, since, errors=None):
        yield Record(
            stable_id="clean:1",
            source="clean",
            subject="ok",
            body="fine",
            created_at=NOW,
            updated_at=NOW,
        )

    def record_shape(self):
        return {}


class _PartiallyFailingRecordSource:
    """Writes one record AND reports one per-file failure — the common case.

    Per-item errors going to the ``errors`` list without aborting the run is
    by design (partial ingest beats total loss); the run still must not
    report success.
    """

    name = "noisy"

    def iter_records(self, since, errors=None):
        if errors is not None:
            errors.append("some/file.pdf: pdf parse failed")
        yield Record(
            stable_id="noisy:1",
            source="noisy",
            subject="ok",
            body="fine",
            created_at=NOW,
            updated_at=NOW,
        )

    def record_shape(self):
        return {}


class _CleanEntitySource:
    name = "sessions"

    def iter_entities(self, since, errors=None):
        yield SessionRow(
            session_id="sess-a",
            root_session_id="sess-a",
            parent_session_id=None,
            kind="session",
            agent_id=None,
            agent_type=None,
            spawned_by_tool_use_id=None,
            cwd=None,
            git_branch=None,
            first_ts=NOW,
            last_ts=NOW,
            jsonl_path="/tmp/a.jsonl",
        )
        yield ObservationRow(
            obs_id="o-a-1",
            session_id="sess-a",
            root_session_id="sess-a",
            parent_obs_id=None,
            type="user",
            ts=NOW,
            model=None,
            input_tokens=None,
            output_tokens=None,
            tool_name=None,
            tool_use_id=None,
            body="hello",
        )

    def record_shape(self):
        return {}


class _PartiallyFailingEntitySource(_CleanEntitySource):
    def iter_entities(self, since, errors=None):
        if errors is not None:
            errors.append("bad.jsonl: line 12 is not JSON")
        yield from _CleanEntitySource.iter_entities(self, since)


def test_clean_records_run_exits_zero(tmp_data_home):
    store = Store()
    store.migrate()
    rc = cli.main(
        ["ingest", "clean"], _store=store, _sources={"clean": _CleanRecordSource()}
    )
    assert rc == 0


def test_records_run_with_errors_exits_three(tmp_data_home, capsys):
    store = Store()
    store.migrate()
    rc = cli.main(
        ["ingest", "noisy"],
        _store=store,
        _sources={"noisy": _PartiallyFailingRecordSource()},
    )
    assert rc == 3
    assert "pdf parse failed" in capsys.readouterr().err


def test_partial_success_still_exits_three(tmp_data_home):
    """The record DID land. The run still failed: a partially-successful run
    is not a successful run, and a timer must not read it as one."""
    store = Store()
    store.migrate()
    rc = cli.main(
        ["ingest", "noisy"],
        _store=store,
        _sources={"noisy": _PartiallyFailingRecordSource()},
    )
    assert rc == 3
    stored = {r.stable_id for r in store.query(QueryAST(source="noisy"))}
    assert stored == {"noisy:1"}


def test_clean_entities_run_exits_zero(tmp_data_home):
    store = Store()
    store.migrate()
    rc = cli.main(
        ["ingest", "sessions"],
        _store=store,
        _sources={"sessions": _CleanEntitySource()},
    )
    assert rc == 0


def test_entities_run_with_errors_exits_three(tmp_data_home, capsys):
    store = Store()
    store.migrate()
    rc = cli.main(
        ["ingest", "sessions"],
        _store=store,
        _sources={"sessions": _PartiallyFailingEntitySource()},
    )
    assert rc == 3
    assert "not JSON" in capsys.readouterr().err


def test_unknown_source_still_exits_two(tmp_data_home):
    """Exit-code meanings for other failures are preserved, not renumbered."""
    store = Store()
    store.migrate()
    rc = cli.main(
        ["ingest", "nope"],
        _store=store,
        _sources={"clean": _CleanRecordSource()},
    )
    assert rc == 2


def test_bad_since_still_exits_two(tmp_data_home):
    store = Store()
    store.migrate()
    rc = cli.main(
        ["ingest", "clean", "--since", "not-a-date"],
        _store=store,
        _sources={"clean": _CleanRecordSource()},
    )
    assert rc == 2


def test_raising_source_still_exits_one(tmp_data_home):
    class _Exploding:
        name = "boom"

        def iter_records(self, since, errors=None):
            raise RuntimeError("upstream 500")
            yield  # pragma: no cover -- makes this a generator

        def record_shape(self):
            return {}

    store = Store()
    store.migrate()
    rc = cli.main(
        ["ingest", "boom"], _store=store, _sources={"boom": _Exploding()}
    )
    assert rc == 1
