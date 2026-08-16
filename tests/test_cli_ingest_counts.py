"""The ingest summary must report what the run actually did.

Both write paths are upserts (``INSERT ... ON CONFLICT DO UPDATE``), so once
the write has happened there is no way to tell an insert from an overwrite.
The counts therefore have to be established BEFORE the write, by probing which
primary keys the store already holds — exactly what ``imports/store_sink.py``
does on the new runner path.

The regression this pins: ``added = len(records)`` with a hardcoded
``updated=0`` prints the same three numbers no matter what happened, so a
no-op re-run is indistinguishable from a real import. The github timer logged
``added=313 updated=0 errors=0`` on every run for weeks.
"""
from __future__ import annotations

from datetime import UTC, datetime

from aggregator import cli
from aggregator.core.store import Store
from aggregator.sources.base import IngestResult, Record


class _StubRecordSource:
    name = "github"

    def rebuild_input(self) -> str:
        """``sources.base.SupportsRebuild``: --rebuild is opt-in per source,
        and this stub stands in for one whose input a machine keeps current."""
        return "a stub input this test controls entirely"

    def __init__(self, records: list[Record]):
        self._records = records

    def iter_records(self, since, errors=None):
        yield from self._records

    def record_shape(self):
        return {}


def _rec(sid: str, subject: str = "s", body: str = "b") -> Record:
    ts = datetime(2026, 8, 1, tzinfo=UTC)
    return Record(
        stable_id=sid,
        source="github",
        subject=subject,
        body=body,
        tags=["pr"],
        created_at=ts,
        updated_at=ts,
    )


def _summary(capsys) -> str:
    return capsys.readouterr().out.strip().splitlines()[-1]


def test_first_ingest_reports_added(tmp_data_home, capsys):
    store = Store()
    store.migrate()
    source = _StubRecordSource([_rec("github:a/b:1"), _rec("github:a/b:2")])
    rc = cli.main(["ingest", "github"], _store=store, _sources={"github": source})
    assert rc == 0
    line = _summary(capsys)
    assert "added=2" in line, line
    assert "updated=0" in line, line


def test_second_identical_ingest_reports_updated_not_added(
    tmp_data_home, capsys
):
    """The exact regression the old code could not express.

    Nothing about the input changed, so a truthful summary says 0 added and
    2 updated. The old code said ``added=2 updated=0`` — identical to the
    first run's line.
    """
    store = Store()
    store.migrate()
    source = _StubRecordSource([_rec("github:a/b:1"), _rec("github:a/b:2")])
    cli.main(["ingest", "github"], _store=store, _sources={"github": source})
    capsys.readouterr()

    rc = cli.main(["ingest", "github"], _store=store, _sources={"github": source})
    assert rc == 0
    line = _summary(capsys)
    assert "added=0" in line, line
    assert "updated=2" in line, line


def test_partially_new_ingest_splits_added_and_updated(tmp_data_home, capsys):
    store = Store()
    store.migrate()
    first = _StubRecordSource([_rec("github:a/b:1")])
    cli.main(["ingest", "github"], _store=store, _sources={"github": first})
    capsys.readouterr()

    second = _StubRecordSource([_rec("github:a/b:1"), _rec("github:a/b:2")])
    rc = cli.main(["ingest", "github"], _store=store, _sources={"github": second})
    assert rc == 0
    line = _summary(capsys)
    assert "added=1" in line, line
    assert "updated=1" in line, line


def test_duplicate_ids_within_one_batch_count_once(tmp_data_home, capsys):
    """Two items with the same stable_id are one row, so one add."""
    store = Store()
    store.migrate()
    source = _StubRecordSource([_rec("github:a/b:1"), _rec("github:a/b:1")])
    rc = cli.main(["ingest", "github"], _store=store, _sources={"github": source})
    assert rc == 0
    line = _summary(capsys)
    assert "added=1" in line, line
    assert "updated=1" in line, line


def test_rebuild_counts_against_pre_run_state(tmp_data_home, capsys):
    """--rebuild drops the source's rows first, but the counts still answer
    "was this id already known?", probed before anything is deleted."""
    store = Store()
    store.migrate()
    source = _StubRecordSource([_rec("github:a/b:1")])
    cli.main(["ingest", "github"], _store=store, _sources={"github": source})
    capsys.readouterr()

    rc = cli.main(
        ["ingest", "github", "--rebuild"],
        _store=store,
        _sources={"github": source},
    )
    assert rc == 0
    line = _summary(capsys)
    assert "added=0" in line, line
    assert "updated=1" in line, line


class _IngestResultSource:
    """Legacy stub exposing only ``ingest()`` — the count-only path."""

    name = "legacy"

    def ingest(self, since):
        return IngestResult(added=3, updated=4, skipped=5)


def test_ingest_result_path_reports_all_three_counts(tmp_data_home, capsys):
    """A source that reports its own counts must have them printed, not
    have ``updated``/``skipped`` overwritten with a hardcoded 0."""
    store = Store()
    store.migrate()
    rc = cli.main(
        ["ingest", "legacy"],
        _store=store,
        _sources={"legacy": _IngestResultSource()},
    )
    assert rc == 0
    line = _summary(capsys)
    assert "added=3" in line, line
    assert "updated=4" in line, line
    assert "skipped=5" in line, line
