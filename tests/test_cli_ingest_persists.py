"""Regression: `aggregator ingest` must actually persist records to the store.

Pre-fix behaviour (BLOCKER from advisor round-1): ``_cmd_ingest`` called
``src.ingest(since)`` which only counted records and threw them away; the
store passed in was discarded (``_ = store``). Result: the CLI happily
reported "added=N" while ``store.query()`` still returned zero rows.

This test drives the fix. It seeds a stub source that yields two records,
runs the ``ingest`` subcommand against a temp store, and asserts the store
actually holds them afterwards. It also covers the ``--rebuild`` path: the
store must be dropped for the source BEFORE the new records are written,
not after (else the rebuild wipes the freshly-written data).
"""
from __future__ import annotations

from datetime import UTC, datetime

from aggregator import cli
from aggregator.core.store import Store
from aggregator.sources.base import IngestResult, QueryAST, Record


class _StubSource:
    """Yields a fixed set of records via ``iter_records``.

    ``ingest`` is retained for backwards-compat with the Source protocol;
    the CLI is expected to call ``iter_records`` and persist those records
    itself, not to delegate persistence to ``ingest``.
    """

    name = "sessions"

    def __init__(self, records: list[Record]):
        self._records = records

    def iter_records(self, since):
        for r in self._records:
            if since and r.updated_at and r.updated_at < since:
                continue
            yield r

    def ingest(self, since):
        # Preserved for protocol conformance; not called by the CLI post-fix.
        return IngestResult(
            added=sum(1 for _ in self.iter_records(since)),
            updated=0,
            skipped=0,
        )

    def search(self, ast):  # pragma: no cover - unused in this test
        return []

    def record_shape(self):
        return {}


def _mk_records() -> list[Record]:
    return [
        Record(
            stable_id="sessions:persist-1",
            source="sessions",
            subject="first",
            body="body one",
            tags=["proj-alpha"],
            created_at=datetime(2026, 7, 25, tzinfo=UTC),
            updated_at=datetime(2026, 7, 25, tzinfo=UTC),
        ),
        Record(
            stable_id="sessions:persist-2",
            source="sessions",
            subject="second",
            body="body two",
            tags=["proj-alpha"],
            created_at=datetime(2026, 7, 26, tzinfo=UTC),
            updated_at=datetime(2026, 7, 26, tzinfo=UTC),
        ),
    ]


def test_ingest_persists_records_to_store(tmp_data_home):
    """RED against the pre-fix CLI: source-yielded records must land in the store."""
    store = Store()
    store.migrate()
    source = _StubSource(_mk_records())

    rc = cli.main(
        ["ingest", "sessions"], _store=store, _sources={"sessions": source}
    )
    assert rc == 0

    stored = store.query(QueryAST(source="sessions"))
    stored_ids = {r.stable_id for r in stored}
    assert stored_ids == {"sessions:persist-1", "sessions:persist-2"}


def test_ingest_rebuild_drops_before_persist(tmp_data_home):
    """``--rebuild`` must drop the source's rows BEFORE persisting new ones.

    Otherwise a rebuild pass would wipe the records it just wrote. Seed a
    stale row for the source; after ``ingest --rebuild`` the stale row must
    be gone AND the fresh records must be present.
    """
    store = Store()
    store.migrate()
    stale = Record(
        stable_id="sessions:stale",
        source="sessions",
        subject="stale",
        body="stale body",
        created_at=datetime(2026, 7, 1, tzinfo=UTC),
        updated_at=datetime(2026, 7, 1, tzinfo=UTC),
    )
    store.upsert([stale])

    source = _StubSource(_mk_records())
    rc = cli.main(
        ["ingest", "sessions", "--rebuild"],
        _store=store,
        _sources={"sessions": source},
    )
    assert rc == 0

    stored_ids = {r.stable_id for r in store.query(QueryAST(source="sessions"))}
    assert "sessions:stale" not in stored_ids
    assert stored_ids == {"sessions:persist-1", "sessions:persist-2"}
