"""Round-3 HIGH: silent data-wipe on transient failure.

Pre-fix behaviour: `_default_api_fetcher` returns [] on any
`CalledProcessError` / `FileNotFoundError` / `TimeoutExpired` / JSON error;
`iter_records` per-endpoint try/except appends the error and continues. If
every endpoint degrades to [] (network hiccup, gh outage), the CLI's
`--rebuild` path called `store.rebuild_and_upsert("github", [])` which
atomically DELETEd all rows for the source and committed — silent wipe.

Fix: two-layer defense.

1. CLI plumbs the errors list into `iter_records(since, errors=errors)`.
   If `records == []` AND `errors != []` (any endpoint failed): refuse
   `--rebuild`, print an actionable error, exit 1. Store is untouched.
2. `Store.rebuild_and_upsert` grows an optional `min_records: int = 0`
   guard. When `len(records) < min_records`, raises `EmptyRebuildRefusedError`.
   CLI passes `min_records=1` when the source's tables in the store are
   currently non-empty. Belt-and-braces against future callers.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from aggregator import cli
from aggregator.core.store import EmptyRebuildRefusedError, Store
from aggregator.sources.base import IngestResult, QueryAST, Record


class _StubSource:
    """Configurable stub: yields ``records`` and appends ``errors`` (via the
    ``errors`` sink from ``iter_records``) so tests can drive the exact
    combination of records/errors the CLI must react to."""

    name = "github"

    def __init__(self, records: list[Record], errors: list[str] | None = None):
        self._records = list(records)
        self._errors = list(errors or [])

    def iter_records(self, since, errors=None):
        if errors is not None:
            errors.extend(self._errors)
        yield from self._records

    def ingest(self, since):  # pragma: no cover - not called on iter_records path
        return IngestResult(added=len(self._records), updated=0, skipped=0)

    def search(self, ast):  # pragma: no cover
        return []

    def record_shape(self):  # pragma: no cover
        return {}


def _seed(store: Store, n: int, source: str = "github") -> None:
    """Seed ``n`` records for ``source`` — used to make the CLI treat the
    store as "historically non-empty" for that source."""
    store.upsert(
        [
            Record(
                stable_id=f"{source}:seed-{i}",
                source=source,
                subject=f"seed {i}",
                body="seed body",
                created_at=datetime(2026, 6, 1, tzinfo=UTC),
                updated_at=datetime(2026, 6, 1, tzinfo=UTC),
            )
            for i in range(n)
        ]
    )


# --- CLI layer -------------------------------------------------------------


def test_rebuild_refuses_when_records_empty_and_errors_nonempty(
    tmp_data_home, capsys
):
    """All four endpoints degraded to [] with errors: --rebuild MUST refuse,
    exit 1, and leave the store untouched. This is the wipe-on-transient-
    failure scenario the round-3 advisor caught."""
    store = Store()
    store.migrate()
    _seed(store, 5)

    source = _StubSource(records=[], errors=[
        "/search/issues?q=is:pr+author:@me: transport failure",
        "/search/issues?q=is:pr+review-requested:@me: transport failure",
        "/search/issues?q=is:issue+author:@me: transport failure",
        "/search/issues?q=is:issue+assignee:@me: transport failure",
    ])
    rc = cli.main(
        ["ingest", "github", "--rebuild"],
        _store=store,
        _sources={"github": source},
    )
    assert rc == 1, "CLI must exit 1 when refusing rebuild"
    # Store still holds the seeded rows.
    remaining = store.query(QueryAST(source="github"))
    assert len(remaining) == 5, "store must be untouched when rebuild refused"
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "refusing" in combined.lower()
    assert "github" in combined
    assert "left intact" in combined.lower() or "untouched" in combined.lower()


def test_rebuild_proceeds_when_records_empty_and_errors_empty(tmp_data_home):
    """Legitimate empty result on a fresh source (no historical data, no
    errors): --rebuild should proceed. Nothing to wipe means no wipe risk."""
    store = Store()
    store.migrate()
    # No seed — store is empty for github.

    source = _StubSource(records=[], errors=[])
    rc = cli.main(
        ["ingest", "github", "--rebuild"],
        _store=store,
        _sources={"github": source},
    )
    assert rc == 0
    # Still empty afterwards, but the command succeeded.
    assert store.query(QueryAST(source="github")) == []


def test_rebuild_refuses_when_store_nonempty_and_zero_records_zero_errors(
    tmp_data_home, capsys
):
    """Belt-and-braces via the store guard: even if the source reports zero
    errors, an empty record set against a historically-non-empty source
    still smells like a bug. The store's ``min_records=1`` guard refuses."""
    store = Store()
    store.migrate()
    _seed(store, 3)

    source = _StubSource(records=[], errors=[])
    rc = cli.main(
        ["ingest", "github", "--rebuild"],
        _store=store,
        _sources={"github": source},
    )
    assert rc == 1
    remaining = store.query(QueryAST(source="github"))
    assert len(remaining) == 3, "store guard must block wipe of nonempty source"
    # (No further assertion — the important thing is the exit code + row count.)
    _ = capsys  # keep fixture in signature; assertions above are the contract


def test_rebuild_succeeds_with_records_and_errors(tmp_data_home):
    """Partial ingest is legitimate: some endpoints succeeded, some failed.
    As long as at least one record arrived, --rebuild proceeds (errors are
    surfaced via warning, not by refusing)."""
    store = Store()
    store.migrate()
    _seed(store, 3)

    good = Record(
        stable_id="github:acme/api:99",
        source="github",
        subject="new pr",
        body="body",
        created_at=datetime(2026, 7, 25, tzinfo=UTC),
        updated_at=datetime(2026, 7, 25, tzinfo=UTC),
    )
    source = _StubSource(records=[good], errors=["one endpoint failed"])
    rc = cli.main(
        ["ingest", "github", "--rebuild"],
        _store=store,
        _sources={"github": source},
    )
    assert rc == 0
    ids = {r.stable_id for r in store.query(QueryAST(source="github"))}
    assert ids == {"github:acme/api:99"}, "rebuild should have replaced seed rows"


# --- Store layer -----------------------------------------------------------


def test_store_rebuild_and_upsert_min_records_guard(tmp_data_home):
    """Direct API test: passing ``min_records=1`` with an empty record list
    must raise ``EmptyRebuildRefusedError`` and leave existing rows intact."""
    s = Store()
    s.migrate()
    _seed(s, 3)

    with pytest.raises(EmptyRebuildRefusedError):
        s.rebuild_and_upsert("github", [], min_records=1)

    rows = s.query(QueryAST(source="github"))
    assert len(rows) == 3, "guard must not commit the DELETE"


def test_store_rebuild_and_upsert_min_records_zero_still_permits_empty(
    tmp_data_home,
):
    """Default ``min_records=0`` preserves the pre-fix API: passing an empty
    list still succeeds (some callers may legitimately want to clear a
    source). The new guard only bites when the caller opts in."""
    s = Store()
    s.migrate()
    _seed(s, 3)

    # Explicit min_records=0 (or omission) still clears.
    s.rebuild_and_upsert("github", [])
    assert s.query(QueryAST(source="github")) == []


def test_empty_rebuild_refused_is_exception_type():
    """Sanity: ``EmptyRebuildRefusedError`` must be an Exception subclass that
    callers can catch specifically (not a bare RuntimeError)."""
    assert issubclass(EmptyRebuildRefusedError, Exception)


def test_cli_surfaces_endpoint_errors_after_successful_ingest(
    tmp_data_home, capsys
):
    """MEDIUM #3 (collapsed into HIGH): CLI must actually plumb errors from
    iter_records into its error print path. Pre-fix, the local
    ``errors: list[str] = []`` was initialized but never populated."""
    store = Store()
    store.migrate()

    good = Record(
        stable_id="github:acme/api:1",
        source="github",
        subject="pr",
        body="body",
        created_at=datetime(2026, 7, 25, tzinfo=UTC),
        updated_at=datetime(2026, 7, 25, tzinfo=UTC),
    )
    source = _StubSource(records=[good], errors=["endpoint X failed: boom"])
    rc = cli.main(
        ["ingest", "github"],
        _store=store,
        _sources={"github": source},
    )
    assert rc == 0
    cap = capsys.readouterr()
    combined = cap.out + cap.err
    # The endpoint error text should appear in the CLI output/stderr.
    assert "boom" in combined
