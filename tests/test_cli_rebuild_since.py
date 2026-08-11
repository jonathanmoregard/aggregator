"""Round-2 HIGH-3: ``--rebuild`` and ``--since`` cannot both mean what they say.

``--rebuild`` DELETEs every row the re-scan did not reproduce. ``--since``
narrows the scan to a window. Together they delete the history OUTSIDE the
window, which is data the run never looked at — and the guards do not save it:
the ratio guard is bypassed entirely for a store of <=100 rows, and a wide
window keeps the shrink under 20% for a big one. Nothing prints a ``deleted=``
count either, so the summary of a run that dropped 4 of 5 rows reads
``added=1 updated=0 skipped=0 errors=0``.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from aggregator.cli import main
from aggregator.core.store import Store
from aggregator.sources.base import Record


def _record(n: int, ts: datetime) -> Record:
    return Record(
        stable_id=f"alpha:{n}",
        source="alpha",
        subject=f"alpha {n}",
        body="body",
        created_at=ts,
        updated_at=ts,
    )


class _WindowedSource:
    """A source that honours ``since``, as every incremental source does."""

    name = "alpha"

    def __init__(self) -> None:
        self.all = [
            _record(n, datetime(2026, 7, n + 1, tzinfo=UTC)) for n in range(1, 5)
        ] + [_record(5, datetime(2026, 8, 5, tzinfo=UTC))]

    def iter_records(self, since, errors=None):
        for r in self.all:
            if since is None or r.updated_at >= since:
                yield r


def _store(tmp_path) -> Store:
    store = Store(db_path=tmp_path / "cache.db")
    store.migrate()
    return store


def test_rebuild_with_since_is_a_usage_error(tmp_path, capsys):
    """THE finding. The windowed scan reproduces 1 of the 5 stored rows and
    the DELETE takes the other 4 — under the ratio guard's 100-row floor, at
    exit 0, with no ``deleted=`` count anywhere in the summary."""
    store = _store(tmp_path)
    src = _WindowedSource()
    store.upsert(src.all)
    assert store.count_by_source("alpha") == 5

    rc = main(
        ["ingest", "alpha", "--rebuild", "--since", "2026-08-01"],
        _store=store,
        _sources={"alpha": src},
    )

    assert rc == 2
    assert store.count_by_source("alpha") == 5, "no row may be deleted"
    err = capsys.readouterr().err
    assert "--rebuild" in err
    assert "--since" in err


def test_the_refusal_names_both_ways_out(tmp_path, capsys):
    store = _store(tmp_path)

    main(
        ["ingest", "alpha", "--rebuild", "--since", "2026-08-01"],
        _store=store,
        _sources={"alpha": _WindowedSource()},
    )

    err = capsys.readouterr().err
    assert "full re-scan" in err or "drop --since" in err


def test_rebuild_alone_is_still_allowed(tmp_path):
    """The other half: a full re-scan is what --rebuild is FOR."""
    store = _store(tmp_path)
    src = _WindowedSource()
    store.upsert(src.all)

    rc = main(
        ["ingest", "alpha", "--rebuild"], _store=store, _sources={"alpha": src}
    )

    assert rc == 0
    assert store.count_by_source("alpha") == 5


def test_since_alone_is_still_allowed(tmp_path):
    store = _store(tmp_path)
    src = _WindowedSource()
    store.upsert(src.all)

    rc = main(
        ["ingest", "alpha", "--since", "2026-08-01"],
        _store=store,
        _sources={"alpha": src},
    )

    assert rc == 0
    assert store.count_by_source("alpha") == 5, "an upsert deletes nothing"


@pytest.mark.parametrize("flags", [["--all", "--since", "2026-08-01"]])
def test_since_still_works_with_all(tmp_path, flags):
    """--all is upsert-only, so a window there deletes nothing."""
    store = _store(tmp_path)

    assert main(["ingest", *flags], _store=store, _adapters=[]) == 0
