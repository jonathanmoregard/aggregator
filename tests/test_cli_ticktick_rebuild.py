"""Round-1 MEDIUM: ``ingest ticktick --rebuild`` destroyed inferred state.

``--rebuild`` adds exactly one thing over a plain ingest: it DELETEs the rows a
re-scan did not produce. For ticktick every such row is unregenerable.

* ``api-inferred-complete`` rows. The Open API serves OPEN tasks only and
  reports a completion exactly once — as a disappearance between two polls —
  so a completed task is never in a poll again. Nothing regenerates the row.
* CSV rows. The backup is a manual export and the local archive is described
  in ``sources/ticktick.py`` as its only surviving copy.

And the shrink guard does not catch it: a rebuild that drops under 20% of the
source clears the guard with no prompt and no warning at all.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from aggregator import cli
from aggregator.core.store import Store
from aggregator.sources import ticktick_api
from aggregator.sources.base import QueryAST, Record
from aggregator.sources.ticktick import TickTickSource


@pytest.fixture(autouse=True)
def _no_network_or_real_credentials(monkeypatch, tmp_path):
    """Same guard the ticktick source tests use: never the real API, never the
    developer's own token."""

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


def _store(tmp_path) -> Store:
    store = Store(db_path=tmp_path / "cache.db")
    store.migrate()
    return store


def _rows(store: Store, n: int, prefix: str, extra=None) -> None:
    store.upsert(
        [
            Record(
                stable_id=f"ticktick:{prefix}-{i}",
                source="ticktick",
                subject=f"task {i}",
                body="body",
                created_at=datetime(2026, 6, 1, tzinfo=UTC),
                updated_at=datetime(2026, 6, 1, tzinfo=UTC),
                extra=extra or {},
            )
            for i in range(n)
        ]
    )


def _ids(store: Store) -> set[str]:
    return {r.stable_id for r in store.query(QueryAST(source="ticktick"))}


class _StubTickTick:
    """Yields whatever it is given; stands in for the merged source."""

    name = "ticktick"

    def __init__(self, records):
        self._records = list(records)
        self.iterated = 0

    def iter_records(self, since, errors=None):
        self.iterated += 1
        yield from self._records


def test_rebuild_is_refused_and_says_why(tmp_path, capsys):
    store = _store(tmp_path)
    src = _StubTickTick([])

    rc = cli.main(
        ["ingest", "ticktick", "--rebuild"],
        _store=store,
        _sources={"ticktick": src},
    )

    assert rc == 2
    err = capsys.readouterr().err
    assert "--rebuild" in err
    assert "ticktick" in err
    assert "regenerate" in err
    assert src.iterated == 0


def test_a_stored_inferred_completion_is_not_deleted(tmp_path):
    """THE finding. 150 CSV rows re-scanned against 151 stored is a 0.7%
    shrink — nowhere near the 20% guard — and the one row it drops is the one
    nothing can put back."""
    store = _store(tmp_path)
    _rows(store, 150, "t")
    _rows(store, 1, "done", extra={"provenance": "api-inferred-complete"})
    assert "ticktick:done-0" in _ids(store)

    rescan = [
        Record(
            stable_id=f"ticktick:t-{i}",
            source="ticktick",
            subject=f"task {i}",
            body="body",
        )
        for i in range(150)
    ]

    rc = cli.main(
        ["ingest", "ticktick", "--rebuild"],
        _store=store,
        _sources={"ticktick": _StubTickTick(rescan)},
    )

    assert rc == 2
    assert "ticktick:done-0" in _ids(store)
    assert len(_ids(store)) == 151


def test_plain_ingest_still_refreshes_every_row_a_rescan_can_produce(tmp_path):
    """Refusing --rebuild costs nothing an operator actually wanted: the
    everyday upsert already overwrites each re-scanned row in place."""
    store = _store(tmp_path)
    _rows(store, 2, "t")
    _rows(store, 1, "done", extra={"provenance": "api-inferred-complete"})

    fixed = [
        Record(
            stable_id="ticktick:t-0",
            source="ticktick",
            subject="reparsed title",
            body="reparsed body",
        )
    ]
    rc = cli.main(
        ["ingest", "ticktick"],
        _store=store,
        _sources={"ticktick": _StubTickTick(fixed)},
    )

    assert rc == 0
    subjects = {
        r.stable_id: r.subject for r in store.query(QueryAST(source="ticktick"))
    }
    assert subjects["ticktick:t-0"] == "reparsed title"
    assert "ticktick:done-0" in subjects


def test_a_refused_rebuild_leaves_the_open_task_baseline_alone(
    tmp_path, monkeypatch
):
    """Driven through the REAL source, because the loss this prevents happens
    inside its poll: ``reconcile_open_tasks`` advances open_tasks.json during
    iteration, and a run that then refuses has consumed a completion it never
    stored."""
    store = _store(tmp_path)
    _rows(store, 200, "t")

    state_file = tmp_path / "open_tasks.json"
    state_file.write_text(
        json.dumps(
            {
                "t1": {"task": {"id": "t1", "title": "open"}, "last_seen": "x"},
                "t2": {"task": {"id": "t2", "title": "done"}, "last_seen": "x"},
            }
        ),
        encoding="utf-8",
    )
    # This poll serves only t1, so t2 reads as an inferred completion.
    monkeypatch.setattr(
        ticktick_api,
        "fetch_open_tasks",
        lambda token, errors=None: [{"id": "t1", "title": "open"}],
    )
    src = TickTickSource(
        backup_dir=tmp_path / "no-downloads",
        archive_dir=tmp_path / "no-archive",
        token="fake-token",
        state_file=state_file,
    )

    rc = cli.main(
        ["ingest", "ticktick", "--rebuild"],
        _store=store,
        _sources={"ticktick": src},
    )

    assert rc == 2
    assert set(json.loads(state_file.read_text())) == {"t1", "t2"}, (
        "a refused run must not have consumed the open-task baseline"
    )
    assert len(_ids(store)) == 200
