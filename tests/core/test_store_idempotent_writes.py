"""Re-writing a row that did not change must cost NOTHING.

THE MEASUREMENT THIS EXISTS FOR. Between 21:18 and 21:58 on 2026-08-15 the
live ``cache.db`` grew 929 MB -> 943 MB while the ``observations`` row count sat
flat at ~372,4xx and ``max(rowid)`` climbed 547,998 -> 569,289. That is
delete-and-reinsert churn at roughly 1:1 — every re-ingested observation freed
its page and allocated a new one, so the file recorded the historical peak of a
run's churn rather than the size of the data. SQLite's default
``auto_vacuum=NONE`` never returns those pages to the OS.

Worse, the expensive half was never SQLite at all: every re-emitted observation
was run through Presidio's scrubber before the write. At 827 rows/min the
pipeline was spending essentially all of its wall clock re-scrubbing text it
had already scrubbed.

So idempotency here is not "the same rows end up stored" — that was already
true. It is "an unchanged row costs no scrub, no page write, and no rowid".
That is what makes an interrupted run resumable in practice rather than only on
paper: the next run re-reads from the stale cursor and walks past everything it
already has almost for free.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from aggregator.core.store import Store
from aggregator.sources.base import ObservationRow, Record, SessionRow

TS = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "cache.db")
    s.migrate()
    yield s
    s.close()


def session(sid: str = "s1", last: datetime = TS) -> SessionRow:
    return SessionRow(
        session_id=sid,
        root_session_id=sid,
        parent_session_id=None,
        kind="session",
        agent_id=None,
        agent_type=None,
        spawned_by_tool_use_id=None,
        cwd="/tmp",
        git_branch="main",
        first_ts=last - timedelta(minutes=5),
        last_ts=last,
        jsonl_path="/tmp/s1.jsonl",
    )


def observation(oid: str = "o1", sid: str = "s1", body: str = "hello") -> ObservationRow:
    return ObservationRow(
        obs_id=oid,
        session_id=sid,
        root_session_id=sid,
        parent_obs_id=None,
        type="user",
        ts=TS,
        model=None,
        input_tokens=None,
        output_tokens=None,
        tool_name=None,
        tool_use_id=None,
        body=body,
    )


def record(rid: str = "github:x:1", body: str = "body") -> Record:
    return Record(
        stable_id=rid,
        source="github",
        subject="subject",
        body=body,
        tags=["a"],
        created_at=TS,
        updated_at=TS,
        extra={"k": "v"},
    )


def _rowids(store: Store, table: str, key: str) -> dict[str, int]:
    return {
        row[0]: row[1]
        for row in store._c().execute(f"SELECT {key}, rowid FROM {table}")  # noqa: S608 - test-local, fixed literals
    }


# --- observations ---------------------------------------------------------


def test_reupserting_identical_observations_writes_no_rows(store):
    """``total_changes`` is the whole point: zero means zero page writes."""
    store.upsert_entities([session(), observation()])
    before = store._c().total_changes
    store.upsert_entities([session(), observation()])
    assert store._c().total_changes == before


def test_reupserting_identical_observations_keeps_the_rowid(store):
    """Rowid stability is what stops the freelist growing on every run.

    Delete-and-reinsert allocates a fresh rowid (and a fresh page) for a row
    whose content never moved, which is exactly the ~1:1 churn measured against
    the live database.
    """
    store.upsert_entities([session(), observation()])
    first = _rowids(store, "observations", "obs_id")
    store.upsert_entities([session(), observation()])
    assert _rowids(store, "observations", "obs_id") == first


def test_a_changed_observation_is_still_written_and_reindexed(store):
    """The guard must not become a way to miss a real edit."""
    store.upsert_entities([session(), observation(body="hello")])
    store.upsert_entities([session(), observation(body="goodbye")])
    row = store._c().execute(
        "SELECT body FROM observations WHERE obs_id = 'o1'"
    ).fetchone()
    assert row[0] == "goodbye"
    hits = store._c().execute(
        "SELECT obs_id FROM observations WHERE rowid IN "
        "(SELECT rowid FROM obs_fts WHERE obs_fts MATCH 'goodbye')"
    ).fetchall()
    assert [h[0] for h in hits] == ["o1"]
    stale = store._c().execute(
        "SELECT rowid FROM obs_fts WHERE obs_fts MATCH 'hello'"
    ).fetchall()
    assert stale == []


def test_unchanged_observations_are_counted_not_hidden(store):
    """A skip nobody can see reads as full coverage. Report the number."""
    store.upsert_entities([session(), observation()])
    assert store.upsert_entities([session(), observation()]) == 2
    assert store.upsert_entities([session(), observation(body="new")]) == 1


def test_an_unchanged_row_is_not_scrubbed_again(store, monkeypatch):
    """The wall-clock fix. Presidio, not SQLite, is what 827 rows/min buys.

    Asserted by counting scrub calls rather than by timing, so it stays a fact
    about the code path instead of a fact about this machine.
    """
    import aggregator.core.store as store_mod

    calls: list[str] = []
    real = store_mod.scrub

    def counting_scrub(text):
        calls.append(text)
        return real(text)

    monkeypatch.setattr(store_mod, "scrub", counting_scrub)
    store.upsert_entities([session(), observation()])
    first = len(calls)
    assert first >= 1
    store.upsert_entities([session(), observation()])
    assert len(calls) == first


# --- records --------------------------------------------------------------


def test_reupserting_identical_records_writes_no_rows(store):
    store.upsert([record()])
    before = store._c().total_changes
    store.upsert([record()])
    assert store._c().total_changes == before


def test_a_changed_record_is_still_written_and_reindexed(store):
    store.upsert([record(body="alpha")])
    store.upsert([record(body="omega")])
    row = store._c().execute(
        "SELECT body FROM records WHERE stable_id = 'github:x:1'"
    ).fetchone()
    assert row[0] == "omega"
    hits = store._c().execute(
        "SELECT stable_id FROM records_fts WHERE records_fts MATCH 'omega'"
    ).fetchall()
    assert [h[0] for h in hits] == ["github:x:1"]


def test_records_unchanged_count_is_returned(store):
    store.upsert([record()])
    assert store.upsert([record()]) == 1
    assert store.upsert([record(body="different")]) == 0


def test_a_record_missing_a_date_still_gains_one_later(store):
    """The COALESCE merge must survive the unchanged-guard.

    ``updated_at = COALESCE(excluded.updated_at, records.updated_at)`` exists so
    a dateless re-observation cannot erase a stored date. The reverse — a stored
    NULL later filled in — has to keep working, or the guard turns "no date yet"
    into "no date ever".
    """
    dateless = record()
    dateless.created_at = None
    dateless.updated_at = None
    store.upsert([dateless])
    assert store.upsert([record()]) == 0
    row = store._c().execute(
        "SELECT created_at, updated_at FROM records WHERE stable_id = 'github:x:1'"
    ).fetchone()
    assert row[0] is not None
    assert row[1] is not None


def test_a_dateless_reobservation_does_not_erase_a_stored_date(store):
    store.upsert([record()])
    dateless = record()
    dateless.updated_at = None
    # Nothing effective changes, so this must be a no-op rather than a wipe.
    store.upsert([dateless])
    row = store._c().execute(
        "SELECT updated_at FROM records WHERE stable_id = 'github:x:1'"
    ).fetchone()
    assert row[0] is not None


# --- migration ------------------------------------------------------------


def test_v3_to_v4_upgrades_in_place_and_self_heals(tmp_path):
    """A v3 database must gain the fingerprint columns without a rebuild.

    The live cache.db holds ~372k observations. A rewriting migration would be
    exactly the hours-long unattended operation this whole change exists to
    abolish, so ``ADD COLUMN`` with no default (metadata-only) is the only
    acceptable shape — and there is deliberately NO backfill: every pre-v4 row
    arrives with a NULL fingerprint, which matches nothing, so the first ingest
    after the upgrade re-writes it once and stamps it. Self-healing over
    exactly one run, and that run is chunked and interruptible.
    """
    s = Store(tmp_path / "cache.db")
    s.migrate()
    s.upsert_entities([session(), observation()])
    s.upsert([record()])
    c = s._c()
    for table in ("sessions", "observations", "records"):
        c.execute(f"UPDATE {table} SET src_hash = NULL")  # noqa: S608 - fixed literals
    c.execute("PRAGMA user_version = 3")
    c.commit()
    s.close()

    s2 = Store(tmp_path / "cache.db")
    s2.migrate()
    try:
        assert s2.schema_version() == 4
        # Nothing matches a NULL fingerprint, so the first pass rewrites.
        assert s2.upsert_entities([session(), observation()]) == 0
        assert s2.upsert([record()]) == 0
        # ...and every pass after it is free.
        assert s2.upsert_entities([session(), observation()]) == 2
        assert s2.upsert([record()]) == 1
        # The rows survived: this is an upgrade, not a rebuild.
        assert s2._c().execute("SELECT count(*) FROM observations").fetchone()[0] == 1
        assert s2._c().execute("SELECT count(*) FROM records").fetchone()[0] == 1
    finally:
        s2.close()


def test_bumping_the_scrub_fingerprint_forces_a_rewrite(store, monkeypatch):
    """The escape hatch, and the reason it has to exist.

    Skipping unchanged rows means a change to ``core/scrub.py`` would never
    reach the rows already stored — the input did not move, so nothing would
    re-scrub it, and a newly-detected secret would sit in the index forever.
    The fingerprint rides in the stored hash precisely so bumping it re-scrubs
    the whole corpus on the next run.
    """
    import aggregator.core.store as store_mod

    store.upsert_entities([session(), observation()])
    monkeypatch.setattr(store_mod, "SCRUB_FINGERPRINT", "different")
    assert store.upsert_entities([session(), observation()]) == 1
