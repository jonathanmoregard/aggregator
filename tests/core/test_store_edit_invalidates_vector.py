"""An edited body must not keep the vector of the text it used to hold.

The ``ON CONFLICT DO UPDATE`` branches rewrite ``body`` and ``src_hash`` and
nothing else. The ``observations_au`` trigger keeps ``obs_fts`` in step, so
the KEYWORD arm is always correct after an edit. The vector arm has no
trigger and no equivalent: ``embedding_state`` stays ``'ok'`` and the
``vec_*`` row keeps the old embedding.

``select_unembedded`` is ``WHERE embedding_state IS NULL``, and the only
path that ever clears the column is the poison-ledger requeue for
``state='error'``. So an edited row with ``state='ok'`` keeps a vector of
its OLD body permanently — the two arms of the same hybrid query disagree
about what the row says, and only the keyword one is right.

This is not an edge case on this corpus. GitHub PR rows are re-ingested on
every tick and their bodies genuinely change; a ``SCRUB_FINGERPRINT`` bump
rewrites every body wholesale.

The other half matters just as much: an UNCHANGED re-ingest must not
invalidate anything. The backfill is 25-30 days of CPU, and a reset that
fired on every tick would mean it never finishes.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from aggregator.core.store import _VEC_DIM, Store
from aggregator.sources.base import ObservationRow, Record, SessionRow

TS = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "cache.db")
    s.migrate()
    yield s
    s.close()


def session(sid: str = "s1") -> SessionRow:
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
        first_ts=TS - timedelta(minutes=5),
        last_ts=TS,
        jsonl_path="/tmp/s1.jsonl",
    )


def observation(oid: str = "o1", body: str = "hello") -> ObservationRow:
    return ObservationRow(
        obs_id=oid,
        session_id="s1",
        root_session_id="s1",
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


def _unit(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(_VEC_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _embed_obs(store, obs_id, seed=0):
    store.upsert_vec_observations([(obs_id, _unit(seed))])
    store.mark_embedded("observations", [obs_id], "ok")


def _embed_rec(store, stable_id, seed=0):
    store.upsert_vec_records([(stable_id, _unit(seed))])
    store.mark_embedded("records", [stable_id], "ok")


def _state(store, table, col, key):
    row = store._c().execute(
        f"SELECT embedding_state FROM {table} WHERE {col} = ?",  # noqa: S608 - fixed literals
        (key,),
    ).fetchone()
    return row[0]


# --- observations -----------------------------------------------------------


def test_editing_a_body_returns_the_observation_to_the_embed_backlog(store):
    store.upsert_entities([session(), observation(body="the original text")])
    _embed_obs(store, "o1")
    assert _state(store, "observations", "obs_id", "o1") == "ok"

    store.upsert_entities([session(), observation(body="something else entirely")])

    assert _state(store, "observations", "obs_id", "o1") is None
    assert {r["obs_id"] for r in store.select_unembedded("observations")} == {"o1"}


def test_editing_a_body_drops_the_vector_of_the_old_text(store):
    """A stale vector left in place is served by KNN as if it were current."""
    store.upsert_entities([session(), observation(body="the original text")])
    _embed_obs(store, "o1")
    assert store.count_vec_rows("observations") == 1

    store.upsert_entities([session(), observation(body="something else entirely")])

    assert store.count_vec_rows("observations") == 0
    assert store._vec_obs_scored(_unit(0), k=5) == []


def test_an_unchanged_reingest_leaves_the_vector_and_the_watermark_alone(store):
    """The backfill is weeks long; a reset per tick means it never lands."""
    store.upsert_entities([session(), observation(body="the original text")])
    _embed_obs(store, "o1")

    store.upsert_entities([session(), observation(body="the original text")])

    assert _state(store, "observations", "obs_id", "o1") == "ok"
    assert store.count_vec_rows("observations") == 1
    assert store.select_unembedded("observations") == []


def test_editing_one_observation_does_not_disturb_its_neighbours(store):
    store.upsert_entities(
        [session(), observation("o1", "first body"), observation("o2", "second body")]
    )
    _embed_obs(store, "o1", seed=1)
    _embed_obs(store, "o2", seed=2)

    store.upsert_entities([session(), observation("o1", "first body, rewritten")])

    assert _state(store, "observations", "obs_id", "o2") == "ok"
    assert [i for i, _ in store._vec_obs_scored(_unit(2), k=5)] == ["o2"]


def test_every_chunk_row_of_an_edited_observation_is_dropped(store):
    """A long body embeds as ``<id>:N`` chunk rows, not as one row."""
    store.upsert_entities([session(), observation(body="long original")])
    store.upsert_vec_observations(
        [(f"o1:{i}", _unit(i)) for i in range(4)]
    )
    store.mark_embedded("observations", ["o1"], "ok")
    assert store.count_vec_rows("observations") == 4

    store.upsert_entities([session(), observation(body="short")])

    assert store.count_vec_rows("observations") == 0


def test_a_skipped_row_that_gains_a_body_is_re_embedded(store):
    """'skip' means "nothing to embed" — an edit can make that false."""
    store.upsert_entities([session(), observation(body="")])
    store.mark_embedded("observations", ["o1"], "skip")

    store.upsert_entities([session(), observation(body="now it has words")])

    assert _state(store, "observations", "obs_id", "o1") is None


# --- records ----------------------------------------------------------------


def test_editing_a_record_returns_it_to_the_backlog_and_drops_its_vector(store):
    store.upsert([record(body="original record body")])
    _embed_rec(store, "github:x:1")
    assert store.count_vec_rows("records") == 1

    store.upsert([record(body="rewritten record body")])

    assert _state(store, "records", "stable_id", "github:x:1") is None
    assert store.count_vec_rows("records") == 0


def test_an_unchanged_record_reingest_leaves_its_vector_alone(store):
    store.upsert([record(body="original record body")])
    _embed_rec(store, "github:x:1")

    store.upsert([record(body="original record body")])

    assert _state(store, "records", "stable_id", "github:x:1") == "ok"
    assert store.count_vec_rows("records") == 1


def test_invalidation_works_without_sqlite_vec(tmp_path, monkeypatch):
    """The watermark is a plain column and must reset even with no extension.

    Otherwise a cache that ingested edits while the extension was broken
    would come back with those rows marked embedded and never revisited.
    """
    import sqlite3

    from aggregator.core import store as store_mod

    def _boom(conn):
        raise sqlite3.OperationalError("simulated sqlite-vec ABI mismatch")

    monkeypatch.setattr(store_mod, "_load_sqlite_vec", _boom)
    monkeypatch.setattr(store_mod, "_VEC_LOAD_WARNED", False)

    s = Store(tmp_path / "cache.db")
    s.migrate()
    assert s.vector_available is False
    s.upsert_entities([session(), observation(body="original")])
    # Written directly: this stages a row embedded BEFORE the extension broke.
    # ``mark_embedded(state='ok')`` refuses here by design, since as an API
    # call it asserts a vector was just written.
    c = s._c()
    c.execute("UPDATE observations SET embedding_state = 'ok' WHERE obs_id = 'o1'")
    c.commit()

    s.upsert_entities([session(), observation(body="edited")])

    assert _state(s, "observations", "obs_id", "o1") is None
    s.close()
