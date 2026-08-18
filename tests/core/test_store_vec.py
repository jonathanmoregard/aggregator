"""Vector arm: upsert vec rows, KNN reader, watermark helpers.

Two halves, and the second one is the load-bearing half. The first is the happy
path — vectors go in, KNN comes back ordered, ``embedding_state`` advances. The
second is what happens when sqlite-vec is not there: reads raise
:class:`VectorIndexUnavailableError` by name, writes no-op, and the parts of
this surface that are ordinary SQL (``select_unembedded`` / ``mark_embedded``
are plain column reads and writes) keep working, because the embed worker's
backlog bookkeeping has no business depending on a native extension.
"""

import logging
import sqlite3

import numpy as np
import pytest

from aggregator.core import store as store_mod
from aggregator.core.store import Store, VectorIndexUnavailableError


@pytest.fixture
def store(tmp_path):
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    return s


@pytest.fixture
def no_vec_store(tmp_path, monkeypatch):
    """A store whose sqlite-vec load fails, the way a real one would."""

    def _boom(conn):
        raise sqlite3.OperationalError("simulated sqlite-vec ABI mismatch")

    monkeypatch.setattr(store_mod, "_load_sqlite_vec", _boom)
    monkeypatch.setattr(store_mod, "_VEC_LOAD_WARNED", False)
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    assert s.vector_available is False
    return s


def _seed_observation(s, obs_id, body="hello"):
    c = s._c()
    c.execute(
        "INSERT OR IGNORE INTO sessions(session_id, root_session_id, kind, "
        "first_ts, last_ts, jsonl_path, origin) VALUES ('sid', 'sid', "
        "'session', '2026-01-01', '2026-01-01', '/tmp/x.jsonl', 'claude-code')"
    )
    c.execute(
        "INSERT INTO observations(obs_id, session_id, root_session_id, "
        "type, ts, body) VALUES (?, ?, ?, 'user', '2026-01-01', ?)",
        (obs_id, "sid", "sid", body),
    )
    c.commit()


def _seed_record(s, stable_id, body="hello"):
    c = s._c()
    c.execute(
        "INSERT INTO records(stable_id, source, subject, body, tags, "
        "created_at, updated_at) VALUES (?, 'github', 'subj', ?, '[]', "
        "'2026-01-01', '2026-01-01')",
        (stable_id, body),
    )
    c.commit()


# --- happy path -------------------------------------------------------------


def test_upsert_and_read_vec_obs(store):
    _seed_observation(store, "o1")
    vec = np.random.rand(768).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    store.upsert_vec_observations([("o1", vec)])
    query = vec  # exact match
    hits = store._vec_obs_ids(query, k=5)
    assert hits[0] == "o1"


def test_knn_returns_topk_ordered(store):
    for i in range(5):
        _seed_observation(store, f"o{i}")
    vecs = np.eye(5, 768, dtype=np.float32)  # 5 orthogonal unit vectors
    store.upsert_vec_observations([(f"o{i}", vecs[i]) for i in range(5)])
    hits = store._vec_obs_ids(vecs[2], k=3)
    assert hits[0] == "o2"
    assert len(hits) == 3


def test_upsert_vec_obs_is_idempotent(store):
    """Re-embedding the same id must replace, not duplicate — vec0 has no UPSERT."""
    _seed_observation(store, "o1")
    vec = np.eye(1, 768, dtype=np.float32)[0]
    store.upsert_vec_observations([("o1", vec)])
    store.upsert_vec_observations([("o1", vec)])
    n = store._c().execute("SELECT COUNT(*) AS n FROM vec_observations").fetchone()["n"]
    assert n == 1


def test_upsert_and_read_vec_records(store):
    _seed_record(store, "github:1")
    _seed_record(store, "github:2")
    vecs = np.eye(2, 768, dtype=np.float32)
    store.upsert_vec_records([("github:1", vecs[0]), ("github:2", vecs[1])])
    assert store._vec_record_ids(vecs[1], k=1) == ["github:2"]


def test_select_unembedded_observations(store):
    for i in range(3):
        _seed_observation(store, f"u{i}")
    rows = store.select_unembedded("observations", limit=10)
    assert {r["obs_id"] for r in rows} == {"u0", "u1", "u2"}


def test_select_unembedded_records(store):
    _seed_record(store, "github:1")
    rows = store.select_unembedded("records", limit=10)
    assert {r["stable_id"] for r in rows} == {"github:1"}


def test_select_unembedded_rejects_unknown_kind(store):
    with pytest.raises(ValueError, match="unknown kind"):
        store.select_unembedded("nope", limit=1)


def test_mark_embedded_flips_state(store):
    _seed_observation(store, "u0")
    store.mark_embedded("observations", ["u0"], state="ok")
    rows = store.select_unembedded("observations", limit=10)
    assert not any(r["obs_id"] == "u0" for r in rows)


def test_mark_embedded_error_state(store):
    _seed_observation(store, "u0")
    store.mark_embedded("observations", ["u0"], state="error")
    c = store._c()
    row = c.execute(
        "SELECT embedding_state FROM observations WHERE obs_id = ?", ("u0",)
    ).fetchone()
    assert row["embedding_state"] == "error"


def test_mark_embedded_empty_ids_is_a_noop(store):
    """An empty batch must not render ``IN ()``, which is a SQL syntax error.

    The worker reaches this whenever a batch is entirely skips or entirely
    successes: one of the two ``mark_embedded`` calls gets an empty list.
    """
    _seed_observation(store, "u0")
    store.mark_embedded("observations", [], state="ok")
    rows = store.select_unembedded("observations", limit=10)
    assert {r["obs_id"] for r in rows} == {"u0"}


def test_mark_embedded_rejects_invalid_state(store):
    with pytest.raises(ValueError, match="invalid state"):
        store.mark_embedded("observations", ["u0"], state="done")


def test_mark_embedded_rejects_unknown_kind(store):
    with pytest.raises(ValueError, match="unknown kind"):
        store.mark_embedded("nope", ["u0"], state="ok")


def test_vec_writes_refuse_on_a_read_only_store(tmp_path):
    writable = Store(db_path=tmp_path / "cache.db")
    writable.migrate()
    writable.close()
    ro = Store(db_path=tmp_path / "cache.db", read_only=True)
    vec = np.eye(1, 768, dtype=np.float32)[0]
    with pytest.raises(RuntimeError, match="read-only Store"):
        ro.upsert_vec_observations([("o1", vec)])
    with pytest.raises(RuntimeError, match="read-only Store"):
        ro.mark_embedded("observations", ["o1"], state="ok")


# --- degraded: sqlite-vec did not load --------------------------------------


def test_vec_read_raises_named_error_when_unavailable(no_vec_store):
    """Never a bare ``no such table: vec_observations``."""
    vec = np.eye(1, 768, dtype=np.float32)[0]
    with pytest.raises(VectorIndexUnavailableError, match="sqlite-vec"):
        no_vec_store._vec_obs_ids(vec, k=5)
    with pytest.raises(VectorIndexUnavailableError, match="sqlite-vec"):
        no_vec_store._vec_record_ids(vec, k=5)


def test_vec_write_is_a_noop_when_unavailable(no_vec_store, caplog):
    """The writer degrades quietly-ish: no crash, one warning, no rows."""
    caplog.set_level(logging.WARNING, logger=store_mod.__name__)
    _seed_observation(no_vec_store, "o1")
    vec = np.eye(1, 768, dtype=np.float32)[0]
    no_vec_store.upsert_vec_observations([("o1", vec)])
    no_vec_store.upsert_vec_records([("github:1", vec)])
    assert any("vector index unavailable" in r.getMessage() for r in caplog.records)


def test_backlog_bookkeeping_still_works_when_unavailable(no_vec_store):
    """``embedding_state`` is a plain column — it must not need the extension."""
    _seed_observation(no_vec_store, "u0")
    _seed_observation(no_vec_store, "u1")
    assert len(no_vec_store.select_unembedded("observations", limit=10)) == 2
    no_vec_store.mark_embedded("observations", ["u0"], state="ok")
    assert {r["obs_id"] for r in no_vec_store.select_unembedded("observations", limit=10)} == {"u1"}


# --- count_vec_rows: how much of the corpus the vector arm can actually reach -


def test_count_vec_rows_is_zero_on_a_fresh_cache(store):
    assert store.count_vec_rows("observations") == 0
    assert store.count_vec_rows("records") == 0


def test_count_vec_rows_counts_written_vectors(store):
    _seed_observation(store, "o0")
    _seed_observation(store, "o1")
    vecs = np.eye(2, 768, dtype=np.float32)
    store.upsert_vec_observations([("o0", vecs[0]), ("o1", vecs[1])])
    assert store.count_vec_rows("observations") == 2
    assert store.count_vec_rows("records") == 0


def test_count_vec_rows_counts_chunks_not_documents(store):
    """One row can hold several vectors — the embed worker writes ``id:N``
    chunk ids for a long body. This counts VECTORS, which is what the vector
    arm can retrieve; document-level progress lives in ``embedding_state``.
    """
    _seed_observation(store, "o0")
    vecs = np.eye(3, 768, dtype=np.float32)
    store.upsert_vec_observations(
        [("o0:0", vecs[0]), ("o0:1", vecs[1]), ("o0:2", vecs[2])]
    )
    assert store.count_vec_rows("observations") == 3


def test_count_vec_rows_counts_records(store):
    _seed_record(store, "github:1")
    vec = np.eye(1, 768, dtype=np.float32)[0]
    store.upsert_vec_records([("github:1", vec)])
    assert store.count_vec_rows("records") == 1


def test_count_vec_rows_rejects_unknown_kind(store):
    with pytest.raises(ValueError, match="unknown kind"):
        store.count_vec_rows("nope")


def test_count_vec_rows_raises_named_error_when_unavailable(no_vec_store):
    """A read of the vector arm, so it obeys the reads-RAISE half of the
    contract: a caller must be able to tell "no vector arm here" apart from
    "nothing embedded yet". Returning 0 would conflate exactly those two.
    """
    with pytest.raises(VectorIndexUnavailableError, match="sqlite-vec"):
        no_vec_store.count_vec_rows("observations")
    with pytest.raises(VectorIndexUnavailableError, match="sqlite-vec"):
        no_vec_store.count_vec_rows("records")


def test_count_vec_rows_rejects_unknown_kind_before_touching_the_extension(
    no_vec_store,
):
    """A bad ``kind`` is a programming error either way — it must not be
    masked by the environment-dependent availability check."""
    with pytest.raises(ValueError, match="unknown kind"):
        no_vec_store.count_vec_rows("nope")


def test_count_embedding_states_tallies_the_backlog(store):
    for i in range(4):
        _seed_observation(store, f"o{i}")
    store.mark_embedded("observations", ["o0", "o1"], state="ok")
    store.mark_embedded("observations", ["o2"], state="skip")
    counts = store.count_embedding_states("observations")
    assert counts["total"] == 4
    assert counts["ok"] == 2
    assert counts["skip"] == 1
    assert counts["error"] == 0
    assert counts["pending"] == 1


def test_count_embedding_states_zeroes_every_key_on_an_empty_table(store):
    """Absent states must still be present as 0 — a caller reading
    ``counts["pending"]`` should not have to guard for a missing key."""
    assert store.count_embedding_states("records") == {
        "total": 0,
        "pending": 0,
        "ok": 0,
        "skip": 0,
        "error": 0,
    }


def test_count_embedding_states_rejects_unknown_kind(store):
    with pytest.raises(ValueError, match="unknown kind"):
        store.count_embedding_states("nope")


def test_count_embedding_states_works_without_the_extension(no_vec_store):
    """``embedding_state`` is a plain column. How far the backfill got is
    exactly what an operator needs when the vector arm is broken, so this
    read must not depend on the native extension."""
    _seed_observation(no_vec_store, "o0")
    _seed_observation(no_vec_store, "o1")
    no_vec_store.mark_embedded("observations", ["o0"], state="ok")
    counts = no_vec_store.count_embedding_states("observations")
    assert counts["total"] == 2
    assert counts["ok"] == 1
    assert counts["pending"] == 1


def test_count_vec_rows_raises_when_the_vec_table_is_missing(tmp_path, monkeypatch):
    """Migrated WITHOUT the extension, then opened WITH it: the extension
    loads but ``vec_observations`` was never created. That is still "no
    vector arm on this cache", and it must not escape as a bare
    ``no such table``.
    """

    def _boom(conn):
        raise sqlite3.OperationalError("simulated sqlite-vec ABI mismatch")

    monkeypatch.setattr(store_mod, "_load_sqlite_vec", _boom)
    monkeypatch.setattr(store_mod, "_VEC_LOAD_WARNED", False)
    degraded = Store(db_path=tmp_path / "cache.db")
    degraded.migrate()
    assert degraded.vector_available is False
    degraded.close()

    monkeypatch.undo()
    healthy = Store(db_path=tmp_path / "cache.db", read_only=True)
    assert healthy.vector_available is True
    with pytest.raises(VectorIndexUnavailableError):
        healthy.count_vec_rows("observations")
