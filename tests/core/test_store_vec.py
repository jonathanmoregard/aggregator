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


def _record_embedding(s, kind, chunk_id, owner_id=None):
    """Stage an index row WITHOUT a vector, for a cache whose extension broke.

    The real writer puts both halves down together (``record_chunk_embedding``);
    this stages the history a broken-extension cache already carries, which is
    the state ``count_embedding_states`` most has to be able to report on.
    """
    c = s._c()
    c.execute(
        "INSERT INTO chunk_embeddings(chunk_id, model, kind, owner_id, dim, "
        "content_sha256, created_at) VALUES (?, ?, ?, ?, 768, NULL, '2026-01-01')",
        (chunk_id, s.embedding_model, kind, owner_id or chunk_id),
    )
    c.commit()


def _force_state(s, kind, ids, state):
    """Write ``embedding_state`` directly, bypassing ``mark_embedded``.

    For the cache that WAS embedded and whose extension broke afterwards — a
    real and unremarkable state, and the one an operator most needs reported.
    ``mark_embedded(state='ok')`` deliberately refuses without the extension,
    because as an API call it asserts a vector was just written; it is the
    wrong tool for staging history that already happened.
    """
    col = "obs_id" if kind == "observations" else "stable_id"
    c = s._c()
    c.executemany(
        f"UPDATE {kind} SET embedding_state = ? WHERE {col} = ?",  # noqa: S608 - test literals
        [(state, i) for i in ids],
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
    hits = [i for i, _ in store._vec_obs_scored(query, k=5)]
    assert hits[0] == "o1"


def test_knn_returns_topk_ordered(store):
    for i in range(5):
        _seed_observation(store, f"o{i}")
    vecs = np.eye(5, 768, dtype=np.float32)  # 5 orthogonal unit vectors
    store.upsert_vec_observations([(f"o{i}", vecs[i]) for i in range(5)])
    hits = [i for i, _ in store._vec_obs_scored(vecs[2], k=3)]
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
    assert [i for i, _ in store._vec_record_scored(vecs[1], k=1)] == ["github:2"]


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


def test_a_vector_is_what_takes_a_row_out_of_the_backlog(store):
    """``'ok'`` ALONE NO LONGER MEANS EMBEDDED, and that is criterion E.

    The backlog is a LEFT JOIN against ``chunk_embeddings``, so what removes a
    row is the existence of an embedding under this model — not a column. A
    column cannot say WHICH model embedded the row, which makes it wrong in
    both directions: it hides the whole corpus from a new model's backfill, and
    a source rebuild that resets it hands ~483k already-embedded rows back.
    """
    _seed_observation(store, "u0")
    store.mark_embedded("observations", ["u0"], state="ok")
    assert any(
        r["obs_id"] == "u0" for r in store.select_unembedded("observations", limit=10)
    ), "a bare 'ok' column took a row with no vector out of the backlog"

    store.upsert_vec_observations([("u0", np.eye(1, 768, dtype=np.float32)[0])])
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
        no_vec_store._vec_obs_scored(vec, k=5)
    with pytest.raises(VectorIndexUnavailableError, match="sqlite-vec"):
        no_vec_store._vec_record_scored(vec, k=5)


def test_vec_write_is_a_noop_when_unavailable(no_vec_store, caplog):
    """The writer degrades quietly-ish: no crash, one warning, no rows."""
    caplog.set_level(logging.WARNING, logger=store_mod.__name__)
    _seed_observation(no_vec_store, "o1")
    vec = np.eye(1, 768, dtype=np.float32)[0]
    no_vec_store.upsert_vec_observations([("o1", vec)])
    no_vec_store.upsert_vec_records([("github:1", vec)])
    assert any("vector index unavailable" in r.getMessage() for r in caplog.records)


def test_backlog_bookkeeping_still_works_when_unavailable(no_vec_store):
    """``embedding_state`` is a plain column — it must not need the extension.

    Marked ``'skip'`` rather than ``'ok'``: ``'ok'`` asserts that a vector was
    written, which is exactly what cannot have happened here, and it now
    refuses. ``'skip'`` and ``'error'`` are the two states that remain TRUE
    with no extension, and draining the backlog past them is the property this
    test is about.
    """
    _seed_observation(no_vec_store, "u0")
    _seed_observation(no_vec_store, "u1")
    assert len(no_vec_store.select_unembedded("observations", limit=10)) == 2
    no_vec_store.mark_embedded("observations", ["u0"], state="skip")
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
    # ``ok`` is counted off the embedding index, so the tally needs vectors
    # rather than a column value — see the note in ``count_embedding_states``.
    store.upsert_vec_observations(
        [(f"o{i}", np.eye(1, 768, dtype=np.float32)[0]) for i in range(2)]
    )
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
    """How far the backfill got is exactly what an operator needs when the
    vector arm is broken, so this read must not depend on the native
    extension. ``chunk_embeddings`` is an ordinary table for that reason —
    the index of record is plain SQL even though the vectors are not."""
    _seed_observation(no_vec_store, "o0")
    _seed_observation(no_vec_store, "o1")
    _record_embedding(no_vec_store, "observations", "o0")
    counts = no_vec_store.count_embedding_states("observations")
    assert counts["total"] == 2
    assert counts["ok"] == 1
    assert counts["pending"] == 1


# --- has_embedded_rows: the routing probe, and why it is not count_vec_rows --
#
# MEASURED, NOT ASSUMED. ``SELECT COUNT(*)`` on a vec0 virtual table is O(n):
# 4 ms at 20k vectors, 13 ms at 100k, 70-83 ms at 400k — and 400k is the size
# of the live cache. Retrieval asks "is there anything for the vector arm to
# find" on every text query, on up to two ontologies, so a linear probe puts a
# tenth of a second of pure overhead on the recall path and grows with the
# corpus. ``embedding_state`` is a plain indexed column and answers the same
# question in microseconds. count_vec_rows keeps its exact count for
# capabilities, where being slow and precise is the right trade.


def test_has_embedded_rows_is_false_on_a_fresh_cache(store):
    assert store.has_embedded_rows("observations") is False
    assert store.has_embedded_rows("records") is False


def test_has_embedded_rows_is_true_once_a_row_is_embedded(store):
    _seed_observation(store, "o0")
    store.upsert_vec_observations([("o0", np.eye(1, 768, dtype=np.float32)[0])])
    store.mark_embedded("observations", ["o0"], state="ok")
    assert store.has_embedded_rows("observations") is True


def test_has_embedded_rows_ignores_skipped_and_failed_rows(store):
    """'skip' means the body had nothing embeddable and 'error' means the
    attempt failed — neither put a vector in the index, so neither gives the
    vector arm anything to return."""
    _seed_observation(store, "o0")
    _seed_observation(store, "o1")
    store.mark_embedded("observations", ["o0"], state="skip")
    store.mark_embedded("observations", ["o1"], state="error")
    assert store.has_embedded_rows("observations") is False


def test_has_embedded_rows_is_per_ontology(store):
    _seed_record(store, "github:1")
    store.upsert_vec_records([("github:1", np.eye(1, 768, dtype=np.float32)[0])])
    store.mark_embedded("records", ["github:1"], state="ok")
    assert store.has_embedded_rows("records") is True
    assert store.has_embedded_rows("observations") is False


def test_has_embedded_rows_rejects_unknown_kind(store):
    with pytest.raises(ValueError, match="unknown kind"):
        store.has_embedded_rows("nope")


def test_has_embedded_rows_raises_when_the_extension_is_missing(no_vec_store):
    """Routing must not engage the vector arm on a machine that cannot run
    it, and the caller has to be told which of the two it is."""
    with pytest.raises(VectorIndexUnavailableError, match="sqlite-vec"):
        no_vec_store.has_embedded_rows("observations")


def test_has_embedded_rows_uses_the_indexed_column_not_a_vec_table_scan(store):
    """Pins the implementation choice, since the reason for it is a
    performance property no assertion about the return value can see."""
    _seed_observation(store, "o0")
    store.mark_embedded("observations", ["o0"], state="ok")
    plan = store._c().execute(
        "EXPLAIN QUERY PLAN SELECT 1 FROM observations "
        "WHERE embedding_state = 'ok' LIMIT 1"
    ).fetchall()
    rendered = " ".join(str(r["detail"]) for r in plan)
    assert "obs_embedding_state" in rendered, rendered
    assert "vec_observations" not in rendered


# --- capabilities()["vector_index"]: three situations, three answers ---------
#
# "backfill in progress", "vector arm unavailable" and "nothing embedded yet"
# are three different facts with three different remedies (wait / fix the
# install / start the worker). Any output that renders two of them identically
# is a bug, so each gets its own test.


def test_capabilities_vector_index_reports_nothing_embedded_yet(store):
    for i in range(3):
        _seed_observation(store, f"o{i}")
    vi = store.capabilities()["vector_index"]
    assert vi["available"] is True
    assert vi["state"] == "not_started"
    assert vi["observations"]["pending"] == 3
    assert vi["observations"]["vectors"] == 0


def test_capabilities_vector_index_reports_backfill_in_progress(store):
    for i in range(3):
        _seed_observation(store, f"o{i}")
    vec = np.eye(1, 768, dtype=np.float32)[0]
    store.upsert_vec_observations([("o0", vec)])
    store.mark_embedded("observations", ["o0"], state="ok")
    vi = store.capabilities()["vector_index"]
    assert vi["available"] is True
    assert vi["state"] == "backfilling"
    assert vi["observations"]["ok"] == 1
    assert vi["observations"]["pending"] == 2
    assert vi["observations"]["vectors"] == 1


def test_capabilities_vector_index_reports_unavailable_arm(no_vec_store):
    """The distinguishing test: the corpus here is IDENTICAL to the
    nothing-embedded-yet case, and the two must not read the same."""
    for i in range(3):
        _seed_observation(no_vec_store, f"o{i}")
    vi = no_vec_store.capabilities()["vector_index"]
    assert vi["available"] is False
    assert vi["state"] == "unavailable"
    assert vi["reason"]
    assert "sqlite-vec" in vi["reason"]
    # The backlog is still reportable — it is plain column arithmetic.
    assert vi["observations"]["pending"] == 3
    # But the vector count is unknown, NOT zero.
    assert vi["observations"]["vectors"] is None


def test_capabilities_vector_index_three_states_are_all_distinguishable(
    tmp_path, monkeypatch
):
    """Belt and braces over the three tests above: render each situation and
    assert no two of them produce equal output."""
    rendered = []

    fresh = Store(db_path=tmp_path / "a.db")
    fresh.migrate()
    _seed_observation(fresh, "o0")
    rendered.append(fresh.capabilities()["vector_index"])

    partial = Store(db_path=tmp_path / "b.db")
    partial.migrate()
    _seed_observation(partial, "o0")
    _seed_observation(partial, "o1")
    partial.upsert_vec_observations([("o0", np.eye(1, 768, dtype=np.float32)[0])])
    partial.mark_embedded("observations", ["o0"], state="ok")
    rendered.append(partial.capabilities()["vector_index"])

    def _boom(conn):
        raise sqlite3.OperationalError("simulated sqlite-vec ABI mismatch")

    monkeypatch.setattr(store_mod, "_load_sqlite_vec", _boom)
    monkeypatch.setattr(store_mod, "_VEC_LOAD_WARNED", False)
    broken = Store(db_path=tmp_path / "c.db")
    broken.migrate()
    _seed_observation(broken, "o0")
    rendered.append(broken.capabilities()["vector_index"])

    states = [vi["state"] for vi in rendered]
    assert len(set(states)) == 3, f"states collapsed: {states}"


def test_capabilities_vector_index_reports_complete_when_backlog_is_drained(store):
    _seed_observation(store, "o0")
    store.upsert_vec_observations([("o0", np.eye(1, 768, dtype=np.float32)[0])])
    store.mark_embedded("observations", ["o0"], state="ok")
    vi = store.capabilities()["vector_index"]
    assert vi["state"] == "complete"


def test_capabilities_vector_index_reports_empty_corpus(store):
    """Nothing to embed is not the same as nothing embedded — a fresh cache
    with no rows must not read as an idle backfill worker."""
    vi = store.capabilities()["vector_index"]
    assert vi["available"] is True
    assert vi["state"] == "empty"


def test_capabilities_vector_index_covers_both_ontologies(store):
    _seed_observation(store, "o0")
    _seed_record(store, "github:1")
    vi = store.capabilities()["vector_index"]
    assert vi["observations"]["total"] == 1
    assert vi["records"]["total"] == 1


def test_capabilities_still_works_when_the_vector_arm_is_broken(no_vec_store):
    """Regression guard: capabilities is the tool a caller reaches for WHEN
    something is wrong. It must not be the thing that breaks."""
    _seed_observation(no_vec_store, "o0")
    caps = no_vec_store.capabilities()
    assert caps["schema_version"] == store_mod.SCHEMA_VERSION
    assert caps["counts"]["observations"] == 1


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


# --- the API refuses the unsafe composition, rather than trusting its caller -
#
# ``upsert_vec_*`` no-ops when sqlite-vec is absent, and ``mark_embedded``
# was callable regardless, so the obvious composition —
#
#     store.upsert_vec_observations(vectors)   # silently writes nothing
#     store.mark_embedded(kind, ids, "ok")     # happily advances anyway
#
# marked rows as embedded that have no vector and that nothing will ever come
# back for: ``select_unembedded`` only sees NULL. Today the only thing standing
# between that and the corpus is a guard in ``cli._cmd_embed``, i.e. one
# caller's discipline. A second caller — a script, a test, a future worker —
# gets no such protection.
#
# 'ok' is the only state that ASSERTS a vector exists. 'skip' (nothing to
# embed) and 'error' (it failed) are both true without one, so they stay
# available: the backlog bookkeeping is a plain column and must keep working
# with the extension broken.


def test_marking_a_row_ok_refuses_when_no_vector_could_have_been_written(
    no_vec_store,
):
    _seed_observation(no_vec_store, "u0")
    with pytest.raises(VectorIndexUnavailableError):
        no_vec_store.mark_embedded("observations", ["u0"], state="ok")
    assert {
        r["obs_id"] for r in no_vec_store.select_unembedded("observations")
    } == {"u0"}


def test_marking_records_ok_refuses_on_the_same_terms(no_vec_store):
    _seed_record(no_vec_store, "github:1")
    with pytest.raises(VectorIndexUnavailableError):
        no_vec_store.mark_embedded("records", ["github:1"], state="ok")


def test_skip_and_error_are_still_writable_without_the_extension(no_vec_store):
    """Both are TRUE with no vector, and the backlog has to drain past them."""
    _seed_observation(no_vec_store, "u0")
    _seed_observation(no_vec_store, "u1")
    no_vec_store.mark_embedded("observations", ["u0"], state="skip")
    no_vec_store.mark_embedded("observations", ["u1"], state="error")
    assert no_vec_store.select_unembedded("observations") == []


def test_marking_an_empty_id_list_ok_is_still_a_noop(no_vec_store):
    """A healthy run reaches this on every all-skip batch; it asserts nothing."""
    no_vec_store.mark_embedded("observations", [], state="ok")


def test_vec_writes_report_how_many_vectors_they_wrote(store):
    _seed_observation(store, "o1")
    _seed_observation(store, "o2")
    vecs = np.eye(2, 768, dtype=np.float32)
    assert store.upsert_vec_observations([("o1", vecs[0]), ("o2", vecs[1])]) == 2


def test_vec_writes_report_zero_when_they_were_discarded(no_vec_store):
    """The count is what makes the no-op checkable from the outside."""
    vec = np.eye(1, 768, dtype=np.float32)[0]
    assert no_vec_store.upsert_vec_observations([("o1", vec)]) == 0
    assert no_vec_store.upsert_vec_records([("github:1", vec)]) == 0
