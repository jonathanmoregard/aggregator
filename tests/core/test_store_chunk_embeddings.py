"""Embeddings keyed ``(chunk_id, model)`` — criterion E, the reference design.

The shape this replaces treated the vector index as a singleton: one vector per
chunk, one model implied by the whole file, and a model change expressed as a
migration that DELETES everything and re-computes it. On this hardware that is
a multi-week outage (``docs/embedding-throughput.md``), which is why every
round so far spent its effort building consent gates in front of the deletion
rather than removing the need for one.

Four rules, from the research report's §7, all cheap and all skipped by most
implementations:

1. **Key on ``(chunk_id, model)``.** Two models then coexist as extra ROWS.
   A model change becomes a background job that finishes, not an outage that
   has to be survived.
2. **Hash the chunk content.** ``content_sha256`` answers "can this embedding
   be reused" across a re-chunk. Our corpus is chat transcripts that get
   APPENDED to rather than rewritten, so the leading chunks of an edited
   document are byte-identical and their vectors are still correct.
3. **Make the backfill a QUERY, not a ledger.** A LEFT JOIN against the
   embedding table is restart-safe by construction, because the store IS the
   ledger. There is no second table to get out of step with the first, and
   running the job twice equals running it once.
4. **Flip a pointer.** ``completed_at`` on the version row, so a half-built
   index for a NEW model is invisible while the old one keeps serving.

Plus the guard the report calls the highest-value line in the section: refuse
to serve on a model mismatch, and never lazily re-embed on read. One stale
vector in a top-k is not a small error — it is an apples-to-rulers comparison
that fails silently with plausible scores.
"""

import hashlib

import numpy as np
import pytest

from aggregator.core.store import _VEC_DIM, Store, VectorIndexUnavailableError


def _unit(seed=0):
    rng = np.random.default_rng(seed)
    v = rng.random(_VEC_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.delenv("AGGREGATOR_EMBED_BACKEND", raising=False)
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    c = s._c()
    c.execute(
        "INSERT INTO sessions(session_id, root_session_id, kind, first_ts, "
        "last_ts, jsonl_path) VALUES ('sid','sid','session','2026-01-01',"
        "'2026-01-01','/tmp/x.jsonl')"
    )
    for i in range(4):
        c.execute(
            "INSERT INTO observations(obs_id, session_id, root_session_id, "
            "type, ts, body) VALUES (?, 'sid','sid','user', ?, ?)",
            (f"o{i}", f"2026-01-0{i + 1}", f"body number {i}"),
        )
    c.commit()
    return s


# --- 1. the (chunk_id, model) key -------------------------------------------


def test_two_models_coexist_as_rows_not_as_a_migration(store):
    """THE WHOLE POINT. Writing model B must not disturb model A."""
    store.commit_embed_batch(
        "observations",
        vectors=[("o0", "o0", _unit(1))],
        ok_ids=["o0"],
        skip_ids=[],
        error_ids=[],
        expected={"o0": None},
    )
    store.record_chunk_embedding(
        "observations",
        chunk_id="o0",
        owner_id="o0",
        model="acme/model-b@768/chunk-x/norm-l2",
        content_sha256=_sha("body number 0"),
        embedding=_unit(2),
    )

    keys = {
        (r["chunk_id"], r["model"])
        for r in store._c().execute("SELECT chunk_id, model FROM chunk_embeddings")
    }
    assert len(keys) == 2, keys
    assert {k[0] for k in keys} == {"o0"}


def test_the_same_chunk_under_the_same_model_is_one_row(store):
    """``INSERT ... ON CONFLICT DO NOTHING``: running the job twice equals
    running it once."""
    for _ in range(3):
        store.commit_embed_batch(
            "observations",
            vectors=[("o0", "o0", _unit(1))],
            ok_ids=["o0"],
            skip_ids=[],
            error_ids=[],
            expected={"o0": None},
        )
    n = store._c().execute(
        "SELECT COUNT(*) FROM chunk_embeddings WHERE chunk_id = 'o0'"
    ).fetchone()[0]
    assert n == 1


def test_a_knn_read_only_sees_the_model_it_asked_for(store):
    """Never silently mix embedding spaces. A vector from another model in a
    top-k is not a small error; it is an apples-to-rulers comparison."""
    store.commit_embed_batch(
        "observations",
        vectors=[("o0", "o0", _unit(1))],
        ok_ids=["o0"],
        skip_ids=[],
        error_ids=[],
        expected={"o0": None},
    )
    store.record_chunk_embedding(
        "observations",
        chunk_id="o1",
        owner_id="o1",
        model="acme/model-b@768/chunk-x/norm-l2",
        content_sha256=None,
        embedding=_unit(1),  # identical vector, foreign space
    )

    hits = [i for i, _ in store._vec_obs_scored(_unit(1), k=10)]
    assert hits == ["o0"], hits


# --- 2. content_sha256 reuse -------------------------------------------------


def test_an_unchanged_chunk_carries_its_embedding_across_a_rechunk(store):
    """The append-only case: chunk 0 is byte-identical, so its vector holds."""
    vec = _unit(7)
    store.commit_embed_batch(
        "observations",
        vectors=[("o0", "o0:0", vec)],
        ok_ids=["o0"],
        skip_ids=[],
        error_ids=[],
        expected={"o0": None},
        hashes={"o0:0": _sha("paragraph one")},
    )

    reusable = store.reusable_chunk_vectors([_sha("paragraph one"), _sha("new")])

    assert set(reusable) == {_sha("paragraph one")}
    assert np.allclose(reusable[_sha("paragraph one")], vec, atol=1e-6)


def test_reuse_is_scoped_to_the_model(store):
    """A hash match under a DIFFERENT model is not a reusable embedding —
    that is exactly the silent mixing this whole key exists to prevent."""
    store.record_chunk_embedding(
        "observations",
        chunk_id="o0:0",
        owner_id="o0",
        model="acme/model-b@768/chunk-x/norm-l2",
        content_sha256=_sha("paragraph one"),
        embedding=_unit(7),
    )
    assert store.reusable_chunk_vectors([_sha("paragraph one")]) == {}


def test_a_chunk_written_without_a_hash_is_never_reused(store):
    """NULL means "we cannot prove the content matched", and the honest
    answer to an unprovable reuse is to embed it again."""
    store.commit_embed_batch(
        "observations",
        vectors=[("o0", "o0", _unit(1))],
        ok_ids=["o0"],
        skip_ids=[],
        error_ids=[],
        expected={"o0": None},
    )
    assert store.reusable_chunk_vectors([_sha("body number 0")]) == {}


# --- 3. the backfill is a LEFT JOIN, not a ledger ----------------------------


def test_the_backlog_is_derived_from_the_embedding_table(store):
    """No separate ledger to fall out of step: the store IS the ledger."""
    assert len(store.select_unembedded("observations", limit=10)) == 4
    store.commit_embed_batch(
        "observations",
        vectors=[("o0", "o0", _unit(1))],
        ok_ids=["o0"],
        skip_ids=[],
        error_ids=[],
        expected={"o0": None},
    )
    remaining = {r["obs_id"] for r in store.select_unembedded("observations", limit=10)}
    assert remaining == {"o1", "o2", "o3"}


def test_a_new_model_puts_the_whole_corpus_back_in_the_backlog(store):
    """Which is what makes a model change a background JOB rather than an
    outage: the old vectors are untouched and still keyed to the old model."""
    for i in range(4):
        store.record_chunk_embedding(
            "observations",
            chunk_id=f"o{i}",
            owner_id=f"o{i}",
            model="acme/model-b@768/chunk-x/norm-l2",
            content_sha256=None,
            embedding=_unit(i),
        )
    # Nothing exists under THIS build's model, so all four are still due.
    assert len(store.select_unembedded("observations", limit=10)) == 4


def test_skipped_and_errored_rows_still_leave_the_backlog(store):
    """``embedding_state`` keeps the two NEGATIVE outcomes and nothing else.

    'skip' (nothing embeddable) and 'error' (set aside by the poison ledger)
    are facts about the row that no ``chunk_embeddings`` row could record,
    because in both cases there is no embedding to record.
    """
    store.mark_embedded("observations", ["o0"], "skip")
    store.mark_embedded("observations", ["o1"], "error")
    remaining = {r["obs_id"] for r in store.select_unembedded("observations", limit=10)}
    assert remaining == {"o2", "o3"}


def test_a_stale_ok_mark_does_not_hide_a_row_from_a_new_model(store):
    """THE ROUND-4 DEFECT CLASS, structurally. A column that says "embedded"
    cannot say WHICH MODEL embedded it, so it is wrong the moment the model
    moves — and it is equally wrong when a rebuild resets it."""
    store._c().execute(
        "UPDATE observations SET embedding_state = 'ok' WHERE obs_id = 'o0'"
    )
    store._c().commit()
    due = {r["obs_id"] for r in store.select_unembedded("observations", limit=10)}
    assert "o0" in due, "a bare 'ok' column still held a row out of the backlog"


# --- 4. completed_at: a half-built index is invisible ------------------------


def test_the_first_index_serves_while_it_builds(store):
    """The bootstrap exception, named. There is no previous index to fall back
    to, so refusing here would mean no vector arm at all for weeks — and the
    watermark already tells callers how far the backfill got."""
    store.commit_embed_batch(
        "observations",
        vectors=[("o0", "o0", _unit(1))],
        ok_ids=["o0"],
        skip_ids=[],
        error_ids=[],
        expected={"o0": None},
    )
    assert store.serving_embedding_model() is not None
    assert store.has_embedded_rows("observations")


def test_a_second_model_is_invisible_until_it_completes(tmp_path, monkeypatch):
    """The real job of the pointer: a model change must not serve a partial
    index, because a half-filled space answers with plausible scores."""
    monkeypatch.delenv("AGGREGATOR_EMBED_BACKEND", raising=False)
    db = tmp_path / "cache.db"
    first = Store(db_path=db)
    first.migrate()
    first.mark_embedding_version_complete()
    first.close()

    monkeypatch.setenv("AGGREGATOR_EMBED_BACKEND", "gguf")
    second = Store(db_path=db)
    second.migrate()
    # Half a batch of the new model's index exists. It must not be served.
    second.record_chunk_embedding(
        "observations", chunk_id="o0", owner_id="o0", embedding=_unit(3)
    )

    assert second.serving_embedding_model() is None
    with pytest.raises(VectorIndexUnavailableError) as e:
        second._vec_obs_scored(_unit(0), k=5)
    assert "still being built" in str(e.value)


def test_the_old_models_vectors_survive_the_switch(tmp_path, monkeypatch):
    """A model change is a background job, not an outage. Nothing is deleted.

    This is the failure every consent gate in this file was built to prevent,
    removed at the root instead: with the model in the key there is no reason
    to delete anything to make room.
    """
    monkeypatch.delenv("AGGREGATOR_EMBED_BACKEND", raising=False)
    db = tmp_path / "cache.db"
    first = Store(db_path=db)
    first.migrate()
    old_model = first.embedding_model
    first.record_chunk_embedding(
        "observations", chunk_id="o0", owner_id="o0", embedding=_unit(1)
    )
    first.close()

    monkeypatch.setenv("AGGREGATOR_EMBED_BACKEND", "gguf")
    second = Store(db_path=db)
    second.migrate()

    kept = second._c().execute(
        "SELECT COUNT(*) FROM chunk_embeddings WHERE model = ?", (old_model,)
    ).fetchone()[0]
    assert kept == 1, "the previous model's index was destroyed by a re-migrate"
    assert second.vector_quarantine is None


def test_completing_the_second_model_makes_it_visible(tmp_path, monkeypatch):
    monkeypatch.delenv("AGGREGATOR_EMBED_BACKEND", raising=False)
    db = tmp_path / "cache.db"
    first = Store(db_path=db)
    first.migrate()
    first.mark_embedding_version_complete()
    first.close()

    monkeypatch.setenv("AGGREGATOR_EMBED_BACKEND", "gguf")
    second = Store(db_path=db)
    second.migrate()
    second.mark_embedding_version_complete()

    assert second.serving_embedding_model() is not None


def test_completion_is_refused_while_the_backlog_still_has_rows(store):
    """A pointer that can be flipped over a half-built index is not a pointer,
    it is a comment."""
    with pytest.raises(RuntimeError) as e:
        store.mark_embedding_version_complete()
    assert "4" in str(e.value)
    assert store.embedding_version_state()["completed_at"] is None
