"""A compare-and-swap must not invent the value it compares against.

Round 3's M1. ``mark_embedded`` and ``commit_embed_batch`` guard every write
with ``AND src_hash IS ?``, reading the expected fingerprint out of the map the
worker built from ``select_unembedded``'s own row set. Both looked it up with
``expected.get(row_id)`` — so an id MISSING from the map did not fail the
lookup, it silently became ``None``, and the guard became ``src_hash IS NULL``.

That is worse than an unguarded write, and the direction is what makes it
worse. ``src_hash IS NULL``:

* does not match an ordinary row, so a correct row is skipped and
  ``cli.py``'s outcome accounting books it as a benign ``superseded`` — a
  silent no-op reported as a self-healing edit;
* DOES match a legacy row that never got a fingerprint (every pre-v4 row
  arrives that way, and ``_ensure_src_hash_columns`` deliberately does not
  backfill them). So the write lands, guarded against a condition the caller
  never asked for, on precisely the rows least able to prove they are
  unchanged.

One missing key therefore produces a wrong answer, not a refusal. A caller
that supplies a map and leaves an id out of it has a bug, and this is the
layer that can still say so.
"""

import numpy as np
import pytest

from aggregator.core.store import _VEC_DIM, Store


def _unit(seed=0):
    rng = np.random.default_rng(seed)
    v = rng.random(_VEC_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


@pytest.fixture
def cache(tmp_path):
    """Two rows: one fingerprinted, one legacy with a NULL ``src_hash``."""
    store = Store(db_path=tmp_path / "cache.db")
    store.migrate()
    c = store._c()
    c.execute(
        "INSERT INTO sessions(session_id, root_session_id, kind, first_ts, "
        "last_ts, jsonl_path) VALUES ('sid','sid','session','2026-01-01',"
        "'2026-01-01','/tmp/x.jsonl')"
    )
    c.execute(
        "INSERT INTO observations(obs_id, session_id, root_session_id, type, "
        "ts, body, src_hash) VALUES ('hashed','sid','sid','user',"
        "'2026-01-02','a body','deadbeef')"
    )
    c.execute(
        "INSERT INTO observations(obs_id, session_id, root_session_id, type, "
        "ts, body, src_hash) VALUES ('legacy','sid','sid','user',"
        "'2026-01-01','a pre-v4 body', NULL)"
    )
    c.commit()
    return store


def _state(store, obs_id):
    return store._c().execute(
        "SELECT embedding_state FROM observations WHERE obs_id = ?", (obs_id,)
    ).fetchone()["embedding_state"]


# --- mark_embedded ----------------------------------------------------------


def test_a_missing_expectation_is_refused_not_guessed(cache):
    """THE M1 REGRESSION on the row least able to defend itself.

    ``legacy`` has a NULL ``src_hash``, so the invented ``IS NULL`` guard
    MATCHED and the row was marked embedded against a claim nobody made.
    """
    with pytest.raises(KeyError) as excinfo:
        cache.mark_embedded("observations", ["legacy"], "skip", {})

    assert "legacy" in str(excinfo.value)
    assert _state(cache, "legacy") is None, "the row was written anyway"


def test_a_missing_expectation_is_refused_for_a_fingerprinted_row_too(cache):
    """The other half: silently skipped, then booked as a benign supersede."""
    with pytest.raises(KeyError):
        cache.mark_embedded("observations", ["hashed"], "skip", {})
    assert _state(cache, "hashed") is None


def test_a_partial_map_is_refused_before_any_row_is_written(cache):
    """All-or-nothing: the check runs before the first UPDATE.

    A caller with one id missing has a bug in how it built the map, and
    writing the ids that happened to be present would leave the store in a
    state neither the caller nor the ledger describes.
    """
    with pytest.raises(KeyError):
        cache.mark_embedded(
            "observations", ["hashed", "legacy"], "skip", {"hashed": "deadbeef"}
        )
    assert _state(cache, "hashed") is None
    assert _state(cache, "legacy") is None


def test_a_complete_map_still_marks_normally(cache):
    written = cache.mark_embedded(
        "observations", ["hashed", "legacy"], "skip",
        {"hashed": "deadbeef", "legacy": None},
    )
    assert sorted(written) == ["hashed", "legacy"]


def test_an_explicit_none_is_still_the_unguarded_form(cache):
    """``'error'`` makes no claim about content and must stay writable."""
    assert cache.mark_embedded("observations", ["legacy"], "error", None) == [
        "legacy"
    ]
    assert _state(cache, "legacy") == "error"


def test_a_stale_expectation_still_fails_the_swap_quietly(cache):
    """A row that MOVED is not a programming error — it is the normal race."""
    written = cache.mark_embedded(
        "observations", ["hashed"], "skip", {"hashed": "a-different-hash"}
    )
    assert written == []
    assert _state(cache, "hashed") is None


# --- commit_embed_batch's vector write --------------------------------------


def test_the_vector_write_refuses_a_missing_expectation_too(cache):
    """The same ``.get`` sat in the guarded INSERT ... WHERE EXISTS."""
    with pytest.raises(KeyError):
        cache.commit_embed_batch(
            "observations",
            vectors=[("legacy", "legacy", _unit())],
            ok_ids=[],
            skip_ids=[],
            error_ids=[],
            expected={},
        )
    assert cache.count_vec_rows("observations") == 0
