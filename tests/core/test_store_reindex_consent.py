"""Who may delete a month of computed vectors, and how they say so.

Round 2 closed the first half of this: a provenance mismatch stopped
DISCARDING the index and started REFUSING it, so a stray
``AGGREGATOR_EMBED_BACKEND`` could no longer turn ``aggregator query`` into a
demolition. Deleting became opt-in, behind ``AGGREGATOR_VECTOR_REINDEX=1``.

Round 3 found the opt-in itself was the wrong shape, and two reviewers flagged
it independently:

* **Scope.** ``migrate()`` runs on EVERY subcommand, so the consent was
  process-wide. Export ``AGGREGATOR_VECTOR_REINDEX=1`` once — for the rebuild
  you genuinely meant — and every later command in that shell carried it,
  including reads. A ``aggregator query`` with a stray backend var then
  deleted 25-30 days of CPU, which is precisely the failure round 2 fixed,
  reintroduced through the fix's own escape hatch.
* **Ergonomics.** A sticky env var is honoured silently by everything, with no
  prompt and no count preview — while ``ingest --rebuild``'s row drop, an
  operation of comparable cost, demands a ``y`` on stdin. The loud path was the
  hard one.

The rule these pin: the reindex is an EMBED-side maintenance action. Only the
embed command can authorise it, it is authorised per invocation, and nothing
in the environment can authorise it at all. S1 is untouched — an unconsented
mismatch still refuses, still deletes nothing, and still switches the arm off.
"""

import sqlite3

import numpy as np
import pytest
import sqlite_vec

from aggregator.core.store import _VEC_DIM, Store, VectorIndexUnavailableError


def _unit(dim, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _raw_vec_conn(db):
    c = sqlite3.connect(db)
    c.enable_load_extension(True)
    sqlite_vec.load(c)
    c.enable_load_extension(False)
    return c


def _seed_obs(store, obs_id, body="hello"):
    c = store._c()
    c.execute(
        "INSERT OR IGNORE INTO sessions(session_id, root_session_id, kind, "
        "first_ts, last_ts, jsonl_path, origin) VALUES ('sid','sid','session',"
        "'2026-01-01','2026-01-01','/tmp/x.jsonl','claude-code')"
    )
    c.execute(
        "INSERT OR REPLACE INTO observations(obs_id, session_id, root_session_id,"
        " type, ts, body) VALUES (?,'sid','sid','user','2026-01-01',?)",
        (obs_id, body),
    )
    c.commit()


def _vector_count(db):
    raw = _raw_vec_conn(db)
    n = raw.execute("SELECT COUNT(*) FROM vec_observations").fetchone()[0]
    raw.close()
    return n


@pytest.fixture
def populated(tmp_path, monkeypatch):
    """A cache holding five real vectors, stamped by THIS build."""
    db = tmp_path / "cache.db"
    monkeypatch.delenv("AGGREGATOR_EMBED_BACKEND", raising=False)
    store = Store(db_path=db)
    store.migrate()
    for i in range(5):
        _seed_obs(store, f"o{i}")
    store.upsert_vec_observations([(f"o{i}", _unit(_VEC_DIM, seed=i)) for i in range(5)])
    store.mark_embedded("observations", [f"o{i}" for i in range(5)], "ok", None)
    store.close()
    assert _vector_count(db) == 5
    # Now the shell disagrees with what wrote the index.
    monkeypatch.setenv("AGGREGATOR_EMBED_BACKEND", "gguf")
    return db


def test_environment_alone_cannot_authorise_a_reindex(populated, monkeypatch):
    """THE ROUND-3 H1 REGRESSION.

    ``AGGREGATOR_VECTOR_REINDEX=1`` left over in a shell used to be honoured
    by ``migrate()``, which every subcommand calls — so the next unrelated
    command deleted the index. No environment variable is consent any more:
    the default ``migrate()`` refuses, whatever is exported.
    """
    monkeypatch.setenv("AGGREGATOR_VECTOR_REINDEX", "1")
    store = Store(db_path=populated)
    store.migrate()
    assert _vector_count(populated) == 5, (
        "a plain migrate() deleted computed vectors on the strength of an "
        "environment variable"
    )
    with pytest.raises(VectorIndexUnavailableError):
        store._vec_obs_ids(_unit(_VEC_DIM), k=3)


def test_migrate_default_keeps_every_vector(populated):
    """S1's guarantee, reached without a quarantine at all.

    Criterion E moved the protection from "refuse the index" to "key the
    index". A plain ``migrate()`` under a stray backend variable now ADOPTS
    the cache — the existing vectors carry their own model and are in no
    danger from being looked at — and the refusal happens at read time, where
    it can be specific about which model is missing. What has not changed, and
    is the whole point, is that nothing is deleted.
    """
    store = Store(db_path=populated)
    store.migrate()
    assert _vector_count(populated) == 5
    with pytest.raises(VectorIndexUnavailableError) as e:
        store.has_embedded_rows("observations")
    assert "NOTHING WAS DELETED" in str(e.value)


def test_the_new_models_backlog_is_the_whole_corpus(populated):
    """A model change is a background JOB: everything is due under the new
    key, and the old vectors are not touched while it runs."""
    store = Store(db_path=populated)
    store.migrate()
    assert len(store.select_unembedded("observations", limit=10)) == 5
    assert _vector_count(populated) == 5


def test_the_explicit_parameter_is_what_deletes(populated):
    """Consent is a per-call argument, not ambient state."""
    store = Store(db_path=populated)
    store.migrate(allow_vector_reindex=True)
    assert _vector_count(populated) == 0
    assert store.vector_quarantine is None
    assert len(store.select_unembedded("observations", limit=10)) == 5


def test_preview_reports_what_a_reindex_would_cost(populated):
    """The count the operator is shown BEFORE being asked to confirm."""
    store = Store(db_path=populated)
    assert store.vector_reindex_preview() == (5, 5)


def test_preview_is_safe_on_a_cache_that_never_migrated(tmp_path):
    """It runs before ``migrate()``, so no table may be assumed to exist."""
    store = Store(db_path=tmp_path / "fresh.db")
    assert store.vector_reindex_preview() == (0, 0)


def test_refusal_names_the_two_opposite_fixes(populated):
    """They ARE opposite, so the message has to name both and say which is
    which: unset a stray variable, or build the new index deliberately."""
    store = Store(db_path=populated)
    store.migrate()
    with pytest.raises(VectorIndexUnavailableError) as e:
        store.has_embedded_rows("observations")
    reason = str(e.value)
    assert "unset AGGREGATOR_EMBED_BACKEND" in reason
    assert "embed --catchup" in reason
    assert "AGGREGATOR_VECTOR_REINDEX" not in reason, (
        "the refusal still tells the operator to export a variable nothing reads"
    )
