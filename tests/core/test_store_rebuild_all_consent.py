"""``rebuild_all()`` is the THIRD wholesale-delete path, and it had no gate.

Round 4's seeded hypothesis, confirmed: rounds 1-3 spent three iterations
making ``migrate()`` the only thing that may destroy computed vectors —
``AGGREGATOR_VECTOR_REINDEX`` deleted, replaced by a per-call
``allow_vector_reindex=``, fronted by a preview and a ``y`` on stdin — and
``rebuild_all()`` ran ``_VEC_DROP_ALL`` unconditionally the whole time.
``scripts/reingest_v2.py`` calls it on line 32 as its first act.

So ``migrate()``'s claim to be the only wholesale delete was false, and the
consent machinery in front of it was a fence with a gap in it. The cost is the
same on either path: on this hardware the last measured backfill rate makes a
full re-embed of this corpus a multi-week operation (see
``docs/embedding-throughput.md``), and no script that says "re-ingest" is
asking for that.

The gate is the SAME gate, deliberately — same parameter name, same refusal
text, same command named as the way through. A second vocabulary for the same
decision is how the first gap opened.
"""

import sqlite3

import numpy as np
import pytest
import sqlite_vec

from aggregator.core.store import _VEC_DIM, Store, VectorReindexNotConsentedError


def _unit(dim, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


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
    raw = sqlite3.connect(db)
    raw.enable_load_extension(True)
    sqlite_vec.load(raw)
    raw.enable_load_extension(False)
    n = raw.execute("SELECT COUNT(*) FROM vec_observations").fetchone()[0]
    raw.close()
    return n


@pytest.fixture
def populated(tmp_path, monkeypatch):
    """A cache holding five real vectors written by THIS build."""
    monkeypatch.delenv("AGGREGATOR_EMBED_BACKEND", raising=False)
    db = tmp_path / "cache.db"
    store = Store(db_path=db)
    store.migrate()
    for i in range(5):
        _seed_obs(store, f"o{i}")
    store.upsert_vec_observations(
        [(f"o{i}", _unit(_VEC_DIM, seed=i)) for i in range(5)]
    )
    store.close()
    assert _vector_count(db) == 5
    return db


def test_rebuild_all_refuses_to_destroy_computed_vectors_unasked(populated):
    """THE ROUND-4 REGRESSION: the gap in the fence.

    Nothing about "re-ingest the corpus" implies consent to discard the
    vector index — the rows come back in minutes, the vectors do not.
    """
    store = Store(db_path=populated)
    with pytest.raises(VectorReindexNotConsentedError) as e:
        store.rebuild_all()
    assert _vector_count(populated) == 5, "vectors were destroyed by the refusal"
    assert "NOTHING WAS DELETED" in str(e.value)


def test_the_refusal_names_the_way_through(populated):
    """A refusal an operator cannot act on is an outage, not a guard."""
    store = Store(db_path=populated)
    with pytest.raises(VectorReindexNotConsentedError) as e:
        store.rebuild_all()
    assert "allow_vector_reindex" in str(e.value)
    assert "5 vector(s)" in str(e.value)


def test_explicit_consent_rebuilds_everything(populated):
    """The parameter is spent by the call that passes it, like migrate()'s."""
    store = Store(db_path=populated)
    store.rebuild_all(allow_vector_reindex=True)
    assert _vector_count(populated) == 0
    assert store.schema_version() > 0


def test_a_cold_index_needs_no_consent(tmp_path, monkeypatch):
    """Nothing computed exists, so nothing is at stake. Prompting here is how
    a real prompt stops working."""
    monkeypatch.delenv("AGGREGATOR_EMBED_BACKEND", raising=False)
    store = Store(db_path=tmp_path / "cold.db")
    store.migrate()
    store.rebuild_all()
    assert store.schema_version() > 0


def test_a_cache_without_the_extension_needs_no_consent(tmp_path, monkeypatch):
    """No sqlite-vec means no vec tables were ever written, so there is
    nothing to weigh and the refusal would be pure obstruction."""
    monkeypatch.delenv("AGGREGATOR_EMBED_BACKEND", raising=False)
    monkeypatch.setattr(
        "aggregator.core.store._try_load_sqlite_vec", lambda _c: False
    )
    store = Store(db_path=tmp_path / "novec.db")
    store.migrate()
    store.rebuild_all()
    assert store.schema_version() > 0
