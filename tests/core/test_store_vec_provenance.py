"""``migrate()`` must not adopt vector state it cannot vouch for.

Before v5 stamped provenance, ``migrate()`` probed only for EXISTENCE —
``PRAGMA table_info`` for the columns, ``CREATE VIRTUAL TABLE IF NOT
EXISTS`` for the vec tables — and nothing recorded which model or which
dimension produced the vectors already on disk.

That is not a hypothetical on this project. The live cache carries the
``vec_*`` tables and both ``embedding_state`` columns *while stamped v4*,
because an abandoned 2026-08-08 branch numbered its schema 4 as well and
evidently ran. Three distinct harms were reproduced against the unstamped
code, and each test below was watched to fail:

* **wrong dimension** — a foreign ``float[1024]`` table survives ``IF NOT
  EXISTS`` untouched, ``migrate()`` reports success and stamps v5, and then
  every write AND every KNN read raises ``Dimension mismatch``. On a timer
  that is a crash loop, not a one-off.
* **stale vectors, right dimension** — adopted wholesale and served as
  current, with nothing on disk recording which model wrote them.
* **pre-set ``embedding_state``** — rows claiming ``'ok'`` with an empty
  index drain the backlog to nothing, make ``has_embedded_rows`` true, and
  make ``vector_index_state`` report ``complete`` over ``vectors: 0``. The
  index is empty, permanently, and every count says it is finished.

The rule this pins: vector state is adopted only when the stamp on disk
matches the model and dimension this build would produce. Otherwise it is
discarded and the rows go back into the backlog to be re-embedded — which
costs CPU and is the only outcome that cannot serve a wrong answer.
"""

import sqlite3

import numpy as np
import pytest
import sqlite_vec

from aggregator.core.store import _VEC_DIM, Store, vector_provenance


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


def _plant_foreign_vec_table(db, dim, n=3, table="vec_observations", col="obs_id"):
    """Create a vec table this code did not write, with ``n`` rows in it."""
    c = _raw_vec_conn(db)
    c.execute(
        f"CREATE VIRTUAL TABLE {table} USING vec0("  # noqa: S608 - test literals
        f"{col} TEXT PRIMARY KEY, embedding float[{dim}])"
    )
    c.executemany(
        f"INSERT INTO {table}({col}, embedding) VALUES (?,?)",  # noqa: S608
        [(f"ghost{i}", _unit(dim, seed=i).tobytes()) for i in range(n)],
    )
    c.commit()
    c.close()


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


def _meta(store, key):
    row = store._c().execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return None if row is None else row[0]


# --- the three reproduced harms --------------------------------------------


def test_a_foreign_vec_table_of_the_wrong_dimension_is_rebuilt(tmp_path):
    """Otherwise every write and every KNN read raises, forever."""
    db = tmp_path / "cache.db"
    _plant_foreign_vec_table(db, dim=1024)

    store = Store(db_path=db)
    store.migrate()

    _seed_obs(store, "o1")
    store.upsert_vec_observations([("o1", _unit(_VEC_DIM))])
    assert store._vec_obs_ids(_unit(_VEC_DIM), k=5) == ["o1"]


def test_b_foreign_vectors_of_the_right_dimension_are_not_adopted(tmp_path):
    db = tmp_path / "cache.db"
    _plant_foreign_vec_table(db, dim=_VEC_DIM, n=5)

    store = Store(db_path=db)
    store.migrate()

    assert store.count_vec_rows("observations") == 0


def test_c_pre_set_embedding_state_goes_back_into_the_backlog(tmp_path):
    """Rows claiming 'ok' over an empty index must be re-embedded, not trusted."""
    db = tmp_path / "cache.db"
    seed = Store(db_path=db)
    seed.migrate()
    _seed_obs(seed, "o1")
    _seed_obs(seed, "o2")
    seed.mark_embedded("observations", ["o1", "o2"], "ok")
    # The abandoned branch's shape: rows marked embedded, no vectors, and no
    # provenance stamp because that build did not write one.
    seed._c().execute("DELETE FROM meta WHERE key='vector_provenance'")
    seed._c().execute("DELETE FROM vec_observations")
    seed._c().commit()
    seed.close()

    store = Store(db_path=db)
    store.migrate()

    backlog = {r["obs_id"] for r in store.select_unembedded("observations")}
    assert backlog == {"o1", "o2"}
    assert store.has_embedded_rows("observations") is False
    assert store.vector_index_state()["state"] == "not_started"


# --- the stamp itself -------------------------------------------------------


def test_migrate_stamps_the_model_and_dimension_it_built_for(tmp_path):
    store = Store(db_path=tmp_path / "cache.db")
    store.migrate()

    stamped = _meta(store, "vector_provenance")
    assert stamped is not None
    model, dim = vector_provenance()
    assert model in stamped
    assert str(dim) in stamped


def test_a_matching_stamp_leaves_real_vectors_alone(tmp_path):
    """The reset must fire on foreign state ONLY — never on our own work."""
    db = tmp_path / "cache.db"
    store = Store(db_path=db)
    store.migrate()
    _seed_obs(store, "o1")
    store.upsert_vec_observations([("o1", _unit(_VEC_DIM))])
    store.mark_embedded("observations", ["o1"], "ok")
    store.close()

    again = Store(db_path=db)
    again.migrate()  # every CLI invocation runs this

    assert again.count_vec_rows("observations") == 1
    assert again.has_embedded_rows("observations") is True
    assert again.select_unembedded("observations") == []


def test_a_changed_model_discards_the_index_it_no_longer_matches(tmp_path):
    db = tmp_path / "cache.db"
    store = Store(db_path=db)
    store.migrate()
    _seed_obs(store, "o1")
    store.upsert_vec_observations([("o1", _unit(_VEC_DIM))])
    store.mark_embedded("observations", ["o1"], "ok")
    store._c().execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES "
        "('vector_provenance', ?)",
        (f'{{"model": "some/other-embedding-model", "dim": {_VEC_DIM}}}',),
    )
    store._c().commit()
    store.close()

    again = Store(db_path=db)
    again.migrate()

    assert again.count_vec_rows("observations") == 0
    assert {r["obs_id"] for r in again.select_unembedded("observations")} == {"o1"}


def test_records_are_reset_on_the_same_terms_as_observations(tmp_path):
    db = tmp_path / "cache.db"
    _plant_foreign_vec_table(
        db, dim=1024, n=2, table="vec_records", col="stable_id"
    )
    seed = Store(db_path=db)
    seed.migrate()
    seed._c().execute(
        "INSERT INTO records(stable_id, source, subject, body, tags, "
        "created_at, updated_at, embedding_state) VALUES "
        "('r1','github','s','b','[]','2026-01-01','2026-01-01','ok')"
    )
    seed._c().execute("DELETE FROM meta WHERE key='vector_provenance'")
    seed._c().commit()
    seed.close()

    store = Store(db_path=db)
    store.migrate()

    assert {r["stable_id"] for r in store.select_unembedded("records")} == {"r1"}
    store.upsert_vec_records([("r1", _unit(_VEC_DIM))])
    assert store.count_vec_rows("records") == 1


def test_migration_without_sqlite_vec_stamps_nothing_and_still_serves_fts(
    tmp_path, monkeypatch
):
    """No extension means no way to validate, so defer rather than guess.

    Stamping here would tell the NEXT migration — the one that can actually
    see the vec tables — that the state on disk had already been vouched for.
    """
    from aggregator.core import store as store_mod

    def _boom(conn):
        raise sqlite3.OperationalError("simulated sqlite-vec ABI mismatch")

    monkeypatch.setattr(store_mod, "_load_sqlite_vec", _boom)
    monkeypatch.setattr(store_mod, "_VEC_LOAD_WARNED", False)

    store = Store(db_path=tmp_path / "cache.db")
    store.migrate()

    assert store.vector_available is False
    assert _meta(store, "vector_provenance") is None
    assert store._c().execute("PRAGMA user_version").fetchone()[0] == 5


def test_vector_provenance_reports_the_model_the_embedder_would_load():
    """The stamp is worthless if it names a model the worker does not use."""
    from aggregator.core.embed import _DEFAULT_MODEL_ST

    model, dim = vector_provenance()
    assert model == _DEFAULT_MODEL_ST
    assert dim == _VEC_DIM


def test_resolving_provenance_does_not_import_the_model_stack():
    """``migrate()`` runs on every CLI invocation; it must stay cheap."""
    import subprocess
    import sys
    import textwrap

    probe = textwrap.dedent(
        """
        import sys
        from aggregator.core.store import vector_provenance
        vector_provenance()
        print("sentence_transformers" in sys.modules)
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip().endswith("False")


@pytest.mark.parametrize("kind", ["observations", "records"])
def test_reset_is_announced_not_silent(tmp_path, caplog, kind):
    """Discarding an index is expensive news; it may not happen quietly."""
    db = tmp_path / "cache.db"
    table = "vec_observations" if kind == "observations" else "vec_records"
    col = "obs_id" if kind == "observations" else "stable_id"
    _plant_foreign_vec_table(db, dim=_VEC_DIM, n=2, table=table, col=col)

    with caplog.at_level("WARNING"):
        Store(db_path=db).migrate()

    assert any(
        "vector" in r.message.lower() and "re-embed" in r.message.lower()
        for r in caplog.records
    ), caplog.text
