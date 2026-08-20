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
matches the model and dimension this build would produce.

WHAT "OTHERWISE" MEANS CHANGED IN ROUND 2, and the tests below moved with it.
The first answer was "discard, always". But ``migrate()`` runs on EVERY CLI
invocation and the stamp is derived from ``AGGREGATOR_EMBED_BACKEND``, so a
shell that happened to export ``gguf`` turned ``aggregator query`` — a read —
into an operation that deleted 1.33 GB and 25-30 days of CPU on a log line,
and the sentence-transformers-pinned timer then deleted the rebuild on its
next tick. The protection was right; the price was not consented to.

So the answer is now proportional to what is at stake:

* **nothing computed on disk** — adopt for free: recreate the vec tables (this
  is what repairs a foreign table of the wrong WIDTH) and requeue the rows. A
  fresh cache takes this branch, which is why it is not optional.
* **vectors on disk** — REFUSE. Nothing is deleted, nothing is stamped, and
  the vector arm switches off for the process: reads raise
  ``VectorIndexUnavailableError``, writes no-op, ``vector_index_state`` says
  ``unavailable``, FTS5 is untouched. A mismatch still cannot serve a vector
  whose model is unknown. Deleting requires saying so.

ROUND 3 MOVED WHERE "SAYING SO" HAPPENS. The opt-in was
``AGGREGATOR_VECTOR_REINDEX=1``, read inside ``migrate()`` — which every
subcommand calls — so the consent was ambient and outlived the command it was
meant for. It is now an argument, ``migrate(allow_vector_reindex=True)``,
passed by exactly one caller: ``aggregator embed --reindex``. See
``test_store_reindex_consent.py`` for that boundary; the tests here pin what
happens on either side of it.
"""

import sqlite3

import numpy as np
import pytest
import sqlite_vec

from aggregator.core.store import (
    _VEC_DIM,
    VECTOR_REINDEX_COMMAND,
    Store,
    VectorIndexUnavailableError,
    vector_provenance,
)


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


# --- round 2, S1: what a stray env var may and may not cost -----------------


def test_a_shell_env_var_cannot_destroy_the_index(tmp_path, monkeypatch):
    """The regression this file's second revision exists for.

    ``migrate()`` runs on every CLI command, ``aggregator query`` included, and
    the stamp is derived from ``AGGREGATOR_EMBED_BACKEND``. So exporting
    ``gguf`` in any shell used to make a READ delete the entire vector index —
    weeks of CPU — after which the sentence-transformers-pinned timer deleted
    the rebuild on its next tick. Env ping-pong, unbounded.
    """
    db = tmp_path / "cache.db"
    monkeypatch.delenv("AGGREGATOR_EMBED_BACKEND", raising=False)
    store = Store(db_path=db)
    store.migrate()
    for i in range(5):
        _seed_obs(store, f"o{i}")
    store.upsert_vec_observations(
        [(f"o{i}", _unit(_VEC_DIM, seed=i)) for i in range(5)]
    )
    store.mark_embedded("observations", [f"o{i}" for i in range(5)], "ok")
    store.close()

    monkeypatch.setenv("AGGREGATOR_EMBED_BACKEND", "gguf")
    Store(db_path=db).migrate()

    raw = _raw_vec_conn(db)
    assert raw.execute("SELECT COUNT(*) FROM vec_observations").fetchone()[0] == 5
    raw.close()
    # …and the backlog was not re-opened either, so the timer has nothing to
    # re-do once the variable goes away.
    monkeypatch.delenv("AGGREGATOR_EMBED_BACKEND")
    healthy = Store(db_path=db)
    healthy.migrate()
    assert healthy.count_vec_rows("observations") == 5
    assert healthy.select_unembedded("observations") == []


# --- the three reproduced harms --------------------------------------------


def test_a_foreign_vec_table_of_the_wrong_dimension_never_serves_a_query(tmp_path):
    """Otherwise every write and every KNN read raises ``Dimension mismatch``.

    The harm was the CRASH LOOP, not the table's continued existence: on a
    30-minute timer an unhandled dimension error is a permanent failure with
    no recovery. Switching the arm off converts it into the degrade this
    codebase already handles everywhere — a typed refusal, and FTS5 keeps
    answering.
    """
    db = tmp_path / "cache.db"
    _plant_foreign_vec_table(db, dim=1024)

    store = Store(db_path=db)
    store.migrate()

    _seed_obs(store, "o1")
    # No ``Dimension mismatch`` escapes: the write is refused, not attempted.
    assert store.upsert_vec_observations([("o1", _unit(_VEC_DIM))]) == 0
    with pytest.raises(VectorIndexUnavailableError):
        store._vec_obs_ids(_unit(_VEC_DIM), k=5)
    # And the row stays in the backlog rather than being marked embedded.
    assert {r["obs_id"] for r in store.select_unembedded("observations")} == {"o1"}


def test_a_wrong_width_table_is_repaired_once_consent_is_given(tmp_path):
    """The rebuild still exists — it is just no longer implicit."""
    db = tmp_path / "cache.db"
    _plant_foreign_vec_table(db, dim=1024)

    store = Store(db_path=db)
    store.migrate()  # refuses; the foreign table survives
    store.close()

    again = Store(db_path=db)
    again.migrate(allow_vector_reindex=True)

    _seed_obs(again, "o1")
    assert again.upsert_vec_observations([("o1", _unit(_VEC_DIM))]) == 1
    assert again._vec_obs_ids(_unit(_VEC_DIM), k=5) == ["o1"]


def test_b_foreign_vectors_of_the_right_dimension_are_not_adopted(tmp_path):
    """Right width, unknown model: the arm must not answer from them."""
    db = tmp_path / "cache.db"
    _plant_foreign_vec_table(db, dim=_VEC_DIM, n=5)

    store = Store(db_path=db)
    store.migrate()

    with pytest.raises(VectorIndexUnavailableError):
        store.count_vec_rows("observations")
    with pytest.raises(VectorIndexUnavailableError):
        store._vec_obs_ids(_unit(_VEC_DIM), k=5)
    assert store.vector_index_state()["state"] == "unavailable"


def test_a_mismatch_keeps_every_vector_it_found(tmp_path):
    """The whole point of round 2's S1: a read may not bill weeks of CPU."""
    db = tmp_path / "cache.db"
    _plant_foreign_vec_table(db, dim=_VEC_DIM, n=5)

    Store(db_path=db).migrate()

    raw = _raw_vec_conn(db)
    assert raw.execute("SELECT COUNT(*) FROM vec_observations").fetchone()[0] == 5
    raw.close()


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


def _stamp_a_different_model(store):
    store._c().execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES "
        "('vector_provenance', ?)",
        (f'{{"model": "some/other-embedding-model", "dim": {_VEC_DIM}}}',),
    )
    store._c().commit()


def test_a_changed_model_refuses_to_serve_the_index_it_did_not_write(
    tmp_path, monkeypatch
):
    """Refuse, KEEP, and say so — criterion E's form of round 1's H1.

    What changed is where the refusal comes from. Vectors are keyed
    ``(chunk_id, model)`` now, so a model change is a background job and the
    old index is neither deleted nor quarantined: it keeps serving any process
    configured for it. What must NOT happen is this process filtering the KNN
    to its own empty partition, returning nothing, and looking exactly like a
    corpus that has not been backfilled yet — a silent degradation where there
    used to be a loud one.
    """
    monkeypatch.delenv("AGGREGATOR_EMBED_BACKEND", raising=False)
    db = tmp_path / "cache.db"
    store = Store(db_path=db)
    store.migrate()
    _seed_obs(store, "o1")
    store.upsert_vec_observations([("o1", _unit(_VEC_DIM))])
    store.close()

    monkeypatch.setenv("AGGREGATOR_EMBED_BACKEND", "gguf")
    again = Store(db_path=db)
    again.migrate()

    with pytest.raises(VectorIndexUnavailableError) as e:
        again.has_embedded_rows("observations")
    assert "NOTHING WAS DELETED" in str(e.value)
    assert "unset AGGREGATOR_EMBED_BACKEND" in str(e.value)
    with pytest.raises(VectorIndexUnavailableError):
        again._vec_obs_ids(_unit(_VEC_DIM), k=3)

    raw = _raw_vec_conn(db)
    assert raw.execute("SELECT COUNT(*) FROM vec_observations").fetchone()[0] == 1
    raw.close()


def test_the_previous_models_index_still_serves_its_own_process(
    tmp_path, monkeypatch
):
    """The other half, and the reason the refusal above can be non-destructive:
    the vectors are not damaged by another process having looked at them."""
    monkeypatch.delenv("AGGREGATOR_EMBED_BACKEND", raising=False)
    db = tmp_path / "cache.db"
    store = Store(db_path=db)
    store.migrate()
    _seed_obs(store, "o1")
    store.upsert_vec_observations([("o1", _unit(_VEC_DIM))])
    store.close()

    monkeypatch.setenv("AGGREGATOR_EMBED_BACKEND", "gguf")
    Store(db_path=db).migrate()

    monkeypatch.delenv("AGGREGATOR_EMBED_BACKEND")
    back = Store(db_path=db)
    back.migrate()
    assert back.has_embedded_rows("observations") is True
    assert back._vec_obs_ids(_unit(_VEC_DIM), k=3) == ["o1"]


def test_a_changed_model_discards_the_index_when_consent_is_given(tmp_path):
    db = tmp_path / "cache.db"
    store = Store(db_path=db)
    store.migrate()
    _seed_obs(store, "o1")
    store.upsert_vec_observations([("o1", _unit(_VEC_DIM))])
    store.mark_embedded("observations", ["o1"], "ok")
    _stamp_a_different_model(store)
    store.close()

    again = Store(db_path=db)
    again.migrate(allow_vector_reindex=True)

    assert again.count_vec_rows("observations") == 0
    assert {r["obs_id"] for r in again.select_unembedded("observations")} == {"o1"}


def test_records_are_judged_on_the_same_terms_as_observations(tmp_path):
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

    with pytest.raises(VectorIndexUnavailableError):
        store.count_vec_rows("records")
    assert store.upsert_vec_records([("r1", _unit(_VEC_DIM))]) == 0


def test_an_empty_foreign_table_is_repaired_without_consent(tmp_path):
    """Nothing computed is at stake, so nothing has to be asked.

    This branch is load-bearing rather than an optimisation: a fresh cache
    reaches ``_reconcile_vector_provenance`` with no stamp, and if that
    refused, no machine could ever bootstrap the vector arm.
    """
    db = tmp_path / "cache.db"
    _plant_foreign_vec_table(db, dim=1024, n=0)

    store = Store(db_path=db)
    store.migrate()

    _seed_obs(store, "o1")
    assert store.upsert_vec_observations([("o1", _unit(_VEC_DIM))]) == 1
    assert store._vec_obs_ids(_unit(_VEC_DIM), k=5) == ["o1"]


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
    # A VERSION STRING, not a bare repo id — criterion E. The repo id is
    # carried; so are the quantization, the width, the chunker geometry and
    # the normalization, because each of those changes the bytes of every
    # vector while leaving the model name untouched.
    assert model.startswith(_DEFAULT_MODEL_ST)
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
def test_a_refusal_is_announced_and_names_both_ways_out(tmp_path, caplog, kind):
    """Silence is the failure mode: the arm switches off with no other signal.

    And it must name BOTH commands, because they are opposite actions and only
    the operator knows which was meant — unset a stray env var, or consent to
    a rebuild that costs weeks.
    """
    db = tmp_path / "cache.db"
    table = "vec_observations" if kind == "observations" else "vec_records"
    col = "obs_id" if kind == "observations" else "stable_id"
    _plant_foreign_vec_table(db, dim=_VEC_DIM, n=2, table=table, col=col)

    with caplog.at_level("WARNING"):
        Store(db_path=db).migrate()

    said = "\n".join(r.message for r in caplog.records)
    assert "unset AGGREGATOR_EMBED_BACKEND" in said, caplog.text
    assert VECTOR_REINDEX_COMMAND in said, caplog.text
    assert "NOTHING WAS DELETED" in said, caplog.text


@pytest.mark.parametrize("kind", ["observations", "records"])
def test_a_consented_discard_is_announced_not_silent(tmp_path, caplog, kind):
    """Throwing away an index is expensive news even when it was asked for."""
    db = tmp_path / "cache.db"
    table = "vec_observations" if kind == "observations" else "vec_records"
    col = "obs_id" if kind == "observations" else "stable_id"
    _plant_foreign_vec_table(db, dim=_VEC_DIM, n=2, table=table, col=col)

    with caplog.at_level("WARNING"):
        Store(db_path=db).migrate(allow_vector_reindex=True)

    assert any(
        "vector" in r.message.lower() and "re-embed" in r.message.lower()
        for r in caplog.records
    ), caplog.text


# --- the read path, which never migrates ------------------------------------


def test_a_read_only_store_refuses_a_mismatched_index_too(tmp_path, monkeypatch):
    """``Store(read_only=True)`` is the MCP server — the surface most queried.

    Under the old answer this needed no check: the mismatched vectors had
    already been deleted by whichever writable command ran first. Refusing
    leaves them on disk, so the read path has to reach the same verdict on its
    own or H1's protection is gone exactly where it matters most.

    The read-only store NEVER MIGRATES, so it cannot re-stamp its way out of
    the disagreement — which makes it the strictest test of the guard, and the
    one that proves the refusal is not a side effect of the write path.
    """
    monkeypatch.delenv("AGGREGATOR_EMBED_BACKEND", raising=False)
    db = tmp_path / "cache.db"
    store = Store(db_path=db)
    store.migrate()
    _seed_obs(store, "o1")
    store.upsert_vec_observations([("o1", _unit(_VEC_DIM))])
    store.close()

    monkeypatch.setenv("AGGREGATOR_EMBED_BACKEND", "gguf")
    ro = Store(db_path=db, read_only=True)
    with pytest.raises(VectorIndexUnavailableError):
        ro.has_embedded_rows("observations")
    with pytest.raises(VectorIndexUnavailableError):
        ro._vec_obs_ids(_unit(_VEC_DIM), k=5)


def test_a_read_only_store_serves_a_matching_index(tmp_path):
    """The guard above must not cost the healthy case its vector arm."""
    db = tmp_path / "cache.db"
    store = Store(db_path=db)
    store.migrate()
    _seed_obs(store, "o1")
    store.upsert_vec_observations([("o1", _unit(_VEC_DIM))])
    store.mark_embedded("observations", ["o1"], "ok")
    store.close()

    ro = Store(db_path=db, read_only=True)
    assert ro.has_embedded_rows("observations") is True
    assert ro._vec_obs_ids(_unit(_VEC_DIM), k=5) == ["o1"]
