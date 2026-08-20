"""``aggregator embed --reindex`` — the only thing that may delete vectors.

Round 3's H1, CLI half. The store now refuses to reindex unless a caller
passes ``allow_vector_reindex=True`` (see
``tests/core/test_store_reindex_consent.py``); this file pins WHICH caller may
pass it and what it must do first.

Two properties, and they pull in opposite directions on purpose:

* **A read can never reach it.** ``aggregator query`` migrates like every
  other subcommand, and no argv or environment it can be given makes that
  migration destructive.
* **The deliberate rebuild is still one command.** It prints the count, it
  asks, and ``--yes`` scripts it. The gate ``ingest --rebuild`` already puts
  on a row drop, applied to the operation that costs 25-30 days of CPU.
"""

import argparse
import io
import sqlite3

import numpy as np
import pytest
import sqlite_vec

from aggregator.cli import _cmd_embed, build_parser, main
from aggregator.core.store import _VEC_DIM, Store


def _unit(seed=0):
    rng = np.random.default_rng(seed)
    v = rng.random(_VEC_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _vector_count(db):
    c = sqlite3.connect(db)
    c.enable_load_extension(True)
    sqlite_vec.load(c)
    c.enable_load_extension(False)
    n = c.execute("SELECT COUNT(*) FROM vec_observations").fetchone()[0]
    c.close()
    return n


@pytest.fixture
def mismatched(tmp_path, monkeypatch):
    """A cache with five real vectors, and a shell that disagrees about them."""
    db = tmp_path / "cache.db"
    monkeypatch.delenv("AGGREGATOR_EMBED_BACKEND", raising=False)
    store = Store(db_path=db)
    store.migrate()
    c = store._c()
    c.execute(
        "INSERT INTO sessions(session_id, root_session_id, kind, first_ts, "
        "last_ts, jsonl_path) VALUES ('sid','sid','session','2026-01-01',"
        "'2026-01-01','/tmp/x.jsonl')"
    )
    for i in range(5):
        c.execute(
            "INSERT INTO observations(obs_id, session_id, root_session_id, "
            "type, ts, body) VALUES (?, 'sid', 'sid', 'user', '2026-01-01', ?)",
            (f"o{i}", f"body {i}"),
        )
    c.commit()
    store.upsert_vec_observations([(f"o{i}", _unit(i)) for i in range(5)])
    store.mark_embedded("observations", [f"o{i}" for i in range(5)], "ok", None)
    store.close()
    assert _vector_count(db) == 5
    monkeypatch.setenv("AGGREGATOR_EMBED_BACKEND", "gguf")
    return db


def _ns(**kw):
    ns = argparse.Namespace(
        catchup=True,
        once=False,
        seed_models=False,
        source="observations",
        batch_size=500,
        reindex=False,
        yes=False,
    )
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


# --- the read path may not destroy anything, however the shell is set --------


def test_query_cannot_reindex_however_the_environment_is_set(
    mismatched, monkeypatch, capsys
):
    """THE H1 REGRESSION, end to end through ``main``.

    With ``AGGREGATOR_VECTOR_REINDEX=1`` exported and a stray backend var, a
    plain ``aggregator query`` used to delete the whole vector index on its way
    through ``migrate()``.
    """
    monkeypatch.setenv("AGGREGATOR_VECTOR_REINDEX", "1")
    rc = main(["query", "body"], _store=Store(db_path=mismatched))
    capsys.readouterr()
    assert rc == 0
    assert _vector_count(mismatched) == 5, (
        "a read command deleted computed vectors"
    )


def test_reindex_is_not_a_query_flag(mismatched):
    """It is not merely ignored on the read path — it does not exist there."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["query", "body", "--reindex"])


# --- the deliberate rebuild ------------------------------------------------


def test_reindex_previews_the_cost_and_asks_before_deleting(
    mismatched, monkeypatch, capsys
):
    _stub_embedder(monkeypatch)
    monkeypatch.setattr("sys.stdin", io.StringIO("y\n"))
    rc = _cmd_embed(_ns(reindex=True), _store=Store(db_path=mismatched))
    out = capsys.readouterr()
    assert rc == 0
    assert "will DELETE 5 vector(s)" in out.err
    assert "5 row(s) to the embed backlog" in out.err
    assert "25-30 days" in out.err


def test_declining_the_prompt_deletes_nothing(mismatched, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("n\n"))
    rc = _cmd_embed(_ns(reindex=True), _store=Store(db_path=mismatched))
    out = capsys.readouterr()
    assert rc == 1
    assert "NOTHING WAS DELETED" in out.err
    assert _vector_count(mismatched) == 5


def test_eof_on_stdin_is_a_refusal_not_a_yes(mismatched, monkeypatch, capsys):
    """An unattended run must not be read as consent."""
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    rc = _cmd_embed(_ns(reindex=True), _store=Store(db_path=mismatched))
    capsys.readouterr()
    assert rc == 1
    assert _vector_count(mismatched) == 5


def test_yes_scripts_the_prompt_but_still_reports(
    mismatched, monkeypatch, capsys
):
    _stub_embedder(monkeypatch)
    rc = _cmd_embed(_ns(reindex=True, yes=True), _store=Store(db_path=mismatched))
    out = capsys.readouterr()
    assert rc == 0
    assert "will DELETE 5 vector(s)" in out.err, (
        "--yes skipped the report, not just the question"
    )
    # The old vectors are gone and the rows were re-embedded under this build.
    store = Store(db_path=mismatched)
    assert store.select_unembedded("observations", limit=10) == []


def test_reindex_with_nothing_to_delete_does_not_prompt(tmp_path, capsys):
    """A cold index is adopted for free; a prompt there teaches 'y' reflexes."""
    _ = tmp_path
    store = Store(db_path=tmp_path / "cache.db")
    rc = _cmd_embed(_ns(reindex=True), _store=store)
    out = capsys.readouterr()
    assert rc == 0
    assert "nothing to delete" in out.out


def _stub_embedder(monkeypatch):
    class StubEmbedder:
        def __init__(self, *a, **kw):
            pass

        def embed_documents(self, docs):
            return np.array(
                [[float(i)] * _VEC_DIM for i in range(len(docs))], dtype=np.float32
            )

    monkeypatch.setattr("aggregator.cli.Embedder", StubEmbedder)
