"""``aggregator embed`` must refuse a quarantined index BEFORE it embeds.

Round 3's H2. ``_cmd_embed`` checked ``vector_available`` — "did the extension
load?" — and stopped there. It never asked the other question the store can
answer: "is the index on disk one this build may write to?"

Round 2's S1 made that a live state rather than a hypothetical. A provenance
mismatch no longer deletes the index; it QUARANTINES it, leaving the vectors
on disk and switching the arm off. ``vector_available`` stays True — the
extension loaded fine — so the worker sailed past the guard, embedded a full
500-row batch, and only found out at ``commit_embed_batch``, where
``_require_vector`` raises. Uncaught.

What that cost, per tick, twice an hour, forever:

* a bare ``VectorIndexUnavailableError`` traceback instead of a diagnosis;
* the unit marked ``failed``, firing a CRITICAL desktop toast whose text names
  missing weights and a missing sqlite-vec wheel — neither of which applies;
* the run's ledger report discarded, so anything it HAD set aside went
  unreported;
* 500 rows of CPU burned on work thrown away before it was written.

The rule: refuse before the batch, name the real cause, exit non-zero.
"""

import argparse
import sqlite3

import numpy as np
import pytest
import sqlite_vec

from aggregator.cli import _cmd_embed
from aggregator.core.store import _VEC_DIM, Store


def _unit(seed=0):
    rng = np.random.default_rng(seed)
    v = rng.random(_VEC_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _stub_embedder(monkeypatch, counter=None):
    class StubEmbedder:
        def __init__(self, *a, **kw):
            pass

        def embed_documents(self, docs):
            if counter is not None:
                counter.append(len(docs))
            return np.array(
                [[float(i)] * _VEC_DIM for i in range(len(docs))], dtype=np.float32
            )

    monkeypatch.setattr("aggregator.cli.Embedder", StubEmbedder)


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


def _vector_count(db):
    c = sqlite3.connect(db)
    c.enable_load_extension(True)
    sqlite_vec.load(c)
    c.enable_load_extension(False)
    n = c.execute("SELECT COUNT(*) FROM vec_observations").fetchone()[0]
    c.close()
    return n


@pytest.fixture
def quarantined(tmp_path, monkeypatch):
    """Vectors on disk, written by a model this process is not configured for.

    Exactly the S1 refusal state: the extension loads, the index is intact,
    and the arm is switched off because its provenance cannot be vouched for.
    """
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
    for i in range(4):
        c.execute(
            "INSERT INTO observations(obs_id, session_id, root_session_id, "
            "type, ts, body) VALUES (?, 'sid', 'sid', 'user', '2026-01-01', ?)",
            (f"o{i}", f"body {i}"),
        )
    c.commit()
    # Two rows embedded under the old model, two still in the backlog.
    store.upsert_vec_observations([(f"o{i}", _unit(i)) for i in range(2)])
    store.mark_embedded("observations", ["o0", "o1"], "ok", None)
    store.close()
    monkeypatch.setenv("AGGREGATOR_EMBED_BACKEND", "gguf")
    return db


def test_quarantine_is_refused_before_any_row_is_embedded(
    quarantined, monkeypatch, capsys
):
    """THE H2 REGRESSION. No traceback, no wasted batch, a real diagnosis."""
    embedded: list[int] = []
    _stub_embedder(monkeypatch, counter=embedded)

    rc = _cmd_embed(_ns(), _store=Store(db_path=quarantined))
    err = capsys.readouterr().err

    assert rc == 1, "a quarantined index must fail the run, loudly"
    assert embedded == [], (
        f"embedded {embedded} document(s) before discovering the refusal"
    )
    assert "refusing to use the vector index" in err, err


def test_the_refusal_names_the_cause_that_actually_applies(
    quarantined, monkeypatch, capsys
):
    """The old failure toast named two causes, neither of them true here."""
    _stub_embedder(monkeypatch)
    _cmd_embed(_ns(), _store=Store(db_path=quarantined))
    err = capsys.readouterr().err

    assert "unset AGGREGATOR_EMBED_BACKEND" in err
    assert "--reindex" in err
    assert "sqlite-vec` wheel" not in err, (
        "named the missing-extension cause on a run where the extension loaded"
    )


def test_the_backlog_and_the_vectors_are_left_exactly_as_they_were(
    quarantined, monkeypatch, capsys
):
    _stub_embedder(monkeypatch)
    _cmd_embed(_ns(), _store=Store(db_path=quarantined))
    capsys.readouterr()

    assert _vector_count(quarantined) == 2
    store = Store(db_path=quarantined)
    assert {r["obs_id"] for r in store.select_unembedded("observations")} == {
        "o2",
        "o3",
    }


def test_a_healthy_index_is_not_refused(tmp_path, monkeypatch, capsys):
    """The guard must not fire on the ordinary case it sits in front of."""
    monkeypatch.delenv("AGGREGATOR_EMBED_BACKEND", raising=False)
    db = tmp_path / "cache.db"
    store = Store(db_path=db)
    store.migrate()
    c = store._c()
    c.execute(
        "INSERT INTO sessions(session_id, root_session_id, kind, first_ts, "
        "last_ts, jsonl_path) VALUES ('sid','sid','session','2026-01-01',"
        "'2026-01-01','/tmp/x.jsonl')"
    )
    c.execute(
        "INSERT INTO observations(obs_id, session_id, root_session_id, type, "
        "ts, body) VALUES ('o0','sid','sid','user','2026-01-01','hello')"
    )
    c.commit()
    store.close()

    _stub_embedder(monkeypatch)
    rc = _cmd_embed(_ns(), _store=Store(db_path=db))
    capsys.readouterr()
    assert rc == 0
    assert Store(db_path=db).select_unembedded("observations") == []
