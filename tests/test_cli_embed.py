"""``aggregator embed [--catchup|--once]`` — flock, backlog, watermark.

The watermark discipline here is the same one the ingest path runs on, and it
is the reason two of these tests exist at all: ``embedding_state`` must never
get ahead of the vectors it claims to describe. A row marked ``'ok'`` with no
row in ``vec_observations`` is never looked at again, so the vector arm quietly
serves a hole while every count says the index is complete.
"""

import argparse
import fcntl
import os
import sqlite3

import numpy as np
import pytest

from aggregator.core import store as store_mod
from aggregator.core.store import Store


@pytest.fixture
def cache(tmp_path):
    db = tmp_path / "cache.db"
    s = Store(db_path=db)
    s.migrate()
    c = s._c()
    c.execute(
        "INSERT INTO sessions(session_id, root_session_id, kind, first_ts, "
        "last_ts, jsonl_path) VALUES ('sid', 'sid', 'session', "
        "'2026-01-01', '2026-01-01', '/tmp/x.jsonl')"
    )
    for i in range(3):
        c.execute(
            "INSERT INTO observations(obs_id, session_id, root_session_id, "
            "type, ts, body) VALUES (?, 'sid', 'sid', 'user', '2026-01-01', ?)",
            (f"o{i}", f"body text {i}"),
        )
    c.commit()
    s.close()
    return db


def _make_stub_embedder(monkeypatch):
    """Patch Embedder so tests don't require the model."""

    class StubEmbedder:
        def __init__(self, *a, **kw):
            pass

        def embed_documents(self, docs):
            return np.array(
                [[float(i)] * 768 for i in range(len(docs))], dtype=np.float32
            )

        def embed_query(self, q):
            return np.zeros(768, dtype=np.float32)

    monkeypatch.setattr("aggregator.core.embed.Embedder", StubEmbedder)
    monkeypatch.setattr("aggregator.cli.Embedder", StubEmbedder)


def argparse_ns(**kw):
    ns = argparse.Namespace(
        catchup=False, once=False, source="observations", batch_size=500
    )
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_catchup_embeds_all(cache, monkeypatch):
    _make_stub_embedder(monkeypatch)
    from aggregator.cli import _cmd_embed

    rc = _cmd_embed(
        argparse_ns(catchup=True, once=False, source="observations"),
        _store=Store(db_path=cache),
    )
    assert rc == 0
    s = Store(db_path=cache)
    assert not s.select_unembedded("observations", limit=10)


def test_once_embeds_one_batch(cache, monkeypatch):
    _make_stub_embedder(monkeypatch)
    from aggregator.cli import _cmd_embed

    _cmd_embed(
        argparse_ns(catchup=False, once=True, source="observations", batch_size=2),
        _store=Store(db_path=cache),
    )
    s = Store(db_path=cache)
    remaining = s.select_unembedded("observations", limit=10)
    assert len(remaining) == 1  # 3 seeded - 2 embedded


def test_second_catchup_is_noop(cache, monkeypatch):
    _make_stub_embedder(monkeypatch)
    from aggregator.cli import _cmd_embed

    ns = argparse_ns(catchup=True, once=False, source="observations")
    _cmd_embed(ns, _store=Store(db_path=cache))
    s = Store(db_path=cache)
    c = s._c()
    n1 = c.execute("SELECT COUNT(*) AS n FROM vec_observations").fetchone()["n"]
    _cmd_embed(ns, _store=Store(db_path=cache))
    n2 = c.execute("SELECT COUNT(*) AS n FROM vec_observations").fetchone()["n"]
    assert n1 == n2 == 3


def test_empty_bodies_are_marked_skip_not_ok(cache, monkeypatch):
    """A body with nothing in it produces no chunks and therefore no vector.

    It still has to leave the backlog, or the worker re-reads it every run
    forever — but as ``'skip'``, so the distinction between "embedded" and
    "nothing to embed" survives in the table.
    """
    _make_stub_embedder(monkeypatch)
    from aggregator.cli import _cmd_embed

    s = Store(db_path=cache)
    c = s._c()
    c.execute(
        "INSERT INTO observations(obs_id, session_id, root_session_id, "
        "type, ts, body) VALUES ('empty', 'sid', 'sid', 'user', "
        "'2026-01-01', '   ')"
    )
    c.commit()
    s.close()

    _cmd_embed(argparse_ns(catchup=True), _store=Store(db_path=cache))
    s2 = Store(db_path=cache)
    row = s2._c().execute(
        "SELECT embedding_state FROM observations WHERE obs_id = 'empty'"
    ).fetchone()
    assert row["embedding_state"] == "skip"


def test_records_source_embeds_records(cache, monkeypatch):
    _make_stub_embedder(monkeypatch)
    from aggregator.cli import _cmd_embed

    s = Store(db_path=cache)
    s._c().execute(
        "INSERT INTO records(stable_id, source, subject, body, tags, "
        "created_at, updated_at) VALUES ('github:1', 'github', 'subj', "
        "'record body', '[]', '2026-01-01', '2026-01-01')"
    )
    s.commit()
    s.close()

    _cmd_embed(argparse_ns(catchup=True, source="records"), _store=Store(db_path=cache))
    s2 = Store(db_path=cache)
    assert not s2.select_unembedded("records", limit=10)
    n = s2._c().execute("SELECT COUNT(*) AS n FROM vec_records").fetchone()["n"]
    assert n == 1


def test_concurrent_worker_is_refused_by_flock(cache, monkeypatch, capsys):
    """Two workers must not fight over one backlog."""
    _make_stub_embedder(monkeypatch)
    from aggregator.cli import _cmd_embed

    lock_path = str(cache) + ".embed.lock"
    with open(lock_path, "w"):
        pass
    held = os.open(lock_path, os.O_RDWR)
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        rc = _cmd_embed(argparse_ns(catchup=True), _store=Store(db_path=cache))
    finally:
        fcntl.flock(held, fcntl.LOCK_UN)
        os.close(held)
    assert rc == 0  # a skipped tick is healthy, not a failure
    assert "another embed worker" in capsys.readouterr().out
    # ...and it touched nothing.
    s = Store(db_path=cache)
    assert len(s.select_unembedded("observations", limit=10)) == 3


def test_embed_refuses_when_vector_extension_is_missing(cache, monkeypatch, capsys):
    """FAIL LOUDLY AND LEAVE THE BACKLOG ALONE.

    With sqlite-vec missing, ``upsert_vec_*`` no-ops. Marking the rows ``'ok'``
    anyway would advance the watermark past data that was never written — the
    exact shape of silent, permanent loss the ingest watermark rules exist to
    forbid. So the worker refuses the whole run, exits non-zero, and every row
    stays in the backlog.
    """
    _make_stub_embedder(monkeypatch)

    def _boom(conn):
        raise sqlite3.OperationalError("simulated sqlite-vec ABI mismatch")

    monkeypatch.setattr(store_mod, "_load_sqlite_vec", _boom)
    monkeypatch.setattr(store_mod, "_VEC_LOAD_WARNED", False)
    from aggregator.cli import _cmd_embed

    rc = _cmd_embed(argparse_ns(catchup=True), _store=Store(db_path=cache))
    assert rc != 0
    assert "sqlite-vec" in capsys.readouterr().err

    monkeypatch.undo()
    s = Store(db_path=cache)
    assert len(s.select_unembedded("observations", limit=10)) == 3


def test_main_dispatches_embed(cache, monkeypatch):
    _make_stub_embedder(monkeypatch)
    from aggregator.cli import main

    rc = main(["embed", "--catchup"], _store=Store(db_path=cache))
    assert rc == 0
    s = Store(db_path=cache)
    assert not s.select_unembedded("observations", limit=10)


def test_embed_requires_a_mode(cache):
    from aggregator.cli import main

    with pytest.raises(SystemExit):
        main(["embed"], _store=Store(db_path=cache))
