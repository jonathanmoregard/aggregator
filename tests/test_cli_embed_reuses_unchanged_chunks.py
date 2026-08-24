"""Never embed the same text twice — criterion E's ``content_sha256`` path.

Rule 2 of the reference design. The embedder is the entire cost of this
pipeline: measured on this hardware at ~40 tokens/second, which puts one
4000-character chunk at ~19 seconds (``docs/embedding-throughput.md``). Every
chunk whose bytes have already been embedded under this model is ~19 seconds
that does not have to be spent, and the answer is identical because the input
was identical.

Three ways the same bytes come round again:

* **A rebuild.** ``ingest --rebuild`` DELETEs and re-INSERTs rows whose content
  never changed. Round 4 found this returned ~483k rows to the backlog; with
  the index keyed on the chunk rather than on a column of the row, they are
  not even selected — this file covers the residual case where the chunk id
  moved but the text did not.
* **Duplicate content across rows.** Chat transcripts repeat themselves
  constantly: quoted text, re-pasted tool output, the same file read twice.
* **A re-chunk.** Changing the chunker geometry changes the version string and
  therefore re-embeds everything — except that most chunks of most documents
  come out byte-identical, and those carry across.

WHAT IS DELIBERATELY *NOT* COVERED: a row whose BODY was edited. Ingest drops
that row's vectors the moment it notices, so nothing stale is ever served, and
the vector reuse would need those exact bytes to still be queryable. Serving a
vector for text the row no longer contains is the failure this project cares
about most, so the recall hole wins over the CPU saving. Named here rather than
discovered later.
"""

import argparse
import hashlib

import numpy as np
import pytest

from aggregator.cli import _cmd_embed
from aggregator.core.store import _VEC_DIM, Store


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


class _CountingEmbedder:
    """Counts the documents actually handed to the model."""

    seen: list[str] = []

    def __init__(self, *a, **kw):
        self.model_id = "acme/counting"
        self.backend = "st"

    def embed_documents(self, docs):
        _CountingEmbedder.seen.extend(docs)
        return np.array(
            [[float(len(d) % 7)] * _VEC_DIM for d in docs], dtype=np.float32
        )


@pytest.fixture(autouse=True)
def _reset_counter():
    _CountingEmbedder.seen = []
    yield
    _CountingEmbedder.seen = []


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.delenv("AGGREGATOR_EMBED_BACKEND", raising=False)
    monkeypatch.setattr("aggregator.cli.Embedder", _CountingEmbedder)
    db = tmp_path / "cache.db"
    s = Store(db_path=db)
    s.migrate()
    c = s._c()
    c.execute(
        "INSERT INTO sessions(session_id, root_session_id, kind, first_ts, "
        "last_ts, jsonl_path) VALUES ('sid','sid','session','2026-01-01',"
        "'2026-01-01','/tmp/x.jsonl')"
    )
    c.commit()
    s.close()
    return db


def _add_obs(db, obs_id, body, ts="2026-01-01"):
    s = Store(db_path=db)
    c = s._c()
    c.execute(
        "INSERT OR REPLACE INTO observations(obs_id, session_id, "
        "root_session_id, type, ts, body) VALUES (?, 'sid','sid','user',?,?)",
        (obs_id, ts, body),
    )
    c.commit()
    s.close()


def test_identical_bodies_in_two_rows_are_embedded_once(cache, capsys):
    """Chat transcripts repeat themselves; the encoder should not."""
    _add_obs(cache, "o0", "the exact same paragraph of text")
    _add_obs(cache, "o1", "the exact same paragraph of text")

    _cmd_embed(_ns(), _store=Store(db_path=cache))
    capsys.readouterr()

    assert _CountingEmbedder.seen == ["the exact same paragraph of text"], (
        f"embedded {len(_CountingEmbedder.seen)} document(s) for one distinct body"
    )
    # Both rows still leave the backlog, and both get a vector.
    s = Store(db_path=cache)
    assert s.select_unembedded("observations", limit=10) == []
    n = s._c().execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0]
    assert n == 2


def test_a_body_already_embedded_is_not_embedded_again(cache, capsys):
    """The cross-run half: the reuse index survives the process."""
    _add_obs(cache, "o0", "a paragraph worth keeping")
    _cmd_embed(_ns(), _store=Store(db_path=cache))
    capsys.readouterr()
    assert len(_CountingEmbedder.seen) == 1

    _CountingEmbedder.seen = []
    _add_obs(cache, "o1", "a paragraph worth keeping", ts="2026-02-01")
    _cmd_embed(_ns(), _store=Store(db_path=cache))
    capsys.readouterr()

    assert _CountingEmbedder.seen == [], (
        "re-embedded text this cache already holds a vector for"
    )


def test_the_reused_vector_is_the_same_vector(cache, capsys):
    """A reuse that returns a different vector is a silent corruption."""
    _add_obs(cache, "o0", "identical content here")
    _add_obs(cache, "o1", "identical content here")
    _cmd_embed(_ns(), _store=Store(db_path=cache))
    capsys.readouterr()

    s = Store(db_path=cache)
    rows = s._c().execute(
        "SELECT obs_id, embedding FROM vec_observations ORDER BY obs_id"
    ).fetchall()
    assert len(rows) == 2
    a = np.frombuffer(rows[0]["embedding"], dtype=np.float32)
    b = np.frombuffer(rows[1]["embedding"], dtype=np.float32)
    assert np.array_equal(a, b)


def test_different_bodies_are_still_embedded_separately(cache, capsys):
    """The guard must not fire on the ordinary case it sits in front of."""
    _add_obs(cache, "o0", "first distinct body")
    _add_obs(cache, "o1", "second distinct body")
    _cmd_embed(_ns(), _store=Store(db_path=cache))
    capsys.readouterr()

    assert sorted(_CountingEmbedder.seen) == [
        "first distinct body",
        "second distinct body",
    ]


def test_a_drained_catchup_publishes_the_index(cache, capsys):
    """The ``completed_at`` pointer, WIRED — round 4's lesson.

    Three reviewers independently found a round-3 fix that no production path
    called. A pointer only the tests flip is the same defect: it would leave
    every real model change serving a half-built index forever, which is the
    thing it exists to prevent.
    """
    _add_obs(cache, "o0", "something to embed")
    s = Store(db_path=cache)
    assert s.embedding_version_state()["completed_at"] is None
    s.close()

    _cmd_embed(_ns(source="both"), _store=Store(db_path=cache))
    out = capsys.readouterr().out

    s = Store(db_path=cache)
    assert s.embedding_version_state()["completed_at"] is not None
    assert "index complete" in out, out


def test_an_interrupted_catchup_publishes_nothing(cache, capsys):
    """A run that stopped where it chose has not seen the whole backlog, so
    it is not entitled to say the index is finished."""
    _add_obs(cache, "o0", "something to embed")
    from aggregator.cli import _EmbedOutcome, _flip_completed_pointer

    s = Store(db_path=cache)
    outcome = _EmbedOutcome()
    outcome.interrupted = True
    _flip_completed_pointer(s, _ns(source="both"), outcome)
    assert s.embedding_version_state()["completed_at"] is None


def test_the_stored_hash_is_of_the_chunk_text(cache, capsys):
    """The hash has to describe the bytes the encoder saw, or reuse is a lie."""
    body = "a body that stays in one chunk"
    _add_obs(cache, "o0", body)
    _cmd_embed(_ns(), _store=Store(db_path=cache))
    capsys.readouterr()

    s = Store(db_path=cache)
    row = s._c().execute(
        "SELECT content_sha256 FROM chunk_embeddings WHERE chunk_id = 'o0'"
    ).fetchone()
    assert row["content_sha256"] == hashlib.sha256(body.encode("utf-8")).hexdigest()
