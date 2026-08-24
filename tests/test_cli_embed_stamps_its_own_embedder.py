"""The write path must stamp the embedder it BUILT, not the one it assumed.

Round 3's M2 added ``migrate(embedder=)`` so the provenance stamp could name
the model that actually filled the index. Round 4 found the wiring missing —
independently, by all three reviewers — and the reproduction is a grep:

    $ grep -rn 'migrate(embedder' aggregator/
    (nothing)

``_cmd_embed`` migrated first and constructed its ``Embedder`` sixty lines
later, so the only write path in the codebase stamped from
``AGGREGATOR_EMBED_BACKEND`` and ``vector_provenance(embedder)`` was dead code.
A fix with no production caller is not a fix; it is a passing test.

Why the difference bites. The stamp is the input to three separate refusals —
adopt a foreign index (round 1's H1), delete one unasked (round 2's S1), and
authorise ``--reindex`` (round 3's H1) — and a stamp derived from the
environment rather than from the object that did the work fails all three the
same way: a foreign index reads as native. It also now carries the QUANTIZATION
and the CHUNKER version, neither of which the environment knows.

These tests drive an ``Embedder`` whose identity DISAGREES with the process
environment, which is the only way to tell a wired stamp from an inert one.
"""

import argparse
import json

import numpy as np
import pytest

from aggregator.cli import _cmd_embed
from aggregator.core.store import _VEC_DIM, VECTOR_PROVENANCE_KEY, Store


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


def _stub_embedder(monkeypatch, model_id, backend="st"):
    """An embedder that reports an identity the environment cannot produce."""

    class StubEmbedder:
        def __init__(self, *a, **kw):
            self.model_id = model_id
            self.backend = backend

        def embed_documents(self, docs):
            return np.array(
                [[float(i)] * _VEC_DIM for i in range(len(docs))], dtype=np.float32
            )

    monkeypatch.setattr("aggregator.cli.Embedder", StubEmbedder)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.delenv("AGGREGATOR_EMBED_BACKEND", raising=False)
    db = tmp_path / "cache.db"
    s = Store(db_path=db)
    s.migrate()
    c = s._c()
    c.execute(
        "INSERT INTO sessions(session_id, root_session_id, kind, first_ts, "
        "last_ts, jsonl_path) VALUES ('sid','sid','session','2026-01-01',"
        "'2026-01-01','/tmp/x.jsonl')"
    )
    c.execute(
        "INSERT INTO observations(obs_id, session_id, root_session_id, type, "
        "ts, body) VALUES ('o0','sid','sid','user','2026-01-01','hello there')"
    )
    c.commit()
    s.close()
    return db


def _stamp(db):
    s = Store(db_path=db)
    row = s._c().execute(
        "SELECT value FROM meta WHERE key = ?", (VECTOR_PROVENANCE_KEY,)
    ).fetchone()
    s.close()
    return None if row is None else json.loads(row[0])


def test_the_stamp_names_the_embedder_the_command_built(cache, monkeypatch, capsys):
    """THE ROUND-4 REGRESSION: the wiring, not the parameter."""
    _stub_embedder(monkeypatch, "acme/embedder-that-actually-ran")

    rc = _cmd_embed(_ns(), _store=Store(db_path=cache))
    capsys.readouterr()

    assert rc == 0
    assert "acme/embedder-that-actually-ran" in _stamp(cache)["model"], _stamp(cache)


def test_the_stamp_is_not_the_environments_guess(cache, monkeypatch, capsys):
    """If the environment could still win, the wiring would be decorative."""
    _stub_embedder(monkeypatch, "acme/embedder-that-actually-ran")
    _cmd_embed(_ns(), _store=Store(db_path=cache))
    capsys.readouterr()

    from aggregator.core.embed import embedding_version

    assert _stamp(cache)["model"] != embedding_version(), (
        "the stamp still describes what the process environment implies"
    )


def test_the_stamp_carries_quantization_chunker_and_dimension(
    cache, monkeypatch, capsys
):
    """The version string has to cover everything that changes the bytes.

    A vector produced by the same weights at a different quantization, a
    different MRL width or a different chunk geometry is not interchangeable
    with one produced here, and none of those differences is visible in a bare
    repo id.
    """
    _stub_embedder(monkeypatch, "acme/model-a", backend="gguf")
    _cmd_embed(_ns(), _store=Store(db_path=cache))
    capsys.readouterr()

    stamped = _stamp(cache)["model"]
    assert "acme/model-a" in stamped
    assert "q4_k_m" in stamped, stamped
    assert f"@{_VEC_DIM}" in stamped, stamped
    assert "chunk-" in stamped, stamped
    assert "norm-l2" in stamped, stamped


def test_the_version_string_does_not_move_with_the_build(cache):
    """A git hash in here re-embeds the whole corpus on every release.

    The version must be a function of the model, the quantization, the width,
    the chunker and the normalization — and of nothing else. Two calls in the
    same process trivially agree; the point is that the value is derived from
    named constants rather than from anything a deploy touches.
    """
    from aggregator.core import chunk as chunk_mod
    from aggregator.core.embed import embedding_version

    version = embedding_version()
    assert chunk_mod.CHUNKER_VERSION in version
    for moving in ("dirty", "+", "git", "20260", "sha"):
        assert moving not in version.lower(), version
