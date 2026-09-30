"""``aggregator_search_memory`` with the embed server stopped answers from FTS5.

The existing dead-embedder test (``test_mcp_hybrid``) swaps in an object
whose ``embed_query`` raises. That proves the handler, not the wiring: with
the ``server`` backend the failure now starts at a SOCKET, inside the real
``Embedder`` the real ``_get_embedder`` builds, and it can happen at two
moments — the unit already stopped when the first query arrives (construction
fails, so the singleton must stay unbuilt and be retried next time), or
stopped after the singleton was built (every ``embed_query`` fails). Offline-AI
mode produces both, for hours at a time, so both are driven here through the
production path with nothing stubbed but the process on the other end of the
socket.
"""

from __future__ import annotations

import pytest

import aggregator.core.embed as embed_mod
import aggregator.mcp as mcp_mod
from aggregator.core.store import Store
from aggregator.mcp import aggregator_query
from tests.embed_server_stub import EmbedServerStub, short_socket_path
from tests.test_mcp_hybrid import _embed, _seed_sessions


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.delenv("AGGREGATOR_EMBED_BACKEND", raising=False)
    # The real lazy singleton, starting unbuilt, restored afterwards.
    monkeypatch.setattr(mcp_mod, "_embedder", None)
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    _seed_sessions(s, [("o1", "quadratic voting", 1), ("o2", "pigeon roost", 2)])
    # A warm vector index, so the query really does engage the vector arm
    # and reach the embedder rather than short-circuiting on an empty one.
    _embed(s, "observations", [("o1", "quadratic voting"), ("o2", "pigeon roost")])
    return s


def test_a_server_stopped_before_the_first_query(store, monkeypatch):
    # The deployed transport, stopped: systemd has removed the socket.
    monkeypatch.setenv(embed_mod.EMBED_URL_ENV, f"unix://{short_socket_path()}")

    result = aggregator_query("voting", _store=store)

    assert result["ok"] is True
    assert result["total"] == 1
    assert mcp_mod._embedder is None, (
        "a failed construction was cached, so the vector arm would stay off "
        "after the server came back"
    )


def test_a_server_stopped_after_the_embedder_was_built(store, monkeypatch):
    with EmbedServerStub(unix_path=short_socket_path()) as stub:
        monkeypatch.setenv(embed_mod.EMBED_URL_ENV, stub.url)
        warm = aggregator_query("voting", _store=store)
        assert warm["ok"] is True
        assert stub.embed_requests, "the vector arm never asked the server"
        stub.dead = True

        cold = aggregator_query("voting", _store=store)

    assert cold["ok"] is True
    assert cold["total"] == 1
