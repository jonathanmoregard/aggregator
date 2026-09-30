"""The ``server`` embed backend: a local llama-server doing the arithmetic.

What has to hold, whatever the HTTP plumbing looks like:

* **The stamp vouches for what the server serves.** The URL is an environment
  value; the stamp is a source constant. So the backend checks, before a
  single vector is produced, that the process at the URL is serving the
  pinned file — a URL must not be able to change what the stamp says.
* **Vectors come back in the space the index is keyed on:** 768-wide, unit
  norm, one per input, in INPUT order whatever order the server answered in.
* **The instruction goes on queries and never on documents** — the same rule
  ``embedding_version`` leans on to leave the instruction out of the stamp.
* **An absent server is a named, typed condition**, not a generic socket
  error: the worker must be able to tell "the unit is stopped" from "this row
  is bad", and the MCP path degrades to FTS5 on it.

Every test runs against ``tests/embed_server_stub.py`` on a real loopback
socket. Nothing here can reach a real model or the network.
"""

from __future__ import annotations

import numpy as np
import pytest

import aggregator.core.embed as embed_mod
from tests.embed_server_stub import (
    EmbedServerStub,
    closed_port_url,
    vector_for,
)


@pytest.fixture
def stub(monkeypatch):
    with EmbedServerStub() as s:
        monkeypatch.setenv(embed_mod.EMBED_URL_ENV, s.url)
        yield s


def _expected(text: str) -> np.ndarray:
    raw = vector_for(text)[None, :]
    return embed_mod.Embedder._truncate_and_normalize(raw)[0]


# -- what the deployment is keyed on -------------------------------------------


def test_the_source_default_is_the_server_backend(monkeypatch):
    """A model change is made in SOURCE (see
    ``cli._would_start_a_second_index_by_accident``), so with nothing exported
    the stamp must name the server's Q8_0 GGUF — the index this build fills."""
    monkeypatch.delenv("AGGREGATOR_EMBED_BACKEND", raising=False)
    assert embed_mod.embedding_version() == (
        "Qwen/Qwen3-Embedding-0.6B-GGUF-q8_0@768/chunk-4000-400/norm-l2"
    )


def test_no_two_backends_share_a_stamp(monkeypatch):
    """Different runtimes and precisions write different bytes; a shared
    stamp would let a KNN compare them with nothing on disk saying so."""
    stamps = set()
    for backend in embed_mod._QUANTIZATION:
        monkeypatch.setenv("AGGREGATOR_EMBED_BACKEND", backend)
        stamps.add(embed_mod.embedding_version())
    assert len(stamps) == len(embed_mod._QUANTIZATION)


def test_the_embedder_stamps_what_it_verified(stub):
    e = embed_mod.Embedder(backend="server")
    assert embed_mod.embedding_version(e) == embed_mod.embedding_version()


# -- the vectors ---------------------------------------------------------------


def test_documents_come_back_in_input_order_unit_norm_and_768_wide(stub):
    docs = ["alpha", "beta", "gamma"]
    out = embed_mod.Embedder(backend="server").embed_documents(docs)

    assert out.shape == (3, 768)
    assert out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-5)
    # The stub answers in reverse; each row must still be ITS input's vector.
    for row, text in zip(out, docs, strict=True):
        np.testing.assert_allclose(row, _expected(text), rtol=1e-5, atol=1e-6)


def test_a_batch_is_one_request(stub):
    """The GPU is only fast if it sees the batch; one round-trip per chunk
    would spend the win on HTTP."""
    embed_mod.Embedder(backend="server").embed_documents(["a", "b", "c", "d"])
    assert stub.embed_requests == [["a", "b", "c", "d"]]


def test_queries_carry_the_instruction_and_documents_do_not(stub):
    e = embed_mod.Embedder(backend="server")
    e.embed_documents(["a document"])
    e.embed_query("a question")

    doc_req, query_req = stub.embed_requests
    assert doc_req == ["a document"]
    assert query_req == [f"{embed_mod.QWEN3_QUERY_PREFIX}a question"]


# -- a URL cannot change what the stamp vouches for ----------------------------


@pytest.mark.parametrize(
    "served",
    [
        "/models/Qwen3-Embedding-0.6B-f16.gguf",
        "/models/Qwen3-Embedding-0.6B-Q4_K_M.gguf",
        "/models/some-other-embedder-Q8_0.gguf",
    ],
)
def test_a_server_serving_another_file_is_refused(stub, served):
    stub.model_path = served
    with pytest.raises(embed_mod.EmbedServerMismatchError) as excinfo:
        embed_mod.Embedder(backend="server")
    assert embed_mod.QWEN3_EMBEDDING_GGUF_FILENAME in str(excinfo.value)
    assert stub.embed_requests == [], "it embedded before checking the model"


def test_a_server_reporting_another_width_or_precision_is_refused(stub):
    stub.n_embd = 768
    with pytest.raises(embed_mod.EmbedServerMismatchError):
        embed_mod.Embedder(backend="server")
    stub.n_embd = embed_mod._NATIVE_DIM
    stub.ftype = "F16"
    with pytest.raises(embed_mod.EmbedServerMismatchError):
        embed_mod.Embedder(backend="server")


# -- an absent server ----------------------------------------------------------


def test_an_unreachable_server_is_a_named_condition(monkeypatch):
    url = closed_port_url()
    monkeypatch.setenv(embed_mod.EMBED_URL_ENV, url)
    with pytest.raises(embed_mod.EmbedServerUnavailableError) as excinfo:
        embed_mod.Embedder(backend="server")
    message = str(excinfo.value)
    assert url in message
    # The operator's next command, not a socket errno.
    assert "aggregator-embed-server" in message


def test_a_server_that_goes_away_mid_run_raises_the_same_condition(stub):
    e = embed_mod.Embedder(backend="server")
    stub.die_after = 1
    e.embed_documents(["served"])
    with pytest.raises(embed_mod.EmbedServerUnavailableError):
        e.embed_documents(["dropped"])


def test_a_server_still_loading_its_model_is_waited_out(stub):
    """llama-server answers 503 while the model loads. The worker is ordered
    After= the server unit, which counts as started the moment it forks, so a
    fresh boot would otherwise fail its first tick on a race."""
    stub.loading_replies = 2
    e = embed_mod.Embedder(backend="server")
    assert e.embed_documents(["x"]).shape == (1, 768)
