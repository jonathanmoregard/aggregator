"""Qwen3 embedder: prefix template, MRL truncation, determinism."""

import numpy as np
import pytest

from aggregator.core.embed import QWEN3_QUERY_PREFIX, Embedder


@pytest.fixture(scope="module")
def embedder():
    # Uses the default model configured in Embedder (safetensors path).
    # Tests mark themselves skip if the model isn't available locally,
    # so the test can run in CI environments that don't cache the weight.
    try:
        return Embedder()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"embedder unavailable: {e}")


def test_query_gets_prefix(monkeypatch, embedder):
    calls: list[str] = []
    monkeypatch.setattr(
        embedder,
        "_encode",
        lambda texts: (
            calls.extend(texts),
            np.zeros((len(texts), 768), dtype=np.float32),
        )[1],
    )
    embedder.embed_query("quadratic voting")
    assert calls == [f"{QWEN3_QUERY_PREFIX}quadratic voting"]


def test_document_no_prefix(monkeypatch, embedder):
    calls: list[list[str]] = []
    monkeypatch.setattr(
        embedder,
        "_encode",
        lambda texts: (
            calls.append(list(texts)),
            np.zeros((len(texts), 768), dtype=np.float32),
        )[1],
    )
    embedder.embed_documents(["doc a", "doc b"])
    assert calls == [["doc a", "doc b"]]


def test_output_shape_and_dtype(embedder):
    out = embedder.embed_documents(["hello world"])
    assert out.shape == (1, 768)
    assert out.dtype == np.float32


def test_l2_normalized(embedder):
    out = embedder.embed_documents(["hello world", "second doc"])
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-3)


def test_deterministic(embedder):
    a = embedder.embed_query("test query")
    b = embedder.embed_query("test query")
    np.testing.assert_array_equal(a, b)
