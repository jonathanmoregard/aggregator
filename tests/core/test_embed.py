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
    # ``embedder.query_prompt``, not the module constant: the instruction is
    # read off the loaded checkpoint and the constant is only the fallback for
    # a load that exposes no registry. Asserting the constant here would pass
    # for the wrong reason on the day the two diverge. That they agree for the
    # pinned model is pinned separately, in test_embed_query_instruction.py.
    assert calls == [f"{embedder.query_prompt}quadratic voting"]
    assert embedder.query_prompt == QWEN3_QUERY_PREFIX


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


# --- pinned artifacts: the deployment rule applies to model weights too -----


def test_the_embedder_pins_a_revision():
    """"Pinned artifact, no in-place update" has to cover the weights.

    Without ``revision=``, every load resolves ``main`` on the hub, so the
    bytes a unit executes can change under a rev-pinned deployment without a
    single commit anywhere. The pin is a commit sha rather than a tag because
    a tag is repointable by the repo owner.
    """
    import aggregator.core.embed as embed_mod

    seen = {}

    class _FakeST:
        def __init__(self, model_name, **kwargs):
            seen["model_name"] = model_name
            seen["kwargs"] = kwargs

    monkey = pytest.MonkeyPatch()
    monkey.setattr("sentence_transformers.SentenceTransformer", _FakeST)
    try:
        embed_mod.Embedder()
    finally:
        monkey.undo()

    revision = seen["kwargs"].get("revision")
    assert revision == embed_mod.QWEN3_EMBEDDING_REVISION
    assert len(revision) == 40
    assert all(ch in "0123456789abcdef" for ch in revision)


def test_a_caller_supplied_model_is_not_given_someone_elses_revision():
    """A pin only means anything for the model it was taken from."""
    import aggregator.core.embed as embed_mod

    seen = {}

    class _FakeST:
        def __init__(self, model_name, **kwargs):
            seen["kwargs"] = kwargs

    monkey = pytest.MonkeyPatch()
    monkey.setattr("sentence_transformers.SentenceTransformer", _FakeST)
    try:
        embed_mod.Embedder(model_name="some/other-model")
    finally:
        monkey.undo()

    assert seen["kwargs"].get("revision") is None


def test_the_pinned_embedding_revision_is_the_one_on_disk():
    """A pin naming a revision nobody has is an outage, not a safeguard."""
    from pathlib import Path

    import aggregator.core.embed as embed_mod

    snapshots = (
        Path.home()
        / ".cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots"
    )
    if not snapshots.is_dir():
        pytest.skip("Qwen3-Embedding weights are not cached on this machine")
    assert (snapshots / embed_mod.QWEN3_EMBEDDING_REVISION).is_dir()
