"""Qwen3-Reranker: score returns per-doc floats, deterministic, higher
score wins — and it does all of that WITHOUT executing hub-hosted code.
"""

import numpy as np
import pytest

from aggregator.core.rerank import Reranker


@pytest.fixture(scope="module")
def reranker():
    try:
        return Reranker()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"reranker unavailable: {e}")


def test_score_shape(reranker):
    scores = reranker.score("what is RRF?", ["reciprocal rank fusion", "convex hull algorithm"])
    assert scores.shape == (2,)


def test_deterministic(reranker):
    a = reranker.score("q", ["d1", "d2"])
    b = reranker.score("q", ["d1", "d2"])
    np.testing.assert_array_equal(a, b)


def test_relevant_beats_irrelevant(reranker):
    scores = reranker.score(
        "what is reciprocal rank fusion",
        [
            "Reciprocal rank fusion (RRF) is a hybrid retrieval scoring method.",
            "The convex hull of a set of points is the smallest convex polygon.",
        ],
    )
    assert scores[0] > scores[1]


# --- no arbitrary code execution on the recall path -------------------------


def test_reranker_never_enables_trust_remote_code(monkeypatch):
    """``rerank=True`` must not license the hub to run code in this process.

    ``Reranker()`` is constructed lazily INSIDE the MCP server — the process
    holding the user's entire personal history — and that server is registered
    bare, with no ``HF_HUB_OFFLINE`` wrapper (only the timer-driven embed unit
    sets it). With ``trust_remote_code=True`` a single ``rerank=True`` query is
    therefore enough to fetch and execute repository-controlled Python there.

    Nothing is given up by refusing: the weights carry no custom modeling code,
    and the architecture is native to transformers — see the test below, which
    proves it against the real model rather than asserting it.
    """
    seen = {}

    class _FakeCrossEncoder:
        def __init__(self, model_name, **kwargs):
            seen["model_name"] = model_name
            seen["kwargs"] = kwargs

    monkeypatch.setattr(
        "sentence_transformers.CrossEncoder", _FakeCrossEncoder, raising=True
    )
    Reranker()

    assert seen["kwargs"].get("trust_remote_code") in (None, False)


def test_the_reranker_architecture_is_native_to_transformers(reranker):
    """The evidence that dropping the flag costs nothing.

    If this model genuinely needed hub-hosted code, the class below would live
    in a ``transformers_modules.*`` package written by the repository. It does
    not — it is the in-tree Qwen3 implementation that ships with transformers.
    """
    assert type(reranker._model.model).__module__.startswith("transformers.models")


def test_the_reranker_pins_a_revision():
    """Same rule as the embedder, and it matters more here.

    This model is loaded inside the long-lived MCP server, so an unpinned
    ``main`` is a moving artifact executing in the process that holds the
    user's whole history.
    """
    import aggregator.core.rerank as rerank_mod

    seen = {}

    class _FakeCrossEncoder:
        def __init__(self, model_name, **kwargs):
            seen["kwargs"] = kwargs

    monkey = pytest.MonkeyPatch()
    monkey.setattr("sentence_transformers.CrossEncoder", _FakeCrossEncoder)
    try:
        rerank_mod.Reranker()
    finally:
        monkey.undo()

    revision = seen["kwargs"].get("revision")
    assert revision == rerank_mod.QWEN3_RERANKER_REVISION
    assert len(revision) == 40


def test_a_caller_supplied_reranker_gets_no_borrowed_revision():
    import aggregator.core.rerank as rerank_mod

    seen = {}

    class _FakeCrossEncoder:
        def __init__(self, model_name, **kwargs):
            seen["kwargs"] = kwargs

    monkey = pytest.MonkeyPatch()
    monkey.setattr("sentence_transformers.CrossEncoder", _FakeCrossEncoder)
    try:
        rerank_mod.Reranker(model_name="some/other-reranker")
    finally:
        monkey.undo()

    assert seen["kwargs"].get("revision") is None


def test_the_pinned_reranker_revision_is_the_one_on_disk():
    from pathlib import Path

    import aggregator.core.rerank as rerank_mod

    snapshots = (
        Path.home()
        / ".cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots"
    )
    if not snapshots.is_dir():
        pytest.skip("Qwen3-Reranker weights are not cached on this machine")
    assert (snapshots / rerank_mod.QWEN3_RERANKER_REVISION).is_dir()
