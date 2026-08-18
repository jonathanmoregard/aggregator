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
