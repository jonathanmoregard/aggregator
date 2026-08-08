"""Qwen3-Reranker: score returns per-doc floats, deterministic, higher
score wins."""

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
