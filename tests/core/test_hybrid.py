"""RRF k=60 fusion: arithmetic, empty-arm fallthroughs, tie order."""

from aggregator.core.hybrid import rrf_fuse


def test_rrf_arithmetic_pure_intersect():
    fts = ["a", "b", "c"]
    vec = ["a", "b", "c"]
    fused = rrf_fuse(fts, vec, k=60)
    # a at rank 1 in both: 1/(60+1) + 1/(60+1) = 2/61
    ids = [i for i, _ in fused]
    assert ids[:3] == ["a", "b", "c"]


def test_rrf_arithmetic_pure_disjoint():
    fused = rrf_fuse(["a", "b"], ["c", "d"], k=60)
    scores = dict(fused)
    assert scores["a"] == 1 / 61
    assert scores["c"] == 1 / 61
    assert scores["b"] == 1 / 62
    assert scores["d"] == 1 / 62


def test_empty_vec_falls_through_to_fts():
    fused = rrf_fuse(["a", "b", "c"], [], k=60)
    ids = [i for i, _ in fused]
    assert ids == ["a", "b", "c"]


def test_empty_fts_falls_through_to_vec():
    fused = rrf_fuse([], ["a", "b"], k=60)
    ids = [i for i, _ in fused]
    assert ids == ["a", "b"]


def test_both_empty_returns_empty():
    assert rrf_fuse([], [], k=60) == []


def test_scoring_promotes_dual_matches_above_single():
    fused = rrf_fuse(["a", "b", "c"], ["c", "d"], k=60)
    ids = [i for i, _ in fused]
    # c appears in both, others in one each: c should top
    assert ids[0] == "c"
