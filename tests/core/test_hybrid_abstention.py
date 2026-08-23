"""The per-arm abstention rule, and the fusion depth it operates on.

WHY A RELATIVE RULE AND NOT A CONSTANT. An absolute cosine floor (0.7 is the
folk default) assumes one score scale across the corpus, and this corpus is
heterogeneous by construction: a 40-turn chat transcript chunk and a 3-line
TickTick item do not sit on the same distance scale, so a constant that
abstains correctly for one throws away the other. A per-query z-score over the
arm's own candidates has no constant in it at all — it asks whether ANYTHING in
this candidate set stands out from the rest of it, which is a question about
this query rather than about the corpus.

WHY THE VECTOR ARM AND NOT THE FUSED SCORE. RRF scores are not probabilities
and have no absolute meaning across queries: 0.031 means "both arms ranked it
about 5th", never "this is relevant". Thresholding there is the one thing the
design forbids, so the rule is applied per-arm, BEFORE fusion.

WHY NO BM25 FLOOR. Deliberately asymmetric, copying Weaviate: it exposes
max-vector-distance for the vector component and has no BM25 equivalent,
because BM25 scores are unnormalized and unbounded so no universal threshold
means anything. The asymmetry is the design, not an omission.
"""

from statistics import NormalDist

import pytest

from aggregator.core.hybrid import (
    FUSION_ARM_DEPTH,
    RRF_K,
    VECTOR_FLOOR_MIN_SAMPLE,
    VECTOR_FLOOR_Z,
    relative_z,
    vector_floor,
)


def _normal_tail(n: int, centre: float = 0.60, spread: float = 0.02) -> list[float]:
    """A deterministic no-answer tail: the exact quantiles of a normal.

    THE SHAPE A QUERY WITH NO ANSWER PRODUCES. The arm returns its ``k``
    nearest neighbours whatever they are, so for an unanswerable query the
    distances are a smooth slice of the corpus distribution with nothing
    standing out. Exact quantiles rather than a seeded sample so the test
    cannot flake on the tail draw it happens to get.
    """
    nd = NormalDist()
    return [centre + spread * nd.inv_cdf((i + 0.5) / n) for i in range(n)]


def _uniform_tail(n: int, centre: float = 0.60, step: float = 0.0005) -> list[float]:
    """The same idea with the other plausible tail shape.

    Worth pinning separately: the minimum of a UNIFORM sample sits only ~1.73
    standard deviations below its mean however large the sample, which is the
    tightest noise floor either shape produces and therefore the hardest case
    for a threshold to clear.
    """
    return [centre + i * step for i in range(n)]


# --- the depth the arms are retrieved to ------------------------------------


def test_fusion_depth_is_150_per_arm():
    """Below ~50 per arm RRF degenerates: too few documents appear in both
    lists, so the cross-arm agreement signal that makes RRF work never fires.
    Fusion cannot rescue a document neither list contained."""
    assert FUSION_ARM_DEPTH == 150


def test_the_rrf_constant_is_unchanged():
    """k=60 is confirmed correct by the reference design and must not move
    when the depth does — they are independent knobs and only one of them was
    wrong."""
    assert RRF_K == 60


# --- the primitive ----------------------------------------------------------


def test_relative_z_scores_a_clear_outlier_far_above_the_rest():
    values = [0.05] + [0.60 + i * 0.001 for i in range(40)]
    zs = relative_z(values, higher_is_better=False, min_sample=VECTOR_FLOOR_MIN_SAMPLE)
    assert zs is not None
    assert zs[0] > 5.0
    assert all(z < 1.0 for z in zs[1:])


@pytest.mark.parametrize("tail", [_normal_tail(60), _uniform_tail(60)])
def test_relative_z_finds_no_outlier_in_a_no_answer_tail(tail):
    """The shape a no-answer query produces: the k nearest neighbours are all
    about equally far away, because 'nearest' is not 'relevant'."""
    zs = relative_z(tail, higher_is_better=False, min_sample=VECTOR_FLOOR_MIN_SAMPLE)
    assert zs is not None
    assert max(zs) < VECTOR_FLOOR_Z


@pytest.mark.parametrize("tail", [_normal_tail(150), _uniform_tail(150)])
def test_the_reference_designs_1_5_bar_sits_inside_the_noise_floor(tail):
    """WHY ``VECTOR_FLOOR_Z`` IS 3.0 AND NOT THE 1.5 THE RESEARCH SUGGESTS.

    The candidates are already the extreme left tail of the corpus distance
    distribution, so the best of them beats its own window's mean by 1.7-2.4
    standard deviations BY CONSTRUCTION — no relevance required. A 1.5 bar is
    therefore below the noise floor and abstains on nothing. Pinned as a
    failing case so the constant cannot drift back without this showing up.
    """
    zs = relative_z(tail, higher_is_better=False, min_sample=VECTOR_FLOOR_MIN_SAMPLE)
    assert zs is not None
    assert max(zs) > 1.5


def test_relative_z_flips_for_scores_where_higher_is_better():
    values = [0.9] + [0.1 + i * 0.001 for i in range(40)]
    zs = relative_z(values, higher_is_better=True, min_sample=VECTOR_FLOOR_MIN_SAMPLE)
    assert zs is not None
    assert zs[0] > 5.0


def test_relative_z_refuses_a_sample_too_small_to_estimate_a_spread():
    """``None`` is 'undecidable', which is not the same answer as 'no outlier'
    and must not share a representation with it — a caller that read 0.0 here
    would abstain on a corpus it simply has not measured."""
    assert relative_z([0.1, 0.9], higher_is_better=False, min_sample=VECTOR_FLOOR_MIN_SAMPLE) is None


def test_relative_z_refuses_a_sample_with_no_spread_at_all():
    """Every candidate equidistant: the rule has no discriminating power, and
    reporting a z of 0 for each would read as 'measured, nothing stands out'."""
    assert relative_z([0.5] * 40, higher_is_better=False, min_sample=VECTOR_FLOOR_MIN_SAMPLE) is None


# --- the floor --------------------------------------------------------------


def test_the_floor_keeps_the_outlier_and_drops_the_tail():
    scored = [("hit", 0.05)] + [
        (f"noise{i}", 0.60 + i * 0.001) for i in range(40)
    ]
    assert vector_floor(scored) == ["hit"]


@pytest.mark.parametrize("tail", [_normal_tail(150), _uniform_tail(150)])
def test_the_floor_empties_an_arm_that_found_only_a_no_answer_tail(tail):
    """THE FAILURE BEING FIXED, named: a query about German stock-option
    taxation against a corpus of recipes still returns five recipes. The arm
    is a ranking primitive and is neutral about whether the neighbours are
    relevant, so the honest answer for this shape is nothing at all."""
    scored = [(f"recipe{i}", d) for i, d in enumerate(tail)]
    assert vector_floor(scored) == []


def test_the_floor_preserves_the_arms_own_ordering():
    scored = [("a", 0.01), ("b", 0.02), ("c", 0.03)] + [
        (f"noise{i}", 0.80 + i * 0.001) for i in range(50)
    ]
    kept = vector_floor(scored)
    assert kept == ["a", "b", "c"]


def test_the_floor_passes_everything_through_below_the_minimum_sample():
    """A handful of candidates cannot estimate a spread, and a floor that
    fires on noise is how a recall tool stops finding the thing the user knows
    is in there. Production depth is ``FUSION_ARM_DEPTH``; this guard only
    covers a corpus far too small to have a distribution."""
    scored = [(f"o{i}", 0.1 * i) for i in range(VECTOR_FLOOR_MIN_SAMPLE - 1)]
    assert vector_floor(scored) == [id_ for id_, _ in scored]


def test_the_floor_passes_everything_through_when_there_is_no_spread():
    scored = [(f"o{i}", 0.42) for i in range(40)]
    assert vector_floor(scored) == [id_ for id_, _ in scored]


def test_the_floor_on_an_empty_arm_is_empty():
    assert vector_floor([]) == []


def test_the_floor_never_invents_an_id():
    scored = [("hit", 0.05)] + [(f"n{i}", 0.6) for i in range(40)]
    assert set(vector_floor(scored)) <= {id_ for id_, _ in scored}


@pytest.mark.parametrize("z", [0.5, 1.5, 3.0])
def test_a_stricter_threshold_never_keeps_more(z):
    scored = [(f"o{i}", 0.1 + i * 0.01) for i in range(60)]
    loose = vector_floor(scored, z_threshold=0.5)
    tight = vector_floor(scored, z_threshold=z)
    assert set(tight) <= set(loose)
