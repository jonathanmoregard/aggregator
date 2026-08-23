"""The per-arm abstention rule, and the fusion depth it operates on.

WHY AN ABSOLUTE DISTANCE AND NOT A PER-WINDOW Z-SCORE. The rule used to keep a
neighbour only if it stood 3 standard deviations below the mean of its own
candidate set. That looks scale-free and self-calibrating, and it is neither.
The candidate set is the ``k`` NEAREST neighbours of ~400k documents — the
extreme left tail of the corpus distance distribution, not a sample of it — and
the relevant documents inside it ARE the spread the z-score divides by. So the
bar rose with the evidence: the arm abstained MORE the more the corpus knew,
and abstained on almost nothing when it knew nothing. That is the exact inverse
of what abstention is for, and this file now asserts the property that catches
it (``test_more_evidence_never_makes_the_arm_abstain_more_often``) against both
the shipped rule and the one it replaced.

WHY THE HETEROGENEITY ARGUMENT FOR THE Z-SCORE DID NOT SURVIVE MEASUREMENT.
The old docstring argued that "a 40-turn chat transcript chunk and a 3-line
TickTick item do not sit on the same distance scale", so no constant could serve
both. Measured over 228 real cache documents spanning every source, they do: the
per-query spread is 0.03-0.07 and the scale is set by the QUERY, not by the
document. What moves is the location — an answerable query's background sits at
~1.21 and an unanswerable one's at ~1.33 — and that separation is the signal an
absolute floor reads.

WHY THE VECTOR ARM AND NOT THE FUSED SCORE. RRF scores are not probabilities
and have no absolute meaning across queries: 0.031 means "both arms ranked it
about 5th", never "this is relevant". Thresholding there is the one thing the
design forbids, so the rule is applied per-arm, BEFORE fusion.

WHY NO BM25 FLOOR. Deliberately asymmetric, copying Weaviate: it exposes
max-vector-distance for the vector component and has no BM25 equivalent,
because BM25 scores are unnormalized and unbounded so no universal threshold
means anything. The asymmetry is the design, not an omission.

EVERY NUMBER IN THIS FILE IS RE-RUNNABLE. The window model and the Monte Carlo
come from ``scripts/vector_floor_calibration.py``, imported below rather than
copied, so the harness a reviewer runs and the harness these tests assert on
cannot drift apart. Its ``spot-check`` subcommand produces the measured
constants; ``simulate`` produces the curves.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest

from aggregator.core.hybrid import (
    FUSION_ARM_DEPTH,
    RRF_K,
    VECTOR_FLOOR_MAX_DISTANCE,
    vector_floor,
)

_CALIBRATION = (
    Path(__file__).resolve().parents[2] / "scripts" / "vector_floor_calibration.py"
)


def _load_calibration():
    """Import the calibration harness by path — ``scripts/`` is not a package.

    Deliberate: a test that re-implemented the retrieval-window model would be
    asserting against its own copy of the thing under test, and the copy is
    where the last derivation went wrong.
    """
    spec = importlib.util.spec_from_file_location(
        "vector_floor_calibration", _CALIBRATION
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


calib = _load_calibration()

# Measured by ``vector_floor_calibration.py spot-check`` over 228 real cache
# documents on 2026-08-23, embedded with the production Qwen3-Embedding-0.6B.
# L2 on unit-normalized vectors, which is what sqlite-vec returns.
_OFF_DOMAIN_BG = (1.33, 0.034)  # query whose subject is nowhere in the corpus
_ON_DOMAIN_BG = (1.21, 0.050)  # query about something the corpus is full of
_RELEVANT = (1.09, 0.09)  # documents that actually mention the subject

# Where the 150 nearest of 400k land for an off-domain query under a gaussian
# corpus tail: ``simulate`` reports the nearest at 1.172 (p05 1.154) and the
# 150th at 1.215. A no-answer window is that band, whatever shape it has
# INSIDE it — which is what the three fixtures below vary.
_NO_ANSWER_LO, _NO_ANSWER_HI = 1.172, 1.215

# The same window under a gumbel-min (exponential) corpus tail: the nearest
# coincidence lands at 0.993 instead. The floor does NOT empty this one, on
# purpose — see ``test_the_floor_fails_open_on_a_heavier_tail_than_measured``.
_HEAVY_TAIL_LO, _HEAVY_TAIL_HI = 0.993, 1.136


def _normal_window(n: int, lo: float, hi: float) -> list[float]:
    """A no-answer window whose interior is normal-shaped, spanning ``lo..hi``.

    Exact quantiles rather than a seeded sample so the test cannot flake on the
    draw it happens to get.
    """
    nd = NormalDist()
    zs = [nd.inv_cdf((i + 0.5) / n) for i in range(n)]
    span = zs[-1] - zs[0]
    return [lo + (z - zs[0]) / span * (hi - lo) for z in zs]


def _uniform_window(n: int, lo: float, hi: float) -> list[float]:
    """Evenly spaced. The tightest interior either tail shape produces, and
    therefore the hardest case for any rule that reads the window's own
    spread — which is exactly why the old z-score rule passed it and nothing
    else."""
    return [lo + i * (hi - lo) / (n - 1) for i in range(n)]


def _skewed_window(n: int, lo: float, hi: float) -> list[float]:
    """A long left tail INSIDE the window: the closest candidate sits far
    below the rest of its own set with nothing relevant anywhere in it. The
    shape that makes a relative rule cry outlier at pure coincidence."""
    raw = [math.expm1(3.0 * i / (n - 1)) for i in range(n)]
    span = raw[-1] - raw[0]
    return [lo + (r - raw[0]) / span * (hi - lo) for r in raw]


_SHAPES = {
    "normal": _normal_window,
    "uniform": _uniform_window,
    "skewed": _skewed_window,
}


def _scored(distances) -> list[tuple[str, float]]:
    return [(f"o{i:03d}", float(d)) for i, d in enumerate(distances)]


def _shipped_rule(distances):
    """The harness's rule interface, backed by the SHIPPED ``vector_floor``.

    Not ``calib.keep_absolute``: a Monte Carlo run against the harness's own
    copy of the rule proves something about the harness. This one goes through
    the function production calls, so unwiring or changing it turns the
    monotonicity test red.
    """
    scored = _scored(distances)
    kept = set(vector_floor(scored))
    return np.array([doc_id in kept for doc_id, _ in scored])


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


# --- the threshold ----------------------------------------------------------


def test_the_floor_is_the_cosine_half_way_point():
    """``d = sqrt(2 - 2·cos)`` on unit-normalized vectors, so ``d <= 1.0`` is
    ``cos >= 0.5``. Stated because sqlite-vec returns L2 and everyone reasons
    in cosine, and a reader who converts it wrong will misjudge the constant by
    a factor that looks plausible."""
    assert VECTOR_FLOOR_MAX_DISTANCE == 1.00
    cosine = 1.0 - VECTOR_FLOOR_MAX_DISTANCE**2 / 2.0
    assert cosine == pytest.approx(0.5)


def test_the_floor_sits_between_the_measured_populations():
    """THE WHOLE CALIBRATION, IN ONE ASSERTION. It has to be above the closest
    genuinely relevant documents measured in this embedding space (0.79, 0.91,
    0.95 for the three answerable queries) and below where the nearest of 400k
    coincidences lands for an unanswerable one (1.172). Both numbers come from
    ``vector_floor_calibration.py``; if either moves the constant is void."""
    assert 0.953 < VECTOR_FLOOR_MAX_DISTANCE < _NO_ANSWER_LO


def test_the_harness_models_the_rule_that_actually_ships():
    """The Monte Carlo below is only evidence if it exercises the shipped
    function. ``_assert_matches_shipped`` imports ``hybrid.vector_floor`` and
    fails if the harness has drifted into modelling something else."""
    calib._assert_matches_shipped(VECTOR_FLOOR_MAX_DISTANCE)


# --- a no-answer window empties the arm, whatever shape it has --------------


@pytest.mark.parametrize("shape", list(_SHAPES))
def test_the_floor_empties_an_arm_that_found_only_a_no_answer_window(shape):
    """THE FAILURE BEING FIXED, named: a query about German stock-option
    taxation against a corpus of recipes still returns five recipes. The arm is
    a ranking primitive and is neutral about whether the neighbours are
    relevant, so the honest answer for this shape is nothing at all.

    All three interiors, not just the one the old model happened to satisfy:
    the z-score rule passed the uniform fixture and failed the other two, and
    that single green light was the whole evidence for its threshold.
    """
    window = _SHAPES[shape](FUSION_ARM_DEPTH, _NO_ANSWER_LO, _NO_ANSWER_HI)
    assert vector_floor(_scored(window)) == []


@pytest.mark.parametrize("shape", list(_SHAPES))
def test_a_no_answer_window_is_emptied_at_any_size_the_arm_can_return(shape):
    """The arm returns fewer than ``FUSION_ARM_DEPTH`` candidates whenever the
    index is smaller than the depth — which is every day of the 25-30 day
    backfill. The old rule failed open below 20 candidates because it needed a
    spread to estimate; this one needs nothing, so the answer must not change
    with the count."""
    for n in (1, 3, 19, 20, FUSION_ARM_DEPTH):
        window = _SHAPES[shape](max(n, 2), _NO_ANSWER_LO, _NO_ANSWER_HI)[:n]
        assert vector_floor(_scored(window)) == [], n


def test_the_floor_fails_open_on_a_heavier_tail_than_measured():
    """A DELIBERATE LIMIT, PINNED SO IT IS NOT MISTAKEN FOR A GUARANTEE.

    Under a gumbel-min corpus tail the nearest of 400k coincidences lands at
    0.993 rather than 1.172, and the floor then keeps it. That is the fail-open
    direction and it is the correct one here: the same 0.993 is where a
    genuinely relevant document sits for an answerable query, so a threshold
    low enough to abstain under this tail would delete real answers to catch a
    case the measurements do not support. 228 real documents put the observed
    tail closer to gaussian than to gumbel — ``(mu-min)/sigma`` came out
    2.6-3.4 where gaussian predicts 2.86 and gumbel-min 3.78 — but three
    queries is not enough to rule it out, so the rule declines to bet on it.
    """
    window = _skewed_window(FUSION_ARM_DEPTH, _HEAVY_TAIL_LO, _HEAVY_TAIL_HI)
    assert vector_floor(_scored(window)) != []


# --- the property the old rule violated -------------------------------------


def test_more_evidence_never_makes_the_arm_abstain_more_often():
    """THE REGRESSION THAT ATE THE LAST DERIVATION, as a property.

    As the corpus holds more relevant documents, the chance that the arm comes
    back empty must not go UP. The z-score rule failed this because the
    relevant documents are part of the spread it divides by; any rule derived
    from the window's own moments will fail it the same way, which is why this
    is asserted rather than reasoned about.

    Monte Carlo over the window the arm actually produces — the ``k`` smallest
    of ``N``, not a sample of the corpus — across all three background shapes.
    """
    counts = (0, 1, 3, 10, 30, 60)
    for shape in calib.SHAPES:
        curve = calib.p_empty_curve(
            _shipped_rule,
            shape=shape,
            bg_mu=_ON_DOMAIN_BG[0],
            bg_sigma=_ON_DOMAIN_BG[1],
            rel_mu=_RELEVANT[0],
            rel_sigma=_RELEVANT[1],
            relevant_counts=counts,
            trials=120,
            seed=20260823,
        )
        for a, b in zip(counts, counts[1:], strict=False):
            assert curve[a] >= curve[b] - 1e-12, (
                f"{shape}: P(arm emptied) rose from {curve[a]:.2f} at m={a} to "
                f"{curve[b]:.2f} at m={b} — the arm abstains MORE the more "
                f"evidence the corpus holds"
            )


def test_the_replaced_z_score_rule_still_violates_that_property():
    """The other half of the evidence, kept executable.

    Without this, "the z-score was backwards" is a claim in a commit message.
    With it, the harness demonstrates the violation on every run, so a future
    round cannot re-derive the same mechanism and believe it is new.
    """
    counts = (0, 1, 3, 10, 30, 60)
    violations = 0
    for shape in calib.SHAPES:
        curve = calib.p_empty_curve(
            lambda d: calib.keep_zscore(d, 3.0),
            shape=shape,
            bg_mu=_ON_DOMAIN_BG[0],
            bg_sigma=_ON_DOMAIN_BG[1],
            rel_mu=_RELEVANT[0],
            rel_sigma=_RELEVANT[1],
            relevant_counts=counts,
            trials=120,
            seed=20260823,
        )
        violations += any(
            curve[b] > curve[a] + 1e-12
            for a, b in zip(counts, counts[1:], strict=False)
        )
    assert violations, (
        "the z-score rule no longer abstains more with more evidence — either "
        "the harness stopped modelling the retrieval window, or the finding "
        "this file exists to pin has evaporated"
    )


def test_adding_a_candidate_never_evicts_one_that_was_surviving():
    """The structural reason the property above holds, asserted directly.

    Each candidate is judged against a constant and never against the others,
    so the survivor set can only grow. This is the invariant a future
    "improvement" that reads the window would break.
    """
    base = _scored(_uniform_window(40, 0.80, 1.30))
    survivors = set(vector_floor(base))
    for extra in (0.05, 0.99, 1.01, 1.40):
        wider = sorted([*base, ("extra", extra)], key=lambda kv: kv[1])
        assert survivors <= set(vector_floor(wider)), extra


# --- an answerable window keeps its answer ----------------------------------


def test_the_floor_keeps_the_neighbours_of_an_answerable_query():
    """The measured shape of a query the corpus can answer: genuinely relevant
    documents at 0.79-0.95, and the coincidence floor of a 400k ON-DOMAIN
    corpus at 0.98 and up, which is closer than most relevant documents are.

    IT IS AN ABSTENTION RULE AND NOT A PRECISION FILTER, and this is where that
    shows: plenty of the coincidental background survives alongside the real
    neighbours, because at 400k documents the two populations overlap and no
    absolute threshold separates them. What the floor guarantees is the thing
    it is for — the answer is not thrown away, and an arm with nothing in range
    comes back empty rather than serving its ``k`` nearest coincidences.
    """
    relevant = [0.786, 0.906, 0.953]
    background = _uniform_window(147, 0.978, 1.041)
    window = _scored(sorted(relevant + background))
    kept = set(vector_floor(window))
    assert {doc_id for doc_id, d in window if d in relevant} <= kept
    assert kept == {doc_id for doc_id, d in window if d <= VECTOR_FLOOR_MAX_DISTANCE}


def test_the_floor_preserves_the_arms_own_ordering():
    scored = [("a", 0.01), ("b", 0.02), ("c", 0.03)] + [
        (f"noise{i}", 1.20 + i * 0.001) for i in range(50)
    ]
    assert vector_floor(scored) == ["a", "b", "c"]


def test_the_floor_on_an_empty_arm_is_empty():
    assert vector_floor([]) == []


def test_the_floor_never_invents_an_id():
    scored = [("hit", 0.05)] + [(f"n{i}", 1.3) for i in range(40)]
    assert set(vector_floor(scored)) <= {id_ for id_, _ in scored}


def test_the_boundary_is_inclusive():
    """A neighbour exactly at the threshold is kept, not dropped — the
    fail-open direction, and the difference is visible in a test that pins the
    constant."""
    assert vector_floor([("edge", VECTOR_FLOOR_MAX_DISTANCE)]) == ["edge"]


@pytest.mark.parametrize("max_distance", [0.5, 1.0, 1.5])
def test_a_stricter_threshold_never_keeps_more(max_distance):
    scored = [(f"o{i}", 0.1 + i * 0.02) for i in range(60)]
    loose = vector_floor(scored, max_distance=1.5)
    tight = vector_floor(scored, max_distance=max_distance)
    assert set(tight) <= set(loose)
