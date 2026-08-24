"""Confidence signal 3: does anything on this page stand out to the reranker?

WHAT WAS WRONG WITH IT. ``has_standout`` refused any sample smaller than
``VECTOR_FLOOR_MIN_SAMPLE`` — a constant belonging to the *vector floor*, a
different rule reading a different distribution — and ``mcp._RERANK_WINDOW``, a
latency budget documented as "not a quality knob", is exactly that same 20. So
the signal was ``None`` for every page under 20 rows, and lowering the latency
budget by one would have made it ``None`` forever, silently. Two unrelated
constants, one accidental equality, and a quality signal whose life depended on
it.

AND THE BAR WAS THE n=20 SPECIAL CASE OF ITS OWN DERIVATION. The shipped 2.5
came from "the largest of 20 draws sits about 1.9 sd above its own mean whether
or not any of them is relevant; 2.5 is the first round bar above that" — a
statement about 20 draws, applied to whatever number of documents the page
happened to hold, since ``_maybe_rerank`` scores ``min(len(items), window)``.
The null maximum grows with the sample, so a fixed bar is too strict on a short
page and too loose on a long one. What ships now is the general rule the 2.5 was
one point on: the null maximum for THIS sample, plus the headroom 2.5 already
encoded.
"""

from __future__ import annotations

import math

import pytest

from aggregator.core.hybrid import (
    RERANK_STANDOUT_MARGIN,
    RERANK_STANDOUT_MIN_SAMPLE,
    RERANK_STANDOUT_Z,
    expected_max_z,
    has_standout,
    relative_z,
    standout_z_threshold,
)


def _spiked(n: int) -> list[float]:
    """``n`` scores, one of them the largest a sample of ``n`` can produce."""
    return [0.01] * (n - 1) + [0.99]


# --- the bar is a function of the sample, not a constant --------------------


def test_the_bar_at_a_full_window_reproduces_the_constant_that_used_to_ship():
    """2.5 was never wrong — it was right for 20 documents and applied to all
    of them. Pinned so the generalisation cannot drift away from the value it
    replaced."""
    assert standout_z_threshold(20) == pytest.approx(2.5, abs=0.01)
    assert pytest.approx(2.5, abs=0.01) == RERANK_STANDOUT_Z


def test_the_null_maximum_this_is_built_on_matches_the_derivation():
    """"The largest of 20 draws sits about 1.9 sd above its own mean" — the
    sentence the old constant was derived from, now computed rather than
    recalled."""
    assert expected_max_z(20) == pytest.approx(1.87, abs=0.02)


def test_the_bar_rises_with_the_sample_because_the_null_maximum_does():
    """The whole reason a fixed bar is wrong: the more documents you score, the
    further the best of them beats its own mean with no relevance involved."""
    bars = [standout_z_threshold(n) for n in (9, 12, 20, 50, 200)]
    assert bars == sorted(bars)
    assert len(set(bars)) == len(bars)


def test_the_minimum_sample_is_derived_from_the_bar_and_not_from_a_window():
    """Two conditions, both about the same headroom. A standout must clear the
    null maximum by ``RERANK_STANDOUT_MARGIN``; and the sample must be large
    enough that clearing it is not the same thing as BEING the largest value
    the sample can hold, which is a page of n-1 identical scores that no
    cross-encoder produces. ``sqrt(n-1)`` is that ceiling.
    """
    n = RERANK_STANDOUT_MIN_SAMPLE
    assert math.sqrt(n - 1) >= standout_z_threshold(n) + RERANK_STANDOUT_MARGIN
    assert (
        math.sqrt(n - 2) < standout_z_threshold(n - 1) + RERANK_STANDOUT_MARGIN
    ), "a smaller sample would also satisfy it, so this is not the minimum"


def test_the_minimum_sample_is_below_the_rerank_window():
    """THE COUPLING THAT WAS THE FINDING, now asserted rather than assumed. The
    window is a latency budget and may be tuned down; the day it drops below
    the smallest sample this rule can judge, the quality signal dies. It must
    not be able to die quietly."""
    from aggregator.mcp import _RERANK_WINDOW

    assert _RERANK_WINDOW >= RERANK_STANDOUT_MIN_SAMPLE


# --- the signal on a page shorter than the window ---------------------------


def test_a_short_page_with_a_clear_winner_answers_true():
    """Ten documents is half the window and used to be ``None``: "the question
    could not be asked", about a page where one document plainly dominates."""
    assert has_standout(_spiked(10), higher_is_better=True) is True


def test_a_short_flat_page_answers_false_and_not_none():
    """"Twenty documents scored identically" was already worth reporting; so is
    ten. ``None`` claimed the question was unaskable."""
    assert has_standout([0.5] * 10, higher_is_better=True) is False


def test_a_short_page_with_no_winner_answers_false():
    scores = [0.40 + i * 0.01 for i in range(10)]
    assert has_standout(scores, higher_is_better=True) is False


def test_a_page_too_short_to_judge_still_answers_none():
    """The refusal survives — it just has a reason of its own now. Below the
    minimum the bar is unreachable, so ``False`` would be an artefact of the
    sample size rather than an observation about the documents."""
    n = RERANK_STANDOUT_MIN_SAMPLE - 1
    assert has_standout(_spiked(n), higher_is_better=True) is None


def test_the_signal_flips_for_a_distance_where_lower_is_better():
    """One primitive, both orientations — the reranker scores up, the vector
    arm scores down, and two near-identical helpers would drift."""
    dists = [0.99] * 9 + [0.01]
    assert has_standout(dists, higher_is_better=False) is True
    assert has_standout(dists, higher_is_better=True) is False


def test_an_explicit_threshold_still_overrides_the_derived_one():
    """The escape hatch a caller with its own calibration needs, kept so the
    derivation is a default rather than a wall."""
    scores = _spiked(10)
    assert has_standout(scores, higher_is_better=True, z_threshold=99.0) is False


# --- the primitive underneath -----------------------------------------------
#
# ``relative_z`` used to live under the vector floor as well; the floor is an
# absolute distance now (see ``test_hybrid_abstention.py``), so ``has_standout``
# is its only caller and its tests belong here.


def test_relative_z_scores_a_clear_outlier_far_above_the_rest():
    values = [0.05] + [0.60 + i * 0.001 for i in range(40)]
    zs = relative_z(values, higher_is_better=False, min_sample=20)
    assert zs is not None
    assert zs[0] > 5.0
    assert all(z < 1.0 for z in zs[1:])


def test_relative_z_flips_for_scores_where_higher_is_better():
    values = [0.9] + [0.1 + i * 0.001 for i in range(40)]
    zs = relative_z(values, higher_is_better=True, min_sample=20)
    assert zs is not None
    assert zs[0] > 5.0


def test_relative_z_refuses_a_sample_smaller_than_the_caller_asked_for():
    """``None`` is 'undecidable', which is not the same answer as 'no outlier'
    and must not share a representation with it — a caller that read 0.0 here
    would abstain on a corpus it simply has not measured."""
    assert relative_z([0.1, 0.9], higher_is_better=False, min_sample=20) is None


def test_relative_z_refuses_a_sample_with_no_spread_at_all():
    """Every value identical: the rule has no discriminating power, and
    reporting a z of 0 for each would read as 'measured, nothing stands out'."""
    assert relative_z([0.5] * 40, higher_is_better=False, min_sample=20) is None


def test_relative_z_has_no_minimum_of_its_own_to_fall_back_on():
    """The finding this file opens with, at the primitive: a shared default is
    what let a latency budget decide when a quality signal was allowed to
    speak. Every caller now states its own minimum, so there is nothing to
    inherit by accident."""
    with pytest.raises(TypeError):
        relative_z([0.1] * 40, higher_is_better=False)
