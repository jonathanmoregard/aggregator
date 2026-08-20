"""Retrieval metrics: the labelled three, and the label-free drift metric.

TWO FAMILIES WITH OPPOSITE FAILURE MODES, which is why they live in one file.

* ``ndcg_at_k`` / ``recall_at_k`` / ``mrr_at_k`` need relevance labels. Nobody
  has labelled anything yet, so the dangerous behaviour is a metric that
  returns ``0.0`` for "unlabelled" — indistinguishable from "the retriever
  found nothing relevant", and it would quietly report a perfect system as
  broken (or, with a 1.0 convention, a broken system as perfect). They raise.
* ``drift`` needs no labels at all. It is the metric that has to work today.
"""

import pytest

from aggregator.evals.metrics import (
    NoLabelsError,
    drift,
    mrr_at_k,
    ndcg_at_k,
    overlap_at_k,
    rank_biased_overlap,
    recall_at_k,
    top1_changed,
)

TEN = [f"o{i}" for i in range(1, 11)]


# --- nDCG@10 ----------------------------------------------------------------


def test_ndcg_at_10_is_one_when_the_ranking_is_ideal():
    labels = {"o1": 3, "o2": 2, "o3": 1}
    assert ndcg_at_k(TEN, labels, k=10) == pytest.approx(1.0)


def test_ndcg_at_10_drops_when_the_best_document_is_pushed_down():
    labels = {"o1": 3, "o2": 2, "o3": 1}
    demoted = ["o2", "o3", *[i for i in TEN if i not in ("o2", "o3")]]
    assert ndcg_at_k(demoted, labels, k=10) < ndcg_at_k(TEN, labels, k=10)


def test_ndcg_at_10_honours_graded_relevance():
    """A 3 at rank 1 must beat a 1 at rank 1 — this is why nDCG is primary."""
    graded = ndcg_at_k(["a", "b"], {"a": 3, "b": 1}, k=10)
    flipped = ndcg_at_k(["b", "a"], {"a": 3, "b": 1}, k=10)
    assert graded > flipped


def test_ndcg_at_10_ignores_documents_below_the_cutoff():
    labels = {"o11": 3}
    assert ndcg_at_k([*TEN, "o11"], labels, k=10) == pytest.approx(0.0)


def test_ndcg_at_10_refuses_to_score_an_unlabelled_query():
    with pytest.raises(NoLabelsError):
        ndcg_at_k(TEN, {}, k=10)


def test_ndcg_at_10_refuses_a_label_set_with_nothing_relevant():
    """All-zero grades carry as little information as no grades. Same refusal."""
    with pytest.raises(NoLabelsError):
        ndcg_at_k(TEN, {"o1": 0, "o2": 0}, k=10)


# --- Recall@50 --------------------------------------------------------------


def test_recall_at_50_is_the_fraction_of_relevant_docs_inside_the_window():
    labels = {"o1": 1, "o2": 1, "missing": 1, "also-missing": 1}
    assert recall_at_k(TEN, labels, k=50) == pytest.approx(0.5)


def test_recall_at_50_ignores_hits_past_the_window():
    ranked = [f"pad{i}" for i in range(50)] + ["gold"]
    assert recall_at_k(ranked, {"gold": 2}, k=50) == pytest.approx(0.0)


def test_recall_at_50_refuses_to_score_an_unlabelled_query():
    with pytest.raises(NoLabelsError):
        recall_at_k(TEN, {}, k=50)


# --- MRR@10 -----------------------------------------------------------------


def test_mrr_at_10_is_the_reciprocal_rank_of_the_first_relevant_hit():
    assert mrr_at_k(TEN, {"o4": 1}, k=10) == pytest.approx(0.25)


def test_mrr_at_10_is_zero_when_nothing_relevant_ranks_inside_the_window():
    """A real 0.0: the query IS labelled, the retriever just missed."""
    assert mrr_at_k(TEN, {"elsewhere": 1}, k=10) == pytest.approx(0.0)


def test_mrr_at_10_refuses_to_score_an_unlabelled_query():
    with pytest.raises(NoLabelsError):
        mrr_at_k(TEN, {}, k=10)


# --- drift (label-free) -----------------------------------------------------


def test_identical_rankings_have_zero_drift():
    assert drift(TEN, list(TEN)) == pytest.approx(0.0)


def test_disjoint_rankings_have_maximum_drift():
    other = [f"x{i}" for i in range(1, 11)]
    assert drift(TEN, other) == pytest.approx(1.0)


def test_drift_weights_the_top_of_the_ranking_more_than_the_tail():
    swap_head = ["o2", "o1", *TEN[2:]]
    swap_tail = [*TEN[:8], "o10", "o9"]
    assert drift(TEN, swap_head) > drift(TEN, swap_tail) > 0.0


def test_two_empty_rankings_do_not_drift():
    """Both abstained. A negative query that still abstains is not a change."""
    assert drift([], []) == pytest.approx(0.0)


def test_abstention_turning_into_results_is_maximum_drift():
    """The label-free signal for criterion D: the system stopped abstaining."""
    assert drift([], ["o1"]) == pytest.approx(1.0)


def test_results_turning_into_abstention_is_maximum_drift():
    assert drift(["o1"], []) == pytest.approx(1.0)


def test_rank_biased_overlap_is_one_for_an_identical_prefix():
    """Normalized RBO, so an unchanged ranking scores exactly 1.0 — not 1-p^d."""
    assert rank_biased_overlap(TEN, list(TEN)) == pytest.approx(1.0)


def test_rank_biased_overlap_and_drift_are_complements():
    current = ["o3", "o1", "o9", *TEN[3:]]
    assert rank_biased_overlap(TEN, current) + drift(TEN, current) == pytest.approx(1.0)


def test_drift_is_symmetric():
    current = ["o5", "o1", "o2"]
    assert drift(TEN, current) == pytest.approx(drift(current, TEN))


def test_reordering_the_same_ids_still_drifts_even_though_overlap_is_perfect():
    """Set overlap alone cannot see a reordering; the eval must."""
    reversed_ = list(reversed(TEN))
    assert overlap_at_k(TEN, reversed_, k=10) == pytest.approx(1.0)
    assert drift(TEN, reversed_) > 0.0


def test_overlap_at_k_is_the_shared_fraction_of_the_two_heads():
    current = ["o1", "o2", "o3", "o4", "o5", "z6", "z7", "z8", "z9", "z10"]
    assert overlap_at_k(TEN, current, k=10) == pytest.approx(0.5)


def test_overlap_of_two_empty_heads_is_total():
    assert overlap_at_k([], [], k=10) == pytest.approx(1.0)


def test_top1_changed_notices_a_new_first_result():
    assert top1_changed(TEN, ["o2", *TEN]) is True
    assert top1_changed(TEN, list(TEN)) is False


def test_top1_changed_notices_abstention_appearing():
    assert top1_changed(TEN, []) is True
    assert top1_changed([], []) is False
