"""Retrieval metrics — the labelled three, and the label-free drift metric.

TWO FAMILIES, AND THE DIFFERENCE BETWEEN THEM IS THE WHOLE DESIGN.

``ndcg_at_k`` / ``recall_at_k`` / ``mrr_at_k`` are the standard IR metrics and
they need relevance judgements. Nobody has made any yet. The tempting shortcut
— return 0.0 when the label set is empty — produces a number that is
indistinguishable from "the retriever found nothing relevant", so an
unlabelled harness would report a healthy system as catastrophically broken and
a broken one as unchanged. These functions RAISE :class:`NoLabelsError`
instead, and the harness turns that into an explicit "no labels" in the report.

``drift`` needs no labels at all. It compares a run's ranking against a frozen
baseline ranking, which is all you need to catch a regression: something that
used to come back first and now does not is visible without anyone deciding
whether it was the right answer.

WHAT DRIFT DELIBERATELY DOES NOT MEASURE: whether the change was an
improvement. Drift is directionless by construction. A fix that finally makes
``power-on`` return results scores exactly as much drift as a bug that makes it
return garbage. Only labels — or a human looking — can tell those apart. Drift
says WHERE to look, never WHETHER to be happy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

#: Reciprocal-rank persistence for :func:`rank_biased_overlap`. At 0.9 the
#: first ten ranks carry ~65% of the weight, which matches a harness whose
#: baseline is a top-10.
DEFAULT_RBO_P = 0.9

#: Depth at which drift is evaluated. Ten, because that is the baseline depth.
DEFAULT_DRIFT_DEPTH = 10


class NoLabelsError(ValueError):
    """Raised when a labelled metric is asked to score an unlabelled query.

    Its own type so the harness can catch it and report "no labels" rather
    than letting a 0.0 masquerade as a measurement.
    """


def _relevant(relevance: Mapping[str, int]) -> dict[str, int]:
    """The positively-graded entries, or raise.

    An empty mapping and an all-zero mapping are treated identically on
    purpose: both carry zero information about what a good answer looks like,
    so both are "unlabelled" as far as a retrieval metric is concerned.
    """
    positive = {rid: int(g) for rid, g in relevance.items() if int(g) > 0}
    if not positive:
        raise NoLabelsError(
            "no relevant documents are labelled for this query; "
            "nDCG/Recall/MRR are not computable and must not be reported as 0.0"
        )
    return positive


def ndcg_at_k(
    ranked_ids: Sequence[str], relevance: Mapping[str, int], k: int = 10
) -> float:
    """Normalized discounted cumulative gain over the top ``k``.

    Primary metric: it handles graded relevance and it is rank-aware, which
    rank-unaware Precision/Recall and binary MRR/MAP are not.
    """
    positive = _relevant(relevance)
    dcg = sum(
        positive.get(rid, 0) / math.log2(rank + 1)
        for rank, rid in enumerate(ranked_ids[:k], start=1)
    )
    ideal = sorted(positive.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(rank + 1) for rank, g in enumerate(ideal, start=1))
    return dcg / idcg


def recall_at_k(
    ranked_ids: Sequence[str], relevance: Mapping[str, int], k: int = 50
) -> float:
    """Fraction of labelled-relevant documents that appear in the top ``k``.

    The rerank ceiling: a cross-encoder can only reorder what retrieval already
    surfaced, so low Recall@50 means fix retrieval, not the reranker.
    """
    positive = _relevant(relevance)
    found = sum(1 for rid in set(ranked_ids[:k]) if rid in positive)
    return found / len(positive)


def mrr_at_k(
    ranked_ids: Sequence[str], relevance: Mapping[str, int], k: int = 10
) -> float:
    """Reciprocal rank of the first relevant hit inside the top ``k``.

    0.0 here is a real measurement — the query is labelled and the retriever
    missed — which is precisely why the unlabelled case must raise instead.

    The rerank need: high Recall@50 with low MRR@10 is the signature of a
    system whose right answer is present but sitting at rank 8-12.
    """
    positive = _relevant(relevance)
    for rank, rid in enumerate(ranked_ids[:k], start=1):
        if rid in positive:
            return 1.0 / rank
    return 0.0


def overlap_at_k(
    baseline: Sequence[str], current: Sequence[str], k: int = DEFAULT_DRIFT_DEPTH
) -> float:
    """Set overlap of the two top-``k`` heads, ignoring order.

    Reported alongside drift because the two disagree usefully: overlap 1.0
    with non-zero drift means the same documents came back in a different
    order, which is what a fusion or rerank change looks like.
    """
    head_b = list(baseline[:k])
    head_c = list(current[:k])
    if not head_b and not head_c:
        return 1.0
    denominator = max(len(head_b), len(head_c))
    return len(set(head_b) & set(head_c)) / denominator


def rank_biased_overlap(
    baseline: Sequence[str],
    current: Sequence[str],
    p: float = DEFAULT_RBO_P,
    depth: int = DEFAULT_DRIFT_DEPTH,
) -> float:
    """Top-weighted similarity of two ranked lists (Webber et al., 2010).

    ``sum_i w_i * A_i`` where ``A_i`` is the set overlap of the two length-``i``
    prefixes and ``w_i = p^(i-1)``, NORMALIZED BY THE WEIGHT MASS ACTUALLY
    USED. Textbook RBO leaves the sum unnormalized, which scores two identical
    lists at ``1 - p^d`` rather than 1.0 — fine for comparing systems, wrong
    for a regression gate, where "nothing changed" has to read as exactly zero
    drift. The normalization is the only deviation, and it preserves the
    property that matters: a swap at rank 1 costs far more than a swap at rank
    9.

    Two empty lists score 1.0 (both abstained, nothing changed). One empty and
    one not scores 0.0 (abstention appeared or disappeared).
    """
    head_b = list(baseline[:depth])
    head_c = list(current[:depth])
    d = max(len(head_b), len(head_c))
    if d == 0:
        return 1.0
    weighted = 0.0
    mass = 0.0
    seen_b: set[str] = set()
    seen_c: set[str] = set()
    for i in range(1, d + 1):
        if i <= len(head_b):
            seen_b.add(head_b[i - 1])
        if i <= len(head_c):
            seen_c.add(head_c[i - 1])
        weight = p ** (i - 1)
        weighted += weight * (len(seen_b & seen_c) / i)
        mass += weight
    return weighted / mass


def drift(
    baseline: Sequence[str],
    current: Sequence[str],
    p: float = DEFAULT_RBO_P,
    depth: int = DEFAULT_DRIFT_DEPTH,
) -> float:
    """How far a ranking moved away from its frozen baseline, in ``[0, 1]``.

    ``1 - rank_biased_overlap``. 0.0 means byte-identical top-``depth``; 1.0
    means the two rankings share nothing at any prefix — including the case
    where one of them is empty and the other is not.

    Symmetric, and deliberately unsigned: see the module docstring.
    """
    return 1.0 - rank_biased_overlap(baseline, current, p=p, depth=depth)


def top1_changed(baseline: Sequence[str], current: Sequence[str]) -> bool:
    """Whether the single most visible result changed.

    Broken out from drift because it is the one number a human can act on
    without reading a distribution: the first hit is what an agent reads.
    """
    first_b = baseline[0] if baseline else None
    first_c = current[0] if current else None
    return first_b != first_c
