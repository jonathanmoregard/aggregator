"""Reciprocal Rank Fusion of the FTS5 and vector retrieval arms.

RRF with k=60 (Cormack et al., SIGIR 2009) is the SOTA cheap fusion
default: score-agnostic (no BM25-vs-cosine normalization problem), one
constant, dominates BM25-alone and vector-alone on most heterogeneous
corpora. Upgrade path: tuned convex combination once we have 50-100
labeled query pairs (deferred per spec §Non-goals).

The retriever surface is DELIBERATELY id-list-in, id-list-out — no
Store or Embedder knowledge here. Callers assemble the two id lists
(FTS5 arm + vector arm), pass them in, receive fused ids ordered by
score descending. This keeps the fusion pure and trivially testable
without a live store.

ABSTENTION LIVES HERE TOO, AND IT IS PER-ARM ON PURPOSE. See
``vector_floor``: the one hard prohibition in the design is that nothing
may threshold the FUSED score, so the only place a floor can go is
before fusion, on an arm whose scores mean something on their own.

``vector_floor``'s PRODUCTION CALLER IS ``mcp._fused_id_scope``, which applies
it to ``Store._vec_obs_scored`` / ``_vec_record_scored`` on every default query
before the arms are fused and before the page token freezes the survivors. It
had no caller at all until 2026-08-21: those two store reads ran
``ORDER BY distance`` and then selected the id column alone, so the number this
rule needs was computed by sqlite-vec and discarded one layer below the caller.
The rule was fully implemented and fully tested throughout, which is exactly why
this paragraph is here — a green unit test is not evidence that a guard is live,
and ``tests/test_mcp_vector_floor_wired.py`` asserts the production call rather
than the function.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from statistics import NormalDist

#: The RRF constant (Cormack et al., SIGIR 2009). CONFIRMED CORRECT by the
#: reference design and deliberately separated from ``FUSION_ARM_DEPTH``
#: below: they are independent knobs, and only the depth was ever wrong.
RRF_K = 60

#: How many candidates each arm retrieves BEFORE fusion.
#:
#: NOT A LATENCY BUDGET — it is what makes RRF work at all. Below roughly 50
#: per arm the fusion degenerates: too few documents appear in BOTH lists, so
#: the cross-arm agreement signal that RRF exists to exploit never fires and
#: the result is two concatenated single-arm rankings wearing a fused score.
#: Fusion cannot rescue a document that neither list contained, so depth is
#: the one parameter that bounds what fusion is able to do.
FUSION_ARM_DEPTH = 150

#: How far above its own candidate set a vector neighbour must stand to
#: survive :func:`vector_floor`, in standard deviations.
#:
#: A RELATIVE THRESHOLD AND NOT A COSINE CONSTANT. 0.7 is a folk default, not
#: a calibrated value, and it presumes one distance scale across the corpus —
#: which this corpus does not have. Chat-transcript chunks and three-line task
#: items land in different parts of the distance distribution, so a constant
#: that abstains correctly for one silently deletes the other. A z-score has
#: no constant about the corpus in it: it asks only whether anything in THIS
#: candidate set stands out from the rest of THIS candidate set.
#:
#: 3.0 AND NOT THE 1.5 THE REFERENCE DESIGN SUGGESTS, and the reason is an
#: extreme-value fact about the window this rule reads rather than a taste
#: difference. The candidates are the ``k`` NEAREST neighbours — already the
#: extreme left tail of the corpus distance distribution — so the best of them
#: is BY CONSTRUCTION well below its own window's mean whether or not the query
#: has an answer. For a smooth no-answer tail the minimum lands about 1.7 sd
#: below the window mean if the tail is uniform and about 2.4 sd below it if
#: the tail is normal, both of which clear a 1.5 bar. A 1.5 threshold therefore
#: sits inside the noise floor and abstains on nothing; it is pinned as a
#: failing case in ``tests/core/test_hybrid_abstention.py``. 3.0 is the first
#: round bar above that noise floor. It is DERIVED, NOT CALIBRATED — the
#: argument above is an extreme-value one about the shape of the window, and no
#: number in it came off the real index. Recalibrating it needs labels, which
#: the golden set does not have yet; ``aggregator retrieval-regression --mode
#: mcp`` is the surface that will score a change to it when they arrive.
VECTOR_FLOOR_Z = 3.0

#: Fewer candidates than this and :func:`vector_floor` does not fire.
#:
#: A spread estimated from a handful of points is noise, and a floor that
#: fires on noise produces "search got smarter and stopped finding the thing I
#: know is in there" — the failure that ends a recall tool's usefulness. The
#: production arm returns ``FUSION_ARM_DEPTH`` candidates, an order of
#: magnitude above this, so the guard only ever covers corpora too small to
#: have a distribution in the first place.
VECTOR_FLOOR_MIN_SAMPLE = 20


def relative_z(
    values: Sequence[float], *, higher_is_better: bool, min_sample: int
) -> list[float] | None:
    """Per-query z-scores, oriented so bigger always means better.

    Returns one z per input value, or ``None`` when the sample cannot support
    the estimate — fewer than ``min_sample`` values, or no spread at all.

    ``min_sample`` IS THE CALLER'S TO STATE, and is required rather than
    defaulted for a reason that already bit once. It used to default to
    :data:`VECTOR_FLOOR_MIN_SAMPLE`, so :func:`has_standout` — a rule about
    cross-encoder scores, with nothing to do with the vector floor — inherited
    the vector floor's 20, which happened to equal ``mcp._RERANK_WINDOW``, a
    latency budget. The reranker signal was therefore ``None`` on every page
    under 20 rows, and one latency tweak away from being ``None`` forever. A
    shared default is how two unrelated rules end up with one knob.

    ``None`` AND NOT A LIST OF ZEROS. "Undecidable" and "measured, nothing
    stands out" are opposite facts and must not share a representation: a
    caller that read 0.0 out of an unmeasurable sample would abstain on
    evidence it never had. This is the same rule the eval harness applies to
    unlabelled metrics, for the same reason.

    ``higher_is_better=False`` is the vector arm — sqlite-vec returns a
    DISTANCE, so the good end is the low end and the sign flips. The reranker
    is the other orientation. One primitive rather than two near-identical
    ones, because the two would drift.
    """
    if len(values) < min_sample:
        return None
    spread = statistics.pstdev(values)
    if spread == 0.0:
        return None
    mean = statistics.fmean(values)
    sign = 1.0 if higher_is_better else -1.0
    return [sign * (v - mean) / spread for v in values]


#: How much a reranker score has to beat its own page's NULL MAXIMUM by before
#: it counts as standing out.
#:
#: THE HALF OF THE OLD ``RERANK_STANDOUT_Z = 2.5`` THAT WAS ACTUALLY A CHOICE.
#: That constant was derived as "the largest of 20 draws from any smooth
#: distribution sits about 1.9 standard deviations above its own mean whether
#: or not any of them is relevant; 2.5 is the first round bar above that" — a
#: null term (1.87, see :func:`expected_max_z`) plus 0.63 of headroom. Only the
#: headroom is a judgement; the null term is a fact about the sample size, and
#: freezing the pair at ``n = 20`` made the bar too strict on a short page and
#: too loose on a long one. ``_maybe_rerank`` scores ``min(len(items), window)``
#: documents, so it was already being applied at sizes it was not derived for.
RERANK_STANDOUT_MARGIN = 0.63


def expected_max_z(n: int) -> float:
    """Where the largest of ``n`` draws sits, in sd above their own mean.

    WITH NOTHING RELEVANT AMONG THEM — this is the null, and it is the number a
    standout has to beat. Blom's plotting position, the standard approximation
    to the expected value of the largest normal order statistic: it gives 1.87
    at ``n = 20``, which is the "about 1.9" the shipped constant was built on.

    Grows with ``n``, which is the whole point: a fixed bar over a variable
    window silently changes what it is asserting every time the page size
    changes.
    """
    if n < 2:
        return 0.0
    return NormalDist().inv_cdf((n - 0.375) / (n + 0.25))


def standout_z_threshold(n: int) -> float:
    """The bar for a page of ``n`` scored documents."""
    return expected_max_z(n) + RERANK_STANDOUT_MARGIN


def _smallest_judgeable_sample() -> int:
    """The smallest page this rule can answer ``True`` about for real reasons.

    TWO CONDITIONS, BOTH ABOUT THE SAME HEADROOM. A standout has to clear the
    null maximum by :data:`RERANK_STANDOUT_MARGIN`; and the sample has to be
    large enough that clearing it is not the same thing as BEING the largest
    value the sample can hold. That ceiling is ``sqrt(n-1)`` — the z of one
    value among ``n-1`` identical others — and a page of identical
    cross-encoder scores is not a thing that happens, so a bar within a hair of
    it is a bar nothing can clear. Requiring the same margin between the bar
    and the ceiling is what keeps ``True`` reachable rather than theoretical.

    DERIVED AT IMPORT AND NOT WRITTEN DOWN, so it cannot drift away from the
    margin it comes from; ``tests/core/test_rerank_standout.py`` pins the value
    and both conditions.
    """
    n = 2
    while math.sqrt(n - 1) < standout_z_threshold(n) + RERANK_STANDOUT_MARGIN:
        n += 1
        if n > 1_000:  # pragma: no cover — the loop converges by n≈9
            raise RuntimeError("no sample size satisfies the standout margin")
    return n


#: Fewer scored documents than this and :func:`has_standout` answers ``None``.
RERANK_STANDOUT_MIN_SAMPLE = _smallest_judgeable_sample()

#: What the rule resolves to at a full ``mcp._RERANK_WINDOW`` page — the value
#: that used to be hard-coded for every page. Kept as a reference point for the
#: docs and pinned by a test; nothing reads it to make a decision.
RERANK_STANDOUT_Z = standout_z_threshold(20)


def has_standout(
    values: Sequence[float],
    *,
    higher_is_better: bool,
    z_threshold: float | None = None,
) -> bool | None:
    """Does anything in ``values`` stand out from the rest? ``None`` = can't say.

    ``z_threshold`` defaults to :func:`standout_z_threshold` FOR THIS SAMPLE.
    Passing one overrides it, which is the escape hatch for a caller with its
    own calibration — but the default is the rule, and it is a function rather
    than a constant because the null it has to beat moves with the sample size.

    THREE-VALUED ON PURPOSE. "Too few scores to judge" is not "nothing was
    relevant", and a caller that collapsed them would report low confidence for
    a three-hit page that simply had nothing to compare against.

    NO SPREAD ANSWERS ``False``, WHICH IS THE OPPOSITE OF WHAT
    :func:`vector_floor` DOES WITH THE SAME INPUT, and the asymmetry is
    deliberate rather than an oversight. A cross-encoder that scored twenty
    documents identically has told you something: none of them stands out. That
    is worth reporting. The floor's answer to the same evidence is to keep
    every candidate, because it DELETES rows and a wrong deletion costs the
    user a document they know exists. Reporting a hedge costs a sentence.
    """
    if len(values) < RERANK_STANDOUT_MIN_SAMPLE:
        return None
    bar = standout_z_threshold(len(values)) if z_threshold is None else z_threshold
    zs = relative_z(
        values,
        higher_is_better=higher_is_better,
        min_sample=RERANK_STANDOUT_MIN_SAMPLE,
    )
    if zs is None:
        return False
    return max(zs) >= bar


def vector_floor(
    scored: Sequence[tuple[str, float]],
    *,
    z_threshold: float = VECTOR_FLOOR_Z,
) -> list[str]:
    """Drop vector neighbours that are not outliers in their own candidate set.

    ``scored`` is ``(id, distance)`` in the arm's own order (ascending
    distance, best first); the surviving ids come back in that same order, so
    fusion downstream sees an unchanged ranking with fewer members.

    THE FAILURE THIS EXISTS TO FIX, named: vector search is a ranking
    primitive and is neutral about whether the neighbours are relevant. A
    query about German stock-option taxation, run against a corpus of recipes,
    returns five recipes. Distance is not relevance; distance is "this was the
    closest thing we had". A ``k``-nearest search always returns ``k``.

    APPLIED BEFORE FUSION, NEVER AFTER. RRF scores are not probabilities and
    carry no absolute meaning across queries — a fused 0.031 says "both arms
    ranked it about fifth", not "this is relevant" — so a floor on the fused
    score is thresholding a number that does not mean anything. The vector
    arm's distances do mean something relative to each other, which is exactly
    what this reads.

    AND THERE IS NO BM25 EQUIVALENT, ON PURPOSE. The asymmetry is copied from
    Weaviate, which exposes a max-vector-distance and deliberately ships no
    BM25 counterpart because BM25 scores are neither normalized nor bounded,
    so no universal threshold is meaningful for them. Adding a symmetric
    keyword floor would look tidier and be wrong.

    FAILS OPEN, and says so out loud. When the rule cannot decide — too few
    candidates, or no spread among them — every candidate is kept. On a
    personal recall tool a false "nothing found" is the worse failure: the
    user knows the document exists, and a tool that hides it is a tool they
    stop trusting. Abstention has to be evidence, not the absence of it.
    """
    zs = relative_z(
        [d for _, d in scored],
        higher_is_better=False,
        min_sample=VECTOR_FLOOR_MIN_SAMPLE,
    )
    if zs is None:
        return [doc_id for doc_id, _ in scored]
    return [doc_id for (doc_id, _), z in zip(scored, zs, strict=True) if z >= z_threshold]


def rrf_fuse(
    fts_ids: list[str],
    vec_ids: list[str],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """Fuse two ranked id lists via reciprocal rank fusion.

    Returns ``(id, score)`` pairs ordered by score descending. Score
    is ``sum(1 / (k + rank_i))`` across every arm that returned the id
    (rank is 1-indexed). Empty arms are skipped; if both arms are empty
    the result is ``[]``.
    """
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(fts_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    for rank, doc_id in enumerate(vec_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
