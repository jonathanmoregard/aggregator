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

THAT FLOOR IS AN ABSOLUTE DISTANCE AS OF 2026-08-23, and the rule it replaced
is worth knowing about because its failure shape is easy to rebuild. A
z-score over the arm's own candidates looks scale-free and self-calibrating,
and it is neither: the candidates ARE the extreme left tail of the corpus
distance distribution, and the relevant documents among them ARE the spread
the z-score divides by. So the bar rose with the evidence and the arm
abstained more the more the corpus knew. Anything derived from the window's
own moments has that defect; ``VECTOR_FLOOR_MAX_DISTANCE`` carries the
measurements and ``scripts/vector_floor_calibration.py`` reproduces them.

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

#: How far a vector neighbour may be from the query and still count as a
#: candidate: L2 distance between unit-normalized 768-dim Qwen3 embeddings,
#: which is what ``sqlite-vec`` returns. ``1.00`` is exactly cosine 0.50 —
#: ``d = sqrt(2 - 2·cos)`` — so the rule reads "at least half-aligned".
#:
#: AN ABSOLUTE FLOOR, REPLACING A PER-WINDOW Z-SCORE THAT WAS BACKWARDS.
#: The rule used to ask whether a neighbour stood ``VECTOR_FLOOR_Z = 3.0``
#: standard deviations below the mean of its own candidate set. Its derivation
#: modelled that set as a sample of the corpus distance distribution when the
#: set is, by construction, the extreme left tail of one — the 150 nearest of
#: ~400k — and the error ran in the worst possible direction: THE RELEVANT
#: DOCUMENTS ARE THE SPREAD. Every relevant neighbour that arrives raises the
#: window's standard deviation and lowers its mean, so the bar rises with the
#: evidence and the arm abstains MORE the more the corpus knows. Monte Carlo
#: over the window the arm actually produces, 300 trials, all three background
#: shapes: a no-answer window cleared the z=3.0 bar (so the arm answered) in
#: 99% of gaussian and skewed corpora, while a corpus holding 60 relevant
#: documents emptied the arm 9-58% of the time. Corroborated on real
#: Qwen3 vectors over real cache text: 150-document pools with 25 lexical hits
#: kept 5, 3 and 2 of them.
#:
#: WHY THE HETEROGENEITY OBJECTION THAT MOTIVATED THE Z-SCORE DOES NOT HOLD.
#: The old comment argued that "a 40-turn chat transcript chunk and a 3-line
#: TickTick item do not sit on the same distance scale", so no constant could
#: serve both. Measured, they do: over 228 real cache documents spanning every
#: source, the background distance to an off-domain query pooled to
#: mu=1.275 sd=0.072, and the per-query spread was 0.03-0.07 — the scale is
#: set by the query, not by the document's length or source. What DOES move
#: with the query is the LOCATION: an answerable query's background sits at
#: ~1.21 and an unanswerable one's at ~1.33, and that separation is the signal
#: this floor reads.
#:
#: HOW 1.00 WAS CHOSEN — ``scripts/vector_floor_calibration.py``, both
#: subcommands, 2026-08-23:
#:
#: * ``spot-check`` measured the space. Three queries whose subject IS in the
#:   corpus put their lexical hits at min 0.79/0.91/0.95 and their background
#:   at min 1.02/1.06/1.13. Three off-domain queries (beef wellington,
#:   medieval Portuguese maritime law, a raglan sweater pattern) put their
#:   NEAREST of the same 228 documents at 1.20/1.21/1.26. A floor at 1.00
#:   therefore empties the arm for all three unanswerable queries and keeps
#:   1-3 real neighbours for each answerable one, on real vectors.
#: * ``simulate`` scaled that to the corpus. For an off-domain query the
#:   nearest of 400k background documents lands at 1.172 (gaussian tail) or
#:   1.271 (uniform); 1.00 sits 0.17 below the closer of those, so the
#:   no-answer arm empties in 100% of trials under both. For an ON-DOMAIN
#:   query with no answer the same figure is 0.978, which 1.00 does NOT clear
#:   — see the fail-open note below, that is deliberate.
#: * A second ``spot-check`` over 37 PRODUCTION-SIZED chunks (bodies of 3000+
#:   characters, which the first sample had few of) put the same six queries
#:   within 0.03 of the same places: off-domain nearest 1.262/1.278/1.293,
#:   on-domain background 1.203-1.263. So the constant is not an artefact of
#:   the short bodies that dominate this cache. It did move the LEXICAL hits up
#:   by about 0.04 — a 4000-character chunk that mentions the subject once is
#:   mostly about something else — which is the recall this floor gives away,
#:   named rather than hidden.
#:
#: AN INDEPENDENT MEASUREMENT AGREES, AND IT IS THE STRICTER OF THE TWO. Task
#: M ran ``scripts/rag_rollout_smoke.py`` over a copy of the live cache with 88
#: queries from the user's own recorded searches plus 10 verified-absent
#: subjects, and reported COSINE distances (``d = sqrt(2·c)``): the nearest
#: irrelevant chunk moves from 0.61 at 5k chunks to 0.55 at the 422k the full
#: backfill produces — L2 1.05. So on a warm corpus a floor at or above L2 1.05
#: can never fire, which is what caps this constant from above; 1.00 sits 0.05
#: inside it. The same measurement prices the trade: documents reachable ONLY
#: by the vector arm sit at cosine p25 0.41 / p50 0.54, so a floor at cosine
#: 0.50 keeps roughly their closest 45% and gives up the rest. That cost is
#: real, it is the reason the previous round shipped no floor at all, and it is
#: accepted here because the alternative — an arm that answers every
#: unanswerable query with its ``k`` nearest coincidences — is the failure the
#: rule exists for. ``tests/test_mcp_hybrid.py`` carries the full table.
#:
#: THE TWO ESTIMATES DISAGREE BY 0.12 AND 1.00 IS UNDER BOTH. The spot-check
#: plus simulation put the no-answer neighbour at L2 1.17; the smoke run puts
#: it at 1.05. Both are extrapolations — 228 documents can resolve a 1-in-228
#: tail and the arm reads a 1-in-400k one, and the smoke run extrapolated from
#: ~500 chunks — so the constant takes the stricter and leaves margin under it
#: rather than splitting the difference. On the sample that was actually
#: measured rather than extrapolated, 1.00 sits ~0.20 clear on both sides: the
#: nearest real off-domain neighbour landed at 1.204 and the closest real
#: relevant documents at 0.786-0.953.
#:
#: WHAT IT COSTS ON A WARM CORPUS: the on-domain retrieval window spans about
#: 0.978-1.041, so 1.00 also drops the weaker half of that window — the rows
#: RRF ranks last — and it does NOT empty the arm for an on-domain query with
#: no answer, whose nearest coincidence is 0.978. That case is left to the
#: ``low_confidence`` hedge in ``mcp._note_confidence``, which reports a page
#: the keyword arm never corroborated.
#:
#: WHAT THIS NUMBER DEPENDS ON, so a reviewer knows when it is void: the
#: embedding model (``Qwen/Qwen3-Embedding-0.6B`` MRL-truncated to 768 and
#: L2-normalized) and nothing else. Not the corpus size — an absolute floor
#: never reads the window — and not the document mix, per the measurement
#: above. Change the model and this must be re-derived; ``store`` partitions
#: the vector tables by model id, so the two spaces cannot silently mix.
#:
#: IT CANNOT YET BE CALIBRATED AGAINST A REAL POPULATED INDEX. No cache has
#: embeddings in it: the backfill is a measured 25-30 days of CPU and has not
#: run. So the evidence is simulation over the modelled retrieval window plus
#: spot checks that embed real cache text on demand — not a query against a
#: warm index, and not labelled relevance. ``aggregator retrieval-regression
#: --mode mcp`` is the surface that will score a change to it once labels
#: exist.
VECTOR_FLOOR_MAX_DISTANCE = 1.00


def relative_z(
    values: Sequence[float], *, higher_is_better: bool, min_sample: int
) -> list[float] | None:
    """Per-query z-scores, oriented so bigger always means better.

    Returns one z per input value, or ``None`` when the sample cannot support
    the estimate — fewer than ``min_sample`` values, or no spread at all.

    ``min_sample`` IS THE CALLER'S TO STATE, and is required rather than
    defaulted for a reason that already bit once. It used to default to
    ``VECTOR_FLOOR_MIN_SAMPLE = 20``, a constant belonging to the vector
    floor's since-replaced z-score rule, so :func:`has_standout` — a rule about
    cross-encoder scores, with nothing to do with the vector arm — inherited
    that 20, which happened to equal ``mcp._RERANK_WINDOW``, a latency budget.
    The reranker signal was therefore ``None`` on every page under 20 rows, and
    one latency tweak away from being ``None`` forever. A shared default is how
    two unrelated rules end up with one knob.

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

    NO SPREAD ANSWERS ``False``, NOT ``None``. A cross-encoder that scored
    twenty documents identically has told you something: none of them stands
    out. That is worth reporting, and it costs a sentence — unlike
    :func:`vector_floor`, which deletes rows, so a wrong call there costs the
    user a document they know exists.
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
    max_distance: float = VECTOR_FLOOR_MAX_DISTANCE,
) -> list[str]:
    """Drop vector neighbours that are simply too far from the query.

    ``scored`` is ``(id, distance)`` in the arm's own order (ascending
    distance, best first); the surviving ids come back in that same order, so
    fusion downstream sees an unchanged ranking with fewer members.

    EVERY CANDIDATE IS JUDGED ALONE, AND THAT IS THE PROPERTY THAT MATTERS.
    Nothing here reads the other candidates, so adding a document to the window
    can never evict one that was already surviving, and the probability that
    the arm comes back empty is therefore NON-INCREASING in the number of
    relevant documents the corpus holds. The rule this replaced read the
    window's own mean and standard deviation, which the relevant documents are
    part of, so more evidence made it abstain more often — see
    :data:`VECTOR_FLOOR_MAX_DISTANCE` for the measurements, and
    ``tests/core/test_hybrid_abstention.py`` for the monotonicity property
    asserted against both rules.

    IT IS ALSO WHAT THE REFERENCE IMPLEMENTATIONS DO. Weaviate exposes
    ``distance`` / ``certainty`` on the vector operand, Qdrant a
    ``score_threshold``; both are absolute, per-vector, and neither derives a
    cutoff from the result set. The z-score was the hand-rolled deviation.

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

    FAILS OPEN, AND THE THRESHOLD IS WHERE IT IS FOR THAT REASON. On a personal
    recall tool a false "nothing found" is the worse failure: the user knows the
    document exists, and a tool that hides it is a tool they stop trusting. So
    the number is set to catch the case the rule is named for — a query whose
    subject is nowhere in the corpus, whose nearest neighbour of 400k lands at
    1.17 — and to fail open on the case it cannot honestly decide: an ON-DOMAIN
    query with no answer, whose nearest coincidence lands at 0.978, which is
    CLOSER than a typical genuinely relevant document (mu 1.09). Those two
    populations overlap in this embedding space, measurably, and no absolute
    floor can separate them. Sitting above the overlap keeps the documents;
    sitting below it would delete them to catch a case the evidence does not
    support catching. There is no minimum-sample guard any more because there
    is nothing to estimate: a lone candidate is judged by the same number as a
    window of 150.
    """
    return [doc_id for doc_id, distance in scored if distance <= max_distance]


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
