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
"""

from __future__ import annotations


def rrf_fuse(
    fts_ids: list[str],
    vec_ids: list[str],
    k: int = 60,
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
