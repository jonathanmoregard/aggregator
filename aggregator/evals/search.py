"""The ``(query, limit) -> [result ids]`` callables the harness measures.

THE HARNESS IS ID-LIST-IN, ID-LIST-OUT, exactly like ``core.hybrid``. It knows
nothing about stores, embedders or ontologies — which is what lets the same
frozen golden set and the same drift metric measure a lexical-only arm today
and the full fused pipeline once criterion H exposes search modes as a
first-class surface. Adding a mode here is adding one function and one registry
entry.

ONE RULE THESE FUNCTIONS DO NOT SHARE WITH THE MCP SERVER: THEY NEVER DEGRADE.
``aggregator.mcp`` falls back to FTS5 whenever the vector arm is unavailable,
and that is right for a user, who would rather have keyword results than an
error. It is wrong here. A harness that degrades silently measures the lexical
arm, files the numbers under "hybrid", and reports no drift — the
empty-result-looks-like-success failure this project bans by name. So a hybrid
run on a machine with no vector index RAISES.

Result ids are OBSERVATION ids. Observations are the bulk of the corpus and
the only ontology both arms index today; records (GitHub) have their own
``vec_records`` table and can be added as a second mode when something needs
measuring there.
"""

from __future__ import annotations

from collections.abc import Callable

from aggregator.core.dsl import parse
from aggregator.core.hybrid import rrf_fuse

#: ``(query_text, limit) -> ranked result ids``.
SearchFn = Callable[[str, int], list[str]]

#: Modes the harness can measure today. Criterion H owns the user-facing
#: version of this list; this one exists so an eval run names what it measured.
SEARCH_MODES = ("lexical", "hybrid")

#: Per-arm depth for the fused mode. Deliberately generous: fusion quality is
#: sensitive to arm depth, and the harness exists to make that measurable.
FUSION_ARM_DEPTH = 150


def lexical_search_fn(store) -> SearchFn:
    """FTS5 arm only, in the store's own result order.

    NOT a relevance ranking — ``query_observations`` orders by timestamp — and
    that is not a defect to fix here. The harness measures what the pipeline
    returns; if the ordering is wrong, the fix belongs in the pipeline and this
    harness is how you will see it land.
    """

    def search(text: str, limit: int) -> list[str]:
        ast = parse(text)
        rows = store.query_observations(ast, limit=limit)
        return [row.obs_id for row in rows]

    return search


def hybrid_search_fn(store, embedder) -> SearchFn:
    """FTS5 + vector KNN, fused with RRF, in RRF rank order.

    THE RRF ORDERING IS KEPT. ``aggregator.mcp`` computes the same fusion and
    then discards the ranking, because its page tokens address a recency-
    ordered stream and re-sorting would silently invalidate every token already
    handed out. The eval has no page tokens and every reason to see the
    relevance order, so it keeps it. Say so plainly: reading the two call sites
    side by side, the difference looks like a bug in one of them.

    Raises ``VectorIndexUnavailableError`` when the vector arm is missing.
    """

    def search(text: str, limit: int) -> list[str]:
        embedding = embedder.embed_query(text)
        vec_ids = store._vec_obs_ids(embedding, FUSION_ARM_DEPTH)
        # NOTHING IS CAUGHT HERE, AND THE ABSENCE IS THE FEATURE. This call
        # used to sit inside ``except sqlite3.OperationalError: fts_ids = []``,
        # under a comment calling it "Criterion B's bug: an unescaped MATCH".
        # b4eab9b whitelisted every string that reaches MATCH, so that cause is
        # gone — demonstrated over all 86 frozen golden queries in
        # ``tests/evals/test_search_modes.py``, the 25 that used to raise
        # included.
        #
        # A swallow whose reason has been removed is not harmless, it is a
        # trap. What can still raise here is a locked cache, a corrupt index or
        # an FTS5 that changed under us, and catching any of those would fuse
        # the vector arm alone, return a full-looking result list and report it
        # as ``hybrid`` — the NEVER DEGRADE rule in this module's own docstring,
        # broken by this module. So it propagates, the run stops, and the
        # operator finds out.
        fts_ids = store._fts_obs_ids(text)[:FUSION_ARM_DEPTH]
        return [doc_id for doc_id, _ in rrf_fuse(fts_ids, vec_ids)][:limit]

    return search


def resolve_search_fn(mode: str, store, embedder=None) -> SearchFn:
    """Build the search callable for ``mode``, or raise saying why not."""
    if mode == "lexical":
        return lexical_search_fn(store)
    if mode == "hybrid":
        if embedder is None:
            raise ValueError(
                "mode 'hybrid' needs an embedder; pass one rather than letting the "
                "run silently measure the lexical arm"
            )
        return hybrid_search_fn(store, embedder)
    raise ValueError(f"unknown search mode {mode!r}; expected one of {SEARCH_MODES}")
