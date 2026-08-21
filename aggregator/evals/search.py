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

RESULT IDS ARE NOT THE SAME KIND OF THING IN EVERY MODE, and pretending they
were would be the quietest possible bug. ``lexical`` and ``hybrid`` return
OBSERVATION ids, because they call the Store and observations are the bulk of
the corpus. ``mcp`` returns whatever the SERVER hands its caller, which for a
free-text query is session and record ``stable_id``s. Baselines are stored per
mode and are never compared across modes, so the two id spaces never meet — but
a drift number from one mode is not comparable with one from another, and
``harness.MODE_SCOPE`` says so in the printed report.

WHY ``mcp`` EXISTS AT ALL. The two store-level modes cannot see anything in
``aggregator.mcp``: not the RRF membership rule, not ``hybrid.vector_floor``,
not the confidence signal, not pagination, not the response shape. Criteria D,
G and H each changed exactly that module and each measured 0.000 drift — a
number produced by running a metric over a path those criteria did not touch.
That is a structural blind spot rather than a result, and a harness that cannot
see the layer under test must not report a confident zero.
"""

from __future__ import annotations

from collections.abc import Callable

from aggregator.core.dsl import parse

#: Per-arm depth for the fused mode, IMPORTED AND NOT RE-DECLARED.
#:
#: This module used to carry its own ``= 150`` beside ``aggregator.mcp``'s own
#: ``= 50``, so the pipeline the harness measured was not the pipeline the
#: server served and a clean eval run described a configuration nobody used.
#: The two numbers agree today; re-typing either is how they stop agreeing
#: without anyone noticing. Re-exported under this module's name because the
#: harness and its tests read it from here.
from aggregator.core.hybrid import FUSION_ARM_DEPTH, rrf_fuse

#: ``(query_text, limit) -> ranked result ids``.
SearchFn = Callable[[str, int], list[str]]

#: Modes the harness can measure today. ``lexical`` and ``hybrid`` are ARMS,
#: read straight off the Store; ``mcp`` is not an arm at all but the whole
#: server surface, driven through the same entry point the agent calls.
#: ``harness.MODE_SCOPE`` carries one note per entry here and is asserted to
#: cover all of them.
SEARCH_MODES = ("lexical", "hybrid", "mcp")


class McpModeUnavailableError(RuntimeError):
    """The ``mcp`` mode could not measure the server path it names.

    Its own type so the CLI can report it as "the harness could not run"
    (exit 2) rather than as a retrieval regression. The alternative — carrying
    on — is a run that measured FTS5, filed the numbers under ``mcp``, and said
    nothing.
    """


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
        # THE RAW ARM, DELIBERATELY UN-FLOORED. ``hybrid.vector_floor`` runs in
        # ``mcp._fused_id_scope`` and is measurable through the ``mcp`` mode
        # below; keeping it out of this arm is what makes the two modes
        # separable — drift between them is the floor's contribution, and a
        # mode that also applied it would have nothing to compare against.
        vec_ids = [doc_id for doc_id, _ in store._vec_obs_scored(embedding, FUSION_ARM_DEPTH)]
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


def _item_id(item: dict) -> str | None:
    """The id an MCP result item is addressed by, whichever shape it has.

    Session cards and records carry ``stable_id``; drilldown observation rows
    carry ``obs_id``. Neither key is optional in its own shape, so a ``None``
    here means the response grew a fourth item shape and this function has to
    learn it — which is why it returns ``None`` instead of guessing.
    """
    return item.get("stable_id") or item.get("obs_id")


def mcp_search_fn(store) -> SearchFn:
    """THE WHOLE SERVER, through the entry point the agent calls.

    Not an arm and not a reimplementation of one: this drives
    ``aggregator.mcp.aggregator_query`` with default arguments — no
    ``search_mode``, no ``rerank``, no ``fields`` — so what is measured is the
    call an agent makes. Everything the store-level modes are blind to is in
    scope here: RRF membership, ``hybrid.vector_floor``, the confidence signal,
    routing and the response shape.

    RESOLVED LAZILY, INSIDE THE CLOSURE'S MODULE IMPORT. ``aggregator.mcp`` is
    imported at build time but ``aggregator_query`` is looked up on the module
    at call time, so a test that patches the module attribute patches what this
    calls — and, more importantly, so does anything that reloads the server.

    REFUSES A COLD VECTOR ARM rather than measuring the fallback. The server
    degrades to FTS5 when the vector index is empty, silently and correctly for
    a user; measured here that fallback would file the keyword arm's answer
    under ``mcp`` and quietly drop the floor and the fusion out of a run that
    claimed to cover them. This is the common case, not an edge one: a full
    backfill is 25-30 days of CPU, so most caches this is pointed at will be
    cold.

    REFUSES ``ok: False`` rather than reading it as zero hits. A refusal scored
    as an empty result list is maximum drift with its reason thrown away.

    ``page_size=limit`` AND ONE PAGE ONLY. The harness asks for ``RUN_DEPTH``
    ids; paginating to reach more would measure the page-token path as well and
    make every run depend on token stability, which is a separate question with
    its own tests.
    """
    import aggregator.mcp as mcp_mod

    warm = [
        kind
        for kind in ("observations", "records")
        if store.has_embedded_rows(kind)
    ]
    if not warm:
        raise McpModeUnavailableError(
            "mode 'mcp' measures the server's hybrid path, and nothing in this "
            "cache is embedded — every query would be answered by FTS5 alone "
            "and reported as 'mcp'. Run `aggregator embed --catchup --source "
            "both` first, or measure the keyword arm honestly with "
            "--mode lexical."
        )

    def search(text: str, limit: int) -> list[str]:
        result = mcp_mod.aggregator_query(dsl=text, page_size=limit, _store=store)
        if not result.get("ok"):
            raise McpModeUnavailableError(
                f"the server refused the golden query {text!r}: "
                f"{result.get('reason', result)}"
            )
        ids = [_item_id(item) for item in result.get("records", [])]
        unknown = sum(1 for i in ids if i is None)
        if unknown:
            raise McpModeUnavailableError(
                f"{unknown} of {len(ids)} items in the response for {text!r} "
                "carry neither 'stable_id' nor 'obs_id'; the result shape "
                "changed and evals/search._item_id has to learn it rather than "
                "silently score a shorter list"
            )
        return [i for i in ids if i][:limit]

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
    if mode == "mcp":
        # NO EMBEDDER IS PASSED, and that is not an oversight. The server builds
        # its own lazily, and handing it one from here would measure a wiring
        # that does not ship.
        return mcp_search_fn(store)
    raise ValueError(f"unknown search mode {mode!r}; expected one of {SEARCH_MODES}")
