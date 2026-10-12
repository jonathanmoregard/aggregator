"""The ``(query, limit) -> [result ids]`` callables the harness measures.

THE HARNESS IS ID-LIST-IN, ID-LIST-OUT. It knows nothing about stores or
ontologies — which is what lets the same frozen golden set and the same drift
metric measure the store's keyword arm and the full server surface. Adding a
mode here is adding one function and one registry entry.

THESE FUNCTIONS NEVER DEGRADE. A harness that silently measured something other
than the mode it names would file the numbers under the wrong name and report
no drift — the empty-result-looks-like-success failure this project bans by
name. So a mode that cannot run RAISES.

RESULT IDS ARE NOT THE SAME KIND OF THING IN EVERY MODE. ``lexical`` returns
OBSERVATION ids, because it calls the Store and observations are the bulk of
the corpus. ``mcp`` returns whatever the SERVER hands its caller, which for a
free-text query is session and record ``stable_id``s. Baselines are stored per
mode and are never compared across modes, so the two id spaces never meet — but
a drift number from one mode is not comparable with one from another, and
``harness.MODE_SCOPE`` says so in the printed report.

WHY ``mcp`` EXISTS AT ALL. The store-level mode cannot see anything in
``aggregator.mcp``: not the confidence signal, not routing, not pagination, not
the response shape. A harness that cannot see the layer under test must not
report a confident zero.
"""

from __future__ import annotations

from collections.abc import Callable

from aggregator.core.dsl import parse

#: ``(query_text, limit) -> ranked result ids``.
SearchFn = Callable[[str, int], list[str]]

#: Modes the harness can measure. ``lexical`` is read straight off the Store;
#: ``mcp`` is the whole server surface, driven through the same entry point
#: the agent calls. ``harness.MODE_SCOPE`` carries one note per entry here and
#: is asserted to cover all of them.
SEARCH_MODES = ("lexical", "mcp")


class McpModeUnavailableError(RuntimeError):
    """The ``mcp`` mode could not measure the server path it names.

    Its own type so the CLI can report it as "the harness could not run"
    (exit 2) rather than as a retrieval regression.
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
    ``rerank``, no ``fields`` — so what is measured is the call an agent
    makes. Everything the store-level mode is blind to is in scope here: the
    confidence signal, routing and the response shape.

    RESOLVED LAZILY, INSIDE THE CLOSURE'S MODULE IMPORT. ``aggregator.mcp`` is
    imported at build time but ``aggregator_query`` is looked up on the module
    at call time, so a test that patches the module attribute patches what this
    calls — and, more importantly, so does anything that reloads the server.

    REFUSES ``ok: False`` rather than reading it as zero hits. A refusal scored
    as an empty result list is maximum drift with its reason thrown away.

    ``page_size=limit`` AND ONE PAGE ONLY. The harness asks for ``RUN_DEPTH``
    ids; paginating to reach more would measure the page-token path as well and
    make every run depend on token stability, which is a separate question with
    its own tests.
    """
    import aggregator.mcp as mcp_mod

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


def resolve_search_fn(mode: str, store) -> SearchFn:
    """Build the search callable for ``mode``, or raise saying why not."""
    if mode == "lexical":
        return lexical_search_fn(store)
    if mode == "mcp":
        return mcp_search_fn(store)
    raise ValueError(f"unknown search mode {mode!r}; expected one of {SEARCH_MODES}")
