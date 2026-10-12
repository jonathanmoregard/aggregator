"""The search callables the harness measures.

The harness is deliberately id-list-in: it takes a ``(query, limit) -> [ids]``
callable and knows nothing about how the ids were produced. That is what lets
the same golden set and the same drift metric measure the store's keyword arm
and the whole server surface.

ONE RULE THESE FUNCTIONS FOLLOW: they never degrade. An eval harness that
silently measured a different system would report "no drift" — the exact
"empty-result-looks-like-success" failure the project bans.
"""

import sqlite3
from datetime import UTC, datetime

import pytest

from aggregator.core.store import Store
from aggregator.evals.search import (
    SEARCH_MODES,
    lexical_search_fn,
    resolve_search_fn,
)
from aggregator.sources.base import ObservationRow, SessionRow


def _session(session_id, day=1):
    ts = datetime(2026, 7, day, 8, 0, tzinfo=UTC)
    return SessionRow(
        session_id=session_id,
        root_session_id=session_id,
        parent_session_id=None,
        kind="session",
        agent_id=None,
        agent_type=None,
        spawned_by_tool_use_id=None,
        cwd="/x",
        git_branch="main",
        first_ts=ts,
        last_ts=ts,
        jsonl_path=f"/tmp/{session_id}.jsonl",
    )


def _obs(obs_id, session_id, body, day=1):
    return ObservationRow(
        obs_id=obs_id,
        session_id=session_id,
        root_session_id=session_id,
        parent_obs_id=None,
        type="user",
        ts=datetime(2026, 7, day, 8, 0, tzinfo=UTC),
        model=None,
        input_tokens=None,
        output_tokens=None,
        tool_name=None,
        tool_use_id=None,
        body=body,
    )


def _seed(store, docs):
    entities = []
    for obs_id, body, day in docs:
        sid = f"s-{obs_id}"
        entities.append(_session(sid, day=day))
        entities.append(_obs(obs_id, sid, body, day=day))
    store.upsert_entities(entities)


DOCS = [
    ("o1", "quadratic voting is the governance mechanism we picked", 1),
    ("o2", "quadratic funding differs from quadratic voting", 2),
    ("o3", "the pigeon will roost on the balcony", 3),
]


@pytest.fixture
def store(tmp_path):
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    _seed(s, DOCS)
    return s


# --- lexical ----------------------------------------------------------------


def test_lexical_search_returns_the_observations_that_match(store):
    hits = lexical_search_fn(store)("quadratic", 50)
    assert set(hits) == {"o1", "o2"}


def test_lexical_search_respects_the_limit(store):
    assert len(lexical_search_fn(store)("quadratic", 1)) == 1


def test_lexical_search_returns_nothing_for_an_absent_term(store):
    assert lexical_search_fn(store)("axolotl", 50) == []


def test_a_query_fts5_could_not_parse_now_searches_for_its_words(store):
    """``power-on`` used to raise ``no such column: on`` and the store
    abstained. ``fts5_match_query`` rewrites it to ``"power" "on"``, so it is
    an ordinary two-term search. Seeded so the answer is non-empty, because an
    empty list here would pass under either regime."""
    _seed(store, [("o4", "the power-on self test writes to the console", 4)])
    assert lexical_search_fn(store)("power-on", 50) == ["o4"]


def test_no_frozen_golden_query_can_make_the_lexical_arm_raise(store):
    """Every frozen golden query runs through the shipped FTS5 arm without a
    syntax error. Anything that raises here is a query the whitelist missed."""
    from aggregator.evals.golden import load_golden_queries

    raised: list[tuple[str, str]] = []
    for query in load_golden_queries():
        try:
            store._fts_obs_ids(query.query)
        except sqlite3.OperationalError as e:  # pragma: no cover - the assertion
            raised.append((query.id, str(e)))
    assert raised == [], f"the FTS5 arm raises for frozen golden queries: {raised}"


# --- mode resolution --------------------------------------------------------


def test_the_modes_on_offer_are_named(store):
    assert set(SEARCH_MODES) == {"lexical", "mcp"}


def test_resolving_an_unknown_mode_fails_loudly(store):
    with pytest.raises(ValueError, match="telepathy"):
        resolve_search_fn("telepathy", store)


def test_resolving_lexical_gives_a_working_callable(store):
    assert set(resolve_search_fn("lexical", store)("quadratic", 50)) == {"o1", "o2"}


# --- mcp: the surface the agent actually queries ----------------------------
#
# ``lexical`` talks to the Store, so it cannot see anything in
# ``aggregator.mcp``: not the confidence signal, not routing, not the response
# layer.


def test_mcp_mode_routes_through_the_server_entry_point(store, monkeypatch):
    """Spy on the exact callable ``aggregator.mcp`` exposes. Reimplementing
    the server's routing here would be a second pipeline to keep in step."""
    import aggregator.mcp as mcp_mod
    from aggregator.evals.search import mcp_search_fn

    seen: list[dict] = []
    real = mcp_mod.aggregator_query

    def spy(**kwargs):
        seen.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(mcp_mod, "aggregator_query", spy)
    mcp_search_fn(store)("quadratic", 50)
    assert seen, "the mcp mode did not reach aggregator.mcp.aggregator_query"
    assert seen[0]["dsl"] == "quadratic"


def test_mcp_mode_returns_the_ids_the_server_returned(store):
    """SESSION ids, not observation ids: the server answers a free-text query
    with cards, and this mode reports what the agent is handed rather than what
    the store holds. Baselines are per-mode, so the two id spaces never meet."""
    from aggregator.evals.search import mcp_search_fn

    assert set(mcp_search_fn(store)("quadratic", 50)) == {"s-o1", "s-o2"}
    assert set(lexical_search_fn(store)("quadratic", 50)) == {"o1", "o2"}


def test_mcp_mode_respects_the_limit(store):
    from aggregator.evals.search import mcp_search_fn

    assert len(mcp_search_fn(store)("quadratic", 1)) == 1


def test_mcp_mode_stops_when_the_server_refuses_a_query(store, monkeypatch):
    """``ok: False`` is a refusal, not a result. Reading it as zero hits would
    score a server error as maximum drift and bury the reason."""
    import aggregator.mcp as mcp_mod
    from aggregator.evals.search import McpModeUnavailableError, mcp_search_fn

    monkeypatch.setattr(
        mcp_mod,
        "aggregator_query",
        lambda **kw: {"ok": False, "reason": "cache unavailable: disk I/O error"},
    )
    with pytest.raises(McpModeUnavailableError, match="disk I/O error"):
        mcp_search_fn(store)("quadratic", 50)


def test_resolving_mcp_gives_a_working_callable(store):
    assert set(resolve_search_fn("mcp", store)("quadratic", 50)) == {"s-o1", "s-o2"}
