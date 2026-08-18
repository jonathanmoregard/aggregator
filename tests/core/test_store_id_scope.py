"""``QueryAST.id_scope`` — the seam the hybrid retriever hands ids through.

WHY A SEPARATE FIELD RATHER THAN REUSING ``text``. The FTS arm and the vector
arm produce id lists, and RRF fuses them into a third id list that no FTS5
MATCH expression can express. So the MCP layer computes the fused ids and asks
the store to filter on exactly those, with every other filter (date, type,
source, session scope) still applied by the store as usual. ``id_scope`` is
INTERNAL: it has no DSL key, no parser support, and no way for a caller to set
it — the DSL surface is unchanged.

The empty set is the interesting case. ``id_scope=None`` means "no id filter";
``id_scope=frozenset()`` means "nothing matched", which must render as a
no-match clause and not as SQL ``IN ()``, which is a syntax error rather than
an empty result.
"""

from datetime import UTC, datetime

import pytest

from aggregator.core.store import Store
from aggregator.sources.base import ObservationRow, QueryAST, Record, SessionRow


@pytest.fixture
def store(tmp_path):
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    return s


def _rec(stable_id, body="body", day=1):
    ts = datetime(2026, 7, day, 8, 0, tzinfo=UTC)
    return Record(
        stable_id=stable_id,
        source="github",
        subject=f"subject {stable_id}",
        body=body,
        tags=[],
        created_at=ts,
        updated_at=ts,
    )


def _session(session_id, kind="session", root=None, day=1):
    ts = datetime(2026, 7, day, 8, 0, tzinfo=UTC)
    return SessionRow(
        session_id=session_id,
        root_session_id=root or session_id,
        parent_session_id=None,
        kind=kind,
        agent_id=None,
        agent_type=None,
        spawned_by_tool_use_id=None,
        cwd="/x",
        git_branch="main",
        first_ts=ts,
        last_ts=ts,
        jsonl_path=f"/tmp/{session_id}.jsonl",
    )


def _obs(obs_id, session_id, root=None, body="body", obs_type="user", day=1):
    return ObservationRow(
        obs_id=obs_id,
        session_id=session_id,
        root_session_id=root or session_id,
        parent_obs_id=None,
        type=obs_type,
        ts=datetime(2026, 7, day, 8, 0, tzinfo=UTC),
        model=None,
        input_tokens=None,
        output_tokens=None,
        tool_name=None,
        tool_use_id=None,
        body=body,
    )


# --- records ontology --------------------------------------------------------


def test_records_id_scope_restricts_to_those_ids(store):
    store.upsert([_rec(f"github:{i}") for i in range(5)])
    ast = QueryAST(id_scope=frozenset({"github:1", "github:3"}))
    got = {r.stable_id for r in store.query(ast)}
    assert got == {"github:1", "github:3"}


def test_records_empty_id_scope_matches_nothing_without_a_sql_error(store):
    """``IN ()`` is a syntax error, not an empty set. An arm that fused to
    nothing must return no rows, not blow up the tool."""
    store.upsert([_rec(f"github:{i}") for i in range(3)])
    ast = QueryAST(id_scope=frozenset())
    assert store.query(ast) == []
    assert store.count(ast) == 0


def test_records_id_scope_none_is_no_filter(store):
    store.upsert([_rec(f"github:{i}") for i in range(3)])
    assert len(store.query(QueryAST())) == 3
    assert len(store.query(QueryAST(id_scope=None))) == 3


def test_records_count_honours_id_scope(store):
    store.upsert([_rec(f"github:{i}") for i in range(5)])
    ast = QueryAST(id_scope=frozenset({"github:0", "github:2", "github:4"}))
    assert store.count(ast) == 3


def test_records_id_scope_intersects_with_other_filters(store):
    """The fused id list narrows; it never widens past the caller's filters."""
    store.upsert([_rec(f"github:{i}", day=1 + i) for i in range(5)])
    ast = QueryAST(
        id_scope=frozenset({"github:0", "github:1", "github:4"}),
        from_date=datetime(2026, 7, 4, tzinfo=UTC),
    )
    assert {r.stable_id for r in store.query(ast)} == {"github:4"}


def test_records_id_scope_survives_pagination(store):
    store.upsert([_rec(f"github:{i}", day=1 + i) for i in range(6)])
    scope = frozenset({"github:0", "github:2", "github:4"})
    page1 = store.query(QueryAST(id_scope=scope), limit=2, offset=0)
    page2 = store.query(QueryAST(id_scope=scope), limit=2, offset=2)
    ids = [r.stable_id for r in [*page1, *page2]]
    assert sorted(ids) == ["github:0", "github:2", "github:4"]
    assert len(set(ids)) == 3


# --- observations ontology ---------------------------------------------------


def test_observations_id_scope_restricts_to_those_ids(store):
    store.upsert_entities(
        [
            _session("s1"),
            *[_obs(f"o{i}", "s1") for i in range(5)],
        ]
    )
    ast = QueryAST(id_scope=frozenset({"o1", "o4"}))
    assert {o.obs_id for o in store.query_observations(ast)} == {"o1", "o4"}


def test_observations_empty_id_scope_matches_nothing(store):
    store.upsert_entities([_session("s1"), _obs("o0", "s1")])
    ast = QueryAST(id_scope=frozenset())
    assert store.query_observations(ast) == []
    assert store.count_observations(ast) == 0


def test_observations_count_honours_id_scope(store):
    store.upsert_entities(
        [_session("s1"), *[_obs(f"o{i}", "s1") for i in range(5)]]
    )
    ast = QueryAST(id_scope=frozenset({"o0", "o3"}))
    assert store.count_observations(ast) == 2


def test_observations_id_scope_intersects_with_type_filter(store):
    store.upsert_entities(
        [
            _session("s1"),
            _obs("o0", "s1", obs_type="user"),
            _obs("o1", "s1", obs_type="assistant"),
        ]
    )
    ast = QueryAST(id_scope=frozenset({"o0", "o1"}), obs_type="assistant")
    assert {o.obs_id for o in store.query_observations(ast)} == {"o1"}


# --- sessions ontology: obs ids project up to session cards ------------------


def test_sessions_id_scope_projects_obs_hits_up_to_session_cards(store):
    """A fused id list is obs ids; the session hit-list is session rows. The
    projection has to match what FTS already does — a hit anywhere under a
    root surfaces the top-level card."""
    store.upsert_entities(
        [
            _session("s1"),
            _session("s2"),
            _obs("o1", "s1"),
            _obs("o2", "s2"),
        ]
    )
    ast = QueryAST(id_scope=frozenset({"o2"}))
    assert {s.session_id for s in store.query_sessions(ast)} == {"s2"}
    assert store.count_sessions(ast) == 1


def test_sessions_id_scope_surfaces_the_root_when_a_subagent_matched(store):
    """Parity with ``_fts_hit_scope``: a hit inside a subagent stream must
    still surface the top-level session card."""
    store.upsert_entities(
        [
            _session("s1"),
            _session("s1:agent-a", kind="subagent", root="s1"),
            _obs("o-sub", "s1:agent-a", root="s1"),
        ]
    )
    ast = QueryAST(id_scope=frozenset({"o-sub"}))
    got = {s.session_id for s in store.query_sessions(ast)}
    assert "s1" in got
    assert "s1:agent-a" in got


def test_sessions_id_scope_does_not_surface_sibling_subagents(store):
    """A subagent card appears only when its OWN stream matched — the same
    rule ``_fts_hit_scope`` enforces for the FTS arm."""
    store.upsert_entities(
        [
            _session("s1"),
            _session("s1:agent-a", kind="subagent", root="s1"),
            _session("s1:agent-b", kind="subagent", root="s1"),
            _obs("o-a", "s1:agent-a", root="s1"),
            _obs("o-b", "s1:agent-b", root="s1"),
        ]
    )
    ast = QueryAST(id_scope=frozenset({"o-a"}))
    got = {s.session_id for s in store.query_sessions(ast)}
    assert got == {"s1", "s1:agent-a"}


def test_sessions_empty_id_scope_matches_nothing(store):
    store.upsert_entities([_session("s1"), _obs("o1", "s1")])
    ast = QueryAST(id_scope=frozenset())
    assert store.query_sessions(ast) == []
    assert store.count_sessions(ast) == 0


def test_sessions_id_scope_respects_the_type_filter(store):
    """``type:`` narrows which observations count as a hit, exactly as it
    does on the FTS arm."""
    store.upsert_entities(
        [
            _session("s1"),
            _session("s2"),
            _obs("o1", "s1", obs_type="user"),
            _obs("o2", "s2", obs_type="assistant"),
        ]
    )
    ast = QueryAST(id_scope=frozenset({"o1", "o2"}), obs_type="assistant")
    assert {s.session_id for s in store.query_sessions(ast)} == {"s2"}


def test_sessions_id_scope_survives_a_scope_bigger_than_one_sql_page(store):
    """The projection query chunks its parameters; more ids than one chunk
    must not silently drop the tail."""
    entities = [_session("s1")]
    entities += [_obs(f"o{i:04d}", "s1") for i in range(1200)]
    store.upsert_entities(entities)
    scope = frozenset(f"o{i:04d}" for i in range(1200))
    assert {s.session_id for s in store.query_sessions(QueryAST(id_scope=scope))} == {
        "s1"
    }
    assert store.count_observations(QueryAST(id_scope=scope)) == 1200
