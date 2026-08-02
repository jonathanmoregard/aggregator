"""Tests for aggregator_query MCP tool (v2, Schema B).

Two routing modes:

* Records path (github): default when no session keys and no ``source:sessions``.
* Sessions path: any of ``session:``, ``top:``, ``agent:``, ``type:``,
  ``active:`` OR ``source:sessions``. Default returns session-level hit list
  with ``matching_observations`` per session; ``drilldown=True`` returns
  observation rows.
"""
from __future__ import annotations

from datetime import UTC, datetime

from aggregator.core.store import Store
from aggregator.mcp import aggregator_query
from aggregator.sources.base import ObservationRow, Record, SessionRow


def _rec(sid: str, subject: str, body: str, tags=()) -> Record:
    return Record(
        stable_id=sid,
        source="github",
        subject=subject,
        body=body,
        tags=list(tags),
        created_at=datetime(2026, 7, 25, tzinfo=UTC),
        updated_at=datetime(2026, 7, 25, tzinfo=UTC),
    )


def _sess(session_id: str, *, kind="session", root=None, parent=None, agent_id=None):
    return SessionRow(
        session_id=session_id,
        root_session_id=root or session_id,
        parent_session_id=parent,
        kind=kind,
        agent_id=agent_id,
        agent_type=None,
        spawned_by_tool_use_id=None,
        cwd=None,
        git_branch=None,
        first_ts=datetime(2026, 7, 25, 10, 0, tzinfo=UTC),
        last_ts=datetime(2026, 7, 25, 10, 5, tzinfo=UTC),
        jsonl_path="/tmp/x.jsonl",
    )


def _obs(obs_id: str, session_id: str, body: str, *, obs_type="user", root=None):
    return ObservationRow(
        obs_id=obs_id,
        session_id=session_id,
        root_session_id=root or session_id,
        parent_obs_id=None,
        type=obs_type,
        ts=datetime(2026, 7, 25, 10, 0, tzinfo=UTC),
        model=None,
        input_tokens=None,
        output_tokens=None,
        tool_name=None,
        tool_use_id=None,
        body=body,
    )


def _seed_records(store, n=1):
    store.migrate()
    store.upsert(
        [
            _rec(
                f"github:acme/api:{i}",
                f"pr {i}",
                f"refactor foo{i}.py",
                tags=["pr"],
            )
            for i in range(n)
        ]
    )


def _seed_sessions(store):
    store.migrate()
    store.upsert_entities(
        [
            _sess("sess-alpha"),
            _obs("u-alpha-1", "sess-alpha", "please refactor foo.py"),
            _obs("a-alpha-1", "sess-alpha", "sure, refactored", obs_type="assistant"),
        ]
    )


# --- Records path (github) ------------------------------------------------


def test_query_records_happy_path(tmp_data_home):
    store = Store()
    _seed_records(store)
    result = aggregator_query(dsl="source:github", fields="full", _store=store)
    assert result["ok"] is True
    assert result["mode"] == "records"
    assert result["total"] == 1


def test_query_records_wraps_external_content(tmp_data_home):
    store = Store()
    _seed_records(store)
    result = aggregator_query(dsl="source:github", fields="full", _store=store)
    body = result["records"][0]["content"]
    assert '<ExternalContent source="github:acme/api:0">' in body


def test_query_records_summary_omits_body(tmp_data_home):
    store = Store()
    _seed_records(store)
    result = aggregator_query(dsl="source:github", fields="summary", _store=store)
    body = result["records"][0]["content"]
    assert "refactor foo0.py" not in body
    assert "notice" in result


def test_query_records_full_includes_body(tmp_data_home):
    store = Store()
    _seed_records(store)
    result = aggregator_query(dsl="source:github", fields="full", _store=store)
    body = result["records"][0]["content"]
    assert "refactor foo0.py" in body


def test_query_records_pagination(tmp_data_home):
    store = Store()
    _seed_records(store, n=5)
    result = aggregator_query(
        dsl="source:github", fields="full", page_size=2, _store=store
    )
    assert len(result["records"]) == 2
    assert result["total"] == 5
    assert "next_page_token" in result


def test_query_records_pagination_second_page(tmp_data_home):
    store = Store()
    _seed_records(store, n=3)
    first = aggregator_query(
        dsl="source:github", fields="full", page_size=2, _store=store
    )
    second = aggregator_query(
        dsl="source:github",
        fields="full",
        page_size=2,
        page_token=first["next_page_token"],
        _store=store,
    )
    assert second["ok"] is True
    assert len(second["records"]) == 1


def test_query_bad_dsl_returns_structured_error(tmp_data_home):
    store = Store()
    store.migrate()
    result = aggregator_query(dsl="from:not-a-date", _store=store)
    assert result["ok"] is False
    assert "reason" in result
    assert "remediation" in result
    assert "Traceback" not in result["reason"]


def test_query_bad_fts_syntax_returns_structured_error(tmp_data_home):
    store = Store()
    _seed_records(store)
    result = aggregator_query(dsl='"unbalanced', _store=store)
    assert result["ok"] is False
    assert "reason" in result
    assert "remediation" in result


# --- Sessions path --------------------------------------------------------


def test_query_sessions_default_returns_session_hit_list(tmp_data_home):
    store = Store()
    _seed_sessions(store)
    result = aggregator_query(dsl="source:sessions", fields="full", _store=store)
    assert result["ok"] is True
    assert result["mode"] == "sessions"
    assert result["total"] == 1
    card = result["records"][0]
    assert card["stable_id"] == "sess-alpha"
    assert card["matching_observations"] == 2  # user + assistant
    assert "refactor foo.py" in card["subject"]


def test_query_sessions_drilldown_returns_observations(tmp_data_home):
    store = Store()
    _seed_sessions(store)
    result = aggregator_query(
        dsl="source:sessions", fields="full", drilldown=True, _store=store
    )
    assert result["mode"] == "observations"
    assert result["total"] == 2
    ids = {r["obs_id"] for r in result["records"]}
    assert ids == {"u-alpha-1", "a-alpha-1"}


def test_query_session_key_routes_to_sessions_path(tmp_data_home):
    store = Store()
    _seed_sessions(store)
    result = aggregator_query(dsl="session:sess-alpha", fields="summary", _store=store)
    assert result["mode"] == "sessions"


def test_query_agent_key_routes_to_sessions_path(tmp_data_home):
    store = Store()
    _seed_sessions(store)
    result = aggregator_query(dsl="agent:foo", fields="summary", _store=store)
    assert result["mode"] == "sessions"


def test_query_type_key_routes_to_sessions_path(tmp_data_home):
    store = Store()
    _seed_sessions(store)
    result = aggregator_query(dsl="type:user", fields="summary", _store=store)
    assert result["mode"] == "sessions"


def test_query_active_key_routes_to_sessions_path(tmp_data_home):
    store = Store()
    _seed_sessions(store)
    result = aggregator_query(
        dsl="active:2026-07-01..2026-07-31", fields="summary", _store=store
    )
    assert result["mode"] == "sessions"


def test_top_returns_single_top_row_when_top_and_subagents_present(tmp_data_home):
    """B2 plan test: seed 1 top + 2 subagents; top:X returns 1 row (the top).

    Regression guard: this codifies the intended semantics from the plan --
    top: yields exactly the top-level session, session: yields top + subagents,
    agent: yields just the named subagent.
    """
    store = Store()
    store.migrate()
    store.upsert_entities(
        [
            _sess("root-b2"),
            _sess("root-b2:aY", kind="subagent", root="root-b2",
                  parent="root-b2", agent_id="aY"),
            _sess("root-b2:aZ", kind="subagent", root="root-b2",
                  parent="root-b2", agent_id="aZ"),
        ]
    )
    r_top = aggregator_query(dsl="top:root-b2", _store=store)
    r_sess = aggregator_query(dsl="session:root-b2", _store=store)
    r_agent = aggregator_query(dsl="agent:aY", _store=store)
    assert r_top["total"] == 1, r_top
    assert r_sess["total"] == 3, r_sess
    assert r_agent["total"] == 1, r_agent


def test_matching_observations_correct_for_subagents(tmp_data_home):
    """B3 regression: session-level cards show real match counts even for
    subagent hits.

    Bug: ``matching_observations`` used ``session_id`` filter which routed
    through ``root_session_id``. For subagents that filter never matched
    because their obs carry the PARENT's root_session_id, not the composite
    subagent id. Fix: filter observations by exact session_id (top:) and
    keep the FTS text so counts reflect actual matches.
    """
    store = Store()
    store.migrate()
    parent = "parent-b3"
    sub = f"{parent}:agentB3"
    store.upsert_entities(
        [
            _sess(parent),
            _sess(sub, kind="subagent", root=parent, parent=parent,
                  agent_id="agentB3"),
            # observation on the subagent — root_session_id = parent
            _obs(
                "obs-b3-1", sub, "please suggest_doc_edit for foo",
                obs_type="tool_use", root=parent,
            ),
            _obs(
                "obs-b3-2", sub, "unrelated body", obs_type="assistant",
                root=parent,
            ),
        ]
    )
    r = aggregator_query(dsl="type:tool_use suggest_doc_edit", _store=store)
    assert r["total"] >= 1
    subagent_cards = [c for c in r["records"] if c["stable_id"] == sub]
    assert subagent_cards, f"expected a card for subagent, got {r['records']}"
    card = subagent_cards[0]
    assert card["matching_observations"] == 1, (
        f"expected matching_observations=1 for the tool_use hit, "
        f"got {card['matching_observations']}"
    )


def test_top_synthesises_orphan_root_when_top_not_ingested(tmp_data_home):
    """B2 real-data repro: subagents present but top-level JSONL was live
    at ingest time so parent session_id row was never written. Users still
    want ``top:X`` to surface the session (via subagent roots) rather than
    silently return 0. Fix: synthesise a minimal orphan-root SessionRow so
    the caller sees "yes, this session exists" and can then drill down.
    """
    store = Store()
    store.migrate()
    store.upsert_entities(
        [
            # NO top row -- simulates live-file skip at ingest
            _sess("orphan-root:aA", kind="subagent", root="orphan-root",
                  parent="orphan-root", agent_id="aA"),
            _sess("orphan-root:aB", kind="subagent", root="orphan-root",
                  parent="orphan-root", agent_id="aB"),
        ]
    )
    r_top = aggregator_query(dsl="top:orphan-root", _store=store)
    assert r_top["total"] == 1, (
        f"top: should synthesise orphan-root row when subagents reference it "
        f"but the top-level JSONL was skipped at ingest (got total={r_top['total']})"
    )
    # Session queries should still return the two subagents (no orphan for
    # session: because subagents already carry the root).
    r_sess = aggregator_query(dsl="session:orphan-root", _store=store)
    assert r_sess["total"] == 2, r_sess


def test_query_sessions_observations_wrapped_content(tmp_data_home):
    store = Store()
    _seed_sessions(store)
    result = aggregator_query(
        dsl="source:sessions", fields="full", drilldown=True, _store=store
    )
    for item in result["records"]:
        assert "<ExternalContent" in item["content"]
        assert "</ExternalContent>" in item["content"]


def test_query_sessions_scrubs_observation_bodies_on_return(tmp_data_home):
    """Pre-return scrub catches secrets even if they slipped past pre-write."""
    store = Store()
    store.migrate()
    # Bypass upsert_entities scrub by writing raw into the observations table.
    session = _sess("sess-leak")
    store.upsert_entities([session])
    secret = "sk-" + "ant-" + "api03-" + "z" * 44
    conn = store._c()
    conn.execute(
        "INSERT INTO observations(obs_id, session_id, root_session_id, "
        "parent_obs_id, type, ts, model, input_tokens, output_tokens, "
        "tool_name, tool_use_id, body) VALUES "
        "('o-leak', 'sess-leak', 'sess-leak', NULL, 'user', "
        "'2026-07-25T10:00:00+00:00', NULL, NULL, NULL, NULL, NULL, ?)",
        (secret,),
    )
    conn.commit()
    result = aggregator_query(
        dsl="source:sessions", fields="full", drilldown=True, _store=store
    )
    for item in result["records"]:
        assert "sk-ant-api03" not in item["content"]


def test_query_sessions_notice_when_summary(tmp_data_home):
    store = Store()
    _seed_sessions(store)
    result = aggregator_query(dsl="source:sessions", _store=store)
    assert "notice" in result


def test_summary_mode_omits_external_content_wrap(tmp_data_home):
    """M1: summary mode returns hit list without bodies; wrapping the empty
    body in <ExternalContent> is misleading (looks like real content is
    present when it isn't). Wrap only in drilldown / fields=full.
    """
    store = Store()
    _seed_sessions(store)
    result = aggregator_query(
        dsl="source:sessions", fields="summary", _store=store
    )
    for rec in result["records"]:
        assert "<ExternalContent" not in rec["content"], (
            f"summary mode should not wrap; got: {rec['content']!r}"
        )
    # Full mode must still wrap (parity with prior behaviour).
    result_full = aggregator_query(
        dsl="source:sessions", fields="full", _store=store
    )
    for rec in result_full["records"]:
        assert "<ExternalContent" in rec["content"]


def test_records_summary_omits_external_content_wrap(tmp_data_home):
    """M1 records-path parity: same behaviour for github-shaped hits."""
    store = Store()
    _seed_records(store)
    result = aggregator_query(
        dsl="source:github", fields="summary", _store=store
    )
    for rec in result["records"]:
        assert "<ExternalContent" not in rec["content"], (
            f"summary mode should not wrap; got: {rec['content']!r}"
        )


def test_query_scrubs_records_secrets_on_return(tmp_data_home):
    """Pre-return scrub of Record path (parity with pre-v2 behaviour)."""
    store = Store()
    store.migrate()
    secret = "sk-" + "ant-" + "api03-" + "x" * 44
    conn = store._c()
    conn.execute(
        "INSERT INTO records(stable_id, source, subject, body, tags, "
        "created_at, updated_at, extra) VALUES "
        "('github:leak', 'github', 's', ?, '[]', NULL, NULL, '{}')",
        (secret,),
    )
    conn.execute(
        "INSERT INTO records_fts(stable_id, source, subject, body, tags) "
        "VALUES ('github:leak', 'github', 's', ?, '')",
        (secret,),
    )
    conn.commit()
    result = aggregator_query(dsl="source:github", fields="full", _store=store)
    assert result["ok"] is True
    for rec in result["records"]:
        assert "sk-ant-api03" not in rec["content"]
