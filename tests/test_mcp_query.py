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
