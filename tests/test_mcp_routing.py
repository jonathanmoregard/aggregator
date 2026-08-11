"""Routing tests for aggregator_query (v2, Schema B).

Formalises the two-ontology split introduced in v2:

* ``records`` + ``records_fts`` — row-per-unit-of-work sources (GitHub PRs +
  issues; future: Gmail, Calendar).
* ``sessions`` + ``observations`` + ``obs_fts`` — Claude Code conversation
  streams (Langfuse-derived).

Routing contract exercised here:

* ``source:github`` → records table.
* ``source:sessions`` → sessions table.
* Session-only keys (``session:``, ``top:``, ``agent:``, ``type:``,
  ``active:``) → sessions path. If combined with ``source:github`` the query
  returns empty with a structured ``notice`` explaining the ontology
  mismatch (records don't carry session ids).
* Records-only keys (``state:``, ``check:``, ``mergeable:``) → records path.
  Combined with ``source:sessions`` returns empty with a structured
  ``notice``.
* Cross-source date-only queries (``from:D`` / ``to:D`` with no source
  hint and no ontology-specific keys) hit BOTH tables and UNION the
  results — the caller asked "what happened in this window?" and the
  aggregator surfaces both PRs and sessions.
"""
from __future__ import annotations

from datetime import UTC, datetime

from aggregator.core.store import Store
from aggregator.mcp import aggregator_query
from aggregator.sources.base import ObservationRow, Record, SessionRow


def _rec(
    sid: str,
    subject: str = "s",
    body: str = "b",
    tags=(),
    created: datetime | None = None,
    updated: datetime | None = None,
) -> Record:
    default = datetime(2026, 7, 25, tzinfo=UTC)
    return Record(
        stable_id=sid,
        source="github",
        subject=subject,
        body=body,
        tags=list(tags),
        created_at=created or default,
        updated_at=updated or default,
    )


def _sess(
    session_id: str,
    *,
    kind="session",
    root=None,
    parent=None,
    agent_id=None,
    first_ts: datetime | None = None,
    last_ts: datetime | None = None,
    origin: str | None = None,
) -> SessionRow:
    kwargs = {} if origin is None else {"origin": origin}
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
        first_ts=first_ts or datetime(2026, 7, 25, 10, 0, tzinfo=UTC),
        last_ts=last_ts or datetime(2026, 7, 25, 10, 5, tzinfo=UTC),
        jsonl_path="/tmp/x.jsonl",
        **kwargs,
    )


def _obs(obs_id: str, session_id: str, body: str, *, obs_type="user", root=None,
         ts: datetime | None = None):
    return ObservationRow(
        obs_id=obs_id,
        session_id=session_id,
        root_session_id=root or session_id,
        parent_obs_id=None,
        type=obs_type,
        ts=ts or datetime(2026, 7, 25, 10, 0, tzinfo=UTC),
        model=None,
        input_tokens=None,
        output_tokens=None,
        tool_name=None,
        tool_use_id=None,
        body=body,
    )


def _seed_both(store: Store) -> None:
    """One github record and one session, at distinct dates."""
    store.migrate()
    store.upsert(
        [
            _rec(
                "github:acme/api:1",
                subject="pr in july",
                body="refactor foo",
                tags=["pr"],
                created=datetime(2026, 7, 20, tzinfo=UTC),
                updated=datetime(2026, 7, 20, tzinfo=UTC),
            ),
            _rec(
                "github:acme/api:2",
                subject="pr in june",
                body="old pr",
                tags=["pr"],
                created=datetime(2026, 6, 1, tzinfo=UTC),
                updated=datetime(2026, 6, 1, tzinfo=UTC),
            ),
        ]
    )
    store.upsert_entities(
        [
            _sess(
                "sess-july",
                first_ts=datetime(2026, 7, 22, tzinfo=UTC),
                last_ts=datetime(2026, 7, 22, 12, tzinfo=UTC),
            ),
            _obs(
                "obs-july-1",
                "sess-july",
                "chat body",
                ts=datetime(2026, 7, 22, tzinfo=UTC),
            ),
            _sess(
                "sess-may",
                first_ts=datetime(2026, 5, 1, tzinfo=UTC),
                last_ts=datetime(2026, 5, 1, 12, tzinfo=UTC),
            ),
            _obs(
                "obs-may-1",
                "sess-may",
                "old chat",
                ts=datetime(2026, 5, 1, tzinfo=UTC),
            ),
        ]
    )


# --- explicit source routing ----------------------------------------------


def test_source_github_returns_records_shape(tmp_data_home):
    store = Store()
    _seed_both(store)
    r = aggregator_query(dsl="source:github", _store=store)
    assert r["ok"] is True
    assert r["mode"] == "records"
    ids = {rec["stable_id"] for rec in r["records"]}
    assert ids == {"github:acme/api:1", "github:acme/api:2"}


def test_source_sessions_returns_sessions_shape(tmp_data_home):
    store = Store()
    _seed_both(store)
    r = aggregator_query(dsl="source:sessions", _store=store)
    assert r["ok"] is True
    assert r["mode"] == "sessions"
    ids = {rec["stable_id"] for rec in r["records"]}
    assert ids == {"sess-july", "sess-may"}


# --- session-only keys with wrong source ---------------------------------


def test_session_key_with_source_github_returns_notice(tmp_data_home):
    """session: on records-shaped source is an ontology mismatch.

    Records don't carry session ids — return empty results and surface a
    structured notice so the caller knows *why* the hit list is empty
    (not "no matches"; "records don't have this concept").
    """
    store = Store()
    _seed_both(store)
    r = aggregator_query(
        dsl="source:github session:sess-july", _store=store
    )
    assert r["ok"] is True
    assert r["total"] == 0
    assert r["records"] == []
    assert "notice" in r
    assert "session" in r["notice"].lower()


def test_type_key_with_source_github_returns_notice(tmp_data_home):
    store = Store()
    _seed_both(store)
    r = aggregator_query(
        dsl="source:github type:tool_use", _store=store
    )
    assert r["ok"] is True
    assert r["total"] == 0
    assert "notice" in r
    assert "type" in r["notice"].lower() or "session" in r["notice"].lower()


def test_top_key_with_source_github_returns_notice(tmp_data_home):
    store = Store()
    _seed_both(store)
    r = aggregator_query(dsl="source:github top:sess-july", _store=store)
    assert r["ok"] is True
    assert r["total"] == 0
    assert "notice" in r


def test_agent_key_with_source_github_returns_notice(tmp_data_home):
    store = Store()
    _seed_both(store)
    r = aggregator_query(dsl="source:github agent:foo", _store=store)
    assert r["ok"] is True
    assert r["total"] == 0
    assert "notice" in r


# --- records-only keys with wrong source ---------------------------------


def test_state_key_with_source_sessions_returns_notice(tmp_data_home):
    """state:/check:/mergeable: are records-only. Combined with a sessions
    source they can't match — return empty + notice."""
    store = Store()
    _seed_both(store)
    r = aggregator_query(
        dsl="source:sessions state:open", _store=store
    )
    assert r["ok"] is True
    assert r["total"] == 0
    assert "notice" in r
    assert (
        "state" in r["notice"].lower()
        or "records" in r["notice"].lower()
        or "github" in r["notice"].lower()
    )


def test_check_key_with_source_sessions_returns_notice(tmp_data_home):
    store = Store()
    _seed_both(store)
    r = aggregator_query(dsl="source:sessions check:pass", _store=store)
    assert r["ok"] is True
    assert r["total"] == 0
    assert "notice" in r


def test_mergeable_key_with_source_sessions_returns_notice(tmp_data_home):
    store = Store()
    _seed_both(store)
    r = aggregator_query(
        dsl="source:sessions mergeable:conflict", _store=store
    )
    assert r["ok"] is True
    assert r["total"] == 0
    assert "notice" in r


# --- session-only key with NO source hint routes to sessions -------------


def test_session_key_without_source_routes_to_sessions(tmp_data_home):
    store = Store()
    _seed_both(store)
    r = aggregator_query(dsl="session:sess-july", _store=store)
    assert r["mode"] == "sessions"
    ids = {rec["stable_id"] for rec in r["records"]}
    assert "sess-july" in ids


# --- records-only key with NO source hint routes to records --------------


def test_state_key_without_source_routes_to_records(tmp_data_home):
    """A records-only extra key with no source hint targets the records
    table (pre-v2 behaviour parity)."""
    store = Store()
    _seed_both(store)
    r = aggregator_query(dsl="state:open", _store=store)
    assert r["mode"] == "records"


# --- cross-source date-only union ----------------------------------------


def test_cross_source_date_query_unions_records_and_sessions(tmp_data_home):
    """Date-only query with no source hint hits BOTH tables.

    The user asked "what happened in July?" — the aggregator surfaces both
    PRs updated then and sessions active then. Rows from June and May are
    excluded from both sides. Result rides on a ``mode: "union"`` shape so
    the caller can distinguish record-shaped from session-shaped items via
    each item's ``source`` field.
    """
    store = Store()
    _seed_both(store)
    r = aggregator_query(
        dsl="from:2026-07-01 to:2026-07-31", _store=store
    )
    assert r["ok"] is True
    assert r["mode"] == "union"
    ids = {rec["stable_id"] for rec in r["records"]}
    # July record + July session; May session + June record excluded.
    assert "github:acme/api:1" in ids
    assert "sess-july" in ids
    assert "github:acme/api:2" not in ids
    assert "sess-may" not in ids
    # Total should reflect the merged count.
    assert r["total"] == 2


def test_union_pagination_walks_all_items_without_loss(tmp_data_home):
    """Round-1 HIGH-1: UNION deep-page reachability, no FTS.

    50 records + 50 sessions on distinct dates. Walk every page with
    page_size=10 and expect exactly 100 unique ids across all pages.
    Baseline reachability regardless of the fetch strategy.
    """
    store = Store()
    store.migrate()
    # 50 records at descending dates in Jul 2026.
    records = []
    for i in range(50):
        day = 1 + (i % 30)  # 1..30
        records.append(
            _rec(
                f"github:acme/api:r{i:03d}",
                subject=f"rec {i}",
                body="body",
                created=datetime(2026, 7, day, 8, 0, tzinfo=UTC),
                updated=datetime(2026, 7, day, 8, 0, tzinfo=UTC),
            )
        )
    store.upsert(records)
    # 50 sessions active in Aug 2026 (mixed with records after sort).
    sessions = []
    for i in range(50):
        day = 1 + (i % 30)
        ts = datetime(2026, 8, day, 12, 0, tzinfo=UTC)
        sessions.append(
            _sess(
                f"sess-u{i:03d}",
                first_ts=ts,
                last_ts=ts,
            )
        )
    store.upsert_entities(sessions)

    seen: set[str] = set()
    page_token: str | None = None
    pages = 0
    while True:
        pages += 1
        assert pages < 50, "runaway pagination"
        r = aggregator_query(
            dsl="",
            fields="summary",
            page_size=10,
            page_token=page_token,
            _store=store,
        )
        assert r["ok"] is True
        assert r["mode"] == "union"
        for rec in r["records"]:
            assert rec["stable_id"] not in seen, (
                f"duplicate across pages: {rec['stable_id']}"
            )
            seen.add(rec["stable_id"])
        page_token = r.get("next_page_token")
        if page_token is None:
            break
    # All 100 items must be reachable via pagination.
    assert len(seen) == 100, (
        f"expected 100 unique ids across all pages, got {len(seen)}"
    )
    assert r["total"] == 100


def test_union_pagination_records_fts_reaches_deep_matches(tmp_data_home):
    """Round-1 HIGH-1: UNION with FTS text under-returns records-side matches.

    Root cause: ``store.query`` applies ``LIMIT OFFSET`` in SQL BEFORE the
    Python-side FTS-id filter (records path). When 50 records exist but only
    the OLDEST 10 match the FTS text, over-fetching ``offset+page_size+1``
    from the top of the ORDER BY updated_at DESC gives you the newest N
    rows — none of which match the FTS term — so the FTS filter drops all
    of them. The 10 matching (older) rows are unreachable via union.

    Fix (option a in the round-1 note): fetch ALL matches per side (limit=
    None), merge, then paginate over the merged list.
    """
    store = Store()
    store.migrate()
    # 40 newer records with body "boring" (won't match FTS "needle").
    boring = [
        _rec(
            f"github:acme/api:b{i:03d}",
            subject=f"boring {i}",
            body="boring body",
            created=datetime(2026, 8, 1 + (i % 28), 8, 0, tzinfo=UTC),
            updated=datetime(2026, 8, 1 + (i % 28), 8, 0, tzinfo=UTC),
        )
        for i in range(40)
    ]
    # 10 older records with body "needle" (will match FTS).
    needles = [
        _rec(
            f"github:acme/api:n{i:03d}",
            subject=f"needle {i}",
            body="needle content here",
            created=datetime(2026, 7, 1 + i, 8, 0, tzinfo=UTC),
            updated=datetime(2026, 7, 1 + i, 8, 0, tzinfo=UTC),
        )
        for i in range(10)
    ]
    store.upsert([*boring, *needles])
    # Also a couple of sessions (unrelated to the needle text).
    store.upsert_entities(
        [
            _sess("sess-x1", first_ts=datetime(2026, 8, 15, tzinfo=UTC),
                  last_ts=datetime(2026, 8, 15, tzinfo=UTC)),
        ]
    )

    r = aggregator_query(
        dsl="needle", fields="summary", page_size=20, _store=store
    )
    assert r["ok"] is True
    assert r["mode"] == "union"
    needle_ids = {rec["stable_id"] for rec in r["records"]
                  if rec["stable_id"].startswith("github:acme/api:n")}
    assert len(needle_ids) == 10, (
        f"expected all 10 needle records reachable in a single page, "
        f"got {len(needle_ids)}; ids={sorted(needle_ids)}"
    )


# --- v3: chat-export origins route to the sessions path -------------------


def test_source_chatgpt_routes_to_sessions_mode(tmp_data_home):
    """``source:chatgpt`` is session-shaped — must route mode='sessions'
    and return only chatgpt-origin rows, labelled with their origin."""
    store = Store()
    store.migrate()
    store.upsert_entities(
        [
            _sess("cc-sess"),
            _sess("chatgpt:conv-1", origin="chatgpt"),
            _obs("cg-o1", "chatgpt:conv-1", "exported chat body"),
        ]
    )
    r = aggregator_query(dsl="source:chatgpt", _store=store)
    assert r["ok"] is True
    assert r["mode"] == "sessions"
    ids = {rec["stable_id"] for rec in r["records"]}
    assert ids == {"chatgpt:conv-1"}
    assert all(rec["source"] == "chatgpt" for rec in r["records"])


def test_source_claude_web_routes_to_sessions_mode(tmp_data_home):
    store = Store()
    store.migrate()
    store.upsert_entities(
        [
            _sess("cc-sess"),
            _sess("claude-web:conv-1", origin="claude-web"),
        ]
    )
    r = aggregator_query(dsl="source:claude-web", _store=store)
    assert r["ok"] is True
    assert r["mode"] == "sessions"
    ids = {rec["stable_id"] for rec in r["records"]}
    assert ids == {"claude-web:conv-1"}


# --- Chunk 4: research is records-shaped ----------------------------------


def test_source_research_routes_to_records_mode(tmp_data_home):
    """``source:research`` targets the records table ONLY. Before the
    routing set learned about research it fell through to union mode,
    where the sessions side (no origin filter for unknown sources)
    returned EVERY session row — a leak, not a filter."""
    store = Store()
    store.migrate()
    store.upsert(
        [
            Record(
                stable_id="research:abc123",
                source="research",
                subject="Export formats survey",
                body="report body",
                tags=["research"],
                created_at=datetime(2026, 7, 30, tzinfo=UTC),
                updated_at=datetime(2026, 7, 30, tzinfo=UTC),
            )
        ]
    )
    store.upsert_entities([_sess("cc-sess"), _obs("o1", "cc-sess", "chat")])
    r = aggregator_query(dsl="source:research", _store=store)
    assert r["ok"] is True
    assert r["mode"] == "records"
    ids = {rec["stable_id"] for rec in r["records"]}
    assert ids == {"research:abc123"}
    assert "cc-sess" not in ids


def test_source_research_with_session_key_returns_notice(tmp_data_home):
    """Session-only keys on the records-shaped research source mismatch."""
    store = Store()
    store.migrate()
    r = aggregator_query(dsl="source:research session:x", _store=store)
    assert r["ok"] is True
    assert r["total"] == 0
    assert "notice" in r


# --- Chunk 7: sota-watch is records-shaped --------------------------------


def test_source_sota_watch_routes_to_records_mode(tmp_data_home):
    """``source:sota-watch`` targets the records table. If it fell through
    to union mode the sessions side would leak every session row."""
    store = Store()
    store.migrate()
    store.upsert(
        [
            Record(
                stable_id="sota-watch:2026-07-31-tts",
                source="sota-watch",
                subject="TTS state of the art",
                body="proposal body",
                tags=["sota-watch"],
                created_at=datetime(2026, 7, 31, tzinfo=UTC),
                updated_at=datetime(2026, 7, 31, tzinfo=UTC),
            )
        ]
    )
    store.upsert_entities([_sess("cc-sess"), _obs("o1", "cc-sess", "chat")])
    r = aggregator_query(dsl="source:sota-watch", _store=store)
    assert r["ok"] is True
    assert r["mode"] == "records"
    ids = {rec["stable_id"] for rec in r["records"]}
    assert ids == {"sota-watch:2026-07-31-tts"}
    assert "cc-sess" not in ids


def test_source_sota_watch_with_session_key_returns_notice(tmp_data_home):
    """Session-only keys on the records-shaped sota-watch source mismatch."""
    store = Store()
    store.migrate()
    r = aggregator_query(dsl="source:sota-watch session:x", _store=store)
    assert r["ok"] is True
    assert r["total"] == 0
    assert "notice" in r


# --- Task 8: ticktick is records-shaped -----------------------------------


def test_source_ticktick_routes_to_records_mode(tmp_data_home):
    """``source:ticktick`` targets the records table. Unlisted, it falls
    through to union mode, whose sessions side has no origin filter for an
    unknown source and would hand back every session row — the source becomes
    ingestible and simultaneously unqueryable."""
    store = Store()
    store.migrate()
    store.upsert(
        [
            Record(
                stable_id="ticktick:abc123",
                source="ticktick",
                subject="Buy milk",
                body="from the good shop",
                tags=["errand", "open"],
                created_at=datetime(2026, 8, 1, tzinfo=UTC),
                updated_at=datetime(2026, 8, 1, tzinfo=UTC),
            )
        ]
    )
    store.upsert_entities([_sess("cc-sess"), _obs("o1", "cc-sess", "chat")])
    r = aggregator_query(dsl="source:ticktick", _store=store)
    assert r["ok"] is True
    assert r["mode"] == "records"
    ids = {rec["stable_id"] for rec in r["records"]}
    assert ids == {"ticktick:abc123"}
    assert "cc-sess" not in ids


def test_cross_source_no_filters_unions(tmp_data_home):
    """No filters at all also unions — "show me everything" surface."""
    store = Store()
    _seed_both(store)
    r = aggregator_query(dsl="", _store=store)
    assert r["ok"] is True
    assert r["mode"] == "union"
    ids = {rec["stable_id"] for rec in r["records"]}
    # All four items present.
    assert ids == {
        "github:acme/api:1",
        "github:acme/api:2",
        "sess-july",
        "sess-may",
    }
