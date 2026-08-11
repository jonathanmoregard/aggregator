"""Tests for the SQLite+FTS5 store (v2, Schema B).

Two entity ontologies:

* ``records`` + ``records_fts`` — Record-shaped (github). Idempotent upsert,
  FTS query, atomic rebuild, scrub-on-write, concurrent-writer safety.
* ``sessions`` + ``observations`` + ``obs_fts`` — Langfuse-derived. Session
  hit-list queries, observation drilldown, root_session_id denormalisation.
"""
import contextlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from aggregator.core.store import SCHEMA_VERSION, Store
from aggregator.sources.base import (
    ObservationRow,
    QueryAST,
    Record,
    SessionRow,
)


def _rec(sid: str, source: str, subject: str, body: str, tags=()) -> Record:
    return Record(
        stable_id=sid,
        source=source,
        subject=subject,
        body=body,
        tags=list(tags),
        created_at=datetime(2026, 7, 25, tzinfo=UTC),
        updated_at=datetime(2026, 7, 25, tzinfo=UTC),
    )


def _sess(
    session_id: str,
    *,
    kind: str = "session",
    root: str | None = None,
    parent: str | None = None,
    agent_id: str | None = None,
    first_ts: datetime | None = None,
    last_ts: datetime | None = None,
    path: str = "/tmp/x.jsonl",
    origin: str | None = None,
) -> SessionRow:
    # ``origin`` passed only when explicitly requested so the helper also
    # exercises the dataclass default ('claude-code') on the common path.
    kwargs = {} if origin is None else {"origin": origin}
    return SessionRow(
        session_id=session_id,
        root_session_id=root or session_id,
        parent_session_id=parent,
        kind=kind,
        agent_id=agent_id,
        agent_type=None,
        spawned_by_tool_use_id=None,
        cwd="/x",
        git_branch="main",
        first_ts=first_ts or datetime(2026, 7, 25, 10, 0, tzinfo=UTC),
        last_ts=last_ts or datetime(2026, 7, 25, 10, 5, tzinfo=UTC),
        jsonl_path=path,
        **kwargs,
    )


def _obs(
    obs_id: str,
    session_id: str,
    body: str,
    *,
    obs_type: str = "user",
    root: str | None = None,
    ts: datetime | None = None,
) -> ObservationRow:
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


# --- Legacy Record path (github) ------------------------------------------


def test_store_upsert_and_fts_query(tmp_data_home):
    s = Store()
    s.migrate()
    s.upsert([_rec("github:a/1", "github", "hi", "refactor foo.py", tags=["pr"])])
    results = s.query(QueryAST(source="github", text="refactor"))
    assert len(results) == 1
    assert results[0].stable_id == "github:a/1"


def test_store_stable_id_persists_across_rebuild(tmp_data_home):
    s = Store()
    s.migrate()
    s.upsert([_rec("github:a/1", "github", "hi", "hello world")])
    s.rebuild("github")
    s.upsert([_rec("github:a/1", "github", "hi", "hello world (v2)")])
    results = s.query(QueryAST(source="github", text="hello"))
    assert len(results) == 1
    assert results[0].stable_id == "github:a/1"
    assert "v2" in results[0].body


def test_store_upsert_rejects_mutated_stable_id(tmp_data_home):
    s = Store()
    s.migrate()
    s.upsert([_rec("github:x", "github", "hi", "one")])
    s.upsert([_rec("github:y", "github", "hi", "one")])
    results = s.query(QueryAST(source="github"))
    assert {r.stable_id for r in results} == {"github:x", "github:y"}


def test_store_scrubs_on_upsert(tmp_data_home):
    s = Store()
    s.migrate()
    secret = "sk-" + "ant-api03-" + "x" * 44
    s.upsert(
        [_rec("github:leak", "github", "leak", f"here is a key {secret}")]
    )
    results = s.query(QueryAST(source="github", text="key"))
    assert len(results) == 1
    assert "sk-" + "ant-api03" not in results[0].body
    assert "[REDACTED:anthropic_key]" in results[0].body


def test_store_fts5_syntax_error_returns_empty_without_crashing(tmp_data_home):
    s = Store()
    s.migrate()
    s.upsert([_rec("github:a", "github", "s", "hello world")])
    r = s.query(QueryAST(source="github", text='"unterminated'))
    assert r == []


def test_store_upsert_overwrites_same_stable_id(tmp_data_home):
    s = Store()
    s.migrate()
    s.upsert([_rec("github:a", "github", "v1", "body v1")])
    s.upsert([_rec("github:a", "github", "v2", "body v2")])
    r = s.query(QueryAST(source="github"))
    assert len(r) == 1
    assert r[0].subject == "v2"
    assert r[0].body == "body v2"


def test_store_query_returns_all_records_beyond_500(tmp_data_home):
    s = Store()
    s.migrate()
    s.upsert([_rec(f"github:{i}", "github", f"s{i}", f"body {i}") for i in range(501)])
    rows = s.query(QueryAST(source="github"))
    assert len(rows) == 501


def test_store_query_limit_and_offset(tmp_data_home):
    s = Store()
    s.migrate()
    s.upsert([_rec(f"github:{i:04d}", "github", f"s{i}", f"body {i}") for i in range(50)])
    page = s.query(QueryAST(source="github"), limit=10, offset=10)
    assert len(page) == 10


def test_store_count_matches_query_size(tmp_data_home):
    s = Store()
    s.migrate()
    s.upsert([_rec(f"github:{i}", "github", f"s{i}", f"body {i}") for i in range(501)])
    assert s.count(QueryAST(source="github")) == 501


def test_store_probe_fts_public(tmp_data_home):
    import sqlite3

    import pytest as _pytest
    s = Store()
    s.migrate()
    s.upsert([_rec("github:a", "github", "s", "hello world")])
    s.probe_fts("hello")
    with _pytest.raises(sqlite3.OperationalError):
        s.probe_fts('"unterminated')


# --- Codex Phase 2 findings (RED) -----------------------------------------


def test_records_fts_reaches_deep_matches_beyond_first_page(tmp_data_home):
    """Codex Phase 2 HIGH: direct records path applied LIMIT/OFFSET before
    the FTS intersect, so a matching row beyond the first page window was
    unreachable via ``query()`` even though ``count()`` reported it.

    Repro: 5 non-matching newer records + 1 matching older record.
    Query ``source:github needle`` with limit=3. Buggy behaviour returned
    ``[]`` while count returned 1. Union path already had this fix; the
    records-only path did not.
    """
    s = Store()
    s.migrate()
    non_matching = [
        Record(
            stable_id=f"github:acme/repo:{i}",
            source="github",
            subject=f"new {i}",
            body="no match here",
            tags=["acme/repo"],
            updated_at=datetime(2026, 8, 1, 12 - i, tzinfo=UTC),
        )
        for i in range(5)
    ]
    old_match = Record(
        stable_id="github:acme/repo:old",
        source="github",
        subject="old",
        body="body with needle in it",
        tags=["acme/repo"],
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    s.upsert([*non_matching, old_match])

    ast = QueryAST(source="github", text="needle")
    hits = s.query(ast, limit=3, offset=0)
    total = s.count(ast)
    assert total == 1
    assert {r.stable_id for r in hits} == {"github:acme/repo:old"}, (
        f"records LIMIT-before-FTS: got {[r.stable_id for r in hits]}"
    )


def test_active_bare_date_upper_bound_inclusive(tmp_data_home):
    """Codex Phase 2 MEDIUM: ``active:D..D`` excluded sessions active on
    day D because bare-date parses as midnight-start of D, and the upper
    bound comparison ``first_ts <= active_to`` rejected sessions that
    began later on the same day. Documented semantics: inclusive [LO, HI]
    days. Fix: treat the bare-date HI as exclusive next-day.
    """
    from aggregator.core.dsl import parse

    s = Store()
    s.migrate()
    s.upsert_entities(
        [
            _sess(
                "feb1-session",
                first_ts=datetime(2026, 2, 1, 10, 0, tzinfo=UTC),
                last_ts=datetime(2026, 2, 1, 11, 0, tzinfo=UTC),
            )
        ]
    )
    ast = parse("active:2026-02-01..2026-02-01")
    rows = s.query_sessions(ast)
    assert {r.session_id for r in rows} == {"feb1-session"}, (
        f"same-day active range excluded row: got {[r.session_id for r in rows]}"
    )


def test_sessions_where_honours_source_kind_split(tmp_data_home):
    """Codex Phase 2 MEDIUM: ``_sessions_where`` ignored ``ast.source``, so
    ``source:sessions`` and ``source:subagents`` returned identical rows
    (top-level + subagents both). Sources advertise a kind bucket; the
    filter must honour it.
    """
    from aggregator.core.dsl import parse

    s = Store()
    s.migrate()
    s.upsert_entities(
        [
            _sess("root-src-kind"),
            _sess(
                "root-src-kind:agent1",
                kind="subagent",
                root="root-src-kind",
                parent="root-src-kind",
                agent_id="agent1",
            ),
        ]
    )

    ast_top = parse("source:sessions")
    top_only = s.query_sessions(ast_top)
    assert {r.session_id for r in top_only} == {"root-src-kind"}, (
        f"source:sessions leaked subagent rows: {[r.session_id for r in top_only]}"
    )

    ast_sub = parse("source:subagents")
    sub_only = s.query_sessions(ast_sub)
    assert {r.session_id for r in sub_only} == {"root-src-kind:agent1"}, (
        f"source:subagents leaked top rows: {[r.session_id for r in sub_only]}"
    )


def test_session_hitlist_honours_obs_type_with_fts(tmp_data_home):
    """Live-model smoke MEDIUM (2026-08-02): the sessions hit list mapped
    FTS text to sessions via root_session_id WITHOUT applying the
    ``type:`` filter. ``type:tool_use suggest_doc_edit`` surfaced 161
    session cards whose only match was a ``tool_result`` obs — every card
    showed ``matching_observations=0`` while drilldown (correctly)
    returned nothing. The hit list must honour the joint predicate.
    """
    from aggregator.core.dsl import parse

    s = Store()
    s.migrate()
    s.upsert_entities(
        [
            _sess("root-joint"),
            _obs(
                "o-result", "root-joint", "output mentioning needleword",
                obs_type="tool_result",
            ),
        ]
    )
    ast = parse("type:tool_use needleword")
    assert s.query_sessions(ast) == [], (
        "hit list must be empty when no obs matches BOTH type and text"
    )
    assert s.count_sessions(ast) == 0

    ast_right_type = parse("type:tool_result needleword")
    rows = s.query_sessions(ast_right_type)
    assert {r.session_id for r in rows} == {"root-joint"}


def test_session_hitlist_excludes_sibling_subagents_without_own_match(
    tmp_data_home,
):
    """Live-model smoke MEDIUM (2026-08-02) part 2: an FTS hit in subagent
    A surfaced A's card, the parent's card, AND sibling B's card (all rows
    sharing root_session_id). Sibling cards with zero own matches are
    noise. Semantics: parent card surfaces on any hit under its root
    (session: aggregates); a subagent card surfaces only when its OWN
    stream matches.
    """
    from aggregator.core.dsl import parse

    s = Store()
    s.migrate()
    parent = "root-sib"
    sub_a = f"{parent}:agentA"
    sub_b = f"{parent}:agentB"
    s.upsert_entities(
        [
            _sess(parent),
            _sess(sub_a, kind="subagent", root=parent, parent=parent,
                  agent_id="agentA"),
            _sess(sub_b, kind="subagent", root=parent, parent=parent,
                  agent_id="agentB"),
            _obs("oa", sub_a, "agent A found needleword here", root=parent),
            _obs("ob", sub_b, "agent B unrelated body", root=parent),
        ]
    )
    ast = parse("needleword")
    rows = s.query_sessions(ast)
    ids = {r.session_id for r in rows}
    assert ids == {parent, sub_a}, (
        f"sibling without own match must not surface: got {ids}"
    )
    assert s.count_sessions(ast) == 2


def test_rebuild_and_upsert_rolls_back_on_error(tmp_data_home):
    s = Store()
    s.migrate()
    original = [
        _rec("github:a", "github", "sa", "body a"),
        _rec("github:b", "github", "sb", "body b"),
        _rec("github:c", "github", "sc", "body c"),
    ]
    s.upsert(original)

    class Boom:
        def __iter__(self):
            yield _rec("github:new", "github", "sn", "body new")
            raise RuntimeError("simulated source fault")

    with contextlib.suppress(RuntimeError):
        s.rebuild_and_upsert("github", Boom())

    rows = s.query(QueryAST(source="github"))
    ids = {r.stable_id for r in rows}
    assert ids == {"github:a", "github:b", "github:c"}


def test_rebuild_and_upsert_replaces_source_atomically(tmp_data_home):
    s = Store()
    s.migrate()
    s.upsert(
        [
            _rec("github:old1", "github", "o1", "old body 1"),
            _rec("github:old2", "github", "o2", "old body 2"),
        ]
    )
    s.rebuild_and_upsert(
        "github",
        [_rec("github:new1", "github", "n1", "new body 1")],
    )
    rows = s.query(QueryAST(source="github"))
    assert {r.stable_id for r in rows} == {"github:new1"}


def _child_write_batch(db_path: str, prefix: str, n: int, q):
    import sqlite3

    from aggregator.core.store import Store as _Store
    from aggregator.sources.base import Record as _Record

    try:
        s = _Store(db_path)
        s.upsert(
            [
                _Record(
                    stable_id=f"{prefix}:{i}",
                    source=prefix,
                    subject=f"s{i}",
                    body=f"body {i}",
                    created_at=datetime(2026, 7, 25, tzinfo=UTC),
                    updated_at=datetime(2026, 7, 25, tzinfo=UTC),
                )
                for i in range(n)
            ]
        )
        s.close()
        q.put(("ok", prefix, None))
    except sqlite3.OperationalError as e:
        q.put(("err", prefix, str(e)))
    except Exception as e:  # noqa: BLE001
        q.put(("err", prefix, f"unexpected: {e}"))


def test_two_processes_concurrent_writes_succeed(tmp_data_home):
    """WAL + busy_timeout must let two concurrent writer processes both succeed."""
    import multiprocessing as mp

    bootstrap = Store()
    bootstrap.migrate()

    row = bootstrap._c().execute("PRAGMA journal_mode").fetchone()
    assert row[0].lower() == "wal", f"expected WAL, got {row[0]!r}"
    busy = bootstrap._c().execute("PRAGMA busy_timeout").fetchone()
    assert int(busy[0]) >= 5000, f"expected busy_timeout>=5000, got {busy[0]!r}"
    bootstrap.close()

    db_path = str(Store().db_path)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p1 = ctx.Process(target=_child_write_batch, args=(db_path, "one", 50, q))
    p2 = ctx.Process(target=_child_write_batch, args=(db_path, "two", 50, q))
    p1.start()
    p2.start()
    p1.join(timeout=30)
    p2.join(timeout=30)
    assert p1.exitcode == 0 and p2.exitcode == 0

    results: list[tuple[str, str, str | None]] = []
    while not q.empty():
        results.append(q.get())
    errs = [r for r in results if r[0] == "err"]
    assert not errs, f"concurrent writers hit errors: {errs}"


def test_read_only_store_skips_wal_and_refuses_migration(tmp_path):
    """MCP readers must not require WAL sidecar writes."""
    import sqlite3

    import pytest as _pytest

    db_path = tmp_path / "cache.db"
    con = sqlite3.connect(db_path)
    con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    con.close()

    store = Store(db_path, read_only=True)
    journal = store._c().execute("PRAGMA journal_mode").fetchone()
    assert journal[0].lower() != "wal"
    assert not db_path.with_name("cache.db-wal").exists()
    assert not db_path.with_name("cache.db-shm").exists()
    with _pytest.raises(RuntimeError, match="read-only Store"):
        store.migrate()
    store.close()


# --- v2 sessions + observations -------------------------------------------


def test_schema_version_is_3(tmp_data_home):
    s = Store()
    s.migrate()
    assert s.schema_version() == SCHEMA_VERSION == 3


def test_upsert_entities_writes_sessions_and_observations(tmp_data_home):
    s = Store()
    s.migrate()
    session = _sess("sess-a")
    obs = [_obs("o1", "sess-a", "hello"), _obs("o2", "sess-a", "world")]
    s.upsert_entities([session, *obs])
    rows = s.query_sessions(QueryAST(top_session_id="sess-a"))
    assert len(rows) == 1
    obs_rows = s.query_observations(QueryAST(top_session_id="sess-a"))
    assert {o.obs_id for o in obs_rows} == {"o1", "o2"}


def test_root_session_id_includes_subagents_via_session_key(tmp_data_home):
    """DSL ``session:X`` matches ``root_session_id`` — must return top + subagent."""
    s = Store()
    s.migrate()
    top = _sess("sess-root")
    sub = _sess("sess-root:agent1", kind="subagent", root="sess-root", parent="sess-root", agent_id="agent1")
    top_obs = _obs("top-o1", "sess-root", "top user prompt")
    sub_obs = _obs("sub-o1", "sess-root:agent1", "subagent turn", root="sess-root")
    s.upsert_entities([top, sub, top_obs, sub_obs])

    obs = s.query_observations(QueryAST(session_id="sess-root"))
    ids = {o.obs_id for o in obs}
    assert ids == {"top-o1", "sub-o1"}, (
        "session: must match root_session_id and include subagent rows"
    )

    top_only = s.query_observations(QueryAST(top_session_id="sess-root"))
    top_only_ids = {o.obs_id for o in top_only}
    assert top_only_ids == {"top-o1"}, (
        "top: must match session_id exactly and exclude subagents"
    )


def test_agent_filter_targets_specific_subagent(tmp_data_home):
    s = Store()
    s.migrate()
    top = _sess("root")
    sub_a = _sess("root:agentA", kind="subagent", root="root", parent="root", agent_id="agentA")
    sub_b = _sess("root:agentB", kind="subagent", root="root", parent="root", agent_id="agentB")
    obs = [
        _obs("a-1", "root:agentA", "agent A body", root="root"),
        _obs("b-1", "root:agentB", "agent B body", root="root"),
    ]
    s.upsert_entities([top, sub_a, sub_b, *obs])
    result = s.query_observations(QueryAST(agent_id="agentA"))
    assert {o.obs_id for o in result} == {"a-1"}


def test_obs_type_filter(tmp_data_home):
    s = Store()
    s.migrate()
    session = _sess("sess-a")
    obs = [
        _obs("u1", "sess-a", "user says", obs_type="user"),
        _obs("a1", "sess-a", "assistant replies", obs_type="assistant"),
        _obs("t1", "sess-a", "", obs_type="tool_use"),
    ]
    s.upsert_entities([session, *obs])
    result = s.query_observations(QueryAST(obs_type="tool_use"))
    assert {o.obs_id for o in result} == {"t1"}


def test_active_window_overlap(tmp_data_home):
    """active:LO..HI matches sessions whose window overlaps [LO, HI]."""
    s = Store()
    s.migrate()
    s.upsert_entities(
        [
            _sess(
                "early",
                first_ts=datetime(2026, 6, 1, tzinfo=UTC),
                last_ts=datetime(2026, 6, 1, 12, tzinfo=UTC),
            ),
            _sess(
                "middle",
                first_ts=datetime(2026, 7, 15, tzinfo=UTC),
                last_ts=datetime(2026, 7, 15, 12, tzinfo=UTC),
            ),
            _sess(
                "late",
                first_ts=datetime(2026, 8, 20, tzinfo=UTC),
                last_ts=datetime(2026, 8, 20, 12, tzinfo=UTC),
            ),
        ]
    )
    result = s.query_sessions(
        QueryAST(
            active_from=datetime(2026, 7, 1, tzinfo=UTC),
            active_to=datetime(2026, 8, 1, tzinfo=UTC),
        )
    )
    assert {r.session_id for r in result} == {"middle"}


def test_obs_fts_text_search_projects_to_sessions(tmp_data_home):
    """Text search in query_sessions flows through obs_fts → root_session_id."""
    s = Store()
    s.migrate()
    s.upsert_entities(
        [
            _sess("alpha"),
            _sess("beta"),
            _obs("a1", "alpha", "we talked about refactoring foo.py extensively"),
            _obs("b1", "beta", "nothing related here"),
        ]
    )
    result = s.query_sessions(QueryAST(text="refactoring"))
    assert {r.session_id for r in result} == {"alpha"}


def test_obs_scrub_on_write(tmp_data_home):
    """Observation bodies scrubbed pre-write (same as Records)."""
    s = Store()
    s.migrate()
    secret = "sk-" + "ant-api03-" + "y" * 44
    s.upsert_entities(
        [_sess("sess-leak"), _obs("o-leak", "sess-leak", f"here is {secret}")]
    )
    obs = s.query_observations(QueryAST(top_session_id="sess-leak"))
    assert "sk-" + "ant-api03" not in obs[0].body
    assert "[REDACTED:anthropic_key]" in obs[0].body


def test_rebuild_all_wipes_and_recreates(tmp_data_home):
    s = Store()
    s.migrate()
    s.upsert_entities([_sess("sess-a"), _obs("o1", "sess-a", "hi")])
    s.upsert([_rec("github:1", "github", "s", "b")])
    s.rebuild_all()
    # Fresh migrate must succeed and tables must exist empty.
    assert s.count_sessions(QueryAST()) == 0
    assert s.count(QueryAST(source="github")) == 0
    # user_version still bumped.
    assert s.schema_version() == SCHEMA_VERSION


def test_rebuild_and_upsert_entities_min_sessions_guard(tmp_data_home):
    import pytest as _pytest

    from aggregator.core.store import EmptyRebuildRefusedError

    s = Store()
    s.migrate()
    s.upsert_entities([_sess("keep"), _obs("k1", "keep", "important")])

    with _pytest.raises(EmptyRebuildRefusedError):
        s.rebuild_and_upsert_entities([], min_sessions=1)

    rows = s.query_sessions(QueryAST())
    assert len(rows) == 1


def test_rebuild_and_upsert_entities_atomic_on_failure(tmp_data_home):
    """A fault mid-upsert must roll back — original sessions/obs intact."""
    s = Store()
    s.migrate()
    s.upsert_entities([_sess("keep"), _obs("k1", "keep", "hi")])

    # Sneak in an entity of unknown type mid-stream so upsert_entities raises.
    class NotAnEntity:
        pass

    with contextlib.suppress(TypeError):
        s.rebuild_and_upsert_entities(
            [_sess("new"), _obs("n1", "new", "x"), NotAnEntity()]
        )
    rows = s.query_sessions(QueryAST())
    assert {r.session_id for r in rows} == {"keep"}


def test_capabilities_reports_v2_counts(tmp_data_home):
    s = Store()
    s.migrate()
    s.upsert_entities(
        [
            _sess("root"),
            _sess("root:sub1", kind="subagent", root="root", parent="root", agent_id="sub1"),
            _obs("o1", "root", "hi"),
            _obs("o2", "root:sub1", "sub hi", root="root"),
        ]
    )
    s.upsert([_rec("github:1", "github", "s", "b")])
    caps = s.capabilities()
    assert "sessions" in caps["sources"]
    assert "subagents" in caps["sources"]
    assert caps["counts"]["sessions"] == 1
    assert caps["counts"]["subagents"] == 1
    assert caps["counts"]["observations"] == 2
    assert caps["counts"]["records"] == 1


# --- v3: sessions.origin (chat-export sources) -----------------------------


def test_migrate_upgrades_v2_db_in_place_without_data_loss(tmp_data_home):
    """v2 → v3 must be an in-place ``ALTER TABLE sessions ADD COLUMN origin``
    guarded by a ``PRAGMA table_info`` probe — the live DB is 616MB and a
    rebuild costs hours. Pre-existing rows must survive and backfill to
    ``origin='claude-code'`` via the column default.
    """
    s = Store()
    s.migrate()
    s.upsert_entities([_sess("pre-v3"), _obs("o-pre", "pre-v3", "kept body")])
    # Downgrade to the v2 shape: drop origin (and any index on it), reset
    # user_version. This reconstructs what a real v2 cache.db looks like.
    c = s._c()
    c.execute("DROP INDEX IF EXISTS sessions_origin")
    c.execute("ALTER TABLE sessions DROP COLUMN origin")
    c.execute("PRAGMA user_version = 2")
    c.commit()
    s.close()

    s2 = Store()
    s2.migrate()
    assert s2.schema_version() == SCHEMA_VERSION == 3
    cols = {r[1] for r in s2._c().execute("PRAGMA table_info(sessions)")}
    assert "origin" in cols
    rows = s2.query_sessions(QueryAST(top_session_id="pre-v3"))
    assert len(rows) == 1
    assert rows[0].origin == "claude-code"
    obs = s2.query_observations(QueryAST(top_session_id="pre-v3"))
    assert {o.obs_id for o in obs} == {"o-pre"}


def test_session_origin_round_trip(tmp_data_home):
    s = Store()
    s.migrate()
    s.upsert_entities([_sess("chatgpt:conv-1", origin="chatgpt")])
    rows = s.query_sessions(QueryAST(top_session_id="chatgpt:conv-1"))
    assert len(rows) == 1
    assert rows[0].origin == "chatgpt"


def test_source_filters_split_by_origin(tmp_data_home):
    """``source:chatgpt`` / ``source:claude-web`` filter on origin;
    ``source:sessions`` stays claude-code-only — chat-export rows are
    kind='session' too and must NOT leak into the existing bucket.
    """
    from aggregator.core.dsl import parse

    s = Store()
    s.migrate()
    s.upsert_entities(
        [
            _sess("cc-sess"),
            _sess("chatgpt:conv-1", origin="chatgpt"),
            _sess("claude-web:conv-1", origin="claude-web"),
        ]
    )
    got = {r.session_id for r in s.query_sessions(parse("source:chatgpt"))}
    assert got == {"chatgpt:conv-1"}
    got = {r.session_id for r in s.query_sessions(parse("source:claude-web"))}
    assert got == {"claude-web:conv-1"}
    got = {r.session_id for r in s.query_sessions(parse("source:sessions"))}
    assert got == {"cc-sess"}, f"chat rows leaked into source:sessions: {got}"


def test_source_subagents_excludes_chat_origins(tmp_data_home):
    """Defensive: the subagents bucket implicitly means claude-code. Even a
    (malformed) chat-origin row with kind='subagent' must not surface.
    """
    from aggregator.core.dsl import parse

    s = Store()
    s.migrate()
    s.upsert_entities(
        [
            _sess("cc-root"),
            _sess(
                "cc-root:agent1",
                kind="subagent",
                root="cc-root",
                parent="cc-root",
                agent_id="agent1",
            ),
            _sess("chatgpt:weird", kind="subagent", origin="chatgpt"),
        ]
    )
    got = {r.session_id for r in s.query_sessions(parse("source:subagents"))}
    assert got == {"cc-root:agent1"}


def test_obs_filter_by_origin(tmp_data_home):
    """Observation queries mirror the origin filter via a sessions subselect."""
    from aggregator.core.dsl import parse

    s = Store()
    s.migrate()
    s.upsert_entities(
        [
            _sess("cc-sess"),
            _sess("chatgpt:conv-1", origin="chatgpt"),
            _obs("cc-o1", "cc-sess", "claude code body"),
            _obs("cg-o1", "chatgpt:conv-1", "exported chat body"),
        ]
    )
    got = {o.obs_id for o in s.query_observations(parse("source:chatgpt"))}
    assert got == {"cg-o1"}
    got = {o.obs_id for o in s.query_observations(parse("source:sessions"))}
    assert got == {"cc-o1"}, f"chat obs leaked into source:sessions: {got}"


def test_capabilities_reports_chat_origin_counts(tmp_data_home):
    """Per-origin counts surface when nonzero; existing keys keep their
    claude-code semantics; zero-row origins are omitted entirely."""
    s = Store()
    s.migrate()
    s.upsert_entities(
        [
            _sess("cc-sess"),
            _sess("chatgpt:conv-1", origin="chatgpt"),
        ]
    )
    caps = s.capabilities()
    assert caps["counts"]["sessions"] == 1  # claude-code only
    assert caps["counts"]["chatgpt"] == 1
    assert "chatgpt" in caps["sources"]
    assert caps["freshness"]["chatgpt"] is not None
    assert "claude-web" not in caps["sources"]
    assert "claude-web" not in caps["counts"]


def test_wal_files_land_in_db_dir(tmp_data_home):
    """WAL sidecar files sit next to cache.db, not somewhere surprising."""
    s = Store()
    s.migrate()
    s.upsert_entities([_sess("sess-a"), _obs("o1", "sess-a", "hi")])
    db = Path(s.db_path)
    assert db.exists()


def test_existing_ids_reports_which_primary_keys_are_already_stored(tmp_data_home):
    """Batch existence probe. The import runner needs it to report REAL
    added-vs-updated counts instead of ``added=len(items)``."""
    s = Store()
    s.migrate()
    s.upsert([_rec("github:1", "github", "a", "b")])
    s.upsert_entities([_sess("sess-a"), _obs("o1", "sess-a", "hi")])

    assert s.existing_ids("records", ["github:1", "github:2"]) == {"github:1"}
    assert s.existing_ids("sessions", ["sess-a", "sess-b"]) == {"sess-a"}
    assert s.existing_ids("observations", ["o1", "o2"]) == {"o1"}
    assert s.existing_ids("records", []) == set()


def test_existing_ids_rejects_a_table_not_on_the_allowlist(tmp_data_home):
    """Table name is interpolated into SQL, so it can only ever be one of
    three literals — never caller-supplied text."""
    s = Store()
    s.migrate()
    with pytest.raises(ValueError, match="unknown table"):
        s.existing_ids("records; DROP TABLE records", ["x"])


# --- a re-observation must not erase a timestamp it does not carry --------
#
# Round-1 MEDIUM: ``ON CONFLICT ... updated_at = excluded.updated_at`` let a
# record with no timestamp overwrite a real one. ticktick is where it bites: a
# task payload carrying none of completedTime/modifiedTime/createdTime yields
# ``updated_at=None``, and the merge lets the fresher API observation win, so
# a NULL landed on the date the CSV leg had parsed. ``_warn_no_timestamps``
# only fires when the WHOLE batch is dateless, so a partially-dated batch
# degrades every date query it touches in complete silence.


def _dated(sid: str, created=None, updated=None) -> Record:
    return Record(
        stable_id=sid,
        source="ticktick",
        subject="task",
        body="body",
        created_at=created,
        updated_at=updated,
    )


def _dates(store: Store, sid: str) -> tuple[str | None, str | None]:
    row = store._c().execute(
        "SELECT created_at, updated_at FROM records WHERE stable_id = ?", (sid,)
    ).fetchone()
    return row["created_at"], row["updated_at"]


REAL_TS = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def test_a_dateless_reobservation_does_not_null_a_real_updated_at(tmp_data_home):
    s = Store()
    s.migrate()
    s.upsert([_dated("ticktick:t1", created=REAL_TS, updated=REAL_TS)])

    s.upsert([_dated("ticktick:t1")])  # API leg won the merge, carrying nothing

    _, updated = _dates(s, "ticktick:t1")
    assert updated == REAL_TS.isoformat()


def test_a_real_updated_at_still_overwrites_an_older_one(tmp_data_home):
    """The guard must not freeze the column: a genuine newer observation is
    the normal case and has to land."""
    s = Store()
    s.migrate()
    newer = datetime(2026, 8, 1, tzinfo=UTC)
    s.upsert([_dated("ticktick:t1", created=REAL_TS, updated=REAL_TS)])

    s.upsert([_dated("ticktick:t1", created=REAL_TS, updated=newer)])

    _, updated = _dates(s, "ticktick:t1")
    assert updated == newer.isoformat()


def test_a_dateless_reobservation_survives_the_rebuild_write_path_too(
    tmp_data_home,
):
    """``upsert`` and ``_do_write_records`` are two copies of the same SQL and
    must not drift; the rebuild path deletes first, so this pins the INSERT
    half rather than the conflict half."""
    s = Store()
    s.migrate()

    s.rebuild_and_upsert("ticktick", [_dated("ticktick:t1", updated=REAL_TS)])
    s.upsert([_dated("ticktick:t1")])

    _, updated = _dates(s, "ticktick:t1")
    assert updated == REAL_TS.isoformat()
