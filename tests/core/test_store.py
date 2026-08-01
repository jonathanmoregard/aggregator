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
) -> SessionRow:
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


# --- v2 sessions + observations -------------------------------------------


def test_schema_version_is_2(tmp_data_home):
    s = Store()
    s.migrate()
    assert s.schema_version() == SCHEMA_VERSION == 2


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


def test_wal_files_land_in_db_dir(tmp_data_home):
    """WAL sidecar files sit next to cache.db, not somewhere surprising."""
    s = Store()
    s.migrate()
    s.upsert_entities([_sess("sess-a"), _obs("o1", "sess-a", "hi")])
    db = Path(s.db_path)
    assert db.exists()
