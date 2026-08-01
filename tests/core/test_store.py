"""Tests for the SQLite+FTS5 store (M2).

The store owns four contracts:

* Upsert-idempotent writes keyed by ``stable_id`` (spec §Storage).
* FTS5 querying scoped by AST filters (source, tags, date range, text).
* Rebuild = ``drop_source(name)`` + re-ingest yields the SAME stable IDs
  (stable-ID discipline, spec constraint 5).
* Every write goes through the scrubber (defense in depth, spec constraint 3).
"""
import contextlib
from datetime import UTC, datetime

from aggregator.core.store import Store
from aggregator.sources.base import QueryAST, Record


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


def test_store_upsert_and_fts_query(tmp_data_home):
    s = Store()
    s.migrate()
    s.upsert(
        [_rec("sessions:a1", "sessions", "hi", "refactor foo.py", tags=["proj-alpha"])]
    )
    results = s.query(QueryAST(source="sessions", text="refactor"))
    assert len(results) == 1
    assert results[0].stable_id == "sessions:a1"


def test_store_stable_id_persists_across_rebuild(tmp_data_home):
    s = Store()
    s.migrate()
    s.upsert([_rec("sessions:a1", "sessions", "hi", "hello world")])
    s.rebuild("sessions")  # drop rows for the source; source re-ingests same IDs
    s.upsert([_rec("sessions:a1", "sessions", "hi", "hello world (v2)")])
    results = s.query(QueryAST(source="sessions", text="hello"))
    assert len(results) == 1
    assert results[0].stable_id == "sessions:a1"
    assert "v2" in results[0].body


def test_store_upsert_rejects_mutated_stable_id(tmp_data_home):
    """Different stable_id => distinct records, never silently merged.

    Two records with identical (subject, body, tags) but different stable_ids
    must remain two rows.
    """
    s = Store()
    s.migrate()
    s.upsert([_rec("sessions:x", "sessions", "hi", "one")])
    s.upsert([_rec("sessions:y", "sessions", "hi", "one")])
    results = s.query(QueryAST(source="sessions"))
    ids = {r.stable_id for r in results}
    assert ids == {"sessions:x", "sessions:y"}


def test_store_tag_filter(tmp_data_home):
    s = Store()
    s.migrate()
    s.upsert(
        [
            _rec("sessions:a", "sessions", "a", "aaa", tags=["proj-alpha"]),
            _rec("sessions:b", "sessions", "b", "bbb", tags=["proj-beta"]),
        ]
    )
    r = s.query(QueryAST(source="sessions", tags=["proj-alpha"]))
    assert len(r) == 1
    assert r[0].stable_id == "sessions:a"


def test_store_source_filter(tmp_data_home):
    s = Store()
    s.migrate()
    s.upsert(
        [
            _rec("sessions:a", "sessions", "a", "aaa"),
            _rec("github:owner/repo:1", "github", "pr", "bbb"),
        ]
    )
    assert {r.stable_id for r in s.query(QueryAST(source="sessions"))} == {"sessions:a"}
    assert {r.stable_id for r in s.query(QueryAST(source="github"))} == {
        "github:owner/repo:1"
    }


def test_store_scrubs_on_upsert(tmp_data_home):
    """Pre-store scrubbing — defense in depth (spec §Security constraint 3).

    The stored body must not contain the raw secret even if the caller passed
    one in. Constructed inline via split literals so this test file itself is
    not gitleaks-flaggable.
    """
    s = Store()
    s.migrate()
    secret = "sk-" + "ant-api03-" + "x" * 44  # constructed inline; NOT a real key
    s.upsert(
        [_rec("sessions:leak", "sessions", "leak", f"here is a key {secret}")]
    )
    results = s.query(QueryAST(source="sessions", text="key"))
    assert len(results) == 1
    assert "sk-" + "ant-api03" not in results[0].body
    assert "[REDACTED:anthropic_key]" in results[0].body


def test_store_capabilities(tmp_data_home):
    s = Store()
    s.migrate()
    s.upsert([_rec("sessions:a", "sessions", "s", "b", tags=["proj-alpha"])])
    caps = s.capabilities()
    assert "sessions" in caps["sources"]
    assert caps["cache_path"].endswith("cache.db")
    assert "sessions" in caps["freshness"]


def test_store_date_range_filter(tmp_data_home):
    s = Store()
    s.migrate()
    early = Record(
        stable_id="sessions:early",
        source="sessions",
        subject="early",
        body="early",
        created_at=datetime(2026, 6, 1, tzinfo=UTC),
        updated_at=datetime(2026, 6, 1, tzinfo=UTC),
    )
    late = Record(
        stable_id="sessions:late",
        source="sessions",
        subject="late",
        body="late",
        created_at=datetime(2026, 8, 1, tzinfo=UTC),
        updated_at=datetime(2026, 8, 1, tzinfo=UTC),
    )
    s.upsert([early, late])
    ast = QueryAST(
        source="sessions",
        from_date=datetime(2026, 7, 1, tzinfo=UTC),
        to_date=datetime(2026, 9, 1, tzinfo=UTC),
    )
    r = s.query(ast)
    assert {x.stable_id for x in r} == {"sessions:late"}


def test_store_fts5_syntax_error_returns_empty_without_crashing(tmp_data_home):
    """Malformed FTS query (unbalanced quote) must not raise (spec §DSL:
    return ``ok: false`` in surface layer; store returns [] and logs)."""
    s = Store()
    s.migrate()
    s.upsert([_rec("sessions:a", "sessions", "s", "hello world")])
    # `"unterminated` is malformed FTS5 syntax (unbalanced double quote).
    r = s.query(QueryAST(source="sessions", text='"unterminated'))
    assert r == []


def test_store_upsert_overwrites_same_stable_id(tmp_data_home):
    """Idempotent ingest: same stable_id re-upsert = update, not duplicate."""
    s = Store()
    s.migrate()
    s.upsert([_rec("sessions:a", "sessions", "v1", "body v1")])
    s.upsert([_rec("sessions:a", "sessions", "v2", "body v2")])
    r = s.query(QueryAST(source="sessions"))
    assert len(r) == 1
    assert r[0].subject == "v2"
    assert r[0].body == "body v2"


# --- HIGH-3: no silent LIMIT 500 truncation, plus pagination + count ------


def test_store_query_returns_all_records_beyond_500(tmp_data_home):
    """Pre-fix (advisor HIGH-3): Store.query hardcoded LIMIT 500, silently
    truncating without any signal to the caller. Post-fix: no default limit;
    the store returns every matching row unless the caller asks otherwise.
    """
    s = Store()
    s.migrate()
    s.upsert(
        [_rec(f"sessions:{i}", "sessions", f"s{i}", f"body {i}") for i in range(501)]
    )
    rows = s.query(QueryAST(source="sessions"))
    assert len(rows) == 501


def test_store_query_limit_and_offset(tmp_data_home):
    """Explicit limit/offset carve out a page. Enables MCP pagination on top
    without every caller paying the "load everything into memory" cost."""
    s = Store()
    s.migrate()
    s.upsert(
        [_rec(f"sessions:{i:04d}", "sessions", f"s{i}", f"body {i}") for i in range(50)]
    )
    page = s.query(QueryAST(source="sessions"), limit=10, offset=10)
    assert len(page) == 10


def test_store_count_matches_query_size(tmp_data_home):
    """Store.count(ast) is the new source-of-truth for MCP's ``total`` field."""
    s = Store()
    s.migrate()
    s.upsert(
        [_rec(f"sessions:{i}", "sessions", f"s{i}", f"body {i}") for i in range(501)]
    )
    assert s.count(QueryAST(source="sessions")) == 501


def test_store_probe_fts_public(tmp_data_home):
    """MEDIUM: MCP no longer reaches into store._c(); use Store.probe_fts."""
    import sqlite3

    s = Store()
    s.migrate()
    s.upsert([_rec("sessions:a", "sessions", "s", "hello world")])
    # Well-formed query: no exception.
    s.probe_fts("hello")
    # Malformed FTS5 syntax: raise OperationalError so callers can convert
    # to a structured error (same behaviour as the private probe used to have).
    import pytest as _pytest
    with _pytest.raises(sqlite3.OperationalError):
        s.probe_fts('"unterminated')


# --- MEDIUM (round-2): atomic rebuild_and_upsert --------------------------


def test_rebuild_and_upsert_rolls_back_on_error(tmp_data_home):
    """Round-2 MEDIUM: pre-fix, ``store.rebuild`` committed the DELETE
    immediately; if the caller's subsequent ``upsert`` faulted the store
    was left empty. Post-fix, ``rebuild_and_upsert`` runs the DELETE +
    upsert inside a single transaction so a failure mid-upsert rolls
    back to the pre-rebuild state.
    """
    s = Store()
    s.migrate()
    original = [
        _rec("sessions:a", "sessions", "sa", "body a"),
        _rec("sessions:b", "sessions", "sb", "body b"),
        _rec("sessions:c", "sessions", "sc", "body c"),
    ]
    s.upsert(original)

    class Boom:
        """Iterable that yields one good record then raises. Simulates a
        source that faults partway through iteration."""

        def __iter__(self):
            yield _rec("sessions:new", "sessions", "sn", "body new")
            raise RuntimeError("simulated source fault")

    with contextlib.suppress(RuntimeError):
        s.rebuild_and_upsert("sessions", Boom())  # expected to raise

    # Original 3 rows must still be present; the partial "sessions:new" must
    # NOT be committed.
    rows = s.query(QueryAST(source="sessions"))
    ids = {r.stable_id for r in rows}
    assert ids == {"sessions:a", "sessions:b", "sessions:c"}, (
        f"rebuild_and_upsert should have rolled back; got {ids}"
    )


def test_rebuild_and_upsert_replaces_source_atomically(tmp_data_home):
    """Happy path: successful call clears old rows and writes new ones,
    leaving other sources untouched."""
    s = Store()
    s.migrate()
    s.upsert(
        [
            _rec("sessions:old1", "sessions", "o1", "old body 1"),
            _rec("sessions:old2", "sessions", "o2", "old body 2"),
            _rec("github:owner/repo:1", "github", "pr", "unrelated"),
        ]
    )
    s.rebuild_and_upsert(
        "sessions",
        [
            _rec("sessions:new1", "sessions", "n1", "new body 1"),
        ],
    )
    session_rows = s.query(QueryAST(source="sessions"))
    assert {r.stable_id for r in session_rows} == {"sessions:new1"}
    # Other source untouched.
    gh_rows = s.query(QueryAST(source="github"))
    assert {r.stable_id for r in gh_rows} == {"github:owner/repo:1"}


# --- Codex Phase 2 MEDIUM #2: concurrent-writer safety --------------------


def _child_write_batch(db_path: str, prefix: str, n: int, q):
    """Multiprocessing child target — must be top-level (picklable).

    Opens its own Store against the same cache.db and writes ``n`` rows.
    Reports any ``sqlite3.OperationalError`` via the queue so the parent
    can assert lock-free behaviour.
    """
    import sqlite3

    from aggregator.core.store import Store as _Store
    from aggregator.sources.base import Record as _Record

    try:
        s = _Store(db_path)
        # Both processes hammer .upsert() concurrently. Pre-fix, the second
        # to acquire the write lock hits ``database is locked`` and raises.
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
    """Codex MEDIUM #2: two systemd user timers (sessions + github) fire on
    the same ``*:0/30`` schedule and both open a Store against the same
    ``cache.db``. Pre-fix (default rollback journal, no busy_timeout) the
    second writer races into ``database is locked`` and its transaction
    fails. Post-fix (WAL + busy_timeout=5000) both writers succeed.

    Uses multiprocessing to mirror the actual failure mode (separate
    processes, separate SQLite connections). Threads inside one process
    can't reproduce it — SQLite enforces its own single-thread rule.

    Sanity: also asserts the PRAGMAs are applied on each fresh connection
    so a future edit that removes them fails loudly instead of just being
    slow to reproduce the race.
    """
    import multiprocessing as mp

    # Bootstrap the schema first so both writers race on writes, not on
    # CREATE TABLE (which is a distinct — and much rarer — failure mode).
    bootstrap = Store()
    bootstrap.migrate()

    # PRAGMA sanity — future edits that remove WAL/busy_timeout trip here.
    row = bootstrap._c().execute("PRAGMA journal_mode").fetchone()
    assert row[0].lower() == "wal", f"expected WAL, got {row[0]!r}"
    busy = bootstrap._c().execute("PRAGMA busy_timeout").fetchone()
    assert int(busy[0]) >= 5000, f"expected busy_timeout>=5000, got {busy[0]!r}"
    bootstrap.close()

    db_path = str(Store().db_path)

    # ``spawn`` avoids fork-inherit warnings on Linux and matches how
    # systemd would launch each timer's Python process from scratch.
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p1 = ctx.Process(
        target=_child_write_batch, args=(db_path, "sessions", 50, q)
    )
    p2 = ctx.Process(target=_child_write_batch, args=(db_path, "github", 50, q))
    p1.start()
    p2.start()
    p1.join(timeout=30)
    p2.join(timeout=30)
    assert p1.exitcode == 0 and p2.exitcode == 0, (
        f"child processes exited abnormally: p1={p1.exitcode}, p2={p2.exitcode}"
    )

    results: list[tuple[str, str, str | None]] = []
    while not q.empty():
        results.append(q.get())
    assert len(results) == 2, f"expected 2 results, got {results}"
    errs = [r for r in results if r[0] == "err"]
    assert not errs, f"concurrent writers hit errors: {errs}"

    # Both sources should be fully persisted.
    verify = Store()
    verify.migrate()
    assert verify.count(QueryAST(source="sessions")) == 50
    assert verify.count(QueryAST(source="github")) == 50


def test_store_wraps_records_on_return(tmp_data_home):
    """Store's query returns Records; the wrapping into ``<ExternalContent>``
    happens at the surface layer (MCP/CLI) via ``wrap_records``. Confirm the
    Records are re-hydrated with intact fields ready for wrapping."""
    from aggregator.core.wrap import wrap_records

    s = Store()
    s.migrate()
    s.upsert(
        [_rec("sessions:w", "sessions", "subj", "some body text", tags=["p"])]
    )
    r = s.query(QueryAST(source="sessions"))
    wrapped = wrap_records(r)
    assert '<ExternalContent source="sessions:w">' in wrapped
    assert "some body text" in wrapped
