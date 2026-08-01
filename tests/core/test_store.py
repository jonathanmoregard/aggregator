"""Tests for the SQLite+FTS5 store (M2).

The store owns four contracts:

* Upsert-idempotent writes keyed by ``stable_id`` (spec §Storage).
* FTS5 querying scoped by AST filters (source, tags, date range, text).
* Rebuild = ``drop_source(name)`` + re-ingest yields the SAME stable IDs
  (stable-ID discipline, spec constraint 5).
* Every write goes through the scrubber (defense in depth, spec constraint 3).
"""
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
