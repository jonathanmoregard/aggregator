"""Tests for aggregator_query MCP tool (M3).

Covers:
* Happy path: DSL parses, store returns records, output wrapped in
  ``<ExternalContent>`` delimiters (spec §Security).
* Field selection: default ``summary`` omits body + emits ``notice``; ``full``
  includes body.
* DSL parse error → structured ``{ok: false, reason, remediation}`` (never a
  stack trace).
* FTS5 syntax error → structured error (M2 store swallows to ``[]``; M3
  layer re-detects via a lightweight FTS probe and surfaces to the caller).
* Pagination: ``page_size`` respected; ``total`` reflects unpaginated count;
  ``next_page_token`` returned when more records remain.
* Defense-in-depth scrub: even a leaked secret in the store is redacted on
  return (M2 pre-store scrub + M3 pre-return scrub).
"""
from __future__ import annotations

from datetime import UTC, datetime

from aggregator.core.store import Store
from aggregator.mcp import aggregator_query
from aggregator.sources.base import Record


def _rec(sid: str, subject: str, body: str, tags=()) -> Record:
    return Record(
        stable_id=sid,
        source="sessions",
        subject=subject,
        body=body,
        tags=list(tags),
        created_at=datetime(2026, 7, 25, tzinfo=UTC),
        updated_at=datetime(2026, 7, 25, tzinfo=UTC),
    )


def _seed(store, n: int = 1) -> None:
    store.migrate()
    store.upsert(
        [
            _rec(
                f"sessions:a{i}",
                f"subj{i}",
                f"refactor foo{i}.py",
                tags=["proj-alpha"],
            )
            for i in range(n)
        ]
    )


# --- happy path --------------------------------------------------------------


def test_query_returns_ok_and_records(tmp_data_home):
    store = Store()
    _seed(store)
    result = aggregator_query(dsl="source:sessions", fields="full", _store=store)
    assert result["ok"] is True
    assert result["total"] == 1
    assert len(result["records"]) == 1


def test_query_wraps_content_in_external_content_tags(tmp_data_home):
    store = Store()
    _seed(store)
    result = aggregator_query(dsl="source:sessions", fields="full", _store=store)
    body = result["records"][0]["content"]
    assert '<ExternalContent source="sessions:a0">' in body
    assert "</ExternalContent>" in body


def test_query_record_shape_has_expected_keys(tmp_data_home):
    store = Store()
    _seed(store)
    result = aggregator_query(dsl="source:sessions", fields="full", _store=store)
    r = result["records"][0]
    for key in ("stable_id", "source", "subject", "tags", "updated_at", "content"):
        assert key in r, f"missing key {key!r} in record shape"


# --- field selection ---------------------------------------------------------


def test_query_summary_is_default_and_emits_notice(tmp_data_home):
    store = Store()
    _seed(store)
    result = aggregator_query(dsl="source:sessions", _store=store)
    assert result["ok"] is True
    assert "notice" in result
    # notice must tell the caller HOW to opt in to full bodies
    assert "fields=full" in result["notice"] or "fields='full'" in result["notice"]


def test_query_summary_omits_body(tmp_data_home):
    store = Store()
    _seed(store)
    result = aggregator_query(dsl="source:sessions", fields="summary", _store=store)
    # summary content still wraps in ExternalContent (so downstream tooling
    # treats the wrapper uniformly), but the body between the tags is empty.
    body = result["records"][0]["content"]
    assert "refactor foo0.py" not in body


def test_query_full_includes_body(tmp_data_home):
    store = Store()
    _seed(store)
    result = aggregator_query(dsl="source:sessions", fields="full", _store=store)
    body = result["records"][0]["content"]
    assert "refactor foo0.py" in body


def test_query_full_has_no_notice(tmp_data_home):
    store = Store()
    _seed(store)
    result = aggregator_query(dsl="source:sessions", fields="full", _store=store)
    assert "notice" not in result


# --- structured errors -------------------------------------------------------


def test_query_bad_dsl_date_returns_structured_error(tmp_data_home):
    store = Store()
    store.migrate()
    result = aggregator_query(dsl="from:not-a-date", _store=store)
    assert result["ok"] is False
    assert "reason" in result
    assert "remediation" in result
    # never leak stack trace
    assert "Traceback" not in result["reason"]


def test_query_bad_fts_syntax_returns_structured_error(tmp_data_home):
    """FTS5 raises OperationalError on unbalanced quote; store swallows to [];
    M3 must detect + surface as ok:false with remediation."""
    store = Store()
    _seed(store)
    # Unbalanced double-quote is an FTS5 syntax error.
    result = aggregator_query(dsl='"unbalanced', _store=store)
    assert result["ok"] is False
    assert "reason" in result
    assert "remediation" in result


# --- pagination --------------------------------------------------------------


def test_query_pagination_respects_page_size(tmp_data_home):
    store = Store()
    _seed(store, n=5)
    result = aggregator_query(
        dsl="source:sessions", fields="full", page_size=2, _store=store
    )
    assert len(result["records"]) == 2
    assert result["total"] == 5
    assert "next_page_token" in result


def test_query_pagination_last_page_has_no_next_token(tmp_data_home):
    store = Store()
    _seed(store, n=2)
    result = aggregator_query(
        dsl="source:sessions", fields="full", page_size=10, _store=store
    )
    assert len(result["records"]) == 2
    assert result["total"] == 2
    assert "next_page_token" not in result


def test_query_pagination_beyond_500_rows(tmp_data_home):
    """HIGH-3: pre-fix, store.query capped at 500 so ``total`` under-reported
    the true row count and ``next_page_token`` never appeared past 500.
    Post-fix: total reflects the real count; next_page_token fires whenever
    more rows remain."""
    store = Store()
    _seed(store, n=501)
    result = aggregator_query(
        dsl="source:sessions", fields="summary", page_size=200, _store=store
    )
    assert result["ok"] is True
    assert result["total"] == 501
    assert "next_page_token" in result


def test_query_pagination_second_page_returns_remainder(tmp_data_home):
    store = Store()
    _seed(store, n=3)
    first = aggregator_query(
        dsl="source:sessions", fields="full", page_size=2, _store=store
    )
    assert "next_page_token" in first
    second = aggregator_query(
        dsl="source:sessions",
        fields="full",
        page_size=2,
        page_token=first["next_page_token"],
        _store=store,
    )
    assert second["ok"] is True
    assert len(second["records"]) == 1
    # no third page
    assert "next_page_token" not in second


# --- defense-in-depth scrub --------------------------------------------------


def test_query_scrubs_secrets_on_return(tmp_data_home):
    """Simulate an "old row" where the store's pre-store scrub was bypassed.
    Pre-return scrub in M3 MUST still redact it."""
    store = Store()
    store.migrate()
    secret = "sk-" + "ant-" + "api03-" + "x" * 44
    conn = store._c()
    conn.execute(
        "INSERT INTO records(stable_id, source, subject, body, tags, "
        "created_at, updated_at, extra) VALUES "
        "('sessions:leak', 'sessions', 's', ?, '[]', NULL, NULL, '{}')",
        (secret,),
    )
    conn.execute(
        "INSERT INTO records_fts(stable_id, source, subject, body, tags) "
        "VALUES ('sessions:leak', 'sessions', 's', ?, '')",
        (secret,),
    )
    conn.commit()
    result = aggregator_query(dsl="source:sessions", fields="full", _store=store)
    assert result["ok"] is True
    for rec in result["records"]:
        assert "sk-ant-api03" not in rec["content"]
