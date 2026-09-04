"""LLM tags on the MCP surface: visible in results, truthful in capabilities.

TWO TRUTHFULNESS RULES, both load-bearing:

* An item's ``llm_tags`` is a SEPARATE field from ``tags`` — a caller can
  always tell which kind of tag a record carries, and source tags are never
  silently mixed with machine-generated ones.
* A partially-tagged corpus must never look fully tagged: capabilities
  carries per-source tagged/total counts plus a note saying what they mean
  for a ``tag:`` filter, the same shape as the vector arm's coverage story.
"""

from __future__ import annotations

from datetime import UTC, datetime

from aggregator.cli import main
from aggregator.core.dsl import format_help
from aggregator.core.store import Store
from aggregator.mcp import aggregator_capabilities, aggregator_query
from aggregator.sources.base import ObservationRow, Record, SessionRow

_TS = datetime(2026, 7, 25, tzinfo=UTC)


def _rec(sid: str, subject: str, body: str, tags=(), source="github") -> Record:
    return Record(
        stable_id=sid,
        source=source,
        subject=subject,
        body=body,
        tags=list(tags),
        created_at=_TS,
        updated_at=_TS,
    )


def _store(tmp_path) -> Store:
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    return s


def test_search_result_items_carry_llm_tags_separately(tmp_path):
    s = _store(tmp_path)
    s.upsert([_rec("github:x", "subject", "body", tags=["src-tag"])])
    s.write_llm_tags([("github:x", ["llm-topic", "other-topic"], "h1")])
    result = aggregator_query("tag:llm-topic", search_mode="lexical", _store=s)
    assert result["ok"] is True
    (item,) = result["records"]
    assert item["tags"] == ["src-tag"]
    assert item["llm_tags"] == ["llm-topic", "other-topic"]


def test_untagged_record_item_has_empty_llm_tags(tmp_path):
    s = _store(tmp_path)
    s.upsert([_rec("github:y", "subject", "body", tags=["src-tag"])])
    result = aggregator_query("tag:src-tag", search_mode="lexical", _store=s)
    (item,) = result["records"]
    assert item["llm_tags"] == []


def _store_with_session(tmp_path) -> Store:
    """A store holding ONE tagged record and ONE session with an observation.

    The session half is the leak vector: sessions carry no tags, so a
    ``tag:`` filter must never surface them — neither in union mode (a bare
    ``tag:`` query) nor via the sessions path.
    """
    s = _store(tmp_path)
    s.upsert([_rec("github:acme/api:1", "pr 1", "refactor foo.py", tags=["real-tag"])])
    s.upsert_entities(
        [
            SessionRow(
                session_id="sess-a",
                root_session_id="sess-a",
                parent_session_id=None,
                kind="session",
                agent_id=None,
                agent_type=None,
                spawned_by_tool_use_id=None,
                cwd=None,
                git_branch=None,
                first_ts=_TS,
                last_ts=_TS,
                jsonl_path="/tmp/x.jsonl",
            ),
            ObservationRow(
                obs_id="o1",
                session_id="sess-a",
                root_session_id="sess-a",
                parent_obs_id=None,
                type="user",
                ts=_TS,
                model=None,
                input_tokens=None,
                output_tokens=None,
                tool_name=None,
                tool_use_id=None,
                body="hello there",
            ),
        ]
    )
    return s


def test_tag_only_query_never_returns_untagged_sessions(tmp_path):
    """Union-leak regression: a ``tag:`` filter must not match sessions.

    A bare ``tag:`` query routes to union mode, and the sessions arm used to
    ignore ``ast.tags`` entirely — every session came back unfiltered, so a
    nonexistent tag returned the whole sessions table and a real tag's page 1
    was dominated by sessions that never carried it.
    """
    s = _store_with_session(tmp_path)
    result = aggregator_query(dsl="tag:zzz-nope", _store=s)
    assert result["ok"] is True, result
    assert result["total"] == 0, {
        "total": result["total"],
        "items": result.get("records"),
    }


def test_tag_query_returns_only_the_tagged_record(tmp_path):
    s = _store_with_session(tmp_path)
    result = aggregator_query(dsl="tag:real-tag", _store=s)
    assert result["ok"] is True, result
    assert result["total"] == 1
    (item,) = result["records"]
    assert item["stable_id"] == "github:acme/api:1"


def test_sessions_scoped_tag_query_matches_nothing(tmp_path):
    """``source:sessions tag:x``: sessions have no tags, so nothing matches."""
    s = _store_with_session(tmp_path)
    result = aggregator_query(dsl="source:sessions tag:real-tag", _store=s)
    assert result["ok"] is True, result
    assert result["total"] == 0
    assert result["records"] == []


def test_records_side_tag_filter_still_works_with_source_hint(tmp_path):
    s = _store_with_session(tmp_path)
    result = aggregator_query(dsl="source:github tag:real-tag", _store=s)
    assert result["ok"] is True and result["mode"] == "records"
    assert result["total"] == 1
    assert result["records"][0]["stable_id"] == "github:acme/api:1"


def test_capabilities_exposes_llm_tag_coverage_and_note(tmp_path):
    s = _store(tmp_path)
    s.upsert(
        [
            _rec("github:a", "s", "b"),
            _rec("github:b", "s2", "b2"),
            _rec("research:c", "s3", "b3", source="research"),
        ]
    )
    h = s._c().execute(
        "SELECT src_hash FROM records WHERE stable_id = 'github:a'"
    ).fetchone()["src_hash"]
    s.write_llm_tags([("github:a", ["t-one"], h)])

    caps = aggregator_capabilities(_store=s)
    assert caps["ok"] is True
    coverage = {row["source"]: row for row in caps["llm_tag_coverage"]}
    assert coverage["github"]["tagged"] == 1
    assert coverage["github"]["total"] == 2
    assert coverage["github"]["state"] == "in_progress"
    assert coverage["research"]["state"] == "not_started"
    # The note is what stops a partially-tagged corpus reading as complete:
    # it must say tag coverage is partial-by-source and that untagged records
    # are still reachable by free text.
    note = caps["llm_tag_coverage_note"]
    assert "tag:" in note
    assert "not" in note.lower()


def test_dsl_help_says_tag_covers_llm_tags(tmp_path):
    help_text = format_help(sources=["github"], tags_by_source={"github": []})
    assert "llm" in help_text.lower()
    # The help must warn that tag: coverage is partial until the backfill
    # finishes, pointing at the coverage surface.
    assert "llm_tag_coverage" in help_text


def test_status_prints_llm_tag_coverage(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    s = _store(tmp_path)
    s.upsert([_rec("github:st", "s", "b")])
    assert main(["status"], _store=s) == 0
    out = capsys.readouterr().out
    assert "llm tag coverage" in out
    assert "github: not_started — 0/1 tagged" in out
