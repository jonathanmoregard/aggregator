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
from aggregator.sources.base import Record

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
