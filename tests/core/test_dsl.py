"""DSL parser tests (v2, Schema B keys added).

Grammar:
  source:X tag:a,b from:D to:D
  session:X top:X agent:Y type:T active:LO..HI
  [freeform text]
"""
from datetime import UTC, datetime

import pytest

from aggregator.core.dsl import DSLError, format_help, parse


def test_parse_source_only():
    ast = parse("source:sessions")
    assert ast.source == "sessions"


def test_parse_source_tag_from_to():
    ast = parse("source:github tag:pr,open from:2026-07-01 to:2026-07-31")
    assert ast.source == "github"
    assert set(ast.tags) == {"pr", "open"}
    assert ast.from_date == datetime(2026, 7, 1, tzinfo=UTC)
    assert ast.to_date == datetime(2026, 7, 31, tzinfo=UTC)


def test_parse_freeform_text():
    ast = parse("source:sessions refactor foo.py")
    assert "refactor" in ast.text
    assert "foo.py" in ast.text


def test_parse_per_source_keys_go_to_extra():
    ast = parse("source:github state:open author:@me")
    assert ast.extra["state"] == "open"
    assert ast.extra["author"] == "@me"


def test_parse_bad_date_raises():
    with pytest.raises(DSLError):
        parse("source:sessions from:not-a-date")


def test_parse_empty_query():
    ast = parse("")
    assert ast.source is None


def test_parse_split_on_first_colon_only():
    ast = parse("source:github github:owner/repo:42")
    assert ast.source == "github"
    assert ast.extra.get("github") == "owner/repo:42"


# --- v2 keys --------------------------------------------------------------


def test_parse_session_key_maps_to_session_id():
    ast = parse("session:abc-uuid")
    assert ast.session_id == "abc-uuid"
    assert ast.top_session_id is None


def test_parse_top_key_maps_to_top_session_id():
    ast = parse("top:abc-uuid")
    assert ast.top_session_id == "abc-uuid"
    assert ast.session_id is None


def test_parse_agent_key():
    ast = parse("agent:agent001")
    assert ast.agent_id == "agent001"


def test_parse_type_key():
    ast = parse("type:tool_use")
    assert ast.obs_type == "tool_use"


def test_parse_active_range_both_sides():
    ast = parse("active:2026-07-30..2026-08-01")
    assert ast.active_from == datetime(2026, 7, 30, tzinfo=UTC)
    assert ast.active_to == datetime(2026, 8, 1, tzinfo=UTC)


def test_parse_active_range_open_end():
    ast = parse("active:2026-07-30..")
    assert ast.active_from == datetime(2026, 7, 30, tzinfo=UTC)
    assert ast.active_to is None


def test_parse_active_range_open_start():
    ast = parse("active:..2026-08-01")
    assert ast.active_from is None
    assert ast.active_to == datetime(2026, 8, 1, tzinfo=UTC)


def test_parse_active_range_bad_syntax_raises():
    with pytest.raises(DSLError):
        parse("active:2026-07-30")  # missing ..


def test_parse_active_range_bad_date_raises():
    with pytest.raises(DSLError):
        parse("active:not-a-date..2026-08-01")


def test_format_help_includes_v2_keys():
    help_text = format_help(
        sources=["sessions", "subagents", "github"],
        tags_by_source={"sessions": [], "subagents": [], "github": ["pr"]},
        date_range=("2026-01-01", "2026-07-31"),
    )
    assert "session:" in help_text
    assert "top:" in help_text
    assert "agent:" in help_text
    assert "type:" in help_text
    assert "active:" in help_text


def test_format_help_lists_sources():
    help_text = format_help(
        sources=["sessions", "github"],
        tags_by_source={
            "sessions": [],
            "github": ["pr", "open"],
        },
        date_range=("2026-01-01", "2026-07-31"),
    )
    assert "sessions" in help_text
    assert "github" in help_text
    assert "2026-01-01" in help_text
