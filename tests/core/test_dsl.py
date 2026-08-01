"""Tests for the flat filter DSL parser (M2).

Grammar: ``source:X tag:a,b from:YYYY-MM-DD to:YYYY-MM-DD [freeform text]``
Everything else lands in ``extra`` verbatim so per-source ``Source.search()`` can
interpret it. Split on the FIRST colon only because source-specific IDs (github
stable IDs like ``github:owner/repo:number``) may themselves contain colons.
"""
from datetime import UTC, datetime

import pytest

from aggregator.core.dsl import DSLError, format_help, parse


def test_parse_source_only():
    ast = parse("source:sessions")
    assert ast.source == "sessions"
    assert ast.tags == []


def test_parse_source_tag_from_to():
    ast = parse("source:github tag:pr,open from:2026-07-01 to:2026-07-31")
    assert ast.source == "github"
    assert set(ast.tags) == {"pr", "open"}
    assert ast.from_date == datetime(2026, 7, 1, tzinfo=UTC)
    assert ast.to_date == datetime(2026, 7, 31, tzinfo=UTC)


def test_parse_freeform_text():
    ast = parse("source:sessions refactor foo.py")
    assert ast.source == "sessions"
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
    """A raw stable_id like ``github:owner/repo:42`` in the freeform region has
    two colons; the DSL must not confuse the second colon for a key separator."""
    ast = parse("source:github github:owner/repo:42")
    assert ast.source == "github"
    # The `github:owner/repo:42` token has an unknown key `github`; the whole
    # RHS (which includes the second colon) must be preserved verbatim in extra.
    assert ast.extra.get("github") == "owner/repo:42"


def test_format_help_lists_sources():
    help_text = format_help(
        sources=["sessions", "github"],
        tags_by_source={
            "sessions": ["proj-alpha", "claude-opus-4-7"],
            "github": ["pr", "open"],
        },
        date_range=("2026-01-01", "2026-07-31"),
    )
    assert "sessions" in help_text
    assert "github" in help_text
    assert "proj-alpha" in help_text
    assert "2026-01-01" in help_text
