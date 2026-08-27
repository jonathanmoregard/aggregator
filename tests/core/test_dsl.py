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
    # Bare-date HI is end-of-day inclusive so the documented range covers
    # everything on HI (Codex Phase 2 MEDIUM fix).
    ast = parse("active:2026-07-30..2026-08-01")
    assert ast.active_from == datetime(2026, 7, 30, tzinfo=UTC)
    assert ast.active_to == datetime(2026, 8, 1, 23, 59, 59, 999999, tzinfo=UTC)


def test_parse_active_range_open_end():
    ast = parse("active:2026-07-30..")
    assert ast.active_from == datetime(2026, 7, 30, tzinfo=UTC)
    assert ast.active_to is None


def test_parse_active_range_open_start():
    ast = parse("active:..2026-08-01")
    assert ast.active_from is None
    assert ast.active_to == datetime(2026, 8, 1, 23, 59, 59, 999999, tzinfo=UTC)


def test_parse_active_range_iso_hi_untouched():
    """Full ISO datetime HI is honoured as-is; only bare dates shift to
    end-of-day inclusive."""
    ast = parse("active:..2026-08-01T15:00:00+00:00")
    assert ast.active_to == datetime(2026, 8, 1, 15, tzinfo=UTC)


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


# --- v6: by: (provenance) --------------------------------------------------
#
# ONE MORE KEY, AND FIVE MORE PLACES IT HAS TO BE REGISTERED. Three of them
# break QUIETLY when missed — the page-token fingerprint, the sessions-vs-
# records routing predicate, and the two hit-scope projections — so those are
# pinned at the MCP and store layers rather than here. This file owns the
# grammar half.


def test_parse_by_key_maps_to_provenance():
    assert parse("by:human").provenance == "human"
    assert parse("by:hook").provenance == "hook"


def test_parse_by_key_accepts_the_machine_shorthand():
    assert parse("by:machine").provenance == "machine"


def test_parse_by_key_is_case_insensitive():
    assert parse("by:HUMAN").provenance == "human"


def test_parse_by_key_refuses_an_unknown_value():
    """A CLOSED ENUM, so a typo is a parse error and not an empty page.

    ``type:`` is deliberately open — its value set is additive — and a typo
    there returns nothing. ``by:`` has exactly six accepted values, so the same
    silence would be a bug reported as "there is nothing in my history".
    """
    with pytest.raises(DSLError) as e:
        parse("by:humann")
    assert "humann" in str(e.value)
    assert "human" in str(e.value)


def test_by_is_absent_by_default():
    """No implicit filter, ever. Same contract as ``type:``."""
    assert parse("source:sessions clickable links").provenance is None


def test_format_help_documents_by_and_says_type_is_transport():
    help_text = format_help(
        sources=["sessions"],
        tags_by_source={"sessions": []},
        date_range=("2026-01-01", "2026-08-27"),
    )
    assert "by:" in help_text
    assert "machine" in help_text
    # The docs must say what ``type:`` is not, until every row is classified.
    assert "transport" in help_text.lower()


# --- v6: scope: (conjunction unit) -----------------------------------------
#
# The grammar half only. The four sites where a missed registration breaks
# WITHOUT a test failure — the page-token fingerprint, the sessions-vs-records
# routing predicate, the session-card projection and the fused keyword arm —
# are pinned in ``tests/test_mcp_conjunction_scope.py``.


def test_parse_scope_key():
    assert parse("scope:session green").scope == "session"
    assert parse("scope:observation green").scope == "observation"


def test_parse_scope_key_is_case_insensitive():
    assert parse("scope:SESSION green").scope == "session"


def test_parse_scope_refuses_an_unknown_value():
    """Closed enum, same argument as ``by:``: ``scope:sessions`` falling into
    ``ast.extra`` would leave the default in place and answer a DIFFERENT
    question in silence, which is worse than an empty page."""
    with pytest.raises(DSLError) as e:
        parse("scope:sessions green")
    assert "sessions" in str(e.value)
    assert "observation" in str(e.value)


def test_scope_is_absent_by_default():
    """``None`` and not ``'observation'``. Filling in the default would make
    EVERY ast look like it carries a sessions-ontology key, and records-shaped
    queries would stop routing to ``records``."""
    assert parse("source:sessions clickable links").scope is None


def test_format_help_documents_scope_and_what_a_quoted_run_means():
    help_text = format_help(
        sources=["sessions"],
        tags_by_source={"sessions": []},
        date_range=("2026-01-01", "2026-08-27"),
    )
    assert "scope:" in help_text
    assert "observation" in help_text
    # The default has to be stated, or a caller cannot know what they got.
    assert "DEFAULT" in help_text
    # And what quoting does, because that is the other half of "one term".
    assert "double-quoted" in help_text.lower()
