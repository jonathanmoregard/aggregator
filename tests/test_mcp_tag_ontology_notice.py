"""A ``tag:`` filter on a sessions route must DISCLOSE, never go silent.

Round 1 taught the store that sessions carry no tags (``tag:`` under the
sessions/observations WHERE renders ``1=0``), which fixed the union leak —
and created this round's HIGH: ``source:sessions tag:main`` returned
``{"ok": true, "records": [], "total": 0}`` with no notice, an empty page
indistinguishable from "no session on that topic". Adopted from the live
repro (scratchpad ``test_repro_silent_empty.py``).

DISCLOSURE-ONLY by design: these tests pin that the rows never change —
empty stays empty, hits stay hits — and only the ``notice`` is added. The
key phrase asserted is "carry no tags"; if the wording moves, keep that
phrase or update BOTH the notice and this file together.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from aggregator.core.store import Store
from aggregator.mcp import aggregator_query
from aggregator.sources.base import ObservationRow, Record, SessionRow

_TS = datetime(2026, 7, 25, 10, 0, tzinfo=UTC)

#: The phrase every tag-ontology notice must carry, and no other page may.
_KEY_PHRASE = "carry no tags"


def _rec(sid: str, subject: str, body: str, tags=()) -> Record:
    return Record(
        stable_id=sid,
        source="github",
        subject=subject,
        body=body,
        tags=list(tags),
        created_at=_TS,
        updated_at=_TS,
    )


def _sess(session_id: str, *, git_branch=None) -> SessionRow:
    return SessionRow(
        session_id=session_id,
        root_session_id=session_id,
        parent_session_id=None,
        kind="session",
        agent_id=None,
        agent_type=None,
        spawned_by_tool_use_id=None,
        cwd=None,
        git_branch=git_branch,
        first_ts=_TS,
        last_ts=_TS,
        jsonl_path="/tmp/x.jsonl",
    )


def _obs(obs_id: str, session_id: str, body: str) -> ObservationRow:
    return ObservationRow(
        obs_id=obs_id,
        session_id=session_id,
        root_session_id=session_id,
        parent_obs_id=None,
        type="user",
        ts=_TS,
        model=None,
        input_tokens=None,
        output_tokens=None,
        tool_name=None,
        tool_use_id=None,
        body=body,
    )


@pytest.fixture
def store(tmp_path) -> Store:
    """One tagged record + one session (branch ``main``) with a matching turn.

    The session's ``git_branch`` is the bait: it used to surface as a
    session-card field named ``tags``, which is exactly what led callers to
    try ``source:sessions tag:main`` in the first place.
    """
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert_entities(
        [
            _sess("sess-alpha", git_branch="main"),
            _obs("u-1", "sess-alpha", "work on branch main"),
        ]
    )
    s.upsert([_rec("github:acme/api:1", "pr main", "fix main", tags=["main"])])
    return s


def test_sessions_route_tag_query_discloses_instead_of_going_silent(store):
    """The adopted repro: ``source:sessions tag:main`` — empty, but SAID."""
    result = aggregator_query(dsl="source:sessions tag:main", _store=store)
    assert result["ok"] is True, result
    assert result["mode"] == "sessions"
    # Disclosure-only: the round-1 filtering semantics stand untouched.
    assert result["total"] == 0
    assert result["records"] == []
    notice = result.get("notice") or ""
    assert _KEY_PHRASE in notice, f"silent empty page: notice={notice!r}"
    # The way out is named: drop one key or the other.
    assert "tag:" in notice
    assert "source:sessions" in notice


def test_sessions_key_route_tag_query_discloses_too(store):
    """Any sessions-route KEY (here ``by:``) + ``tag:`` gets the same notice."""
    result = aggregator_query(dsl="tag:main by:human", _store=store)
    assert result["ok"] is True, result
    assert result["mode"] == "sessions"
    assert result["total"] == 0
    notice = result.get("notice") or ""
    assert _KEY_PHRASE in notice, f"silent empty page: notice={notice!r}"


def test_union_route_with_text_keeps_record_hits_and_discloses(store):
    """``tag:main fix`` — the records half answers, the sessions half is
    disclosed as excluded rather than silently absent."""
    result = aggregator_query(dsl="tag:main fix", _store=store)
    assert result["ok"] is True, result
    assert result["mode"] == "union"
    ids = {item["stable_id"] for item in result["records"]}
    # The hits are KEPT — disclosure never suppresses the records arm.
    assert "github:acme/api:1" in ids
    # The session holding "main" in its text does NOT appear (tag guard) …
    assert "sess-alpha" not in ids
    # … and the page says so instead of posing as the complete union.
    notice = result.get("notice") or ""
    assert _KEY_PHRASE in notice, f"undisclosed union page: notice={notice!r}"


def test_pure_tag_query_carries_no_ontology_notice(store):
    """``tag:main`` alone is a records-ontology question answered completely
    on its own terms — no lecture attached to every plain tag page."""
    result = aggregator_query(dsl="tag:main", _store=store)
    assert result["ok"] is True, result
    ids = {item["stable_id"] for item in result["records"]}
    assert ids == {"github:acme/api:1"}
    assert _KEY_PHRASE not in (result.get("notice") or "")


def test_records_route_tag_query_carries_no_ontology_notice(store):
    """``source:github tag:main`` — records ARE the tag-bearing shape."""
    result = aggregator_query(dsl="source:github tag:main", _store=store)
    assert result["ok"] is True, result
    assert result["mode"] == "records"
    assert result["total"] == 1
    assert _KEY_PHRASE not in (result.get("notice") or "")


def test_drilldown_observations_route_discloses_as_well(store):
    """``drilldown=True`` rides the sessions route; same empty, same notice."""
    result = aggregator_query(
        dsl="source:sessions tag:main", drilldown=True, _store=store
    )
    assert result["ok"] is True, result
    assert result["mode"] == "observations"
    assert result["total"] == 0
    assert _KEY_PHRASE in (result.get("notice") or "")
