"""``scope:session``'s empty page gets the same honest diagnosis as every other.

THE DEFECT ADOPTED HERE (scratchpad ``test_repro_scope_session_false_cooccur``):
the ``scope:session`` ladder recorded NOTHING. ``_compute_session_hit_scope``
and ``_session_scope_obs_ids`` ran real FTS5 statements, found the sessions that
carry every conjunct, and never called ``_note_lexical_matched`` — so
``Store.lexical_matches`` stayed ``None`` all the way to the tool boundary, the
filter branch of the diagnosis could not fire, and the page fell through to
"They do not co-occur in any one session".

That sentence is a claim about the corpus, and here it was false: the terms DO
co-occur, in one session, and a ``from:`` filter is what emptied the page. The
caller was then told to use fewer terms — a remedy that cannot undo a filter,
which is the exact failure the empty-page composer was written to end.

Pinned below, in both directions: the confirmed-co-occurrence page names the
filters, and the page where the terms genuinely never co-occur keeps the old
wording, which is a MEASUREMENT there and must not be silenced by the fix.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from aggregator.core.store import Store
from aggregator.mcp import aggregator_query
from aggregator.sources.base import ObservationRow, SessionRow

_TS = datetime(2026, 7, 28, 15, 32, 21, tzinfo=UTC)

#: The corpus claim a filter-emptied ``scope:session`` page may never make.
_DENIAL = "do not co-occur"


def _sess(session_id: str) -> SessionRow:
    return SessionRow(
        session_id=session_id,
        root_session_id=session_id,
        parent_session_id=None,
        kind="session",
        agent_id=None,
        agent_type=None,
        spawned_by_tool_use_id=None,
        cwd=None,
        git_branch=None,
        first_ts=_TS,
        last_ts=_TS + timedelta(hours=1),
        jsonl_path=f"/tmp/{session_id}.jsonl",
    )


def _obs(oid: str, body: str, session_id: str = "sess-both", seconds: int = 0):
    return ObservationRow(
        obs_id=oid,
        session_id=session_id,
        root_session_id=session_id,
        parent_obs_id=None,
        type="user",
        ts=_TS + timedelta(seconds=seconds),
        model=None,
        input_tokens=None,
        output_tokens=None,
        tool_name=None,
        tool_use_id=None,
        body=body,
        provenance=None,
    )


@pytest.fixture
def co_occurring(tmp_path) -> Store:
    """One session carrying "fix" in one turn and "bug" in another."""
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert_entities(
        [
            _sess("sess-both"),
            _obs("o-fix", "please fix the parser"),
            _obs("o-bug", "that bug is in the tokenizer", seconds=60),
        ]
    )
    return s


@pytest.fixture
def apart(tmp_path) -> Store:
    """The same two terms, in two DIFFERENT sessions: no co-occurrence."""
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert_entities(
        [
            _sess("sess-fix"),
            _obs("o-fix", "please fix the parser", "sess-fix"),
            _sess("sess-bug"),
            _obs("o-bug", "that bug is in the tokenizer", "sess-bug"),
        ]
    )
    return s


# --- the adopted repro -------------------------------------------------------


def test_filter_emptied_page_must_not_deny_co_occurrence(co_occurring):
    """`fix` and `bug` DO co-occur in sess-both; only from:2030 emptied the
    page. Adopted verbatim from the scratchpad repro."""
    together, _ = co_occurring._session_hit_scope("fix bug", None, None)
    assert together, "precondition: the terms co-occur in one session"
    result = aggregator_query(
        dsl="source:sessions type:user scope:session fix bug from:2030-01-01",
        fields="summary",
        drilldown=True,
        _store=co_occurring,
    )
    assert result["ok"] is True, result
    notice = " ".join((result.get("notice") or "").split())
    assert result["total"] == 0, result["total"]
    assert _DENIAL not in notice, notice


def test_and_it_names_the_filter_that_did_it(co_occurring):
    """Not denying the corpus is half of it; the other half is the remedy. The
    filter is the fix, and rewriting the query is not."""
    result = aggregator_query(
        dsl="source:sessions type:user scope:session fix bug from:2030-01-01",
        drilldown=True,
        _store=co_occurring,
    )
    notice = result["notice"]
    assert "THE WORDS ARE NOT THE PROBLEM" in notice, notice
    assert "from:2030-01-01" in notice, notice
    assert "the fix is a FILTER" in notice, notice
    # …and never the advice that cannot work here.
    assert "try fewer terms" not in notice, notice


def test_the_session_card_page_gets_the_same_diagnosis(co_occurring):
    """The cards route reaches the intersection through ``_hit_scope`` rather
    than ``_scoped_obs_ids``, so it is a second recording site and a second
    chance to lie."""
    result = aggregator_query(
        dsl="source:sessions scope:session fix bug from:2030-01-01",
        _store=co_occurring,
    )
    assert result["ok"] is True and result["total"] == 0
    notice = result["notice"]
    assert _DENIAL not in notice, notice
    assert "THE WORDS ARE NOT THE PROBLEM" in notice, notice


def test_the_confidence_hedge_agrees_with_it(co_occurring):
    """One page, two sentences. The hedge reads the same measurement."""
    result = aggregator_query(
        dsl="source:sessions type:user scope:session fix bug from:2030-01-01",
        drilldown=True,
        _store=co_occurring,
    )
    reason = result["low_confidence_reason"]
    assert "nothing matched this query on either arm" not in reason, reason
    assert "FILTER" in reason, reason


# --- the measured zero survives the fix --------------------------------------


def test_terms_that_really_never_co_occur_keep_the_old_wording(apart):
    """THE CASE THE WORDING WAS WRITTEN FOR. The intersection ran and came back
    empty, so "they do not co-occur in any one session" is a measurement — and
    ``scope:session`` really is the widest unit this ontology has."""
    result = aggregator_query(
        dsl="source:sessions type:user scope:session fix bug",
        drilldown=True,
        _store=apart,
    )
    assert result["ok"] is True and result["total"] == 0
    notice = result["notice"]
    assert _DENIAL in notice, notice
    assert "widest unit" in notice, notice


def test_a_filter_on_top_of_a_real_miss_still_blames_neither(apart):
    """No co-occurrence AND a filter: the intersection measured zero, so the
    filter is not what emptied the page and must not be named as the remedy."""
    result = aggregator_query(
        dsl="source:sessions type:user scope:session fix bug from:2030-01-01",
        drilldown=True,
        _store=apart,
    )
    assert result["ok"] is True and result["total"] == 0
    notice = result["notice"]
    assert _DENIAL in notice, notice
    assert "THE WORDS ARE NOT THE PROBLEM" not in notice, notice
