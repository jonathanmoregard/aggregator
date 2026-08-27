"""Criterion D, half three — an empty page says WHY it is empty.

MISSION ACCEPTANCE TEST 2, verbatim: a caller who does not know the exact phrase
tries ``"PR link" "status report"``. It must return the 2026-07-28 turn, OR
return nothing WITH a notice explaining same-observation conjunction and
suggesting single-phrase queries. Silently returning irrelevant rows is the
WORST of the three outcomes — and it is what this code did before criterion D:
84 rows on the live corpus, none of them the answer, because the caller's quotes
were discarded and four ordinary words were ANDed instead.

The spec's own headline false negative is here too — twelve remembered words
that can never all land in one 111-character turn. There is no retrieval fix for
that shape: the terms are individually common, this path has no relevance
ordering (``ORDER BY ts ASC``), so OR-ing them would replace nothing with
thousands of rows in timestamp order, which is the worst outcome wearing a
different hat. The fix is that the failure stops being silent, and that the
notice names the one query shape that does work.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

import aggregator.mcp as mcp
from aggregator.core.store import Store
from aggregator.sources.base import ObservationRow, SessionRow

_TS = datetime(2026, 7, 28, 15, 32, 21, tzinfo=UTC)

_ANSWER = (
    "link prs. Also: please somehow make you always link PRs when delivering "
    "a status report: I want clickable links"
)
#: The shape that outranked it: a hook prompt quoting a whole transcript, so it
#: satisfies every loose word in one observation and would survive
#: same-observation conjunction untouched. Criterion C's finding, restated here
#: because it is exactly why D had to be measured after C.
_HOOK = (
    "You are watching a Claude Code session for a specific failure mode. "
    "TRANSCRIPT: the user asked Claude to link the PR in every report on the "
    "status of the branch, and Claude did not. Decide whether Claude complied."
)


def _sess(sid: str) -> SessionRow:
    return SessionRow(
        session_id=sid, root_session_id=sid, parent_session_id=None,
        kind="session", agent_id=None, agent_type=None,
        spawned_by_tool_use_id=None, cwd="/home/jonathan/Repos/aggregator",
        git_branch="main", first_ts=_TS, last_ts=_TS + timedelta(hours=4),
        jsonl_path=f"/tmp/{sid}.jsonl",
    )


def _obs(oid, sid, body, *, seconds=0, obs_type="user"):
    return ObservationRow(
        obs_id=oid, session_id=sid, root_session_id=sid, parent_obs_id=None,
        type=obs_type, ts=_TS + timedelta(seconds=seconds), model=None,
        input_tokens=None, output_tokens=None, tool_name=None,
        tool_use_id=None, body=body, provenance=None,
    )


@pytest.fixture
def store(tmp_path):
    """Its own temporary database. Never the live cache."""
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert_entities(
        [
            _sess("sess-answer"),
            _obs("o-answer", "sess-answer", _ANSWER),
            _sess("sess-hook"),
            _obs("o-hook", "sess-hook", _HOOK),
        ]
    )
    return s


def _query(store, dsl, **kw):
    kw.setdefault("fields", "summary")
    kw.setdefault("drilldown", True)
    result = mcp.aggregator_query(dsl=dsl, _store=store, **kw)
    assert result["ok"] is True, result
    return result


# --- acceptance test 2 ------------------------------------------------------


def test_the_two_phrase_query_abstains_instead_of_handing_back_the_hook_prompt(
    store,
):
    """The hook prompt contains PR, link, status and report — every loose word.
    Only the phrases tell it apart from the answer, and the answer does not
    carry ``PR link`` either ("link prs"), so nothing is the honest result."""
    result = _query(store, 'source:sessions type:user "PR link" "status report"')
    assert result["total"] == 0
    assert result["records"] == []


def test_and_it_says_why(store):
    result = _query(store, 'source:sessions type:user "PR link" "status report"')
    notice = result["notice"]
    assert "ONE observation" in notice, notice
    assert '"PR link"' in notice and '"status report"' in notice, notice
    assert "scope:" in notice, notice
    # "suggesting single-phrase queries", per the acceptance test.
    assert "one phrase" in notice or "single quoted phrase" in notice, notice


def test_dropping_the_quotes_is_the_query_that_used_to_bury_the_answer(store):
    """Unquoted, the words are ANDed and the transcript-quoting hook prompt
    satisfies all four in one observation — which is the pollution criterion C
    made visible, not something the conjunction scope can fix."""
    result = _query(store, "source:sessions type:user PR link status report")
    assert [r["obs_id"] for r in result["records"]] == ["o-hook"]
    assert result["records"][0]["provenance"] is None


# --- the spec's twelve-word false negative ---------------------------------


def test_a_remembered_gist_abstains_and_names_the_shape_that_works(store):
    result = _query(
        store,
        "source:sessions type:user hand back control only when done clickable "
        "PR link executive summary",
    )
    assert result["total"] == 0
    notice = result["notice"]
    assert "11 terms" in notice, notice
    assert "one phrase" in notice or "single quoted phrase" in notice, notice


def test_the_probe_still_runs_for_a_remembered_gist(store, monkeypatch):
    """THE BOUND MUST NOT SKIP THE QUERIES THE NOTICE EXISTS FOR.

    An earlier bound of 8 conjuncts skipped exactly the utterance-length shape
    this notice was written for, and the skip rendered as "no session contains
    all of them" — measured false on the live corpus, where fourteen do. One
    index scan per term costs ~0.08 s there, so the probe is affordable and the
    bound is generous.
    """
    calls: list[str] = []
    real = Store._session_hit_scope
    monkeypatch.setattr(
        Store, "_session_hit_scope",
        lambda self, text, *a, **k: (calls.append(text), real(self, text, *a, **k))[1],
    )
    _query(
        store,
        "source:sessions type:user hand back control only when done clickable "
        "PR link executive summary",
    )
    assert len(calls) == 1


def test_above_the_bound_it_says_it_did_not_look_rather_than_reporting_zero(
    store, monkeypatch
):
    """NOT-MEASURED IS NOT ZERO. Collapsing the two is how a notice starts
    stating a fact nobody established."""
    monkeypatch.setattr(mcp, "_SCOPE_PROBE_MAX_CONJUNCTS", 3)
    probed: list[str] = []
    real = Store._session_hit_scope
    monkeypatch.setattr(
        Store, "_session_hit_scope",
        lambda self, text, *a, **k: (probed.append(text), real(self, text, *a, **k))[1],
    )
    result = _query(
        store,
        "source:sessions type:user hand back control only when done clickable "
        "PR link executive summary",
    )
    assert probed == []
    notice = result["notice"]
    assert "NOT CHECKED" in notice, notice
    assert "No session contains all of them" not in notice, notice


# --- the probe, when it has something to offer ------------------------------


def test_when_the_terms_do_co_occur_in_a_session_the_notice_says_so(tmp_path):
    """The most useful sentence this can emit: not "nothing", but "nothing in
    one turn — here is the key that finds it"."""
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert_entities(
        [
            _sess("sess-split"),
            _obs("o-1", "sess-split", "please include a PR link every time"),
            _obs("o-2", "sess-split", "and put it in the status report",
                 seconds=10800),
        ]
    )
    result = _query(s, 'source:sessions type:user "PR link" "status report"')
    assert result["total"] == 0
    notice = result["notice"]
    assert "scope:session" in notice, notice
    assert "1 session" in notice, notice
    # ...and the key it names actually works.
    widened = _query(
        s, 'source:sessions type:user scope:session "PR link" "status report"'
    )
    assert sorted(r["obs_id"] for r in widened["records"]) == ["o-1", "o-2"]


def test_an_empty_scope_session_page_does_not_recommend_itself(tmp_path):
    """Already widened and still empty: there is no wider unit to suggest, and
    suggesting one would be a lie the caller can check in one call."""
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert_entities([_sess("sess-x"), _obs("o-x", "sess-x", "nothing relevant")])
    result = _query(
        s, 'source:sessions type:user scope:session "PR link" "status report"'
    )
    assert result["total"] == 0
    notice = result["notice"]
    assert "widest unit" in notice, notice
    assert "re-run with `scope:session`" not in notice, notice


# --- it stays quiet when it has nothing to add ------------------------------


def test_a_single_term_that_misses_gets_no_conjunction_notice(store):
    """There is no conjunction to blame, and blaming one would be noise on
    every genuine miss."""
    result = _query(store, 'source:sessions type:user "cassandra"')
    assert result["total"] == 0
    assert "conjunction" not in (result.get("notice") or "")


def test_a_page_with_rows_gets_no_conjunction_notice(store):
    result = _query(store, "source:sessions type:user PR link status report")
    assert result["total"] == 1
    assert "conjunction" not in (result.get("notice") or "")


def test_session_cards_get_the_notice_too(store):
    result = _query(
        store, 'source:sessions type:user "PR link" "status report"',
        drilldown=False,
    )
    assert result["total"] == 0
    assert "ONE observation" in result["notice"]


def test_the_union_path_gets_it_too(store):
    """The spec's own repro was typed WITHOUT a source hint, so it lands in
    union mode — where an unexplained zero is just as opaque."""
    result = _query(store, '"PR link" "status report"')
    assert result["total"] == 0
    assert "ONE observation" in result["notice"]
