"""Criterion D, half three — an AND-miss is either RESCUED or EXPLAINED.

MISSION ACCEPTANCE TEST 2, verbatim: a caller who does not know the exact phrase
tries ``"PR link" "status report"``. It must return the 2026-07-28 turn, OR
return nothing WITH a notice explaining same-observation conjunction and
suggesting single-phrase queries. Silently returning irrelevant rows is the
WORST of the three outcomes — and it is what this code did before criterion D:
84 rows on the live corpus, none of them the answer, because the caller's quotes
were discarded and four ordinary words were ANDed instead.

V7 REACHES THE FIRST OUTCOME. The lexical relaxation ladder (strict AND, then
OR-of-conjuncts, then a prefix on the final term) rescues the AND-dead page and
hands back the rows that DO carry a conjunct — flagged ``lexical_relaxation:
"or"|"prefix"`` with a leading notice sentence, so the looser match can never
pose as exact (see ``tests/test_mcp_relaxation.py`` for that contract end to
end). What this file pins is the DIVISION OF LABOUR that survives v7:

* a rescued page returns the ANSWER, never the pollutant that matches no
  conjunct, and discloses the rescue plus the one query shape that asks the
  precise question (a single quoted phrase);
* when the strict probe finds the conjuncts co-occurring inside one session,
  the rescued page still names ``scope:session`` — the exact-conjunction
  answer the caller originally asked for;
* facts nobody measured are never claimed: above the probe bound the notice
  says NOTHING about sessions, in either direction;
* the conjunction notice proper still owns the pages relaxation cannot touch —
  ``scope:session`` misses and exhausted-relaxation zeros (see
  ``test_mcp_relaxation.py::test_exhausted_relaxation_is_reported...``).
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


def test_the_two_phrase_query_returns_the_answer_never_the_hook_prompt(store):
    """The hook prompt contains PR, link, status and report — every loose word.
    Only the phrases tell it apart from the answer. Strict AND still matches
    nothing (the answer says "link prs", and a quoted run means ADJACENCY), so
    pre-v7 the honest result was an explained zero. The OR tier now does
    better: the answer carries ``status report`` verbatim and comes back ON
    the page, while the hook prompt matches NEITHER phrase and stays off it —
    the rescue must never widen far enough to readmit the pollutant."""
    result = _query(store, 'source:sessions type:user "PR link" "status report"')
    assert result["total"] == 1
    assert [r["obs_id"] for r in result["records"]] == ["o-answer"]
    assert result["lexical_relaxation"] == "or"


def test_and_it_says_why(store):
    """The disclosure that keeps the rescued page honest: the rows are not
    exact conjunction matches, and the one shape that asks the precise
    question — a single quoted phrase — is named, per the acceptance test."""
    result = _query(store, 'source:sessions type:user "PR link" "status report"')
    notice = result["notice"]
    assert "LEXICAL RELAXATION APPLIED" in notice, notice
    assert "NOT exact" in notice, notice
    # "suggesting single-phrase queries", per the acceptance test.
    assert "quote a phrase" in notice, notice
    # No session holds BOTH phrases here, so the notice must not send the
    # caller to a widening that would come back empty.
    assert "scope:session" not in notice, notice


def test_dropping_the_quotes_still_admits_the_hook_prompt(store):
    """Unquoted, the words are ANDed and the transcript-quoting hook prompt
    satisfies all four in one observation — the pollution criterion C made
    visible, not something the conjunction scope can fix. Under porter
    stemming ``prs`` and ``PR`` are one term, so the human answer row now
    satisfies the AND as well and rides alongside: an EXACT page (no
    relaxation marker) where the quoted form above is still what tells the
    answer from the pollutant."""
    result = _query(store, "source:sessions type:user PR link status report")
    assert sorted(r["obs_id"] for r in result["records"]) == [
        "o-answer", "o-hook",
    ]
    assert "lexical_relaxation" not in result
    assert all(r["provenance"] is None for r in result["records"])


# --- the spec's twelve-word false negative, rescued -------------------------


def test_a_remembered_gist_is_rescued_flagged_and_names_the_shape_that_works(
    store,
):
    """Pre-v7 this was the headline false negative: eleven remembered words
    that can never all land in one 111-character turn, so the AND returned
    nothing and the notice was the whole product. The OR tier now returns the
    answer row itself — flagged, with the single-quoted-phrase advice intact,
    so the caller still learns the shape that asks precisely."""
    result = _query(
        store,
        "source:sessions type:user hand back control only when done clickable "
        "PR link executive summary",
    )
    assert result["total"] == 2
    assert "o-answer" in {r["obs_id"] for r in result["records"]}
    assert result["lexical_relaxation"] == "or"
    notice = result["notice"]
    assert "NOT exact" in notice, notice
    assert "quote a phrase" in notice, notice


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


def test_above_the_bound_the_notice_claims_nothing_about_sessions(
    store, monkeypatch
):
    """NOT-MEASURED IS NOT ZERO. Collapsing the two is how a notice starts
    stating a fact nobody established.

    Pre-v7 the empty page's notice discussed sessions either way, so above
    the probe bound it had to say "NOT CHECKED" out loud. The rescued page's
    relaxation notice only mentions sessions when the probe RAN and found
    them — so above the bound the truthful behaviour is silence: no probe,
    and no session claim in either direction."""
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
    assert result["lexical_relaxation"] == "or"
    notice = result["notice"]
    assert "DO contain" not in notice, notice
    assert "No session contains all of them" not in notice, notice
    assert "scope:session" not in notice, notice


# --- the probe, when it has something to offer ------------------------------


def test_when_the_terms_do_co_occur_in_a_session_the_notice_says_so(tmp_path):
    """The most useful sentence this can emit survives the rescue: the OR
    tier fills the page with one-phrase rows, which answers a LOOSER question
    — so when the strict probe finds a session holding ALL the phrases, the
    notice still names ``scope:session``, the exact-conjunction answer the
    caller originally asked for."""
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
    assert result["total"] == 2
    assert result["lexical_relaxation"] == "or"
    notice = result["notice"]
    assert "scope:session" in notice, notice
    assert "1 session" in notice, notice
    # ...and the key it names actually works — exactly, with no marker.
    widened = _query(
        s, 'source:sessions type:user scope:session "PR link" "status report"'
    )
    assert sorted(r["obs_id"] for r in widened["records"]) == ["o-1", "o-2"]
    assert "lexical_relaxation" not in widened


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
    """An exact AND hit (two rows under porter — see the unquoted test above)
    must carry neither the conjunction blame nor the relaxation marker."""
    result = _query(store, "source:sessions type:user PR link status report")
    assert result["total"] == 2
    assert "conjunction" not in (result.get("notice") or "")
    assert "lexical_relaxation" not in result


def test_session_cards_get_the_disclosure_too(store):
    """The cards route (``drilldown=False``) is rescued through the same
    ladder, so it owes the caller the same flag: the one matching card is the
    ANSWER session, marked as a relaxed hit."""
    result = _query(
        store, 'source:sessions type:user "PR link" "status report"',
        drilldown=False,
    )
    assert result["total"] == 1
    assert [r["stable_id"] for r in result["records"]] == ["sess-answer"]
    assert result["lexical_relaxation"] == "or"
    assert "LEXICAL RELAXATION APPLIED" in result["notice"]


def test_the_union_path_gets_it_too(store):
    """The spec's own repro was typed WITHOUT a source hint, so it lands in
    union mode — where an undisclosed rescue is just as opaque as an
    unexplained zero was."""
    result = _query(store, '"PR link" "status report"')
    assert result["total"] == 1
    assert result["lexical_relaxation"] == "or"
    assert "LEXICAL RELAXATION APPLIED" in result["notice"]
