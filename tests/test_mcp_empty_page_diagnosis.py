"""An empty page gets ONE diagnosis, and every sentence on it must be true.

Three notices used to stamp the same page from three places — the relaxation
marker, the tag-ontology disclosure and the conjunction notice — each reading
one fact off the store, each right about its own fact, and together producing
pages that argued with themselves. This file pins the composed result.

THE HEADLINE DEFECT, adopted from the live repro
(scratchpad ``test_repro_false_corpus_claim.py``): ``Store.lexical_relaxation
is None`` meant BOTH "the strict tier answered" and "no tier matched anything".
So when a ``tag:``/``source:``/``from:`` filter excluded every row the words
DID match, the page told the caller the corpus contains none of those terms and
to try different words — three false claims, the last of which cannot be acted
on, because rewriting a query does not undo a filter.
``Store.lexical_matches`` is the missing half, and it is a count the ladder
already had in hand.

The other contradictions pinned here:

* a page whose visible hits are EXACT record matches is never stamped "NOT
  exact matches" because the OTHER union arm relaxed out of sight;
* ``scope:session`` is never offered as a way out of a ``tag:`` query — under
  ``tag:`` that re-run is empty by definition, and the same page says so;
* the union ontology notice does not promise "the record hits here are
  unaffected" on a page with no record hits;
* the sessions-route way out names the key the QUERY used, not a fixed list
  the caller never typed;
* a truly-empty page keeps the honest "every tier came back empty" wording —
  the fix must not silence the case it was written for.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from aggregator.core.store import Store
from aggregator.mcp import aggregator_query
from aggregator.sources.base import ObservationRow, Record, SessionRow

_TS = datetime(2026, 7, 25, 10, 0, tzinfo=UTC)

#: The three claims a filter-emptied page must never make about the corpus.
_CORPUS_BLAME = (
    "contain none of these terms",
    "every tier came back empty",
    "Try different words",
)


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


def _sess(
    session_id: str, origin: str = "claude-code", ts: datetime = _TS
) -> SessionRow:
    return SessionRow(
        session_id=session_id,
        root_session_id=session_id,
        parent_session_id=None,
        kind="session",
        agent_id=None,
        agent_type=None,
        spawned_by_tool_use_id=None,
        cwd="/x",
        git_branch="main",
        first_ts=ts,
        last_ts=ts + timedelta(minutes=5),
        jsonl_path=f"/tmp/{session_id}.jsonl",
        origin=origin,
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


# --- 1. the words matched; a filter took them --------------------------------


@pytest.fixture
def filtered(tmp_path) -> Store:
    """The corpus DOES contain "fix" and "bug" — in an UNTAGGED record."""
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert([_rec("github:acme/api:1", "pr 1", "fix bug")])
    return s


def test_tag_filtered_empty_page_does_not_blame_the_corpus(filtered):
    """The adopted repro. The terms are in the index; the ``tag:`` filter is
    what emptied the page, so none of the corpus-blame sentences may appear."""
    result = aggregator_query(dsl="tag:main fix bug", _store=filtered)
    assert result["ok"] is True, result
    assert result["total"] == 0, "precondition: the tag: filter empties the page"
    notice = result.get("notice") or ""
    for claim in _CORPUS_BLAME:
        assert claim not in notice, (claim, notice)


def test_it_says_what_actually_happened_and_names_the_filter(filtered):
    """Not blaming the corpus is half the job; the other half is telling the
    caller where to look. A count of matched rows, the filters in play by
    name, and "the fix is a FILTER" — because the actionable remedy is the
    whole reason this branch exists."""
    result = aggregator_query(dsl="tag:main fix bug", _store=filtered)
    notice = result["notice"]
    assert "THE WORDS ARE NOT THE PROBLEM" in notice, notice
    assert "matched 1 row(s)" in notice, notice
    assert "tag:main" in notice, notice
    assert "the fix is a FILTER" in notice, notice


def test_a_date_filter_gets_the_same_honest_diagnosis(filtered):
    """``tag:`` is not special here — any row filter that removes every match
    produces the same page, and naming the wrong key would be its own lie."""
    result = aggregator_query(dsl="from:2030-01-01 fix bug", _store=filtered)
    assert result["ok"] is True and result["total"] == 0
    notice = result["notice"]
    assert "from:2030-01-01" in notice, notice
    for claim in _CORPUS_BLAME:
        assert claim not in notice, (claim, notice)


def test_the_records_route_gets_it_too(filtered):
    """``source:github`` routes to records, which never carried an empty-page
    notice at all — a record is a document, so the turn-shaped conjunction
    prose was never true there. The filter diagnosis is ontology-neutral and
    therefore is."""
    result = aggregator_query(
        dsl="source:github tag:main fix bug", _store=filtered
    )
    assert result["ok"] is True and result["mode"] == "records"
    assert result["total"] == 0
    notice = result["notice"]
    assert "THE WORDS ARE NOT THE PROBLEM" in notice, notice
    # …without borrowing the sessions ontology's unit.
    assert "one observation" not in notice, notice


# --- the honest zero survives the fix ----------------------------------------


def test_a_genuinely_empty_corpus_still_says_every_tier_was_tried(tmp_path):
    """THE CASE THE OLD WORDING WAS WRITTEN FOR MUST NOT BE SILENCED. When the
    ladder RAN and no tier matched, "the rows this query can see contain none
    of these terms" is a measurement, and it stays."""
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert_entities([_sess("s1"), _obs("o1", "s1", "nothing relevant here")])
    result = aggregator_query(
        dsl="source:sessions zzzmissing qqqmissing", drilldown=True, _store=s
    )
    assert result["ok"] is True and result["total"] == 0
    notice = result["notice"]
    assert "every tier came back empty" in notice, notice
    assert "contain none of these terms" in notice, notice


# --- 2. the relaxation marker names the arm whose rows are on the page -------


def test_union_page_of_exact_record_hits_is_not_stamped_relaxed(tmp_path):
    """ITEM 2, and it is a contradiction the caller can see in one response.

    Union runs the ladder once per ontology. Here the RECORDS arm matches
    ``deploy token`` strictly, while the OBSERVATIONS arm has to relax to OR to
    find anything — and the request-wide marker keeps the deeper of the two. So
    a page holding nothing but exact record hits came back flagged
    ``lexical_relaxation: "or"`` with a notice saying "these are NOT exact
    matches", about rows that are.
    """
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    # Records side: one row carrying BOTH terms — the strict tier answers.
    s.upsert([_rec("github:exact", "deploy", "deploy token rotation")])
    # Sessions side: two rows carrying ONE term each — strict AND is empty, so
    # the observations ladder relaxes to OR. Both are older than the record, so
    # neither reaches the first page below.
    old = _TS - timedelta(days=30)
    s.upsert_entities(
        [
            _sess("s-old", ts=old),
            _obs("o-a", "s-old", "the deploy failed"),
            _obs("o-b", "s-old", "rotate the token"),
        ]
    )
    result = aggregator_query(
        dsl="deploy token", page_size=1, _store=s
    )
    assert result["ok"] is True
    assert [r["stable_id"] for r in result["records"]] == ["github:exact"]
    assert "lexical_relaxation" not in result, result.get("lexical_relaxation")
    assert "NOT exact" not in (result.get("notice") or ""), result["notice"]


def test_a_union_page_that_does_show_relaxed_rows_is_still_stamped(tmp_path):
    """The attribution must not become a way to hide a rescue: when the
    relaxed arm's rows ARE on the page, the marker stands."""
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert_entities(
        [
            _sess("s1"),
            _obs("o-a", "s1", "the deploy failed"),
            _obs("o-b", "s1", "rotate the token"),
        ]
    )
    result = aggregator_query(dsl="deploy token", _store=s)
    assert result["ok"] is True and result["total"] > 0
    assert result["lexical_relaxation"] == "or"
    assert "NOT exact" in result["notice"]


# --- 3b. scope:session is never offered as a way out of a tag: query ---------


def test_a_relaxed_tag_page_does_not_send_the_caller_to_scope_session(tmp_path):
    """Sessions carry no tags, so ``tag:x scope:session`` is empty BY
    DEFINITION. Offering it beside a notice that says exactly that is a
    guaranteed-empty dead end AND a self-contradiction on one page."""
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    # A tagged record that matches only under the OR tier…
    s.upsert([_rec("github:t", "pr", "deploy notes", tags=["main"])])
    # …plus a session that DOES hold both terms across two turns, which is
    # precisely what makes the probe want to recommend scope:session.
    s.upsert_entities(
        [
            _sess("s1"),
            _obs("o-a", "s1", "the deploy failed"),
            _obs("o-b", "s1", "rotate the token"),
        ]
    )
    result = aggregator_query(dsl="tag:main deploy token", _store=s)
    assert result["ok"] is True
    assert result["lexical_relaxation"] == "or"
    assert "scope:session" not in result["notice"], result["notice"]


# --- 3a. under tag:, the ontology notice owns the sessions-route page --------


def test_sessions_route_tag_page_does_not_also_blame_the_corpus(tmp_path):
    """Two incompatible reasons for one empty page: "empty BY DEFINITION"
    (the tag ontology) and "the corpus contains none of these terms" (the
    conjunction). Only the first is true here, so only it is said."""
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert_entities([_sess("s1"), _obs("o1", "s1", "deploy token rotation")])
    result = aggregator_query(
        dsl="source:sessions tag:main deploy token", _store=s
    )
    assert result["ok"] is True and result["total"] == 0
    notice = result["notice"]
    assert "empty BY DEFINITION" in notice, notice
    for claim in _CORPUS_BLAME:
        assert claim not in notice, (claim, notice)


# --- 3c. no promise of record hits on a page that has none ------------------


def test_union_ontology_notice_does_not_claim_hits_it_does_not_have(filtered):
    """"The record hits here are unaffected" over an empty page reads as
    "your answer is intact" when there is no answer at all."""
    result = aggregator_query(dsl="tag:main fix bug", _store=filtered)
    assert result["total"] == 0 and result["records"] == []
    notice = result["notice"]
    assert "The record hits here are unaffected" not in notice, notice
    assert "no record hits here either" in notice, notice


def test_and_it_still_says_so_when_there_ARE_record_hits(tmp_path):
    """The mirror case: the sentence is conditional, not deleted."""
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert([_rec("github:hit", "pr main", "fix main", tags=["main"])])
    result = aggregator_query(dsl="tag:main fix", _store=s)
    assert result["ok"] is True and result["total"] == 1
    assert "The record hits here are unaffected" in result["notice"]


# --- 3d. the way out names the key the QUERY used ---------------------------


def test_chatgpt_route_way_out_names_the_key_the_caller_typed(tmp_path):
    """``source:chatgpt`` is session-shaped (``_SESSIONS_SOURCES``), so a
    ``tag:`` query routes here — and the way out used to name only
    ``source:sessions/subagents``, keys this caller never typed. A way out
    through a key you did not use leads nowhere."""
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert_entities(
        [_sess("c1", origin="chatgpt"), _obs("o1", "c1", "deploy token")]
    )
    result = aggregator_query(dsl="source:chatgpt tag:main", _store=s)
    assert result["ok"] is True and result["mode"] == "sessions"
    notice = result["notice"]
    assert "source:chatgpt" in notice, notice
    assert "source:sessions/subagents" not in notice, notice


def test_a_by_route_way_out_names_by(tmp_path):
    """Same rule with no ``source:`` at all: ``by:`` is what routed it."""
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert_entities([_sess("s1"), _obs("o1", "s1", "deploy")])
    result = aggregator_query(dsl="tag:main by:human", _store=s)
    assert result["ok"] is True and result["mode"] == "sessions"
    notice = result["notice"]
    assert "by:" in notice, notice
    assert "source:sessions/subagents" not in notice, notice


# --- 4e. tag: case folding is documented where the key is ------------------


def test_tag_case_insensitivity_is_stated_on_both_help_surfaces():
    """``tag:`` matches ``COLLATE NOCASE``, so a hand-typed ``tag:bug`` finds
    github's ``Bug`` label. A recall fact nobody can discover by reading the
    tag inventory, stated identically on the DSL help and the capabilities
    note so the two cannot drift."""
    from aggregator.core.dsl import format_help
    from aggregator.mcp import _LLM_TAG_NOTE

    help_text = format_help(["github"], {"github": ["bug"]})
    for surface in (help_text, _LLM_TAG_NOTE):
        assert "CASE-INSENSITIVE" in surface, surface
        assert "tag:bug" in surface, surface


def test_tag_matching_really_is_case_insensitive(tmp_path):
    """The claim above, measured against the store rather than trusted."""
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert([_rec("github:b", "pr", "body", tags=["Bug"])])
    result = aggregator_query(dsl="tag:bug", _store=s)
    assert [r["stable_id"] for r in result["records"]] == ["github:b"]
