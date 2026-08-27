"""Criterion D, half two — WHICH ROW a conjunction has to be satisfied in.

THE DEFAULT IS THE OBSERVATION AND IT ALREADY WAS. ``obs_fts`` holds one row
per observation, so ``MATCH`` has always ANDed a query's terms inside a single
turn; the spec's "matches split across turns hours apart" was read off bodies
that quote whole transcripts, which is criterion C's finding. This file PINS
that default so nothing quietly widens it, and adds the widening as something a
caller asks for by name.

``scope:session`` is the new capability, not a restored one: it lets the terms
sit in different turns of the same session root, which answers "which session
covered both" rather than "which moment said it". It is never the default,
because the moment is what a recall tool is asked for and a session here runs to
hundreds of turns.

THE FOUR SITES THAT BREAK WITHOUT A TEST FAILURE, each covered below:
the page-token fingerprint (a token minted under one scope continuing under
another), the sessions-vs-records routing predicate, the session-card
projection, and the keyword arm inside the fused hybrid scope.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

import aggregator.mcp as mcp
from aggregator.core.dsl import DSLError, parse
from aggregator.core.store import Store
from aggregator.sources.base import (
    SCOPE_OBSERVATION,
    SCOPE_SESSION,
    ObservationRow,
    QueryAST,
    SessionRow,
)

_TS = datetime(2026, 7, 28, 15, 32, 21, tzinfo=UTC)

# The answer turn from the incident, verbatim.
_ANSWER = (
    "link prs. Also: please somehow make you always link PRs when delivering "
    "a status report: I want clickable links"
)


def _sess(sid: str, hours: int = 4) -> SessionRow:
    return SessionRow(
        session_id=sid, root_session_id=sid, parent_session_id=None,
        kind="session", agent_id=None, agent_type=None,
        spawned_by_tool_use_id=None, cwd="/home/jonathan/Repos/aggregator",
        git_branch="main", first_ts=_TS, last_ts=_TS + timedelta(hours=hours),
        jsonl_path=f"/tmp/{sid}.jsonl",
    )


def _obs(oid, sid, body, *, seconds=0, obs_type="user", provenance=None):
    return ObservationRow(
        obs_id=oid, session_id=sid, root_session_id=sid, parent_obs_id=None,
        type=obs_type, ts=_TS + timedelta(seconds=seconds), model=None,
        input_tokens=None, output_tokens=None, tool_name=None,
        tool_use_id=None, body=body, provenance=provenance,
    )


@pytest.fixture
def store(tmp_path):
    """Its own temporary database. Never the live cache.

    ``sess-split`` is the shape the spec described and the code never had: the
    two terms live in different turns, three hours apart.
    ``sess-together`` carries both in one turn.
    """
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert_entities(
        [
            _sess("sess-split"),
            _obs("o-early", "sess-split", "the gate finally went green", seconds=0),
            _obs("o-late", "sess-split", "now open a pull request on main",
                 seconds=10800),
            _sess("sess-together"),
            _obs("o-both", "sess-together", "the pull request is green", seconds=0),
        ]
    )
    return s


def _query(store, dsl, **kw):
    kw.setdefault("fields", "summary")
    kw.setdefault("drilldown", True)
    result = mcp.aggregator_query(dsl=dsl, _store=store, **kw)
    assert result["ok"] is True, result
    return result


# --- the default, pinned ----------------------------------------------------


def test_the_default_is_one_observation(store):
    """Two terms in two turns of one session do NOT match. This is the fact the
    spec reported the opposite of, and it is measured here rather than assumed.
    """
    result = _query(store, "source:sessions type:user green pull request")
    assert [r["obs_id"] for r in result["records"]] == ["o-both"]


def test_naming_the_default_changes_nothing(store):
    """``scope:observation`` is the same query as no ``scope:`` at all."""
    bare = _query(store, "source:sessions type:user green pull request")
    named = _query(
        store, "source:sessions type:user scope:observation green pull request"
    )
    assert [r["obs_id"] for r in named["records"]] == [
        r["obs_id"] for r in bare["records"]
    ]
    assert named["total"] == bare["total"]


def test_the_default_holds_for_session_cards_too(store):
    result = _query(
        store, "source:sessions type:user green pull request", drilldown=False
    )
    assert [r["stable_id"] for r in result["records"]] == ["sess-together"]


# --- the widening, on request only -----------------------------------------


def test_scope_session_lets_the_terms_sit_in_different_turns(store):
    result = _query(
        store, "source:sessions type:user scope:session green pull request",
        drilldown=False,
    )
    assert sorted(r["stable_id"] for r in result["records"]) == [
        "sess-split", "sess-together"
    ]


def test_scope_session_drills_down_to_every_contributing_turn(store):
    """Once a session qualifies, BOTH turns are part of the answer — demanding
    every term of each row again would collapse back to the default."""
    result = _query(
        store, "source:sessions type:user scope:session green pull request"
    )
    assert sorted(r["obs_id"] for r in result["records"]) == [
        "o-both", "o-early", "o-late"
    ]


def test_scope_session_does_not_reach_across_sessions(store):
    """The widening is bounded by the session root. A term in one session and a
    term in another is still nothing."""
    s = Store(db_path=store.db_path)
    s.upsert_entities(
        [
            _sess("sess-a"), _obs("o-a", "sess-a", "kubernetes rollback plan"),
            _sess("sess-b"), _obs("o-b", "sess-b", "helm chart authoring"),
        ]
    )
    result = _query(
        s, "source:sessions type:user scope:session kubernetes helm"
    )
    assert result["total"] == 0


def test_a_single_term_is_the_same_query_under_both_scopes(store):
    """With nothing to intersect the two scopes must agree exactly, or the
    widening has changed an answer it had no business touching."""
    obs = _query(store, "source:sessions type:user green")
    sess = _query(store, "source:sessions type:user scope:session green")
    assert [r["obs_id"] for r in sess["records"]] == [
        r["obs_id"] for r in obs["records"]
    ]


def test_the_count_and_the_page_agree_under_the_widening(store):
    """A total computed in one scope and a page in another is the "plausible but
    wrong" answer this module refuses everywhere else."""
    ast = parse("source:sessions type:user scope:session green pull request")
    assert store.count_observations(ast) == len(store.query_observations(ast))


def test_scope_session_honours_the_type_filter(store):
    """A session qualifies on turns that pass the query's filters, not on any
    turn at all — otherwise ``type:user`` would be advisory."""
    s = Store(db_path=store.db_path)
    s.upsert_entities(
        [
            _sess("sess-mixed"),
            _obs("o-user", "sess-mixed", "the gate went green"),
            _obs("o-asst", "sess-mixed", "opening a pull request now",
                 seconds=60, obs_type="assistant"),
        ]
    )
    result = _query(
        s, "source:sessions type:user scope:session green pull request"
    )
    assert "sess-mixed" not in {r["session_id"] for r in result["records"]}


# --- the value set is closed ------------------------------------------------


def test_an_unknown_scope_value_is_refused_not_silently_ignored():
    """``scope:sessions`` landing in ``ast.extra`` would leave the default in
    place and answer a different question in silence."""
    with pytest.raises(DSLError) as e:
        parse("source:sessions scope:sessions green")
    assert "sessions" in str(e.value)
    assert SCOPE_SESSION in str(e.value)


def test_the_tool_refuses_it_at_the_boundary(store):
    result = mcp.aggregator_query(
        dsl="source:sessions scope:turn green", _store=store, drilldown=True
    )
    assert result["ok"] is False
    assert "turn" in result["reason"]


def test_absent_stays_none_so_the_ast_does_not_always_carry_a_key():
    """If ``parse`` filled in the default, every AST would look like it carried
    a sessions key and records routing would break."""
    assert parse("green pull request").scope is None
    assert parse("scope:observation green").scope == SCOPE_OBSERVATION


# --- the sites that break QUIETLY -------------------------------------------


def test_the_page_token_fingerprint_binds_the_scope():
    """A token minted under one scope must not continue under the other: the
    widened query cuts its offsets from a strictly larger row set."""
    default = mcp._query_fingerprint(parse("source:sessions a b"), True, "hybrid")
    named = mcp._query_fingerprint(
        parse("source:sessions scope:observation a b"), True, "hybrid"
    )
    widened = mcp._query_fingerprint(
        parse("source:sessions scope:session a b"), True, "hybrid"
    )
    # Absent and the default spelled out are the SAME question and must share
    # a token — refusing a caller who said the default out loud would be a
    # refusal with no cause.
    assert default == named
    assert default != widened


def test_a_token_minted_under_one_scope_is_refused_under_the_other(store):
    first = mcp.aggregator_query(
        dsl="source:sessions type:user scope:session green pull request",
        fields="summary", drilldown=True, page_size=1, _store=store,
    )
    token = first["next_page_token"]
    reused = mcp.aggregator_query(
        dsl="source:sessions type:user green pull request",
        fields="summary", drilldown=True, page_size=1,
        page_token=token, _store=store,
    )
    assert reused["ok"] is False


def test_scope_alone_routes_to_the_sessions_ontology():
    """Omit it from ``_has_sessions_keys`` and ``scope:session`` falls through to
    ``union``, half of which is ``records`` — document-shaped, with no session
    for the conjunction to range over."""
    assert mcp._has_sessions_keys(parse("scope:session green")) is True
    assert mcp._has_sessions_keys(parse("scope:observation green")) is True
    assert mcp._route_mode(parse("scope:session green")) == "sessions"


def test_scope_widens_the_session_card_projection(store):
    """``_text_hit_scope`` is what decides which cards the keyword arm matched.
    Left at the default there, every card ``scope:session`` surfaced would be
    reported as uncorroborated on the query that asked for it."""
    default = store._text_hit_scope(
        QueryAST(text="green pull request", obs_type="user")
    )
    widened = store._text_hit_scope(
        QueryAST(text="green pull request", obs_type="user",
                 scope=SCOPE_SESSION)
    )
    assert default[0] == {"sess-together"}
    assert widened[0] == {"sess-split", "sess-together"}
    assert mcp._lexical_session_ids(
        store, "green pull request", "user",
        frozenset({"o-early", "o-late"}), None, SCOPE_SESSION,
    ) >= frozenset({"sess-split"})


def test_scope_widens_the_keyword_arm_of_the_fused_scope(store, monkeypatch):
    """The hybrid route computes its own keyword arm. Narrower there than the
    FTS5-only route is the "search got smarter and stopped finding it" failure
    ``_fused_id_scope`` refuses by name."""
    monkeypatch.setattr(mcp, "_widen_chunk_ids", lambda ids: list(ids))
    ast = parse("source:sessions type:user scope:session green pull request")
    scope, _hits, lexical = mcp._fused_id_scope(
        store, "observations", ast.text or "", object(), frozen=["o-both"],
        lexical_ast=ast,
    )
    assert lexical == frozenset({"o-both", "o-early", "o-late"})
    assert scope == frozenset({"o-both", "o-early", "o-late"})


# --- records are untouched --------------------------------------------------


def test_records_shaped_sources_keep_document_level_matching(tmp_path):
    """A whole document IS the unit there, so ``scope:`` has no business in it —
    and the records path must keep answering its own way."""
    from aggregator.sources.base import Record

    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert(
        [
            Record(
                stable_id="github:1", source="github", subject="pr",
                body="the gate went green\n\n...\n\nopen a pull request",
                tags=[], created_at=_TS, updated_at=_TS,
            )
        ]
    )
    got = s.query(QueryAST(source="github", text="green pull request"))
    assert [r.stable_id for r in got] == ["github:1"]
