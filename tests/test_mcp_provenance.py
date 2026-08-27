"""Criterion C at the tool surface: the row says who wrote it, loudly.

THE DECISION THIS FILE PINS. The recall path is NOT narrowed to human-only, in
``Store`` or in the tool. A default filter measured as breaking for three
in-repo callers — ``_first_user_prompt`` (29% of sampled user turns are
``hook``-class and headless sessions OPEN with one, so session cards lose their
labels), the frozen eval baseline (criterion E could then no longer separate a
retrieval win from a filter), and ``_count_scope_for`` /
``_session_body_preview``, which would under-count in silence.

Instead: every row carries ``provenance``, ``by:`` narrows on request, and the
``notice`` says out loud how much of the page a machine wrote. That is the same
call the codebase already makes for page tokens at ``_PageTokenError`` — a
silently narrowed result set is "plausible but wrong", and this one is loud.

THE THREE QUIET REGISTRATION SITES. A new AST field breaks without a test
failure in exactly three places, and each is covered below: the page-token
fingerprint (a token minted under one filter would continue under another), the
sessions-vs-records routing predicate (``by:human`` alone would route to
``records``), and the two hit-scope projections that surface session cards.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

import aggregator.mcp as mcp
from aggregator.core.dsl import parse
from aggregator.core.provenance import AGENT, HOOK, HUMAN
from aggregator.core.store import Store
from aggregator.sources.base import ObservationRow, QueryAST, SessionRow

_TS = datetime(2026, 7, 28, 15, 32, 21, tzinfo=UTC)

_HUMAN_TURN = (
    "link prs. Also: please somehow make you always link PRs when delivering "
    "a status report: I want clickable links"
)
_HOOK_PROMPT = (
    "You are watching a Claude Code session for a specific failure mode. The "
    "user asked for clickable links in every status report; decide whether "
    "Claude complied."
)
_AGENT_BRIEF = (
    "Research how the tool returns clickable links and report back with a "
    "summary."
)


def _sess(sid: str) -> SessionRow:
    return SessionRow(
        session_id=sid, root_session_id=sid, parent_session_id=None,
        kind="session", agent_id=None, agent_type=None,
        spawned_by_tool_use_id=None, cwd="/home/jonathan/Repos/aggregator",
        git_branch="main", first_ts=_TS, last_ts=_TS + timedelta(minutes=5),
        jsonl_path=f"/tmp/{sid}.jsonl",
    )


def _obs(obs_id, sid, body, *, provenance, offset=0, obs_type="user"):
    return ObservationRow(
        obs_id=obs_id, session_id=sid, root_session_id=sid, parent_obs_id=None,
        type=obs_type, ts=_TS + timedelta(seconds=offset), model=None,
        input_tokens=None, output_tokens=None, tool_name=None,
        tool_use_id=None, body=body, provenance=provenance,
    )


@pytest.fixture
def store(tmp_path):
    """Its own temporary database. Never the live cache."""
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert_entities(
        [
            _sess("sess-a"),
            _obs("o-human", "sess-a", _HUMAN_TURN, provenance=HUMAN, offset=0),
            _obs("o-hook", "sess-a", _HOOK_PROMPT, provenance=HOOK, offset=10),
            _obs("o-agent", "sess-a", _AGENT_BRIEF, provenance=AGENT, offset=20),
        ]
    )
    return s


def _query(store, dsl, **kw):
    kw.setdefault("fields", "summary")
    kw.setdefault("drilldown", True)
    result = mcp.aggregator_query(dsl=dsl, _store=store, **kw)
    assert result["ok"] is True, result
    return result


# --- the row carries it -----------------------------------------------------


def test_every_observation_row_reports_its_provenance(store):
    """One column of legibility, beside the snippet criterion A added."""
    result = _query(store, 'source:sessions type:user "clickable links"')
    got = {r["obs_id"]: r["provenance"] for r in result["records"]}
    assert got == {"o-human": HUMAN, "o-hook": HOOK, "o-agent": AGENT}


def test_full_mode_carries_it_too(store):
    result = _query(store, 'source:sessions type:user "clickable links"', fields="full")
    assert all("provenance" in r for r in result["records"])


# --- no default filter, anywhere --------------------------------------------


def test_the_default_query_returns_the_machine_rows_as_well(store):
    """NO SILENT NARROWING. The caller sees the whole page and is told."""
    result = _query(store, 'source:sessions type:user "clickable links"')
    assert {r["obs_id"] for r in result["records"]} == {
        "o-human", "o-hook", "o-agent"
    }
    assert result["total"] == 3


def test_session_cards_keep_their_labels(store):
    """``_first_user_prompt`` must not be filtered — this is the concrete
    breakage a human-only default caused: a headless session OPENS with a
    hook-class turn, so its card would fall back to ``session <uuid>``."""
    s = Store(db_path=store.db_path)
    s.upsert_entities(
        [
            _sess("sess-headless"),
            _obs(
                "o-brief", "sess-headless",
                "You are a NixOS config drift analyzer. clickable links.",
                provenance=HOOK, offset=0,
            ),
        ]
    )
    result = _query(
        s, 'source:sessions type:user "clickable links"', drilldown=False
    )
    card = next(
        r for r in result["records"] if r["stable_id"] == "sess-headless"
    )
    assert not card["subject"].startswith("session "), card["subject"]
    assert "NixOS" in card["subject"]


# --- the notice says how much of the page a machine wrote -------------------


def test_the_notice_counts_the_machine_rows_and_names_the_filter(store):
    result = _query(store, 'source:sessions type:user "clickable links"')
    notice = result["notice"]
    assert "2 of 3" in notice, notice
    assert "by:human" in notice, notice


def test_the_notice_is_silent_on_an_all_human_page(store):
    result = _query(store, f"source:sessions type:user by:{HUMAN} clickable")
    assert "by:human" not in (result.get("notice") or "")


def test_the_notice_appears_in_full_mode_too(store):
    """It is a fact about the ROWS, not about the presentation."""
    result = _query(
        store, 'source:sessions type:user "clickable links"', fields="full"
    )
    assert "by:human" in result["notice"]


def test_an_unclassified_page_says_so_rather_than_claiming_authorship(tmp_path):
    """Before the backfill runs, NULL must not be reported as machine or human."""
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert_entities([_sess("sess-u"), _obs("o-u", "sess-u", "clickable links", provenance=None)])
    result = _query(s, 'source:sessions type:user "clickable links"')
    assert result["records"][0]["provenance"] is None
    notice = result["notice"]
    assert "NOT YET CLASSIFIED" in notice, notice
    assert "provenance --backfill" in notice, notice


def test_an_empty_by_page_explains_itself_when_nothing_is_classified(tmp_path):
    """A ``by:`` filter over an unclassified corpus returns nothing, and an
    empty page that cannot say why is the failure this whole mission is about."""
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert_entities([_sess("sess-u"), _obs("o-u", "sess-u", "clickable links", provenance=None)])
    result = _query(s, "source:sessions type:user by:human clickable")
    assert result["total"] == 0
    assert "provenance --backfill" in result["notice"], result["notice"]


# --- by: narrows, on request only -------------------------------------------


def test_by_human_narrows_the_page(store):
    result = _query(store, "source:sessions type:user by:human clickable")
    assert [r["obs_id"] for r in result["records"]] == ["o-human"]


def test_by_machine_is_the_complement(store):
    result = _query(store, "source:sessions type:user by:machine clickable")
    assert sorted(r["obs_id"] for r in result["records"]) == ["o-agent", "o-hook"]


def test_an_unknown_by_value_is_refused_not_silently_empty(store):
    result = mcp.aggregator_query(
        dsl="source:sessions by:humann", _store=store, drilldown=True
    )
    assert result["ok"] is False
    assert "humann" in result["reason"]


# --- the three sites that break QUIETLY -------------------------------------


def test_the_page_token_fingerprint_binds_the_by_filter():
    """Omit it and a token minted under one filter continues under another,
    addressing a row position that never existed in the new result set."""
    unfiltered = mcp._query_fingerprint(parse("source:sessions clickable"), True, "hybrid")
    human = mcp._query_fingerprint(
        parse("source:sessions by:human clickable"), True, "hybrid"
    )
    machine = mcp._query_fingerprint(
        parse("source:sessions by:machine clickable"), True, "hybrid"
    )
    assert len({unfiltered, human, machine}) == 3


def test_a_token_minted_under_one_filter_is_refused_under_another(store):
    first = mcp.aggregator_query(
        dsl='source:sessions type:user "clickable links"',
        fields="summary", drilldown=True, page_size=1, _store=store,
    )
    token = first["next_page_token"]
    reused = mcp.aggregator_query(
        dsl='source:sessions type:user by:human "clickable links"',
        fields="summary", drilldown=True, page_size=1,
        page_token=token, _store=store,
    )
    assert reused["ok"] is False


def test_by_alone_routes_to_the_sessions_ontology():
    """Omit it from ``_has_sessions_keys`` and ``by:human`` hits ``records``,
    which has no provenance at all — an empty page for a valid query."""
    assert mcp._has_sessions_keys(parse("by:human")) is True
    assert mcp._route_mode(parse("by:human")) == "sessions"
    assert mcp._route_mode(parse("by:human clickable")) == "sessions"


def test_by_scopes_the_session_card_projection(store):
    """``_fts_hit_scope`` takes ``obs_type`` explicitly and needed the same
    treatment: a card must surface only on a hit that passes the filter."""
    roots, exacts = store._fts_hit_scope("clickable", provenance=HUMAN)
    assert roots == {"sess-a"}
    only_hook = store._fts_hit_scope("NixOS", provenance=HUMAN)
    assert only_hook == (set(), set())


def test_by_scopes_the_hybrid_id_projection(store):
    """``_obs_id_hit_scope`` is the v5 twin and must stay behaviourally
    identical to its FTS sibling, filter included."""
    all_ids = ["o-human", "o-hook", "o-agent"]
    roots, _exacts = store._obs_id_hit_scope(all_ids, provenance=HUMAN)
    assert roots == {"sess-a"}
    assert store._obs_id_hit_scope(["o-hook"], provenance=HUMAN) == (set(), set())


def test_by_filters_a_session_card_page(store):
    """End to end through the tool, not just the helper."""
    result = _query(store, "source:sessions by:human clickable", drilldown=False)
    assert [r["stable_id"] for r in result["records"]] == ["sess-a"]
    assert result["records"][0]["matching_observations"] == 1


def test_a_session_card_page_filtered_to_nothing_comes_back_empty(store):
    result = _query(store, "source:sessions by:command clickable", drilldown=False)
    assert result["total"] == 0


# --- the store is never narrowed on its own --------------------------------


def test_the_store_applies_no_default_filter(store):
    """The eval harness and ``_first_user_prompt`` both go through here."""
    assert len(store.query_observations(QueryAST())) == 3
    assert store.count_observations(QueryAST()) == 3
