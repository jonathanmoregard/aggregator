"""Lexical relaxation is VISIBLE at the tool boundary, and never lies.

The store's relaxation ladder (see ``tests/core/test_store_fts_relaxation``)
rescues AND-dead gist queries — but a rescued page that looked like an exact
answer would trade the documented false-negative for a silent false-positive,
which is worse. So the contract tested here is truthfulness end to end:

* an exact match NEVER carries the marker — no ``lexical_relaxation`` key,
  no relaxation sentence in the notice;
* a relaxed page ALWAYS carries both — ``lexical_relaxation: "or"|"prefix"``
  in the response and a leading notice sentence saying the rows are not
  exact conjunction matches;
* the CLI prints the marker as its own line, so a terminal reader sees it
  without parsing JSON;
* when even relaxation finds nothing, the empty-page conjunction notice says
  the relaxed tiers were tried too, instead of blaming the conjunction for
  an emptiness the conjunction did not cause;
* ``scope:session`` keeps its strict per-conjunct semantics, unmarked.

Plus the regression the whole feature exists for: a multi-word gist query
that returns 0 under strict AND returns the relevant rows via relaxation.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from aggregator import cli
from aggregator.core.store import Store
from aggregator.mcp import aggregator_query
from aggregator.sources.base import ObservationRow, Record, SessionRow


def _sess(session_id: str) -> SessionRow:
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
        first_ts=datetime(2026, 7, 25, 10, 0, tzinfo=UTC),
        last_ts=datetime(2026, 7, 25, 10, 5, tzinfo=UTC),
        jsonl_path="/tmp/x.jsonl",
    )


def _obs(obs_id: str, session_id: str, body: str) -> ObservationRow:
    return ObservationRow(
        obs_id=obs_id,
        session_id=session_id,
        root_session_id=session_id,
        parent_obs_id=None,
        type="user",
        ts=datetime(2026, 7, 25, 10, 0, tzinfo=UTC),
        model=None,
        input_tokens=None,
        output_tokens=None,
        tool_name=None,
        tool_use_id=None,
        body=body,
    )


def _rec(sid: str, subject: str, body: str) -> Record:
    return Record(
        stable_id=sid,
        source="github",
        subject=subject,
        body=body,
        tags=[],
        created_at=datetime(2026, 7, 25, tzinfo=UTC),
        updated_at=datetime(2026, 7, 25, tzinfo=UTC),
    )


@pytest.fixture
def corpus(tmp_path):
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert_entities(
        [
            _sess("s1"),
            _obs("o-token", "s1", "the deploy failed because the token expired"),
            _obs("o-renew", "s1", "we renewed the certificate afterwards"),
            _obs("o-alphabet", "s1", "alphabet soup for lunch"),
        ]
    )
    s.upsert(
        [
            _rec("r-running", "shoes", "great for running"),
            _rec("r-boots", "boots", "solid walking boots"),
        ]
    )
    yield s
    s.close()


# --- the marker, both ways ---------------------------------------------------


def test_exact_match_never_carries_the_marker(corpus):
    result = aggregator_query(
        "source:sessions deploy token", drilldown=True, _store=corpus
    )
    assert result["ok"] is True
    assert {r["obs_id"] for r in result["records"]} == {"o-token"}
    assert "lexical_relaxation" not in result
    assert "relax" not in result.get("notice", "").lower()


def test_and_miss_is_rescued_by_or_and_flagged(corpus):
    result = aggregator_query(
        "source:sessions token renewed", drilldown=True, _store=corpus
    )
    assert result["ok"] is True
    assert {r["obs_id"] for r in result["records"]} == {"o-token", "o-renew"}
    assert result["lexical_relaxation"] == "or"
    notice = result["notice"]
    assert "relax" in notice.lower(), notice
    assert "not exact" in notice.lower() or "NOT exact" in notice, notice


def test_or_miss_is_rescued_by_prefix_and_flagged(corpus):
    result = aggregator_query(
        "source:sessions zzzmissing alphab", drilldown=True, _store=corpus
    )
    assert result["ok"] is True
    assert {r["obs_id"] for r in result["records"]} == {"o-alphabet"}
    assert result["lexical_relaxation"] == "prefix"
    assert "relax" in result["notice"].lower()


def test_session_cards_route_carries_the_marker_too(corpus):
    result = aggregator_query("source:sessions token renewed", _store=corpus)
    assert result["ok"] is True
    assert {r["stable_id"] for r in result["records"]} == {"s1"}
    assert result["lexical_relaxation"] == "or"


def test_records_route_carries_the_marker(corpus):
    result = aggregator_query("source:github running boots", _store=corpus)
    assert result["ok"] is True
    assert {r["stable_id"] for r in result["records"]} == {
        "r-running",
        "r-boots",
    }
    assert result["lexical_relaxation"] == "or"


def test_union_route_carries_the_marker(corpus):
    result = aggregator_query("token renewed", _store=corpus)
    assert result["ok"] is True
    assert result["total"] > 0
    assert result["lexical_relaxation"] == "or"


def test_marker_resets_between_requests_on_a_shared_store(corpus):
    relaxed = aggregator_query(
        "source:sessions token renewed", drilldown=True, _store=corpus
    )
    assert relaxed["lexical_relaxation"] == "or"
    exact = aggregator_query(
        "source:sessions deploy token", drilldown=True, _store=corpus
    )
    assert "lexical_relaxation" not in exact


# --- the regression the feature exists for -----------------------------------


def test_multiword_gist_query_is_rescued(corpus):
    """The documented worst failure: a remembered-gist query of several words,
    ANDed into one observation, returned nothing. Now it returns the relevant
    turns — flagged as relaxed, never as exact."""
    gist = "source:sessions deploy token expired renewal reminder"
    result = aggregator_query(gist, drilldown=True, _store=corpus)
    assert result["ok"] is True
    assert result["total"] > 0
    ids = {r["obs_id"] for r in result["records"]}
    assert "o-token" in ids, ids
    assert result["lexical_relaxation"] in ("or", "prefix")


# --- honesty when even relaxation finds nothing ------------------------------


def test_exhausted_relaxation_is_reported_not_blamed_on_the_conjunction(corpus):
    result = aggregator_query(
        "source:sessions zzzmissing qqqmissing", drilldown=True, _store=corpus
    )
    assert result["ok"] is True
    assert result["total"] == 0
    assert "lexical_relaxation" not in result
    notice = result["notice"]
    # The old notice blamed the AND ("had to appear in ONE observation") and
    # offered scope:session as the remedy. Both would be lies now: OR and
    # prefix were tried, so no term matches anything and no widening helps —
    # scope:session may be MENTIONED only to say it cannot help.
    assert "relax" in notice.lower(), notice
    assert "had to appear in ONE observation" not in notice, notice
    assert "re-run with `scope:session`" not in notice, notice
    assert "`scope:session` cannot help" in notice, notice
    # The stemming fact rides here so the caller does not go off to
    # hand-pluralise terms the tokenizer already folds.
    assert "stems" in notice, notice


def test_scope_session_stays_strict_and_unmarked(corpus):
    hit = aggregator_query(
        "source:sessions scope:session token renewed", _store=corpus
    )
    assert hit["ok"] is True
    assert {r["stable_id"] for r in hit["records"]} == {"s1"}
    assert "lexical_relaxation" not in hit

    miss = aggregator_query(
        "source:sessions scope:session token zzzmissing", _store=corpus
    )
    assert miss["ok"] is True
    assert miss["total"] == 0
    assert "lexical_relaxation" not in miss


# --- CLI surface -------------------------------------------------------------


def test_cli_prints_a_relaxation_line(tmp_data_home, corpus, capsys):
    rc = cli.main(
        ["query", "source:sessions token renewed", "--drilldown"], _store=corpus
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "# lexical_relaxation: or" in out, out


def test_cli_prints_no_relaxation_line_for_exact_matches(
    tmp_data_home, corpus, capsys
):
    rc = cli.main(
        ["query", "source:sessions deploy token", "--drilldown"], _store=corpus
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "lexical_relaxation" not in out, out
