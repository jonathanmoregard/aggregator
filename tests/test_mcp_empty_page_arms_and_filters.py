"""An empty page's diagnosis must be true of EACH ARM and of EVERY filter.

Three sentences that were each right about a request-wide fact and wrong about
the page they landed on:

* **"the semantic arm, the only one that ran"** — ``lexical_unavailable`` is
  ``any`` over the union's arms, and it preempted the filter branch. So a page
  whose records arm had run, matched, and been emptied by a date filter carried
  that sentence directly beside the composer's "the keyword arm matched 1
  row(s)". If the keyword arm ran and matched, it was not unavailable.

* **"drop or widen one (start with X) and the matching rows come back"** — a
  promise, and a false one whenever two filters each exclude the matched rows
  independently. Dropping ``tag:`` gets the caller a second empty page and a
  reason to stop trusting the notice.

* **"Those rows matched under the RELAXED `or` tier"** — read off the
  request-wide marker, which keeps the DEEPEST tier any arm reached. The union
  runs one ladder per ontology, so rows the records arm matched STRICTLY were
  labelled relaxed because the observations arm relaxed out of sight. The
  per-arm probes from the relaxation-attribution fix already hold the answer.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from aggregator.core.store import Store
from aggregator.mcp import aggregator_query
from aggregator.sources.base import ObservationRow, Record, SessionRow

_TS = datetime(2026, 7, 25, 10, 0, tzinfo=UTC)

#: The unqualified relaxation suffix. True only when EVERY arm that matched
#: relaxed; a lie about a page whose record hits matched strictly.
_ALL_RELAXED = "Those rows matched under the RELAXED"


class StubEmbedder:
    """No real model, ever. One axis, so any text is any other's neighbour."""

    @staticmethod
    def _vec(_text: str) -> np.ndarray:
        v = np.zeros(768, dtype=np.float32)
        v[0] = 1.0
        return v

    def embed_query(self, query: str) -> np.ndarray:
        return self._vec(query)

    def embed_documents(self, docs: list[str]) -> np.ndarray:
        return np.array([self._vec(d) for d in docs], dtype=np.float32)


def _rec(sid: str, body: str, tags=()) -> Record:
    return Record(
        stable_id=sid,
        source="github",
        subject="pr",
        body=body,
        tags=list(tags),
        created_at=_TS,
        updated_at=_TS,
    )


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
        first_ts=_TS,
        last_ts=_TS + timedelta(minutes=5),
        jsonl_path=f"/tmp/{session_id}.jsonl",
    )


def _obs(obs_id: str, body: str, session_id: str = "s1") -> ObservationRow:
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
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    return s


@pytest.fixture
def embedder(monkeypatch) -> StubEmbedder:
    stub = StubEmbedder()
    monkeypatch.setattr("aggregator.mcp._get_embedder", lambda: stub)
    return stub


# --- 1. a dropped arm does not make a matching arm "unavailable" -------------


@pytest.fixture
def one_arm_down(store, embedder, monkeypatch) -> Store:
    """The records arm runs and matches; the observations arm falls over.

    ``from:2030-01-01`` then empties the page, so the request carries both
    facts at once — an arm that failed, and an arm that matched rows a filter
    removed — which is the pair the hedge used to report as one.
    """
    store.upsert([_rec("github:hit", "fix bug in the parser")])
    store.upsert_entities([_sess("s1"), _obs("o1", "fix bug in the parser")])
    vec = StubEmbedder().embed_documents(["fix bug in the parser"])
    store.upsert_vec_observations([("o1", vec[0])])
    store.mark_embedded("observations", ["o1"], state="ok")

    def boom(_text):
        raise sqlite3.OperationalError("simulated FTS5 failure, sessions side")

    monkeypatch.setattr(store, "_fts_obs_ids", boom)
    return store


def test_a_matching_keyword_arm_is_never_reported_as_the_arm_that_did_not_run(
    one_arm_down,
):
    """THE CONTRADICTION, on one page: the diagnosis says the keyword arm
    matched a row, and the hedge beside it said only the semantic arm ran."""
    result = aggregator_query(dsl="from:2030-01-01 fix bug", _store=one_arm_down)
    assert result["ok"] is True and result["total"] == 0, result
    notice = result["notice"]
    reason = result["low_confidence_reason"]
    assert "matched 1 row(s)" in notice, notice
    assert "the only one that ran" not in reason, reason
    assert "the only one that ran" not in notice, notice


def test_and_the_hedge_reports_the_filter_instead(one_arm_down):
    """Replacing the false sentence with silence would be its own regression:
    the page is still empty, and WHY is the whole point of the hedge."""
    result = aggregator_query(dsl="from:2030-01-01 fix bug", _store=one_arm_down)
    reason = result["low_confidence_reason"]
    assert result["low_confidence"] is True
    assert "matched 1 row(s)" in reason, reason
    assert "FILTER" in reason, reason


def test_the_dropped_arm_is_still_disclosed(one_arm_down):
    """And the failure does not get swallowed by the filter sentence — an arm
    that fell over changes what a re-run may return, so it is still said."""
    result = aggregator_query(dsl="from:2030-01-01 fix bug", _store=one_arm_down)
    reason = result["low_confidence_reason"]
    assert "UNAVAILABLE" in reason, reason
    # …without the absolute claim, which the records arm's match disproves.
    assert "has been checked against your words at all" not in reason, reason


# --- 2. more than one filter gets no single-key promise ----------------------


@pytest.fixture
def untagged(store) -> Store:
    """The corpus DOES hold "fix" and "bug" — in an untagged github record."""
    store.upsert([_rec("github:acme/api:1", "fix bug")])
    return store


def test_one_filter_keeps_the_promise(untagged):
    """With a single filter in play, "drop it and the rows come back" is a
    guarantee the query can make good on, and it stays."""
    result = aggregator_query(dsl="from:2030-01-01 fix bug", _store=untagged)
    assert result["ok"] is True and result["total"] == 0
    notice = result["notice"]
    assert "drop or widen `from:2030-01-01`" in notice, notice
    assert "the matching rows come back" in notice, notice


def test_several_filters_get_no_promise_at_all(untagged):
    """``tag:main`` and ``from:2030-01-01`` each exclude the matched row on
    their own, so dropping either one leaves the page exactly as empty. The
    old wording sent the caller to do that and blamed the corpus by
    implication when it did not work."""
    result = aggregator_query(
        dsl="source:github tag:main from:2030-01-01 fix bug", _store=untagged
    )
    assert result["ok"] is True and result["total"] == 0
    notice = result["notice"]
    for key in ("tag:main", "source:github", "from:2030-01-01"):
        assert key in notice, (key, notice)
    assert "the matching rows come back" not in notice, notice
    assert "relaxing only one may not be enough" in notice, notice
    # The actionable half survives: it is still a filter problem, not a words
    # problem, and there is still a place to start.
    assert "the fix is a FILTER" in notice, notice


# --- 3. the relaxed suffix is attributed per arm -----------------------------


@pytest.fixture
def mixed_tiers(store) -> Store:
    """The records arm answers STRICTLY; the observations arm has to relax.

    One record carries both terms, so its ladder stops at tier 1. The session
    carries them in two separate turns, so the strict AND is empty there and
    the OR tier answers. ``from:2030-01-01`` then empties both sides.
    """
    store.upsert([_rec("github:exact", "fix bug")])
    store.upsert_entities(
        [_sess("s1"), _obs("o-fix", "please fix it"), _obs("o-bug", "a bug here")]
    )
    return store


def test_a_strict_arms_rows_are_not_labelled_relaxed(mixed_tiers):
    """The request-wide marker keeps the DEEPEST tier any arm reached, which
    over-discloses safely between sub-evaluations of one arm and simply lies
    across two ontologies."""
    result = aggregator_query(dsl="from:2030-01-01 fix bug", _store=mixed_tiers)
    assert result["ok"] is True and result["total"] == 0
    notice = result["notice"]
    assert "THE WORDS ARE NOT THE PROBLEM" in notice, notice
    assert _ALL_RELAXED not in notice, notice
    # The rescue is not hidden either — half the rows really did come from the
    # OR tier, and a caller acting on them needs to know which half.
    assert "SOME of those rows" in notice, notice
    assert "`or`" in notice, notice


def test_when_every_matching_arm_relaxed_the_plain_wording_stands(store):
    """The mirror case, so the attribution cannot become a way to hide a
    rescue: both ladders relax, so every matched row really is a lead."""
    store.upsert([_rec("github:a", "fix it"), _rec("github:b", "a bug")])
    store.upsert_entities(
        [_sess("s1"), _obs("o-fix", "please fix it"), _obs("o-bug", "a bug here")]
    )
    result = aggregator_query(dsl="from:2030-01-01 fix bug", _store=store)
    assert result["ok"] is True and result["total"] == 0
    notice = result["notice"]
    assert _ALL_RELAXED in notice, notice
    assert "SOME of those rows" not in notice, notice


def test_an_exact_only_page_carries_no_relaxation_suffix(untagged):
    """And the strict case says nothing at all — absence is the exact-match
    claim, so it must never be diluted into a default."""
    result = aggregator_query(dsl="from:2030-01-01 fix bug", _store=untagged)
    notice = result["notice"]
    assert "RELAXED" not in notice, notice
