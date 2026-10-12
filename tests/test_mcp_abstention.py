"""The response says when it does not trust its own answer.

Two signals: nothing matched (or a filter removed every match), and — when
the caller asked for it — the reranker scored the page and found nothing that
stands out.

NOTHING IS TRUNCATED. The signal is a flag on a full page, not a shorter
page: an agent that receives fewer rows cannot tell a confident short answer
from a hedged long one, and silently dropping rows is how a recall tool starts
hiding documents the user knows exist.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from aggregator.core.store import Store
from aggregator.mcp import aggregator_query
from aggregator.sources.base import ObservationRow, SessionRow

_TS = datetime(2026, 7, 1, 8, 0, tzinfo=UTC)


class FlatReranker:
    """Scores everything the same: the cross-encoder found nothing that stands
    out, which is what a page with no relevant document looks like."""

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        return np.full(len(docs), 0.5, dtype=np.float32)


class PeakedReranker:
    """One clear winner and a flat tail — a page that does contain an answer."""

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        out = np.full(len(docs), 0.01, dtype=np.float32)
        if len(out):
            out[-1] = 0.99
        return out


@pytest.fixture
def store(tmp_path):
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    return s


def _seed(store: Store, docs: list[tuple[str, str]]) -> None:
    entities: list = []
    for obs_id, body in docs:
        sid = f"s-{obs_id}"
        entities.append(
            SessionRow(
                session_id=sid, root_session_id=sid, parent_session_id=None,
                kind="session", agent_id=None, agent_type=None,
                spawned_by_tool_use_id=None, cwd="/x", git_branch="main",
                first_ts=_TS, last_ts=_TS, jsonl_path=f"/tmp/{sid}.jsonl",
            )
        )
        entities.append(
            ObservationRow(
                obs_id=obs_id, session_id=sid, root_session_id=sid,
                parent_obs_id=None, type="user", ts=_TS, model=None,
                input_tokens=None, output_tokens=None, tool_name=None,
                tool_use_id=None, body=body,
            )
        )
    store.upsert_entities(entities)


# --- the flag itself --------------------------------------------------------


def test_a_matching_answer_is_not_low_confidence(store):
    _seed(store, [("o1", "quadratic voting is a governance mechanism")])
    result = aggregator_query("source:sessions governance", _store=store)
    assert result["ok"] is True, result
    assert result["low_confidence"] is False
    assert "low_confidence_reason" not in result


def test_the_flag_is_present_on_every_free_text_answer(store):
    """``False`` is asserted explicitly rather than left absent. An absent key
    is unfalsifiable from the caller's side — it reads identically to a server
    that has not been upgraded — and this is a claim about the answer, so it
    has to be stated even when the claim is 'fine'."""
    _seed(store, [("o1", "quadratic voting is a governance mechanism")])
    result = aggregator_query("source:sessions governance", _store=store)
    assert "low_confidence" in result
    assert result["search_mode"] == "lexical"


def test_a_filter_only_query_carries_no_confidence_claim(store):
    """No retrieval ran, so there is nothing to be confident or unconfident
    about. Reporting ``False`` here would assert a judgement nobody made."""
    _seed(store, [("o1", "quadratic voting")])
    result = aggregator_query("source:sessions", _store=store)
    assert result["ok"] is True, result
    assert "low_confidence" not in result
    assert "search_mode" not in result


def test_a_query_that_finds_nothing_says_so_explicitly(store):
    _seed(store, [("o1", "quadratic voting")])
    result = aggregator_query("source:sessions beef wellington", _store=store)
    assert result["ok"] is True, result
    assert result["total"] == 0
    assert result["low_confidence"] is True
    assert result["low_confidence_reason"]
    # A caller that only reads ``notice`` — every human, and any model that
    # was not told about the flag — has to see it too.
    assert "LOW CONFIDENCE" in result["notice"]


# --- the reranker threshold, when the reranker ran --------------------------


def test_a_flat_reranker_page_is_low_confidence(store, monkeypatch):
    """A page where nothing scored above the rest is a page with no answer on
    it. The rows are all still there: the flag hedges, it does not truncate."""
    _seed(store, [(f"o{i}", f"governance note {i}") for i in range(25)])
    monkeypatch.setattr("aggregator.mcp._get_reranker", FlatReranker)
    result = aggregator_query(
        "source:sessions governance", fields="full", rerank=True, _store=store
    )
    assert result["ok"] is True, result
    assert result["rerank_applied"] is True
    assert result["low_confidence"] is True
    assert "rerank" in result["low_confidence_reason"].lower()
    assert len(result["records"]) == result["total"] == 25


def test_a_peaked_reranker_page_is_not_low_confidence(store, monkeypatch):
    _seed(store, [(f"o{i}", f"governance note {i}") for i in range(25)])
    monkeypatch.setattr("aggregator.mcp._get_reranker", PeakedReranker)
    result = aggregator_query(
        "source:sessions governance", fields="full", rerank=True, _store=store
    )
    assert result["ok"] is True, result
    assert result["rerank_applied"] is True
    assert result["low_confidence"] is False


def test_a_failed_reranker_makes_no_confidence_claim_of_its_own(store, monkeypatch):
    """It produced no scores, so it has no opinion. Treating "the model died"
    as "nothing was relevant" would be an answer invented out of a failure."""

    class Exploding:
        def score(self, query, docs):
            raise RuntimeError("cross-encoder failed to load")

    _seed(store, [("o1", "quadratic voting is a governance mechanism")])
    monkeypatch.setattr("aggregator.mcp._get_reranker", Exploding)
    result = aggregator_query(
        "source:sessions governance", fields="full", rerank=True, _store=store
    )
    assert result["ok"] is True, result
    assert result["rerank_applied"] is False
    assert result["low_confidence"] is False
