"""Criterion D: the response says when it does not trust its own answer.

THE FAILURE MODE, NAMED. Vector search is a ranking primitive and is neutral
about whether the neighbours it returns are relevant: a query about German
stock-option taxation, run against a corpus of recipes, returns five recipes.
``k``-nearest always returns ``k``. So a fused result set that the keyword arm
never corroborated is exactly the shape a no-answer query produces, and nothing
in the rows themselves says so — they look like any other page.

WHAT IS DELIBERATELY NOT DONE HERE: no threshold on the fused RRF score, ever.
RRF scores are not probabilities and have no absolute meaning across queries; a
fused 0.031 says "both arms ranked it about fifth", not "this is relevant".
Thresholding there is the one hard prohibition in the design.

AND NOTHING IS TRUNCATED. The signal is a flag on a full page, not a shorter
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
_AXES = {"voting": 0, "governance": 0, "pigeon": 1}

# WHY "PIGEON" IS NOT ORTHOGONAL TO "VOTING" ANY MORE. The vector arm has an
# absolute distance floor (``hybrid.VECTOR_FLOOR_MAX_DISTANCE`` = 1.0, i.e.
# cosine 0.5), so a document on a fully orthogonal axis sits at L2 1.414 and is
# dropped before fusion ever sees it. A "vector-only result set" fixture built
# that way would be testing the floor and reporting it as a test of the hedge:
# the page would come back empty and ``low_confidence`` would be true for the
# wrong reason. Cosine 0.6 (L2 0.894) is inside the floor, so the neighbour
# survives, no keyword matches it, and the hedge is what decides the outcome.
_NEAR_COS = 0.6


class StubEmbedder:
    """No real model, ever — an earlier round named one in a test and pulled
    15 GB off a CDN."""

    @staticmethod
    def _vec_for(text: str) -> np.ndarray:
        v = np.zeros(768, dtype=np.float32)
        lowered = (text or "").lower()
        for word, axis in _AXES.items():
            if word in lowered:
                if axis == 0:
                    v[0] = 1.0
                else:
                    v[0] = _NEAR_COS
                    v[axis] = float(np.sqrt(1.0 - _NEAR_COS**2))
                return v
        v[2] = 1.0
        return v

    def embed_query(self, query: str) -> np.ndarray:
        return self._vec_for(query)

    def embed_documents(self, docs: list[str]) -> np.ndarray:
        return np.array([self._vec_for(d) for d in docs], dtype=np.float32)


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


@pytest.fixture
def embedder(monkeypatch):
    stub = StubEmbedder()
    monkeypatch.setattr("aggregator.mcp._get_embedder", lambda: stub)
    return stub


def _seed(store: Store, docs: list[tuple[str, str]], embed: bool = True) -> None:
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
    if not embed:
        return
    vecs = StubEmbedder().embed_documents([b for _, b in docs])
    store.upsert_vec_observations(
        [(obs_id, vecs[i]) for i, (obs_id, _) in enumerate(docs)]
    )
    store.mark_embedded("observations", [i for i, _ in docs], state="ok")


# --- the flag itself --------------------------------------------------------


def test_a_corroborated_answer_is_not_low_confidence(store, embedder):
    _seed(store, [("o1", "quadratic voting is a governance mechanism")])
    result = aggregator_query("source:sessions governance", _store=store)
    assert result["ok"] is True, result
    assert result["low_confidence"] is False
    assert "low_confidence_reason" not in result


def test_the_flag_is_present_on_every_free_text_answer(store, embedder):
    """``False`` is asserted explicitly rather than left absent. An absent key
    is unfalsifiable from the caller's side — it reads identically to a server
    that has not been upgraded — and this is a claim about the answer, so it
    has to be stated even when the claim is 'fine'."""
    _seed(store, [("o1", "quadratic voting is a governance mechanism")])
    result = aggregator_query("source:sessions governance", _store=store)
    assert "low_confidence" in result


def test_a_filter_only_query_carries_no_confidence_claim(store, embedder):
    """No retrieval ran, so there is nothing to be confident or unconfident
    about. Reporting ``False`` here would assert a judgement nobody made."""
    _seed(store, [("o1", "quadratic voting")])
    result = aggregator_query("source:sessions", _store=store)
    assert result["ok"] is True, result
    assert "low_confidence" not in result


def test_a_query_that_finds_nothing_says_so_explicitly(store, embedder):
    _seed(store, [("o1", "quadratic voting")], embed=False)
    result = aggregator_query(
        "source:sessions beef wellington", _store=store
    )
    assert result["ok"] is True, result
    assert result["total"] == 0
    assert result["low_confidence"] is True
    assert result["low_confidence_reason"]


def test_a_vector_only_result_set_is_low_confidence(store, embedder):
    """THE REPRO for the recipe case. ``pigeon`` matches no word in the corpus,
    but the KNN returns its nearest neighbour anyway — and it is close enough to
    clear the distance floor, so the floor does not save this one. The hedge is
    what tells the caller that nothing about the row matched their words."""
    _seed(store, [("o1", "quadratic voting is a governance mechanism")])
    result = aggregator_query("source:sessions pigeon", _store=store)
    assert result["ok"] is True, result
    assert result["total"] > 0, "the repro needs the vector arm to answer"
    assert result["low_confidence"] is True
    assert "keyword" in result["low_confidence_reason"].lower()


def test_the_low_confidence_answer_is_not_truncated(store, embedder):
    """The rows are all still there. An agent given a shorter page cannot tell
    a confident short answer from a hedged long one."""
    _seed(store, [("o1", "quadratic voting"), ("o2", "governance notes")])
    hedged = aggregator_query("source:sessions pigeon", _store=store)
    assert hedged["low_confidence"] is True
    assert len(hedged["records"]) == hedged["total"] > 0


def test_the_reason_also_reaches_the_prose_notice(store, embedder):
    """A caller that only reads ``notice`` — every human, and any model that
    was not told about the flag — has to see it too."""
    _seed(store, [("o1", "quadratic voting is a governance mechanism")])
    result = aggregator_query("source:sessions pigeon", _store=store)
    assert "LOW CONFIDENCE" in result["notice"]


# --- the reranker threshold, when the reranker ran --------------------------


def test_a_flat_reranker_page_is_low_confidence(store, embedder, monkeypatch):
    """The report's preferred abstention signal, applied where it applies. The
    cross-encoder is OFF BY DEFAULT here (~13.7 s per pair on this hardware),
    so this can never be the only abstention signal — but when it did run, a
    page where nothing scored above the rest is a page with no answer on it."""
    _seed(store, [(f"o{i}", f"governance note {i}") for i in range(25)])
    monkeypatch.setattr("aggregator.mcp._get_reranker", FlatReranker)
    result = aggregator_query(
        "source:sessions governance", fields="full", rerank=True, _store=store
    )
    assert result["ok"] is True, result
    assert result["rerank_applied"] is True
    assert result["low_confidence"] is True
    assert "rerank" in result["low_confidence_reason"].lower()


def test_a_peaked_reranker_page_is_not_low_confidence(
    store, embedder, monkeypatch
):
    _seed(store, [(f"o{i}", f"governance note {i}") for i in range(25)])
    monkeypatch.setattr("aggregator.mcp._get_reranker", PeakedReranker)
    result = aggregator_query(
        "source:sessions governance", fields="full", rerank=True, _store=store
    )
    assert result["ok"] is True, result
    assert result["rerank_applied"] is True
    assert result["low_confidence"] is False


def test_a_failed_reranker_makes_no_confidence_claim_of_its_own(
    store, embedder, monkeypatch
):
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
