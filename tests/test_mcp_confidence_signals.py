"""``search_mode`` and the low-confidence hedge must describe THIS response.

TWO CLAIMS THAT WERE NOT CHECKED AGAINST WHAT ACTUALLY HAPPENED.

``search_mode`` echoed the mode the caller REQUESTED. On a cold vector index —
the state of this machine for the 25-30 days the embedding backfill takes — a
default query answered from FTS5 alone came back as ``search_mode: 'hybrid'``,
which is the answer the caller already knew and not the one they asked for. The
docstring promised "which arms answered" throughout.

The low-confidence hedge says "the keyword arm matched none of THESE ROWS" and
tested ``bool(fts_ids)`` — whether the uncapped keyword arm matched anything
anywhere in the corpus. The fused candidate set is a strict superset of the
keyword arm's ids, so that predicate can only ever be false when the keyword arm
found nothing at all; a page of vector-only rows served while the keyword arm
matched thousands of rows further down the recency order reported
``low_confidence: false``. One golden query's uncapped arm returns 13,650 ids.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from aggregator.core.store import Store
from aggregator.mcp import aggregator_query
from aggregator.sources.base import ObservationRow, SessionRow

_TS = datetime(2026, 7, 1, 8, 0, tzinfo=UTC)
_AXES = {"voting": 0, "governance": 0, "pigeon": 1}


class StubEmbedder:
    """No real model, ever — an earlier round named one in a test and pulled
    15 GB off a CDN. Nothing here is about embedding quality."""

    @staticmethod
    def _vec_for(text: str) -> np.ndarray:
        v = np.zeros(768, dtype=np.float32)
        lowered = (text or "").lower()
        for word, axis in _AXES.items():
            if word in lowered:
                v[axis] = 1.0
                return v
        v[2] = 1.0
        return v

    def embed_query(self, query: str) -> np.ndarray:
        return self._vec_for(query)

    def embed_documents(self, docs: list[str]) -> np.ndarray:
        return np.array([self._vec_for(d) for d in docs], dtype=np.float32)


class FlatReranker:
    """Scores everything the same: nothing on the page stands out."""

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


def _seed(store: Store, docs: list[tuple[str, str, bool]]) -> None:
    """``(obs_id, body, embed)`` — newest first, one session per observation.

    Recency order is what the page window addresses, so the ORDER of this list
    is load-bearing for the page-scoped tests below: the first entry is the
    newest and lands on page 1.
    """
    entities: list = []
    to_embed: list[tuple[str, str]] = []
    for i, (obs_id, body, embed) in enumerate(docs):
        ts = _TS - timedelta(hours=i)
        sid = f"s-{obs_id}"
        entities.append(
            SessionRow(
                session_id=sid, root_session_id=sid, parent_session_id=None,
                kind="session", agent_id=None, agent_type=None,
                spawned_by_tool_use_id=None, cwd="/x", git_branch="main",
                first_ts=ts, last_ts=ts, jsonl_path=f"/tmp/{sid}.jsonl",
            )
        )
        entities.append(
            ObservationRow(
                obs_id=obs_id, session_id=sid, root_session_id=sid,
                parent_obs_id=None, type="user", ts=ts, model=None,
                input_tokens=None, output_tokens=None, tool_name=None,
                tool_use_id=None, body=body,
            )
        )
        if embed:
            to_embed.append((obs_id, body))
    store.upsert_entities(entities)
    if not to_embed:
        return
    vecs = StubEmbedder().embed_documents([b for _, b in to_embed])
    store.upsert_vec_observations(
        [(obs_id, vecs[i]) for i, (obs_id, _) in enumerate(to_embed)]
    )
    store.mark_embedded("observations", [i for i, _ in to_embed], state="ok")


def _ids(result) -> list[str]:
    return [r["stable_id"] for r in result.get("records", [])]


# --- search_mode reports which arms produced the rows -----------------------


def test_a_cold_vector_index_reports_the_arm_that_actually_ran(store, embedder):
    """THE DEGRADED CASE, which had no coverage at all. Nothing is embedded, so
    the vector arm cannot engage and the answer is FTS5's alone — but the caller
    asked for 'hybrid' and used to be told 'hybrid' back. That is the state of
    the live cache for the whole 25-30 day backfill, not an edge case."""
    _seed(store, [("o-lex", "governance working notes", False)])
    result = aggregator_query("source:sessions governance", _store=store)
    assert result["ok"] is True, result
    assert _ids(result) == ["s-o-lex"]
    assert result["search_mode"] == "lexical"


def test_a_result_set_only_the_vector_arm_reached_says_vector(store, embedder):
    """The mirror case. The keyword arm ran and matched nothing; every row came
    from the KNN. Calling that 'hybrid' credits an arm that contributed no
    candidates, which is the same lie in the other direction."""
    _seed(store, [("o-vec", "quadratic voting rollout", True)])
    result = aggregator_query("source:sessions governance", _store=store)
    assert result["ok"] is True, result
    assert _ids(result) == ["s-o-vec"]
    assert result["search_mode"] == "vector"


def test_both_arms_contributing_is_still_hybrid(store, embedder):
    _seed(
        store,
        [
            ("o-vec", "quadratic voting rollout", True),
            ("o-lex", "governance working notes", False),
        ],
    )
    result = aggregator_query("source:sessions governance", _store=store)
    assert result["ok"] is True, result
    assert set(_ids(result)) == {"s-o-vec", "s-o-lex"}
    assert result["search_mode"] == "hybrid"


# --- the hedge is about the rows the caller is holding ----------------------


def test_a_page_of_vector_only_rows_is_low_confidence(store, embedder):
    """THE REPRO. The keyword arm matched ``o-lex`` — it is in the fused set and
    it is on page 2 — so the old ``bool(fts_ids)`` predicate said "corroborated"
    about a page whose single row the keyword arm never saw."""
    _seed(
        store,
        [
            ("o-vec", "quadratic voting rollout", True),
            ("o-lex", "governance working notes", False),
        ],
    )
    page1 = aggregator_query(
        "source:sessions governance", page_size=1, _store=store
    )
    assert page1["ok"] is True, page1
    assert _ids(page1) == ["s-o-vec"], "the repro needs the vector-only row first"
    assert page1["low_confidence"] is True
    assert "keyword" in page1["low_confidence_reason"].lower()


def test_the_next_page_of_corroborated_rows_is_not_low_confidence(
    store, embedder
):
    """The same query, the same fused set, the other page. A predicate that
    reads the corpus cannot tell these two apart; one that reads the page must."""
    _seed(
        store,
        [
            ("o-vec", "quadratic voting rollout", True),
            ("o-lex", "governance working notes", False),
        ],
    )
    page1 = aggregator_query(
        "source:sessions governance", page_size=1, _store=store
    )
    page2 = aggregator_query(
        "source:sessions governance",
        page_size=1,
        page_token=page1["next_page_token"],
        _store=store,
    )
    assert page2["ok"] is True, page2
    assert _ids(page2) == ["s-o-lex"]
    assert page2["low_confidence"] is False


def test_a_drilldown_page_of_vector_only_rows_is_low_confidence(store, embedder):
    """Same predicate, different ontology and different id space: drilldown rows
    are observations, so the keyword arm's ids compare directly. Covered
    separately because the sessions path has to project observation hits up to
    session cards and a fix that only worked in one id space would look fine.

    Seeded newest-first because drilldown orders observations ``ts ASC`` while
    session cards order ``last_ts DESC`` — page 1 here is the OLDEST row.
    """
    _seed(
        store,
        [
            ("o-lex", "governance working notes", False),
            ("o-vec", "quadratic voting rollout", True),
        ],
    )
    page1 = aggregator_query(
        "source:sessions governance", page_size=1, drilldown=True, _store=store
    )
    assert page1["ok"] is True, page1
    assert [r["obs_id"] for r in page1["records"]] == ["o-vec"]
    assert page1["low_confidence"] is True
    assert "keyword" in page1["low_confidence_reason"].lower()


# --- signal 3 reaches a page smaller than the rerank window -----------------


def test_a_ten_row_reranked_page_gets_a_standout_verdict(
    store, embedder, monkeypatch
):
    """CONFIDENCE SIGNAL 3, WHICH WAS STRUCTURALLY DEAD BELOW 20 ROWS. Nothing
    here is embedded, so the vector arm never engages and the keyword hedge
    cannot fire: the only thing that can flag this page is the reranker, and it
    scored ten documents identically."""
    _seed(store, [(f"o{i}", f"governance note {i}", False) for i in range(10)])
    monkeypatch.setattr("aggregator.mcp._get_reranker", FlatReranker)
    result = aggregator_query(
        "source:sessions governance", fields="full", rerank=True, _store=store
    )
    assert result["ok"] is True, result
    assert result["reranked_count"] == 10, "the repro needs a short page"
    assert result["low_confidence"] is True
    assert "rerank" in result["low_confidence_reason"].lower()


def test_a_ten_row_reranked_page_with_a_winner_is_not_hedged(
    store, embedder, monkeypatch
):
    """The other half. A signal that answered ``True`` for every short page
    would be as useless as one that answered ``None``."""
    _seed(store, [(f"o{i}", f"governance note {i}", False) for i in range(10)])
    monkeypatch.setattr("aggregator.mcp._get_reranker", PeakedReranker)
    result = aggregator_query(
        "source:sessions governance", fields="full", rerank=True, _store=store
    )
    assert result["ok"] is True, result
    assert result["reranked_count"] == 10
    assert result["low_confidence"] is False
