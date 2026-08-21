"""Criterion H: the two arms are separately runnable, in one command.

CORRECTNESS TOOLING, NOT CONVENIENCE. The reference design's "weakest link"
finding is that fusing a strong arm with a weak one can score WORSE than the
strong arm alone — TOUCHE measured lexical-alone at 0.650 nDCG@10 against 0.604
fused — and that neither raising per-arm depth nor bolting on a heavyweight
reranker repairs it. Rank fusion is specifically susceptible because a weak
arm's rank-1 document collects the full ``1/(k+1)`` credit regardless of
whether it is relevant. Without arms that can be run separately that condition
is undiagnosable: the fused answer looks fine, and nothing in it says which arm
produced the damage.

``search_mode='vector'`` REFUSES RATHER THAN DEGRADES, and that is the whole
point of it. Everywhere else in this server the vector arm falls back to FTS5,
which is right for a user who would rather have keyword results than an error.
It is wrong for a diagnostic: a vector-mode run that silently answered from the
keyword arm would file the keyword arm's numbers under "vector", and that is
the empty-result-looks-like-success failure this project bans by name.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import numpy as np
import pytest

from aggregator.core import store as store_mod
from aggregator.core.store import Store
from aggregator.mcp import aggregator_query
from aggregator.sources.base import ObservationRow, SessionRow

_TS = datetime(2026, 7, 1, 8, 0, tzinfo=UTC)

# "voting" and "governance" share an axis, so a governance query is at distance
# 0 from a voting document while matching none of its words. That is what makes
# the two arms separable by a test: one row only FTS5 can reach, one row only
# the vector arm can reach.
_AXES = {"voting": 0, "governance": 0, "pigeon": 1}


class StubEmbedder:
    """No real model, ever. An earlier round named one in a test and pulled
    15 GB off a CDN; nothing here is about embedding quality."""

    def __init__(self):
        self.query_calls = 0

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
        self.query_calls += 1
        return self._vec_for(query)

    def embed_documents(self, docs: list[str]) -> np.ndarray:
        return np.array([self._vec_for(d) for d in docs], dtype=np.float32)


class ExplodingEmbedder:
    def embed_query(self, query):
        raise RuntimeError("model weights are corrupt")


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


@pytest.fixture
def no_vec_store(tmp_path, monkeypatch):
    def _boom(conn):
        raise sqlite3.OperationalError("simulated sqlite-vec ABI mismatch")

    monkeypatch.setattr(store_mod, "_load_sqlite_vec", _boom)
    monkeypatch.setattr(store_mod, "_VEC_LOAD_WARNED", False)
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    assert s.vector_available is False
    return s


def _seed(store: Store, embed: bool = True) -> None:
    """One row each arm can reach and one row only the other can.

    * ``o-lex`` says "governance" — the FTS5 arm matches it on the word and the
      vector arm cannot, because its vector is not embedded.
    * ``o-vec`` says "quadratic voting" — no query word matches it, and its
      vector sits on the same axis a "governance" query embeds to.
    """
    entities: list = []
    for obs_id, body in (
        ("o-lex", "governance working notes"),
        ("o-vec", "quadratic voting rollout"),
    ):
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
    vec = StubEmbedder().embed_documents(["quadratic voting rollout"])
    store.upsert_vec_observations([("o-vec", vec[0])])
    store.mark_embedded("observations", ["o-vec"], state="ok")


def _ids(result) -> set[str]:
    return {r["stable_id"] for r in result.get("records", [])}


# --- the three modes --------------------------------------------------------


def test_hybrid_is_the_default_and_returns_both_arms(store, embedder):
    _seed(store)
    result = aggregator_query("source:sessions governance", _store=store)
    assert result["ok"] is True, result
    assert _ids(result) == {"s-o-lex", "s-o-vec"}
    assert result["search_mode"] == "hybrid"


def test_lexical_mode_returns_only_what_the_keyword_arm_reaches(store, embedder):
    _seed(store)
    result = aggregator_query(
        "source:sessions governance", search_mode="lexical", _store=store
    )
    assert result["ok"] is True, result
    assert _ids(result) == {"s-o-lex"}
    assert result["search_mode"] == "lexical"


def test_lexical_mode_never_constructs_the_embedder(store, embedder):
    """A weak-arm diagnosis must not pay for the arm it is excluding, and an
    embedder call here would also mean the mode is advisory rather than real."""
    _seed(store)
    aggregator_query(
        "source:sessions governance", search_mode="lexical", _store=store
    )
    assert embedder.query_calls == 0


def test_lexical_mode_never_runs_the_knn(store, embedder, monkeypatch):
    _seed(store)
    calls: list[int] = []
    original = store._vec_obs_scored
    monkeypatch.setattr(
        store,
        "_vec_obs_scored",
        lambda e, k: (calls.append(k), original(e, k))[1],
    )
    aggregator_query(
        "source:sessions governance", search_mode="lexical", _store=store
    )
    assert calls == []


def test_vector_mode_returns_only_what_the_vector_arm_reaches(store, embedder):
    """THE POINT OF THE MODE: ``s-o-lex`` matches the query word and is still
    absent, because the keyword arm is not running."""
    _seed(store)
    result = aggregator_query(
        "source:sessions governance", search_mode="vector", _store=store
    )
    assert result["ok"] is True, result
    assert _ids(result) == {"s-o-vec"}
    assert result["search_mode"] == "vector"


def test_vector_mode_still_honours_the_dsl_filters(store, embedder):
    """Excluding an arm must not become a way past a filter the caller set."""
    _seed(store)
    result = aggregator_query(
        "source:sessions governance from:2027-01-01",
        search_mode="vector",
        _store=store,
    )
    assert result["ok"] is True, result
    assert _ids(result) == set()


# --- vector mode refuses instead of degrading -------------------------------


def test_vector_mode_refuses_when_nothing_is_embedded(store, embedder):
    """The default path would answer this from FTS5 and say nothing. Here that
    would report the keyword arm's rows as the vector arm's.

    This is the state the live cache is in for most of its sources: the
    backfill is a measured 25-30 days of CPU, so "the corpus is indexed but not
    embedded" is the NORMAL case for weeks, not an edge case.
    """
    _seed(store, embed=False)
    result = aggregator_query(
        "source:sessions governance", search_mode="vector", _store=store
    )
    assert result["ok"] is False, result
    assert "vector" in result["reason"].lower()
    assert result["remediation"]


def test_vector_mode_refuses_when_the_extension_is_missing(no_vec_store, embedder):
    result = aggregator_query(
        "source:sessions governance", search_mode="vector", _store=no_vec_store
    )
    assert result["ok"] is False, result
    assert result["remediation"]


def test_vector_mode_refuses_when_the_embedder_cannot_answer(
    store, monkeypatch
):
    _seed(store)
    monkeypatch.setattr("aggregator.mcp._get_embedder", ExplodingEmbedder)
    result = aggregator_query(
        "source:sessions governance", search_mode="vector", _store=store
    )
    assert result["ok"] is False, result
    assert result["remediation"]


def test_vector_mode_refuses_a_query_with_no_free_text(store, embedder):
    """There is nothing to embed, so there is no vector arm to run. Answering
    from the filters alone and calling it 'vector' would be a lie about which
    arm produced the rows."""
    _seed(store)
    result = aggregator_query(
        "source:sessions", search_mode="vector", _store=store
    )
    assert result["ok"] is False, result
    assert result["remediation"]


def test_lexical_mode_on_a_filter_only_query_is_fine(store, embedder):
    """Symmetric case, opposite answer: a filter-only query IS the lexical
    path, so there is nothing to refuse."""
    _seed(store)
    result = aggregator_query(
        "source:sessions", search_mode="lexical", _store=store
    )
    assert result["ok"] is True, result


# --- the surface ------------------------------------------------------------


def test_an_unknown_search_mode_is_refused_with_the_valid_ones_named(store):
    result = aggregator_query(
        "governance", search_mode="semantic", _store=store
    )
    assert result["ok"] is False, result
    assert "semantic" in result["reason"]
    for mode in ("hybrid", "lexical", "vector"):
        assert mode in result["remediation"]


def test_a_filter_only_query_reports_no_search_mode(store, embedder):
    """The key answers "which arms ran", and for a query with no free text the
    honest answer is "neither" rather than the default the caller passed."""
    _seed(store)
    result = aggregator_query("source:sessions", _store=store)
    assert result["ok"] is True, result
    assert "search_mode" not in result


def test_a_page_token_is_refused_when_the_search_mode_changes(store, embedder):
    """The mode decides MEMBERSHIP, so an offset carried across a change to it
    indexes a set that was never cut. Same hazard the dsl fingerprint closes,
    one argument over."""
    _seed(store)
    page1 = aggregator_query(
        "source:sessions governance", page_size=1, _store=store
    )
    token = page1["next_page_token"]
    reused = aggregator_query(
        "source:sessions governance",
        page_size=1,
        page_token=token,
        search_mode="lexical",
        _store=store,
    )
    assert reused["ok"] is False, reused
    assert "page_token" in reused["reason"], reused


def test_a_page_token_still_works_within_one_search_mode(store, embedder):
    _seed(store)
    page1 = aggregator_query(
        "source:sessions governance",
        page_size=1,
        search_mode="hybrid",
        _store=store,
    )
    page2 = aggregator_query(
        "source:sessions governance",
        page_size=1,
        page_token=page1["next_page_token"],
        search_mode="hybrid",
        _store=store,
    )
    assert page2["ok"] is True, page2
    assert _ids(page1) | _ids(page2) == {"s-o-lex", "s-o-vec"}


def test_the_mcp_tool_adapter_exposes_the_parameter(store):
    """The tool the model calls is ``_tool_aggregator_query``, not the Python
    function under it — a parameter added to only one of them is invisible to
    every caller that matters."""
    import inspect

    from aggregator.mcp import _tool_aggregator_query

    assert "search_mode" in inspect.signature(_tool_aggregator_query).parameters
