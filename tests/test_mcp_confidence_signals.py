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

import sqlite3
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from aggregator.core.store import Store
from aggregator.mcp import aggregator_query
from aggregator.sources.base import ObservationRow, Record, SessionRow

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


# --- a keyword arm that fell over must not be reported as one that missed ---


def test_a_dropped_keyword_arm_says_so_instead_of_blaming_the_rows(
    store, embedder, monkeypatch
):
    """THE SENTENCE IS THE CONTRACT, so the test reads the sentence.

    When ``_fts_obs_ids`` raises, the query still answers from the vector arm —
    a broken index must cost the arm and never the answer. What it must not do
    is report that as "the keyword arm matched none of these rows", which is a
    claim about the documents made out of a failure of the index, and leaves an
    agent unable to tell an unmatched corpus from an FTS5 table that fell over.
    """
    _seed(store, [("o-vec", "quadratic voting rollout", True)])

    def boom(_text):
        raise sqlite3.OperationalError("simulated FTS5 failure")

    monkeypatch.setattr(store, "_fts_obs_ids", boom)
    result = aggregator_query("source:sessions governance", _store=store)
    assert result["ok"] is True, result
    assert _ids(result) == ["s-o-vec"], "the vector arm still has to answer"
    assert result["low_confidence"] is True
    reason = result["low_confidence_reason"].lower()
    assert "unavailable" in reason, reason
    assert "matched none of these rows" not in reason, reason
    assert "LOW CONFIDENCE" in result["notice"]


def test_a_keyword_arm_that_ran_and_missed_still_blames_the_rows(
    store, embedder
):
    """The other half, and the reason the first one needs a distinct state: an
    arm that DID run and found nothing must keep the original sentence. A fix
    that emitted "unavailable" for both would trade one lie for another."""
    _seed(store, [("o-vec", "quadratic voting rollout", True)])
    result = aggregator_query("source:sessions governance", _store=store)
    assert result["ok"] is True, result
    assert _ids(result) == ["s-o-vec"]
    assert result["low_confidence"] is True
    reason = result["low_confidence_reason"].lower()
    assert "matched none of these rows" in reason, reason
    assert "unavailable" not in reason, reason


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


# --- ONE DEFINITION OF "WHICH ARM ANSWERED" ---------------------------------
#
# The wave-5 review found four findings that are one defect: the confidence
# surface grew three different scopes for that question and never reconciled
# them. The reconciliation this file pins:
#
#   search_mode        which arms put rows into THIS RESULT SET (the fused
#                      scope). Result-set scope, NOT page scope, so that it is
#                      stable while a caller paginates.
#   low_confidence     the keyword hedge, about THIS PAGE. Page scope, because
#                      a corpus-wide predicate can only be false when the arm
#                      matched nothing anywhere.
#   the dropped-arm    about the ARM, per ontology consulted. "It raised" is
#   sentence           not "it ran and missed", and one ontology's failure does
#                      not speak for another's success.
#
# The two scopes differ on purpose. What was wrong was that nothing said so and
# the three code paths disagreed about the first one.


def test_an_engaged_vector_arm_that_returned_no_rows_is_not_credited(
    store, embedder
):
    """MEDIUM-3. ``vector_contributed`` was the arm ENGAGING, not the arm
    producing rows, on the records and sessions paths — while ``88d7c86`` had
    added exactly that row-gate to the union path and left the other two alone.
    One query shape, three paths, three answers.

    THE REPRO IS AN EMPTY RESULT SET WITH THE ARM ENGAGED. A date filter the
    embedded row cannot satisfy leaves the vector arm engaged (there is free
    text and there are embedded rows), its candidate in the fused scope, and
    nothing at all in the response. The sessions path then reported
    ``search_mode: 'vector'`` for a page with no rows on it — an arm named as
    having produced rows that do not exist — while the union path, on the same
    corpus and the same text, said ``lexical``.
    """
    _seed(store, [("o-vec", "quadratic voting rollout", True)])
    scoped = aggregator_query(
        "source:sessions governance from:2027-01-01", _store=store
    )
    union = aggregator_query("governance from:2027-01-01", _store=store)
    assert scoped["ok"] is True and union["ok"] is True, (scoped, union)
    assert scoped["total"] == 0 and union["total"] == 0, (scoped, union)
    assert scoped["search_mode"] == union["search_mode"], (
        "the same query text over the same corpus must not get a different "
        f"answer from two paths: sessions={scoped['search_mode']!r} "
        f"union={union['search_mode']!r}"
    )
    assert scoped["search_mode"] == "lexical", (
        "no arm put a row in this result set, and the degenerate case is "
        "declared to be 'lexical' — naming the vector arm claims rows that "
        "are not there"
    )


def test_search_mode_is_stable_across_pages_and_the_hedge_is_not(
    store, embedder
):
    """MEDIUM-4. The docstring header said ``search_mode`` "NAMES THE ARMS THAT
    PRODUCED THESE ROWS" and four lines later said it is "derived from which
    arms CONTRIBUTED CANDIDATES" — the page and the result set, in one
    paragraph, for one field. One response asserted both "hybrid produced these
    rows" and "the keyword arm matched none of these rows".

    THE DECISION, PINNED HERE RATHER THAN ARGUED: ``search_mode`` is
    result-set-scoped and the hedge is page-scoped, and they are allowed to
    disagree because they answer different questions. This test is what makes
    that a property instead of a preference — under a page-scoped
    ``search_mode`` page 1 would say ``vector`` and page 2 ``lexical`` for one
    query, and an agent paginating would watch the retrieval mode flicker.
    """
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
    assert _ids(page1) == ["s-o-vec"] and _ids(page2) == ["s-o-lex"], (
        page1,
        page2,
    )
    assert page1["search_mode"] == page2["search_mode"] == "hybrid", (
        "both arms put rows in this result set on both pages; the field "
        "describes the result set and must not move with the window"
    )
    assert page1["low_confidence"] is True
    assert page2["low_confidence"] is False, (
        "the hedge is the page-scoped one and MUST move with the window"
    )


def test_a_failed_card_projection_does_not_erase_a_keyword_arm_that_matched(
    store, embedder, monkeypatch
):
    """MEDIUM-1. ``_lexical_session_ids`` returned ``LEXICAL_ARM_UNAVAILABLE``
    when the CARD PROJECTION raised, even though ``lexical_ids`` had come back
    healthy and non-empty a few milliseconds earlier.

    The arm ran. It matched this very row. What failed is the second statement
    that maps its observation hits up to session-card ids — so the true answer
    is "we cannot tell whether the arm matched THESE CARDS", and the response
    used to say "nothing here has been checked against your words at all",
    which is a claim about the whole corpus made out of a projection failure.
    """
    _seed(store, [("o-lex", "governance working notes", True)])

    def boom(_text, _obs_type=None, _provenance=None):
        raise sqlite3.OperationalError("simulated FTS5 failure in the projection")

    monkeypatch.setattr(store, "_fts_hit_scope", boom)
    result = aggregator_query("source:sessions governance", _store=store)
    assert result["ok"] is True, result
    assert _ids(result) == ["s-o-lex"], "the page is not in question, only the claim"
    reason = (result.get("low_confidence_reason") or "").lower()
    assert "has been checked against your words at all" not in reason, (
        "the keyword arm ran and matched this row; only the projection onto "
        f"card ids failed. reason={reason!r}"
    )
    assert "matched none of these rows" not in reason, (
        f"nor did it miss them — that is the other lie. reason={reason!r}"
    )


def test_one_ontologys_dropped_arm_does_not_speak_for_the_other(
    store, embedder, monkeypatch
):
    """MEDIUM-2. ``_lexical_arm_failed`` used ``any`` across every ontology in
    the response, so in union mode one side falling over made the whole response
    say "nothing here has been checked against your words at all" — while the
    other side had run, matched, and put the corroborated rows on the page.

    Here the records side's keyword arm raises and the sessions side's answers.
    The sentence has to be true of the rows actually returned.
    """
    _seed(store, [("o-vec", "quadratic voting rollout", True)])
    store.upsert(
        [
            Record(
                stable_id="github:o/r:1",
                source="github",
                subject="governance rollout plan",
                body="governance rollout plan and the vote weighting change",
                created_at=_TS,
                updated_at=_TS,
            )
        ]
    )

    def boom(_text):
        raise sqlite3.OperationalError("simulated FTS5 failure on the sessions side")

    monkeypatch.setattr(store, "_fts_obs_ids", boom)
    result = aggregator_query("governance", _store=store)
    assert result["ok"] is True, result
    assert result["mode"] == "union", result
    assert "github:o/r:1" in _ids(result), (
        "the records side answers through the FTS5-only route, so its row on "
        f"this page IS a keyword match by construction: {_ids(result)}"
    )
    reason = (result.get("low_confidence_reason") or "").lower()
    assert "has been checked against your words at all" not in reason, (
        "one ontology's arm fell over; the other ran, matched, and put a "
        f"corroborated row on this very page. reason={reason!r}"
    )
