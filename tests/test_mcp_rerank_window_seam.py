"""``rerank_applied: true`` did not say how much of the page was ranked.

``_maybe_rerank`` reorders at most ``_RERANK_WINDOW`` = 20 items — a latency
budget, since a cross-encoder pass costs 273 s median on this CPU — while a
page holds up to 200 (summary) or 40 (full). So the response could carry 40
hits of which 20 were scored, and the only thing it said about that was
``rerank_applied: true``.

The reordered rows are byte-for-byte indistinguishable from the untouched
ones. A programmatic caller that asked for relevance ordering and got a page
that is relevance-ordered down to row 20 and recency-ordered below it has been
told something false by omission, and cannot detect it — this is the same
defect round 3 found in the CLI's printed note (fixed in b22eee9, which now
marks the seam in the human-readable output), on the surface that actually
matters: MCP is how this tool is consumed.

Round 3's CLI fix ended at the file boundary and flagged the remainder as a
contract for this file. This is that contract: ``reranked_count``.

Two things it must be, beyond existing:

* HONEST WHEN THE PAGE IS SHORTER THAN THE WINDOW. A 5-hit page under a 20-hit
  window is ranked all the way down. Reporting the window there would invent
  15 rows that do not exist and understate the ranking that does.
* INCAPABLE OF DISAGREEING WITH ``rerank_applied``. Two independently computed
  facts about one event drift; the count IS the fact, and the boolean is a
  view of it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

import aggregator.mcp as mcp
from aggregator.core.store import Store
from aggregator.mcp import _RERANK_WINDOW
from aggregator.sources.base import Record

_TS = datetime(2026, 7, 1, tzinfo=UTC)


class SpyReranker:
    """Never a real cross-encoder — naming one in a test pulled 15 GB once."""

    def __init__(self):
        self.scored: list[int] = []

    def score(self, query, docs):
        self.scored.append(len(docs))
        return list(range(len(docs)))  # ascending => the sort reverses them


@pytest.fixture
def reranker(monkeypatch):
    spy = SpyReranker()
    monkeypatch.setattr(mcp, "_get_reranker", lambda: spy)
    return spy


def _store_with(tmp_path, n: int) -> Store:
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert(
        [
            Record(
                stable_id=f"github:acme/api:{i}",
                source="github",
                subject=f"pr {i} about quadratic voting",
                body=f"body of pull request {i}: whether this row answers it",
                tags=["pr"],
                created_at=_TS + timedelta(minutes=i),
                updated_at=_TS + timedelta(minutes=i),
            )
            for i in range(n)
        ]
    )
    return s


def test_a_caller_can_tell_where_the_ranking_stops(tmp_path, reranker):
    """THE REPRO. A page three times the window, and one boolean about it."""
    store = _store_with(tmp_path, 3 * _RERANK_WINDOW)

    result = mcp.aggregator_query(
        "source:github voting",
        fields="full",
        page_size=3 * _RERANK_WINDOW,
        rerank=True,
        _store=store,
    )

    assert result["rerank_applied"] is True
    assert result.get("reranked_count") == _RERANK_WINDOW, (
        f"the response reports a reranked page of {len(result['records'])} "
        f"hits, of which only {reranker.scored} were actually scored, and "
        f"offers no field that says so: {sorted(result)}"
    )


def test_the_seam_is_computable_without_knowing_the_window(tmp_path, reranker):
    """``_RERANK_WINDOW`` is a private latency budget that may change.

    A caller splitting the page must not have to import it, hardcode it, or
    infer it from the docs — the response has to carry the number.
    """
    store = _store_with(tmp_path, 3 * _RERANK_WINDOW)
    result = mcp.aggregator_query(
        "source:github voting",
        fields="full",
        page_size=3 * _RERANK_WINDOW,
        rerank=True,
        _store=store,
    )

    ranked = result["records"][: result["reranked_count"]]
    unranked = result["records"][result["reranked_count"] :]

    assert len(ranked) == reranker.scored[0], (
        "the field does not agree with what the cross-encoder was handed"
    )
    assert unranked, "this page must have a tail for the split to mean anything"


def test_a_page_shorter_than_the_window_is_ranked_all_the_way_down(
    tmp_path, reranker
):
    """The honesty case in the other direction: do not report 20 for 5 rows."""
    store = _store_with(tmp_path, 5)

    result = mcp.aggregator_query(
        "source:github voting", fields="full", rerank=True, _store=store
    )

    assert result["reranked_count"] == 5, result
    assert result["reranked_count"] == len(result["records"])


def test_a_page_exactly_the_window_reports_no_unranked_tail(tmp_path, reranker):
    """The boundary, where an off-by-one would invent a seam that is not there."""
    store = _store_with(tmp_path, _RERANK_WINDOW)
    result = mcp.aggregator_query(
        "source:github voting", fields="full", rerank=True, _store=store
    )
    assert result["reranked_count"] == len(result["records"]) == _RERANK_WINDOW


class ExplodingReranker:
    def score(self, query, docs):
        raise RuntimeError("cross-encoder died mid-page")


@pytest.mark.parametrize(
    "scenario",
    ["success", "model_failure", "no_free_text", "empty_page"],
)
def test_the_count_and_the_flag_can_never_disagree(tmp_path, monkeypatch, scenario):
    """One fact, two views — checked across every branch that produces them.

    ``rerank_applied`` is derived FROM the count rather than tracked beside
    it, so this cannot be made to fail by adding a branch; that is the point
    of asserting it as an identity rather than per-case constants.
    """
    store = _store_with(tmp_path, 3 * _RERANK_WINDOW)
    dsl = "source:github voting"
    if scenario == "model_failure":
        monkeypatch.setattr(mcp, "_get_reranker", ExplodingReranker)
    else:
        monkeypatch.setattr(mcp, "_get_reranker", SpyReranker)
    if scenario == "no_free_text":
        dsl = "source:github"
    if scenario == "empty_page":
        dsl = "source:github nothingmatchesthisterm"

    result = mcp.aggregator_query(
        dsl,
        fields="full",
        page_size=3 * _RERANK_WINDOW,
        rerank=True,
        _store=store,
    )

    assert result["ok"] is True, result
    assert result["rerank_applied"] == (result["reranked_count"] > 0), (
        f"{scenario}: rerank_applied={result['rerank_applied']} alongside "
        f"reranked_count={result['reranked_count']}"
    )
    assert result["reranked_count"] <= len(result["records"]), (
        f"{scenario}: claims more ranked rows than the page holds"
    )


def test_a_plain_query_carries_no_rerank_keys(tmp_path, reranker):
    """Parity with ``rerank_applied``: an answer to a question only a rerank
    caller asked. A count of 0 on every ordinary response would read as "the
    reranker ran and ranked nothing"."""
    store = _store_with(tmp_path, 5)
    result = mcp.aggregator_query(
        "source:github voting", fields="full", _store=store
    )
    assert "reranked_count" not in result
    assert "rerank_applied" not in result


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dsl": "source:github voting"},
        {"dsl": "voting"},
    ],
    ids=["records", "union"],
)
def test_every_route_reports_the_count(tmp_path, reranker, kwargs):
    """Four paths call ``_maybe_rerank``; the field is not one path's habit."""
    store = _store_with(tmp_path, 5)
    result = mcp.aggregator_query(
        fields="full", rerank=True, _store=store, **kwargs
    )
    assert result["reranked_count"] == len(result["records"]), kwargs


def test_the_tool_description_documents_the_field(tmp_path):
    """The docstring IS the MCP tool description — an undocumented field on
    this surface is a field the LLM caller never learns exists."""
    doc = mcp.aggregator_query.__doc__ or ""
    assert "reranked_count" in doc
