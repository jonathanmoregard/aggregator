"""``rerank=True`` must not spend 47 seconds ranking text it was never given.

``_rerank_doc`` builds the string the cross-encoder scores out of the result
ITEM, not out of the stored row — and an item's ``content`` is deliberately
empty unless ``fields='full'``. So under the DEFAULT ``fields='summary'`` the
reranker was handed bodiless documents. Measured, per route, by spying on the
docs actually passed to ``score()``:

    records,  summary (DEFAULT)  3 docs, 3 with no body: 'pr 0 about …\\n\\n'
    records,  full               3 docs, 0 with no body: subject + body
    obs,      summary (DEFAULT)  3 docs, 3 with no body, ONE DISTINCT STRING:
                                 'user\\n\\n', 'user\\n\\n', 'user\\n\\n'
    sessions, summary (DEFAULT)  3 docs, 3 with no body
    union,    summary (DEFAULT)  6 docs, 6 with no body

The observation route is the pure case: every document is the same literal
string, so the ordering is noise by construction. The others are not empty but
are no better a deal, and this is the part worth being precise about — they
rank on ``subject``, which the response ALREADY returns to the caller. A
47-second cross-encoder pass over fields the caller is holding in its hand
buys nothing it could not compute for free.

The decision is to refuse, not to auto-upgrade ``fields``. Auto-upgrading
would change the caller's payload shape behind its back — summary and full
items differ in more than one key — and would still cost the 47 s. Refusing
costs nothing at all: it happens before any retrieval runs.

Measured cost of the thing being refused: 47 s median, 59 s worst case per
call on this CPU, against 0.65 s for the same query without it.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

import aggregator.mcp as mcp
from aggregator.core.store import Store
from aggregator.sources.base import ObservationRow, Record, SessionRow

_TS = datetime(2026, 7, 1, tzinfo=UTC)


class SpyReranker:
    """Records what it was asked to score, and reverses the order.

    Never a real cross-encoder: naming a real model in a test is how an
    earlier round pulled 15 GB off a CDN before anyone noticed.
    """

    def __init__(self):
        self.docs: list[str] = []
        self.calls = 0

    def score(self, query, docs):
        self.calls += 1
        self.docs.extend(docs)
        return list(range(len(docs)))  # ascending => the sort reverses them


@pytest.fixture
def reranker(monkeypatch):
    spy = SpyReranker()
    monkeypatch.setattr(mcp, "_get_reranker", lambda: spy)
    return spy


@pytest.fixture
def store(tmp_path):
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert(
        [
            Record(
                stable_id=f"github:acme/api:{i}",
                source="github",
                subject=f"pr {i} about quadratic voting",
                body=f"body of pull request {i}: the text that actually says "
                     f"whether this row answers the query",
                tags=["pr"],
                created_at=_TS,
                updated_at=_TS,
            )
            for i in range(3)
        ]
    )
    rows: list = []
    for i in range(3):
        sid = f"s{i}"
        rows.append(
            SessionRow(
                session_id=sid, root_session_id=sid, parent_session_id=None,
                kind="session", agent_id=None, agent_type=None,
                spawned_by_tool_use_id=None, cwd="/x", git_branch="main",
                first_ts=_TS, last_ts=_TS, jsonl_path=f"/tmp/{sid}.jsonl",
            )
        )
        rows.append(
            ObservationRow(
                obs_id=f"o{i}", session_id=sid, root_session_id=sid,
                parent_obs_id=None, type="user", ts=_TS, model=None,
                input_tokens=None, output_tokens=None, tool_name=None,
                tool_use_id=None,
                body=f"observation {i}: a discussion of quadratic voting",
            )
        )
    s.upsert_entities(rows)
    return s


_SUMMARY_ROUTES = [
    ("records", {"dsl": "source:github voting"}),
    ("observations", {"dsl": "source:sessions voting", "drilldown": True}),
    ("sessions", {"dsl": "source:sessions voting"}),
    ("union", {"dsl": "voting"}),
]


# --- the refusal ------------------------------------------------------------


@pytest.mark.parametrize("route,kwargs", _SUMMARY_ROUTES, ids=[r for r, _ in _SUMMARY_ROUTES])
def test_rerank_is_refused_in_summary_mode(route, kwargs, store, reranker):
    """THE REPRO. Every route handed the cross-encoder bodiless documents."""
    result = mcp.aggregator_query(rerank=True, _store=store, **kwargs)
    assert result["ok"] is False, (
        f"{route}: scored {len(reranker.docs)} bodiless docs "
        f"({sorted(set(reranker.docs))[:2]})"
    )
    assert reranker.calls == 0


def test_the_refusal_names_the_fix(store, reranker):
    result = mcp.aggregator_query("voting", rerank=True, _store=store)
    assert "fields" in result["reason"]
    assert "full" in result["remediation"]


def test_the_refusal_is_free(store, reranker, monkeypatch):
    """Refusing after running the query would charge for the answer it
    withholds. The check runs before any retrieval."""

    def _explode(*a, **k):
        raise AssertionError("retrieval ran before the rerank contract check")

    monkeypatch.setattr(store, "query", _explode)
    monkeypatch.setattr(store, "query_sessions", _explode)
    result = mcp.aggregator_query("voting", rerank=True, _store=store)
    assert result["ok"] is False
    # Not vacuous: the union path catches a broken store and also answers
    # ok=False, so the refusal has to be identifiable as THIS refusal.
    assert "rerank" in result["reason"], result["reason"]


# --- and what must keep working --------------------------------------------


def test_rerank_with_fields_full_reorders_the_page(store, reranker):
    plain = mcp.aggregator_query("source:github voting", fields="full", _store=store)
    ranked = mcp.aggregator_query(
        "source:github voting", fields="full", rerank=True, _store=store
    )
    assert ranked["ok"] is True
    assert reranker.calls == 1
    plain_ids = [r["stable_id"] for r in plain["records"]]
    ranked_ids = [r["stable_id"] for r in ranked["records"]]
    assert sorted(plain_ids) == sorted(ranked_ids)
    assert ranked_ids == list(reversed(plain_ids))


def test_the_scored_document_carries_the_body_in_full_mode(store, reranker):
    mcp.aggregator_query(
        "source:github voting", fields="full", rerank=True, _store=store
    )
    assert reranker.docs
    for doc in reranker.docs:
        assert doc.partition("\n\n")[2].strip(), f"still bodiless: {doc!r}"


def test_summary_mode_without_rerank_is_untouched(store, reranker):
    """The default call is the common one and must not have moved."""
    result = mcp.aggregator_query("voting", _store=store)
    assert result["ok"] is True
    assert result["total"] > 0
    assert reranker.calls == 0
