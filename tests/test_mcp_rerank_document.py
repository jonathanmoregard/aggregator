"""A session card was handed to the cross-encoder as its own subject, twice.

``_query_sessions_path`` and ``_query_union_path`` both call
``_session_to_item(s, fields, subject, match_count, subject)`` — the LAST
argument is the ``body_preview`` slot, and it was given the subject. In
``fields='full'`` the card's ``content`` is therefore
``<ExternalContent source="…">\\n{subject}\\n</ExternalContent>``, and
``_rerank_doc`` concatenates head and content, so the string the cross-encoder
scores is the subject followed by the subject.

WHY THAT IS NOT MERELY REDUNDANT. ``rerank=True`` is refused outright unless
``fields='full'`` precisely so the reranker sees real bodies
(``tests/test_mcp_rerank_contract.py``). The union route is the DEFAULT — a
free-text query with no ``source:`` filter lands there — so on the common path
that refusal bought nothing: the caller paid the cross-encoder's full latency
to have near-empty documents ranked against each other, and ``reranked_count``
reported an ordering derived from them.

The fix gives the card the text that actually matched: the session's own
matching observations, which is what a cross-encoder needs to decide whether
this session answers the query, and what a reader wants to see too.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

import aggregator.mcp as mcp
from aggregator.core.store import Store
from aggregator.sources.base import ObservationRow, SessionRow

_TS = datetime(2026, 7, 1, tzinfo=UTC)

_SUBJECT = "please investigate the quadratic voting rollout"
_EVIDENCE = (
    "the quadratic tally applies a square-root cost curve, so the fourth vote "
    "costs sixteen credits and the pelican budget never clears"
)


class SpyReranker:
    """Records what it was asked to score. Never a real cross-encoder —
    naming one in a test is how an earlier round pulled 15 GB off a CDN."""

    def __init__(self):
        self.docs: list[str] = []

    def score(self, query, docs):
        self.docs.extend(docs)
        return list(range(len(docs)))


@pytest.fixture
def reranker(monkeypatch):
    spy = SpyReranker()
    monkeypatch.setattr(mcp, "_get_reranker", lambda: spy)
    return spy


@pytest.fixture
def store(tmp_path):
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    rows: list = []
    for i in range(2):
        sid = f"s{i}"
        rows.append(
            SessionRow(
                session_id=sid, root_session_id=sid, parent_session_id=None,
                kind="session", agent_id=None, agent_type=None,
                spawned_by_tool_use_id=None, cwd="/x", git_branch="main",
                first_ts=_TS, last_ts=_TS + timedelta(minutes=1),
                jsonl_path=f"/tmp/{sid}.jsonl",
            )
        )
        rows.append(
            ObservationRow(
                obs_id=f"{sid}-u", session_id=sid, root_session_id=sid,
                parent_obs_id=None, type="user", ts=_TS, model=None,
                input_tokens=None, output_tokens=None, tool_name=None,
                tool_use_id=None, body=f"{_SUBJECT} number {i}",
            )
        )
        rows.append(
            ObservationRow(
                obs_id=f"{sid}-a", session_id=sid, root_session_id=sid,
                parent_obs_id=None, type="assistant",
                ts=_TS + timedelta(minutes=1), model=None,
                input_tokens=None, output_tokens=None, tool_name=None,
                tool_use_id=None, body=f"{_EVIDENCE} in session {i}",
            )
        )
    s.upsert_entities(rows)
    return s


def _session_docs(spy: SpyReranker) -> list[str]:
    return [d for d in spy.docs if d.startswith(_SUBJECT)]


def _wrapped_body(doc: str) -> str:
    """The payload inside the card's ``<ExternalContent>`` wrapper."""
    _head, _, content = doc.partition("\n\n")
    inner = content.partition(">\n")[2]
    return inner.rpartition("\n</ExternalContent>")[0]


# --- the repro --------------------------------------------------------------


@pytest.mark.parametrize(
    "dsl", ["quadratic", "source:sessions quadratic"], ids=["union", "sessions"]
)
def test_a_session_card_is_not_scored_against_its_own_subject(dsl, store, reranker):
    """THE REPRO. Head and body were the same string."""
    result = mcp.aggregator_query(dsl, fields="full", rerank=True, _store=store)
    assert result["ok"] is True, result
    docs = _session_docs(reranker)
    assert docs, f"no session card reached the reranker: {reranker.docs!r}"
    for doc in docs:
        head = doc.partition("\n\n")[0]
        assert _wrapped_body(doc) != head, (
            f"the cross-encoder scored this document against itself: {doc!r}"
        )


@pytest.mark.parametrize(
    "dsl", ["quadratic", "source:sessions quadratic"], ids=["union", "sessions"]
)
def test_the_scored_session_card_carries_the_text_that_matched(dsl, store, reranker):
    """A card that answers the query has to SAY so somewhere in the document.

    The subject is the session's first user turn; the evidence for relevance
    usually is not. Ranking on the subject alone ranks on a field the response
    already returns, which the caller could have sorted for free.
    """
    mcp.aggregator_query(dsl, fields="full", rerank=True, _store=store)
    docs = _session_docs(reranker)
    assert docs
    assert any("square-root cost curve" in d for d in docs), (
        f"the matching observation never reached the reranker: {docs!r}"
    )


def test_two_sessions_do_not_look_identical_to_the_cross_encoder(store, reranker):
    """The ordering has to be derivable from something. Two cards whose
    documents are equal cannot be ranked against each other at all."""
    mcp.aggregator_query("quadratic", fields="full", rerank=True, _store=store)
    docs = _session_docs(reranker)
    assert len(docs) == 2, docs
    assert docs[0] != docs[1]


# --- and what must keep working --------------------------------------------


def test_the_card_still_wraps_its_body_as_untrusted_content(store, reranker):
    result = mcp.aggregator_query("quadratic", fields="full", _store=store)
    for rec in result["records"]:
        assert '<ExternalContent source="' in rec["content"]
        assert rec["content"].endswith("</ExternalContent>")


def test_summary_mode_does_not_pay_for_the_full_preview(store):
    """The 1 500-character, five-observation preview is a full-mode cost.

    D1 changed WHAT summary mode returns, not how much it spends: the card
    now carries a bounded snippet of ONE matching observation, so the caller
    can act on ``matching_observations`` instead of re-calling at
    ``fields='full'`` to find out what matched. The concatenated preview stays
    a full-mode artefact.
    """
    result = mcp.aggregator_query("quadratic", _store=store)
    for rec in result["records"]:
        assert rec["content"], f"summary card back to an opaque id: {rec!r}"
        inner = _wrapped_body(f"head\n\n{rec['content']}")
        assert len(inner) <= mcp._SNIPPET_CHARS * 2, len(inner)
        # The full preview joins several turns with a ``[type]`` label each.
        assert "[assistant]" not in inner, inner


def test_the_subject_is_still_the_first_user_prompt(store):
    result = mcp.aggregator_query("quadratic", fields="full", _store=store)
    subjects = [r["subject"] for r in result["records"]]
    assert all(s.startswith(_SUBJECT) for s in subjects), subjects


def test_a_session_with_no_matching_observations_still_renders(store, tmp_path):
    """``source:sessions`` with no free text scopes to no FTS hits at all.
    The preview must degrade to something, not raise."""
    result = mcp.aggregator_query("source:sessions", fields="full", _store=store)
    assert result["ok"] is True
    assert len(result["records"]) == 2
    for rec in result["records"]:
        assert isinstance(rec["content"], str)


def test_a_session_with_no_body_at_all_does_not_wrap_an_empty_block(store):
    """The degenerate case has to stay honest in the other direction.

    A session whose observations are all gone or all empty yields no preview.
    Wrapping "" in ``<ExternalContent>`` is the exact thing M1 removed from
    summary mode: a block that looks like content and holds none. Fall back to
    the subject — that card really does contain nothing else.
    """
    store.upsert_entities(
        [
            SessionRow(
                session_id="empty", root_session_id="empty", parent_session_id=None,
                kind="session", agent_id=None, agent_type=None,
                spawned_by_tool_use_id=None, cwd="/x", git_branch="main",
                first_ts=_TS, last_ts=_TS, jsonl_path="/tmp/empty.jsonl",
            )
        ]
    )
    result = mcp.aggregator_query("source:sessions", fields="full", _store=store)
    card = next(r for r in result["records"] if r["stable_id"] == "empty")
    assert card["subject"] in card["content"], card
    assert card["content"].partition(">\n")[2].rpartition("\n</")[0].strip(), (
        f"empty <ExternalContent> block: {card['content']!r}"
    )
