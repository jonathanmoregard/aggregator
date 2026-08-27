"""D1: summary mode returned rows whose ``content`` was the empty string.

THE FIELD REPORT. A caller asked "when did the user first ask for clickable PR
links in status reports?", ran the mandated recall tool, and got five correct
rows back — every one of them ``content: ""``. There was no way to tell a
bullseye from a false positive, so the caller re-called at ``fields='full'``,
spilled 282,110 characters to disk, and abandoned the tool for a grep script.
The retrieval was never the problem; the response format was.

WHY IT WAS NEVER A STORAGE COST. ``_observation_to_item``'s full-mode branch
slices ``o.body[:120]`` for its subject with no extra fetch — the body is
already on the row that summary mode threw away. Emitting a snippet is
therefore a display decision, which is why this fixes it in place rather than
adding a third ``fields='snippet'`` mode.

``_session_to_item`` had the identical shape, and that is what made
``matching_observations: N`` unactionable: a count of turns you cannot read.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

import aggregator.mcp as mcp
from aggregator.core.store import Store
from aggregator.sources.base import ObservationRow, SessionRow

_TS = datetime(2026, 7, 28, 15, 32, 21, tzinfo=UTC)

#: The turn the field report was hunting for, verbatim in shape.
_ANSWER = (
    "link prs. Also: please somehow make you always link PRs when delivering "
    "a status report: I want clickable links"
)

#: A body long enough that a 200-character window cannot contain both ends,
#: with the query term buried in the middle. Proves the window is CENTRED on
#: the hit rather than taken from the head.
_BURIED = (
    "preamble that says nothing useful " * 20
    + "the clickable links request lands here "
    + "and then a long tail of unrelated chatter " * 20
)


def _sess(sid: str, **kw) -> SessionRow:
    return SessionRow(
        session_id=sid, root_session_id=kw.get("root", sid),
        parent_session_id=None, kind="session", agent_id=None, agent_type=None,
        spawned_by_tool_use_id=None, cwd="/home/jonathan/Repos/aggregator",
        git_branch="main", first_ts=_TS, last_ts=_TS + timedelta(minutes=5),
        jsonl_path=f"/tmp/{sid}.jsonl",
    )


def _obs(obs_id: str, sid: str, body: str, *, obs_type="user", offset=0):
    return ObservationRow(
        obs_id=obs_id, session_id=sid, root_session_id=sid, parent_obs_id=None,
        type=obs_type, ts=_TS + timedelta(seconds=offset), model=None,
        input_tokens=None, output_tokens=None, tool_name=None,
        tool_use_id=None, body=body,
    )


@pytest.fixture
def store(tmp_path):
    """Its own temporary database. Never the live cache."""
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert_entities(
        [
            _sess("sess-answer"),
            # The session's OPENING turn does not mention the query terms —
            # so a card whose snippet came from its subject would show
            # nothing, which is the defect ``matching_observations`` had.
            _obs("o-open", "sess-answer", "start a new branch for the parser",
                 offset=0),
            _obs("o-answer", "sess-answer", _ANSWER, offset=10),
            _sess("sess-buried"),
            _obs("o-open-2", "sess-buried", "unrelated opening prompt", offset=0),
            _obs("o-buried", "sess-buried", _BURIED, offset=10),
        ]
    )
    return s


def _obs_rows(store, **kw):
    result = mcp.aggregator_query(
        dsl='source:sessions type:user "clickable links"',
        fields="summary", drilldown=True, _store=store, **kw,
    )
    assert result["ok"] is True, result
    return result


def _inner(content: str) -> str:
    """The payload inside the ``<ExternalContent>`` wrapper."""
    return content.partition(">\n")[2].rpartition("\n</ExternalContent>")[0]


# --- A: observations ------------------------------------------------------


def test_summary_drilldown_returns_a_legible_snippet(store):
    """THE REPRO. Five correct rows, every content an empty string."""
    result = _obs_rows(store)
    assert result["records"], result
    for rec in result["records"]:
        assert rec["content"], f"empty content in summary mode: {rec!r}"
    joined = " ".join(r["content"] for r in result["records"])
    assert "clickable" in joined, joined


def test_the_snippet_marks_the_terms_that_matched(store):
    result = _obs_rows(store)
    joined = " ".join(r["content"] for r in result["records"])
    assert "[[clickable]]" in joined, joined
    assert "[[links]]" in joined, joined


def test_the_snippet_is_bounded(store):
    """A snippet is a display budget, not a body. ``page_size`` bounds rows;
    this is what bounds the row."""
    result = _obs_rows(store)
    for rec in result["records"]:
        assert len(_inner(rec["content"])) <= mcp._SNIPPET_CHARS * 2, (
            f"snippet ran away: {len(_inner(rec['content']))} chars"
        )


def test_the_snippet_is_centred_on_the_hit_not_taken_from_the_head(store):
    """The hit is ~700 characters into the body. A head slice would miss it."""
    result = _obs_rows(store)
    buried = next(r for r in result["records"] if r["obs_id"] == "o-buried")
    inner = _inner(buried["content"])
    assert "[[clickable]]" in inner, inner
    assert inner.startswith("…"), inner
    assert "preamble that says nothing useful preamble" not in inner[:40], inner


def test_the_snippet_is_wrapped_as_untrusted_content(store):
    """Summary mode now returns real body text, so the boundary rule applies.

    M1 skipped the wrapper because the content was empty and an empty
    ``<ExternalContent>`` block is cosmetically misleading. The moment the
    block holds untrusted text again, the tool docstring's promise — "content
    is returned inside <ExternalContent> delimiters" — has to hold too.
    """
    result = _obs_rows(store)
    for rec in result["records"]:
        assert rec["content"].startswith('<ExternalContent source="'), rec
        assert rec["content"].endswith("</ExternalContent>"), rec


def test_summary_mode_no_longer_tells_the_caller_to_re_call_at_full(store):
    """The old notice read "Re-call with fields=full to include observation
    bodies" — the instruction that produced the 282k-character spill."""
    result = _obs_rows(store)
    notice = result.get("notice", "")
    assert "snippet" in notice.lower(), notice
    assert "Re-call with fields=full to include observation bodies" not in notice


# --- A: session cards -----------------------------------------------------


def _session_cards(store):
    result = mcp.aggregator_query(
        dsl='source:sessions "clickable links"', fields="summary", _store=store,
    )
    assert result["ok"] is True, result
    return result


def test_a_session_card_carries_a_snippet_of_what_matched(store):
    """``matching_observations: N`` advertises that something matched without
    ever showing what. The card has to carry the evidence."""
    result = _session_cards(store)
    card = next(r for r in result["records"] if r["stable_id"] == "sess-answer")
    assert card["matching_observations"] >= 1, card
    assert card["content"], f"session card content still empty: {card!r}"
    assert "[[clickable]]" in card["content"], card["content"]


def test_a_session_card_snippet_is_not_a_copy_of_its_own_subject(store):
    """The subject is the session's FIRST user turn; the evidence usually is
    not. A card whose body repeats its header shows the reader nothing."""
    result = _session_cards(store)
    card = next(r for r in result["records"] if r["stable_id"] == "sess-answer")
    assert card["subject"].startswith("start a new branch"), card["subject"]
    assert _inner(card["content"]).strip("…") not in card["subject"]


def test_a_session_card_snippet_is_bounded_and_wrapped(store):
    result = _session_cards(store)
    for card in result["records"]:
        assert card["content"].startswith('<ExternalContent source="'), card
        assert len(_inner(card["content"])) <= mcp._SNIPPET_CHARS * 2, card


def test_the_session_notice_describes_the_snippet(store):
    result = _session_cards(store)
    notice = result.get("notice", "")
    assert "snippet" in notice.lower(), notice


# --- degenerate cases -----------------------------------------------------


def test_a_pure_filter_query_still_gets_a_readable_head(store):
    """No free text means no term to centre on. The head of the body is still
    strictly more legible than the empty string."""
    result = mcp.aggregator_query(
        dsl="source:sessions type:user", fields="summary", drilldown=True,
        _store=store,
    )
    assert result["ok"] is True, result
    assert result["records"]
    assert all(r["content"] for r in result["records"]), result["records"]


def test_an_empty_body_does_not_wrap_an_empty_block(tmp_path):
    """The other direction of honesty: a block that looks like content and
    holds none is what M1 removed from summary mode in the first place."""
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert_entities(
        [_sess("sess-blank"), _obs("o-blank", "sess-blank", "", offset=0)]
    )
    result = mcp.aggregator_query(
        dsl="source:sessions type:user", fields="summary", drilldown=True,
        _store=s,
    )
    assert result["ok"] is True, result
    for rec in result["records"]:
        assert rec["content"] == "", rec


def test_full_mode_still_returns_the_whole_body(store):
    """The snippet is an addition to summary mode, not a cap on full mode."""
    result = mcp.aggregator_query(
        dsl='source:sessions type:user "clickable links"',
        fields="full", drilldown=True, _store=store,
    )
    answer = next(r for r in result["records"] if r["obs_id"] == "o-answer")
    assert _ANSWER in answer["content"], answer["content"]
