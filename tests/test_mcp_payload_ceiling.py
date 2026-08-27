"""D3: nothing bounded the response, so ``page_size=8`` returned 282,110 chars.

THE FIELD REPORT. Two calls in one session overran the tool output limit and
were spilled to files by the harness — 83,909 characters, then 282,110. The
second was ``page_size=8``. ``page_size`` bounds ROWS; a single row can carry a
whole compacted session, so eight of them carried a third of a megabyte. The
spill is not free: it forces a jq round-trip to read anything, which degrades
the tool to the grep workflow it exists to replace, plus the cost of having
called it.

Grepping ``mcp.py`` for size caps before this change found the page-token
budget (``_frozen_over_budget``, ``_MAX_FROZEN_ID_CHARS``) and the
cross-encoder's 512-token pair budget. Neither bounds the response. There was
no governor at all.

WHY TRUNCATION HERE AND REFUSAL THERE. ``_PageTokenError`` deliberately refuses
an over-long page token rather than truncating it, because a truncated frozen
set is "plausible but wrong" — it looks like a working continuation and
silently addresses different rows. Content is the opposite call: a truncated
body is still the right body, and the caller can see it is short. That only
holds while the cut is DECLARED, which is why ``truncated`` and
``content_length`` are on every item rather than only on the cut ones, and why
they are not optional.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

import aggregator.mcp as mcp
from aggregator.core.store import Store
from aggregator.sources.base import ObservationRow, SessionRow

_TS = datetime(2026, 7, 28, tzinfo=UTC)

#: One row of the shape that produced the 282k spill: a whole compacted
#: session in a single observation body.
_HUGE = ("compacted session transcript about the pelican budget review " * 600)
_SMALL = "a short user turn about the pelican budget"


def _sess(sid: str) -> SessionRow:
    return SessionRow(
        session_id=sid, root_session_id=sid, parent_session_id=None,
        kind="session", agent_id=None, agent_type=None,
        spawned_by_tool_use_id=None, cwd="/home/jonathan/Repos/aggregator",
        git_branch="main", first_ts=_TS, last_ts=_TS + timedelta(minutes=5),
        jsonl_path=f"/tmp/{sid}.jsonl",
    )


def _obs(obs_id: str, sid: str, body: str, offset: int = 0):
    return ObservationRow(
        obs_id=obs_id, session_id=sid, root_session_id=sid, parent_obs_id=None,
        type="user", ts=_TS + timedelta(seconds=offset), model=None,
        input_tokens=None, output_tokens=None, tool_name=None,
        tool_use_id=None, body=body,
    )


@pytest.fixture
def fat_store(tmp_path):
    """Eight rows, each carrying a compacted session. Its own temp database."""
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    rows: list = [_sess("sess-fat")]
    rows += [_obs(f"o-fat-{i}", "sess-fat", _HUGE, offset=i) for i in range(8)]
    s.upsert_entities(rows)
    return s


@pytest.fixture
def mixed_store(tmp_path):
    """One giant row beside seven small ones — the starvation case."""
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    rows: list = [_sess("sess-mixed"), _obs("o-giant", "sess-mixed", _HUGE)]
    rows += [
        _obs(f"o-small-{i}", "sess-mixed", _SMALL, offset=i + 1)
        for i in range(7)
    ]
    s.upsert_entities(rows)
    return s


def _query(store, **kw):
    result = mcp.aggregator_query(
        dsl="source:sessions type:user pelican budget",
        fields="full", drilldown=True, _store=store, **kw,
    )
    assert result["ok"] is True, result
    return result


def _size(result) -> int:
    return len(json.dumps(result, default=str, ensure_ascii=False))


# --- the governor ---------------------------------------------------------


def test_a_fat_page_comes_back_under_the_ceiling(fat_store):
    """THE REPRO. Eight rows, no governor, 282,110 characters."""
    result = _query(fat_store, page_size=8)
    assert len(result["records"]) == 8, "rows must not be dropped"
    assert _size(result) <= mcp._MAX_RESPONSE_CHARS, _size(result)


def test_page_size_is_not_the_governor(fat_store):
    """The bug in one line: eight rows is a small page and a huge payload."""
    untruncated = sum(len(r) for r in [_HUGE] * 8)
    assert untruncated > mcp._MAX_RESPONSE_CHARS
    result = _query(fat_store, page_size=8)
    assert _size(result) <= mcp._MAX_RESPONSE_CHARS


def test_every_truncated_item_declares_it(fat_store):
    result = _query(fat_store, page_size=8)
    cut = [r for r in result["records"] if r["truncated"]]
    assert cut, "a page this size cannot have come back whole"
    for rec in cut:
        assert rec["truncated"] is True
        assert len(rec["content"]) < rec["content_length"]


def test_the_declared_length_is_the_true_one(fat_store):
    """``content_length`` has to be the size of the body that EXISTS, not the
    size of what came back — otherwise it says nothing the caller could not
    already measure with len()."""
    whole = _query(fat_store, page_size=1)
    assert whole["records"][0]["truncated"] is False
    true_len = len(whole["records"][0]["content"])

    cut = _query(fat_store, page_size=8)
    for rec in cut["records"]:
        assert rec["content_length"] == true_len, rec["content_length"]


def test_an_untruncated_item_still_states_the_fact(fat_store):
    """Stated, not inferred from a missing key — the same rule
    ``low_confidence`` follows. A caller must never have to read the absence
    of a field as a claim about the body."""
    result = _query(fat_store, page_size=1)
    for rec in result["records"]:
        assert rec["truncated"] is False
        assert rec["content_length"] == len(rec["content"])


def test_truncation_leaves_the_untrusted_wrapper_closed(fat_store):
    """Slicing the assembled string would cut off ``</ExternalContent>`` and
    leave the boundary open — every byte after it reads as trusted."""
    result = _query(fat_store, page_size=8)
    for rec in result["records"]:
        if not rec["content"]:
            continue
        assert rec["content"].count("<ExternalContent") == 1, rec["content"][:200]
        assert "</ExternalContent>" in rec["content"], rec["content"][-200:]


def test_a_truncated_body_says_so_in_band_too(fat_store):
    result = _query(fat_store, page_size=8)
    cut = [r for r in result["records"] if r["truncated"] and r["content"]]
    assert cut
    for rec in cut:
        assert mcp._CONTENT_TRUNCATION_MARKER in rec["content"], rec["content"][-200:]


def test_the_response_says_that_it_cut_something(fat_store):
    result = _query(fat_store, page_size=8)
    notice = result.get("notice", "")
    assert "truncat" in notice.lower(), notice
    assert "content_length" in notice, notice


def test_a_small_page_is_left_alone(mixed_store):
    """The governor is a ceiling, not a policy. Nothing under it is touched."""
    result = mcp.aggregator_query(
        dsl="source:sessions type:user pelican budget",
        fields="summary", drilldown=True, _store=mixed_store,
    )
    assert result["ok"] is True
    assert all(r["truncated"] is False for r in result["records"]), result
    assert "truncat" not in result.get("notice", "").lower()


def test_one_giant_row_does_not_starve_the_small_ones(mixed_store):
    """A flat per-row budget would cut seven readable rows to make room for
    one that cannot fit anyway. Fair-share gives every row what it asks for
    until the budget binds, and only then splits the remainder."""
    result = _query(mixed_store, page_size=8)
    small = [r for r in result["records"] if r["obs_id"].startswith("o-small")]
    assert len(small) == 7
    for rec in small:
        assert rec["truncated"] is False, rec
        assert _SMALL in rec["content"], rec["content"]


# --- the caller's own bound ------------------------------------------------


def test_max_chars_lowers_the_ceiling(mixed_store):
    result = _query(mixed_store, page_size=8, max_chars=6000)
    assert _size(result) <= 6000, _size(result)
    assert any(r["truncated"] for r in result["records"])


def test_max_chars_cannot_raise_the_ceiling(fat_store):
    """The server-side bound is not negotiable — that is the whole point of
    putting it on the server. ``max_chars`` may only ask for less."""
    result = _query(fat_store, page_size=8, max_chars=10_000_000)
    assert _size(result) <= mcp._MAX_RESPONSE_CHARS, _size(result)


# --- the case truncating content cannot fix -------------------------------


def test_a_page_whose_metadata_alone_overruns_says_so(tmp_path):
    """Truncating content cannot bring a 200-row page under the ceiling when
    200 rows of ids and timestamps are already over it. Pretending otherwise
    is the "plausible but wrong" failure in a different costume, so this must
    be stated, with the remediation that actually works.
    """
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    rows: list = [_sess("sess-many")]
    rows += [_obs(f"o-many-{i:04d}", "sess-many", _SMALL, offset=i)
             for i in range(300)]
    s.upsert_entities(rows)
    result = mcp.aggregator_query(
        dsl="source:sessions type:user pelican budget",
        fields="summary", drilldown=True, page_size=300, _store=s,
    )
    assert result["ok"] is True, result
    notice = result.get("notice", "")
    assert "page_size" in notice, notice
    assert all(r["truncated"] is True for r in result["records"]), result["records"]


def test_the_metadata_of_a_default_summary_page_is_already_large(tmp_path):
    """A measured fact this change surfaces rather than fixes.

    ``_DEFAULT_PAGE_SIZE_SUMMARY`` is 200, and 200 observation rows cost
    ~73 000 characters of ids and timestamps BEFORE any body. This is here so
    the number cannot drift silently, and so the next person to look at the
    default page size finds the measurement rather than re-deriving it.
    """
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    rows: list = [_sess("sess-env")]
    rows += [_obs(f"o-env-{i:04d}", "sess-env", _SMALL, offset=i)
             for i in range(220)]
    s.upsert_entities(rows)
    result = mcp.aggregator_query(
        dsl="source:sessions type:user pelican budget",
        fields="summary", drilldown=True, page_size=200, _store=s,
    )
    blanked = json.loads(json.dumps(result, default=str))
    for rec in blanked["records"]:
        rec["content"] = ""
    envelope = len(json.dumps(blanked, ensure_ascii=False))
    assert envelope > mcp._MAX_RESPONSE_CHARS, envelope
