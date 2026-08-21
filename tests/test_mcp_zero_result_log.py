"""Criterion D: a query that finds nothing is written down for the golden set.

THE LOOP THIS CLOSES. The eval harness can only measure queries somebody
thought to freeze, and the queries worth freezing are precisely the ones that
currently fail — which nobody remembers to write down, because a zero-result
answer looks like a question with no answer rather than like a retrieval bug.
``search_misses`` is where they accumulate; ``suggest_from_misses`` surfaces
them at the end of every regression run.

A SEPARATE DATABASE FROM ``cache.db``, and the eval store refuses that name
outright. The harness must never migrate or write to the artifact it measures,
and a baseline has to survive a re-ingest, a vector rebuild and a model swap.
It is co-located with whichever cache is being served, so in production it IS
``$XDG_DATA_HOME/aggregator/retrieval_eval.db`` — the same file
``default_eval_db_path`` resolves — while a test or a scratch cache gets its
own beside itself instead of writing into the developer's home.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from aggregator.core.store import Store
from aggregator.evals.db import EvalStore
from aggregator.mcp import aggregator_query
from aggregator.sources.base import ObservationRow, SessionRow

_TS = datetime(2026, 7, 1, 8, 0, tzinfo=UTC)


@pytest.fixture
def store(tmp_path):
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    sid = "s1"
    s.upsert_entities(
        [
            SessionRow(
                session_id=sid, root_session_id=sid, parent_session_id=None,
                kind="session", agent_id=None, agent_type=None,
                spawned_by_tool_use_id=None, cwd="/x", git_branch="main",
                first_ts=_TS, last_ts=_TS, jsonl_path="/tmp/s1.jsonl",
            ),
            ObservationRow(
                obs_id="o1", session_id=sid, root_session_id=sid,
                parent_obs_id=None, type="user", ts=_TS, model=None,
                input_tokens=None, output_tokens=None, tool_name=None,
                tool_use_id=None, body="quadratic voting is a governance idea",
            ),
        ]
    )
    return s


@pytest.fixture(autouse=True)
def _fresh_miss_log(monkeypatch):
    """The log connection is a module singleton, like the embedder. Tests share
    a process, so one test's connection would otherwise answer another test's
    cache — and point at a ``tmp_path`` that has already been torn down."""
    monkeypatch.setattr("aggregator.mcp._miss_log", None)
    monkeypatch.setattr("aggregator.mcp._miss_log_path", None)


def _misses(tmp_path) -> list[dict]:
    eval_store = EvalStore(tmp_path / "retrieval_eval.db")
    try:
        return eval_store.search_misses()
    finally:
        eval_store.close()


def test_a_zero_result_text_query_is_logged(store, tmp_path):
    result = aggregator_query("beef wellington recipe", _store=store)
    assert result["ok"] is True and result["total"] == 0, result
    misses = _misses(tmp_path)
    assert [m["query_text"] for m in misses] == ["beef wellington recipe"]
    assert misses[0]["result_count"] == 0


def test_the_log_records_which_arms_were_asked(store, tmp_path):
    """A miss under ``lexical`` and a miss under ``hybrid`` are different
    facts about the pipeline, and a log that cannot tell them apart cannot say
    whether the vector arm would have found it."""
    aggregator_query("beef wellington", search_mode="lexical", _store=store)
    assert _misses(tmp_path)[0]["mode"] == "lexical"


def test_a_query_that_found_something_is_not_logged(store, tmp_path):
    result = aggregator_query("voting", _store=store)
    assert result["total"] > 0, result
    assert _misses(tmp_path) == []


def test_a_filter_only_query_is_never_logged(store, tmp_path):
    """The log feeds a RETRIEVAL golden set. A filter that matched no rows is a
    fact about the corpus, not about ranking, and freezing it as a golden query
    would measure nothing."""
    result = aggregator_query("source:github", _store=store)
    assert result["total"] == 0, result
    assert _misses(tmp_path) == []


def test_nothing_is_written_until_something_misses(store, tmp_path):
    """The happy path must not pay a second database's open+migrate."""
    aggregator_query("voting", _store=store)
    assert not (tmp_path / "retrieval_eval.db").exists()


def test_an_ontology_mismatch_is_not_logged_as_a_miss(store, tmp_path):
    """``source:github session:abc voting`` returns nothing because the two
    filter families cannot both apply, not because retrieval failed. Freezing
    it into the golden set would pin a query that can never return a row, and
    it would score as a permanent abstention nobody can fix."""
    result = aggregator_query("source:github session:abc voting", _store=store)
    assert result["ok"] is True and result["total"] == 0, result
    assert "do not apply" in result["notice"]
    assert _misses(tmp_path) == []


def test_a_refused_query_is_not_logged_as_a_miss(store, tmp_path):
    result = aggregator_query("voting", fields="bogus", _store=store)
    assert result["ok"] is False
    assert _misses(tmp_path) == []


def test_repeated_misses_all_land(store, tmp_path):
    """Append-only and not deduplicated on purpose: how OFTEN a query misses is
    the signal for which misses are worth freezing."""
    aggregator_query("beef wellington", _store=store)
    aggregator_query("beef wellington", _store=store)
    assert len(_misses(tmp_path)) == 2


def test_a_broken_miss_log_is_reported_and_does_not_cost_the_answer(
    store, tmp_path, monkeypatch
):
    """Both halves matter. Losing the answer to report a lost log entry is the
    wrong trade; losing the log entry SILENTLY is the banned one — an empty
    zero-result log is indistinguishable from a healthy pipeline."""

    def _boom(*a, **kw):
        raise OSError("read-only file system")

    monkeypatch.setattr("aggregator.evals.db.EvalStore.__init__", _boom)
    result = aggregator_query("beef wellington", _store=store)
    assert result["ok"] is True, result
    assert result["total"] == 0
    assert "zero-result" in result["notice"].lower()
    assert "read-only file system" in result["notice"]
