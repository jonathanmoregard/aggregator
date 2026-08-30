"""A cache the reader does not understand was served as if it did.

``_ensure_cache_ready`` compared the cache's ``PRAGMA user_version`` against
the reader's ``SCHEMA_VERSION`` with ``<``, so it refused a cache that was
BEHIND and waved through a cache that was AHEAD. Measured on this tree before
the fix: there was no guard anywhere against a cache newer than the reader —
not in ``mcp.py``, not in ``Store``, not at any call site.

THAT IS THE MIRROR IMAGE OF THE INCIDENT, and it is the quieter half. A cache
behind the reader produces a refusal on every call, which is at least a fact
somebody can trip over. A cache ahead of the reader produces ANSWERS —
computed against tables whose shape this build does not know, columns it will
not select, and a vector index whose provenance it cannot evaluate. Wrong
recall that looks like recall is worse than no recall, because nothing about it
prompts anyone to look.

WHY THE TWO REFUSALS MUST NOT SHARE A MESSAGE. They have opposite causes and
opposite remedies:

* behind — the WRITER is the lagging side; deploy a newer writer (and until
  then the cache is stale but intelligible);
* ahead — the READER is the lagging side; deploy a newer reader, and touch the
  cache with nothing, because the only thing an old writer can do to a newer
  cache is damage it.

A message that conflates them sends the reader to work on the component that
is already correct. The tests below therefore assert on the CONTENT of each
message and on the fact that the two differ, not merely that a refusal
occurred.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pytest

from aggregator.core.store import SCHEMA_VERSION, Store
from aggregator.mcp import aggregator_capabilities, aggregator_query
from aggregator.sources.base import ObservationRow, SessionRow

_TS = datetime(2026, 8, 30, tzinfo=UTC)


def _seed(store: Store, n: int = 3) -> None:
    rows: list = []
    for i in range(n):
        sid = f"s{i}"
        rows.append(
            SessionRow(
                session_id=sid,
                root_session_id=sid,
                parent_session_id=None,
                kind="session",
                agent_id=None,
                agent_type=None,
                spawned_by_tool_use_id=None,
                cwd="/x",
                git_branch="main",
                first_ts=_TS,
                last_ts=_TS,
                jsonl_path=f"/tmp/{sid}.jsonl",
            )
        )
        rows.append(
            ObservationRow(
                obs_id=f"o{i}",
                session_id=sid,
                root_session_id=sid,
                parent_obs_id=None,
                type="user",
                ts=_TS,
                model=None,
                input_tokens=None,
                output_tokens=None,
                tool_name=None,
                tool_use_id=None,
                body=f"quadratic voting note {i}",
            )
        )
    store.upsert_entities(rows)


def _reader(db) -> Store:
    return Store(db_path=db, read_only=True)


def _stamp(db, version: int) -> None:
    c = sqlite3.connect(db)
    c.execute(f"PRAGMA user_version = {version};")
    c.commit()
    c.close()


@pytest.fixture
def current_db(tmp_path):
    db = tmp_path / "cache.db"
    store = Store(db_path=db)
    store.migrate()
    _seed(store)
    store.close()
    return db


@pytest.fixture
def ahead_db(current_db):
    """A cache written by a build newer than this reader.

    Stamped rather than built, because the gate under test reads the pragma
    and nothing else — ``_ensure_cache_ready`` runs before a single table is
    touched. There is also no newer schema in this tree to build one from,
    which is precisely the situation the guard exists for.

    The fixture proves its own premise: pre-fix, the assertion below is what
    fails, because the query was answered.
    """
    _stamp(current_db, SCHEMA_VERSION + 1)
    probe = _reader(current_db)
    assert probe.schema_version() == SCHEMA_VERSION + 1
    probe.close()
    return current_db


@pytest.fixture
def behind_db(current_db):
    _stamp(current_db, SCHEMA_VERSION - 1)
    return current_db


def _refusal(db) -> dict:
    return aggregator_query("voting", _store=_reader(db))


# --- the guard that did not exist -------------------------------------------


def test_a_cache_ahead_of_the_reader_is_refused(ahead_db):
    """THE REPRO. Pre-fix this returned ``ok: True`` and a result set.

    Answered out of tables this build has no description of, which is the
    whole objection: the rows come back looking exactly like correct ones.
    """
    result = _refusal(ahead_db)
    assert result["ok"] is False, result


def test_capabilities_refuses_an_ahead_cache_too(ahead_db):
    """Both surfaces run the same gate, and a caller can reach either first.

    ``aggregator_capabilities`` is the tool an agent calls to decide whether
    recall is usable at all, so a version of it that reports healthy over an
    unintelligible cache would license every query that follows.
    """
    result = aggregator_capabilities(_store=_reader(ahead_db))
    assert result["ok"] is False, result


def test_the_ahead_refusal_names_both_versions_it_compared(ahead_db):
    result = _refusal(ahead_db)
    blob = f"{result['reason']} {result['remediation']}"
    assert str(SCHEMA_VERSION + 1) in blob, blob
    assert str(SCHEMA_VERSION) in blob, blob


def test_the_ahead_refusal_says_the_cache_is_newer_not_older(ahead_db):
    """The one word that decides which component the reader goes and fixes.

    Reusing the "older than required" sentence here would be worse than
    saying nothing: it names the wrong lagging side with full confidence, and
    the operator's next act is to upgrade a writer that is already current.
    """
    reason = _refusal(ahead_db)["reason"].lower()
    assert "newer" in reason, reason
    assert "older" not in reason, reason


def test_the_ahead_and_behind_refusals_are_different_messages(current_db):
    """Same gate, same shape of failure, opposite diagnosis.

    Both readings are taken from ONE file, re-stamped between them, so the
    only thing that can account for a difference in the messages is the
    version comparison itself. Taken in sequence rather than from the two
    fixtures, because those stamp the same path and the second would silently
    win — the shape of mistake that makes a "both cases covered" test cover
    one case twice.
    """
    _stamp(current_db, SCHEMA_VERSION + 1)
    ahead = _refusal(current_db)
    _stamp(current_db, SCHEMA_VERSION - 1)
    behind = _refusal(current_db)
    assert ahead["ok"] is False and behind["ok"] is False
    assert ahead["reason"] != behind["reason"]
    assert ahead["remediation"] != behind["remediation"]


def test_the_ahead_refusal_sends_the_operator_to_the_reader(ahead_db):
    """The lagging side is this process. Say which process that is.

    A reader told only "versions disagree" has two candidates and no way to
    choose; naming the tree this module was imported from removes the guess.
    """
    remediation = _refusal(ahead_db)["remediation"]
    from aggregator.mcp import _READER_TREE

    assert _READER_TREE in remediation, remediation


def test_the_ahead_refusal_does_not_send_the_operator_to_the_writer(ahead_db):
    """The advice that fits the behind case is actively destructive here.

    Running an older writer against a newer cache is the one action that can
    turn "recall is refused" into "the cache is wrong": pre-fix it re-stamped
    ``user_version`` downward, and even a writer that refuses has no business
    being pointed at as a remedy for a reader that is behind.
    """
    remediation = _refusal(ahead_db)["remediation"].lower()
    assert "aggregator ingest" not in remediation, remediation
    assert "aggregator status" not in remediation, remediation


def test_the_ahead_refusal_never_offers_to_lower_the_cache(ahead_db):
    """Forward only. The cache is the leading side and stays where it is."""
    remediation = _refusal(ahead_db)["remediation"].lower()
    for banned in ("downgrade", "roll back", "rollback", "revert", "rebuild"):
        assert banned not in remediation, (banned, remediation)


# --- the controls -----------------------------------------------------------


def test_the_behind_refusal_is_unchanged_and_still_diagnoses_build_skew(behind_db):
    """The neighbouring branch must survive the widening of the comparison.

    ``<`` became ``!=``; if that rewrite had swallowed the behind branch, this
    is the test that says so.
    """
    remediation = _refusal(behind_db)["remediation"]
    assert "aggregator status" in remediation, remediation
    assert str(SCHEMA_VERSION - 1) in remediation, remediation


def test_a_current_cache_is_still_answered(current_db):
    """The path every real call takes. ``!=`` must not have made it stricter
    than ``<`` for the version that matches exactly."""
    result = aggregator_query("voting", _store=_reader(current_db))
    assert result["ok"] is True, result
    assert result["total"] > 0
