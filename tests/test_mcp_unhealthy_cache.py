"""An unhealthy cache must arrive as a structured refusal, like everything else.

``aggregator_query`` and ``aggregator_capabilities`` promise exactly two
shapes: ``{ok: True, ...}`` or ``{ok: False, reason, remediation}``. A raw
exception is neither. It reaches the MCP client as a transport-level tool
error with a SQLite string in it, which the model then has to interpret with
no remediation attached.

Three real gaps, all reproduced against a genuinely corrupt cache file rather
than a mocked one:

1. ``probe_fts`` ran outside any handler that could see the failure.
   ``sqlite3.DatabaseError`` — which is what SQLITE_CORRUPT raises — is the
   PARENT of ``sqlite3.OperationalError``, so the ``except OperationalError``
   guarding that call never caught it.
2. When the failure IS an ``OperationalError`` (a locked database), the same
   handler caught it and labelled it an **FTS5 syntax error**. The caller was
   then told its query text was malformed when the truth was that another
   process held the write lock. That is a wrong answer, not a missing one.
3. ``store.capabilities()`` had no handler at all.

And the routing probe: ``_vector_arm_engaged`` catches only
``VectorIndexUnavailableError``, and ``_apply_hybrid`` reaches it BEFORE the
per-path ``try``. Anything else from ``has_embedded_rows`` went straight out
of the tool.

The fix reuses ``CacheUnavailableError``'s existing response shape rather
than inventing a second vocabulary for the same condition.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime

import pytest

from aggregator.core.store import Store
from aggregator.mcp import aggregator_capabilities, aggregator_query
from aggregator.sources.base import ObservationRow, SessionRow

_TS = datetime(2026, 7, 1, tzinfo=UTC)


def _seed(store: Store, n: int = 400) -> None:
    rows: list = []
    for i in range(n):
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
                body=f"quadratic voting note {i} " + "y" * 600,
            )
        )
    store.upsert_entities(rows)


@pytest.fixture
def healthy_db(tmp_path):
    db = tmp_path / "cache.db"
    store = Store(db_path=db)
    store.migrate()
    _seed(store)
    store.close()
    return db


@pytest.fixture
def corrupt_db(healthy_db):
    """A cache whose page 1 is intact and whose content pages are not.

    Page 1 carries ``user_version``, so ``_ensure_cache_ready`` reads a
    perfectly good schema version and waves the query through — which is
    precisely why the failure surfaces later, at the first real read, instead
    of at the health check that exists to catch it.
    """
    # Fold the WAL into the main file first; corrupting the main file is
    # pointless while the rows the reader will see still live in the sidecar.
    c = sqlite3.connect(healthy_db)
    c.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    c.execute("PRAGMA journal_mode = DELETE").fetchone()
    c.close()
    size = os.path.getsize(healthy_db)
    with open(healthy_db, "r+b") as fh:
        fh.seek(4096 * 4)
        fh.write(os.urandom(size - 4096 * 4))

    # The fixture must not be able to pass vacuously: prove the file really is
    # unreadable, and that the health check really does still pass it.
    probe = Store(db_path=healthy_db, read_only=True)
    assert probe.schema_version() == 5
    with pytest.raises(sqlite3.DatabaseError):
        probe.probe_fts("voting")
    probe.close()
    return healthy_db


def _reader(db):
    return Store(db_path=db, read_only=True)


# --- the corrupt-cache path -------------------------------------------------


def test_a_malformed_cache_is_a_structured_refusal_not_a_traceback(corrupt_db):
    """THE REPRO. Pre-fix this raised sqlite3.DatabaseError out of the tool."""
    result = aggregator_query("voting", _store=_reader(corrupt_db))
    assert result["ok"] is False
    assert result["remediation"]


def test_a_malformed_cache_is_not_reported_as_an_fts5_syntax_error(corrupt_db):
    """The reviewer's specific warning: a broken DB read as a broken query."""
    result = aggregator_query("voting", _store=_reader(corrupt_db))
    assert "fts5" not in result["reason"].lower()
    assert "syntax" not in result["reason"].lower()
    assert "cache" in result["reason"].lower()


def test_a_filter_only_query_also_refuses_structurally(corrupt_db):
    result = aggregator_query("source:sessions", _store=_reader(corrupt_db))
    assert result["ok"] is False


def test_capabilities_refuses_structurally_on_a_malformed_cache(corrupt_db):
    """``store.capabilities()`` was called with no handler around it at all."""
    result = aggregator_capabilities(_store=_reader(corrupt_db))
    assert result["ok"] is False
    assert result["remediation"]


# --- the routing probe, which runs before every per-path handler ------------


def test_a_locked_db_at_the_routing_probe_does_not_escape(healthy_db, monkeypatch):
    """``_vector_arm_engaged`` catches only VectorIndexUnavailableError.

    A lock taken between the health check and the probe — the ingest timer
    fires every 30 minutes, so this is a scheduled event, not a freak one —
    left ``sqlite3.OperationalError`` to walk out of the tool.
    """
    store = _reader(healthy_db)

    def _locked(_kind):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(store, "has_embedded_rows", _locked)
    result = aggregator_query("voting", _store=store)
    assert result["ok"] is False
    assert result["remediation"]
    assert "syntax" not in result["reason"].lower()


# --- and the thing that must NOT change ------------------------------------


def test_hostile_query_text_on_a_healthy_cache_is_answered_not_refused(healthy_db):
    """Was ``test_a_real_fts5_syntax_error_is_still_reported_as_one``, which
    pinned the old contract: ``'voting AND OR "'`` came back ``ok: False`` with
    "FTS5 syntax error" in the reason. There is no such error left to report —
    ``fts5_match_query`` whitelists the text before it reaches ``MATCH``, so
    the operators and the stray quote are gone and the words are answered.

    The concern that test guarded is not dropped, it moved: what must never
    happen is a CACHE failure being labelled a query-text problem, and
    ``test_a_malformed_cache_is_not_reported_as_an_fts5_syntax_error`` above is
    now the only test that owns that distinction. Here the mirror image is
    pinned instead — hostile text against a healthy cache must never produce a
    cache-unavailable refusal.
    """
    result = aggregator_query('voting AND OR "', _store=_reader(healthy_db))
    assert result["ok"] is True, result
    # '"voting" "AND" "OR"' — all three tokens, AND-ed. The corpus has the
    # first and not the others, so the honest answer is zero hits.
    assert result["total"] == 0
    assert aggregator_query("voting", _store=_reader(healthy_db))["total"] > 0


def test_a_healthy_cache_still_answers(healthy_db):
    result = aggregator_query("voting", _store=_reader(healthy_db))
    assert result["ok"] is True
    assert result["total"] > 0
