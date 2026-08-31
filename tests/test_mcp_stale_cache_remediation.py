"""A cache behind the reader has TWO causes, and the old advice fitted one.

``_ensure_cache_ready`` refuses every cache whose ``user_version`` is below
the reader's ``SCHEMA_VERSION``, and it used to attach one remediation to that
refusal: run ``aggregator status`` or ``aggregator ingest <source>``, because
"the writable aggregator path" can "create or migrate the cache". That advice
is correct for exactly one of the two situations that produce this refusal,
and it is actively harmful in the other.

THE SITUATION IT GETS WRONG IS BUILD SKEW, and build skew is what actually
happened here. The reader ran from a working checkout at ``SCHEMA_VERSION``
6; the writer — the ``aggregator`` CLI and the ``aggregator-ingest.timer``
behind it — was a pinned Nix build at 5. The cache sat at ``user_version``
5. Every thirty minutes the v5 writer opened it, re-stamped ``user_version =
5``, and **exited 0**. ``systemctl show aggregator-ingest.service`` reported
``ExecMainStatus=0``, every source read fresh, and every single recall was
refused. Recall was 100% dead for as long as that lasted, and silent, because
the half of the system with the exit code was the half that was fine.

Handed the old remediation, a caller does the one thing guaranteed not to
work — runs the stale writer — watches it succeed, and concludes the fault is
somewhere else entirely. A confident wrong diagnosis costs more than no
diagnosis, which is why this test exists and why it asserts on the CONTENT of
the remediation rather than merely on its presence.

What the message now owes the reader:

1. The two numbers it compared, so the reader can place itself.
2. Both branches, with the fix for each — and, for the skew branch, the fact
   that running the writer cannot ever be that fix.
3. Honesty about the writer's version, which this read-only surface does not
   measure. Naming the command that measures it is the fix for that; guessing
   at it would reintroduce the exact failure this test guards.
4. No suggestion that the reader should accept the older schema. Skew is
   resolved by bringing the lagging side up, never by pulling the leading
   side back.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime

import pytest

import aggregator.mcp as mcp_mod
from aggregator.core.store import SCHEMA_VERSION, Store
from aggregator.mcp import aggregator_capabilities, aggregator_query
from aggregator.sources.base import ObservationRow, SessionRow

_TS = datetime(2026, 8, 30, tzinfo=UTC)

#: The distinctive clause of the advice this file replaced. Matched as a
#: substring of the remediation the tool RETURNS, never over the module source
#: — the docstring above has to quote the old wording to explain the fix, and a
#: source scan would fail on its own explanation and then be "corrected" by
#: deleting it. Same trap ``test_mcp_unhealthy_cache.py`` documents for the
#: dead FTS5 branch.
_OLD_ADVICE = "so it can create or migrate the cache"


def _seed(store: Store, n: int = 3) -> None:
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
                tool_use_id=None, body=f"quadratic voting note {i}",
            )
        )
    store.upsert_entities(rows)


@pytest.fixture
def current_db(tmp_path):
    """A cache the reader is happy with. The control for everything below."""
    db = tmp_path / "cache.db"
    store = Store(db_path=db)
    store.migrate()
    _seed(store)
    store.close()
    return db


@pytest.fixture
def stale_db(current_db):
    """The same cache, re-stamped one schema version back.

    Written with a raw ``PRAGMA user_version`` rather than by reconstructing
    an old schema shape, because the refusal under test reads that pragma and
    nothing else — ``_ensure_cache_ready`` runs before any table is touched.
    A faithful v5 table layout would make the fixture slower, more brittle,
    and no more discriminating.

    The fixture must not be able to pass vacuously, so it proves both halves
    of its own premise: the stamp took, and the reader really does refuse it.
    """
    c = sqlite3.connect(current_db)
    c.execute(f"PRAGMA user_version = {SCHEMA_VERSION - 1};")
    c.commit()
    c.close()

    probe = Store(db_path=current_db, read_only=True)
    assert probe.schema_version() == SCHEMA_VERSION - 1
    probe.close()

    refusal = aggregator_query("voting", _store=_reader(current_db))
    assert refusal["ok"] is False, refusal
    return current_db


def _reader(db):
    return Store(db_path=db, read_only=True)


def _remediation(db) -> str:
    result = aggregator_query("voting", _store=_reader(db))
    assert result["ok"] is False, result
    return result["remediation"]


# --- what the message must now say ------------------------------------------


def test_the_remediation_names_both_versions_it_compared(stale_db):
    """THE REPRO. Pre-fix the remediation carried no numbers whatsoever.

    A reader who cannot see which two quantities disagreed cannot tell which
    of the two branches below applies to them, so the numbers are load-bearing
    and not decoration.
    """
    remediation = _remediation(stale_db)
    assert str(SCHEMA_VERSION) in remediation, remediation
    assert str(SCHEMA_VERSION - 1) in remediation, remediation


def test_the_remediation_no_longer_promises_that_running_the_writer_fixes_it(
    stale_db,
):
    """THE REPRO. The old sentence stated a fix that cannot work under skew."""
    assert _OLD_ADVICE not in _remediation(stale_db)


def test_the_remediation_names_build_skew_and_says_running_the_writer_cannot_fix_it(
    stale_db,
):
    """THE REPRO. The second branch did not exist in the old message at all.

    Both halves are asserted: that a writer older than the reader is named as
    a possible cause, and that the message says so plainly rather than leaving
    the caller to infer it from a silence.
    """
    remediation = _remediation(stale_db).lower()
    assert "older build" in remediation, remediation
    assert "exits 0" in remediation, remediation


def test_the_remediation_admits_it_did_not_measure_the_writer(stale_db):
    """A guess here is the failure this whole change exists to remove.

    The writer's schema version is only obtainable by executing the writer,
    which a read-only recall surface will not do. So the message says the
    number is unmeasured and hands over the command that measures it, rather
    than asserting skew as established fact.
    """
    remediation = _remediation(stale_db).lower()
    assert "not been measured" in remediation, remediation
    assert "aggregator status" in remediation, remediation
    assert "schema_version:" in remediation, remediation


def test_the_remediation_never_offers_to_downgrade_the_reader(stale_db):
    """Skew is resolved forward. The lagging side comes up; the leading side
    is never pulled back, and offering that as one arm of a choice would cost
    the reader a round trip to reject it."""
    remediation = _remediation(stale_db).lower()
    for banned in ("downgrade", "roll back", "rollback", "revert"):
        assert banned not in remediation, (banned, remediation)


# --- and it must be right about the writer it can see -----------------------


def test_a_nix_store_writer_is_named_as_a_pinned_immutable_build(
    stale_db, monkeypatch
):
    """The single most expensive wrong turn available to a reader here.

    When the writer resolves into ``/nix/store`` there is no in-place update
    path — the store path is immutable and pinned to one source revision — so
    pulling or rebuilding inside a checkout changes nothing about what the
    timer runs. A reader who does not know that will "fix" the checkout,
    observe no change, and be no further forward.
    """
    monkeypatch.setattr(
        mcp_mod.shutil, "which",
        lambda _name: "/nix/store/abc123-aggregator-0.0.1/bin/aggregator",
    )
    remediation = _remediation(stale_db)
    assert "/nix/store/abc123-aggregator-0.0.1/bin/aggregator" in remediation
    assert "immutable" in remediation.lower(), remediation


def test_no_writer_on_path_is_not_reported_as_proof_there_is_none(
    stale_db, monkeypatch
):
    """An MCP server inherits the environment of whatever launched it, which
    is almost never the user's login shell. "Not on PATH" here is a fact about
    this process and nothing more, and the message may not inflate it into a
    claim about the machine."""
    monkeypatch.setattr(mcp_mod.shutil, "which", lambda _name: None)
    remediation = _remediation(stale_db)
    assert "not proof" in remediation.lower(), remediation


def test_the_readers_own_entry_point_is_never_passed_off_as_the_writer(
    stale_db, monkeypatch
):
    """The trap this branch exists for, and it is the LIVE configuration.

    The MCP server runs as ``uv run --directory <checkout> aggregator-mcp``,
    which puts ``<checkout>/.venv/bin`` at the head of PATH. So the writer
    lookup resolves to the checkout's own CLI — the reader's build, at the
    reader's schema version — while the executable that actually maintains the
    cache is the one in the ingest unit's ``ExecStart``. Naming the first as
    "the writer" would reintroduce the confident-wrong-diagnosis bug in a new
    place, one round of fixing later.
    """
    monkeypatch.setattr(
        mcp_mod.shutil, "which",
        lambda _name: os.path.join(mcp_mod._READER_TREE, ".venv/bin/aggregator"),
    )
    remediation = _remediation(stale_db)
    assert "own entry point" in remediation.lower(), remediation
    assert "ExecStart" in remediation, remediation
    # And it must not be dressed up as a Nix deployment problem instead.
    assert "immutable" not in remediation.lower(), remediation


def test_a_sibling_directory_is_not_mistaken_for_the_reader_tree():
    """Containment is over path components. A checkout parked next to this one
    under a longer name shares a character prefix and nothing else, and calling
    it "the reader's own tree" would suppress the correct advice for it."""
    assert mcp_mod._is_inside("/srv/aggregator/bin/x", "/srv/aggregator")
    assert not mcp_mod._is_inside("/srv/aggregator-old/bin/x", "/srv/aggregator")


def test_a_writer_outside_the_nix_store_gets_the_ordinary_reinstall_advice(
    stale_db, monkeypatch
):
    monkeypatch.setattr(
        mcp_mod.shutil, "which", lambda _name: "/home/u/.venv/bin/aggregator"
    )
    remediation = _remediation(stale_db)
    assert "/home/u/.venv/bin/aggregator" in remediation
    assert "immutable" not in remediation.lower(), remediation


def test_a_path_lookup_that_raises_still_produces_a_refusal(
    stale_db, monkeypatch
):
    """This helper exists to make an error message better. It must never be
    the reason the error message fails to arrive — an unreadable PATH entry
    turning a structured refusal into a traceback would be a strictly worse
    bug than the one being fixed."""

    def _boom(_name):
        raise OSError("PATH entry is on a dead NFS mount")

    monkeypatch.setattr(mcp_mod.shutil, "which", _boom)
    result = aggregator_query("voting", _store=_reader(stale_db))
    assert result["ok"] is False
    assert result["remediation"]


# --- the other surface, and the thing that must not change ------------------


def test_capabilities_refuses_a_stale_cache_the_same_way(stale_db):
    """Two tools, one diagnosis. ``aggregator_capabilities`` runs the same
    check, and a caller that reached it first must not get the old advice."""
    result = aggregator_capabilities(_store=_reader(stale_db))
    assert result["ok"] is False
    assert _OLD_ADVICE not in result["remediation"]
    assert str(SCHEMA_VERSION - 1) in result["remediation"]


def test_a_current_cache_is_still_answered_and_not_diagnosed(current_db):
    """Proves the refusal above is caused by the stamp and by nothing else in
    the fixture — the same file, unstamped, answers normally."""
    result = aggregator_query("voting", _store=_reader(current_db))
    assert result["ok"] is True, result
    assert result["total"] > 0
