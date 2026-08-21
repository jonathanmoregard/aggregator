"""``scripts/rag_rollout_smoke.py`` must measure the path that ships.

THE SCRIPT IS NOT A GATE AND THIS FILE DOES NOT MAKE IT ONE. Its own docstring
is explicit: it needs a multi-GB copy of a real cache to say anything, so it
can never be a pytest test. What this file pins is narrower and cheap — that
the ONE helper standing between a raw query string and ``MATCH`` is the shipped
one. Nothing else here touches the script.

WHY IT EARNS A TEST AT ALL. The script produces rollout facts with a shelf
life: "34% of real queries return ok:false" came out of it, and that number is
what put FTS5 sanitization on the critical path. It bound raw user text
straight to ``MATCH``, so after b4eab9b whitelisted every string that reaches
``MATCH`` in ``aggregator.core.store``, the script kept measuring the RAW error
rate — a number about a code path that no longer exists. Re-run today it would
report the same ~34% and someone would conclude the fix never landed, or worse,
size a future decision on it. Nothing in CI referenced the file, which is
exactly why the drift could go unnoticed; one import here is the cheapest
possible fix for that.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

from aggregator.core import store as store_mod

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rag_rollout_smoke.py"


@pytest.fixture(scope="module")
def smoke():
    """Import the script by path. It is not a package, and must not become one."""
    spec = importlib.util.spec_from_file_location("_rag_rollout_smoke", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        sys.modules.pop(spec.name, None)


@pytest.fixture
def conn():
    """The two tables ``_fts_ranked`` joins, and nothing else."""
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(
        """
        CREATE TABLE observations (obs_id TEXT PRIMARY KEY, body TEXT);
        CREATE VIRTUAL TABLE obs_fts USING fts5(
            body, content='observations', content_rowid='rowid',
            tokenize='unicode61 remove_diacritics 2'
        );
        """
    )
    c.execute(
        "INSERT INTO observations(rowid, obs_id, body) VALUES (1, ?, ?)",
        ("o1", "the power-on self test writes to the console"),
    )
    c.execute("INSERT INTO obs_fts(rowid, body) SELECT rowid, body FROM observations")
    c.commit()
    try:
        yield c
    finally:
        c.close()


def test_the_script_uses_the_shipped_sanitizer_and_not_a_copy(
    smoke, conn, monkeypatch
):
    """A second implementation would drift, silently, and be believed.

    Asserted by substitution rather than by identity: the script imports
    ``aggregator`` names inside the call (a module-level import would drag
    spaCy in through ``core.scrub``), so what is checked is that the shipped
    function is the one doing the work at call time.
    """
    seen: list[str] = []
    real = store_mod.fts5_match_query

    def spy(text):
        seen.append(text)
        return real(text)

    monkeypatch.setattr(store_mod, "fts5_match_query", spy)
    smoke.fts_obs_ranked(conn, "power-on", 10)

    assert seen == ["power-on"], (
        "scripts/rag_rollout_smoke.py must route user text through "
        "aggregator.core.store.fts5_match_query. A local copy is how a "
        "measurement script starts reporting on a path that no longer ships."
    )


def test_a_query_that_used_to_be_a_syntax_error_now_returns_its_rows(smoke, conn):
    """THE REPRO. ``power-on`` was 'no such column: on' when bound raw."""
    hits, error = smoke.fts_obs_ranked(conn, "power-on", 10)

    assert error is None, (
        "the smoke script still binds raw user text to MATCH, so it measures "
        f"the raw error rate rather than the shipped sanitized path: {error!r}"
    )
    assert hits == ["o1"]


def test_an_all_punctuation_query_matches_nothing_rather_than_everything(
    smoke, conn
):
    """``fts5_match_query`` returns '' for text with no word characters, and
    the store then runs NO MATCH at all. ``MATCH ''`` is a different question,
    and an unconstrained MATCH returning the whole corpus is the dangerous way
    to get this wrong — in a script whose whole output is hit counts."""
    assert smoke.fts_obs_ranked(conn, "!!!", 10) == ([], None)


def test_a_genuinely_broken_index_still_comes_back_as_an_error(smoke, conn):
    """The error channel is kept, and now means what it says. After the
    rewrite a failure cannot be a syntax error, so it is a lock, a corrupt
    index or a missing table — facts the analysis must keep apart from
    'this query has no matches'."""
    conn.execute("DROP TABLE obs_fts")

    hits, error = smoke.fts_obs_ranked(conn, "power-on", 10)

    assert hits == []
    assert error is not None and "obs_fts" in error
