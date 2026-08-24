"""``scripts/rag_rollout_smoke.py`` must measure the path that ships.

THE SCRIPT IS NOT A GATE AND THIS FILE DOES NOT MAKE IT ONE. Its own docstring
is explicit: it needs a multi-GB copy of a real cache to say anything, so it
can never be a pytest test. What this file pins is narrower and cheap — that
BOTH helpers standing between a raw query string and ``MATCH`` are the shipped
one. Nothing else here touches the script.

THERE ARE TWO, AND AN EARLIER VERSION OF THIS DOCSTRING SAID THERE WAS ONE.
``_fts_ranked`` (via ``fts_obs_ranked``/``fts_rec_ranked``) was fixed and
covered here; ``store_fts_all`` was not, and its only caller swallowed the
resulting ``OperationalError`` into an empty set, so six of seven realistic
queries failed against a real snapshot with nothing going red. The claim was
load-bearing: it is what a later reader consults to decide whether the class
of defect is closed. ``tests/test_fts5_match_site_enumeration.py`` now derives
the list of helpers mechanically instead of asserting it in prose.

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


# --- the SECOND helper: store_fts_all ----------------------------------------
#
# It opens its own connection from a path rather than taking one, so it needs a
# file, and it queries BOTH ontologies, so it needs both tables.


@pytest.fixture
def db_file(tmp_path):
    """A two-ontology cache small enough to build in a millisecond."""
    path = tmp_path / "cache.db"
    c = sqlite3.connect(str(path))
    c.executescript(
        """
        CREATE TABLE observations (obs_id TEXT PRIMARY KEY, body TEXT);
        CREATE VIRTUAL TABLE obs_fts USING fts5(
            body, content='observations', content_rowid='rowid',
            tokenize='unicode61 remove_diacritics 2'
        );
        CREATE VIRTUAL TABLE records_fts USING fts5(
            stable_id UNINDEXED, source UNINDEXED, subject, body, tags,
            tokenize='unicode61 remove_diacritics 2'
        );
        """
    )
    c.execute(
        "INSERT INTO observations(rowid, obs_id, body) VALUES (1, ?, ?)",
        ("o1", "the power-on self test writes to cache.db"),
    )
    c.execute("INSERT INTO obs_fts(rowid, body) SELECT rowid, body FROM observations")
    c.execute(
        "INSERT INTO records_fts(stable_id, source, subject, body, tags) "
        "VALUES (?, ?, ?, ?, ?)",
        ("gh:1", "github", "power-on", "sqlite-vec and C++ 17 notes", ""),
    )
    c.commit()
    c.close()
    return path


@pytest.mark.parametrize(
    "query",
    ["power-on", "#42", "cache.db", "C++ 17", "nixos-rebuild", "sqlite-vec"],
)
def test_store_fts_all_answers_the_queries_that_used_to_raise(smoke, db_file, query):
    """THE REPRO, at the exact shape measured against the 509k-row snapshot.

    Driven through the shipped function, six of these seven raised — ``power-on``
    'no such column: on', ``#42`` 'fts5: syntax error near "#"', and so on — and
    only ``quadratic voting`` survived. Every one of them is an ordinary thing a
    person types.
    """
    smoke.store_fts_all(db_file, query)


def test_store_fts_all_uses_the_shipped_sanitizer_and_not_a_copy(
    smoke, db_file, monkeypatch
):
    """Same substitution check as its sibling, for the same reason: a local
    copy of the whitelist is how the two helpers drift apart again."""
    seen: list[str] = []
    real = store_mod.fts5_match_query

    def spy(text):
        seen.append(text)
        return real(text)

    monkeypatch.setattr(store_mod, "fts5_match_query", spy)
    smoke.store_fts_all(db_file, "power-on")

    assert seen == ["power-on"], seen


def test_store_fts_all_reaches_both_ontologies(smoke, db_file):
    """Uncapped across obs AND records — the exclusion it feeds is "can FTS5
    reach this document at all", and missing one ontology understates it."""
    assert set(smoke.store_fts_all(db_file, "power-on")) == {"o1", "gh:1"}


def test_store_fts_all_matches_nothing_for_an_all_punctuation_query(smoke, db_file):
    """Not everything. An unconstrained MATCH would mark every document
    'already reachable by FTS5' and empty the vector-only population from the
    other direction."""
    assert smoke.store_fts_all(db_file, "!!! ...") == []


def test_store_fts_all_still_raises_when_the_index_is_broken(smoke, db_file):
    """Fail loudly. After the rewrite a failure cannot be a syntax error, so it
    is a lock or a corrupt index — a fact about the CACHE, and one that must
    not reach the caller disguised as a query with no matches."""
    c = sqlite3.connect(str(db_file))
    c.execute("DROP TABLE obs_fts")
    c.commit()
    c.close()

    with pytest.raises(sqlite3.OperationalError):
        smoke.store_fts_all(db_file, "power-on")
