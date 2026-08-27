"""Criterion D, half one — WHAT COUNTS AS ONE CONJUNCT.

A query is a conjunction: every conjunct must match. The defect this file pins
is not which ROW has to satisfy them (that is ``scope:``, and the observation
was already the unit) — it is that a caller's quoted phrase was silently taken
apart into independent words before FTS5 ever saw it, so a precise question
became an imprecise one with no way to tell.

Measured on the live corpus, read-only, before this landed:

* ``"low usage cap"``       1 row  ->  156 rows, the right one at rank 118
* ``"terraform state lock"`` 0 rows ->  1,845 rows (a phrase the corpus does
  not contain anywhere; only the adjacency was selective)
* ``"PR link" "status report"`` 0 rows -> 84 rows, none of them the answer

The last one is mission acceptance test 2, whose stated WORST outcome is
"silently returning irrelevant rows". Restoring the phrase turns it back into
an honest abstention.

The safety property from ``test_store_fts_sanitize.py`` is unchanged and is
re-asserted here in its widened form: still nothing but double-quoted ``\\w+``
runs, now allowing single spaces INSIDE one pair of quotes.
"""
from __future__ import annotations

import re
import sqlite3

import pytest

from aggregator.core.store import fts5_match_conjuncts, fts5_match_query

# --- the split --------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        # A balanced pair is ONE conjunct.
        ('"low usage cap"', ['"low usage cap"']),
        ('"PR link" "status report"', ['"PR link"', '"status report"']),
        ('"Meet more, nag less"', ['"Meet more nag less"']),
        # Quoted and unquoted mix; each unquoted word is its own conjunct.
        ('"branch protection" force merge',
         ['"branch protection"', '"force"', '"merge"']),
        # No quotes at all: exactly as before.
        ("power-on", ['"power"', '"on"']),
        ("hand back control", ['"hand"', '"back"', '"control"']),
        # Nothing to match.
        ("!!! ... @#$", []),
        ("", []),
    ],
)
def test_a_balanced_quoted_run_is_one_conjunct(query, expected):
    assert fts5_match_conjuncts(query) == expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ('"unbalanced quote', ['"unbalanced"', '"quote"']),
        ('unbalanced "', ['"unbalanced"']),
        ('quote " in the middle',
         ['"quote"', '"in"', '"the"', '"middle"']),
        ('"""', []),
    ],
)
def test_an_unbalanced_quote_groups_nothing(query, expected):
    """Guessing where an unclosed phrase ends invents a query nobody wrote, so
    the whole string falls back to the per-word split it had before."""
    assert fts5_match_conjuncts(query) == expected


def test_the_joined_form_is_still_the_match_expression():
    assert fts5_match_query('"PR link" "status report"') == (
        '"PR link" "status report"'
    )
    assert fts5_match_query("power-on") == '"power" "on"'


# --- the safety property, widened by exactly one space --------------------

_SAFE_SHAPE = re.compile(r'^("\w+( \w+)*"( "\w+( \w+)*")*)?$')

HOSTILE = (
    '"body:secret"',
    '"NEAR(a b, 3)"',
    '"prefix*"',
    '"^anchored"',
    '"a AND b OR NOT c"',
    '"-negated"',
    '"a" OR "b"',
    'a"b"c',
    '"\\backslash"',
    '"\x00null"',
    '"emoji 🙂 tail"',
    '""""',
    '" " " "',
    '"a""b"',
)


@pytest.mark.parametrize("hostile", HOSTILE)
def test_nothing_but_quoted_word_phrases_can_still_reach_match(hostile):
    got = fts5_match_query(hostile)
    assert _SAFE_SHAPE.match(got), got
    # And no input quote survives: every ``"`` in the output was written here.
    assert '""' not in got


@pytest.mark.parametrize("hostile", HOSTILE)
def test_hostile_input_still_does_not_raise_against_real_fts5(hostile):
    db = sqlite3.connect(":memory:")
    db.execute("CREATE VIRTUAL TABLE t USING fts5(body)")
    db.execute("INSERT INTO t(body) VALUES ('secret body text')")
    rewritten = fts5_match_query(hostile)
    if not rewritten:
        return
    db.execute("SELECT rowid FROM t WHERE t MATCH ?", (rewritten,)).fetchall()


def test_a_quoted_operator_inside_a_phrase_is_still_a_literal():
    db = sqlite3.connect(":memory:")
    db.execute("CREATE VIRTUAL TABLE t USING fts5(body)")
    db.execute("INSERT INTO t(body) VALUES ('a b')")
    rows = db.execute(
        "SELECT rowid FROM t WHERE t MATCH ?", (fts5_match_query('"a OR b"'),)
    ).fetchall()
    assert rows == [], "OR was honoured as an operator inside a phrase"


# --- what it does to real rows ---------------------------------------------


@pytest.fixture
def fts():
    db = sqlite3.connect(":memory:")
    db.execute("CREATE VIRTUAL TABLE t USING fts5(body)")
    db.executemany(
        "INSERT INTO t(body) VALUES (?)",
        [
            ("disable RSI, I have low usage cap. We need to test its impact.",),
            ("the state of the lock is low, and terraform is not used here",),
            ("usage went up; the cap on the state lock was low all week",),
            ("wdyt? Meet more, nag less. Klaffar finds the best time.",),
            ("meet more, plan less. Omit the reminder and nag nobody.",),
        ],
    )
    return db


def _hits(db, text):
    return [
        r[0]
        for r in db.execute(
            "SELECT rowid FROM t WHERE t MATCH ? ORDER BY rowid",
            (fts5_match_query(text),),
        )
    ]


def test_the_phrase_selects_the_one_row_the_bag_of_words_could_not(fts):
    assert _hits(fts, '"low usage cap"') == [1]
    # The same words, unquoted, are the imprecise question — and it is the
    # caller's to ask, not the rewrite's to substitute.
    assert _hits(fts, "low usage cap") == [1, 3]


def test_a_phrase_the_corpus_does_not_contain_abstains(fts):
    """The over-eagerness canary: three ordinary words, no adjacency."""
    assert _hits(fts, '"terraform state lock"') == []
    assert _hits(fts, "terraform state lock") == [2]


def test_adjacency_tells_apart_two_drafts_of_one_slogan(fts):
    """Which slogan was chosen is the whole question; the word bag merges the
    accepted line with the rejected one."""
    assert _hits(fts, '"Meet more, nag less"') == [4]
    assert _hits(fts, "Meet more, nag less") == [4, 5]
