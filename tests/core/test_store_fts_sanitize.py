"""Criterion B — every string that reaches FTS5 ``MATCH`` is a whitelist.

WHY A WHITELIST AND NOT AN ESCAPE. FTS5 query syntax gives structural meaning
to ``-``, ``:``, ``*``, ``^``, ``(``, ``)``, ``"`` and the barewords ``AND``,
``OR``, ``NOT``, ``NEAR`` — *outside* a quoted string. Inside a quoted string
the only meta-character is ``"``. So the safe move is not "escape the
characters we know bite"; it is "keep only characters that cannot bite, and
quote them anyway". A blacklist has to be right about every character SQLite
will ever add meaning to; the SQLite docs explicitly warn that input which
raises a syntax error today "may be interpreted differently by some future
version of FTS5". A whitelist is right by construction.

Measured before this landed: 25 of the 86 frozen golden queries — 29% —
raised ``sqlite3.OperationalError`` against a fresh index. Not a hypothetical
class: ``power-on`` ("no such column: on"), ``cache.db`` ("syntax error near
'.'"), ``#178``, ``nDCG@10``, ``sqlite-vec``.

THE PROPERTY THAT MAKES THIS SAFE TO SHIP is not "the errors stopped". It is
that queries which already worked return the SAME ROWS WITH THE SAME BM25
SCORES — a rewrite that silently reranks every working query has broken the
thing it claims to repair. ``test_previously_working_queries_are_untouched``
asserts exactly that, comparing ``(rowid, bm25)`` pairs from the original
string against the rewritten one over the whole golden set.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from datetime import UTC, datetime

import pytest

from aggregator.core.dsl import parse
from aggregator.core.store import Store, fts5_match_query
from aggregator.evals.golden import load_golden_queries
from aggregator.sources.base import ObservationRow, QueryAST, Record, SessionRow

# Every row of the research report's failure table, verbatim.
REPORTED_FAILURES = (
    "power-on",
    "on/off toggle",
    "C++ 17",
    "wifi-6E",
    '"unbalanced quote',
)


# --- the rewrite itself -------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("power-on", '"power" "on"'),
        ("on/off toggle", '"on" "off" "toggle"'),
        ("C++ 17", '"C" "17"'),
        ("wifi-6E", '"wifi" "6E"'),
        ('"unbalanced quote', '"unbalanced" "quote"'),
        ("cache.db", '"cache" "db"'),
        ("#178", '"178"'),
        ("nDCG@10", '"nDCG" "10"'),
        ("sqlite-vec", '"sqlite" "vec"'),
        ("aggregator-ingest.service", '"aggregator" "ingest" "service"'),
    ],
)
def test_reported_failure_shapes_become_quoted_phrases(query, expected):
    assert fts5_match_query(query) == expected


def test_an_all_punctuation_query_rewrites_to_nothing():
    """Not to ``""`` and not to a wildcard — to the empty string, which the
    callers below turn into "no lexical matches"."""
    assert fts5_match_query("!!! ... @#$ ***") == ""
    assert fts5_match_query("") == ""
    assert fts5_match_query("   ") == ""


def test_underscore_identifiers_stay_whole():
    """``ERR_TLS_CERT`` is a legal FTS5 bareword TODAY, and this corpus is full
    of them. Splitting it into three phrases changes both the rows and the
    bm25 scores of a query that already worked — see
    ``test_previously_working_queries_are_untouched``, which is the test that
    forced ``_`` into the whitelist."""
    assert fts5_match_query("ERR_TLS_CERT") == '"ERR_TLS_CERT"'
    assert fts5_match_query("root_session_id") == '"root_session_id"'


def test_non_ascii_letters_and_digits_survive():
    assert fts5_match_query("café 日本語 Ω") == '"café" "日本語" "Ω"'


_SAFE_SHAPE = re.compile(r'^("\w+"( "\w+")*)?$')

HOSTILE = (
    'body:secret',
    'NEAR(a b, 3)',
    'prefix*',
    '^anchored',
    'a AND b OR NOT c',
    '-negated',
    'quote " in the middle',
    'unbalanced "',
    '(grouped)',
    'col : val',
    '{body} : x',
    'a"b"c',
    '\\backslash',
    '\x00null',
    'emoji 🙂 tail',
    '"""',
)


@pytest.mark.parametrize("hostile", HOSTILE)
def test_nothing_but_quoted_word_phrases_can_reach_match(hostile):
    """The structural claim, asserted on the STRING rather than on a result:
    the output is a space-joined run of double-quoted ``\\w+`` runs, or empty.
    No operator, column filter, NEAR, prefix ``*`` or stray quote can survive
    that shape, whatever FTS5 decides those characters mean next."""
    assert _SAFE_SHAPE.match(fts5_match_query(hostile)), fts5_match_query(hostile)


@pytest.mark.parametrize("hostile", HOSTILE)
def test_hostile_input_does_not_raise_against_a_real_fts5_table(hostile):
    db = sqlite3.connect(":memory:")
    db.execute("CREATE VIRTUAL TABLE t USING fts5(body)")
    db.execute("INSERT INTO t(body) VALUES ('secret body text')")
    rewritten = fts5_match_query(hostile)
    if not rewritten:
        return
    db.execute("SELECT rowid FROM t WHERE t MATCH ?", (rewritten,)).fetchall()


def test_quoted_operator_words_are_literals_not_operators():
    """``AND``/``OR``/``NOT``/``NEAR`` are letters, so they pass the whitelist —
    and that is fine, because quoting demotes them to ordinary tokens."""
    assert fts5_match_query("a OR b") == '"a" "OR" "b"'
    db = sqlite3.connect(":memory:")
    db.execute("CREATE VIRTUAL TABLE t USING fts5(body)")
    db.execute("INSERT INTO t(body) VALUES ('a b')")
    rows = db.execute(
        "SELECT rowid FROM t WHERE t MATCH ?", (fts5_match_query("a OR b"),)
    ).fetchall()
    assert rows == [], "OR was honoured as an operator, not a literal token"


# --- the identity property ----------------------------------------------------


def _bm25_rows(db, match_expr):
    return [
        (r[0], round(r[1], 6))
        for r in db.execute(
            "SELECT rowid, bm25(t) FROM t WHERE t MATCH ? ORDER BY rowid",
            (match_expr,),
        )
    ]


def test_previously_working_queries_are_untouched():
    """Same rows, same bm25 to 6 decimal places, for every golden query that
    did not raise before the rewrite existed.

    The corpus is built FROM the golden queries so the comparison is not
    vacuous — a run where both sides return nothing proves nothing, so the
    number of queries that actually matched something is asserted too.
    """
    db = sqlite3.connect(":memory:")
    db.execute("CREATE VIRTUAL TABLE t USING fts5(body)")
    golden = load_golden_queries()
    for q in golden:
        db.execute("INSERT INTO t(body) VALUES (?)", (q.query,))
        db.execute(
            "INSERT INTO t(body) VALUES (?)",
            (f"prose around {q.query} and some filler words to vary the length",),
        )

    compared = 0
    non_empty = 0
    for q in golden:
        text = parse(q.query).text
        if not text:
            continue
        try:
            before = _bm25_rows(db, text)
        except sqlite3.OperationalError:
            continue  # raised before the fix; nothing to preserve
        compared += 1
        after = _bm25_rows(db, fts5_match_query(text))
        assert after == before, f"{q.id} ({text!r}) moved: {before} -> {after}"
        if before:
            non_empty += 1

    assert compared >= 40, f"only {compared} golden queries worked before the fix"
    assert non_empty >= 20, f"only {non_empty} comparisons had any rows at all"


# --- the store honours it -----------------------------------------------------


def _session(session_id):
    ts = datetime(2026, 7, 1, 8, 0, tzinfo=UTC)
    return SessionRow(
        session_id=session_id,
        root_session_id=session_id,
        parent_session_id=None,
        kind="session",
        agent_id=None,
        agent_type=None,
        spawned_by_tool_use_id=None,
        cwd="/x",
        git_branch="main",
        first_ts=ts,
        last_ts=ts,
        jsonl_path=f"/tmp/{session_id}.jsonl",
    )


def _obs(obs_id, session_id, body):
    return ObservationRow(
        obs_id=obs_id,
        session_id=session_id,
        root_session_id=session_id,
        parent_obs_id=None,
        type="user",
        ts=datetime(2026, 7, 1, 8, 0, tzinfo=UTC),
        model=None,
        input_tokens=None,
        output_tokens=None,
        tool_name=None,
        tool_use_id=None,
        body=body,
    )


def _record(stable_id, subject, body):
    ts = datetime(2026, 7, 1, 8, 0, tzinfo=UTC)
    return Record(
        stable_id=stable_id,
        source="github",
        subject=subject,
        body=body,
        tags=[],
        created_at=ts,
        updated_at=ts,
    )


@pytest.fixture
def seeded(tmp_path):
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    entities = []
    for i, failing in enumerate(REPORTED_FAILURES):
        sid = f"s{i}"
        entities.append(_session(sid))
        entities.append(_obs(f"o{i}", sid, f"the note says {failing} in context"))
    s.upsert_entities(entities)
    s.upsert(
        [
            _record(f"github:{i}", "github", f"issue text {failing}")
            for i, failing in enumerate(REPORTED_FAILURES)
        ]
    )
    return s


@pytest.mark.parametrize("failing", REPORTED_FAILURES)
def test_observations_path_answers_the_reported_failures(seeded, failing):
    rows = seeded.query_observations(parse(failing))
    assert rows, f"{failing!r} still returns nothing"


@pytest.mark.parametrize("failing", REPORTED_FAILURES)
def test_records_path_answers_the_reported_failures(seeded, failing):
    text = parse(failing).text
    assert seeded.query(QueryAST(source="github", text=text))


@pytest.mark.parametrize("failing", REPORTED_FAILURES)
def test_probe_fts_no_longer_rejects_the_reported_failures(seeded, failing):
    seeded.probe_fts(parse(failing).text)


@pytest.mark.parametrize("failing", REPORTED_FAILURES)
def test_session_cards_surface_for_the_reported_failures(seeded, failing):
    assert seeded.query_sessions(parse(failing))
    assert seeded.count_sessions(parse(failing)) >= 1


def test_count_observations_agrees_with_the_page(seeded):
    ast = parse("power-on")
    assert seeded.count_observations(ast) == len(seeded.query_observations(ast))


def test_an_all_punctuation_query_returns_nothing_rather_than_everything(seeded):
    """The dangerous failure mode of "just drop the bad characters" is an empty
    MATCH that matches the whole corpus."""
    ast = QueryAST(text="!!! ...")
    assert seeded.query_observations(ast) == []
    assert seeded.count_observations(ast) == 0
    assert seeded._fts_obs_ids("!!! ...") == []
    assert seeded._fts_ids("!!! ...") == set()
    assert seeded._fts_hit_scope("!!! ...") == (set(), set())
    assert seeded._fts_root_session_ids("!!! ...") == []
    seeded.probe_fts("!!! ...")


def test_the_store_never_hands_match_an_empty_string(seeded, monkeypatch):
    """``MATCH ''`` is not merely useless, it is a different question. Assert
    the guard is in the store, not in the caller."""
    seen: list[str] = []
    real = Store._c

    class Spy:
        def __init__(self, inner):
            self._inner = inner

        def execute(self, sql, params=()):
            if "MATCH" in sql and params:
                seen.extend(p for p in params if isinstance(p, str))
            return self._inner.execute(sql, params)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    monkeypatch.setattr(Store, "_c", lambda self: Spy(real(self)))
    seeded.query_observations(QueryAST(text="!!! ..."))
    assert "" not in seen


# --- the vector arm keeps the original string ---------------------------------


def test_the_vector_arm_is_handed_the_unsanitised_query(seeded, monkeypatch):
    """Sanitisation is a property of the LEXICAL arm only. ``power-on`` carries
    meaning to an embedding model that ``"power" "on"`` does not, so the fix
    must not leak upstream into the text the embedder sees."""
    import aggregator.mcp as mcp_mod

    asked: list[str] = []

    class SpyEmbedder:
        def embed_query(self, text):
            asked.append(text)
            return [0.0] * 768

    monkeypatch.setattr(mcp_mod, "_get_embedder", lambda: SpyEmbedder())
    mcp_mod.aggregator_query("power-on", _store=seeded)
    assert asked == [] or asked == ["power-on"], asked
    # And directly: the embedding text is never rewritten on the way in.
    assert mcp_mod._query_embedding("power-on") is not None
    assert asked[-1] == "power-on"


# --- defence in depth: the fused path degrades to semantic-only ---------------


def test_a_raising_fts_arm_degrades_to_the_vector_arm_alone(seeded, monkeypatch):
    """If MATCH raises anyway — a lock, a corrupt index, a future FTS5 — the
    hybrid query answers from the vector arm rather than erroring at the agent.

    AND IT REPORTS THE ARM AS UNAVAILABLE, NOT AS UNMATCHED. This used to come
    back as a plain empty id set, which ``_note_confidence`` read and turned
    into "the keyword arm matched none of these rows" — a claim about the
    DOCUMENTS manufactured out of a failure of the INDEX. An agent reading that
    cannot tell a corpus with no keyword match from an FTS5 table that fell
    over, which is the empty-result-looks-like-success failure this project
    bans by name.

    ``LEXICAL_ARM_UNAVAILABLE`` is equal to ``frozenset()`` and behaves as one
    everywhere, so nothing downstream has to special-case it to stay correct;
    only ``is`` tells it apart, and only the sentence changes. It cannot be
    spelled ``None`` — that already means the FTS5-only route, where every row
    IS a keyword match, so reusing it would flip a dropped arm to "fully
    corroborated".
    """
    import aggregator.mcp as mcp_mod

    def boom(_text):
        raise sqlite3.OperationalError("simulated FTS5 failure")

    monkeypatch.setattr(seeded, "_fts_obs_ids", boom)
    monkeypatch.setattr(mcp_mod, "_widen_chunk_ids", lambda ids: list(ids))
    scope, vec_hits, lexical_ids = mcp_mod._fused_id_scope(
        seeded, "observations", "power-on", object(), frozen=["o0", "o1"]
    )
    assert scope == frozenset({"o0", "o1"})
    assert vec_hits == ["o0", "o1"]

    # Identity, because equality cannot carry the distinction — that is the
    # whole design — and this assertion is what goes red if the state is ever
    # collapsed back into an ordinary empty set.
    assert lexical_ids is mcp_mod.LEXICAL_ARM_UNAVAILABLE
    assert mcp_mod._lexical_arm_failed(lexical_ids) is True
    # ...and it really is empty for everything that only wants the ids, so the
    # fusion above and every set operation below it stay correct untouched.
    assert lexical_ids == frozenset()
    assert not lexical_ids
    assert "o0" not in lexical_ids
    assert mcp_mod._lexical_contributed(lexical_ids) is False

    # A HEALTHY ARM THAT MATCHED NOTHING IS THE OTHER STATE, and the two must
    # not be the same object — otherwise the sentence above is emitted for a
    # corpus that was searched correctly and simply had no hit.
    monkeypatch.setattr(seeded, "_fts_obs_ids", lambda _text: [])
    _scope, _hits, empty = mcp_mod._fused_id_scope(
        seeded, "observations", "power-on", object(), frozen=["o0", "o1"]
    )
    assert empty == lexical_ids
    assert empty is not mcp_mod.LEXICAL_ARM_UNAVAILABLE
    assert mcp_mod._lexical_arm_failed(empty) is False


def test_a_raising_fts_arm_is_logged_loudly(seeded, monkeypatch, caplog):
    """Degrading silently is the banned failure — an empty lexical arm looks
    exactly like a corpus with no keyword matches."""
    real = Store._c

    class BoomConn:
        def __init__(self, inner):
            self._inner = inner

        def execute(self, sql, params=()):
            if "obs_fts MATCH" in sql:
                raise sqlite3.OperationalError("simulated FTS5 failure")
            return self._inner.execute(sql, params)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    monkeypatch.setattr(Store, "_c", lambda self: BoomConn(real(self)))
    with (
        caplog.at_level(logging.ERROR, logger="aggregator.core.store"),
        pytest.raises(sqlite3.OperationalError),
    ):
        seeded._fts_obs_ids("power-on")
    assert any(r.levelno >= logging.ERROR for r in caplog.records), caplog.text
    assert "power-on" in caplog.text


def test_the_mcp_health_probe_constant_still_probes_something():
    """``probe_fts`` skips the MATCH when the text has no word characters, so
    the constant ``_fts_probe_is_healthy`` passes must keep some. Left
    unpinned, changing that constant to punctuation would turn the "is the
    cache readable at all" check into an unconditional yes."""
    from aggregator.mcp import _FTS_HEALTH_PROBE

    assert fts5_match_query(_FTS_HEALTH_PROBE)
