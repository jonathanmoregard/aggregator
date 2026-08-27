"""SQLite + FTS5 store — Schema B (Langfuse-derived), two ontologies.

The store holds two intentionally-distinct ontologies side by side:

1. ``records`` + ``records_fts`` — **units-of-work** ontology (Gooen
   stable-ID discipline; one row per PR / issue / eventually Gmail thread /
   Calendar event). Fields: ``subject``, ``body``, ``tags``, ``created_at``,
   ``updated_at``, per-source ``extra`` JSON blob. Query keys:
   ``source:github``, ``state:``, ``check:``, ``mergeable:``, ``author:``,
   ``tag:``. See ``sources/github.py``.

2. ``sessions`` (Langfuse "trace") + ``observations`` (Langfuse
   "observation") + ``obs_fts`` — **conversation-stream** ontology. One
   session per JSONL, one observation per message. Fields: ``kind``,
   ``root_session_id``, ``parent_session_id``, ``agent_type``,
   ``first_ts``, ``last_ts``, tokens, tool metadata. Query keys:
   ``source:sessions``, ``session:``, ``top:``, ``agent:``, ``type:``,
   ``active:``. See ``sources/sessions.py``.

The two schemas are NOT unified. A PR is not naturally a session; a session
is not naturally a unit of work. Attempting to squeeze either into the
other's shape loses meaningful structure (a session-shaped PR has no
observation stream; a records-shaped session collapses turn-by-turn detail
into an opaque body blob). ``aggregator/mcp.py::_route_mode`` implements
transparent routing over the split so DSL callers don't have to know which
table they're hitting; cross-ontology date/text queries fan out to both
tables (UNION mode).

Everything under a session root is an indexed equality on the denormalized
``observations.root_session_id`` — no recursion — the SOTA trick documented
in the research report §2. Same trick lets ``sessions.root_session_id``
group top-level + subagent streams under one query.

SQLite is a **derived index**; JSONLs / GitHub API responses are the source
of truth. v1→v2 migration = full rebuild (schema version bumped to 2 to
signal the break; ``Store.rebuild_all()`` drops every table — records +
sessions + observations + FTS shadows — and re-runs DDL). v2→v3 is additive
and migrates IN PLACE: ``migrate()`` adds ``sessions.origin`` via ALTER
TABLE with a ``'claude-code'`` default (chat-export sources land as
``origin='chatgpt'`` / ``'claude-web'``); no rebuild needed.

FTS5 external-content pattern per ``sqlite.org/fts5.html`` §external content:
one virtual table per base table with ``content=base_table content_rowid=rowid``,
plus AFTER INSERT/DELETE/UPDATE triggers to keep the index in sync without
duplicating body text.

Scrub-on-write (spec constraint 3, defense in depth: also runs pre-return at
MCP/CLI boundary). WAL + busy_timeout=5000 (Codex Phase 2 MEDIUM #2 —
concurrent-writer safety).

v3→v4 is additive and migrates IN PLACE. It adds:

* ``records.src_hash`` / ``observations.src_hash`` — a fingerprint of the row
  as the SOURCE produced it, so a re-observed row that did not change is
  recognised BEFORE it is scrubbed or written. See :func:`_src_hash`.
* ``ingest_state`` — the per-source high-water mark, in this database rather
  than in a sidecar file, so it can be advanced in the SAME TRANSACTION as the
  chunk it describes. Two artifacts updated by two writes cannot be made to
  agree under SIGTERM; a watermark that gets ahead of its data is silent,
  permanent loss. See ``imports/ingest_state.py`` for the policy on top.
* ``quarantine`` — the local analogue of a dead-letter queue. A record that
  cannot be written must not abort the run and must not be retried forever;
  it lands here with an attempt count and a terminal state.

Both ALTERs are metadata-only (``ADD COLUMN`` with no default), which matters:
the live database holds ~372k observations and a rewrite would be exactly the
kind of hours-long unattended operation this schema change exists to abolish.

v4→v5 is the RAG schema, and is additive on the same terms. It adds:

* ``observations.embedding_state`` / ``records.embedding_state`` — a NULL-vector
  watermark for the background embed worker. NULL means "not embedded yet", so
  every pre-v5 row arrives QUEUED FOR BACKFILL. Deliberately not backfilled to
  a done-looking value: that would mark the whole corpus embedded and leave the
  vector arm silently returning nothing.
* ``vec_observations`` / ``vec_records`` — sqlite-vec ``vec0`` virtual tables
  holding 768-dim MRL-truncated Qwen3 embeddings.

THE VECTOR ARM IS OPTIONAL AND THE STORE TREATS IT THAT WAY. ``sqlite-vec`` is
a loadable native extension; it can be absent, ABI-mismatched against this
SQLite, or blocked by a python built without ``enable_load_extension``. Every
read in this product goes through ``Store`` — including
``aggregator_search_memory``, which has no vector dependency at all — so
loading it unconditionally would turn an optional feature into a single point
of failure for recall. Instead: attempt the load, complain loudly exactly once
per process, set :attr:`Store.vector_available`, and keep FTS5 serving. Vector
reads then raise :class:`VectorIndexUnavailableError` by name rather than a
bare ``no such table: vec_observations``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from aggregator.core.provenance import MACHINE, MACHINE_VALUES
from aggregator.core.scrub import scrub
from aggregator.sources.base import (
    SCOPE_SESSION,
    ObservationRow,
    QueryAST,
    Record,
    SessionEntity,
    SessionRow,
)

log = logging.getLogger(__name__)

SCHEMA_VERSION = 6


class VectorIndexUnavailableError(RuntimeError):
    """Raised when a vector read is attempted and sqlite-vec did not load.

    Its own type, and raised by name, so a caller can tell "the vector arm is
    not installed here" apart from "the query was wrong" — which a bare
    ``sqlite3.OperationalError: no such table: vec_observations`` cannot.
    """


class VectorReindexNotConsentedError(RuntimeError):
    """A wholesale delete of the vector index was attempted without consent.

    THE THIRD PATH. Rounds 1-3 spent three iterations narrowing who may
    destroy computed vectors: the ambient ``AGGREGATOR_VECTOR_REINDEX``
    variable was deleted, replaced by a per-call ``allow_vector_reindex=``
    argument that only ``embed --reindex`` passes, behind a printed preview and
    a ``y`` on stdin. All three rounds guarded ``migrate()``.

    ``rebuild_all()`` ran ``_VEC_DROP_ALL`` unconditionally the entire time,
    and ``scripts/reingest_v2.py`` calls it as its first act. So the sentence
    "``migrate()`` is the only thing that may delete computed vectors
    wholesale" — written into three docstrings — was false, and every gate in
    front of it was a fence with a gap beside it.

    Its own type, rather than a bare ``RuntimeError``, because the two callers
    that can hit it want opposite things: a script wants to catch it and ask
    the operator, while a library caller wants it to propagate.
    """


class CacheUnavailableError(sqlite3.OperationalError):
    """The cache file could not be opened at all — usually it is not there yet.

    SUBCLASSES ``sqlite3.OperationalError`` DELIBERATELY. That is the type
    every existing handler on the read path already catches, so widening the
    vocabulary here cannot break one; callers that want to distinguish "there
    is no cache yet" from "that query was malformed" — which are the same
    exception type otherwise — now can.

    Raised only on the read-only path. A writable ``Store`` is allowed to
    create its database: ingest is what brings a cache into existence.
    """


# Set once, per process, the first time the extension fails to load. The
# warning is worth shouting; shouting it on every connection turns a one-line
# actionable message into log noise that gets filtered out.
_VEC_LOAD_WARNED = False


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    """Load the sqlite-vec extension onto ``conn``. Raises on any failure.

    Imported lazily rather than at module scope on purpose: a top-level
    ``import sqlite_vec`` would make an uninstalled or broken extension an
    ImportError on ``aggregator.core.store``, which is imported by literally
    every surface including read-only recall. The optional dependency has to
    stay optional all the way down.

    ``enable_load_extension`` is re-disabled in a ``finally`` so a failed load
    cannot leave the connection able to side-load native code.
    """
    import sqlite_vec

    conn.enable_load_extension(True)
    try:
        sqlite_vec.load(conn)
    finally:
        conn.enable_load_extension(False)


def _try_load_sqlite_vec(conn: sqlite3.Connection) -> bool:
    """Best-effort extension load. Returns whether the vector arm is usable.

    Catches broadly and on purpose. The failure modes are a missing wheel
    (ImportError), an ABI mismatch or a missing entry point
    (sqlite3.OperationalError), and a python compiled without extension
    loading at all (AttributeError) — and none of them is a reason for FTS5
    recall to stop working.
    """
    global _VEC_LOAD_WARNED
    try:
        _load_sqlite_vec(conn)
    except Exception as e:  # noqa: BLE001 - see docstring; degradation is the point
        if not _VEC_LOAD_WARNED:
            _VEC_LOAD_WARNED = True
            log.warning(
                "sqlite-vec extension failed to load (%s: %s) — the vector "
                "retrieval arm is DISABLED for this process. FTS5 keyword "
                "search is unaffected and continues to serve. Fix with "
                "`uv sync` / reinstall the `sqlite-vec` wheel for this "
                "interpreter, then re-run `aggregator embed --catchup`.",
                type(e).__name__,
                e,
            )
        return False
    return True

# WHAT A STORED ``src_hash`` PROMISES ABOUT THE SCRUBBER, and the one-line way
# to break that promise on purpose.
#
# Skipping an unchanged row skips its scrub. That is the entire wall-clock win
# — at 827 rows/min the 2026-08-15 run was spending essentially all of its time
# re-running Presidio over text it had already cleaned — but it means a change
# to ``core/scrub.py`` would never reach the rows already stored: their INPUT
# did not move, so nothing would ever re-scrub them, and a newly-detectable
# secret would sit in the index forever.
#
# So the fingerprint rides inside the hash. Bump this string in the same commit
# that changes what ``scrub`` detects, and every scrubbed row re-scrubs on the
# next ingest — no rebuild, no migration, no flag. Leaving it alone is the
# statement that the scrubber's output for a given input is unchanged.
SCRUB_FINGERPRINT = "presidio+gitleaks/v1"

# v3 chat-export origins. ``sessions.origin`` values are these plus the
# default 'claude-code'. Kept as a module constant so the WHERE builders,
# capabilities, and the MCP routing layer stay in lockstep.
CHAT_ORIGINS = ("chatgpt", "claude-web")

#: The group holding every source the user did not rank. Bracketed so it can
#: never collide with a real source name, and printable, because it appears in
#: ``aggregator status`` next to the ranked ones.
EMBED_REST = "(other)"

#: THE ORDER THE VECTOR ARM IS FILLED IN. User directive, 2026-08-21, verbatim:
#: ``dropbox -> blog -> llm -> claude code``.
#:
#: The full backfill is a measured 25-30 days of continuous CPU on this
#: hardware and the FTS5 arm serves throughout, so this ordering is the whole
#: difference between the vector arm being useful in week one and in week five.
#: It is not a tuning detail and it is not the obvious order: the proposal put
#: claude-code sessions FIRST, on the theory that the agent's own history is
#: what gets searched most, and the user put them LAST. Take it as given.
#:
#: Category names map to cache source names as recorded in
#: session-constraints.md — an assumption, because the user named categories:
#: ``blog`` is ``substack``; ``llm`` is ``claude-web`` plus ``chatgpt`` (which
#: has no rows in this cache today and is listed anyway, so the day an export
#: lands it is already ranked); ``claude code`` is ``sessions`` then
#: ``subagents``.
#:
#: IT SPANS BOTH ONTOLOGIES, which is exactly why the worker cannot drain
#: "observations, then records": two records sources come before every
#: observation and four more come after. The pairs are ordered, and the worker
#: walks them in this sequence.
#:
#: ``EMBED_REST`` LAST, AND IT IS NOT A DUMPING GROUND. The user ranked four
#: categories and said nothing about github, research, ticktick or sota-watch;
#: unranked means later, not unimportant. Its real job is that the groups
#: PARTITION the backlog: a row belonging to no group would never be selected,
#: never embedded, and never reported missing.
EMBED_BACKLOG_ORDER: tuple[tuple[str, str], ...] = (
    ("records", "dropbox"),
    ("records", "substack"),
    ("observations", "claude-web"),
    ("observations", "chatgpt"),
    ("observations", "sessions"),
    ("observations", "subagents"),
    ("records", EMBED_REST),
    ("observations", EMBED_REST),
)

#: Records sources the order names explicitly. Everything else is ``EMBED_REST``.
_RANKED_RECORD_SOURCES = tuple(
    source
    for kind, source in EMBED_BACKLOG_ORDER
    if kind == "records" and source != EMBED_REST
)

#: ``sessions.origin`` values the order names explicitly, for the observations
#: side of the same question.
_RANKED_OBS_ORIGINS = ("claude-code", "claude-web", "chatgpt")

#: Maps one observation row to its backfill source, in SQL, from the session
#: that owns it. Observations have no ``source`` column: which product a
#: transcript came from is ``sessions.origin``, and session-vs-subagent is
#: ``sessions.kind``. Written once and reused by the backlog query and the
#: progress tally, so the two can never disagree about which rows are whose —
#: which would show up as a source reporting more embedded rows than it has.
_OBS_SOURCE_CASE = f"""
    CASE
      WHEN s.origin = 'claude-code' AND s.kind = 'session'  THEN 'sessions'
      WHEN s.origin = 'claude-code' AND s.kind = 'subagent' THEN 'subagents'
      WHEN s.origin IN ('claude-web', 'chatgpt') THEN s.origin
      ELSE '{EMBED_REST}'
    END
    -- Reached through a LEFT JOIN, so ``s.origin`` is NULL for an observation
    -- whose session row is missing: every WHEN is false and it falls to
    -- EMBED_REST, which is what keeps the groups a partition.
"""

#: The same mapping for records, where the column is right there.
_REC_SOURCE_CASE = (
    "CASE WHEN r.source IN ("
    + ",".join(f"'{s}'" for s in _RANKED_RECORD_SOURCES)
    + f") THEN r.source ELSE '{EMBED_REST}' END"
)

# Tables whose primary key ``existing_ids`` may probe, and that key's column.
# Allowlist, not a hint: the table name is interpolated into SQL.
_PK_BY_TABLE = {
    "records": "stable_id",
    "sessions": "session_id",
    "observations": "obs_id",
}


# How many items one hash probe covers. The probe is a single indexed
# ``IN (...)`` per table, so this only bounds how much of the caller's stream is
# held at once — ``upsert_entities`` is public and may be handed a generator
# over the whole corpus, and materialising that would give back exactly the
# memory the streaming pipeline was built to save.
_HASH_PROBE_CHUNK = 1000


def _src_hash(*parts: object) -> str:
    """Fingerprint a row AS THE SOURCE PRODUCED IT, before any scrubbing.

    The point of computing this over the raw input rather than over the stored
    row is ordering: it lets an unchanged row be recognised BEFORE Presidio
    runs, and Presidio is where the wall clock goes. At 827 rows/min the
    2026-08-15 run was not SQLite-bound, it was scrubber-bound — re-cleaning
    text it had already cleaned, 372k rows at a time, every 30 minutes.

    ``blake2b`` at 16 bytes: this is a change detector, not a security
    boundary, and a 128-bit digest makes an accidental collision (which would
    silently drop an edit) not a thing that happens. NULL is a distinct byte
    rather than the empty string, and every part is terminated, so
    ``("ab", "c")`` and ``("a", "bc")`` cannot collide by concatenation.
    """
    h = hashlib.blake2b(digest_size=16)
    for part in parts:
        h.update(b"\x00" if part is None else str(part).encode("utf-8", "replace"))
        h.update(b"\x1e")
    return h.hexdigest()


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _now_iso() -> str:
    """UTC, aware, ISO-8601 — the format every timestamp column here uses."""
    return datetime.now(UTC).isoformat()


def _chunked(items: Iterable, size: int) -> Iterable[list]:
    """Bounded batching over an arbitrary iterable. Consumes lazily."""
    batch: list = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _stored_hashes(
    c: sqlite3.Connection, table: str, key: str, ids: Sequence[str]
) -> dict[str, str | None]:
    """One indexed lookup of the fingerprints already stored for ``ids``."""
    if not ids:
        return {}
    out: dict[str, str | None] = {}
    for page in _chunked(ids, 500):
        placeholders = ",".join("?" * len(page))
        rows = c.execute(
            f"SELECT {key}, src_hash FROM {table} WHERE {key} IN ({placeholders})",  # noqa: S608 - allowlisted literals
            list(page),
        )
        for row in rows:
            out[row[0]] = row[1]
    return out


def _require_expectations(
    ids: Sequence[str], expected: dict[str, str | None]
) -> None:
    """Every id being compare-and-swapped must have a stated expectation.

    A guarded write asks ``AND src_hash IS ?``. Reading that parameter with
    ``expected.get(row_id)`` made a MISSING key indistinguishable from an
    expectation of NULL, which is not a weaker guard but a DIFFERENT one — and
    one that is wrong in both directions: it never matches a fingerprinted row
    (the write vanishes, and ``cli._embed_batch`` books the miss as a benign
    ``superseded``) and it always matches a legacy row (the write lands,
    guarded against a condition nobody asked for). Pre-v4 rows are all NULL by
    design, so the second case is the common one on this corpus.

    Raised as ``KeyError`` because that is what it is: the caller built the map
    from its own row set and lost an id on the way. ``expected=None`` remains
    the way to say "no claim" — it just has to be said.
    """
    missing = [row_id for row_id in ids if row_id not in expected]
    if missing:
        raise KeyError(
            f"no expected src_hash for {missing!r}: a guarded write needs a "
            f"stated expectation for every id. A missing entry would compare "
            f"'src_hash IS NULL', which never matches a fingerprinted row and "
            f"always matches a legacy one. Pass expected=None for a "
            f"deliberately unguarded write."
        )


def _json_id_clause(column: str) -> str:
    """``<column> IN (…)`` that binds ANY NUMBER of ids as ONE parameter.

    The obvious ``IN (?,?,?…)`` breaks at ``SQLITE_MAX_VARIABLE_NUMBER`` —
    32,766 on this build — and every id set in this file that comes from an
    FTS5 or hybrid hit list is routinely bigger than that: the corpus is
    483,193 observations and a common word matches tens of thousands of them.
    ``json_each`` takes the whole list as a single JSON parameter and stays
    index-driven; the plan keeps the id lookup and carries the scope as a LIST
    SUBQUERY. JSON1 has been compiled into SQLite by default since 3.38.

    ``column`` is interpolated and is never caller-supplied: every call site
    passes a literal.
    """
    return f"{column} IN (SELECT value FROM json_each(?))"


#: The ONLY characters allowed to reach FTS5 ``MATCH``: letters, numbers and
#: underscore. ``\w`` under Python's default Unicode semantics is exactly
#: ``[\p{L}\p{N}_]`` — ``str.isalnum()`` plus ``_`` — so this is unicode-aware
#: without a third-party ``regex`` dependency.
#:
#: UNDERSCORE IS IN THE WHITELIST ON PURPOSE, and it is the one deviation from
#: the letters-and-digits-only rule this fix was specified with. Two reasons,
#: one measured and one structural:
#:
#: * Measured. ``_`` is a legal FTS5 bareword character, so ``ERR_TLS_CERT``
#:   and ``root_session_id`` are queries that ALREADY WORKED. FTS5 tokenizes
#:   such a bareword into an adjacency phrase (``err`` ``tls`` ``cert`` next to
#:   each other); rewriting it to three independent phrases is a different
#:   question with different answers. On a seeded corpus it changed both the
#:   rows and the score: bm25 -5.857257 -> -17.571770 for ``ERR_TLS_CERT``.
#:   The whole point of this fix is that queries which worked keep returning
#:   identical rows with identical scores, and on a corpus saturated with
#:   snake_case identifiers, dropping ``_`` breaks precisely that.
#: * Structural. Widening the whitelist by ``_`` gives up nothing. Inside a
#:   quoted FTS5 string the only meta-character is ``"``; ``_`` cannot begin an
#:   operator, a column filter, ``NEAR``, a prefix ``*`` or a quote in any
#:   position, quoted or not. The safety property — "the only thing that
#:   reaches MATCH is quoted word characters" — is unchanged.
_FTS5_TOKEN_RE = re.compile(r"\w+")


def fts5_match_conjuncts(text: str | None) -> list[str]:
    """Split user text into the FTS5 phrases that must ALL match.

    ``power-on`` -> ``['"power"', '"on"']``.
    ``"PR link" "status report"`` -> ``['"PR link"', '"status report"']``.

    A DOUBLE-QUOTED RUN IS ONE CONJUNCT, and that is the whole difference from
    a per-word split. The caller who writes ``"low usage cap"`` is naming three
    words that stand next to each other; three independent conjuncts is a
    different question with a different answer, and on this corpus a much worse
    one — measured, ``"low usage cap"`` goes from the single right row to 156
    rows with the right one at 118, and ``"terraform state lock"`` goes from a
    clean nothing to 1,845 hits, because "state" and "lock" are ordinary words
    here and only the adjacency was selective. Quotes are the one piece of
    query syntax every caller already knows; discarding them silently turns a
    precise question into an imprecise one without saying so.

    ONLY BALANCED QUOTES GROUP. An odd number of ``"`` means the caller's
    phrase never closed, and guessing where it ends invents a query nobody
    wrote — so an unbalanced string falls all the way back to the per-word
    split, exactly as before. ``'"unbalanced quote'`` is still
    ``['"unbalanced"', '"quote"']``.

    THE SAFETY PROPERTY IS UNCHANGED: what reaches ``MATCH`` is still nothing
    but double-quoted ``\\w+`` runs, now with single spaces allowed between
    runs INSIDE a pair of quotes. Inside an FTS5 string the only
    meta-character is ``"``, and no ``"`` from the input survives — the quotes
    in the output are ones this function wrote. So no operator, column filter,
    ``NEAR``, prefix ``*`` or stray quote can be expressed, whatever FTS5
    decides those characters mean next.

    Returns ``[]`` for text with no word characters at all.
    """
    raw = text or ""
    segments = raw.split('"')
    # An even segment count means an odd number of quotes: unbalanced. Treat
    # the whole string as unquoted rather than pairing the quotes by position.
    if len(segments) % 2 == 0:
        segments = [raw.replace('"', " ")]
    out: list[str] = []
    for i, segment in enumerate(segments):
        tokens = _FTS5_TOKEN_RE.findall(segment)
        if not tokens:
            continue
        if i % 2:  # inside a balanced pair of quotes: one adjacency phrase
            out.append('"' + " ".join(tokens) + '"')
        else:
            out.extend(f'"{t}"' for t in tokens)
    return out


def fts5_match_query(text: str | None) -> str:
    """Rewrite arbitrary user text into a safe FTS5 ``MATCH`` expression.

    ``power-on`` -> ``"power" "on"``. ``!!!`` -> ``""``.

    WHITELIST, NOT ESCAPE. Keep the word-character runs, quote each as a
    literal phrase, join with spaces (FTS5's implicit AND). Everything else is
    discarded, so no operator, column filter, ``NEAR``, prefix ``*`` or stray
    quote survives to be interpreted. An escape-based fix has to enumerate
    every character FTS5 gives meaning to and stay right as SQLite changes:
    the docs warn that input which raises a syntax error today "may be
    interpreted differently by some future version of FTS5". A whitelist is
    right by construction, including about characters nobody has thought of.

    Quoting a single-token bareword does not change what it matches — an
    unquoted bareword and a one-token quoted phrase tokenize identically under
    ``unicode61`` — which is why 29% of real queries stop erroring while the
    ones that already worked return the same rows with the same bm25 scores.
    Asserted in ``tests/core/test_store_fts_sanitize.py``.

    A BALANCED QUOTED RUN STAYS ONE PHRASE — see
    :func:`fts5_match_conjuncts`, which this is the joined form of. That
    RESTORES the identity property above rather than weakening it: a quoted
    phrase was already valid FTS5 and already meant adjacency, so shattering it
    into independent words was the one class of previously-working query this
    rewrite silently reranked.

    Returns ``""`` for text with no word characters at all. Callers MUST read
    that as "no lexical matches" and skip the query: ``MATCH ''`` is a
    different question, not a cheaper form of this one.

    THE VECTOR ARM MUST BE HANDED THE ORIGINAL TEXT. This is a property of the
    lexical arm only — ``power-on`` carries meaning to an embedding model that
    ``"power" "on"`` does not. That holds by construction here because the
    rewrite happens inside the FTS5 binding sites below, downstream of every
    caller that also drives the vector arm.
    """
    return " ".join(fts5_match_conjuncts(text))


def fts5_query_terms(text: str | None) -> list[str]:
    """The tokens :func:`fts5_match_query` sends to ``MATCH``, unquoted.

    ONE DEFINITION OF "WHAT COUNTS AS A QUERY TERM", shared with the snippet
    builder in ``mcp.py``. A snippet centred on a term the index did not
    actually match is a lie about why the row is on the page, and a second
    tokenizer in another module is how the two drift apart — the same argument
    that put the FTS5 whitelist in one function rather than at each of its
    binding sites.
    """
    return _FTS5_TOKEN_RE.findall(text or "")


def _table_present(c: sqlite3.Connection, table: str) -> bool:
    """Whether ``table`` exists. Probed once per write group, not per row."""
    return (
        c.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (table,)).fetchone()
        is not None
    )


def _vec_table_usable(c: sqlite3.Connection, table: str) -> bool:
    """Whether ``table`` exists AND ``vec0`` is loaded on THIS connection.

    EXISTENCE IS THE WRONG QUESTION FOR A VIRTUAL TABLE, and asking it was
    round 3's H3. ``sqlite_master`` records that the table was created; it says
    nothing about whether the module implementing it is available now.
    ``sqlite-vec`` is a loadable extension, so the two come apart in a case
    that is ordinary rather than exotic: a cache filled on a working
    interpreter, then opened on one where the wheel is missing or ABI-
    mismatched. Every statement against the table — ``DELETE`` included —
    then raises ``no such module: vec0``.

    That mattered because the probe guarded the ingest write path. An edited
    row's stale vectors are dropped inside the same transaction that writes
    the row, so a raise there did not cost the vector arm (which was already
    gone) — it aborted the WRITE, and took FTS5 keyword recall down with it.
    Serving keyword search when the vector arm cannot load is a founding
    requirement of this branch, not a nicety.

    So the probe runs the cheapest statement that would fail for the same
    reason the real one would. ``LIMIT 1`` on a ``vec0`` table touches its
    shadow index and returns at most one row, and it is asked once per write
    GROUP — never per row — for the same reason the existence probe was.
    """
    if not _table_present(c, table):
        return False
    try:
        c.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone()  # noqa: S608 - fixed literals
    except sqlite3.OperationalError:
        # ``no such module: vec0``. Same degradation as a failed load: the
        # vector arm is off, and the caller's write proceeds without it.
        return False
    return True


def _owner_of_chunk(
    c: sqlite3.Connection, base_table: str, base_key: str, chunk_id: str
) -> str:
    """Which row a chunk id belongs to, for callers that were not told.

    ``commit_embed_batch`` carries the owner explicitly and never comes here,
    because the parse below is genuinely ambiguous: ``github:x:1`` is a real
    record id AND a plausible chunk 1 of ``github:x``. This exists for
    ``upsert_vec_*``, which is handed a bare chunk id.

    THE AMBIGUITY IS RESOLVED BY ASKING THE TABLE, not by preferring one
    reading. A trailing ``:<n>`` is stripped only when the remainder is an
    actual row and the full id is not; anything else is its own owner. Getting
    this wrong is not cosmetic — ``owner_id`` is what the backfill LEFT JOINs
    on, so a wrong answer means a row that is never selected for embedding
    again and never reported as missing.
    """
    head, sep, tail = chunk_id.rpartition(":")
    if not sep or not head or not tail.isdigit():
        return chunk_id
    own = c.execute(
        f"SELECT 1 FROM {base_table} WHERE {base_key} = ?",  # noqa: S608 - allowlisted literals
        (chunk_id,),
    ).fetchone()
    if own is not None:
        return chunk_id
    parent = c.execute(
        f"SELECT 1 FROM {base_table} WHERE {base_key} = ?",  # noqa: S608 - allowlisted literals
        (head,),
    ).fetchone()
    return head if parent is not None else chunk_id


def _drop_row_vectors(
    c: sqlite3.Connection, table: str, key: str, row_id: str
) -> None:
    """Delete every vector belonging to ``row_id``, by EXACT KEY ONLY.

    A body is stored either as one vector under the row's own id, or — when it
    chunked — as ``<id>:0 .. <id>:N-1``, written contiguously by
    ``cli._embed_batch``. So walking the indices from 0 and stopping at the
    first miss removes exactly the set that was written, and every statement is
    a primary-key lookup on the vec0 shadow index.

    NOT the obvious ``substr(key, 1, length(id)+1) = id || ':'``. That is
    correct (it spares ``o10`` while removing ``o1:0``) but it is a full table
    scan: measured at 50k vectors it is 500x slower per row than the key
    lookups, and unlike them it gets worse as the index grows — against the
    422k vectors the backfill produces it would put a ~300 ms scan on every
    edited row of every ingest tick.

    ``rowcount <= 0`` rather than ``== 0`` ends the walk: a virtual table that
    reported -1 would otherwise spin forever.

    ACROSS EVERY MODEL, and the index rows with them. The caller reaches here
    because the row's BODY changed, which invalidates the text that went into
    the encoder — so a vector under a second model is exactly as stale as the
    one under the first. Leaving the ``chunk_embeddings`` rows behind would be
    worse than leaving the vectors: the backfill LEFT JOIN would report the row
    as embedded and never come back for it.
    """
    c.execute(f"DELETE FROM {table} WHERE {key} = ?", (row_id,))  # noqa: S608 - fixed literals
    i = 0
    while True:
        cur = c.execute(
            f"DELETE FROM {table} WHERE {key} = ?",  # noqa: S608 - fixed literals
            (f"{row_id}:{i}",),
        )
        if cur.rowcount <= 0:
            break
        i += 1
    if _table_present(c, "chunk_embeddings"):
        c.execute("DELETE FROM chunk_embeddings WHERE owner_id = ?", (row_id,))


#: ``(vec table, key column, base table, base key)`` for the orphan purge.
_VEC_OWNERSHIP = {
    "observations": ("vec_observations", "obs_id", "observations", "obs_id"),
    "records": ("vec_records", "stable_id", "records", "stable_id"),
}


def _purge_orphan_vectors(c: sqlite3.Connection, kind: str) -> int:
    """Delete vectors whose row is gone. Returns how many.

    WHY THIS IS AN ANTI-JOIN AND NOT A LIST OF IDS. The scoped delete paths
    (``rebuild(source)``, ``rebuild_and_upsert_entities(origins=…)``) remove
    rows by source or origin, and nothing propagates that to the vec tables:
    there is no foreign key into a virtual table and no trigger can span one.
    Sweeping for orphans after the fact costs one scan of the vec table (~40 ms
    at 422k vectors, on an operation that already rewrote the whole corpus) and
    has the useful property of cleaning up orphans left by any EARLIER
    rebuild too, rather than only the one that just ran.

    An orphan is not a wrong answer; it is a quietly smaller one. It still
    occupies one of the ``_VECTOR_ARM_K`` = 50 KNN slots a query gets, then
    matches no row when the fused id set reaches SQL. Enough of them and the
    vector arm returns a full 50 neighbours and contributes nothing, while
    every count still reports the index complete.

    A vector is kept if EITHER reading of its key still has a row: the key
    itself, or the ``<id>:N`` chunk owner. Both are checked because the two are
    genuinely ambiguous — ``github:x:1`` is a real record id AND a plausible
    chunk 1 of ``github:x`` — and keeping a live vector costs one KNN slot
    while dropping one costs the row its recall.
    """
    vec_table, vec_key, base_table, base_key = _VEC_OWNERSHIP[kind]
    # USABLE, not merely present: this runs inside the rebuild's savepoint, so
    # ``no such module: vec0`` here rolls back the re-write it was cleaning up
    # after. See ``_vec_table_usable``.
    if not _vec_table_usable(c, vec_table):
        return 0
    # ``rtrim(key,'0-9')`` strips a trailing chunk index; when what remains
    # ends in ':' the key was ``<owner>:<n>`` and the owner is the rest.
    owner = (
        f"CASE WHEN substr(rtrim({vec_key}, '0123456789'), -1) = ':' "
        f"THEN substr({vec_key}, 1, length(rtrim({vec_key}, '0123456789')) - 1) "
        f"ELSE {vec_key} END"
    )
    cur = c.execute(
        f"DELETE FROM {vec_table} WHERE "  # noqa: S608 - fixed literals
        f"{vec_key} NOT IN (SELECT {base_key} FROM {base_table}) "
        f"AND ({owner}) NOT IN (SELECT {base_key} FROM {base_table})"
    )
    return max(0, cur.rowcount)


class EmptyRebuildRefusedError(RuntimeError):
    """Raised by ``Store.rebuild_and_upsert`` when the incoming record list is
    smaller than the caller-declared ``min_records`` floor.

    Round-3 HIGH: ``rebuild_and_upsert`` runs a DELETE + upsert atomically,
    so calling it with an empty list DELETEs every row for the source and
    commits. That's a silent wipe when the caller genuinely had records the
    last time round and the current pass just failed to fetch them (network
    hiccup, upstream outage). Callers that know the source has held records
    before pass ``min_records=1`` to trip this guard rather than allow the
    wipe. Kept as its own type (not RuntimeError) so CLI/MCP callers can
    catch it specifically and turn it into a structured refusal.
    """


# ---------------------------------------------------------------------------
# DDL — three tables + three FTS shadows, all under one migration.
# ---------------------------------------------------------------------------
_DDL: list[str] = [
    # --- v2: sessions (Langfuse "trace") ------------------------------
    """
    CREATE TABLE IF NOT EXISTS sessions (
        session_id             TEXT PRIMARY KEY,
        root_session_id        TEXT NOT NULL,
        parent_session_id      TEXT,
        kind                   TEXT NOT NULL CHECK(kind IN ('session','subagent')),
        agent_id               TEXT,
        agent_type             TEXT,
        spawned_by_tool_use_id TEXT,
        cwd                    TEXT,
        git_branch             TEXT,
        first_ts               TEXT NOT NULL,
        last_ts                TEXT NOT NULL,
        jsonl_path             TEXT NOT NULL,
        origin                 TEXT NOT NULL DEFAULT 'claude-code',
        -- v4. Nullable and unindexed on purpose: it is only ever read by
        -- primary key, alongside the row it describes. See ``_src_hash``.
        src_hash               TEXT
    );
    """,
    "CREATE INDEX IF NOT EXISTS sessions_root ON sessions(root_session_id);",
    "CREATE INDEX IF NOT EXISTS sessions_kind ON sessions(kind);",
    "CREATE INDEX IF NOT EXISTS sessions_last_ts ON sessions(last_ts);",
    "CREATE INDEX IF NOT EXISTS sessions_origin ON sessions(origin);",
    # --- v2: observations (Langfuse "observation") --------------------
    #
    # ``provenance`` (v6) SAYS WHO COMPOSED THE TEXT, which ``type`` does not:
    # ``type`` is the channel a line arrived on, and 59% of ``type='user'`` rows
    # were written by a machine. It holds one of the five members of
    # ``aggregator.core.provenance.PROVENANCE_VALUES``. NULL means NOT YET
    # CLASSIFIED and is the backfill's cursor, exactly as
    # ``embedding_state IS NULL`` is the embed worker's — never ``'unknown'``,
    # which would collapse "we looked and could not tell" into "nobody looked".
    #
    # THE PROSE LIVES OUT HERE RATHER THAN INSIDE THE STATEMENT because
    # sqlite_schema stores the CREATE TABLE text verbatim and SQLite re-parses
    # it with a reduced tokenizer on ``ALTER TABLE ... DROP COLUMN``. Measured:
    # a multi-line ``--`` block in front of the LAST column made that reparse
    # fail with "incomplete input" on SQLite 3.50.4 (and not on 3.53.3), which
    # would strand a future migration on the live database for the sake of a
    # comment. Column notes inside the statement stay to one trailing line.
    """
    CREATE TABLE IF NOT EXISTS observations (
        obs_id           TEXT PRIMARY KEY,
        session_id       TEXT NOT NULL REFERENCES sessions(session_id),
        root_session_id  TEXT NOT NULL,
        parent_obs_id    TEXT,
        type             TEXT NOT NULL,
        ts               TEXT NOT NULL,
        model            TEXT,
        input_tokens     INTEGER,
        output_tokens    INTEGER,
        tool_name        TEXT,
        tool_use_id      TEXT,
        body             TEXT,
        src_hash         TEXT,         -- v4; see ``_src_hash``
        -- v5. NULL means "not embedded yet" and is what the background embed
        -- worker selects on; 'ok' / 'skip' / 'error' are terminal.
        embedding_state  TEXT,
        provenance       TEXT          -- v6; see aggregator.core.provenance
    );
    """,
    "CREATE INDEX IF NOT EXISTS obs_root_ts ON observations(root_session_id, ts);",
    "CREATE INDEX IF NOT EXISTS obs_session_ts ON observations(session_id, ts);",
    "CREATE INDEX IF NOT EXISTS obs_type ON observations(type);",
    # WITHOUT THIS THE BACKFILL IS QUADRATIC. Its chunk is
    # ``WHERE provenance IS NULL LIMIT n``, and unindexed that scans past every
    # row already classified — 1,100 chunks over 549,952 rows, each starting
    # further in than the last. Indexed it is a seek. It also serves ``by:``.
    "CREATE INDEX IF NOT EXISTS obs_provenance ON observations(provenance);",
    # FTS5 external-content over observations.body. Sync via triggers below.
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS obs_fts USING fts5(
        body,
        content='observations',
        content_rowid='rowid',
        tokenize='unicode61 remove_diacritics 2'
    );
    """,
    """
    CREATE TRIGGER IF NOT EXISTS observations_ai AFTER INSERT ON observations
    BEGIN
        INSERT INTO obs_fts(rowid, body) VALUES (new.rowid, new.body);
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS observations_ad AFTER DELETE ON observations
    BEGIN
        INSERT INTO obs_fts(obs_fts, rowid, body) VALUES ('delete', old.rowid, old.body);
    END;
    """,
    # ``OF body`` IS THE WHOLE POINT, and it is not a micro-optimisation.
    #
    # Without the column list this fires on ANY update to ANY column — an
    # ``embedding_state`` flip, a ``provenance`` stamp — and each firing is a
    # full FTS5 ``'delete'`` plus re-insert of the entire body. With
    # ``auto_vacuum=0`` the pages that frees are never handed back, which is the
    # same mechanism documented at the ingest UPSERT below. Measured on a
    # throwaway database at 1/5 of live scale: a chunked column-only UPDATE ran
    # 87.5 s and grew the file 128 MB wide, versus 7.4 s and no growth narrow.
    # Extrapolated to the live 549,952 rows that is 12x wall clock and
    # 380-640 MB of permanent bloat on a 1.4 GB file.
    #
    # IT INDEXES EXACTLY AS MUCH AS BEFORE. SQLite fires ``UPDATE OF body``
    # whenever ``body`` appears in the SET clause, whether or not the value
    # actually moved, and the ingest UPSERT always names ``body =
    # excluded.body``. Pinned by ``tests/core/test_store_fts_trigger_narrowing
    # .py::test_upsert_keeps_the_fts_index_correct``, which matches a sentinel
    # token through the real upsert rather than restating this paragraph.
    #
    # ``IF NOT EXISTS`` will NOT replace a trigger that is already there, so
    # every database created before this change needs ``_ensure_narrow_fts_
    # update_trigger`` to drop the wide one first — and those are the databases
    # the narrowing is for.
    """
    CREATE TRIGGER IF NOT EXISTS observations_au
    AFTER UPDATE OF body ON observations
    BEGIN
        INSERT INTO obs_fts(obs_fts, rowid, body) VALUES ('delete', old.rowid, old.body);
        INSERT INTO obs_fts(rowid, body) VALUES (new.rowid, new.body);
    END;
    """,
    # --- Legacy: records + records_fts for GitHub-shaped sources -------
    """
    CREATE TABLE IF NOT EXISTS records (
        stable_id  TEXT PRIMARY KEY,
        source     TEXT NOT NULL,
        subject    TEXT NOT NULL,
        body       TEXT NOT NULL,
        tags       TEXT NOT NULL,       -- JSON array
        created_at TEXT,
        updated_at TEXT,
        extra      TEXT NOT NULL DEFAULT '{}',
        src_hash   TEXT,             -- v4; see ``_src_hash``
        embedding_state TEXT         -- v5; NULL = not embedded yet
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_records_source ON records(source);",
    "CREATE INDEX IF NOT EXISTS idx_records_updated ON records(updated_at);",
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS records_fts USING fts5(
        stable_id UNINDEXED,
        source    UNINDEXED,
        subject,
        body,
        tags,
        tokenize='unicode61 remove_diacritics 2'
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    """,
    # --- v4: the per-source high-water mark ----------------------------
    #
    # IN THIS DATABASE, NOT IN A SIDECAR FILE, and that is the whole design.
    # A watermark and the data it describes are two artifacts that must agree.
    # Updated by two separate writes they cannot be made to agree under
    # SIGTERM, and the two failure orders are not symmetric: watermark-first
    # loses records permanently and silently, data-first merely re-reads. Here
    # they are one row and one batch inside one transaction, so neither order
    # is reachable.
    #
    # ``cursor_kind`` is on the row rather than only in code because a source
    # that CANNOT support a watermark has to be able to say so out loud (see
    # ``imports/ingest_state.CursorKind``). Silently full-scanning while
    # reporting like an incremental source is the failure this whole change
    # exists to remove.
    """
    CREATE TABLE IF NOT EXISTS ingest_state (
        source               TEXT PRIMARY KEY,
        cursor_value         TEXT,
        cursor_kind          TEXT NOT NULL,
        last_run_at          TEXT,
        last_ok_at           TEXT,
        rows_seen            INTEGER NOT NULL DEFAULT 0,
        consecutive_failures INTEGER NOT NULL DEFAULT 0,
        last_error           TEXT,
        -- When this source may be tried again, or NULL for "right now".
        -- STORED rather than recomputed on read, because the delay is
        -- jittered: recomputing it would let two reads inside one run
        -- disagree about whether the source runs, which is a decision that
        -- must be made once and stay made.
        next_attempt_at      TEXT
    );
    """,
    # --- v4: the poison-record quarantine ------------------------------
    #
    # A dead-letter queue collapsed into a table, which for one user on one
    # machine is strictly better than a broker: it is transactional with the
    # data writes, which a broker can never be. ``next_retry_at IS NULL`` is
    # terminal — never retried again, never deleted either, because a row
    # nobody can count is a gap that reads as full coverage.
    """
    CREATE TABLE IF NOT EXISTS quarantine (
        source        TEXT NOT NULL,
        record_key    TEXT NOT NULL,
        error_type    TEXT NOT NULL,
        error_detail  TEXT,
        attempts      INTEGER NOT NULL DEFAULT 1,
        first_seen_at TEXT NOT NULL,
        last_seen_at  TEXT NOT NULL,
        next_retry_at TEXT,
        PRIMARY KEY (source, record_key)
    );
    """,
    "CREATE INDEX IF NOT EXISTS quarantine_terminal "
    "ON quarantine(source, next_retry_at);",
    # --- the other half of the quarantine: PERMANENTLY-BAD INPUT -------
    #
    # ``quarantine`` above holds records the SINK refused, which is a thing
    # that might work next time — hence ``attempts`` and ``next_retry_at``.
    # This holds input that will NEVER parse: two malformed lines in a JSONL
    # file are not going to become valid, so there is nothing to retry and the
    # only question is whether a human has already been told. Reported loudly
    # the first time the identity is seen, then quiet, and always listed by
    # ``aggregator status`` — because quiet is only acceptable while it is not
    # the same as forgotten.
    #
    # ``fault_key`` is a hash of source + scope + reason + the EXACT record
    # list, never of the count: suppressing by count would let a different bad
    # line inherit a known one's silence. ``scope_stamp`` is the artifact's
    # mtime+size when the fault was recorded, which is how a rewritten file
    # gets its row dropped instead of reporting a quarantine that is over.
    #
    # No version bump. The table is additive and created by a
    # ``CREATE TABLE IF NOT EXISTS`` that ``migrate()`` runs on every single
    # CLI invocation, so an existing v4 database grows it on the next command;
    # bumping ``user_version`` would instead point every such database at the
    # drop-and-rebuild path for a table that holds nothing anyone can rebuild.
    """
    CREATE TABLE IF NOT EXISTS poison_faults (
        source        TEXT NOT NULL,
        fault_key     TEXT NOT NULL,
        scope         TEXT NOT NULL,
        scope_stamp   TEXT NOT NULL,
        reason        TEXT NOT NULL,
        detail        TEXT NOT NULL,
        record_count  INTEGER NOT NULL DEFAULT 0,
        line          TEXT NOT NULL,
        first_seen_at TEXT NOT NULL,
        last_seen_at  TEXT NOT NULL,
        PRIMARY KEY (source, fault_key)
    );
    """,
    "CREATE INDEX IF NOT EXISTS poison_faults_scope "
    "ON poison_faults(source, scope);",
    # --- v5: the embed worker's backlog probe --------------------------
    #
    # ``select_unembedded`` is ``WHERE embedding_state IS NULL ORDER BY ts``
    # over 372k rows on every worker batch. Without an index that is a full
    # scan per batch, which is the shape of bug this schema keeps producing.
    "CREATE INDEX IF NOT EXISTS obs_embedding_state "
    "ON observations(embedding_state);",
    "CREATE INDEX IF NOT EXISTS rec_embedding_state "
    "ON records(embedding_state);",
    # --- v5: the embedding index of record, keyed (chunk_id, model) ------
    #
    # THE TABLE THE REFERENCE DESIGN PUTS AT THE CENTRE, and the reason a
    # model change stops being an outage. Keying on the chunk ALONE makes the
    # vector index a singleton: there is exactly one embedding per chunk, its
    # model is whatever the file as a whole was last stamped with, and moving
    # to a new model is therefore a DELETE of everything followed by a
    # multi-week recompute (``docs/embedding-throughput.md``). Every consent
    # gate in this file exists because of that shape.
    #
    # With the model in the key, model B's vectors are extra ROWS beside model
    # A's. The backfill for B is a background job that runs to completion while
    # A goes on answering queries, and nothing is ever deleted to make room.
    #
    # ``content_sha256`` is what carries an embedding across a RE-CHUNK. When a
    # document changes in one paragraph most chunk hashes are unchanged, and
    # this corpus is chat transcripts that are APPENDED to rather than
    # rewritten — so the leading chunks of an edited session are byte-identical
    # and their vectors are still correct. NULLABLE on purpose: a vector
    # written through a path that never saw the text (``upsert_vec_*``) cannot
    # prove any content matched, and the honest answer to an unprovable reuse
    # is to embed it again.
    #
    # ``owner_id`` is the row a chunk came from, carried explicitly rather than
    # parsed back out of ``<owner>:<n>``. That parse is genuinely ambiguous —
    # ``github:x:1`` is a real record id AND a plausible chunk 1 of
    # ``github:x`` — and it is the join key the backfill LEFT JOINs on, so
    # guessing it wrong means a row silently never gets embedded.
    #
    # NO FOREIGN KEY, because a chunk's owner lives in one of two tables and
    # SQLite has no polymorphic reference. ``_purge_orphan_vectors`` sweeps.
    """
    CREATE TABLE IF NOT EXISTS chunk_embeddings (
        chunk_id       TEXT    NOT NULL,
        model          TEXT    NOT NULL,
        kind           TEXT    NOT NULL,
        owner_id       TEXT    NOT NULL,
        dim            INTEGER NOT NULL,
        content_sha256 TEXT,
        created_at     TEXT    NOT NULL,
        PRIMARY KEY (chunk_id, model)
    );
    """,
    # The backfill's LEFT JOIN runs against exactly this prefix, once per
    # candidate row, so it is the difference between a probe and a scan.
    "CREATE INDEX IF NOT EXISTS chunk_embeddings_owner "
    "ON chunk_embeddings(model, kind, owner_id);",
    # The reuse lookup: "is there already a vector for this exact text under
    # this exact model?"
    "CREATE INDEX IF NOT EXISTS chunk_embeddings_sha "
    "ON chunk_embeddings(model, content_sha256);",
    # --- v5: the version pointer ----------------------------------------
    #
    # ``completed_at`` is the flip that keeps a half-built index invisible. A
    # partially filled embedding space does not fail loudly — it answers, with
    # plausible-looking scores, from whichever fraction of the corpus happened
    # to be embedded first. That is worse than no vector arm, because nothing
    # about the result says it was drawn from a tenth of the index.
    #
    # See ``Store.serving_embedding_model`` for the one deliberate exception:
    # the FIRST index a cache ever builds has nothing to fall back to.
    """
    CREATE TABLE IF NOT EXISTS embedding_versions (
        model        TEXT PRIMARY KEY,
        dim          INTEGER NOT NULL,
        started_at   TEXT NOT NULL,
        completed_at TEXT
    );
    """,
]


# --- v5: the vector index, kept OUT of ``_DDL`` -----------------------------
#
# Separate because it is the only DDL in this file that can fail for a reason
# that is nobody's bug: a ``vec0`` virtual table cannot be created — or dropped
# — on a connection where the sqlite-vec extension did not load. Running these
# from ``_DDL`` would make ``migrate()`` raise on a machine missing the
# extension, taking the entire schema, and therefore FTS5 recall, down with the
# optional half of the feature. ``migrate()`` runs them only when
# ``vector_available``, and re-runs them on every invocation, so a database
# that migrated without the extension picks the vec tables up on the first
# command after the extension is fixed. No second migration, no version bump.
#: Width of a stored vector. Must equal ``embed._EMBED_DIM``; the equality is
#: asserted by a test rather than by an import, because ``store`` is on the MCP
#: cold-start path and ``embed`` must not be dragged onto it.
_VEC_DIM = 768

#: ``meta`` key holding ``{"model": ..., "dim": ...}`` for the vectors on disk.
#: Its ABSENCE is meaningful: it means the vec tables were written by something
#: that is not this code, and therefore may not be trusted.
VECTOR_PROVENANCE_KEY = "vector_provenance"

#: ``meta`` key naming the row the embed worker is attempting right now. Its
#: presence at STARTUP means the previous worker died on that row without
#: unwinding — see ``Store.claim_embed_row``.
EMBED_CLAIM_KEY = "embed_inflight"

#: The command that may authorise a provenance mismatch to DELETE vectors.
#:
#: NOT AN ENVIRONMENT VARIABLE, and round 3's H1 is why. The first opt-in was
#: ``AGGREGATOR_VECTOR_REINDEX=1``, read by ``reindex_consented()`` inside
#: ``migrate()`` — and ``migrate()`` runs on EVERY subcommand. So the consent
#: was ambient and sticky: export it once for the rebuild you meant, and the
#: next ``aggregator query`` in that shell carried it too. Pair that with a
#: stray ``AGGREGATOR_EMBED_BACKEND`` and a READ deleted 25-30 days of CPU,
#: which is exactly the failure round 2's S1 fix existed to prevent,
#: reintroduced through the fix's own escape hatch.
#:
#: Consent is therefore a PARAMETER — ``migrate(allow_vector_reindex=True)`` —
#: and only ``cli._cmd_embed`` passes it, only when ``--reindex`` was typed on
#: that invocation, and only after printing what it would destroy and taking a
#: ``y`` on stdin. That is the shape ``ingest --rebuild`` already uses for a
#: row drop of comparable cost, and it makes the loud path the easy one: no
#: environment state can authorise this, so nothing can authorise it silently.
VECTOR_REINDEX_COMMAND = "aggregator embed --catchup --source both --reindex"

# ``model`` IS A vec0 PARTITION KEY, and the key column is a plain metadata
# column rather than the PRIMARY KEY it used to be. Both halves are needed for
# ``(chunk_id, model)``:
#
# * vec0's ``primary key`` is globally unique and a partition key does NOT
#   participate in it — measured, not assumed: inserting the same ``chunk_id``
#   under two models against a ``text primary key`` raises "UNIQUE constraint
#   failed on t primary key". So the key column has to stop being the PK.
# * A partition key gives each model its own shadow index, so a KNN with
#   ``AND model = ?`` searches only that model's vectors. Without it the two
#   spaces would share one brute-force scan and a foreign vector could win a
#   top-k slot with a meaningless distance.
#
# Uniqueness of ``(chunk_id, model)`` is enforced by ``chunk_embeddings``, the
# plain table that owns the key; these tables hold the bytes. The writers below
# keep their existing delete-then-insert idempotency.
_VEC_DDL: list[str] = [
    f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_observations USING vec0(
        model      TEXT PARTITION KEY,
        obs_id     TEXT,
        embedding  float[{_VEC_DIM}]
    );
    """,
    f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_records USING vec0(
        model      TEXT PARTITION KEY,
        stable_id  TEXT,
        embedding  float[{_VEC_DIM}]
    );
    """,
]


def vector_provenance(embedder: object | None = None) -> tuple[str, int]:
    """``(model id, dimension)`` the vectors in this cache are supposed to be.

    Imported from ``aggregator.core.embed`` LAZILY and deliberately. That
    module owns which model the worker loads, so reading the answer from
    anywhere else is how the stamp ends up naming a model nobody uses — but
    ``store`` is imported by ``aggregator.mcp`` on every editor cold start, so
    the import may not happen at module scope. Importing the module is cheap;
    ``sentence_transformers`` lives inside ``Embedder.__init__``.

    PASS THE EMBEDDER ON THE WRITE PATH. That is round 3's M2, whose embed-side
    half landed in ``adaa28e``. ``configured_model_id()`` with no argument reads
    ``AGGREGATOR_EMBED_BACKEND``, while ``Embedder`` resolves from its own
    ``backend=``/``model_name=`` arguments FIRST — so an embedder constructed
    with explicit arguments wrote vectors from one model while this stamped
    them with another. The stamp is the input to round 1's H1, round 2's S1 and
    round 3's H1, and all three fail in the same direction on a stamp that
    lies: a foreign index reads as native.

    THE READ PATH KEEPS THE NO-ARGUMENT FORM, deliberately. It asks whether the
    vectors already on disk may be trusted, and it runs on every ``Store`` —
    the read-only MCP one included — before any embedder exists. Threading an
    embedder through it would mean loading a model to answer a question about a
    file.

    THE ANSWER IS A VERSION STRING, NOT A REPO ID, and that is criterion E.
    A bare repo id is silent about quantization, about the MRL width, and
    about the chunker geometry — three things that each change the bytes of
    every vector while leaving the model name untouched. See
    :func:`aggregator.core.embed.embedding_version`.
    """
    from aggregator.core.embed import embedding_version

    return embedding_version(embedder), _VEC_DIM

_VEC_DROP_ALL: list[str] = [
    "DROP TABLE IF EXISTS vec_observations;",
    "DROP TABLE IF EXISTS vec_records;",
]


_DROP_ALL: list[str] = [
    "DROP TABLE IF EXISTS poison_faults;",
    "DROP TABLE IF EXISTS quarantine;",
    "DROP TABLE IF EXISTS ingest_state;",
    "DROP TRIGGER IF EXISTS observations_ai;",
    "DROP TRIGGER IF EXISTS observations_ad;",
    "DROP TRIGGER IF EXISTS observations_au;",
    "DROP TABLE IF EXISTS obs_fts;",
    "DROP TABLE IF EXISTS observations;",
    "DROP TABLE IF EXISTS sessions;",
    "DROP TABLE IF EXISTS records_fts;",
    "DROP TABLE IF EXISTS records;",
    "DROP TABLE IF EXISTS meta;",
]


def _provenance_refusal(stamped: str, expected: str, vectors: int | None) -> str:
    """The one message an operator gets when the index and the build disagree.

    It has to carry three things, because a mismatch is silent otherwise and
    the two ways out are opposite actions: what disagreed, that NOTHING was
    deleted, and the exact command for each intent. ``vectors`` is ``None`` on
    the read path, which declines to pay an O(n) ``COUNT(*)`` to decorate an
    error it has already decided to raise.
    """
    held = (
        "the vectors on disk are intact"
        if vectors is None
        else f"the {vectors} vector(s) on disk are intact"
    )
    return (
        f"refusing to use the vector index on this cache: it is {stamped}, and "
        f"this process is configured for {expected}. NOTHING WAS DELETED — "
        f"{held}. The vector arm is switched "
        f"off for this process instead, so no vector whose model is unknown is "
        f"served; FTS5 keyword search is unaffected. If "
        f"AGGREGATOR_EMBED_BACKEND is exported in this shell then that is the "
        f"cause and `unset AGGREGATOR_EMBED_BACKEND` is the whole fix. If the "
        f"model change is intended, the index cannot be converted and has to "
        f"be rebuilt from scratch (weeks of CPU) — say so once, explicitly, on "
        f"the command that owns the vector index: `{VECTOR_REINDEX_COMMAND}`. "
        f"It prints what it would delete and asks for a 'y' first."
    )


def _unattributable_refusal(stamped: str) -> str:
    """The READ path's refusal: vectors on disk that name no model at all.

    Its own message, rather than :func:`_provenance_refusal`, for two reasons.

    It is a DIFFERENT fault. That one is "the index names a model and it is not
    the one I want", which is now handled by keeping both and choosing at query
    time. This one is "the index names nobody" — a vec table from before
    embeddings were keyed on the model, or one nothing ever stamped — and no
    amount of choosing helps, because there is nothing to choose between.

    And it must NOT NAME THE EXPECTED MODEL, which is what keeps it cheap.
    Computing that value imports ``aggregator.core.embed`` and numpy, and this
    runs under ``capabilities()`` on the MCP connect path, where every import
    is paid before the user's first search returns.
    """
    return (
        f"refusing to use the vector index on this cache: it is {stamped}, and "
        f"its vectors carry no model attribution — either they predate "
        f"model-keyed embeddings (no `model` column on the vec tables) or "
        f"nothing ever stamped them. NOTHING WAS DELETED — the vectors on disk "
        f"are intact. The vector arm is switched off for this process instead, "
        f"so no vector whose model is unknown is served; FTS5 keyword search "
        f"is unaffected. If AGGREGATOR_EMBED_BACKEND is exported in this shell "
        f"then that is worth ruling out first — `unset "
        f"AGGREGATOR_EMBED_BACKEND` costs nothing. Otherwise the index cannot "
        f"be converted, only rebuilt: `{VECTOR_REINDEX_COMMAND}`, which prints "
        f"what it would delete and asks for a 'y' first."
    )


def _default_db_path() -> Path:
    """Resolve ``$XDG_DATA_HOME/aggregator/cache.db`` (creating parents)."""
    root = Path(
        os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    )
    p = root / "aggregator" / "cache.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


class Store:
    """Thin wrapper around a per-process SQLite connection.

    Not thread-safe within a process — v1 assumes a single ingest thread
    per process. Cross-process concurrency IS supported: two systemd user
    timers (sessions + github) fire on ``*:0/30`` and both open a Store
    against the same ``cache.db``. ``_c()`` enables WAL journal mode + a
    30s busy_timeout (Codex Phase 2 bump from 5s) so the second writer
    waits for the first to release its lock instead of failing with
    ``database is locked``. The github timer also jitters by 3 min
    (``RandomizedDelaySec`` in nix/aggregator.nix) to reduce collision
    probability at the tick boundary.
    """

    def __init__(self, db_path: str | Path | None = None, read_only: bool = False):
        self.db_path = Path(db_path) if db_path else _default_db_path()
        self.read_only = read_only
        self._conn: sqlite3.Connection | None = None
        self._vector_available = False
        self._vector_write_warned = False
        #: Why the vector arm is refusing, or ``None`` when it is trusted.
        self._vector_quarantine: str | None = None
        #: Whether the question above has been answered for this connection.
        self._vector_quarantine_decided = False

    # -- connection lifecycle ---------------------------------------------

    @property
    def vector_available(self) -> bool:
        """Whether sqlite-vec loaded on this store's connection.

        Reading it opens the connection, because the answer is not knowable
        until the load has been attempted. False means the vector arm is off
        for this store: vec DDL is skipped, vector writes no-op, and vector
        reads raise :class:`VectorIndexUnavailableError`. FTS5 is unaffected.
        """
        self._c()
        return self._vector_available

    def _c(self) -> sqlite3.Connection:
        if self._conn is None:
            if self.read_only:
                # ``mode=ro`` ONLY. Not ``immutable=1`` — that is a promise to
                # SQLite that the file cannot change, and this file changes
                # constantly: the ingest timer rewrites it every 30 minutes and
                # the embed backfill writes for weeks. Under that promise SQLite
                # skips all locking, ignores the ``-wal`` outright, and holds its
                # page cache across statements forever. Measured consequences on
                # this exact URI, not theory:
                #   * WAL-committed rows invisible, so ``has_embedded_rows`` —
                #     the v5 hybrid routing predicate — calls a warm vector index
                #     cold and silently drops the vector arm;
                #   * after a checkpoint, ``SELECT COUNT(*)`` returned 19,991
                #     where the truth was 40,000, with NO error raised;
                #   * after a VACUUM, ``database disk image is malformed``.
                # ``mode=ro`` is correct in every one of those cases. The single
                # case it cannot serve — a dirty ``-wal`` whose ``-shm`` is absent
                # and whose directory is unwritable — it refuses loudly with
                # "unable to open database file", where ``immutable=1`` answers
                # confidently and wrongly. Loud beats silently wrong here.
                # Read-only is still enforced by ``mode=ro`` plus
                # ``_ensure_writable``; dropping the flag grants no write rights.
                uri = f"file:{self.db_path}?mode=ro"
                try:
                    self._conn = sqlite3.connect(uri, uri=True)
                except sqlite3.OperationalError as e:
                    # Named, because the bare message is
                    # "unable to open database file" — the same type a
                    # malformed query raises, and the commonest cause is
                    # simply that nothing has been ingested yet.
                    raise CacheUnavailableError(
                        f"no readable cache at {self.db_path}: {e}. Run "
                        f"`aggregator ingest --all` to create it; on a fresh "
                        f"machine that is expected rather than a fault."
                    ) from e
            else:
                self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON;")
            # BEST-EFFORT, on read-only connections too — the vector arm is a
            # read feature first. A failure here is not fatal; see
            # ``_try_load_sqlite_vec`` and the v5 note in the module docstring.
            self._vector_available = _try_load_sqlite_vec(self._conn)
            if self.read_only:
                return self._conn
            # Codex Phase 2 MEDIUM #2: concurrent-writer safety. See prior
            # revision docstring; WAL + busy_timeout + synchronous=NORMAL.
            self._conn.execute("PRAGMA journal_mode = WAL").fetchone()
            # Codex Phase 2 MEDIUM: bumped from 5s -> 30s. Steady-state
            # ingests are seconds; 30s absorbs incidental collisions
            # between the sessions and github timers without failing loud
            # ``database is locked`` errors. The pathological case (full
            # sessions rebuild holds a savepoint for minutes) is not fixed
            # by any busy_timeout — sequence manually or set
            # ``services.aggregator.sources.github.enable = false`` while
            # rebuilding.
            self._conn.execute("PRAGMA busy_timeout = 30000")
            self._conn.execute("PRAGMA synchronous = NORMAL")
            # BOUND THE ``-wal`` SIDECAR. A checkpoint does not truncate the
            # WAL unless this is set — it merely starts overwriting from the
            # beginning — so a WAL that grew once during a large transaction
            # stays that size on disk forever. 64 MiB is comfortably more than
            # a chunk-sized transaction needs and stops the sidecar tracking
            # the peak of the worst run this database has ever seen.
            self._conn.execute("PRAGMA journal_size_limit = 67108864")
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            # The provenance verdict describes the FILE, which another process
            # may have re-stamped meanwhile. Re-open, re-decide.
            self._vector_quarantine = None
            self._vector_quarantine_decided = False

    def commit(self) -> None:
        """End the open transaction. For a caller composing SEVERAL writes.

        Every write method here commits on its own by default. This exists for
        the one caller that must not let them: the chunk write, which passes
        ``_commit=False`` to the data write AND to the watermark advance so the
        two land as one atomic unit. See ``imports/store_sink.write_checkpoint``.
        """
        self._c().commit()

    def rollback(self) -> None:
        """Discard the open transaction. The other half of :meth:`commit`."""
        self._c().rollback()

    def _ensure_writable(self) -> None:
        if self.read_only:
            raise RuntimeError("read-only Store cannot write")

    # -- schema -----------------------------------------------------------

    def migrate(
        self, *, allow_vector_reindex: bool = False, embedder: object | None = None
    ) -> None:
        """Create tables + FTS virtual tables + triggers. Idempotent.

        ``embedder`` MAKES THE PROVENANCE STAMP TRUTHFUL. A caller that is
        about to fill the index passes the embedder that will do it, and the
        stamp then records what actually wrote the vectors rather than what
        ``AGGREGATOR_EMBED_BACKEND`` implies. See :func:`vector_provenance`.
        Left ``None`` — which is every caller that is not writing vectors —
        the stamp and the comparison both describe what ``Embedder()`` would
        load in this process, which is the right question for a read.

        ``allow_vector_reindex`` IS THE ONLY THING THAT MAY DELETE COMPUTED
        VECTORS WHOLESALE, and it defaults to refusing. It reaches exactly one
        branch of ``_reconcile_vector_provenance``: a provenance mismatch over
        an index that has vectors in it. Everything else here is additive.

        Keyword-only, and every caller but one leaves it alone. This method
        runs on EVERY subcommand — that is what made the previous environment
        variable so dangerous, since a value exported for one deliberate
        rebuild was then honoured by every read that followed it. A parameter
        cannot leak out of the call that passes it. See
        ``VECTOR_REINDEX_COMMAND``.

        Bumps ``PRAGMA user_version`` to SCHEMA_VERSION. A downgraded schema
        won't be silently touched — callers detect via ``user_version`` and
        run ``rebuild_all()`` to drop + recreate.

        v2 → v3 upgrades in place: ``ALTER TABLE sessions ADD COLUMN origin``
        with the ``'claude-code'`` default backfilling every existing row.
        NO rebuild — the live cache.db is hundreds of MB and SQLite ADD
        COLUMN with a constant default is a metadata-only change. The ALTER
        runs BEFORE the DDL pass because the v3 DDL also creates an index on
        ``origin``, which would fail against a v2-shaped table. Guarded by a
        ``PRAGMA table_info`` probe (not user_version) so it only fires when
        the column is genuinely absent; fresh DBs get the column via DDL.

        v4 → v5 upgrades in place on the same terms: two nullable
        ``embedding_state`` columns, then the vec virtual tables — the latter
        ONLY when sqlite-vec loaded. A machine without the extension still
        migrates, still stamps v5, and still serves FTS5; it simply has no vec
        tables until the extension is fixed, at which point the next
        ``migrate()`` (one runs on every CLI invocation) creates them. Nothing
        here touches ``ingest_state``, ``quarantine`` or ``poison_faults``.
        """
        self._ensure_writable()
        c = self._c()
        self._ensure_sessions_origin_column(c)
        self._ensure_src_hash_columns(c)
        self._ensure_embedding_state_columns(c)
        self._ensure_provenance_column(c)
        self._ensure_narrow_fts_update_trigger(c)
        for stmt in _DDL:
            c.executescript(stmt)
        if self._vector_available:
            # BEFORE the vec DDL: a foreign table of the wrong width survives
            # ``CREATE VIRTUAL TABLE IF NOT EXISTS`` untouched, so the check
            # that might drop it has to run first.
            self._reconcile_vector_provenance(c, allow_vector_reindex, embedder)
            if self._vector_quarantine is None:
                for stmt in _VEC_DDL:
                    c.executescript(stmt)
        # THE VERSION ROW, recorded whether or not sqlite-vec loaded. It is
        # plain-table bookkeeping and the moment an operator most needs to know
        # which model this cache is filling is the moment the extension broke.
        # ``DO NOTHING`` so a re-migration never resets a ``completed_at``
        # earned by a finished backfill — ``migrate()`` runs on every
        # subcommand, and an index that un-completed itself on the next
        # ``aggregator query`` would be a pointer that flips at random.
        model, dim = vector_provenance(embedder)
        c.execute(
            "INSERT INTO embedding_versions(model, dim, started_at) "
            "VALUES (?, ?, ?) ON CONFLICT(model) DO NOTHING",
            (model, dim, _now_iso()),
        )
        c.execute(f"PRAGMA user_version = {SCHEMA_VERSION};")
        c.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        c.commit()

    def _reconcile_vector_provenance(
        self,
        c: sqlite3.Connection,
        allow_reindex: bool = False,
        embedder: object | None = None,
    ) -> None:
        """Adopt the vector index only if this build is what produced it.

        WHY EXISTENCE IS NOT ENOUGH. Every other probe in ``migrate()`` asks
        "is this column/table there?", and for a plain column that is a
        complete question. For the vector index it is not: a ``vec0`` table is
        only meaningful together with the model and the width that filled it,
        and neither of those is recoverable from the table itself. So a
        migration that probes existence alone will adopt whatever it finds.

        That is not a hypothetical here. The live cache carries ``vec_*``
        tables and both ``embedding_state`` columns *while stamped v4*, because
        an abandoned 2026-08-08 branch numbered its schema 4 too and ran. The
        three ways that goes wrong, all reproduced:

        * a foreign ``float[1024]`` table survives ``IF NOT EXISTS``, so
          ``migrate()`` succeeds and stamps v5, and then every vector write and
          every KNN read raises ``Dimension mismatch`` — on a 30-minute timer,
          a crash loop;
        * right-width vectors from another model are served as current, with
          nothing on disk saying otherwise;
        * rows pre-marked ``embedding_state='ok'`` over an empty index drain
          the backlog to nothing, so ``vector_index_state`` reports
          ``complete`` while ``vectors`` is 0 and no row is ever re-embedded.

        The stamp closes all three with one comparison. When it matches, this
        is a no-op and the index is kept.

        WHEN IT DOES NOT MATCH, THE ANSWER DEPENDS ON WHAT IS AT STAKE — and
        that distinction is the round-2 S1 fix, because the first version of
        this method always answered "discard". ``migrate()`` runs on EVERY CLI
        invocation, and the stamp is derived from ``AGGREGATOR_EMBED_BACKEND``,
        so a shell that happened to export ``gguf`` turned a read-only
        ``aggregator query`` into an operation that deleted the whole vector
        index — 1.33 GB and 25-30 days of continuous CPU — on a log line, and
        the sentence-transformers-pinned timer then deleted the rebuild on its
        next tick. Nothing about reading implies consent to that. Two cases:

        * **No vectors on disk.** Nothing computed exists, so adopting costs
          nothing and destroys nothing: drop and recreate the vec tables (this
          is what fixes a foreign table of the wrong WIDTH), return every
          non-NULL ``embedding_state`` to the backlog, stamp. This branch is
          not an optimisation — a fresh database reaches this method with no
          stamp and no tables, and refusing there would mean no cache could
          ever bootstrap the arm.
        * **Vectors on disk.** REFUSE. Nothing is dropped, nothing is
          requeued, nothing is stamped. The vector arm is switched off for
          this Store (``_vector_quarantine``), which is the degrade path this
          file already has: vector reads raise
          :class:`VectorIndexUnavailableError`, vector writes no-op,
          ``mark_embedded('ok')`` refuses, ``vector_index_state`` reports
          ``unavailable`` with the reason, and FTS5 keyword search is
          untouched. So a genuine mismatch still cannot serve a vector whose
          model is unknown — it just says so instead of billing a month of CPU
          for the discovery.

        The refusal names both fixes, because they are opposite actions and
        only the operator knows which one is meant: unset the stray env var,
        or — if the model change is real — consent to the rebuild explicitly
        with ``aggregator embed --reindex``, which is the only thing in this
        codebase that may delete computed vectors WHOLESALE.

        ``allow_reindex`` IS THAT CONSENT, and it arrives as an argument
        because round 3 found the environment variable it replaced could not
        be scoped. ``migrate()`` runs on every subcommand, so a variable
        exported for one intended rebuild authorised every command that
        followed it in that shell — a read included. An argument is spent by
        the call that passes it and cannot be left lying around.

        Not wholesale: ingest's per-row invalidation (``_drop_row_vectors``
        when a body is edited, ``_purge_orphan_vectors`` after a rebuild) still
        runs while quarantined, and should. Those delete vectors that are stale
        or ownerless by definition — never because of the stamp — so they
        cannot turn a mismatch into a mass deletion, and stopping them would
        leave the index pointing at bodies that no longer exist.

        Runs only when sqlite-vec loaded — see the caller. Without the
        extension there is no way to inspect or drop the tables, and stamping
        anyway would tell the next migration, the one that CAN see them, that
        this state had already been vouched for.
        """
        # THE WRITE PATH'S QUESTION, so the embedder answers it when there is
        # one: the stamp has to name whatever will actually fill this index.
        model, dim = vector_provenance(embedder)
        expected = json.dumps({"dim": dim, "model": model}, sort_keys=True)
        row = c.execute(
            "SELECT value FROM meta WHERE key = ?", (VECTOR_PROVENANCE_KEY,)
        ).fetchone()
        keyed = self._vec_tables_are_model_keyed(c)
        if row is not None and row[0] == expected and keyed:
            self._adopt_vectors()
            return

        # EXPLICIT CONSENT STILL WINS, and it is checked before the adopt
        # below. Keying on the model means a model change no longer NEEDS a
        # delete — but ``embed --reindex`` exists for the operator who wants
        # one anyway (a corrupt index, a disk reclaim, a chunker change whose
        # old vectors are simply dead weight), and it has already printed what
        # it would destroy and taken a 'y'. Silently adopting under a flag that
        # says "delete" would be its own kind of lie.
        if allow_reindex and not (row is not None and row[0] == expected):
            self._drop_and_restamp(c, expected, row)
            return

        if row is not None and keyed:
            # A MODEL CHANGE OVER A MODEL-KEYED INDEX IS NOT A CONFLICT, and
            # that is criterion E's whole thesis. The refusal below exists
            # because the index used to be a SINGLETON: one vector per chunk,
            # its model implied by the file, so two models could not be told
            # apart at query time and the only safe answers were "refuse" or
            # "delete everything and spend a month recomputing".
            #
            # With ``(chunk_id, model)`` there is nothing to tell apart. The
            # old model's vectors keep their own key, every KNN filters on the
            # partition, and the new model's backfill is a background job that
            # runs to completion beside them. Nothing is deleted and nothing is
            # refused — ``serving_embedding_model`` is what withholds the new
            # index until it is finished.
            #
            # STILL LOUD, because the commonest cause of arriving here is not a
            # deliberate model change: it is ``AGGREGATOR_EMBED_BACKEND``
            # exported in a shell. That no longer destroys anything, but it
            # does start a multi-week backfill for a model nobody wanted, so it
            # has to be visible and the message has to name the one-word fix.
            log.warning(
                "vector index model changed: this cache is stamped %s and this "
                "process is configured for %s. NOTHING WAS DELETED — vectors "
                "are keyed (chunk_id, model), so the existing ones keep "
                "serving any process configured for them and the new model is "
                "filled by a background backfill. If AGGREGATOR_EMBED_BACKEND "
                "is merely exported in this shell then that is the cause and "
                "`unset AGGREGATOR_EMBED_BACKEND` is the whole fix. If the "
                "change is intended, run `aggregator embed --catchup --source "
                "both`; until it completes, vector search is served by the "
                "previously completed index or, if there is none, degrades to "
                "FTS5 keyword search.",
                row[0],
                expected,
            )
            c.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                (VECTOR_PROVENANCE_KEY, expected),
            )
            self._adopt_vectors()
            return

        stamped = "no stamp" if row is None else f"stamped {row[0]}"
        # O(n) on a vec0 table, and deliberately paid only here: the matching
        # stamp returns above, so this runs once, on the mismatch path.
        on_disk = self._count_vectors_on_disk(c)

        if on_disk:
            # AN UNATTRIBUTABLE INDEX. Everything adoptable returned above, so
            # what is left holds vectors this build cannot key to any model:
            # no stamp, or a pre-keying vec table shape. Those cannot be told
            # apart from correct ones at query time, and nobody asked for them
            # to be deleted. Refuse, keep, and say so.
            self._quarantine_vectors(
                _provenance_refusal(stamped, expected, on_disk)
            )
            log.error("%s", self._vector_quarantine)
            return

        self._drop_and_restamp(c, expected, row)

    def _count_vectors_on_disk(self, c: sqlite3.Connection) -> int:
        total = 0
        for table in ("vec_observations", "vec_records"):
            if not _vec_table_usable(c, table):
                continue
            total += c.execute(
                f"SELECT COUNT(*) FROM {table}"  # noqa: S608 - fixed literals
            ).fetchone()[0]
        return total

    def _drop_and_restamp(
        self, c: sqlite3.Connection, expected: str, row: sqlite3.Row | None
    ) -> None:
        """Discard every vector, re-open the backlog, stamp, adopt.

        Reached two ways, and they are opposite in spirit but identical in
        effect: an index with nothing computed in it (adopting costs nothing)
        and an operator who typed ``--reindex`` and answered the prompt. Kept
        as one method so the second can never diverge from the first — the
        ``chunk_embeddings`` and ``embedding_versions`` deletes below were
        exactly the kind of thing a duplicate would forget.
        """
        stamped = "no stamp" if row is None else f"stamped {row[0]}"
        on_disk = self._count_vectors_on_disk(c)
        for stmt in _VEC_DROP_ALL:
            c.execute(stmt)
        # THE INDEX OF RECORD GOES WITH THE BYTES. Leaving ``chunk_embeddings``
        # behind would make the backfill's LEFT JOIN report every row as
        # already embedded against vectors that no longer exist — a corpus that
        # never re-embeds and a vector arm that returns nothing, with every
        # count claiming the index is complete.
        if _table_present(c, "chunk_embeddings"):
            c.execute("DELETE FROM chunk_embeddings")
        if _table_present(c, "embedding_versions"):
            c.execute("DELETE FROM embedding_versions")
        requeued = 0
        for table in ("observations", "records"):
            cur = c.execute(
                f"UPDATE {table} SET embedding_state = NULL "  # noqa: S608 - fixed literals
                "WHERE embedding_state IS NOT NULL"
            )
            requeued += cur.rowcount or 0

        if on_disk or requeued:
            log.warning(
                "discarding a vector index this build did not write "
                "(%s): %d vector(s) dropped and %d row(s) returned to the "
                "backlog to re-embed. The index on disk carried no provenance "
                "stamp, or named a different model or dimension, and vectors "
                "whose model is unknown cannot be told apart from correct ones "
                "at query time. Run `aggregator embed --catchup --source both` "
                "to refill it; FTS5 keyword search is unaffected meanwhile.",
                stamped,
                on_disk,
                requeued,
            )
        c.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            (VECTOR_PROVENANCE_KEY, expected),
        )
        self._adopt_vectors()

    @staticmethod
    def _vec_tables_are_model_keyed(c: sqlite3.Connection) -> bool:
        """Whether the vec tables on disk carry the ``model`` partition key.

        THE SHAPE IS PART OF THE PROVENANCE. A pre-keying ``vec0`` table holds
        one vector per chunk with no record of which model produced it, so a
        second model's vectors cannot be added to it and its existing ones
        cannot be attributed. That is the state the live cache is in — vec
        tables written by an abandoned 2026-08-08 branch — and it is exactly
        the case the refusal below must keep refusing.

        False when there are no vec tables at all: nothing to adopt, and the
        create-and-stamp path is the right one for a fresh cache.
        """
        seen = False
        for table in ("vec_observations", "vec_records"):
            if not _table_present(c, table):
                continue
            seen = True
            try:
                cols = {r[1] for r in c.execute(f"PRAGMA table_info({table})")}
            except sqlite3.OperationalError:
                return False
            if "model" not in cols:
                return False
        return seen

    @property
    def vector_quarantine(self) -> str | None:
        """Why the vector arm is refusing on this store, or ``None``.

        THE PUBLIC READING OF THE S1 VERDICT, and round 3's H2 is what it is
        for. ``_require_vector`` consults the same answer, but only at the
        moment of a vector write — which in the embed worker is AFTER a full
        500-row batch has been embedded, so the refusal arrived as an uncaught
        exception on top of work already thrown away. A caller that wants to
        refuse BEFORE spending anything needs to be able to ask.

        Same cost and caching as ``_require_vector``: one indexed ``meta``
        lookup per connection, and nothing at all on the FTS5-only path.
        """
        return self._vector_quarantine_reason()

    def vector_reindex_preview(self) -> tuple[int, int]:
        """``(vectors that would be deleted, rows that would be re-embedded)``.

        WHAT THE OPERATOR IS SHOWN BEFORE BEING ASKED TO CONFIRM. A destructive
        prompt that cannot say how much it destroys is a prompt people learn to
        answer 'y' to, and the quantity is the whole argument here: this is the
        difference between discarding a cold index and discarding weeks of CPU.

        Deliberately safe to call BEFORE ``migrate()`` — that is exactly when
        the CLI needs it, since the confirmation has to happen before the
        migration that would act on it. Every table is probed for existence
        first, and the vec counts are additionally guarded against the
        ``no such module: vec0`` this file already handles elsewhere: a cache
        that HAS the tables on an interpreter where the extension did not load
        can still answer "nothing would be deleted here".
        """
        c = self._c()
        vectors = 0
        for table in ("vec_observations", "vec_records"):
            if not _vec_table_usable(c, table):
                continue
            vectors += c.execute(
                f"SELECT COUNT(*) FROM {table}"  # noqa: S608 - fixed literals
            ).fetchone()[0]
        rows = 0
        for table in ("observations", "records"):
            if not _table_present(c, table):
                continue
            rows += c.execute(
                f"SELECT COUNT(*) FROM {table} "  # noqa: S608 - fixed literals
                "WHERE embedding_state IS NOT NULL"
            ).fetchone()[0]
        return vectors, rows

    def _adopt_vectors(self) -> None:
        """The vector state on disk is this build's. Trust it."""
        self._vector_quarantine = None
        self._vector_quarantine_decided = True

    def _quarantine_vectors(self, reason: str) -> None:
        """Switch the vector arm off for this Store, keeping the vectors."""
        self._vector_quarantine = reason
        self._vector_quarantine_decided = True

    def _vector_quarantine_reason(self) -> str | None:
        """Why the vector arm must refuse on THIS connection, or ``None``.

        THE READ PATH NEEDS ITS OWN ANSWER, and that is new in round 2. While a
        mismatch was resolved by deleting the vectors, ``migrate()`` was enough:
        the wrong vectors were gone before anything could read them. Refusing
        instead leaves them on disk, so a Store that never migrates — and the
        MCP server is exactly that, ``Store(read_only=True)``, the surface the
        user actually queries through — would go on serving them. Checking here
        is what keeps H1's protection intact under the new answer.

        Cached per connection. The cost is one indexed ``meta`` lookup plus the
        import of ``aggregator.core.embed`` (numpy, not torch — see
        ``vector_provenance``), paid on the first vector operation and never on
        the FTS5-only path.

        An ABSENT stamp over a cache with no vec tables is not a refusal: that
        is a cache migrated on an interpreter without sqlite-vec, and
        ``count_vec_rows`` already has a better message for it. An absent stamp
        with vec tables present IS a refusal — that is precisely the foreign
        state the stamp exists to catch.
        """
        if self._vector_quarantine_decided:
            return self._vector_quarantine
        c = self._c()
        try:
            row = c.execute(
                "SELECT value FROM meta WHERE key = ?", (VECTOR_PROVENANCE_KEY,)
            ).fetchone()
        except sqlite3.OperationalError:
            # No ``meta`` table: nothing has ever migrated this file.
            row = None
        # THE TWO STRUCTURAL ANSWERS COME FIRST, BEFORE THE EXPENSIVE ONE, and
        # the ordering is load-bearing rather than tidy. Computing ``expected``
        # imports ``aggregator.core.embed`` (hence numpy), and this method sits
        # under ``count_vec_rows`` → ``vector_index_state`` → ``capabilities``,
        # which the MCP server calls at CONNECT time — a path
        # ``test_mcp_cold_start`` guards precisely because every import there is
        # paid before the user's first search returns. Both branches below
        # answer without knowing which model this process would load, so on a
        # healthy cache the import never happens at all.
        present = c.execute(
            "SELECT 1 FROM sqlite_master WHERE name IN "
            "('vec_observations', 'vec_records')"
        ).fetchone()
        if row is None and present is None:
            # Nothing has ever written a vector here.
            self._adopt_vectors()
            return None
        if row is not None and self._vec_tables_are_model_keyed(c):
            # MIRRORS THE WRITE PATH. A stamp over a model-keyed index is
            # adoptable whether or not it names THIS model: those vectors are
            # attributable and every KNN filters on the partition, so nothing
            # here can be served by accident. Whether this model may be served
            # is ``serving_embedding_model``'s question, and it answers with a
            # message naming which of the two situations applies.
            self._adopt_vectors()
            return None
        # WHAT IS LEFT CANNOT BE ATTRIBUTED TO ANY MODEL. Both adoptable cases
        # returned above, so this is a vec table with no ``model`` column — the
        # pre-keying shape, which stores one vector per chunk and no record of
        # what produced it — or one with no stamp at all. Neither can be told
        # apart from a correct index at query time.
        #
        # THE MESSAGE IS BUILT FROM DISK FACTS ONLY, with no reference to what
        # THIS process would load, and that is deliberate: naming the expected
        # model means computing it, which imports ``aggregator.core.embed`` and
        # numpy onto the MCP connect path. It also says nothing useful here —
        # the fault is that the vectors name NOBODY, so no expected value makes
        # it better or worse.
        stamped = "no stamp" if row is None else f"stamped {row[0]}"
        self._quarantine_vectors(_unattributable_refusal(stamped))
        return self._vector_quarantine

    @staticmethod
    def _ensure_column(
        c: sqlite3.Connection, table: str, column: str, ddl: str
    ) -> None:
        """Idempotent ``ALTER TABLE .. ADD COLUMN``.

        Probes ``PRAGMA table_info(<table>)`` rather than ``user_version`` so a
        half-applied state (column present, version stale) cannot run the ALTER
        twice. No-op when the table does not exist yet — a fresh database gets
        the column from the ``CREATE TABLE`` in ``_DDL`` instead.

        ``table``, ``column`` and ``ddl`` are interpolated into SQL and are
        never caller-supplied: every call site below passes literals.
        """
        table_exists = c.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if not table_exists:
            return
        cols = {row[1] for row in c.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            c.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")  # noqa: S608 - fixed literals

    @staticmethod
    def _ensure_sessions_origin_column(c: sqlite3.Connection) -> None:
        """v2 → v3 in-place upgrade: add ``sessions.origin`` when absent.

        Probes ``PRAGMA table_info(sessions)`` rather than ``user_version``
        so a half-applied state (column present, version stale) can't run
        the ALTER twice. No-op on fresh DBs (no sessions table yet — DDL
        creates it with the column) and on already-v3 DBs.
        """
        Store._ensure_column(
            c, "sessions", "origin", "TEXT NOT NULL DEFAULT 'claude-code'"
        )

    @staticmethod
    def _ensure_src_hash_columns(c: sqlite3.Connection) -> None:
        """v3 → v4 in-place upgrade: add ``src_hash`` to records + observations.

        ``ADD COLUMN`` with no default is metadata-only in SQLite — it does not
        touch a single existing page. That is not an optimisation detail here:
        the live database holds ~372k observations, and a rewriting migration
        would be precisely the hours-long unattended operation this schema
        change exists to abolish.

        NO BACKFILL, deliberately. Every pre-v4 row arrives with
        ``src_hash IS NULL``, which never matches a computed hash, so the first
        ingest after the upgrade re-scrubs and re-writes it once and stamps the
        hash. Self-healing, spread over exactly one run, and — because the
        writes are chunked and checkpointed — interruptible.

        Probes ``PRAGMA table_info`` rather than ``user_version`` so a
        half-applied state cannot run the ALTER twice.
        """
        for table in ("records", "observations", "sessions"):
            Store._ensure_column(c, table, "src_hash", "TEXT")

    @staticmethod
    def _ensure_embedding_state_columns(c: sqlite3.Connection) -> None:
        """v4 → v5 in-place upgrade: add ``embedding_state`` where absent.

        NULL FOR EVERY EXISTING ROW, and that is the whole contract. NULL is
        what ``select_unembedded`` selects on, so a pre-v5 corpus comes out of
        the migration QUEUED FOR BACKFILL. Giving the column a non-NULL default
        would be the same bug as advancing a watermark past data that was never
        processed: the embed worker would find nothing to do, the vec tables
        would stay empty, and hybrid retrieval would degrade to keyword-only
        while every status surface reported a healthy, fully-embedded index.

        Metadata-only, like the v4 ALTERs above: ``ADD COLUMN`` with no default
        does not rewrite a single page of the 372k-row observations table.

        Must run BEFORE the ``_DDL`` pass — v5 DDL creates indexes ON
        ``embedding_state``, which cannot exist against a v4-shaped table.
        """
        Store._ensure_column(c, "observations", "embedding_state", "TEXT")
        Store._ensure_column(c, "records", "embedding_state", "TEXT")

    @staticmethod
    def _ensure_provenance_column(c: sqlite3.Connection) -> None:
        """v5 → v6 in-place upgrade: add ``observations.provenance``.

        NULL FOR EVERY EXISTING ROW, and that is the contract — the same one
        ``_ensure_embedding_state_columns`` states one method up. NULL means
        "not classified yet" and is what the standalone backfill selects on, so
        a pre-v6 corpus comes out of this migration QUEUED FOR CLASSIFICATION.
        A non-NULL default would make every row claim an authorship no
        classifier ever assigned it, and the backfill would find nothing to do.

        Metadata-only: ``ADD COLUMN`` with no default rewrites no page.
        Measured on a throwaway database at 1/5 of live scale at **0.006 s**
        and +4 KB — the column is free. What is not free is UPDATING it, which
        is why the FTS trigger narrowing (see
        :meth:`_ensure_narrow_fts_update_trigger`) ships in the same change.

        NOTHING BACKFILLS HERE, deliberately. ``migrate()`` runs on every
        subcommand including read-only queries; a classification pass over
        549,952 rows behind an ``aggregator query`` would be exactly the
        hours-long surprise this schema style exists to abolish. The pass is
        ``aggregator provenance --backfill``, which is resumable and chunked.

        Must run BEFORE the ``_DDL`` pass, which creates an index ON
        ``provenance`` that cannot exist against a v5-shaped table.
        """
        Store._ensure_column(c, "observations", "provenance", "TEXT")

    @staticmethod
    def _ensure_narrow_fts_update_trigger(c: sqlite3.Connection) -> None:
        """Replace a wide ``observations_au`` with the ``OF body`` form.

        DROP AND LET THE DDL PASS RE-CREATE IT, because
        ``CREATE TRIGGER IF NOT EXISTS`` does not replace an existing trigger —
        it is a no-op against one — so shipping the narrow text in ``_DDL``
        alone would fix only databases that do not exist yet. Precedent for
        dropping a trigger by name is in ``_DROP_ALL``.

        PROBES THE ARTIFACT, NOT ``user_version``, for the same reason every
        ``_ensure_*`` helper above does: the stored SQL is the thing that
        decides how the trigger behaves, and a half-applied state (trigger
        replaced, version stale, or the reverse) must converge rather than skip.
        No-op when the trigger is already narrow, and when it is absent
        entirely — a fresh database gets it from ``_DDL``.

        Must run BEFORE the ``_DDL`` pass, which is what puts the new one back.
        """
        row = c.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'trigger' AND name = 'observations_au'"
        ).fetchone()
        if row is None or row[0] is None:
            return
        if "UPDATE OF body" in row[0]:
            return
        c.execute("DROP TRIGGER observations_au")

    def schema_version(self) -> int:
        """Return the DB's ``PRAGMA user_version`` (0 for a fresh file)."""
        c = self._c()
        row = c.execute("PRAGMA user_version").fetchone()
        return int(row[0]) if row else 0

    def rebuild_all(self, *, allow_vector_reindex: bool = False) -> None:
        """Drop every table (records, sessions, observations, FTS shadows,
        meta) and re-run DDL. Migration escape hatch: SQLite is a derived
        index; JSONLs / API responses are source of truth. Callers detecting
        a stale ``user_version`` invoke this before re-ingesting.

        ``allow_vector_reindex`` IS THE SAME CONSENT ``migrate()`` TAKES, and
        this method needing it is round 4's confirmed finding. The rows this
        drops are re-fetched in minutes; the vectors are not, and on this
        hardware refilling them is a multi-week operation
        (``docs/embedding-throughput.md``). Rounds 1-3 built a preview, a
        stdin prompt and a per-call argument in front of ``migrate()`` while
        this method ran ``_VEC_DROP_ALL`` unconditionally — so "``migrate()``
        is the only wholesale delete" was a claim with a documented
        counter-example in ``scripts/reingest_v2.py`` line 32.

        DELIBERATELY THE SAME PARAMETER NAME AND THE SAME REFUSAL TEXT. A
        second vocabulary for the same decision is how the first gap opened:
        a reader auditing "who may delete the index" greps for one spelling.

        Only refuses when there is something to lose. A cold index, an absent
        one, and a cache on an interpreter without sqlite-vec all rebuild
        without a question — prompting where nothing is at stake is how a
        prompt that matters stops being read.
        """
        self._ensure_writable()
        c = self._c()
        if not allow_vector_reindex:
            vectors, rows = self.vector_reindex_preview()
            if vectors:
                raise VectorReindexNotConsentedError(
                    f"refusing to rebuild this cache: it holds {vectors} "
                    f"vector(s) that rebuild_all() would DELETE, and dropping "
                    f"the vector index is not implied by re-ingesting the "
                    f"rows. NOTHING WAS DELETED — the vectors on disk are "
                    f"intact and so is the {rows} row(s) of embed watermark. "
                    f"The rows come back from their sources in minutes; the "
                    f"vectors are recomputed, and on this hardware that is a "
                    f"multi-week operation (see "
                    f"docs/embedding-throughput.md). If the vector index "
                    f"genuinely has to go, say so on the call: "
                    f"rebuild_all(allow_vector_reindex=True), or use "
                    f"`{VECTOR_REINDEX_COMMAND}`, which prints what it would "
                    f"delete and asks for a 'y' first."
                )
        # Vec tables first, and only when the module that owns them is loaded:
        # SQLite runs the vtab's xDestroy to drop one, so ``DROP TABLE IF
        # EXISTS vec_observations`` raises ``no such module: vec0`` rather than
        # no-opping when sqlite-vec is missing. Skipping is correct — without
        # the extension the tables were never created either.
        if self._vector_available:
            for stmt in _VEC_DROP_ALL:
                c.execute(stmt)
        for stmt in _DROP_ALL:
            c.execute(stmt)
        c.commit()
        # Nuke connection state so cached PRAGMAs (foreign_keys) etc. don't
        # linger against dropped tables. Force a fresh connection.
        self.close()
        self.migrate()

    # -- writes: v2 sessions + observations -------------------------------

    def upsert_entities(
        self,
        entities: Iterable[SessionEntity],
        *,
        _commit: bool = True,
    ) -> int:
        """Write ``SessionRow`` + ``ObservationRow`` items into the v2 tables.

        Idempotent per primary key. Scrubs ``ObservationRow.body`` pre-write.
        Sessions carry no user-authored text (only metadata) so scrub isn't
        applied there — cwd / git_branch are structural.

        Session rows must precede their observation rows in the iterable so
        the FK check passes (the caller — ``sessions.py::iter_entities`` —
        yields the session row first, then all its observations). If a caller
        yields out of order, the FK error surfaces immediately.

        ``_commit=False`` skips the trailing ``COMMIT`` so
        ``rebuild_and_upsert_entities`` can nest the writes under its
        SAVEPOINT (COMMIT inside a savepoint releases it, which then breaks
        the surrounding RELEASE).

        A batch that raises part-way through is ROLLED BACK. sqlite3 opens an
        implicit transaction before the first DML and holds it until someone
        ends it; without this, the rows written before the raise sat pending
        on the shared connection and the NEXT writer's COMMIT published them.
        They land while every count that described them was discarded with the
        exception — rows in the store that no report ever mentioned.

        Only when this call owns the transaction. Under ``_commit=False`` the
        caller holds a SAVEPOINT and does its own ``ROLLBACK TO SAVEPOINT``; a
        connection-wide rollback here would destroy the savepoint it is about
        to release.

        RETURNS THE NUMBER OF ROWS THAT DID NOT HAVE TO BE WRITTEN — the ones
        whose ``src_hash`` already matched. It is the number worth watching:
        direct evidence that a re-run is costing what a re-run should cost, and
        the difference between "resumable in principle" and "resumable in an
        amount of time a 30-minute timer can absorb".
        """
        self._ensure_writable()
        c = self._c()
        try:
            unchanged = self._do_write_entities(c, entities)
        except BaseException:
            if _commit:
                c.rollback()
            raise
        if _commit:
            c.commit()
        return unchanged

    @staticmethod
    def _do_write_entities(
        c: sqlite3.Connection, entities: Iterable[SessionEntity]
    ) -> int:
        """The entity write loop. Transaction handling belongs to the caller.

        Streamed in bounded groups so the hash probe can be one indexed query
        per table per group instead of one per row, WITHOUT materialising the
        caller's iterable — which for the sessions source is ~372k rows.
        """
        unchanged = 0
        for group in _chunked(entities, _HASH_PROBE_CHUNK):
            unchanged += Store._write_entity_group(c, group)
        return unchanged

    @staticmethod
    def _write_entity_group(
        c: sqlite3.Connection, group: Sequence[SessionEntity]
    ) -> int:
        """Write one group, skipping rows whose fingerprint already matches.

        THE SKIP HAPPENS BEFORE THE SCRUB, which is the whole reason the
        fingerprint is computed over the source's row rather than the stored
        one. Presidio is the expensive step by an order of magnitude.

        ORDER IS PRESERVED. ``observations.session_id`` is a real foreign key
        with ``PRAGMA foreign_keys`` ON, so a session row must still be written
        before its observations. A session skipped as unchanged is by
        definition already in the table, so the constraint holds either way.

        A REPEATED ID INSIDE ONE GROUP compares against the probe, not against
        what an earlier item in the same group just wrote. Two identical copies
        both count as unchanged (correct); two differing copies both write, and
        the last one wins (correct, at the cost of one extra write).
        """
        session_ids = [e.session_id for e in group if isinstance(e, SessionRow)]
        obs_ids = [e.obs_id for e in group if isinstance(e, ObservationRow)]
        stored_sessions = _stored_hashes(c, "sessions", "session_id", session_ids)
        stored_obs = _stored_hashes(c, "observations", "obs_id", obs_ids)
        # Probed ONCE per group, and it asks whether the table can be WRITTEN,
        # not merely whether it exists. A cache migrated without sqlite-vec has
        # no vec tables at all; a cache FILLED with the extension and then
        # opened without it has them and cannot touch them. Both must leave the
        # row write and the watermark reset below untouched — see
        # ``_vec_table_usable``.
        vec_present = _vec_table_usable(c, "vec_observations")

        unchanged = 0
        for e in group:
            if isinstance(e, SessionRow):
                # NO ``SCRUB_FINGERPRINT`` here: session rows carry no
                # user-authored text (cwd and git_branch are structural) and are
                # never scrubbed, so a change to the scrubber cannot change what
                # is stored for them and must not cost 8k pointless rewrites.
                digest = _src_hash(
                    e.session_id,
                    e.root_session_id,
                    e.parent_session_id,
                    e.kind,
                    e.agent_id,
                    e.agent_type,
                    e.spawned_by_tool_use_id,
                    e.cwd,
                    e.git_branch,
                    _iso(e.first_ts),
                    _iso(e.last_ts),
                    e.jsonl_path,
                    e.origin,
                )
                if stored_sessions.get(e.session_id) == digest:
                    unchanged += 1
                    continue
                c.execute(
                    """
                    INSERT INTO sessions(
                        session_id, root_session_id, parent_session_id, kind,
                        agent_id, agent_type, spawned_by_tool_use_id,
                        cwd, git_branch, first_ts, last_ts, jsonl_path, origin,
                        src_hash
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        root_session_id        = excluded.root_session_id,
                        parent_session_id      = excluded.parent_session_id,
                        kind                   = excluded.kind,
                        agent_id               = excluded.agent_id,
                        agent_type             = excluded.agent_type,
                        spawned_by_tool_use_id = excluded.spawned_by_tool_use_id,
                        cwd                    = excluded.cwd,
                        git_branch             = excluded.git_branch,
                        first_ts               = excluded.first_ts,
                        last_ts                = excluded.last_ts,
                        jsonl_path             = excluded.jsonl_path,
                        origin                 = excluded.origin,
                        src_hash               = excluded.src_hash
                    """,
                    (
                        e.session_id,
                        e.root_session_id,
                        e.parent_session_id,
                        e.kind,
                        e.agent_id,
                        e.agent_type,
                        e.spawned_by_tool_use_id,
                        e.cwd,
                        e.git_branch,
                        e.first_ts.isoformat(),
                        e.last_ts.isoformat(),
                        e.jsonl_path,
                        e.origin,
                        digest,
                    ),
                )
            elif isinstance(e, ObservationRow):
                # ``provenance`` IS DELIBERATELY ABSENT FROM THIS DIGEST, and
                # the ``SCRUB_FINGERPRINT`` comment above actively invites the
                # opposite — so this says no in the place someone would say yes.
                #
                # A classifier is not a scrubber. Putting its output in here,
                # or adding a PROVENANCE_FINGERPRINT beside SCRUB_FINGERPRINT,
                # would make every classifier revision change all 549,952
                # digests. The next ingest then takes the ``DO UPDATE`` branch
                # for every row, which re-runs Presidio on each — ~11 hours at
                # this repo's measured 827 rows/min — and sets
                # ``embedding_state = NULL`` a few lines below, discarding the
                # observation vector arm. That costs nothing TODAY only because
                # that arm is still cold; the moment the embed backlog reaches
                # observations the same edit silently throws away weeks of CPU.
                #
                # The correct wiring is what is here: provenance is written on
                # INSERT and on the ``DO UPDATE`` branch (which is rewriting the
                # row anyway), and is otherwise owned by
                # ``aggregator provenance --backfill``, whose cursor is
                # ``provenance IS NULL``.
                digest = _src_hash(
                    SCRUB_FINGERPRINT,
                    e.obs_id,
                    e.session_id,
                    e.root_session_id,
                    e.parent_obs_id,
                    e.type,
                    _iso(e.ts),
                    e.model,
                    e.input_tokens,
                    e.output_tokens,
                    e.tool_name,
                    e.tool_use_id,
                    e.body,
                )
                if stored_obs.get(e.obs_id) == digest:
                    unchanged += 1
                    continue
                scrubbed_body = scrub(e.body).text if e.body else ""
                # UPSERT, NOT DELETE-THEN-INSERT. The old shape allocated a
                # fresh rowid and a fresh page for a row whose content had not
                # moved, and SQLite's default ``auto_vacuum=NONE`` never hands
                # the freed pages back — measured on the live database as 929 MB
                # -> 943 MB in 40 minutes at a completely flat row count, with
                # ``max(rowid)`` climbing 547,998 -> 569,289. The ``_au``
                # trigger keeps ``obs_fts`` in step on the UPDATE branch exactly
                # as the ``_ad``/``_ai`` pair did on the delete-insert one.
                c.execute(
                    """
                    INSERT INTO observations(
                        obs_id, session_id, root_session_id, parent_obs_id,
                        type, ts, model, input_tokens, output_tokens,
                        tool_name, tool_use_id, body, src_hash, provenance
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(obs_id) DO UPDATE SET
                        session_id      = excluded.session_id,
                        root_session_id = excluded.root_session_id,
                        parent_obs_id   = excluded.parent_obs_id,
                        type            = excluded.type,
                        ts              = excluded.ts,
                        model           = excluded.model,
                        input_tokens    = excluded.input_tokens,
                        output_tokens   = excluded.output_tokens,
                        tool_name       = excluded.tool_name,
                        tool_use_id     = excluded.tool_use_id,
                        body            = excluded.body,
                        src_hash        = excluded.src_hash,
                        -- The body moved, so who wrote it is worth re-stating
                        -- from the source that just produced it. Reached ONLY
                        -- when the digest moved, which provenance is not part
                        -- of — a classifier revision alone never gets here.
                        provenance      = excluded.provenance,
                        -- BACK INTO THE EMBED BACKLOG. The body just changed,
                        -- so any vector held for this row describes text that
                        -- is no longer here. ``observations_au`` keeps obs_fts
                        -- in step for the keyword arm; the vector arm has no
                        -- trigger, and ``select_unembedded`` only ever looks at
                        -- NULL, so without this the row keeps the embedding of
                        -- its OLD body permanently and the two arms of the same
                        -- hybrid query disagree about what it says.
                        embedding_state = NULL
                    """,
                    (
                        e.obs_id,
                        e.session_id,
                        e.root_session_id,
                        e.parent_obs_id,
                        e.type,
                        e.ts.isoformat(),
                        e.model,
                        e.input_tokens,
                        e.output_tokens,
                        e.tool_name,
                        e.tool_use_id,
                        scrubbed_body,
                        digest,
                        e.provenance,
                    ),
                )
                # ONLY for a row that already existed. A brand-new obs_id has
                # no vector to drop, and this sits in the ingest hot loop.
                if vec_present and e.obs_id in stored_obs:
                    _drop_row_vectors(c, "vec_observations", "obs_id", e.obs_id)
            else:
                raise TypeError(f"unknown entity type: {type(e)!r}")
        return unchanged

    def rebuild_and_upsert_entities(
        self,
        entities: Iterable[SessionEntity],
        *,
        min_sessions: int = 0,
        origins: Sequence[str] | None = None,
    ) -> None:
        """Atomic replacement of session + observation rows.

        Materializes the iterable before opening the savepoint so
        ``min_sessions`` guards against silent wipes on transient parse
        failure — same round-3 HIGH pattern as the Record-shaped path.
        Sessions source is monolithic (one call across all JSONLs) so no
        per-file granularity is exposed here.

        ``origins`` SCOPES the DELETE. The sessions and observations tables
        hold three populations that no single source can regenerate for the
        others: ``claude-code`` (rebuildable — the JSONLs are still on disk),
        ``chatgpt`` and ``claude-web`` (NOT rebuildable — their only source is
        a vendor export archive a human downloads by hand, and once the drop
        is gone the rows are the last copy). An unscoped rebuild driven by the
        sessions source therefore deletes rows it cannot put back. Callers
        that know which population they are re-scanning pass it here; the
        DELETE then cannot reach the others. ``None`` keeps the historical
        whole-table behaviour and is only for a caller that genuinely means
        every origin.
        """
        materialised = list(entities)
        session_count = sum(1 for e in materialised if isinstance(e, SessionRow))
        if session_count < min_sessions:
            raise EmptyRebuildRefusedError(
                f"refusing to rebuild sessions: got {session_count} session "
                f"rows, min_sessions={min_sessions}"
            )
        if origins is not None:
            scope = list(origins)
            stray = sorted(
                {
                    e.origin
                    for e in materialised
                    if isinstance(e, SessionRow) and e.origin not in scope
                }
            )
            if stray:
                # The incoming rows would be INSERTed while the DELETE never
                # reached their existing counterparts — a rebuild that is not
                # a rebuild for those origins. Refuse rather than half-apply.
                raise EmptyRebuildRefusedError(
                    f"refusing to rebuild origins {scope}: the entity stream "
                    f"also carries origin(s) {stray}, which the scoped DELETE "
                    f"would not replace"
                )
        self._ensure_writable()
        c = self._c()
        c.execute("SAVEPOINT rebuild_entities")
        try:
            self._delete_entity_rows(c, origins)
            # _commit=False: don't COMMIT inside the savepoint (would release
            # it prematurely and break the surrounding RELEASE).
            self.upsert_entities(materialised, _commit=False)
            # AFTER the re-write, deliberately. A row that came straight back
            # keeps the vector it already had, so recall survives the rebuild
            # window instead of going dark until the next backfill catches up.
            # Only rows that genuinely did not return lose theirs.
            _purge_orphan_vectors(c, "observations")
        except BaseException:
            c.execute("ROLLBACK TO SAVEPOINT rebuild_entities")
            c.execute("RELEASE SAVEPOINT rebuild_entities")
            raise
        c.execute("RELEASE SAVEPOINT rebuild_entities")
        c.commit()

    @staticmethod
    def _delete_entity_rows(
        c: sqlite3.Connection, origins: Sequence[str] | None
    ) -> None:
        """Clear sessions + observations, optionally scoped to ``origins``.

        Observations first: ``observations.session_id`` is a real FK and
        ``PRAGMA foreign_keys`` is ON, so dropping the parents first would
        abort the statement.
        """
        if origins is None:
            c.execute("DELETE FROM observations")
            c.execute("DELETE FROM sessions")
            return
        scope = list(origins)
        if not scope:
            return
        placeholders = ",".join("?" * len(scope))
        c.execute(
            "DELETE FROM observations WHERE session_id IN "
            f"(SELECT session_id FROM sessions WHERE origin IN ({placeholders}))",
            scope,
        )
        c.execute(f"DELETE FROM sessions WHERE origin IN ({placeholders})", scope)

    # -- v5 vector index: writes, reads, and the embed watermark -----------
    #
    # EVERY ENTRY POINT HERE IS GUARDED ON ``vector_available``, and the two
    # halves are guarded differently on purpose:
    #
    # * Writes NO-OP with a warning. The embed worker's job is to make forward
    #   progress over a 372k-row backlog; a missing extension should stop it
    #   writing vectors, not abort the run — and crucially it must not advance
    #   ``embedding_state``, so the backlog is still there to embed once the
    #   extension is fixed.
    # * Reads RAISE, by name. A retrieval caller that silently got an empty
    #   vector arm would degrade to keyword-only while reporting a hybrid
    #   search, which is the "index rots and nobody finds out" failure this
    #   project exists to prevent. The caller decides whether to fall back;
    #   it cannot decide what it is never told.
    #
    # ``select_unembedded`` / ``mark_embedded`` are NOT guarded: they are
    # ordinary reads and writes of a plain column and have no business
    # depending on a native extension.

    # -- the (chunk_id, model) index of record -----------------------------

    @property
    def embedding_model(self) -> str:
        """The version string vectors written by THIS process are keyed on.

        Deliberately the same value the provenance stamp holds — one identity,
        one place it is computed. See
        :func:`aggregator.core.embed.embedding_version`.
        """
        return vector_provenance()[0]

    def _stamped_model(self) -> str | None:
        """The model THIS CACHE's vectors are keyed on, read off disk.

        NO IMPORT, and that is the whole reason it exists beside
        ``embedding_model``. That property answers "what would this process
        load", which needs ``aggregator.core.embed`` and therefore numpy — and
        ``capabilities()`` reaches it, which the MCP server calls at connect
        time on a path ``test_mcp_cold_start`` guards precisely because every
        import there is paid before the user's first search returns.

        The stamp is a string in ``meta``: one indexed lookup, no module graph.
        It is also the better question for a status count, which is asking how
        far the index ON DISK has got rather than what this process would build.
        ``None`` when nothing has ever stamped this cache.
        """
        c = self._c()
        try:
            row = c.execute(
                "SELECT value FROM meta WHERE key = ?", (VECTOR_PROVENANCE_KEY,)
            ).fetchone()
        except sqlite3.OperationalError:
            row = None
        if row is not None:
            try:
                return str(json.loads(row[0])["model"])
            except (ValueError, KeyError, TypeError):
                pass
        # NO STAMP IS NOT THE SAME AS NO ANSWER. The provenance stamp is only
        # written when sqlite-vec loaded, and a cache whose extension broke
        # AFTER a backfill is exactly when an operator most needs the counts.
        # ``embedding_versions`` is a plain table written on every migrate, so
        # it can answer where the stamp cannot — still without an import.
        if not _table_present(c, "embedding_versions"):
            return None
        latest = c.execute(
            "SELECT model FROM embedding_versions ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        return None if latest is None else str(latest["model"])

    def embedding_version_state(self, model: str | None = None) -> dict:
        """``{model, dim, started_at, completed_at}`` for a version row.

        ``completed_at is None`` means the backfill for that model has not
        drained yet. Returns the row for the CONFIGURED model by default.
        """
        model = model or self.embedding_model
        c = self._c()
        if not _table_present(c, "embedding_versions"):
            return {"model": model, "dim": None, "started_at": None, "completed_at": None}
        row = c.execute(
            "SELECT model, dim, started_at, completed_at FROM embedding_versions "
            "WHERE model = ?",
            (model,),
        ).fetchone()
        if row is None:
            return {"model": model, "dim": None, "started_at": None, "completed_at": None}
        return dict(row)

    def serving_embedding_model(self) -> str | None:
        """The model the vector arm may ANSWER FROM, or ``None`` to refuse.

        RULE 4 OF THE REFERENCE DESIGN: flip a pointer, do not expose a
        half-built index. A partially filled embedding space does not fail
        loudly — it answers from whichever fraction of the corpus happened to
        be embedded first, with scores that look exactly like complete ones.
        Nothing in a result says it was drawn from a tenth of the index, so
        the failure is invisible at exactly the moment it matters.

        THE BOOTSTRAP EXCEPTION IS DELIBERATE AND NAMED. If NO version has
        ever completed on this cache, the in-progress one is served. There is
        no previous index to fall back to, so refusing would mean no vector arm
        at all for the length of the first backfill — weeks, on this hardware —
        and the project's own design has always been "a background worker with
        a watermark, so queries never block on the backfill". The watermark
        (``count_embedding_states``) is what reports how far it got. The moment
        any version completes, that argument expires: a later incomplete
        version now has something better to defer to, and stays invisible until
        it finishes.

        Returns ``None`` when the configured model has no version row at all —
        which is a cache whose vectors were written by something else, and is
        the same refusal the provenance stamp makes for the same reason.
        """
        c = self._c()
        if not _table_present(c, "embedding_versions"):
            # A cache migrated before this table existed. The provenance stamp
            # is the guard there; refusing on the absence of bookkeeping that
            # was never written would take the arm down for no gain.
            return self.embedding_model
        model = self.embedding_model
        row = c.execute(
            "SELECT completed_at FROM embedding_versions WHERE model = ?", (model,)
        ).fetchone()
        if row is None:
            return None
        if row["completed_at"] is not None:
            return model
        other = c.execute(
            "SELECT 1 FROM embedding_versions WHERE completed_at IS NOT NULL "
            "AND model <> ? LIMIT 1",
            (model,),
        ).fetchone()
        return None if other is not None else model

    def mark_embedding_version_complete(self) -> str:
        """Flip ``completed_at`` for the configured model. Returns the model.

        REFUSES OVER A BACKLOG THAT IS NOT EMPTY, because a pointer that can be
        flipped early is not a pointer, it is a comment. The check is the same
        LEFT JOIN the worker drains against, so "complete" means precisely
        "that query returns nothing" and cannot drift from it.

        Idempotent: flipping an already-complete version keeps the original
        timestamp, so a second ``--catchup`` over a drained backlog does not
        rewrite when the index was finished.
        """
        self._ensure_writable()
        model = self.embedding_model
        outstanding = 0
        for kind in self._VEC_TABLES:
            outstanding += len(self.select_unembedded(kind, limit=1))
        if outstanding:
            raise RuntimeError(
                f"refusing to mark the embedding index complete: "
                f"{outstanding} kind(s) still have rows with no vector under "
                f"{model!r}. `completed_at` is what keeps a half-built index "
                f"from being served, so setting it over a live backlog would "
                f"publish a partial embedding space that answers with "
                f"plausible scores. Run `aggregator embed --catchup --source "
                f"both` to drain it first."
            )
        c = self._c()
        c.execute(
            "UPDATE embedding_versions SET completed_at = ? "
            "WHERE model = ? AND completed_at IS NULL",
            (_now_iso(), model),
        )
        c.commit()
        return model

    def record_chunk_embedding(
        self,
        kind: str,
        *,
        chunk_id: str,
        owner_id: str,
        embedding: np.ndarray,
        content_sha256: str | None = None,
        model: str | None = None,
        _commit: bool = True,
    ) -> None:
        """Write one ``(chunk_id, model)`` embedding — bytes AND index row.

        ONE CALL WRITES BOTH, deliberately. A vector in the vec table with no
        ``chunk_embeddings`` row is invisible to the backfill query, so it gets
        computed again forever; a row with no vector is a hole the KNN cannot
        fill while every count says the index is complete. They are two halves
        of one fact and there is no correct interleaving of two separate calls.

        ``model`` defaults to this build's version string. It is a parameter so
        a caller can register another model's vectors — which is what makes
        "two models coexist as extra rows" a thing this store can express
        rather than a thing it merely permits.
        """
        self._ensure_writable()
        vec_table = self._VEC_TABLES.get(kind)
        if vec_table is None:
            raise ValueError(f"unknown kind: {kind!r}")
        vec_key = "obs_id" if kind == "observations" else "stable_id"
        model = model or self.embedding_model
        c = self._c()
        if self._vector_writes_enabled():
            c.execute(
                f"DELETE FROM {vec_table} WHERE model = ? AND {vec_key} = ?",  # noqa: S608 - allowlisted literals
                (model, chunk_id),
            )
            c.execute(
                f"INSERT INTO {vec_table}(model, {vec_key}, embedding) "  # noqa: S608 - allowlisted literals
                f"VALUES (?, ?, ?)",
                (model, chunk_id, embedding.astype("float32").tobytes()),
            )
            self._record_chunk_row(
                c, kind, chunk_id, owner_id, model, content_sha256
            )
        if _commit:
            c.commit()

    @staticmethod
    def _record_chunk_row(
        c: sqlite3.Connection,
        kind: str,
        chunk_id: str,
        owner_id: str,
        model: str,
        content_sha256: str | None,
    ) -> None:
        """The index half of :meth:`record_chunk_embedding`, without a commit.

        ``DO UPDATE`` rather than ``DO NOTHING`` on conflict: the bytes above
        were just overwritten, so the hash describing them has to move with
        them or the reuse path would hand out a vector for text it no longer
        holds. Re-running the job over UNCHANGED content still writes the same
        values, so "twice equals once" survives.
        """
        c.execute(
            "INSERT INTO chunk_embeddings("
            "  chunk_id, model, kind, owner_id, dim, content_sha256, created_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(chunk_id, model) DO UPDATE SET "
            "  kind = excluded.kind,"
            "  owner_id = excluded.owner_id,"
            "  dim = excluded.dim,"
            "  content_sha256 = excluded.content_sha256",
            (chunk_id, model, kind, owner_id, _VEC_DIM, content_sha256, _now_iso()),
        )

    def reusable_chunk_vectors(
        self, content_hashes: Sequence[str], model: str | None = None
    ) -> dict[str, np.ndarray]:
        """``{content_sha256: embedding}`` for text already embedded here.

        RULE 2 OF THE REFERENCE DESIGN, and on this corpus it is close to the
        whole cost of a re-index. When a document changes in one paragraph most
        chunk hashes are unchanged; ours are chat transcripts that get APPENDED
        to rather than rewritten, so every chunk but the last is byte-identical
        after an edit and its vector is still exactly right.

        SCOPED TO ONE MODEL, always. A hash match under a different model is
        the same text in a different embedding space, and handing that back
        would be the silent mixing this whole key exists to prevent.

        A ``NULL`` hash never matches: it means the vector was written through
        a path that never saw the text, so nothing can prove the content is the
        same, and the honest answer to an unprovable reuse is to embed again.
        """
        if not content_hashes:
            return {}
        c = self._c()
        if not _table_present(c, "chunk_embeddings"):
            return {}
        model = model or self.embedding_model
        out: dict[str, np.ndarray] = {}
        for page in _chunked(list(dict.fromkeys(content_hashes)), 400):
            placeholders = ",".join("?" * len(page))
            rows = c.execute(
                f"SELECT chunk_id, kind, content_sha256 FROM chunk_embeddings "  # noqa: S608 - fixed literals
                f"WHERE model = ? AND content_sha256 IN ({placeholders})",
                (model, *page),
            ).fetchall()
            for row in rows:
                if row["content_sha256"] in out:
                    continue
                vec = self._chunk_vector(row["kind"], row["chunk_id"], model)
                if vec is not None:
                    out[row["content_sha256"]] = vec
        return out

    def _chunk_vector(
        self, kind: str, chunk_id: str, model: str
    ) -> np.ndarray | None:
        """The stored bytes for one ``(chunk_id, model)``, or ``None``."""
        vec_table = self._VEC_TABLES.get(kind)
        if vec_table is None or not self.vector_available:
            return None
        vec_key = "obs_id" if kind == "observations" else "stable_id"
        try:
            row = self._c().execute(
                f"SELECT embedding FROM {vec_table} "  # noqa: S608 - allowlisted literals
                f"WHERE model = ? AND {vec_key} = ?",
                (model, chunk_id),
            ).fetchone()
        except sqlite3.OperationalError:
            return None
        if row is None:
            return None
        # LOCAL IMPORT, like ``vector_provenance``'s. ``store`` is on the MCP
        # cold-start path and numpy is deliberately TYPE_CHECKING-only at
        # module scope; this is the one method here that has to MAKE an array
        # rather than accept one.
        import numpy

        return numpy.frombuffer(row["embedding"], dtype=numpy.float32)

    def upsert_vec_observations(
        self, rows: list[tuple[str, np.ndarray]]
    ) -> int:
        """Upsert ``(obs_id, embedding)`` rows. Returns how many were WRITTEN.

        Delete-then-insert to keep the upsert idempotent: vec0 virtual tables
        do not support ``ON CONFLICT``.

        The return value is what makes the degrade path checkable from outside:
        0 means the vectors were discarded because the extension is missing,
        which is otherwise indistinguishable from success.

        REGISTERS EACH VECTOR IN ``chunk_embeddings`` TOO — see
        :meth:`record_chunk_embedding` for why the two are one write. This
        entry point never sees the text, so it can prove nothing about content
        and stores no hash: these vectors are correct, but they are not
        reusable across a re-chunk.
        """
        return self._upsert_vectors("observations", rows)

    def upsert_vec_records(self, rows: list[tuple[str, np.ndarray]]) -> int:
        """Upsert ``(stable_id, embedding)`` rows. Returns how many were written."""
        return self._upsert_vectors("records", rows)

    def _upsert_vectors(
        self, kind: str, rows: list[tuple[str, np.ndarray]]
    ) -> int:
        self._ensure_writable()
        if not self._vector_writes_enabled():
            return 0
        c = self._c()
        col, base_table = self._kind_columns(kind)
        for chunk_id, embedding in rows:
            self.record_chunk_embedding(
                kind,
                chunk_id=chunk_id,
                owner_id=_owner_of_chunk(c, base_table, col, chunk_id),
                embedding=embedding,
                content_sha256=None,
                _commit=False,
            )
        c.commit()
        return len(rows)

    def _vector_writes_enabled(self) -> bool:
        """Whether a vector write can proceed; warns once per store if not.

        A quarantined index blocks writes as firmly as a missing extension, and
        for a sharper reason: the writes would be THIS build's vectors landing
        in a table filled by a different model, producing a silently mixed
        index that no later check could untangle. Refusing keeps the two apart.
        """
        if self.vector_available and self._vector_quarantine_reason() is None:
            return True
        if not self._vector_write_warned:
            self._vector_write_warned = True
            log.warning(
                "%s — discarding vector writes for this store. "
                "embedding_state is deliberately NOT advanced, so the backlog "
                "survives and `aggregator embed --catchup` will fill it once "
                "the arm is working.",
                self._vector_quarantine
                or "vector index unavailable (sqlite-vec did not load)",
            )
        return False

    def _require_vector(self) -> None:
        if not self.vector_available:
            raise VectorIndexUnavailableError(
                "vector retrieval is unavailable: the sqlite-vec extension "
                "did not load on this connection. FTS5 keyword search still "
                "works; re-install the `sqlite-vec` wheel for this interpreter "
                "to restore the vector arm."
            )
        # The extension loaded, but the index may not be this build's. Same
        # exception type on purpose: "the arm is off" is one fact with two
        # causes, and every caller already degrades to FTS5 on it.
        quarantine = self._vector_quarantine_reason()
        if quarantine is not None:
            raise VectorIndexUnavailableError(quarantine)

    def _require_vector_readable(self) -> str:
        """Everything :meth:`_require_vector` checks, PLUS ``completed_at``.

        Returns the model a read may answer from. Split from the write guard
        deliberately and in one direction only: the worker filling a new
        model's index must be able to write to it while it is still invisible,
        or the pointer could never flip. Reads may not, because a half-built
        embedding space answers with plausible scores drawn from whatever
        fraction happens to be embedded.

        Same exception type as every other way the arm is off — one fact,
        several causes, and every caller already degrades to FTS5 on it.
        """
        self._require_vector()
        model = self.serving_embedding_model()
        if model is None:
            raise VectorIndexUnavailableError(self._not_serving_reason())
        # AN EMPTY INDEX BESIDE A FULL ONE IS NOT A COLD CACHE. Keying on the
        # model removed the need to DELETE anything on a model change, but it
        # introduced a way to degrade silently: a process configured for a
        # model this cache has never embedded would filter the KNN to its own
        # empty partition, return nothing, and look exactly like a corpus that
        # has not been backfilled yet — while a perfectly good index for
        # another model sat beside it. That is round 1's H1 in its new form,
        # and it gets round 1's answer: refuse, name both models, delete
        # nothing. A genuinely cold cache (no vectors under ANY model) is not
        # this case and still answers "nothing embedded yet".
        foreign = self.other_indexed_model(model)
        if foreign is None:
            return model
        raise VectorIndexUnavailableError(
            f"this cache holds vectors for {foreign!r} and none at all "
            f"for {model!r}, which is what this process is configured to "
            f"query. Serving the query anyway would silently return nothing "
            f"and look identical to a corpus that has not been embedded yet. "
            f"NOTHING WAS DELETED — vectors are keyed (chunk_id, model), so "
            f"the existing index is intact and still serves any process "
            f"configured for it. If AGGREGATOR_EMBED_BACKEND is exported in "
            f"this shell then that is the cause and `unset "
            f"AGGREGATOR_EMBED_BACKEND` is the whole fix. If the model change "
            f"is intended, run `aggregator embed --catchup --source both` to "
            f"build the new index; FTS5 keyword search is unaffected "
            f"meanwhile."
        )

    def other_indexed_model(self, model: str | None = None) -> str | None:
        """A model this cache HAS embedded, when ``model`` has nothing at all.

        ``None`` in the two healthy cases: this model has vectors (so there is
        nothing to compare it against), or the cache has no vectors under any
        model (a genuinely cold index, which is not a disagreement).

        Two callers, one question. The read path refuses to serve an empty
        partition while a full one sits beside it — otherwise a model change
        would degrade silently to "no results" and look like an unfinished
        backfill. The embed worker asks the same thing before deciding whether
        it is about to start a SECOND index by accident.
        """
        model = model or self.embedding_model
        c = self._c()
        if not _table_present(c, "chunk_embeddings"):
            return None
        mine = c.execute(
            "SELECT 1 FROM chunk_embeddings WHERE model = ? LIMIT 1", (model,)
        ).fetchone()
        if mine is not None:
            return None
        other = c.execute(
            "SELECT model FROM chunk_embeddings WHERE model <> ? LIMIT 1", (model,)
        ).fetchone()
        return None if other is None else str(other["model"])

    def _not_serving_reason(self) -> str:
        """Why the configured model may not be served. Two causes, two fixes.

        Both are actionable and they are OPPOSITE actions, which is why the
        message has to say which one applies rather than offering both.
        """
        state = self.embedding_version_state()
        wanted = self.embedding_model
        if state["started_at"] is None:
            return (
                f"this cache holds no vector index for {wanted!r} — nothing "
                f"has ever been embedded under that model here. NOTHING WAS "
                f"DELETED; vectors are keyed (chunk_id, model), so whatever "
                f"other model this cache holds is intact and still serves any "
                f"process configured for it. If AGGREGATOR_EMBED_BACKEND is "
                f"exported in this shell then that is the cause and `unset "
                f"AGGREGATOR_EMBED_BACKEND` is the whole fix. If the model "
                f"change is intended, run `aggregator embed --catchup --source "
                f"both` to build it; FTS5 keyword search is unaffected "
                f"meanwhile."
            )
        return (
            f"the vector index for {wanted!r} is still being built, and a "
            f"previously completed index exists on this cache, so this partial "
            f"one is not served. NOTHING IS WRONG and nothing was deleted — a "
            f"half-filled embedding space answers with plausible scores drawn "
            f"from whichever rows happened to be embedded first, which is "
            f"worse than not answering at all. FTS5 keyword search is "
            f"unaffected. Run `aggregator embed --catchup --source both` to "
            f"finish it; `aggregator status` reports how far it has got "
            f"(started {state['started_at']})."
        )

    @staticmethod
    def _embed_source_clause(kind: str, source: str) -> tuple[str, str, list]:
        """``(extra join, extra WHERE, params)`` narrowing a backlog to one group.

        The SAME predicate the progress tally groups by, expressed as a filter
        — both are derived from ``_OBS_SOURCE_CASE`` / ``_REC_SOURCE_CASE`` so
        "which rows are dropbox's" has one definition. Two definitions here
        would show up as a source reporting more rows embedded than it holds,
        which is the shape of bug that makes a progress display worse than none.

        AN UNKNOWN NAME RAISES. Silently selecting nothing would read as "that
        source is fully embedded" to every caller above — a typo in a source
        name turning into a clean bill of health for a source nobody touched.
        """
        if kind == "observations":
            # LEFT, so that an observation whose session row is missing still
            # reaches the CASE and lands in EMBED_REST. The foreign key makes
            # that row unwritable through this API, but PRAGMA foreign_keys is
            # per-connection and defaults OFF everywhere else, and SQLite never
            # re-validates existing rows — so a cache someone has repaired by
            # hand can hold one. Under an INNER JOIN it would belong to no
            # group at all: never selected, never embedded, never reported
            # missing, and blocking mark_embedding_version_complete forever.
            join = "LEFT JOIN sessions s ON s.session_id = o.session_id"
            if source == "sessions":
                return join, "s.origin = 'claude-code' AND s.kind = 'session'", []
            if source == "subagents":
                return join, "s.origin = 'claude-code' AND s.kind = 'subagent'", []
            if source in CHAT_ORIGINS:
                return join, "s.origin = ?", [source]
            if source == EMBED_REST:
                # ``s.origin IS NULL`` is the orphan. It has to be spelled out:
                # ``NOT IN`` is NULL for a NULL left-hand side, which is not
                # true, so the catch-all would drop exactly the row it exists
                # to catch. The ranked branches need no such clause — their
                # equalities are false for NULL, which is what they want.
                marks = ",".join("?" * len(_RANKED_OBS_ORIGINS))
                return (
                    join,
                    f"(s.origin IS NULL OR s.origin NOT IN ({marks}))",
                    list(_RANKED_OBS_ORIGINS),
                )
        elif kind == "records":
            if source in _RANKED_RECORD_SOURCES:
                return "", "r.source = ?", [source]
            if source == EMBED_REST:
                marks = ",".join("?" * len(_RANKED_RECORD_SOURCES))
                return "", f"r.source NOT IN ({marks})", list(_RANKED_RECORD_SOURCES)
        else:
            raise ValueError(f"unknown kind: {kind!r}")
        raise ValueError(
            f"unknown embed source {source!r} for kind {kind!r}; expected one "
            f"of {[s for k, s in EMBED_BACKLOG_ORDER if k == kind]}. Selecting "
            f"nothing for an unrecognised name would read as 'that source is "
            f"fully embedded'."
        )

    def select_unembedded(
        self,
        kind: str,
        limit: int = 500,
        model: str | None = None,
        source: str | None = None,
    ) -> list[sqlite3.Row]:
        """The embed worker's backlog: rows with no vector under ``model``.

        ``source`` NARROWS IT TO ONE GROUP OF ``EMBED_BACKLOG_ORDER``, which is
        how the user's 2026-08-21 priority is actually executed: the worker
        walks the groups in order and drains each before starting the next.
        ``None`` is the whole backlog and stays the default, because plenty of
        callers legitimately want "what is left" without caring whose it is.

        RESUMABLE MID-SOURCE FALLS OUT OF THE LEFT JOIN and needs nothing else.
        A run killed halfway through dropbox comes back, asks the same
        question, and gets the rows it had not reached — no cursor, no ledger,
        nothing to fall out of step. That is the same property the unscoped
        form already had, narrowed; adding a per-source cursor would have
        reintroduced exactly the second source of truth this design removed.

        A QUERY, NOT A LEDGER — rule 3 of the reference design, and the reason
        this is a LEFT JOIN against ``chunk_embeddings`` rather than a scan of
        an ``embedding_state`` column. The store IS the ledger, so the backlog
        is restart-safe by construction: there is no second table to fall out
        of step with the first, every write is idempotent, and running the job
        twice equals running it once.

        The column could never have carried this. ``embedding_state = 'ok'``
        cannot say WHICH MODEL embedded the row, so it is wrong the moment the
        model moves — a model change with a column-based backlog selects
        nothing and the new index stays empty forever. It is wrong in the other
        direction too: round 4 found that a source rebuild re-INSERTs rows
        without the column, returning ~483k of them to the backlog while their
        vectors are still perfectly good. Both are the same bug, which is that
        the ledger and the vectors are two separate facts that must agree.

        ``embedding_state`` KEEPS THE TWO NEGATIVE OUTCOMES, and only those.
        ``'skip'`` (nothing embeddable in this body) and ``'error'`` (set aside
        by the poison ledger) are facts no ``chunk_embeddings`` row could
        record, because in both cases there is no embedding to record. They
        hold a row out of the backlog; ``'ok'`` no longer does anything.

        Newest first: a fresh observation is the one most likely to be searched
        for, so a partially-embedded corpus is useful long before it is
        complete.

        ``src_hash`` COMES BACK WITH THE BODY, from the same statement, and
        that is the whole point of it being here. The worker has to be able to
        say "the body I embedded was the one carrying this fingerprint", and a
        fingerprint read by a LATER statement cannot say that: ingest can land
        in between, so the "before" snapshot already describes the new body and
        the row reads as unmoved while the text in hand is stale. Reproduced —
        the old body's vector stored as current with the row marked ``'ok'``,
        which ``select_unembedded`` never returns again. See
        ``commit_embed_batch``, which is where the value is spent.
        """
        if kind not in ("observations", "records"):
            raise ValueError(f"unknown kind: {kind!r}")
        c = self._c()
        model = model or self.embedding_model
        # Raises for an unknown source BEFORE any SQL is built, so a typo
        # cannot come back as an empty backlog.
        extra_join, extra_where, extra_params = (
            self._embed_source_clause(kind, source) if source is not None else ("", "", [])
        )
        scope = f"  AND ({extra_where}) " if extra_where else ""
        if kind == "observations":
            return list(
                c.execute(
                    "SELECT o.obs_id AS obs_id, o.body AS body, "  # noqa: S608 - allowlisted literals
                    "       o.src_hash AS src_hash "
                    "FROM observations o "
                    f"{extra_join} "
                    "LEFT JOIN chunk_embeddings e "
                    "  ON e.owner_id = o.obs_id AND e.kind = 'observations' "
                    "  AND e.model = ? "
                    "WHERE e.chunk_id IS NULL "
                    "  AND o.embedding_state IS NOT 'skip' "
                    "  AND o.embedding_state IS NOT 'error' "
                    f"{scope}"
                    "ORDER BY o.ts DESC LIMIT ?",
                    (model, *extra_params, limit),
                )
            )
        return list(
            c.execute(
                "SELECT r.stable_id AS stable_id, r.subject AS subject, "  # noqa: S608 - allowlisted literals
                "       r.body AS body, r.src_hash AS src_hash "
                "FROM records r "
                f"{extra_join} "
                "LEFT JOIN chunk_embeddings e "
                "  ON e.owner_id = r.stable_id AND e.kind = 'records' "
                "  AND e.model = ? "
                "WHERE e.chunk_id IS NULL "
                "  AND r.embedding_state IS NOT 'skip' "
                "  AND r.embedding_state IS NOT 'error' "
                f"{scope}"
                "ORDER BY r.updated_at DESC LIMIT ?",
                (model, *extra_params, limit),
            )
        )

    def mark_embedded(
        self,
        kind: str,
        ids: list[str],
        state: str,
        expected: dict[str, str | None] | None = None,
        *,
        _commit: bool = True,
    ) -> list[str]:
        """Advance ``embedding_state`` for ``ids``. Returns the ids WRITTEN.

        ``expected`` maps id → the ``src_hash`` the caller read alongside the
        body, and turns this into a compare-and-set: ``WHERE id = ? AND
        src_hash IS ?``. A row whose body moved since then does not match, so
        it is not marked and stays in the backlog for the next pass — which is
        what makes the return value load-bearing rather than decorative. ``IS``
        rather than ``=`` on purpose: a legacy row with a NULL ``src_hash``
        compares equal to an expected NULL and marks normally, where ``=``
        would silently never match it and strand it in the backlog forever.

        ``expected=None`` is the unguarded form, for callers making no claim
        about content — see ``'error'`` below — and returns ``ids`` unchanged.
        It is an explicit choice and stays available. What is NOT available is
        making it BY ACCIDENT: an id in ``ids`` and absent from a supplied
        ``expected`` raises ``KeyError``.

        WHY THAT IS AN ERROR AND NOT A DEFAULT. The lookup used to be
        ``expected.get(row_id)``, so a missing key became ``None`` and the
        guard became ``src_hash IS NULL`` — a condition the caller never
        expressed, and one that is wrong in BOTH directions at once. Against an
        ordinary row it never matches, so the row is skipped and
        ``cli._embed_batch`` books the miss as a benign ``superseded``: a
        silent no-op reported as a self-healing edit. Against a legacy row it
        MATCHES — every pre-v4 row has a NULL ``src_hash`` and
        ``_ensure_src_hash_columns`` deliberately never backfills them — so the
        write lands, guarded against nothing, on exactly the rows least able to
        prove they are unchanged. Reproduced: ``mark_embedded(ids=['hashed',
        'legacy'], expected={})`` returned ``['legacy']`` and marked it.

        Checked for the WHOLE id list before the first UPDATE. A caller whose
        map is missing an id has a bug in how it built the map, and writing the
        ids that happened to be present would leave the store in a state
        neither that caller nor the ledger describes.

        ``state`` is one of ``'ok'`` / ``'skip'`` / ``'error'``; anything else
        raises rather than writing a value nothing selects on. An EMPTY ``ids``
        list is a no-op and not an error — the worker reaches it on any batch
        that was all-successes or all-skips, and ``IN ()`` is a SQL syntax
        error, not an empty set.

        ``'ok'`` REFUSES WHEN THE VECTOR ARM IS UNAVAILABLE, because it is the
        only one of the three that asserts a vector exists. ``upsert_vec_*``
        no-ops without the extension, so the obvious composition — write the
        vectors, then advance the watermark — otherwise marks rows embedded
        that have no vector and that ``select_unembedded`` will never return
        again. Until now the only thing preventing that was a guard in
        ``cli._cmd_embed``: one caller's discipline, protecting nothing from
        the second caller. ``'skip'`` and ``'error'`` are both true with no
        vector, so they stay writable and the backlog still drains.
        """
        self._ensure_writable()
        if state not in ("ok", "skip", "error"):
            raise ValueError(f"invalid state: {state!r}")
        col, table = self._kind_columns(kind)
        if not ids:
            return []
        if state == "ok":
            self._require_vector()
        c = self._c()
        if expected is None:
            for page in _chunked(ids, 500):
                placeholders = ",".join("?" * len(page))
                c.execute(
                    f"UPDATE {table} SET embedding_state = ? "  # noqa: S608 - allowlisted literals
                    f"WHERE {col} IN ({placeholders})",
                    (state, *page),
                )
            written = list(ids)
        else:
            # BEFORE THE FIRST UPDATE, so a caller with an incomplete map gets
            # a refusal rather than a partly-applied batch. See the docstring:
            # the old ``.get`` turned a missing key into ``src_hash IS NULL``,
            # which silently skips ordinary rows and silently MARKS legacy ones.
            _require_expectations(ids, expected)
            # One primary-key UPDATE per id rather than a clever set-based
            # statement: a batch is 500 rows, each of these is microseconds,
            # and ``rowcount`` per statement is how the caller learns WHICH
            # rows it actually claimed. ``_drop_row_vectors`` loops for the
            # same reason.
            written = []
            for row_id in ids:
                cur = c.execute(
                    f"UPDATE {table} SET embedding_state = ? "  # noqa: S608 - allowlisted literals
                    f"WHERE {col} = ? AND src_hash IS ?",
                    (state, row_id, expected[row_id]),
                )
                if cur.rowcount:
                    written.append(row_id)
        if _commit:
            c.commit()
        return written

    def commit_embed_batch(
        self,
        kind: str,
        *,
        vectors: list[tuple[str, str, np.ndarray]],
        ok_ids: list[str],
        skip_ids: list[str],
        error_ids: list[str],
        expected: dict[str, str | None],
        hashes: dict[str, str] | None = None,
    ) -> tuple[list[str], list[str]]:
        """The embed worker's commit point. Returns ``(ok written, skip written)``.

        ONE TRANSACTION, ONE GUARD, and both halves are the point.

        THE GUARD closes the last window in the read-embed-write cycle. Ingest
        sets ``embedding_state`` back to NULL and drops a row's vectors when
        its body changes — that is the only thing that ever re-embeds an edited
        row — and the worker used to undo it: it embedded the OLD body and then
        wrote that vector back and marked the row ``'ok'``, unconditionally.
        Re-reading the fingerprints just before the writes shrank the window
        but could not close it, because the "before" value came from a
        different statement than the body: an edit landing between the SELECT
        and that re-read makes both reads see the NEW hash, so the row looks
        untouched while the text in hand is stale. Reproduced exactly that way.
        Here the "before" value is ``select_unembedded``'s own ``src_hash``, and
        every write carries ``AND src_hash IS ?`` in the same statement that
        does the writing, so there is no interval left to land in.

        ONE TRANSACTION removes the second interval, between the vector commit
        and the watermark commit. Nothing widens the lock: both writes already
        happened together at the end of a batch. The direction of a crash is
        unchanged and still the safe one — a death anywhere here rolls the
        whole batch back to NULL and the next run redoes it, which costs one
        batch. The mirrored failure, rows marked embedded with no vector, is
        the one nothing ever comes back for, and it remains impossible.

        ``vectors`` is ``(owner row id, chunk id, embedding)``: a body over the
        window size becomes ``<id>:0 .. <id>:N-1``, so the guard has to be
        expressed against the OWNER while the write targets the chunk.

        ``error_ids`` are marked UNGUARDED, deliberately. ``'error'`` is not a
        claim about a row's content — it says the worker could not embed it —
        and the quarantine ledger, not this column, decides when it comes back.
        Guarding it would leave a failed row at NULL, which
        ``select_unembedded`` re-selects on the very next batch of the same
        run: the abort loop this whole path exists to prevent.
        """
        self._ensure_writable()
        vec_table = self._VEC_TABLES.get(kind)
        if vec_table is None:
            raise ValueError(f"unknown kind: {kind!r}")
        col, table = self._kind_columns(kind)
        vec_key = "obs_id" if kind == "observations" else "stable_id"
        if ok_ids:
            # Round 1's M4, kept: 'ok' asserts a vector exists, so it may not
            # be written on a store that cannot hold one.
            self._require_vector()
        c = self._c()
        # Every OWNER whose vector is about to be written must have a stated
        # expectation, checked before the first DELETE. The guarded INSERT
        # below reads the same map, and a missing entry there would write the
        # vector under ``src_hash IS NULL`` — landing it on a legacy row and
        # losing it on every other. Same failure as ``mark_embedded``'s, one
        # table over.
        _require_expectations([row_id for row_id, _, _ in vectors], expected)
        writes_enabled = self._vector_writes_enabled()
        model = self.embedding_model
        hashes = hashes or {}
        for row_id, chunk_id, embedding in vectors:
            # Unconditional, and scoped to THIS model: if the row moved, ingest
            # already dropped every model's vectors for it and this removes
            # nothing. If it did not, this is the delete half of the idempotent
            # delete-then-insert. Another model's vector for the same chunk is
            # not touched — that coexistence is the point of the key.
            c.execute(
                f"DELETE FROM {vec_table} WHERE model = ? AND {vec_key} = ?",  # noqa: S608 - allowlisted literals
                (model, chunk_id),
            )
            if not writes_enabled:
                continue
            cur = c.execute(
                f"INSERT INTO {vec_table}(model, {vec_key}, embedding) "  # noqa: S608 - allowlisted literals
                f"SELECT ?, ?, ? WHERE EXISTS ("
                f"SELECT 1 FROM {table} WHERE {col} = ? AND src_hash IS ?)",
                (
                    model,
                    chunk_id,
                    embedding.astype("float32").tobytes(),
                    row_id,
                    expected[row_id],
                ),
            )
            # THE INDEX ROW RIDES THE SAME GUARD. It is written only when the
            # vector was, so ``chunk_embeddings`` can never claim an embedding
            # that a compare-and-swap rejected — which would take the row out
            # of the backfill query permanently, for a vector that does not
            # exist.
            if cur.rowcount:
                self._record_chunk_row(
                    c, kind, chunk_id, row_id, model, hashes.get(chunk_id)
                )
        written_ok = self.mark_embedded(kind, ok_ids, "ok", expected, _commit=False)
        written_skip = self.mark_embedded(
            kind, skip_ids, "skip", expected, _commit=False
        )
        # UNGUARDED, and said so explicitly rather than by omission — see the
        # note above on ``error_ids``.
        self.mark_embedded(kind, error_ids, "error", None, _commit=False)
        c.commit()
        return written_ok, written_skip

    def requeue_embedding(self, kind: str, ids: list[str]) -> None:
        """Put rows back INTO the backlog: ``embedding_state`` → NULL.

        The other half of ``mark_embedded(state='error')``, and the reason that
        state exists at all. ``select_unembedded`` is ``WHERE embedding_state
        IS NULL ORDER BY ts DESC``, so a row left NULL after it failed is
        re-selected by the very next batch of the same ``--catchup`` — the old
        abort loop converted into an in-process spin. ``'error'`` therefore
        means "out of the backlog RIGHT NOW", and the quarantine ledger, not
        this column, owns the question of when it comes back. This is how it
        comes back.

        Unguarded on ``vector_available`` like its two neighbours: a plain
        column has no business depending on a native extension. An empty
        ``ids`` is a no-op, not an error — the caller reaches it on every
        healthy run, and ``IN ()`` is a syntax error rather than an empty set.
        """
        self._ensure_writable()
        col, table = self._kind_columns(kind)
        if not ids:
            return
        c = self._c()
        for page in _chunked(ids, 500):
            placeholders = ",".join("?" * len(page))
            c.execute(
                f"UPDATE {table} SET embedding_state = NULL "  # noqa: S608 - allowlisted literals
                f"WHERE {col} IN ({placeholders})",
                tuple(page),
            )
        c.commit()

    # -- the in-flight claim: what a row that KILLED the worker leaves behind -
    #
    # Every other failure path in the embed worker starts with an exception.
    # This one cannot: an OOM kill, a segfault in a native tokenizer or torch
    # kernel, or a SIGKILL ends the process with no unwinding, so nothing is
    # caught, nothing is held in the ledger and ``embedding_state`` never
    # moves. ``select_unembedded`` is ``ORDER BY ts DESC``, so the next tick
    # picks the same row first and dies identically — twice an hour, forever,
    # with an empty ledger and no stderr line to say so.
    #
    # So the row is written down BEFORE it is attempted, and committed, which
    # is the only part a kill cannot undo. A run that finds a claim knows the
    # previous process did not survive that row.

    @property
    def embed_claim_path(self) -> Path:
        """Where the in-flight claim lives. NEXT TO THE DATABASE, NOT IN IT.

        THE CLAIM IS A CRASH DETECTOR, AND IT USED TO BE STORED INSIDE THE
        RESOURCE WHOSE UNAVAILABILITY IS THE COMMONEST NON-CRASH FAILURE. It
        was a ``meta`` row, so disowning a row meant writing to sqlite — and
        the 30-minute ingest timer takes that write lock. Observed in
        production on 2026-08-27, four times in three hours: the worker
        embedded a row, ``release_embed_claim`` raised
        ``database is locked``, the run exited 1, and the NEXT run read the
        surviving claim as a kill and booked a healthy row into the poison
        ledger. Three of those make a row terminal — silently absent from the
        vector arm forever.

        A retry loop around the DELETE narrows that window and cannot close
        it: the failure mode IS that the database stays unavailable longer
        than the process is willing to wait. Any design where "I exited
        cleanly" has to be written to a contended resource fails exactly when
        it is needed.

        A FILE SEPARATES THE TWO CASES THE DATABASE COULD NOT. Removing it
        needs no lock, so a clean exit can always disown its row, however
        thoroughly sqlite is wedged; a SIGKILL still runs no code, so a real
        crash still leaves it. Same evidence for the case it was built for,
        none of the evidence it was fabricating.

        Beside ``<db>.embed.lock``, which the worker already holds for its
        lifetime — same directory, same owner, same cleanup story, and the
        adjacency is a hint to anyone who finds one of them by hand.
        """
        return Path(str(self.db_path) + ".embed.claim")

    def claim_embed_row(self, kind: str, row_id: str) -> None:
        """Record — durably — which row is about to be embedded.

        DURABLY IS THE ENTIRE POINT: this has to outlive a process that gets
        no chance to unwind, so the bytes are on the platter before the
        encode starts. Written to a temporary name and renamed, so a kill
        during the write cannot leave a half-written claim naming a partial
        row id — ``rename`` is atomic within a directory, and a claim that
        names the wrong row is worse than no claim at all.
        """
        payload = json.dumps({"kind": kind, "row_id": row_id}, sort_keys=True)
        path = self.embed_claim_path
        tmp = path.with_suffix(path.suffix + ".new")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

    def release_embed_claim(self) -> None:
        """The claimed row resolved. Clear the claim.

        NEVER RAISES, and that is load-bearing rather than defensive. This is
        called on the success path of every row and again from the handlers
        that abort a run; if it could throw it would become one more way to
        exit while leaving a claim behind, which is the bug it exists to end.
        Nothing is lost by swallowing: a claim that could not be removed is
        re-examined next run, and the worst case is the false blame that this
        whole mechanism is being reshaped to avoid.
        """
        try:
            self.embed_claim_path.unlink(missing_ok=True)
        except OSError as e:  # pragma: no cover - unreadable claim directory
            log.warning(
                "could not remove the embed claim at %s (%s: %s); the next "
                "run may read it as a crash.",
                self.embed_claim_path,
                type(e).__name__,
                e,
            )

    def pending_embed_claim(self) -> tuple[str, str] | None:
        """``(kind, row_id)`` a previous process died on, or ``None``."""
        # A LEGACY CLAIM IN ``meta`` CONVICTS NOBODY, and is cleared on sight.
        # It was written by a build that could not tell a kill from a
        # lock-on-release, so its presence is consistent with both and carries
        # no information about which happened. Reading it as a crash is what
        # condemned four good rows on 2026-08-27. It is deleted rather than
        # ignored so it cannot be re-examined by every future run — and the
        # delete is best-effort, because the same lock that created this mess
        # may still be held.
        try:
            legacy = self._c().execute(
                "SELECT value FROM meta WHERE key = ?", (EMBED_CLAIM_KEY,)
            ).fetchone()
            if legacy is not None:
                log.warning(
                    "discarding an in-database embed claim left by an older "
                    "build (%s). It cannot be distinguished from a lock on "
                    "the release path, so no row is blamed for it.",
                    legacy[0],
                )
                c = self._c()
                c.execute("DELETE FROM meta WHERE key = ?", (EMBED_CLAIM_KEY,))
                c.commit()
        except sqlite3.Error:
            pass

        try:
            raw = self.embed_claim_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError:
            return None
        try:
            claim = json.loads(raw)
            return str(claim["kind"]), str(claim["row_id"])
        except (ValueError, KeyError, TypeError):
            # An unreadable claim names nobody, so it can blame nobody. Say
            # nothing rather than condemning an arbitrary row.
            return None

    @staticmethod
    def _kind_columns(kind: str) -> tuple[str, str]:
        """``(id column, table)`` for an ontology name, or raise.

        One mapping for the three ``embedding_state`` writers, so a new
        ontology cannot be half-added — the allowlist that makes their f-string
        SQL safe is the same allowlist in every one of them.
        """
        if kind == "observations":
            return "obs_id", "observations"
        if kind == "records":
            return "stable_id", "records"
        raise ValueError(f"unknown kind: {kind!r}")

    _VEC_TABLES = {
        "observations": "vec_observations",
        "records": "vec_records",
    }

    def count_vec_rows(self, kind: str) -> int:
        """How many VECTORS the vector arm holds for ``kind``.

        Counts rows in the ``vec_<kind>`` virtual table, which is chunks and
        not documents: the embed worker writes one row per chunk, keyed
        ``<id>:<n>`` when a body needed more than one. Document-level progress
        is ``embedding_state``; this is "how much can KNN actually reach".

        A read of the vector arm, so it RAISES rather than answering 0 when
        the arm is off — the whole point of the number is telling "nothing
        embedded yet" apart from "no vector index on this cache", and a 0
        that means either is worth less than no number at all. Both the
        extension-missing case and the tables-missing case (a cache migrated
        on a machine without sqlite-vec, opened later on one with it) come
        back as :class:`VectorIndexUnavailableError`.
        """
        table = self._VEC_TABLES.get(kind)
        if table is None:
            raise ValueError(f"unknown kind: {kind!r}")
        self._require_vector()
        c = self._c()
        try:
            row = c.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()  # noqa: S608 - allowlisted literals
        except sqlite3.OperationalError as e:
            raise VectorIndexUnavailableError(
                f"vector table {table!r} is missing from this cache: {e}. The "
                "cache was migrated on an interpreter without the sqlite-vec "
                "extension. Run `aggregator embed --catchup` (or any writable "
                "aggregator command) on an interpreter that has it to create "
                "the vector tables and fill them."
            ) from e
        return int(row["n"]) if row else 0

    def has_embedded_rows(self, kind: str) -> bool:
        """Is there anything for the vector arm to find? The ROUTING probe.

        Deliberately not ``count_vec_rows(kind) > 0``, and the reason is
        measured rather than assumed: ``COUNT(*)`` on a ``vec0`` virtual table
        is O(n) — about 4 ms at 20k vectors, 13 ms at 100k and 70 ms at 400k,
        which is the size of the live cache. Retrieval asks this question on
        every free-text query and on up to two ontologies, so a linear probe
        would put a tenth of a second of pure overhead on the recall path and
        make it worse every time the corpus grows.

        ``chunk_embeddings`` is a plain table with an index on
        ``(model, kind, …)``, so the same question costs microseconds — and it
        is the more honest predicate for a second reason round 4 made concrete.
        The old form read ``embedding_state = 'ok'``, a column a source rebuild
        silently resets: after one, ~483k rows returned to NULL and this method
        reported the arm OFF over a fully populated index. The embedding index
        is not touched by a row rewrite, so it cannot lie in that direction.

        Raises when the extension is missing, like every other vector read —
        routing has to be able to tell "nothing embedded yet" from "this
        machine cannot run the vector arm at all".
        """
        if kind not in self._VEC_TABLES:
            raise ValueError(f"unknown kind: {kind!r}")
        model = self._require_vector_readable()
        c = self._c()
        if not _table_present(c, "chunk_embeddings"):
            return False
        row = c.execute(
            "SELECT 1 FROM chunk_embeddings WHERE model = ? AND kind = ? LIMIT 1",
            (model, kind),
        ).fetchone()
        return row is not None

    def embed_progress_by_source(self, model: str | None = None) -> list[dict]:
        """How far the backfill has got, PER SOURCE, in priority order.

        "Which sources are fully embedded" is the question a user actually asks
        of a 25-30 day backfill, and a global percentage cannot answer it. A
        single "62% embedded" is compatible with dropbox being untouched and
        with dropbox being finished, and those are opposite answers to "can I
        search my notes yet".

        One row per entry in :data:`EMBED_BACKLOG_ORDER`, in that order, always
        all of them — a source with nothing in it still gets a row, because
        omitting it is how "no rows here" becomes indistinguishable from "not
        reached yet". Each carries ``kind``, ``source``, ``total``,
        ``embedded``, ``skipped``, ``errors``, ``pending`` and a ``state``.

        ``state`` USES THE SAME VOCABULARY AS :meth:`vector_index_state`, for
        the same reasons, one source at a time:

        * ``empty``       — the source holds no rows at all. NEVER ``complete``:
          a source with nothing in it and a source fully embedded return the
          same zero vector hits for every query, and telling them apart is the
          whole point of this method.
        * ``not_started`` — rows are waiting and none has a vector yet.
        * ``in_progress`` — some embedded, some still pending.
        * ``degraded``    — nothing pending and yet rows are missing, because
          the worker set them aside as ``'error'``. Waiting cannot fix it,
          which is exactly why it may not be reported as ``complete``.
        * ``complete``    — every row either embedded or legitimately skipped.

        FOUR QUERIES, NOT FOUR PER SOURCE. Grouped in SQL rather than looped in
        Python. Measured end to end against a read-only snapshot of the live
        cache (505k observations, 4.3k records, 1.3 GB): **0.27 s warm, 4.3 s
        on a cold page cache** — the cold figure is the first touch of the file
        in a process and is what an operator pays for the first
        ``aggregator status`` after a boot. Thirty-two separate counts would
        have made this something an operator learns not to run.

        DELIBERATELY NOT PART OF ``capabilities()``. That is the MCP connect
        path, and a full table scan belongs on a command someone typed, not on
        every client handshake — least of all the cold one.

        ``embedded`` COMES OFF ``chunk_embeddings`` AND IS KEYED ON THE MODEL,
        never off ``embedding_state``: the column cannot say which model
        embedded a row, so after a model change it would report a complete
        index for an embedding space holding nothing. Same reasoning as the
        backlog query, and it must be the same answer or the two disagree
        about the same source.
        """
        model = model or self._stamped_model() or self.embedding_model
        c = self._c()
        totals: dict[tuple[str, str], dict[str, int]] = {
            (kind, source): {
                "total": 0,
                "embedded": 0,
                "skipped": 0,
                "errors": 0,
            }
            for kind, source in EMBED_BACKLOG_ORDER
        }

        def bucket(kind: str, source: str) -> dict[str, int]:
            # An unrecognised source cannot happen — both CASE expressions end
            # in ELSE EMBED_REST — but answering into a discarded dict would be
            # the one way this method could under-report, so it is spelled out.
            return totals.setdefault(
                (kind, source),
                {"total": 0, "embedded": 0, "skipped": 0, "errors": 0},
            )

        for kind, alias, sql_case, tally_from, embedded_from in (
            (
                "observations",
                "o",
                _OBS_SOURCE_CASE,
                # LEFT for the same reason the backlog's join is LEFT: a tally
                # that drops the orphan reports a source 100% embedded while
                # the worker still has the row. See _embed_source_clause.
                "observations o LEFT JOIN sessions s ON s.session_id = o.session_id",
                "chunk_embeddings e "
                "JOIN observations o ON o.obs_id = e.owner_id "
                "LEFT JOIN sessions s ON s.session_id = o.session_id",
            ),
            (
                "records",
                "r",
                _REC_SOURCE_CASE,
                "records r",
                "chunk_embeddings e JOIN records r ON r.stable_id = e.owner_id",
            ),
        ):
            for row in c.execute(
                f"SELECT {sql_case} AS src, {alias}.embedding_state AS st, "  # noqa: S608 - allowlisted literals
                f"COUNT(*) AS n FROM {tally_from} GROUP BY src, st"
            ):
                entry = bucket(kind, str(row["src"]))
                n = int(row["n"])
                entry["total"] += n
                if row["st"] == "skip":
                    entry["skipped"] += n
                elif row["st"] == "error":
                    entry["errors"] += n
            if not _table_present(c, "chunk_embeddings"):
                continue
            for row in c.execute(
                f"SELECT {sql_case} AS src, COUNT(DISTINCT e.owner_id) AS n "  # noqa: S608 - allowlisted literals
                f"FROM {embedded_from} "
                "WHERE e.model = ? AND e.kind = ? GROUP BY src",
                (model, kind),
            ):
                bucket(kind, str(row["src"]))["embedded"] += int(row["n"])

        out: list[dict] = []
        for kind, source in EMBED_BACKLOG_ORDER:
            entry = totals[(kind, source)]
            pending = max(
                0,
                entry["total"]
                - entry["embedded"]
                - entry["skipped"]
                - entry["errors"],
            )
            if entry["total"] == 0:
                state = "empty"
            elif pending == 0 and entry["errors"] == 0:
                state = "complete"
            elif pending == 0:
                state = "degraded"
            elif entry["embedded"] == 0:
                state = "not_started"
            else:
                state = "in_progress"
            out.append(
                {
                    "kind": kind,
                    "source": source,
                    "state": state,
                    "pending": pending,
                    **entry,
                }
            )
        return out

    def count_embedding_states(self, kind: str) -> dict[str, int]:
        """Tally of ``embedding_state`` for ``kind``: the backfill watermark.

        Keys, always all present: ``total``, plus ``pending`` (NULL — still in
        the backlog), ``ok``, ``skip`` (nothing embeddable in the body) and
        ``error``. A caller reading ``["pending"]`` must never have to guard
        for a missing key.

        Plain column arithmetic with NO dependency on the native extension,
        deliberately — the moment an operator most needs to know how far the
        backfill got is the moment the vector arm is broken.
        """
        if kind not in self._VEC_TABLES:
            raise ValueError(f"unknown kind: {kind!r}")
        c = self._c()
        counts = {"total": 0, "pending": 0, "ok": 0, "skip": 0, "error": 0}
        rows = c.execute(
            f"SELECT embedding_state AS s, COUNT(*) AS n FROM {kind} "  # noqa: S608 - allowlisted literals
            "GROUP BY embedding_state"
        ).fetchall()
        for row in rows:
            n = int(row["n"])
            counts["total"] += n
            if row["s"] in ("skip", "error"):
                counts[str(row["s"])] += n
        # ``ok`` COMES OFF THE EMBEDDING INDEX, not the column, for the same
        # reason the backlog query does: the column cannot say which model, so
        # after a model change it would report a complete index for a space
        # that holds nothing. Counting owners rather than chunks keeps this a
        # DOCUMENT-level tally, comparable with ``total``.
        # ``_stamped_model``, not ``embedding_model``: this is reached from
        # ``capabilities()`` at MCP connect time, and the property would drag
        # ``aggregator.core.embed`` (and numpy) onto the cold-start path the
        # ``test_mcp_cold_start`` guard exists to keep clear. The stamp is also
        # the right question here — "how far has the index on disk got".
        model = self._stamped_model()
        if model is not None and _table_present(c, "chunk_embeddings"):
            counts["ok"] = int(
                c.execute(
                    "SELECT COUNT(DISTINCT owner_id) AS n FROM chunk_embeddings "
                    "WHERE model = ? AND kind = ?",
                    (model, kind),
                ).fetchone()["n"]
            )
        counts["pending"] = max(
            0, counts["total"] - counts["ok"] - counts["skip"] - counts["error"]
        )
        return counts

    def _vec_obs_scored(
        self, query_embedding: np.ndarray, k: int
    ) -> list[tuple[str, float]]:
        """Vector KNN over ``vec_observations``.

        Returns top-K ``(obs_id, distance)`` ordered by ascending distance —
        best match first, which is the order RRF expects.

        THE DISTANCE COMES BACK, AND THAT IS THE WHOLE REASON FOR THE NAME.
        This method used to be ``_vec_obs_ids`` and selected the id column
        alone, so sqlite-vec computed the distance, ``ORDER BY`` used it, and
        then it was thrown away one layer below the only caller that needs it —
        which left ``hybrid.vector_floor``, criterion D's per-arm abstention
        rule, fully implemented, fully tested and never executed on any
        production path. RENAMED rather than re-typed: a stale caller now dies
        with ``AttributeError`` instead of silently treating ``(id, distance)``
        tuples as ids, which is hashable, fuses without complaint, and matches
        nothing.

        ``AND model = ?`` IS LOAD-BEARING, not a filter for tidiness. Two
        models coexist in this table by design, and their vectors are not
        comparable: a distance computed between a query in one embedding space
        and a document in another is an apples-to-rulers number that looks
        exactly like a good score. The partition key means this costs less than
        the unfiltered scan did, not more.
        """
        model = self._require_vector_readable()
        c = self._c()
        rows = c.execute(
            """
            SELECT obs_id, distance
            FROM vec_observations
            WHERE embedding MATCH ?
              AND model = ?
              AND k = ?
            ORDER BY distance
            """,
            (query_embedding.astype("float32").tobytes(), model, k),
        ).fetchall()
        return [(r["obs_id"], float(r["distance"])) for r in rows]

    def _vec_record_scored(
        self, query_embedding: np.ndarray, k: int
    ) -> list[tuple[str, float]]:
        """Vector KNN over ``vec_records``. Same contract as
        ``_vec_obs_scored``, including the returned distance and the
        load-bearing ``AND model = ?``."""
        model = self._require_vector_readable()
        c = self._c()
        rows = c.execute(
            """
            SELECT stable_id, distance
            FROM vec_records
            WHERE embedding MATCH ?
              AND model = ?
              AND k = ?
            ORDER BY distance
            """,
            (query_embedding.astype("float32").tobytes(), model, k),
        ).fetchall()
        return [(r["stable_id"], float(r["distance"])) for r in rows]

    # -- ingest state: the per-source high-water mark ----------------------

    def read_ingest_state(self, source: str) -> dict[str, object]:
        """One source's row from ``ingest_state``, or ``{}`` if it has none.

        ``{}`` is the first-run answer and means "no window" — a full scan,
        which is what a first run must be. Never raises for a missing row; a
        watermark that could not be read has to degrade to re-reading, never to
        skipping.
        """
        row = self._c().execute(
            "SELECT source, cursor_value, cursor_kind, last_run_at, last_ok_at, "
            "rows_seen, consecutive_failures, last_error, next_attempt_at "
            "FROM ingest_state WHERE source = ?",
            (source,),
        ).fetchone()
        return dict(row) if row is not None else {}

    def all_ingest_state(self) -> dict[str, dict[str, object]]:
        """Every source's ingest state, for ``aggregator status``."""
        rows = self._c().execute(
            "SELECT source, cursor_value, cursor_kind, last_run_at, last_ok_at, "
            "rows_seen, consecutive_failures, last_error, next_attempt_at "
            "FROM ingest_state ORDER BY source"
        )
        return {row["source"]: dict(row) for row in rows}

    def advance_ingest_cursor(
        self,
        source: str,
        *,
        cursor_kind: str,
        cursor_value: str | None,
        rows: int,
        at: str,
        _commit: bool = True,
    ) -> None:
        """Move one source's mark forward. FORWARD ONLY, and never to NULL.

        THE MONOTONICITY GUARD IS IN THE SQL, not in the caller, so no code
        path can route around it: ``cursor_value`` is only taken when it is
        strictly greater than what is stored. A retry of an older chunk, a
        clock that stepped backwards, or two runs racing would otherwise rewind
        the window — and everything between the two values then gets re-read
        (harmless) or, if the guard were ever missing on the way up, skipped
        forever (not).

        ISO-8601 strings compare lexicographically in the same order as the
        instants they name, as long as they share an offset and a precision.
        Every value written here comes from ``datetime.isoformat()`` on an
        aware UTC datetime produced by this codebase, so that holds — and the
        Python-side ``Watermarks`` never hands over a value it did not parse.

        ``COALESCE(?, cursor_value)`` is the empty-run rule: a pass that found
        nothing still stamps ``last_run_at`` and clears the failure counter,
        but must not write NULL over a live mark. The same defect is on record
        in another implementation of this pattern (an empty sync nuking state
        to ``{}``); here it would full-scan 372k rows on the very next tick.

        ``_commit=False`` leaves the transaction open, which is the whole point
        of the mark living in this database: the caller commits it together
        with the chunk it describes, so no crash can leave one without the
        other.
        """
        self._ensure_writable()
        c = self._c()
        try:
            c.execute(
                """
                INSERT INTO ingest_state(
                    source, cursor_value, cursor_kind, last_run_at, last_ok_at,
                    rows_seen, consecutive_failures, last_error, next_attempt_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 0, NULL, NULL)
                ON CONFLICT(source) DO UPDATE SET
                    cursor_value = CASE
                        WHEN excluded.cursor_value IS NOT NULL
                         AND (ingest_state.cursor_value IS NULL
                              OR excluded.cursor_value > ingest_state.cursor_value)
                        THEN excluded.cursor_value
                        ELSE ingest_state.cursor_value
                    END,
                    cursor_kind          = excluded.cursor_kind,
                    last_run_at          = excluded.last_run_at,
                    last_ok_at           = excluded.last_ok_at,
                    rows_seen            = ingest_state.rows_seen + excluded.rows_seen,
                    consecutive_failures = 0,
                    last_error           = NULL,
                    next_attempt_at      = NULL
                """,
                (source, cursor_value, cursor_kind, at, at, rows),
            )
        except BaseException:
            if _commit:
                c.rollback()
            raise
        if _commit:
            c.commit()

    def record_ingest_failure(
        self,
        source: str,
        *,
        cursor_kind: str,
        error: str,
        at: str,
        next_attempt_at: str | None,
        _commit: bool = True,
    ) -> None:
        """Count a failed pass. DELIBERATELY LEAVES THE MARK WHERE IT WAS.

        A failed run must not advance anything: the next run re-reads the same
        window — cheap, because the apply is idempotent — and nothing between
        the old mark and wherever this run got to can be lost. Advancing on
        failure is the one ordering that loses records silently.

        ``next_attempt_at`` is computed by the caller (which owns the backoff
        policy) and STORED, because the delay is jittered: recomputing it on
        every read would let two reads inside one run disagree about whether
        the source runs at all.
        """
        self._ensure_writable()
        c = self._c()
        try:
            c.execute(
                """
                INSERT INTO ingest_state(
                    source, cursor_value, cursor_kind, last_run_at, last_ok_at,
                    rows_seen, consecutive_failures, last_error, next_attempt_at
                )
                VALUES (?, NULL, ?, ?, NULL, 0, 1, ?, ?)
                ON CONFLICT(source) DO UPDATE SET
                    cursor_kind          = excluded.cursor_kind,
                    last_run_at          = excluded.last_run_at,
                    consecutive_failures = ingest_state.consecutive_failures + 1,
                    last_error           = excluded.last_error,
                    next_attempt_at      = excluded.next_attempt_at
                """,
                (source, cursor_kind, at, error, next_attempt_at),
            )
        except BaseException:
            if _commit:
                c.rollback()
            raise
        if _commit:
            c.commit()

    # -- ingest state: records that could not be written -------------------

    def read_poison(self, source: str) -> list[dict[str, object]]:
        """Every held record for one source, due or not. Policy is the caller's."""
        rows = self._c().execute(
            "SELECT source, record_key, error_type, error_detail, attempts, "
            "first_seen_at, last_seen_at, next_retry_at "
            "FROM quarantine WHERE source = ?",
            (source,),
        )
        return [dict(row) for row in rows]

    def hold_poison(
        self,
        source: str,
        record_key: str,
        *,
        error_type: str,
        error_detail: str,
        at: str,
        next_retry_at: str | None,
        _commit: bool = True,
    ) -> None:
        """Record one record that could not be written, or bump its attempt count.

        ``first_seen_at`` is preserved across attempts — how long something has
        been broken is the number that decides whether a human cares.
        """
        self._ensure_writable()
        c = self._c()
        try:
            c.execute(
                """
                INSERT INTO quarantine(
                    source, record_key, error_type, error_detail, attempts,
                    first_seen_at, last_seen_at, next_retry_at
                )
                VALUES (?, ?, ?, ?, 1, ?, ?, ?)
                ON CONFLICT(source, record_key) DO UPDATE SET
                    error_type    = excluded.error_type,
                    error_detail  = excluded.error_detail,
                    attempts      = quarantine.attempts + 1,
                    last_seen_at  = excluded.last_seen_at,
                    next_retry_at = excluded.next_retry_at
                """,
                (source, record_key, error_type, error_detail, at, at, next_retry_at),
            )
        except BaseException:
            if _commit:
                c.rollback()
            raise
        if _commit:
            c.commit()

    def release_poison(self, source: str, record_key: str, *, _commit: bool = True) -> None:
        """A record that used to fail wrote cleanly. Stop holding it."""
        self._ensure_writable()
        c = self._c()
        c.execute(
            "DELETE FROM quarantine WHERE source = ? AND record_key = ?",
            (source, record_key),
        )
        if _commit:
            c.commit()

    def poison_summary(self) -> list[dict[str, object]]:
        """One row per (source, error_type, terminal-or-not) with a count.

        The entire "what is broken" report, and the reason held rows are never
        deleted: a failure nobody can count is a gap that reads as coverage.
        """
        rows = self._c().execute(
            "SELECT source, error_type, "
            "       next_retry_at IS NULL AS terminal, count(*) AS n "
            "FROM quarantine GROUP BY source, error_type, terminal "
            "ORDER BY source, error_type"
        )
        return [
            {
                "source": row["source"],
                "error_type": row["error_type"],
                "terminal": bool(row["terminal"]),
                "count": int(row["n"]),
            }
            for row in rows
        ]

    # -- ingest state: input that will never parse -------------------------

    def read_faults(self, source: str) -> list[dict[str, object]]:
        """Every permanent fault recorded for one source. Policy is the caller's."""
        rows = self._c().execute(
            "SELECT source, fault_key, scope, scope_stamp, reason, detail, "
            "record_count, line, first_seen_at, last_seen_at "
            "FROM poison_faults WHERE source = ? ORDER BY scope, reason",
            (source,),
        )
        return [dict(row) for row in rows]

    def record_fault(
        self,
        source: str,
        fault_key: str,
        *,
        scope: str,
        scope_stamp: str,
        reason: str,
        detail: str,
        record_count: int,
        line: str,
        at: str,
        _commit: bool = True,
    ) -> None:
        """Remember one permanently-bad input, or refresh what is known about it.

        ``first_seen_at`` survives every refresh — how long something has been
        broken is the number that decides whether a human cares, and it is the
        one ``aggregator status`` prints. ``scope_stamp`` does NOT survive: it
        is refreshed on every sighting so that "the file changed" keeps meaning
        "changed since we last looked at it", which is what lets a fault
        outlive an append to its file without re-alarming.
        """
        self._ensure_writable()
        c = self._c()
        try:
            c.execute(
                """
                INSERT INTO poison_faults(
                    source, fault_key, scope, scope_stamp, reason, detail,
                    record_count, line, first_seen_at, last_seen_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source, fault_key) DO UPDATE SET
                    scope_stamp  = excluded.scope_stamp,
                    record_count = excluded.record_count,
                    line         = excluded.line,
                    last_seen_at = excluded.last_seen_at
                """,
                (
                    source,
                    fault_key,
                    scope,
                    scope_stamp,
                    reason,
                    detail,
                    record_count,
                    line,
                    at,
                    at,
                ),
            )
        except BaseException:
            if _commit:
                c.rollback()
            raise
        if _commit:
            c.commit()

    def forget_fault(self, source: str, fault_key: str, *, _commit: bool = True) -> None:
        """This input parses again (or is gone). Stop reporting it as quarantined.

        Deleted rather than tombstoned, unlike ``quarantine``: a held RECORD is
        kept forever because a failure nobody can count is a gap that reads as
        coverage, but a fault that no longer reproduces describes no gap at all
        — the records are in the index. Keeping the row would make
        ``aggregator status`` overstate the damage, which is the same lie in
        the other direction.
        """
        self._ensure_writable()
        c = self._c()
        c.execute(
            "DELETE FROM poison_faults WHERE source = ? AND fault_key = ?",
            (source, fault_key),
        )
        if _commit:
            c.commit()

    def fault_summary(self) -> list[dict[str, object]]:
        """Every permanent fault, newest-source-first — the whole quiet set.

        Rows rather than counts, because the question ``aggregator status``
        answers is "what exactly is being held quiet, and since when", and an
        aggregate cannot name the file.
        """
        rows = self._c().execute(
            "SELECT source, fault_key, scope, reason, detail, record_count, "
            "       line, first_seen_at, last_seen_at "
            "FROM poison_faults ORDER BY source, scope, reason"
        )
        return [
            {
                "source": row["source"],
                "fault_key": row["fault_key"],
                "scope": row["scope"],
                "reason": row["reason"],
                "detail": row["detail"],
                "count": int(row["record_count"]),
                "line": row["line"],
                "first_seen_at": row["first_seen_at"],
                "last_seen_at": row["last_seen_at"],
            }
            for row in rows
        ]

    # -- writes: legacy records (GitHub) ----------------------------------

    def upsert(self, records: list[Record], *, _commit: bool = True) -> int:
        """Write records to the store, scrubbing every field pre-write.

        Idempotent per ``stable_id``: re-upsert of the same ID overwrites the
        row (INSERT ... ON CONFLICT DO UPDATE); a fresh ID inserts a new row.
        A row whose ``src_hash`` already matches is not written at all and is
        counted in the return value — see :meth:`upsert_entities`.

        A batch that raises part-way through is ROLLED BACK, for the same
        reason as ``upsert_entities`` — the rows written before the raise
        would otherwise sit in the implicit transaction and be committed by
        the next writer, uncounted by any report.

        ``_commit=False`` leaves the transaction open, so a caller can make
        this write and the watermark advance that describes it ONE atomic unit.
        Same escape hatch, and same rollback rule, as ``upsert_entities``.
        """
        self._ensure_writable()
        c = self._c()
        try:
            unchanged = self._do_write_records(c, list(records))
        except BaseException:
            if _commit:
                c.rollback()
            raise
        if _commit:
            c.commit()
        return unchanged

    def rebuild(self, source: str) -> None:
        """Drop all Record-shaped rows for one source; caller re-ingests."""
        self._ensure_writable()
        c = self._c()
        c.execute("DELETE FROM records WHERE source = ?", (source,))
        c.execute("DELETE FROM records_fts WHERE source = ?", (source,))
        _purge_orphan_vectors(c, "records")
        c.commit()

    def rebuild_and_upsert(
        self,
        source: str,
        records: Iterable[Record],
        *,
        min_records: int = 0,
    ) -> None:
        """Atomic replacement of one source's records (GitHub path).

        Round-2 MEDIUM / round-3 HIGH: DELETE + upsert inside one savepoint;
        ``min_records`` guards against silent wipes.
        """
        record_list = list(records)
        if len(record_list) < min_records:
            raise EmptyRebuildRefusedError(
                f"refusing to rebuild {source!r}: got {len(record_list)} "
                f"records, min_records={min_records}"
            )
        self._ensure_writable()
        c = self._c()
        c.execute("SAVEPOINT rebuild_and_upsert")
        try:
            c.execute("DELETE FROM records WHERE source = ?", (source,))
            c.execute("DELETE FROM records_fts WHERE source = ?", (source,))
            self._do_write_records(c, record_list)
            # AFTER the re-write, so a row that came back keeps its vector
            # instead of being purged and re-embedded for no reason.
            _purge_orphan_vectors(c, "records")
        except BaseException:
            c.execute("ROLLBACK TO SAVEPOINT rebuild_and_upsert")
            c.execute("RELEASE SAVEPOINT rebuild_and_upsert")
            raise
        c.execute("RELEASE SAVEPOINT rebuild_and_upsert")
        c.commit()

    @staticmethod
    def _do_write_records(c: sqlite3.Connection, records: list[Record]) -> int:
        """Shared write body between ``upsert`` and ``rebuild_and_upsert``.

        Kept static so the savepoint scope in the atomic path can call it
        without touching module-level state. Body kept in lockstep with
        ``upsert``; if the write path grows a new field, update both.

        Returns how many rows were already stored identically and therefore
        neither scrubbed nor written.
        """
        unchanged = 0
        for group in _chunked(records, _HASH_PROBE_CHUNK):
            unchanged += Store._write_record_group(c, group)
        return unchanged

    @staticmethod
    def _write_record_group(
        c: sqlite3.Connection, records: Sequence[Record]
    ) -> int:
        """Write one group of records, skipping the ones that did not change.

        THE FINGERPRINT IS OVER THE EFFECTIVE ROW, NOT THE INCOMING ONE. Two of
        the columns below are merged rather than overwritten (``created_at``
        never moves once known; a dateless re-observation must not erase a
        stored ``updated_at``), so hashing the payload as it arrived would make
        every dateless TickTick re-observation of a dated row look like a change
        and rewrite it on every single tick — the doom loop in miniature. The
        merge is therefore applied here first, against the stored values, and
        the hash describes what the row WILL be.
        """
        ids = [r.stable_id for r in records]
        stored: dict[str, tuple[str | None, str | None, str | None]] = {}
        for page in _chunked(ids, 500):
            placeholders = ",".join("?" * len(page))
            rows = c.execute(
                "SELECT stable_id, src_hash, created_at, updated_at FROM records "
                f"WHERE stable_id IN ({placeholders})",  # noqa: S608 - fixed literals
                list(page),
            )
            for row in rows:
                stored[row[0]] = (row[1], row[2], row[3])

        # Once per group, matching the observations path. The probe runs a
        # statement against the vec table, so asking it per row would put it in
        # the write loop for no added truth.
        vec_usable = _vec_table_usable(c, "vec_records")

        unchanged = 0
        for r in records:
            held_hash, held_created, held_updated = stored.get(
                r.stable_id, (None, None, None)
            )
            effective_created = held_created or _iso(r.created_at)
            effective_updated = _iso(r.updated_at) or held_updated
            digest = _src_hash(
                SCRUB_FINGERPRINT,
                r.stable_id,
                r.source,
                r.subject,
                r.body,
                json.dumps(r.tags),
                effective_created,
                effective_updated,
                json.dumps(r.extra, default=str),
            )
            if held_hash == digest:
                unchanged += 1
                continue
            Store._write_one_record(
                c, r, digest, r.stable_id in stored, vec_usable=vec_usable
            )
        return unchanged

    @staticmethod
    def _write_one_record(
        c: sqlite3.Connection,
        r: Record,
        digest: str,
        existed: bool = False,
        *,
        vec_usable: bool = False,
    ) -> None:
        scrubbed_body = scrub(r.body).text
        scrubbed_subject = scrub(r.subject).text
        c.execute(
            """
            INSERT INTO records(
                stable_id, source, subject, body, tags,
                created_at, updated_at, extra, src_hash
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(stable_id) DO UPDATE SET
                subject    = excluded.subject,
                body       = excluded.body,
                tags       = excluded.tags,
                src_hash   = excluded.src_hash,
                -- Existing value first, so a creation time never moves
                -- once known — that is why created_at was left out of
                -- this SET list originally. Omitting it entirely went too
                -- far: a row first written dateless kept its NULL forever,
                -- including after ticktick's CREATED_FIELD is corrected,
                -- which breaks the tripwire's promise that fixing the
                -- field names also repairs the records it already wrote.
                created_at = COALESCE(records.created_at, excluded.created_at),
                -- COALESCE, not a plain overwrite: a re-observation that
                -- carries NO timestamp must not erase the one already
                -- stored. ticktick is where this bites — a task payload
                -- with none of completedTime/modifiedTime/createdTime
                -- yields updated_at=None, and the merge lets the fresher
                -- API observation win, so a NULL landed on top of the real
                -- date the CSV leg parsed. The row then drops out of every
                -- date query while looking perfectly healthy, and only a
                -- WHOLE batch being dateless trips the API tripwire, so a
                -- partially-dated batch degrades in silence.
                updated_at = COALESCE(excluded.updated_at, records.updated_at),
                extra      = excluded.extra,
                -- BACK INTO THE EMBED BACKLOG, for the same reason as
                -- observations: subject and body just changed, so a vector
                -- held for this row describes text that is no longer here.
                -- records_fts is rebuilt three lines down; this is the vector
                -- arm's half of the same refresh.
                embedding_state = NULL
            """,
            (
                r.stable_id,
                r.source,
                scrubbed_subject,
                scrubbed_body,
                json.dumps(r.tags),
                r.created_at.isoformat() if r.created_at else None,
                r.updated_at.isoformat() if r.updated_at else None,
                json.dumps(r.extra, default=str),
                digest,
            ),
        )
        # ONLY when the row was already there, matching the observations path.
        # During a rebuild the row was just DELETEd, so ``existed`` is False and
        # its vector survives the re-insert — the row is back at
        # ``embedding_state IS NULL`` and will be re-embedded, but it keeps
        # serving the vector arm in the meantime instead of going dark.
        # ``vec_usable`` is probed once per group by the caller, and defaults to
        # False so a future call site that forgets it degrades to "leave the
        # vectors alone" rather than to a raise inside the row write.
        if existed and vec_usable:
            _drop_row_vectors(c, "vec_records", "stable_id", r.stable_id)
        c.execute("DELETE FROM records_fts WHERE stable_id = ?", (r.stable_id,))
        c.execute(
            "INSERT INTO records_fts(stable_id, source, subject, body, tags) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                r.stable_id,
                r.source,
                scrubbed_subject,
                scrubbed_body,
                " ".join(r.tags),
            ),
        )

    # -- reads: legacy Record-shaped path (GitHub) ------------------------

    def _build_where(self, ast: QueryAST) -> tuple[str, list]:
        """WHERE builder for records queries."""
        clauses = ["1=1"]
        params: list = []
        if ast.source and ast.source != "sessions":
            clauses.append("source = ?")
            params.append(ast.source)
        if ast.from_date:
            clauses.append("(created_at >= ? OR updated_at >= ?)")
            iso = ast.from_date.isoformat()
            params.extend([iso, iso])
        if ast.to_date:
            clauses.append("(created_at <= ? OR updated_at <= ?)")
            iso = ast.to_date.isoformat()
            params.extend([iso, iso])
        for tag in ast.tags:
            clauses.append("tags LIKE ?")
            params.append(f'%"{tag}"%')
        self._apply_id_scope(ast, "stable_id", clauses, params)
        return " AND ".join(clauses), params

    @staticmethod
    def _apply_id_scope(
        ast: QueryAST, column: str, clauses: list[str], params: list
    ) -> None:
        """Append the v5 hybrid ``id_scope`` filter, if the caller set one.

        ``None`` is no filter at all. An EMPTY scope is not the same thing: it
        means the retriever fused to nothing, and it renders as ``1=0`` because
        SQL ``IN ()`` is a syntax error, not an empty set. Getting that
        backwards turns "no results" into a 500 at the tool boundary.

        ONE BOUND PARAMETER, NOT ONE PER ID. The obvious
        ``IN (?,?,?…)`` breaks at ``SQLITE_MAX_VARIABLE_NUMBER`` — 32,766 here
        — and this scope is routinely bigger than that: it is the FTS5 arm
        UNION the vector arm, and the FTS5 arm is deliberately uncapped (see
        ``mcp._fused_id_scope``), so it holds one id per row the term matched.
        On a 483k-observation corpus a common word clears the limit easily, so
        the binding failed on precisely the queries most worth running — and
        failed two different ways: ``query_observations`` swallowed the
        ``OperationalError`` and returned an empty page, while
        ``count_observations`` let it propagate.

        ``json_each`` takes the whole list as a single JSON parameter and is
        still index-driven — the plan stays
        ``SEARCH … USING INDEX (obs_id=?)`` with the scope as a LIST SUBQUERY,
        and 60k ids resolve in ~20 ms. JSON1 has been compiled into SQLite by
        default since 3.38; this build is 3.53. Shared with the FTS5 arm's
        three binding sites via :func:`_json_id_clause` — round 2's S2 found
        the same defect there, and one technique is enough.
        """
        if ast.id_scope is None:
            return
        if not ast.id_scope:
            clauses.append("1=0")
            return
        clauses.append(_json_id_clause(column))
        params.append(json.dumps(sorted(ast.id_scope)))

    def _fts_rows(self, sql: str, params: Sequence, text: str) -> list[sqlite3.Row]:
        """Run one MATCH-bearing SELECT, and be LOUD if it still fails.

        After :func:`fts5_match_query` nothing reaching ``MATCH`` can be a
        syntax error, so a failure here is a lock, a corrupt index, or an FTS5
        that has changed under us. The exception is re-raised UNCHANGED — the
        callers above each have a considered answer to it, and swallowing it
        here would turn "the index is broken" into "there are no matches",
        which is the empty-result-looks-like-success failure this project bans
        by name. The log line exists so the degrade-to-vector-only path in
        ``mcp._fused_id_scope`` can never be silent: that path answers from the
        vector arm alone, which is right for the caller, but a query answered
        by one arm must say so somewhere.
        """
        try:
            return self._c().execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            log.exception(
                "FTS5 MATCH failed for %r (rewritten to %r) — the lexical arm "
                "contributes nothing to this query",
                text,
                params[0] if params else "",
            )
            raise

    def _fts_ids(self, text: str) -> set[str]:
        """Records-path FTS5 arm. Empty set when the text has no word chars."""
        match_expr = fts5_match_query(text)
        if not match_expr:
            return set()
        rows = self._fts_rows(
            "SELECT stable_id FROM records_fts WHERE records_fts MATCH ?",
            (match_expr,),
            text,
        )
        return {row["stable_id"] for row in rows}

    def query(
        self,
        ast: QueryAST,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Record]:
        """Return records matching the AST, ordered by ``updated_at`` desc.

        Records-shaped path (GitHub). For sessions use ``query_sessions`` or
        ``query_observations``.

        ``source == 'sessions'`` falls through here as a no-op — sessions
        aren't stored in ``records`` in v2. Callers that want session hits
        should route through ``query_sessions``.
        """
        if ast.source == "sessions":
            return []
        where, params = self._build_where(ast)
        c = self._c()
        base_sql = f"SELECT * FROM records WHERE {where} ORDER BY updated_at DESC"
        # Codex Phase 2 HIGH: when FTS text is present, we must intersect
        # BEFORE pagination — SQL LIMIT applied ahead of the FTS filter
        # drops legitimate matches beyond the first page and can return an
        # empty page while ``count()`` reports hits. Fetch the full ordered
        # side + Python-side FTS intersect + Python-side slice. Matches
        # the union path's approach; safe at v2 scale (records table is
        # bounded by GitHub search API pagination).
        if ast.text:
            try:
                fts_ids = self._fts_ids(ast.text)
            except sqlite3.OperationalError as e:
                log.warning("FTS5 syntax error for query %r: %s", ast.text, e)
                return []
            try:
                rows = list(c.execute(base_sql, params))
            except sqlite3.OperationalError as e:
                log.warning("store.query base SELECT failed for ast=%r: %s", ast, e)
                return []
            filtered = [row for row in rows if row["stable_id"] in fts_ids]
            lo = int(offset)
            hi = None if limit is None else lo + int(limit)
            return [_row_to_record(row) for row in filtered[lo:hi]]

        sql = base_sql
        if limit is not None or offset:
            sql += " LIMIT ? OFFSET ?"
            params = [*params, (-1 if limit is None else int(limit)), int(offset)]
        try:
            rows = list(c.execute(sql, params))
        except sqlite3.OperationalError as e:
            log.warning("store.query base SELECT failed for ast=%r: %s", ast, e)
            return []
        return [_row_to_record(row) for row in rows]

    def count(self, ast: QueryAST) -> int:
        """Return the total number of records matching ``ast`` (Records path)."""
        if ast.source == "sessions":
            return 0
        where, params = self._build_where(ast)
        c = self._c()
        if ast.text:
            try:
                fts_ids = self._fts_ids(ast.text)
            except sqlite3.OperationalError as e:
                log.warning("FTS5 syntax error in count for %r: %s", ast.text, e)
                return 0
            id_rows = c.execute(
                f"SELECT stable_id FROM records WHERE {where}", params
            ).fetchall()
            return sum(1 for row in id_rows if row["stable_id"] in fts_ids)
        row = c.execute(
            f"SELECT COUNT(*) AS n FROM records WHERE {where}", params
        ).fetchone()
        return int(row["n"]) if row else 0

    def count_by_source(self, source: str) -> int:
        """Return the number of rows currently held for ``source``.

        For ``sessions`` this counts the sessions table (not observations —
        the guard cares about "is there historical data here" per source).
        """
        c = self._c()
        if source == "sessions":
            row = c.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()
        else:
            row = c.execute(
                "SELECT COUNT(*) AS n FROM records WHERE source = ?", (source,)
            ).fetchone()
        return int(row["n"]) if row else 0

    def count_sessions_by_origin(self, origins: Sequence[str] | None = None) -> int:
        """Rows in ``sessions``, optionally restricted to ``origins``.

        Exists so the CLI's shrink guard can measure the SAME population its
        DELETE will reach. Measured against the whole table instead, a rebuild
        that replaces 840 claude-code sessions in a store also holding 160
        claude-web ones reads as a 16% shrink — inside the 20% slack, so no
        prompt fires — while actually destroying every claude-web row.
        """
        c = self._c()
        if origins is None:
            row = c.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()
        else:
            scope = list(origins)
            if not scope:
                return 0
            placeholders = ",".join("?" * len(scope))
            row = c.execute(
                f"SELECT COUNT(*) AS n FROM sessions WHERE origin IN ({placeholders})",
                scope,
            ).fetchone()
        return int(row["n"]) if row else 0

    def existing_ids(self, table: str, ids: Iterable[str]) -> set[str]:
        """Return the subset of ``ids`` already present in ``table``.

        Read-only batch existence probe. Exists so an import sink can report
        REAL added-vs-updated counts: both write paths are upserts, so after
        the fact there is no way to tell a fresh row from an overwritten one,
        and a summary that always says ``added=<batch size>`` is the bug
        ``cli.py`` currently ships.

        ``table`` is interpolated into the SQL (SQLite cannot parameterise an
        identifier) and is therefore constrained to the three-entry allowlist
        below — never caller-supplied text. The ids themselves are bound.
        """
        pk = _PK_BY_TABLE.get(table)
        if pk is None:
            raise ValueError(
                f"unknown table {table!r}; expected one of {sorted(_PK_BY_TABLE)}"
            )
        id_list = list(ids)
        if not id_list:
            return set()
        found: set[str] = set()
        c = self._c()
        # Chunked to stay under SQLITE_MAX_VARIABLE_NUMBER (999 on older
        # builds) regardless of the caller's batch size.
        for start in range(0, len(id_list), 500):
            chunk = id_list[start : start + 500]
            placeholders = ",".join("?" * len(chunk))
            rows = c.execute(
                f"SELECT {pk} AS id FROM {table} WHERE {pk} IN ({placeholders})",
                chunk,
            ).fetchall()
            found.update(row["id"] for row in rows)
        return found

    def probe_fts(self, text: str) -> None:
        """Run cheap MATCH probes over both FTS indexes.

        Checks both records_fts and obs_fts so a failure is caught regardless
        of which index the actual query would hit.

        THIS NO LONGER REJECTS USER TEXT. It used to be the place a caller
        learned its freeform query was malformed — ``power-on``, ``#178``,
        ``cache.db`` all raised here — but :func:`fts5_match_query` means the
        expression handed to ``MATCH`` cannot be malformed. What survives is
        the OTHER half of the job, which is the half that matters more: the
        probe runs the same statement the real query will run against the same
        tables, so a locked or corrupt cache still raises here and is reported
        as a cache problem rather than as a bad query.

        Text with no word characters at all probes nothing, because the query
        it stands in for will not run a ``MATCH`` either. Callers that use this
        as a HEALTH probe must therefore pass text with word characters in it;
        ``tests/core/test_store_fts_sanitize.py`` pins that for the one
        constant that does.
        """
        match_expr = fts5_match_query(text)
        if not match_expr:
            return
        self._fts_rows(
            "SELECT rowid FROM records_fts WHERE records_fts MATCH ? LIMIT 1",
            (match_expr,),
            text,
        )
        self._fts_rows(
            "SELECT rowid FROM obs_fts WHERE obs_fts MATCH ? LIMIT 1",
            (match_expr,),
            text,
        )

    # -- reads: v2 sessions + observations --------------------------------

    def _sessions_where(self, ast: QueryAST) -> tuple[str, list]:
        """Build WHERE for a ``sessions`` query.

        Precedence: ``top_session_id`` (exact ``session_id``) > ``session_id``
        (matches ``root_session_id`` — includes subagents) >
        ``agent_id`` filter.

        Codex Phase 2 MEDIUM: honour ``ast.source`` kind split.
        ``source:sessions`` filters to top-level rows (``kind='session'``);
        ``source:subagents`` filters to subagent rows (``kind='subagent'``).
        Both route through the sessions ontology upstream, but without the
        kind filter they returned identical rows.

        v3: the kind buckets implicitly mean claude-code, so both also pin
        ``origin='claude-code'`` — chat-export rows are kind='session' too
        and must not leak in. ``source:chatgpt`` / ``source:claude-web``
        filter on origin alone (chat exports carry no subagents, so no kind
        constraint).
        """
        clauses = ["1=1"]
        params: list = []
        if ast.source == "sessions":
            clauses.append("kind = ? AND origin = 'claude-code'")
            params.append("session")
        elif ast.source == "subagents":
            clauses.append("kind = ? AND origin = 'claude-code'")
            params.append("subagent")
        elif ast.source in CHAT_ORIGINS:
            clauses.append("origin = ?")
            params.append(ast.source)
        if ast.top_session_id:
            clauses.append("session_id = ?")
            params.append(ast.top_session_id)
        elif ast.session_id:
            clauses.append("root_session_id = ?")
            params.append(ast.session_id)
        if ast.agent_id:
            clauses.append("agent_id = ?")
            params.append(ast.agent_id)
        if ast.active_from:
            clauses.append("last_ts >= ?")
            params.append(ast.active_from.isoformat())
        if ast.active_to:
            clauses.append("first_ts <= ?")
            params.append(ast.active_to.isoformat())
        # from_date/to_date map to activity window as well for backwards compat
        # when the caller passes plain from:/to: rather than active_from/to.
        if ast.from_date and not ast.active_from:
            clauses.append("last_ts >= ?")
            params.append(ast.from_date.isoformat())
        if ast.to_date and not ast.active_to:
            clauses.append("first_ts <= ?")
            params.append(ast.to_date.isoformat())
        return " AND ".join(clauses), params

    @staticmethod
    def _provenance_clause(provenance: str, column: str = "provenance") -> tuple[str, list]:
        """``(SQL fragment, params)`` for one ``by:`` value.

        ``machine`` EXPANDS TO THE FOUR NON-HUMAN MEMBERS rather than to
        ``!= 'human'``, and that difference is the whole point: ``!=`` in
        SQLite is false against NULL, which would be right by accident today
        and wrong the moment anyone writes ``NOT (provenance = 'human')``. An
        explicit ``IN`` says what is meant — a positive machine claim — and
        keeps unclassified rows out of BOTH sides, where they belong: NULL
        means nobody has looked, which is not a fact about authorship.
        """
        if provenance == MACHINE:
            marks = ",".join("?" * len(MACHINE_VALUES))
            return f"{column} IN ({marks})", list(MACHINE_VALUES)
        return f"{column} = ?", [provenance]

    def _apply_provenance(
        self, ast: QueryAST, clauses: list[str], params: list
    ) -> None:
        """Append the ``by:`` predicate, or nothing at all.

        NOTHING AT ALL IS THE DEFAULT AND STAYS THE DEFAULT. A human-only
        default here would narrow ``_first_user_prompt`` (which labels every
        session card), ``count_observations`` (which is
        ``matching_observations``), ``_session_body_preview`` and the frozen
        eval baseline — four callers that never asked for a filter, in one
        edit, invisibly.
        """
        if not ast.provenance:
            return
        clause, extra = self._provenance_clause(ast.provenance)
        clauses.append(clause)
        params.extend(extra)

    def has_unclassified_observations(self) -> bool:
        """Is any observation still ``provenance IS NULL``?

        ONE INDEXED PROBE, not a count: ``LIMIT 1`` against ``obs_provenance``.
        It exists so an empty ``by:`` page can say WHY it is empty. Before the
        backfill has run, every row is NULL and every ``by:`` filter matches
        nothing — an answer indistinguishable from "you never said that",
        which is precisely the failure this mission exists to remove.
        """
        row = self._c().execute(
            "SELECT 1 FROM observations WHERE provenance IS NULL LIMIT 1"
        ).fetchone()
        return row is not None

    def count_unclassified_observations(self) -> int:
        """How many observations are still ``provenance IS NULL``.

        The backfill's remaining-work number, printed at the end of a run.
        A COUNT rather than the ``LIMIT 1`` probe above because "how much is
        left" is the question an operator asks after an interrupted pass, and
        "some" does not answer it. It is an index-only scan of
        ``obs_provenance`` and is paid once per run, not per query.
        """
        return int(
            self._c().execute(
                "SELECT count(*) FROM observations WHERE provenance IS NULL"
            ).fetchone()[0]
        )

    def _obs_where(self, ast: QueryAST) -> tuple[str, list]:
        """Build WHERE for an ``observations`` query.

        ``session_id`` matches ``root_session_id`` (Langfuse-style: includes
        subagents). ``top_session_id`` matches the exact ``session_id`` (only
        the top-level stream). ``agent_id`` filters via the parent session
        row.

        Codex Phase 2 MEDIUM: ``source:sessions`` / ``source:subagents``
        restrict to observations owned by that kind (via the sessions
        table). Kept in lockstep with ``_sessions_where``.

        v3: kind buckets also pin ``origin='claude-code'``;
        ``source:chatgpt`` / ``source:claude-web`` filter on the owning
        session's origin via the same subselect pattern.
        """
        clauses = ["1=1"]
        params: list = []
        if ast.source in ("sessions", "subagents"):
            kind = "session" if ast.source == "sessions" else "subagent"
            clauses.append(
                "session_id IN (SELECT session_id FROM sessions "
                "WHERE kind = ? AND origin = 'claude-code')"
            )
            params.append(kind)
        elif ast.source in CHAT_ORIGINS:
            clauses.append(
                "session_id IN (SELECT session_id FROM sessions WHERE origin = ?)"
            )
            params.append(ast.source)
        if ast.top_session_id:
            clauses.append("session_id = ?")
            params.append(ast.top_session_id)
        elif ast.session_id:
            clauses.append("root_session_id = ?")
            params.append(ast.session_id)
        if ast.agent_id:
            clauses.append(
                "session_id IN (SELECT session_id FROM sessions WHERE agent_id = ?)"
            )
            params.append(ast.agent_id)
        if ast.obs_type:
            clauses.append("type = ?")
            params.append(ast.obs_type)
        self._apply_provenance(ast, clauses, params)
        if ast.from_date:
            clauses.append("ts >= ?")
            params.append(ast.from_date.isoformat())
        if ast.to_date:
            clauses.append("ts <= ?")
            params.append(ast.to_date.isoformat())
        self._apply_id_scope(ast, "obs_id", clauses, params)
        return " AND ".join(clauses), params

    def query_sessions(
        self,
        ast: QueryAST,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[SessionRow]:
        """Return session rows matching the AST.

        Ordered by ``last_ts`` descending — most-recently-active first.
        Text search flows through obs_fts and back to sessions via the
        denormalized ``root_session_id`` so a hit anywhere in a subagent
        stream still surfaces the top-level session.

        Orphan-root synthesis (B2 fix): if ``top_session_id=X`` is set and
        no top-level session with ``session_id=X`` exists, but subagents
        reference X via ``root_session_id``, synthesise one placeholder
        SessionRow so the caller sees the session exists. Common cause: the
        top-level JSONL was still being written when ingest ran and got
        skipped by the 5-min live-file guard, while its subagent files
        (older, quiescent) were ingested.
        """
        where, params = self._sessions_where(ast)
        c = self._c()
        if ast.text or ast.id_scope is not None:
            try:
                root_ids, exact_ids = self._hit_scope(ast)
            except sqlite3.OperationalError as e:
                log.warning("query_sessions FTS5 syntax %r: %s", ast.text, e)
                return []
            if not root_ids and not exact_ids:
                return []
            clause, scope_params = self._fts_scope_clause(root_ids, exact_ids)
            where += " AND " + clause
            params = [*params, *scope_params]
        sql = f"SELECT * FROM sessions WHERE {where} ORDER BY last_ts DESC"
        if limit is not None or offset:
            sql += " LIMIT ? OFFSET ?"
            params = [*params, (-1 if limit is None else int(limit)), int(offset)]
        try:
            rows = list(c.execute(sql, params))
        except sqlite3.OperationalError as e:
            log.warning("query_sessions failed: %s", e)
            return []
        sessions = [_row_to_session(row) for row in rows]
        if (
            ast.top_session_id
            and not sessions
            and not ast.text
            and ast.id_scope is None
            and offset == 0
        ):
            orphan = self._synthesise_orphan_root(ast.top_session_id)
            if orphan is not None:
                sessions.append(orphan)
        return sessions

    def count_sessions(self, ast: QueryAST) -> int:
        """Match count for ``query_sessions`` (for MCP ``total``)."""
        where, params = self._sessions_where(ast)
        c = self._c()
        if ast.text or ast.id_scope is not None:
            try:
                root_ids, exact_ids = self._hit_scope(ast)
            except sqlite3.OperationalError:
                return 0
            if not root_ids and not exact_ids:
                return 0
            clause, scope_params = self._fts_scope_clause(root_ids, exact_ids)
            where += " AND " + clause
            params = [*params, *scope_params]
        row = c.execute(
            f"SELECT COUNT(*) AS n FROM sessions WHERE {where}", params
        ).fetchone()
        n = int(row["n"]) if row else 0
        if (
            n == 0
            and ast.top_session_id
            and not ast.text
            and ast.id_scope is None
            and self._synthesise_orphan_root(ast.top_session_id) is not None
        ):
            # Orphan-root synth: count 1 when only subagents exist.
            n = 1
        return n

    def _synthesise_orphan_root(self, top_sid: str) -> SessionRow | None:
        """Return a placeholder ``kind='session'`` row for a session id whose
        subagents were ingested but whose top-level JSONL is missing (B2).

        Uses the subagents' first_ts/last_ts range as the parent window and
        borrows cwd/git_branch via ``MIN()`` across the subagents — the
        alphabetically smallest non-null string wins, chosen just so the
        surface has a stable non-null display metadata (not an "earliest"
        signal — round-1 MEDIUM docstring correction). Returns None when
        no subagents reference the id either — genuinely unknown session,
        don't fabricate.
        """
        c = self._c()
        row = c.execute(
            """
            SELECT MIN(first_ts) AS lo, MAX(last_ts) AS hi,
                   MIN(cwd)      AS cwd, MIN(git_branch) AS git_branch
            FROM sessions
            WHERE kind='subagent' AND root_session_id=?
            """,
            (top_sid,),
        ).fetchone()
        if not row or row["lo"] is None:
            return None
        return SessionRow(
            session_id=top_sid,
            root_session_id=top_sid,
            parent_session_id=None,
            kind="session",
            agent_id=None,
            agent_type=None,
            spawned_by_tool_use_id=None,
            cwd=row["cwd"],
            git_branch=row["git_branch"],
            first_ts=_parse_iso(row["lo"]),
            last_ts=_parse_iso(row["hi"]),
            jsonl_path="(not ingested — top-level JSONL missing at scan time)",
        )

    def query_observations(
        self,
        ast: QueryAST,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[ObservationRow]:
        """Return observation rows matching the AST, ordered by ``ts``.

        ONE BOUND PARAMETER FOR THE HIT LIST, not one per hit — see
        :func:`_json_id_clause`. Binding per hit broke above 32,766 matches,
        which a common word clears easily on a 483k-observation corpus, and it
        broke silently here: the ``OperationalError`` handler below turned
        "tens of thousands of hits" into an empty page, so the recall tool
        answered "nothing found" on precisely the queries most worth running.
        """
        where, params = self._obs_where(ast)
        c = self._c()
        if ast.text:
            try:
                obs_ids = self._scoped_obs_ids(ast)
            except sqlite3.OperationalError as e:
                log.warning("FTS5 syntax error in query_observations %r: %s", ast.text, e)
                return []
            if not obs_ids:
                return []
            where += " AND " + _json_id_clause("obs_id")
            params = [*params, json.dumps(sorted(obs_ids))]
        sql = f"SELECT * FROM observations WHERE {where} ORDER BY ts ASC"
        if limit is not None or offset:
            sql += " LIMIT ? OFFSET ?"
            params = [*params, (-1 if limit is None else int(limit)), int(offset)]
        try:
            rows = list(c.execute(sql, params))
        except sqlite3.OperationalError as e:
            log.warning("query_observations failed: %s", e)
            return []
        return [_row_to_observation(row) for row in rows]

    def count_observations(self, ast: QueryAST) -> int:
        """Match count for ``query_observations`` (for MCP ``total``).

        Same one-parameter binding as its twin, and it failed the OTHER way
        before: the ``COUNT`` below is unwrapped, so an oversized hit list
        propagated ``too many SQL variables`` to the caller while
        ``query_observations`` silently returned nothing for the same query.
        """
        where, params = self._obs_where(ast)
        c = self._c()
        if ast.text:
            try:
                obs_ids = self._scoped_obs_ids(ast)
            except sqlite3.OperationalError:
                return 0
            if not obs_ids:
                return 0
            where += " AND " + _json_id_clause("obs_id")
            params = [*params, json.dumps(sorted(obs_ids))]
        row = c.execute(
            f"SELECT COUNT(*) AS n FROM observations WHERE {where}", params
        ).fetchone()
        return int(row["n"]) if row else 0

    def _fts_root_session_ids(self, text: str) -> list[str]:
        """FTS text → list of matching root_session_ids (dedup ordered).

        Used to project obs_fts hits back up to the sessions layer for
        session-level hit lists.
        """
        match_expr = fts5_match_query(text)
        if not match_expr:
            return []
        rows = self._fts_rows(
            """
            SELECT DISTINCT o.root_session_id AS root
            FROM obs_fts f
            JOIN observations o ON o.rowid = f.rowid
            WHERE obs_fts MATCH ?
            """,
            (match_expr,),
            text,
        )
        return [r["root"] for r in rows if r["root"]]

    @staticmethod
    def _fts_scope_clause(
        root_ids: set[str], exact_ids: set[str]
    ) -> tuple[str, list[str]]:
        """``(SQL fragment, params)`` pairing kinds with their FTS hit scope.

        Top-level cards surface on any hit under their root; subagent cards
        only on hits in their own stream. Empty sets render as ``1=0`` (SQL
        disallows ``IN ()``).

        RETURNS ITS OWN PARAMS rather than documenting an order for the caller
        to reproduce, because there is now at most one per part and getting
        the pairing wrong is silent. Each part binds ONE parameter however
        many ids it holds — see :func:`_json_id_clause`. Per-id binding broke
        above 32,766 ids, and broke asymmetrically: ``query_sessions``
        swallowed the ``OperationalError`` into an empty page while
        ``count_sessions`` raised it at the caller.
        """
        params: list[str] = []
        if root_ids:
            root_part = (
                "(kind = 'session' AND " + _json_id_clause("root_session_id") + ")"
            )
            params.append(json.dumps(sorted(root_ids)))
        else:
            root_part = "1=0"
        if exact_ids:
            exact_part = (
                "(kind = 'subagent' AND " + _json_id_clause("session_id") + ")"
            )
            params.append(json.dumps(sorted(exact_ids)))
        else:
            exact_part = "1=0"
        return f"({root_part} OR {exact_part})", params

    def _fts_hit_scope(
        self,
        text: str,
        obs_type: str | None = None,
        provenance: str | None = None,
    ) -> tuple[set[str], set[str]]:
        """FTS text (+ optional obs type and authorship) → ``(root_ids, exact_ids)``.

        ``root_ids``  — distinct ``root_session_id`` of matching obs; used
        to surface top-level session cards (a hit anywhere under the root
        counts — ``session:`` aggregates subagents).
        ``exact_ids`` — distinct ``session_id`` (composite for subagents);
        used so a subagent card surfaces only when its OWN stream matches.

        Live-model smoke MEDIUM (2026-08-02): the previous root-only
        mapping ignored ``type:`` and surfaced sibling subagents with
        zero own matches.

        ``provenance`` (v6) rides here for the identical reason ``obs_type``
        does. A card is supposed to surface only on a hit that PASSES the
        query's filters; a ``by:`` filter left out of this projection would
        surface cards whose only matching observation the caller just excluded
        — the same silent over-surfacing, one filter later.
        """
        match_expr = fts5_match_query(text)
        if not match_expr:
            return set(), set()
        sql = """
            SELECT DISTINCT o.root_session_id AS root, o.session_id AS sid
            FROM obs_fts f
            JOIN observations o ON o.rowid = f.rowid
            WHERE obs_fts MATCH ?
        """
        params: list = [match_expr]
        if obs_type:
            sql += " AND o.type = ?"
            params.append(obs_type)
        if provenance:
            clause, extra = self._provenance_clause(provenance, "o.provenance")
            sql += f" AND {clause}"
            params.extend(extra)
        rows = self._fts_rows(sql, params, text)
        roots = {r["root"] for r in rows if r["root"]}
        exacts = {r["sid"] for r in rows if r["sid"]}
        return roots, exacts

    def _session_hit_scope(
        self,
        text: str,
        obs_type: str | None = None,
        provenance: str | None = None,
    ) -> tuple[set[str], set[str]]:
        """``scope:session``: every conjunct satisfied SOMEWHERE under one root.

        The conjuncts are evaluated separately and their session sets are
        INTERSECTED, which is the whole difference from the default — under
        ``scope:observation`` a single row has to carry all of them, here one
        turn may carry "PR link" and a turn three hours later may carry "status
        report". That is a real question a caller asks ("the session where we
        discussed both"), and it is the one this ontology could not express at
        all until now; it is never the default because the answer it gives back
        is a session, not the moment, and the moment is what recall is for.

        Early-exit on an empty intersection: a conjunct nothing satisfies makes
        every later query pointless, and with N conjuncts each costing an FTS5
        scan that saving is the reason a long query stays affordable.

        THE PER-CONJUNCT STATEMENT IS INLINE RATHER THAN IN A HELPER, and that
        is not a style choice. ``tests/test_fts5_match_site_enumeration.py``
        requires the function that OWNS an FTS5 ``MATCH`` to be the function
        that rewrites the text, precisely because "a caller sanitizes it" is the
        belief that produced three instances of the unescaped-MATCH defect. A
        helper taking a pre-built ``expr`` would move the statement out from
        under that rule.
        """
        conjuncts = fts5_match_conjuncts(text)
        if not conjuncts:
            return set(), set()
        base = """
            SELECT DISTINCT o.root_session_id AS root, o.session_id AS sid
            FROM obs_fts f
            JOIN observations o ON o.rowid = f.rowid
            WHERE obs_fts MATCH ?
        """
        filters = ""
        extra_params: list = []
        if obs_type:
            filters += " AND o.type = ?"
            extra_params.append(obs_type)
        if provenance:
            clause, extra = self._provenance_clause(provenance, "o.provenance")
            filters += f" AND {clause}"
            extra_params.extend(extra)
        roots: set[str] | None = None
        exacts: set[str] | None = None
        for expr in conjuncts:
            rows = self._fts_rows(base + filters, [expr, *extra_params], text)
            got_roots = {r["root"] for r in rows if r["root"]}
            got_exacts = {r["sid"] for r in rows if r["sid"]}
            roots = got_roots if roots is None else roots & got_roots
            exacts = got_exacts if exacts is None else exacts & got_exacts
            if not roots and not exacts:
                return set(), set()
        return roots or set(), exacts or set()

    def _text_hit_scope(self, ast: QueryAST) -> tuple[set[str], set[str]]:
        """Whichever conjunction scope the AST asked for. ``scope:`` lives here
        and not in :meth:`_obs_where` because it is not a row predicate: it
        changes which ROWS THE TEXT MATCHES, not which of the matched rows
        survive a filter. Anyone adding a key to the registration list in the
        design note should read that difference off this method."""
        if ast.scope == SCOPE_SESSION:
            return self._session_hit_scope(
                ast.text or "", ast.obs_type, ast.provenance
            )
        return self._fts_hit_scope(ast.text or "", ast.obs_type, ast.provenance)

    def _obs_id_hit_scope(
        self,
        obs_ids: Iterable[str],
        obs_type: str | None = None,
        provenance: str | None = None,
    ) -> tuple[set[str], set[str]]:
        """``(root_ids, exact_ids)`` for a set of obs ids — the v5 twin of
        :meth:`_fts_hit_scope`, and it must stay behaviourally identical to it.

        The hybrid retriever fuses obs ids, but the hit list this feeds is
        session CARDS, so the ids have to be projected up the same way FTS
        hits are: a hit anywhere under a root surfaces the top-level card,
        while a subagent card surfaces only on a hit in its own stream.
        Diverging here would make hybrid and FTS5 answer the same question
        with differently-shaped hit lists.

        ``scope:`` IS DELIBERATELY NOT A PARAMETER HERE, and that is the one
        place in the v6 registration list where the answer is "nothing to
        register". This method projects an id set that some arm already
        produced; the conjunction it was produced by is upstream, and a fused
        hybrid id set has no conjunct structure left to intersect. Widening
        happens in :meth:`_text_hit_scope`, which :meth:`_hit_scope` then
        INTERSECTS with this projection — so a ``scope:session`` query through
        the hybrid path is widened exactly once, on the lexical side, rather
        than twice or not at all.

        Parameters are chunked because a fused scope can carry thousands of
        ids and SQLite caps host parameters per statement.
        """
        roots: set[str] = set()
        exacts: set[str] = set()
        c = self._c()
        for page in _chunked(sorted(obs_ids), 500):
            sql = (
                "SELECT DISTINCT root_session_id AS root, session_id AS sid "  # noqa: S608 - placeholders only
                "FROM observations "
                f"WHERE obs_id IN ({','.join('?' * len(page))})"
            )
            params: list = list(page)
            if obs_type:
                sql += " AND type = ?"
                params.append(obs_type)
            if provenance:
                clause, extra = self._provenance_clause(provenance)
                sql += f" AND {clause}"
                params.extend(extra)
            for row in c.execute(sql, params):
                if row["root"]:
                    roots.add(row["root"])
                if row["sid"]:
                    exacts.add(row["sid"])
        return roots, exacts

    def _hit_scope(self, ast: QueryAST) -> tuple[set[str], set[str]]:
        """Session-card hit scope for whichever arm(s) the AST carries.

        ``text`` alone is the FTS5 arm; ``id_scope`` alone is the hybrid arm.
        Both set means both must hold, so the scopes INTERSECT — a narrowing
        filter never widens the result, whichever order the arms arrive in.
        """
        if ast.id_scope is None:
            return self._text_hit_scope(ast)
        roots, exacts = self._obs_id_hit_scope(
            ast.id_scope, ast.obs_type, ast.provenance
        )
        if ast.text:
            fts_roots, fts_exacts = self._text_hit_scope(ast)
            roots &= fts_roots
            exacts &= fts_exacts
        return roots, exacts

    def _fts_obs_ids(self, text: str) -> list[str]:
        """Observations-path FTS5 arm. Empty when the text has no word chars."""
        match_expr = fts5_match_query(text)
        if not match_expr:
            return []
        rows = self._fts_rows(
            """
            SELECT o.obs_id AS obs_id
            FROM obs_fts f
            JOIN observations o ON o.rowid = f.rowid
            WHERE obs_fts MATCH ?
            """,
            (match_expr,),
            text,
        )
        return [r["obs_id"] for r in rows]

    def _session_scope_obs_ids(self, text: str, roots: set[str]) -> list[str]:
        """Observations under ``roots`` matching ANY conjunct.

        THE ROW SET IS WIDER THAN THE DEFAULT AND THAT IS THE POINT. Once a
        session has qualified by carrying every conjunct somewhere, the turns
        worth showing are the ones that carry any of them — the turn saying
        "PR link" and the turn saying "status report" are both part of the
        answer, and demanding both of each row again would collapse straight
        back to ``scope:observation``.

        One statement per conjunct rather than an ``OR``-joined expression, so
        the invariant that only quoted ``\\w+`` runs reach ``MATCH`` survives
        a widening that had no reason to spend it.
        """
        if not roots:
            return []
        scope_json = json.dumps(sorted(roots))
        ids: set[str] = set()
        for expr in fts5_match_conjuncts(text):
            rows = self._fts_rows(
                "SELECT o.obs_id AS obs_id FROM obs_fts f "  # noqa: S608 - placeholders only
                "JOIN observations o ON o.rowid = f.rowid "
                "WHERE obs_fts MATCH ? AND "
                + _json_id_clause("o.root_session_id"),
                (expr, scope_json),
                text,
            )
            ids.update(r["obs_id"] for r in rows)
        return sorted(ids)

    def _scoped_obs_ids(self, ast: QueryAST) -> list[str]:
        """The observation ids the AST's free text matches, in its scope.

        The ONE seam between ``scope:`` and the observations page. Both
        :meth:`query_observations` and :meth:`count_observations` go through
        it, because a page and a total computed under different scopes is the
        "plausible but wrong" answer this module refuses everywhere else.
        """
        text = ast.text or ""
        if ast.scope != SCOPE_SESSION:
            return self._fts_obs_ids(text)
        roots, _exacts = self._session_hit_scope(
            text, ast.obs_type, ast.provenance
        )
        return self._session_scope_obs_ids(text, roots)

    # -- capabilities -----------------------------------------------------

    def capabilities(self) -> dict:
        """Read-only summary used by ``aggregator_capabilities``.

        Reports records-shaped sources (github) and sessions/subagent counts
        under a single ``sources`` list, plus session-side inventory (agent
        types seen, activity range).

        v3: the ``sessions``/``subagents`` buckets count claude-code rows
        only (matching the query-filter semantics); chat-export origins
        (chatgpt, claude-web) surface as their own source + count +
        freshness entries when nonzero. Zero-row origins are omitted so the
        pre-v3 response shape is unchanged until chat data actually lands.
        """
        c = self._c()
        sources: list[str] = [
            r["source"] for r in c.execute("SELECT DISTINCT source FROM records")
        ]
        # v2: session presence is source "sessions"; subagent presence adds a
        # nominal "subagents" bucket for capability discovery.
        sess_count = c.execute(
            "SELECT COUNT(*) AS n FROM sessions "
            "WHERE kind='session' AND origin='claude-code'"
        ).fetchone()
        sub_count = c.execute(
            "SELECT COUNT(*) AS n FROM sessions "
            "WHERE kind='subagent' AND origin='claude-code'"
        ).fetchone()
        if sess_count and sess_count["n"] > 0:
            sources.insert(0, "sessions")
        if sub_count and sub_count["n"] > 0:
            sources.insert(1 if "sessions" in sources else 0, "subagents")
        # v3: chat-export origins, only when rows exist.
        origin_counts: dict[str, int] = {}
        for origin in CHAT_ORIGINS:
            row = c.execute(
                "SELECT COUNT(*) AS n FROM sessions WHERE origin = ?", (origin,)
            ).fetchone()
            if row and row["n"] > 0:
                origin_counts[origin] = int(row["n"])
                sources.append(origin)

        freshness: dict[str, str | None] = {}
        tags_by_source: dict[str, list[str]] = {}
        session_shaped = {"sessions", "subagents", *CHAT_ORIGINS}
        for s in [x for x in sources if x not in session_shaped]:
            row = c.execute(
                "SELECT MAX(updated_at) AS m FROM records WHERE source = ?", (s,)
            ).fetchone()
            freshness[s] = row["m"] if row else None
            tag_counter: dict[str, int] = {}
            for row in c.execute(
                "SELECT tags FROM records WHERE source = ?", (s,)
            ):
                for t in json.loads(row["tags"]):
                    tag_counter[t] = tag_counter.get(t, 0) + 1
            tags_by_source[s] = [
                t for t, _ in sorted(
                    tag_counter.items(), key=lambda kv: kv[1], reverse=True
                )
            ][:20]

        if "sessions" in sources:
            row = c.execute(
                "SELECT MAX(last_ts) AS m FROM sessions "
                "WHERE kind='session' AND origin='claude-code'"
            ).fetchone()
            freshness["sessions"] = row["m"] if row else None
            tags_by_source["sessions"] = []
        if "subagents" in sources:
            row = c.execute(
                "SELECT MAX(last_ts) AS m FROM sessions "
                "WHERE kind='subagent' AND origin='claude-code'"
            ).fetchone()
            freshness["subagents"] = row["m"] if row else None
            agents = [r["at"] for r in c.execute(
                "SELECT DISTINCT agent_type AS at FROM sessions "
                "WHERE kind='subagent' AND agent_type IS NOT NULL LIMIT 20"
            )]
            tags_by_source["subagents"] = agents
        for origin in origin_counts:
            row = c.execute(
                "SELECT MAX(last_ts) AS m FROM sessions WHERE origin = ?",
                (origin,),
            ).fetchone()
            freshness[origin] = row["m"] if row else None
            tags_by_source[origin] = []

        # Date range across everything (records + sessions).
        row = c.execute(
            "SELECT MIN(created_at) AS lo, MAX(updated_at) AS hi FROM records"
        ).fetchone()
        lo, hi = (row["lo"], row["hi"]) if row else (None, None)
        s_row = c.execute(
            "SELECT MIN(first_ts) AS lo, MAX(last_ts) AS hi FROM sessions"
        ).fetchone()
        if s_row and s_row["lo"]:
            lo = min(lo, s_row["lo"]) if lo else s_row["lo"]
        if s_row and s_row["hi"]:
            hi = max(hi, s_row["hi"]) if hi else s_row["hi"]
        date_range: tuple[str, str] | None = None
        if lo and hi:
            date_range = (lo[:10], hi[:10])

        # Chunk 4: per-record-source counts (github: N, research: M, ...) so
        # records-shaped sources register under ``counts`` like the chat
        # origins do — not just in the aggregate ``records`` total.
        record_counts = {
            row["source"]: int(row["n"])
            for row in c.execute(
                "SELECT source, COUNT(*) AS n FROM records GROUP BY source"
            )
        }
        counts = {
            "sessions": int(sess_count["n"]) if sess_count else 0,
            "subagents": int(sub_count["n"]) if sub_count else 0,
            "observations": int(
                c.execute("SELECT COUNT(*) AS n FROM observations").fetchone()["n"]
            ),
            "records": int(
                c.execute("SELECT COUNT(*) AS n FROM records").fetchone()["n"]
            ),
            **record_counts,
            **origin_counts,
        }

        return {
            "sources": sources,
            "freshness": freshness,
            "tags_by_source": tags_by_source,
            "date_range": date_range,
            "cache_path": str(self.db_path),
            "schema_version": SCHEMA_VERSION,
            "counts": counts,
            "vector_index": self.vector_index_state(),
        }

    def vector_index_state(self) -> dict:
        """v5: how much of the corpus hybrid retrieval can actually reach.

        FOUR SITUATIONS THAT LOOK ALIKE AND ARE NOT, which is the whole
        reason this is a structured value rather than a row count:

        * **the vector arm is unavailable** — sqlite-vec did not load, or this
          cache has no vec tables. Hybrid is off and no amount of waiting
          fixes it; somebody has to repair the install.
        * **nothing embedded yet** — the arm works, the backlog is full, the
          worker has not run. Fix: run ``aggregator embed``.
        * **backfill in progress** — the arm works and is partway through.
          Fix: wait. Recall is already better than FTS5 alone and improving.
        * **backfill drained but incomplete** — nothing is pending and yet
          rows are missing, because the worker set them aside as ``'error'``.
          Waiting cannot fix this one, which is exactly why it may not be
          reported as ``complete``. Fix: ``aggregator status`` names the held
          rows (ledger sources ``embed:observations`` / ``embed:records``)
          and says whether each will retry or is terminal; the ones still due
          come back by themselves, the terminal ones never do.

        Reported as ``state`` ∈ ``unavailable`` | ``empty`` | ``not_started`` |
        ``backfilling`` | ``degraded`` | ``complete``, with the raw numbers
        alongside so a caller can render a percentage without re-deriving the
        verdict.

        ``vectors`` is ``None``, never ``0``, when the arm is unavailable:
        the count is genuinely unknown, and a 0 there is precisely the lie
        that would make "broken" read as "not started yet". The
        ``embedding_state`` tallies stay populated in every case — they are
        plain columns, and the backlog size is exactly what an operator
        needs when the arm is broken.
        """
        available = True
        reason: str | None = None
        per_kind: dict[str, dict] = {}
        for kind in ("observations", "records"):
            tally = dict(self.count_embedding_states(kind))
            vectors: int | None = None
            if available:
                try:
                    vectors = self.count_vec_rows(kind)
                except VectorIndexUnavailableError as e:
                    available = False
                    reason = str(e)
            tally["vectors"] = vectors
            per_kind[kind] = tally
        if not available:
            # The first kind may have answered before the second one failed;
            # an arm is available for both ontologies or neither.
            for tally in per_kind.values():
                tally["vectors"] = None

        total = sum(t["total"] for t in per_kind.values())
        pending = sum(t["pending"] for t in per_kind.values())
        errors = sum(t["error"] for t in per_kind.values())
        vectors_total = (
            None if not available else sum(t["vectors"] for t in per_kind.values())
        )
        if not available:
            state = "unavailable"
        elif total == 0:
            state = "empty"
        elif pending == 0 and errors == 0:
            state = "complete"
        elif pending == 0:
            # DRAINED, BUT NOT WHOLE. ``'error'`` rows leave ``pending``, so
            # without this branch a cache whose only unembedded rows are ones
            # the worker gave up on reports ``complete`` — the index rotting
            # while every count says it is fine, which is the exact failure
            # this project exists to prevent. ``degraded`` is the honest
            # verdict: the backfill has gone as far as it can and did not
            # reach everything. ``aggregator status`` names the rows.
            state = "degraded"
        elif vectors_total == 0:
            state = "not_started"
        else:
            state = "backfilling"
        return {
            "available": available,
            "reason": reason,
            "state": state,
            # Summed here so a caller can render "n unreachable" without
            # re-deriving it per ontology and getting the sum wrong.
            "errors": errors,
            **per_kind,
        }


def _row_to_record(row: sqlite3.Row) -> Record:
    return Record(
        stable_id=row["stable_id"],
        source=row["source"],
        subject=row["subject"],
        body=row["body"],
        tags=json.loads(row["tags"]),
        created_at=_parse_iso(row["created_at"]),
        updated_at=_parse_iso(row["updated_at"]),
        extra=json.loads(row["extra"] or "{}"),
    )


def _row_to_session(row: sqlite3.Row) -> SessionRow:
    return SessionRow(
        session_id=row["session_id"],
        root_session_id=row["root_session_id"],
        parent_session_id=row["parent_session_id"],
        kind=row["kind"],
        agent_id=row["agent_id"],
        agent_type=row["agent_type"],
        spawned_by_tool_use_id=row["spawned_by_tool_use_id"],
        cwd=row["cwd"],
        git_branch=row["git_branch"],
        first_ts=_parse_iso(row["first_ts"]),
        last_ts=_parse_iso(row["last_ts"]),
        jsonl_path=row["jsonl_path"],
        origin=row["origin"],
    )


def _row_to_observation(row: sqlite3.Row) -> ObservationRow:
    return ObservationRow(
        obs_id=row["obs_id"],
        session_id=row["session_id"],
        root_session_id=row["root_session_id"],
        parent_obs_id=row["parent_obs_id"],
        type=row["type"],
        ts=_parse_iso(row["ts"]),
        model=row["model"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        tool_name=row["tool_name"],
        tool_use_id=row["tool_use_id"],
        body=row["body"] or "",
        # v6. Carried so the MCP can put it on the row beside the snippet —
        # a caller seeing ``provenance: hook`` next to a hit knows why it is
        # there, which is the whole legibility win. NULL stays NULL: it means
        # "not classified yet", and coercing it to 'human' on the way out
        # would turn an unrun backfill into a corpus-wide authorship claim.
        provenance=row["provenance"],
    )


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None
