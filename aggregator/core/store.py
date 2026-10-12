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

Scrub-on-write (spec constraint 3; ``aggregator rescrub`` brings older rows up
to date, and results are returned as stored). WAL + busy_timeout=5000 (Codex Phase 2 MEDIUM #2 —
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

v5 added a vector index (sqlite-vec tables, ``embedding_state`` columns)
and v8 removed it again; search is FTS5 only. See
``Store._drop_vector_index``.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from aggregator.core.provenance import MACHINE, MACHINE_VALUES
from aggregator.core.scrub import scrub_many
from aggregator.sources.base import (
    SCOPE_SESSION,
    ObservationRow,
    QueryAST,
    Record,
    SessionEntity,
    SessionRow,
)

log = logging.getLogger(__name__)

# v8: the vector index is removed. The version stamp is bookkeeping only —
# every migration step is guarded by probing its artifact in ``sqlite_master``
# or ``PRAGMA table_info``, so a half-applied state converges rather than
# skips. See ``Store._drop_vector_index``.
SCHEMA_VERSION = 8

#: How long SQLite's own busy handler waits for the write lock before raising
#: ``database is locked``, in milliseconds.
#:
#: NAMED, NOT INLINED, because a test has to be able to shorten it. The
#: property worth testing is "a writer that loses the race waits instead of
#: dying", and at 30s the only test that could observe it would have to hold a
#: transaction for longer than that — half a minute of wall clock per
#: assertion. See ``tests/core/test_store_cross_process_write_lock.py``, which
#: turns it down to milliseconds and holds for two seconds.
#:
#: THIS NUMBER IS NOT WHAT MAKES CONCURRENT WRITES SAFE, and history here is
#: worth keeping: it was 5s, then bumped to 30s to buy margin, and the bump
#: bought a little and settled nothing. Measured on the live cache
#: 2026-08-31, a single ordinary ``aggregator ingest --all`` held the write
#: lock in contiguous blocks of 49s, 29s and 139s — so 30s was not short of
#: the mark by a margin a bigger number would close, it was short of it by
#: 4.6x on one transaction, on a corpus that only grows. What makes the
#: overlap safe is ``_CacheWriteLock`` below, which serialises the writers
#: before SQLite is ever asked. This timeout is now only a backstop against a
#: writer that predates the lock (a mid-deploy mix of builds) or one outside
#: this codebase entirely (a human at the ``sqlite3`` prompt).
_BUSY_TIMEOUT_MS = 30_000


class SchemaAheadError(RuntimeError):
    """This build found a cache written by a NEWER build, and stopped.

    THE INCIDENT. Two builds of the same program ran as different roles against
    one shared cache. The MCP reader ran out of the live checkout at
    ``SCHEMA_VERSION`` 6; the ``aggregator`` CLI and the ingest timer behind it
    were a Nix build pinned at 5. ``migrate()`` stamped
    ``PRAGMA user_version = SCHEMA_VERSION`` unconditionally, so every thirty
    minutes the v5 writer re-stamped the cache down to 5 and exited 0. The unit
    reported ``ExecMainStatus=0``, every source read fresh, and every recall was
    refused for weeks. Nothing noticed, because A SUCCESSFUL DOWNGRADE LOOKS
    EXACTLY LIKE A SUCCESSFUL MIGRATION — same code path, same exit code, same
    log line.

    WHY REFUSING IS THE ONLY SAFE MOVE, and not merely the tidy one. A cache
    stamped above this build's ``SCHEMA_VERSION`` was written by code this
    process has never seen. It does not know which columns the newer schema
    added, which of them are ``NOT NULL``, which triggers now fire on write,
    or whether the FTS shadows still mean what they used to. Every alternative to stopping is worse:

    * stamp anyway (the bug) — silently breaks every newer reader;
    * skip the stamp and keep writing — writes rows against a schema this
      build cannot describe, turning "the cache is unreadable by one component"
      into "the cache is wrong for all of them";
    * rebuild the cache at this version — that is a downgrade wearing a
      different hat, and it throws away rows only their sources can restore.

    So nothing is written at all, and the caller is told which two versions
    disagreed. THE FIX IS ALWAYS FORWARD: bring the lagging side — the build
    raising this — up to the cache. The cache is the leading side and stays
    exactly where it is.

    ITS OWN TYPE, and not a bare ``RuntimeError``, because the CLI has to
    catch precisely this and nothing else. An unhandled exception out of
    ``migrate()`` reaches ``aggregator-ingest.service`` as a traceback on every
    thirty-minute tick, and that unit's ``OnFailure=`` notifier has no
    debounce — forty-eight critical toasts a day about a condition only a
    deploy can change. See ``cli._report_schema_ahead`` for the alarm that
    replaces it, and the "deliberately no fourth code" paragraph in
    ``README.md`` for why alarm fatigue is treated here as a form of silence.

    CARRIES THE TWO NUMBERS AS ATTRIBUTES, not only inside the message. The
    message is written for a human reading a journal and is long on purpose;
    the desktop toast that the CLI raises from it has to stay short enough to
    be read at a glance, and re-deriving ``found`` by parsing the prose would
    be a parser of our own sentences — the kind of coupling that breaks the
    moment somebody improves the wording.
    """

    def __init__(self, message: str, *, found: int, expected: int) -> None:
        super().__init__(message)
        #: The cache's ``PRAGMA user_version`` — the LEADING side.
        self.found = found
        #: This build's ``SCHEMA_VERSION`` — the side that has to come up.
        self.expected = expected


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
#
# Exception: a change that only ADDS secret detection ships with
# ``aggregator rescrub`` instead of a bump — see core/rescrub.py for why.
SCRUB_FINGERPRINT = "presidio+gitleaks/v1"

# v3 chat-export origins. ``sessions.origin`` values are these plus the
# default 'claude-code'. Kept as a module constant so the WHERE builders,
# capabilities, and the MCP routing layer stay in lockstep.
CHAT_ORIGINS = ("chatgpt", "claude-web")

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


def _json_id_clause(column: str) -> str:
    """``<column> IN (…)`` that binds ANY NUMBER of ids as ONE parameter.

    The obvious ``IN (?,?,?…)`` breaks at ``SQLITE_MAX_VARIABLE_NUMBER`` —
    32,766 on this build — and every id set in this file that comes from an
    FTS5 hit list is routinely bigger than that: the corpus is
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

#: How many distinct ``scope:session`` intersections one read-only store keeps.
#: A single request asks one or two distinct questions of it — the page and its
#: per-card counts — so this is a guard against unbounded growth on a store
#: somebody holds open, not a working set to tune.
_SESSION_SCOPE_MEMO_MAX = 8


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


#: The two relaxed tiers, by name. These strings travel all the way to the
#: tool boundary (``lexical_relaxation`` in the MCP response, a ``#`` line in
#: CLI output), so they are part of the response vocabulary, not internals.
LEXICAL_RELAX_OR = "or"
LEXICAL_RELAX_PREFIX = "prefix"

#: Tier depth, for the sticky-deepest marker: ``None`` (exact) < or < prefix.
_LEXICAL_RELAX_DEPTH = {None: 0, LEXICAL_RELAX_OR: 1, LEXICAL_RELAX_PREFIX: 2}


@dataclass
class LexicalProbe:
    """What the relaxation ladder did inside one scope. Two facts, not one.

    ``tier`` alone could not answer the question the empty-page notice asks,
    and that is the whole reason this type exists. ``tier is None`` conflates
    "the STRICT tier answered" with "no tier matched anything at all" —
    indistinguishable states in the marker, opposite states in a diagnosis.
    When a ``tag:``/``source:``/``from:`` filter then removes every matching
    row, the page is empty for a reason that has nothing to do with the words,
    and the notice built on ``tier`` alone told the caller the corpus contains
    none of their terms. It does; the filter took them.

    ``matched`` is the missing half: how many rows the tier that answered
    returned, AS THE LADDER SAW THEM — that is, before the outer SELECT
    applied ``tag:``, ``source:``, ``from:``/``to:`` or the id-scope keys. It
    is a count the ladder already had in hand (``len(rows)``), so this costs
    no extra query.

    THREE-VALUED ON PURPOSE: ``None`` means the ladder never ran (a
    pure-filter query), ``0`` means it ran and matched nothing, and
    ``> 0`` means the words are in the corpus. A notice that collapsed the
    first two would state a fact about the corpus out of a query that never
    asked it.
    """

    #: Deepest relaxation tier that returned rows in this scope.
    tier: str | None = None
    #: Pre-filter rows the answering tier returned; ``None`` = ladder unused.
    matched: int | None = None


def fts5_relaxation_tiers(
    conjuncts: list[str],
) -> list[tuple[str | None, str]]:
    """The progressive-relaxation ladder: ``(tier, MATCH expression)`` pairs.

    THE FAILURE THIS REMOVES is the index's documented worst one: free text
    is an implicit AND inside one row, so a remembered-gist query of several
    words reliably returned nothing at all. The ladder keeps strict AND as
    tier 1 — a query that matches exactly returns identical rows to before —
    and adds two fallbacks a caller only ever sees when every earlier tier
    was EMPTY: the same conjuncts joined with ``OR``, then ``OR`` with a
    prefix ``*`` on the final conjunct (the token most likely to be
    half-remembered or half-typed).

    TAKES SANITIZED CONJUNCTS, NOT RAW TEXT, and that is deliberate twice
    over. First, the safety property: every character of user origin inside
    the returned expressions was emitted by :func:`fts5_match_conjuncts` —
    quoted ``\\w+`` runs — and the ``OR`` and trailing ``*`` are written
    HERE, by this function, so no operator can be smuggled in through the
    ladder. (``"a b"*`` is FTS5's phrase-prefix form: adjacency is kept and
    only the last token of the phrase prefix-matches — a caller's quoted
    phrase stays a phrase in every tier.) Second, the enumeration rule:
    ``tests/test_fts5_match_site_enumeration.py`` requires the function that
    OWNS a ``MATCH`` statement to call the sanitizer itself, so the binding
    sites call ``fts5_match_conjuncts`` and hand the result here rather than
    passing raw text through one more layer.

    THE OR TIER IS SKIPPED FOR A SINGLE CONJUNCT — it would be byte-identical
    to the strict tier, and re-running an FTS scan to learn nothing is how a
    "cheap" fallback stops being cheap.

    A NOTE ON PREFIX UNDER PORTER: the index holds STEMMED tokens, and FTS5
    does not stem a prefix query's token (stem-then-prefix is incoherent), so
    ``configurati*`` misses a corpus that only holds ``configur``. The prefix
    tier is therefore a genuine last resort for truncated tokens
    (``alphab*``), not a general recall widener — which matches its position
    at the bottom of the ladder.

    Callers run the tiers IN ORDER and stop at the first tier with rows:
    exactly one tier's rows are ever returned, so relaxed and exact results
    can never interleave, and the tier that answered is recorded via
    :meth:`Store._note_lexical_relaxation` for the output layer to surface.
    """
    if not conjuncts:
        return []
    tiers: list[tuple[str | None, str]] = [(None, " ".join(conjuncts))]
    if len(conjuncts) > 1:
        tiers.append((LEXICAL_RELAX_OR, " OR ".join(conjuncts)))
    prefixed = [*conjuncts[:-1], conjuncts[-1] + "*"]
    tiers.append((LEXICAL_RELAX_PREFIX, " OR ".join(prefixed)))
    return tiers


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

#: ONE tokenizer string for BOTH FTS tables, and the reason it is a constant:
#: the v7 migration probes the stored ``sqlite_master`` SQL against it, so a
#: second spelling in a second CREATE would make one table migrate and the
#: other silently keep the old tokenizer.
#:
#: ``porter`` IS THE v7 CHANGE. Without it, ``report`` and ``reports`` are
#: different terms, and a multi-word gist query — every term ANDed — loses to
#: morphology before the conjunction even gets its turn. Porter stems the
#: indexed tokens AND the query tokens with the same rules, so ``running``
#: finds ``run fast`` and ``run`` finds ``running shoes``. It wraps
#: ``unicode61 remove_diacritics 2``, which keeps the existing case- and
#: diacritic-folding exactly as it was.
_FTS_TOKENIZE = "porter unicode61 remove_diacritics 2"

#: The FTS table CREATEs and the obs sync triggers, as named constants rather
#: than inline ``_DDL`` strings, because the v7 migration has to recreate
#: exactly these artifacts (a tokenizer cannot be ALTERed — see
#: ``Store._ensure_porter_fts_tokenizer``) and a second copy of the text is
#: how the migrated shape and the fresh shape drift apart.

# FTS5 external-content over observations.body. Sync via the triggers below.
_OBS_FTS_DDL = f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS obs_fts USING fts5(
        body,
        content='observations',
        content_rowid='rowid',
        tokenize='{_FTS_TOKENIZE}'
    );
"""

_OBS_FTS_TRIGGERS: tuple[str, ...] = (
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
    # Without the column list this fires on ANY update to ANY column — a
    # ``provenance`` stamp, say — and each firing is a
    # full FTS5 ``'delete'`` plus re-insert of the entire body. With
    # ``auto_vacuum=0`` the pages that frees are never handed back, which is the
    # same mechanism documented at the ingest UPSERT below. Measured on a
    # throwaway database at 1/5 of live scale: a chunked column-only rewrite
    # took 87.5 s and grew the file 128 MB wide, versus 7.4 s and no growth
    # narrow.
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
)

_RECORDS_FTS_DDL = f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS records_fts USING fts5(
        stable_id UNINDEXED,
        source    UNINDEXED,
        subject,
        body,
        tags,
        tokenize='{_FTS_TOKENIZE}'
    );
"""

#: The text the ``records_fts`` ``tags`` column indexes for one row: the
#: space-joined UNION of the source-written ``tags`` array and the LLM-written
#: ``llm_tags`` array, read from the ``records`` row itself. One SQL
#: expression shared by every site that (re)builds a ``records_fts`` row —
#: the porter rebuild, the row-write path and ``write_llm_tags`` — because a
#: second spelling is how the index and the columns would come to disagree
#: about whether a record is reachable by its tags. ``COALESCE`` on
#: ``llm_tags``: NULL means "never tagged" and must read as no tags, not as a
#: ``json_each`` error.
_RECORDS_FTS_TAGS_SQL = (
    "TRIM(COALESCE((SELECT group_concat(je.value, ' ') "
    "FROM json_each(records.tags) AS je), '') || ' ' || "
    "COALESCE((SELECT group_concat(je.value, ' ') "
    "FROM json_each(COALESCE(records.llm_tags, '[]')) AS je), ''))"
)

#: Rebuild one record's ``records_fts`` row from the stored row. Callers
#: DELETE the old FTS row first (records_fts is not external-content, so
#: delete+insert is the update idiom — see ``_write_one_record``). Reading
#: from ``records`` rather than from caller-held values is what keeps the
#: index equal to the columns by construction: whatever the row holds — a
#: fresh ingest's body, a tag write's llm_tags — is what gets indexed.
_RECORDS_FTS_REFRESH_SQL = (
    "INSERT INTO records_fts(stable_id, source, subject, body, tags) "
    "SELECT stable_id, source, subject, body, " + _RECORDS_FTS_TAGS_SQL + " "
    "FROM records WHERE stable_id = ?"
)

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
    # CLASSIFIED and is the backfill's cursor — never ``'unknown'``, which
    # would collapse "we looked and could not tell" into "nobody looked".
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
    # FTS5 shadow + sync triggers — shared with the v7 rebuild migration,
    # which is why they are module constants rather than inline strings here.
    _OBS_FTS_DDL,
    *_OBS_FTS_TRIGGERS,
    # --- Legacy: records + records_fts for GitHub-shaped sources -------
    """
    CREATE TABLE IF NOT EXISTS records (
        stable_id  TEXT PRIMARY KEY,
        source     TEXT NOT NULL,
        subject    TEXT NOT NULL,
        body       TEXT NOT NULL,
        tags       TEXT NOT NULL,       -- JSON array; SOURCE-written, pristine
        created_at TEXT,
        updated_at TEXT,
        extra      TEXT NOT NULL DEFAULT '{}',
        src_hash   TEXT,             -- v4; see ``_src_hash``
        llm_tags   TEXT,             -- JSON array; see ``_ensure_llm_tags_columns``
        llm_tags_src_hash TEXT       -- src_hash the llm_tags were computed from
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_records_source ON records(source);",
    "CREATE INDEX IF NOT EXISTS idx_records_updated ON records(updated_at);",
    _RECORDS_FTS_DDL,
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


def _default_db_path() -> Path:
    """Resolve ``$XDG_DATA_HOME/aggregator/cache.db`` (creating parents)."""
    root = Path(
        os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    )
    p = root / "aggregator" / "cache.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


class _CacheWriteLock:
    """The cache's single writer slot, WAITED FOR rather than raced for.

    SQLite already has a single-writer rule; what it does not have is a way to
    queue for it. ``busy_timeout`` is a retry schedule with a deadline, and a
    deadline is the wrong instrument when the thing being waited on is another
    aggregator process doing legitimate, unbounded work. This is the queue:
    one ``flock`` per cache file, taken before SQLite is asked for anything,
    released when the transaction that took it ends.

    WHY AN OS LOCK AND NOT A BIGGER TIMEOUT. The two are not the same fix with
    different constants. A busy handler that never gives up still cannot make
    SQLite's lock acquisition fair — the loser re-tries on a backoff schedule
    and a writer committing back-to-back can keep winning — and it still holds
    the losing connection's read snapshot open while it spins. ``flock``
    blocks in the kernel, wakes on release, and costs nothing while waiting.
    The failure mode it replaces is not "waited and was unlucky", it is
    "waited 30s against a 139s transaction and was guaranteed to lose".

    WHY IT WRAPS THE TRANSACTION AND NOT THE RUN. Holding this for the life of
    a process would serialise every writer behind the longest-running one, which
    is the same starvation with the blame moved. Scoped to the transaction, a
    writer waits only for the transaction in front of it.

    HELD PER PROCESS, NOT PER STORE. ``flock`` is associated with the open
    file description, so two descriptors in ONE process are two independent
    locks and the second would block on the first forever. Hence the registry
    below: one descriptor per cache path per process, and a holder count so
    the nested writers this codebase actually has — ``rebuild_and_upsert_
    entities`` calling ``upsert_entities(_commit=False)``, ``write_checkpoint``
    composing a data write and a watermark advance into one transaction — take
    the baton once and hand it back once.

    A PROCESS THAT DIES HOLDING IT RELEASES IT. That is the kernel's doing,
    not ours: descriptors close on exit, however the exit happened, so a
    SIGKILL mid-transaction cannot wedge another writer, and it is why the
    baton is a file lock rather than a row in the database it is protecting.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fd: int | None = None
        #: How many Stores in THIS process are inside a write transaction.
        self._holders = 0

    def acquire(self) -> None:
        if self._holders == 0:
            if self._fd is None:
                try:
                    self._fd = os.open(
                        str(self.path), os.O_CREAT | os.O_RDWR, 0o600
                    )
                except OSError as e:
                    # Named, because the bare errno arrives as a puzzle at a
                    # call site that was trying to write a row. A cache
                    # directory this process cannot create a file in is a
                    # cache it could not have written to either, so this is
                    # the same refusal arriving one step earlier.
                    raise RuntimeError(
                        f"cannot take the cache write lock at {self.path}: {e}. "
                        f"Writing to the cache needs a lock file beside it, so "
                        f"the directory has to be writable by this user."
                    ) from e
            fcntl.flock(self._fd, fcntl.LOCK_EX)
        self._holders += 1

    def release(self) -> None:
        if self._holders == 0:
            return
        self._holders -= 1
        if self._holders == 0 and self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)


#: One ``_CacheWriteLock`` per cache path per process. See the class docstring
#: for why a second descriptor on the same file would deadlock against the
#: first rather than queue behind it.
_WRITE_LOCKS: dict[str, _CacheWriteLock] = {}
_WRITE_LOCKS_GUARD = threading.Lock()


def _write_lock_for(db_path: Path) -> _CacheWriteLock:
    """The baton for one cache file, keyed by its RESOLVED path.

    Resolved on both halves — the registry key and the lock file's own name —
    because two processes are only excluded from each other if they agree on
    which file to lock. ``~/.local/share/aggregator/cache.db`` reached through
    a symlinked home and reached directly are the same database, and a lock
    named after the spelling rather than the file would protect neither.
    """
    resolved = Path(db_path).resolve()
    key = str(resolved)
    with _WRITE_LOCKS_GUARD:
        lock = _WRITE_LOCKS.get(key)
        if lock is None:
            lock = _CacheWriteLock(Path(key + ".write.lock"))
            _WRITE_LOCKS[key] = lock
        return lock


class _WriteLockedConnection(sqlite3.Connection):
    """A connection that hands the write baton back the moment it can.

    THE RELEASE LIVES HERE BECAUSE THE TRANSACTION ENDS HERE. ``Store`` has 27
    places that finish a transaction and every one of them calls ``commit()``
    or ``rollback()`` on the connection — most directly as ``c.commit()``,
    which a ``Store.commit()`` override would never see. Overriding the
    connection catches all of them at once, so no future write method can
    forget to give the baton back, and no call site had to change to get it
    right. (``.cursor()`` appears nowhere in this module, so there is no
    second path to a transaction that would slip past this.)

    ``executescript`` commits without coming through here, which leaves the
    baton held slightly longer than SQLite's own lock — deliberately the safe
    direction. Holding a moment too long makes another process wait; letting
    go a moment too early is the bug this file exists to fix.
    """

    #: The baton for this connection's cache, or ``None`` on a read-only
    #: connection — which never writes, so it never queues for anything. That
    #: is what keeps WAL's real bargain intact: ``aggregator_search_memory``
    #: reads at full speed straight through a long ingest.
    cache_write_lock: _CacheWriteLock | None = None
    _holds_write_lock: bool = False

    def take_cache_write_lock(self) -> None:
        """Queue for the writer slot. Idempotent for the life of a transaction."""
        if self.cache_write_lock is None or self._holds_write_lock:
            return
        self.cache_write_lock.acquire()
        self._holds_write_lock = True

    def _drop_cache_write_lock(self) -> None:
        if not self._holds_write_lock:
            return
        # Cleared BEFORE the release, so a release that somehow raises cannot
        # leave the flag claiming we still hold a baton we have let go of.
        self._holds_write_lock = False
        self.cache_write_lock.release()

    def commit(self) -> None:
        try:
            super().commit()
        finally:
            self._drop_cache_write_lock()

    def rollback(self) -> None:
        try:
            super().rollback()
        finally:
            self._drop_cache_write_lock()

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._drop_cache_write_lock()


class Store:
    """Thin wrapper around a per-process SQLite connection.

    Not thread-safe within a process — v1 assumes a single ingest thread
    per process. Cross-process concurrency IS supported: the ingest timer,
    the tag timer, a hand-run CLI command and the MCP reader all open the
    same ``cache.db``, and their runs overlap.

    WHAT MAKES THAT SAFE IS ``_CacheWriteLock``, not the timeout. ``_c()``
    enables WAL so readers never block, and every writer queues for the OS
    lock in ``_ensure_writable`` before SQLite is asked for anything. The
    ``busy_timeout`` above it is a backstop for writers that are not part of
    this scheme, not the mechanism: a single ingest transaction has been
    measured holding the write lock for 139 seconds.
    """

    def __init__(self, db_path: str | Path | None = None, read_only: bool = False):
        self.db_path = Path(db_path) if db_path else _default_db_path()
        self.read_only = read_only
        self._conn: sqlite3.Connection | None = None
        #: ``scope:session`` answers computed on this store, keyed on
        #: ``(what, text, obs_type, provenance)``. Written only when
        #: ``read_only`` — see :meth:`_session_scope_cached` for why, and for
        #: the 200x it saves on a session-card page.
        self._session_scope_memo: dict[tuple, Any] = {}
        #: The lexical-ladder observation stack. Element 0 is the REQUEST's
        #: probe — what :meth:`lexical_relaxation` and :meth:`lexical_matches`
        #: report — and :meth:`lexical_probe` pushes narrower ones on top so a
        #: caller driving two ontologies from one query can ask what EACH arm
        #: did. Every recorder writes to every probe on the stack, so the
        #: request-wide answer is unaffected by who is watching.
        self._lexical_probes: list[LexicalProbe] = [LexicalProbe()]

    # -- connection lifecycle ---------------------------------------------

    def _c(self) -> sqlite3.Connection:
        if self._conn is None:
            if self.read_only:
                # ``mode=ro`` ONLY. Not ``immutable=1`` — that is a promise to
                # SQLite that the file cannot change, and this file changes
                # constantly: the ingest timer rewrites it every 30 minutes.
                # Under that promise SQLite skips all locking, ignores the
                # ``-wal`` outright, and holds its page cache across statements
                # forever. Measured consequences on this exact URI, not theory:
                #   * WAL-committed rows invisible;
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
                    self._conn = sqlite3.connect(
                        uri, uri=True, factory=_WriteLockedConnection
                    )
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
                self._conn = sqlite3.connect(
                    self.db_path, factory=_WriteLockedConnection
                )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON;")
            if self.read_only:
                return self._conn
            # THE BATON THIS CONNECTION WILL QUEUE FOR, attached here and only
            # for writable stores. A read-only connection leaves it ``None``
            # and never waits for anything — see ``_WriteLockedConnection``.
            self._conn.cache_write_lock = _write_lock_for(self.db_path)
            # Codex Phase 2 MEDIUM #2: concurrent-writer safety. See prior
            # revision docstring; WAL + busy_timeout + synchronous=NORMAL.
            self._conn.execute("PRAGMA journal_mode = WAL").fetchone()
            self._conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
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
        """The gate every write in this file passes through — now also the queue.

        THE BATON IS TAKEN HERE BECAUSE THIS IS ALREADY THE CHOKE POINT. All
        19 write methods call it first, and a method that skipped it would
        also be skipping the read-only guard, i.e. would already be a bug of a
        kind this file is built to make obvious. Putting the acquire here
        means no write path can reach SQLite without having queued, and no new
        write method has to remember to.

        BLOCKING, AND THAT IS THE POINT. The caller waits for however long the
        other writer legitimately needs — 139 seconds is a measured real value
        — instead of failing after a deadline that has nothing to do with the
        work in front of it.
        """
        if self.read_only:
            raise RuntimeError("read-only Store cannot write")
        self._c().take_cache_write_lock()

    # -- schema -----------------------------------------------------------

    def _refuse_if_schema_ahead(self, c: sqlite3.Connection) -> None:
        """Stop dead if this cache was written by a build newer than this one.

        ONE ``PRAGMA`` READ, AND IT IS FREE ON THE PATHS THAT MATTER.
        ``user_version`` lives in the header of page 1, which SQLite has
        already read to open the file at all, so the two overwhelmingly common
        cases — a cache at exactly this version, and a fresh database at 0 —
        pay for one integer comparison and take no new branch. That mattered
        enough to check: ``migrate()`` runs on EVERY subcommand, so anything
        expensive here would be a tax on the whole CLI.

        ``> SCHEMA_VERSION``, NOT ``!= SCHEMA_VERSION``. Below is the ordinary
        upgrade and is the entire reason this method exists; refusing in both
        directions would turn one incident into two, and would brick every
        machine mid-rollout. Above is the dangerous direction and the only one
        guarded here, because this build cannot know what the newer schema
        contains — see :class:`SchemaAheadError` for why every alternative to
        stopping is worse than stopping.

        THE MESSAGE IS PART OF THE FIX. An operator meeting a schema error
        fears a half-migrated cache, and reaching for a wholesale reset to
        repair damage that has not happened is how a cache gets deleted for
        nothing. So it says plainly that nothing was written,
        names both numbers, and names the one remedy that is not a downgrade
        in disguise: deploy a build that understands the cache. Nothing here
        offers to lower the cache to suit this process, because that is not an
        available answer — skew is resolved by bringing the lagging side up.
        """
        found = int(c.execute("PRAGMA user_version").fetchone()[0])
        if found <= SCHEMA_VERSION:
            return
        raise SchemaAheadError(
            f"refusing to migrate {self.db_path}: it is stamped at schema "
            f"version {found}, and this build only understands schema version "
            f"{SCHEMA_VERSION}. NOTHING WAS WRITTEN — the cache on disk is "
            f"exactly as it was, `PRAGMA user_version` included, and it is not "
            f"damaged. A cache newer than the process opening it means an OLD "
            f"BUILD is running against a CURRENT cache: this process cannot "
            f"know which columns schema {found} added, which of them are NOT "
            f"NULL, or which triggers now fire, so writing into it would take "
            f"a cache that one component merely cannot read and make it wrong "
            f"for all of them. Upgrade this build to one that understands "
            f"schema {found} — for a Nix deployment that means bumping the "
            f"pinned aggregator revision and re-running the switch, since the "
            f"installed store path is immutable and editing a source checkout "
            f"changes nothing about what runs on the timer. Until then this "
            f"writer stays stopped, on purpose: the cache is the side that is "
            f"current, and it keeps its version.",
            found=found,
            expected=SCHEMA_VERSION,
        )

    def migrate(self) -> None:
        """Create tables + FTS virtual tables + triggers. Idempotent.

        Bumps ``PRAGMA user_version`` to SCHEMA_VERSION — FORWARD ONLY. A cache
        already stamped above this build raises :class:`SchemaAheadError`
        before anything is written; see :meth:`_refuse_if_schema_ahead` for the
        argument, and note that the unconditional stamp this replaced is what
        made the 2026-08 skew incident invisible. A cache stamped BELOW is the
        ordinary upgrade and is migrated in place as it always was.

        v2 → v3 upgrades in place: ``ALTER TABLE sessions ADD COLUMN origin``
        with the ``'claude-code'`` default backfilling every existing row.
        NO rebuild — the live cache.db is hundreds of MB and SQLite ADD
        COLUMN with a constant default is a metadata-only change. The ALTER
        runs BEFORE the DDL pass because the v3 DDL also creates an index on
        ``origin``, which would fail against a v2-shaped table. Guarded by a
        ``PRAGMA table_info`` probe (not user_version) so it only fires when
        the column is genuinely absent; fresh DBs get the column via DDL.

        v7 → v8 removes the retired vector index; see
        :meth:`_drop_vector_index`. Nothing here touches ``ingest_state``,
        ``quarantine`` or ``poison_faults``.
        """
        self._ensure_writable()
        c = self._c()
        # FIRST, BEFORE A SINGLE DDL STATEMENT. Raising further down would
        # satisfy a test that only catches the exception while reproducing the
        # incident exactly: the damage is the write, not the return.
        self._refuse_if_schema_ahead(c)
        self._ensure_sessions_origin_column(c)
        self._ensure_src_hash_columns(c)
        self._ensure_provenance_column(c)
        self._ensure_llm_tags_columns(c)
        self._ensure_narrow_fts_update_trigger(c)
        self._ensure_porter_fts_tokenizer(c)
        for stmt in _DDL:
            c.executescript(stmt)
        self._drop_vector_index(c)
        c.execute(f"PRAGMA user_version = {SCHEMA_VERSION};")
        c.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        c.commit()

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
    def _ensure_provenance_column(c: sqlite3.Connection) -> None:
        """v5 → v6 in-place upgrade: add ``observations.provenance``.

        NULL FOR EVERY EXISTING ROW, and that is the contract. NULL means
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
    def _ensure_llm_tags_columns(c: sqlite3.Connection) -> None:
        """Additive (no version bump): ``records.llm_tags`` + its watermark.

        ``llm_tags`` holds a JSON array of LLM-generated topic tags;
        ``llm_tags_src_hash`` records the ``records.src_hash`` value those
        tags were computed FROM. NULL for every existing row, and — the same
        contract as ``provenance`` above — that NULL
        is the whole watermark: ``aggregator tag`` selects rows whose
        ``llm_tags_src_hash`` is NULL or disagrees with the current
        ``src_hash``, so a pre-existing corpus comes out of this migration
        QUEUED FOR TAGGING and a record whose content changes re-queues
        itself. No sidecar ledger, nothing to fall out of step.

        SOURCE TAGS STAY PRISTINE. The LLM never writes into ``records.tags``
        — the union of the two is computed at query/index time
        (:data:`_RECORDS_FTS_TAGS_SQL`, ``_build_where``) so a bad tagging
        run can always be re-done without having corrupted source data.

        Additive columns ship WITHOUT a schema-version bump, per the rule at
        the top of ``_DDL``: old readers ignore columns they do not know.
        Metadata-only ALTERs, like every ``_ensure_*`` above. Must run BEFORE
        ``_ensure_porter_fts_tokenizer``, whose records repopulate reads the
        ``llm_tags`` column through :data:`_RECORDS_FTS_TAGS_SQL`.
        """
        Store._ensure_column(c, "records", "llm_tags", "TEXT")
        Store._ensure_column(c, "records", "llm_tags_src_hash", "TEXT")

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

    @staticmethod
    def _fts_table_sql(c: sqlite3.Connection, table: str) -> str | None:
        """The stored CREATE text of ``table``, or None when absent."""
        row = c.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        return row[0] if row is not None else None

    @staticmethod
    def _ensure_porter_fts_tokenizer(c: sqlite3.Connection) -> None:
        """v6 → v7: rebuild both FTS5 tables with porter stemming.

        A TOKENIZER CANNOT BE ALTERED. It is baked into the FTS5 shadow tables
        at CREATE time, so the only path from ``unicode61`` to
        ``porter unicode61`` is drop + recreate + re-tokenize every row:
        ``obs_fts`` is external-content, so its rebuild is FTS5's own
        ``INSERT INTO obs_fts(obs_fts) VALUES('rebuild')`` over the
        observations table it shadows; ``records_fts`` is standalone, so it is
        repopulated from ``records`` — with ``tags`` space-joined out of the
        stored JSON array via ``json_each``, exactly the shape the row-write
        path stores (see the INSERT in ``_write_record_group``).

        PROBES THE ARTIFACT, NOT ``user_version`` — the same rule as every
        ``_ensure_*`` helper above, and here it is what makes an interrupted
        migration safe rather than silent: the whole swap for each table runs
        inside one SAVEPOINT, so a process killed mid-rebuild rolls back to
        the intact old-tokenizer table and the next ``migrate()`` (one runs on
        every CLI invocation) starts over. The probe can therefore never see
        ``porter`` in ``sqlite_master`` next to a partial index — which is the
        invariant that matters, because a migrated database that silently
        served an empty or half-built FTS index would look exactly like a
        corpus with nothing to say.

        THE COST IS PAID HERE, ON PURPOSE, AND ANNOUNCED OUT LOUD. On the live
        cache this re-tokenizes ~657k observation rows (~772 MB of text) and
        runs for minutes, on whichever process first executes the new build —
        possibly the MCP reader while other writers queue
        on the write lock. The alternative — a chunked background backfill —
        would mean a window in which queries silently answer from a partial
        index, the exact empty-looks-like-success failure this project bans by
        name. A migration that blocks once, says why, and leaves a complete
        index behind is the honest trade. The recommended controlled first
        run is one ``aggregator status`` immediately after deploy, so the
        rebuild happens in a terminal a human is watching rather than under a
        timer.

        Runs BEFORE the ``_DDL`` pass. On a fresh database neither table
        exists yet, so this no-ops and the DDL creates them with porter
        directly. The triggers are recreated inside the same savepoint — not
        left to the DDL pass — so "the savepoint committed" and "the index is
        complete AND kept in sync" are the same fact.
        """
        obs_sql = Store._fts_table_sql(c, "obs_fts")
        rec_sql = Store._fts_table_sql(c, "records_fts")
        obs_stale = obs_sql is not None and "porter" not in obs_sql
        rec_stale = rec_sql is not None and "porter" not in rec_sql
        if not obs_stale and not rec_stale:
            return
        obs_rows = (
            c.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
            if obs_stale
            else 0
        )
        rec_rows = (
            c.execute("SELECT COUNT(*) FROM records").fetchone()[0]
            if rec_stale
            else 0
        )
        log.warning(
            "Rebuilding the FTS5 index with porter stemming (schema v7): "
            "re-tokenizing %d observation row(s) and %d record row(s). This "
            "is a one-time migration — a tokenizer cannot be changed in "
            "place — and on the full cache it takes MINUTES while holding "
            "the write lock. Do not interrupt it; an interrupted run rolls "
            "back and starts over on the next command.",
            obs_rows,
            rec_rows,
        )
        c.execute("SAVEPOINT porter_fts")
        try:
            if obs_stale:
                for trig in (
                    "observations_ai",
                    "observations_ad",
                    "observations_au",
                ):
                    c.execute(f"DROP TRIGGER IF EXISTS {trig}")
                # Porter migration: FTS5 tokenizer cannot be ALTERed; this is
                # the deliberate drop+recreate+rebuild inside the SAVEPOINT.
                # arftl-allow: sql-drop
                c.execute("DROP TABLE obs_fts")
                c.execute(_OBS_FTS_DDL)
                c.execute("INSERT INTO obs_fts(obs_fts) VALUES('rebuild')")
                for stmt in _OBS_FTS_TRIGGERS:
                    c.execute(stmt)
            if rec_stale:
                # arftl-allow: sql-drop — same porter migration, records side.
                c.execute("DROP TABLE records_fts")
                c.execute(_RECORDS_FTS_DDL)
                c.execute(
                    "INSERT INTO records_fts"
                    "(stable_id, source, subject, body, tags) "
                    "SELECT stable_id, source, subject, body, "
                    + _RECORDS_FTS_TAGS_SQL
                    + " FROM records"
                )
        except BaseException:
            c.execute("ROLLBACK TO SAVEPOINT porter_fts")
            c.execute("RELEASE SAVEPOINT porter_fts")
            raise
        c.execute("RELEASE SAVEPOINT porter_fts")
        # The commit makes the minutes-long rebuild durable NOW, so a failure
        # later in migrate() cannot roll it back and pay the cost twice — but
        # ``_WriteLockedConnection.commit`` also hands the flock baton back,
        # and the REST of migrate() (DDL pass, vec reconcile, version stamp)
        # must not run unlocked on the first post-deploy invocation. Re-queue
        # immediately: ``take_cache_write_lock`` is idempotent, and if another
        # writer slipped in between commit and re-acquire it saw a consistent,
        # fully-rebuilt index. Guarded because the parameter type is the plain
        # ``sqlite3.Connection`` protocol; migrate() always passes the
        # write-locked subclass.
        c.commit()
        if isinstance(c, _WriteLockedConnection):
            c.take_cache_write_lock()
        log.info(
            "FTS5 porter rebuild complete: %d observation row(s), %d record "
            "row(s) re-tokenized.",
            obs_rows,
            rec_rows,
        )

    @staticmethod
    def _drop_vector_index(c: sqlite3.Connection) -> None:
        """v7 → v8: remove every artifact of the retired vector search.

        The artifacts are the ``vec_observations`` / ``vec_records`` virtual
        tables (and their shadow tables), ``chunk_embeddings``,
        ``embedding_versions``, the ``embedding_state`` column on
        ``observations`` and ``records`` with its two indexes, the
        ``vector_provenance`` / ``embed_inflight`` rows in ``meta``, and the
        embed worker's held rows in ``quarantine`` (sources
        ``embed:observations`` / ``embed:records``).

        PROBES THE ARTIFACTS, NOT ``user_version``, like every ``_ensure_*``
        helper: each step runs only when its artifact is present, so a fresh
        database and an already-migrated one are no-ops, and an interrupted run
        converges on the next ``migrate()``. Everything runs inside
        ``migrate()``'s one transaction.

        THE vec0 TABLES ARE DROPPED WITHOUT THE vec0 MODULE. ``DROP TABLE`` on
        a virtual table calls its module, so with sqlite-vec no longer loaded
        it raises ``no such module: vec0``. Instead the virtual table's own row
        is deleted from ``sqlite_master`` under ``PRAGMA writable_schema`` and
        its shadow tables — ordinary tables named ``<vtab>_<suffix>`` — are
        dropped normally. Those DROPs change the schema cookie, so every other
        open connection re-reads the schema.

        The file does not shrink: freed pages stay in the database until a
        ``VACUUM``, which is an operator's call and never run from here.
        """
        vtabs = [
            row[0]
            for row in c.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name IN ('vec_observations', 'vec_records')"
            )
        ]
        shadows = [
            row[0]
            for row in c.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND ("
                "name LIKE 'vec\\_observations\\_%' ESCAPE '\\' OR "
                "name LIKE 'vec\\_records\\_%' ESCAPE '\\')"
            )
        ]
        columns = [
            table
            for table in ("observations", "records")
            if "embedding_state"
            in {row[1] for row in c.execute(f"PRAGMA table_info({table})")}
        ]
        plain = [
            row[0]
            for row in c.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name IN ('chunk_embeddings', 'embedding_versions')"
            )
        ]
        if vtabs or shadows or columns or plain:
            log.warning(
                "Removing the retired vector index (schema v8): dropping %s. "
                "One-time migration; the freed space stays inside cache.db "
                "until a VACUUM.",
                ", ".join(
                    [*vtabs, *plain, *(f"{t}.embedding_state" for t in columns)]
                ),
            )
        if vtabs:
            c.execute("PRAGMA writable_schema = ON")
            try:
                c.execute(
                    "DELETE FROM sqlite_master WHERE type = 'table' AND name IN "
                    "('vec_observations', 'vec_records')"
                )
            finally:
                c.execute("PRAGMA writable_schema = OFF")
        for table in (*shadows, *plain):
            c.execute(f'DROP TABLE "{table}"')  # noqa: S608 - names read from sqlite_master
        c.execute("DROP INDEX IF EXISTS obs_embedding_state")
        c.execute("DROP INDEX IF EXISTS rec_embedding_state")
        for table in columns:
            c.execute(f"ALTER TABLE {table} DROP COLUMN embedding_state")
        c.execute(
            "DELETE FROM meta WHERE key IN ('vector_provenance', 'embed_inflight')"
        )
        c.execute(
            "DELETE FROM quarantine WHERE source IN "
            "('embed:observations', 'embed:records')"
        )

    def schema_version(self) -> int:
        """Return the DB's ``PRAGMA user_version`` (0 for a fresh file)."""
        c = self._c()
        row = c.execute("PRAGMA user_version").fetchone()
        return int(row[0]) if row else 0

    def rebuild_all(self) -> None:
        """Drop every table (records, sessions, observations, FTS shadows,
        meta) and re-run DDL. Migration escape hatch: SQLite is a derived
        index; JSONLs / API responses are source of truth. Callers detecting
        a stale ``user_version`` invoke this before re-ingesting.
        """
        self._ensure_writable()
        c = self._c()
        # BEFORE THE FIRST ``DROP``. This method drops every table and then
        # calls ``migrate()``, so guarding only ``migrate()`` would turn it into
        # "destroy the cache, then refuse to recreate it" — strictly worse than
        # the incident that guard exists for, which cost recall rather than the
        # data. A schema this build cannot read is never replaced by an older
        # one.
        self._refuse_if_schema_ahead(c)
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

        # DIGESTS FIRST, THEN ONE SCRUB FOR THE WHOLE GROUP. The secret scanner
        # is a ``secret-redact`` process whose start-up alone is ~100 ms, so
        # scrubbing row by row would put a process spawn on every changed row;
        # batched it is one spawn per group. ``scrub_many`` raises
        # ``ScrubUnavailableError`` when the scanner cannot run — nothing in the
        # group is written then (fail closed).
        obs_digest: dict[int, str] = {}
        to_scrub: list[int] = []
        for pos, e in enumerate(group):
            if not isinstance(e, ObservationRow):
                continue
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
            obs_digest[pos] = digest
            if stored_obs.get(e.obs_id) != digest and e.body:
                to_scrub.append(pos)
        scrubbed_by_pos = dict(
            zip(
                to_scrub,
                (r.text for r in scrub_many([group[p].body for p in to_scrub])),
                strict=True,
            )
        )

        unchanged = 0
        for pos, e in enumerate(group):
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
                # for every row, which re-runs the scrubber on each — ~11 hours
                # at this repo's measured 827 rows/min.
                #
                # The correct wiring is what is here: provenance is written on
                # INSERT and on the ``DO UPDATE`` branch (which is rewriting the
                # row anyway), and is otherwise owned by
                # ``aggregator provenance --backfill``, whose cursor is
                # ``provenance IS NULL``. (The digest is computed above, before
                # the group's one batched scrub.)
                digest = obs_digest[pos]
                if stored_obs.get(e.obs_id) == digest:
                    unchanged += 1
                    continue
                scrubbed_body = scrubbed_by_pos.get(pos, "")
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
                        provenance      = excluded.provenance
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

        unchanged = 0
        to_write: list[tuple[Record, str]] = []
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
            to_write.append((r, digest))
        # One batched scrub for every changed row's subject and body — see the
        # observations path for why (one scanner process per group, and fail
        # closed: a ScrubUnavailableError here writes nothing).
        scrubbed = scrub_many(
            [r.subject for r, _ in to_write] + [r.body for r, _ in to_write]
        )
        n = len(to_write)
        for i, (r, digest) in enumerate(to_write):
            Store._write_one_record(
                c, r, digest, scrubbed=(scrubbed[i].text, scrubbed[n + i].text)
            )
        return unchanged

    @staticmethod
    def _write_one_record(
        c: sqlite3.Connection,
        r: Record,
        digest: str,
        *,
        scrubbed: tuple[str, str],
    ) -> None:
        # ``scrubbed`` = (subject, body) already scrubbed by the caller's batch.
        scrubbed_subject, scrubbed_body = scrubbed
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
                extra      = excluded.extra
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
        c.execute("DELETE FROM records_fts WHERE stable_id = ?", (r.stable_id,))
        # INSERT..SELECT FROM the row just written, not VALUES from the
        # incoming Record. The difference is ``llm_tags``: the tagger's
        # columns survive an upsert (they are not in the SET list above), so
        # the FTS row has to carry them too or every ingest tick would strip
        # the LLM tags from the lexical index while the columns still claimed
        # the record tagged. The SELECT reads the scrubbed subject/body the
        # statement above stored, so the indexed text is identical either way.
        c.execute(_RECORDS_FTS_REFRESH_SQL, (r.stable_id,))

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
            # UNION of the source-written and LLM-written tag arrays: a
            # ``tag:`` filter must reach a record however it earned the tag.
            #
            # EXACT membership via json_each, not the old ``LIKE '%"tag"%'``
            # substring probe, which had two real defects: LIKE wildcards in
            # the tag value leaked through (``tag:a_b`` matched a record
            # tagged ``axb``), and source tags are stored ensure_ascii
            # ``json.dumps``-escaped, so a non-ASCII tag (``blåbär`` →
            # ``blåbär`` on disk) could never match the raw JSON
            # text. json_each decodes the escapes and ``=`` has no wildcard
            # vocabulary, so both go away without any ESCAPE bookkeeping.
            # COLLATE NOCASE keeps LIKE's ASCII case folding — callers copy
            # tag values from capabilities, but a hand-typed ``tag:bug`` must
            # still find the github label ``Bug`` like it always did.
            # ``COALESCE(llm_tags, '[]')``: NULL means never tagged and must
            # read as no tags — the same guard ``_RECORDS_FTS_TAGS_SQL`` uses.
            clauses.append(
                "(EXISTS (SELECT 1 FROM json_each(records.tags) "
                "WHERE json_each.value = ? COLLATE NOCASE) "
                "OR EXISTS (SELECT 1 FROM json_each(COALESCE(records.llm_tags, '[]')) "
                "WHERE json_each.value = ? COLLATE NOCASE))"
            )
            params.append(tag)
            params.append(tag)
        return " AND ".join(clauses), params

    @property
    def lexical_relaxation(self) -> str | None:
        """Which relaxation tier answered the lexical arm, if any.

        ``None`` means every lexical evaluation on this store since the last
        :meth:`reset_lexical_relaxation` was answered by the STRICT tier (or
        found nothing at any tier) — so ``None`` is a hard guarantee that no
        relaxed row is in the caller's hands. ``"or"`` / ``"prefix"`` name
        the DEEPEST tier that contributed rows to any evaluation in the
        request: a request can run the lexical arm more than once (the page
        and its count, the union's two ontologies), and "did anything on this
        page come from a relaxed tier" is the question the output layer has
        to answer — so the marker over-discloses across sub-evaluations
        rather than ever under-disclosing. The direction is deliberate: a
        relaxed match reported as exact is the lie this field exists to
        prevent; an exact match reported alongside a relaxed sibling is
        merely cautious.

        RESET AT THE REQUEST BOUNDARY, NOT PER CALL. ``aggregator_query``
        resets on entry; within one request every evaluation runs the same
        ladder over the same text and corpus, so the recorded tier is
        deterministic for the response it describes.

        NOT A DIAGNOSIS ON ITS OWN. ``None`` here is two different worlds —
        the strict tier answered, or nothing matched anywhere — and telling
        them apart is :meth:`lexical_matches`. See :class:`LexicalProbe`.
        """
        return self._lexical_probes[0].tier

    @property
    def lexical_matches(self) -> int | None:
        """How many rows the lexical ladder matched BEFORE the query's filters.

        ``None`` = the ladder never ran on this request. ``0`` = it ran and
        no tier matched. ``> 0`` = the terms are in the corpus, whatever the
        page ended up showing. See :class:`LexicalProbe` for why the empty-page
        notice cannot be honest without this.
        """
        return self._lexical_probes[0].matched

    @contextmanager
    def lexical_probe(self) -> Iterator[LexicalProbe]:
        """Watch what the ladder does inside this block, per arm.

        The union route runs the ladder once per ontology and the request-wide
        marker keeps the DEEPEST of the two — correct as an over-disclosure,
        wrong as an attribution, which is the bug this exists to fix: a page
        showing exact RECORD hits was stamped "NOT exact matches" because the
        observations arm, whose rows are not on the page, had relaxed.

        Re-entrant and exception-safe: the probe is removed by identity, so a
        nested block cannot pop somebody else's.
        """
        probe = LexicalProbe()
        self._lexical_probes.append(probe)
        try:
            yield probe
        finally:
            # By identity: ``list.remove`` compares by equality, and two
            # probes that saw the same thing are equal.
            for i in range(len(self._lexical_probes) - 1, 0, -1):
                if self._lexical_probes[i] is probe:
                    del self._lexical_probes[i]
                    break

    def reset_lexical_relaxation(self) -> None:
        """Start a fresh request: forget what the last one's ladder did.

        Drops any per-arm probes along with the request probe. Called at the
        request boundary, where no arm can still be open.
        """
        self._lexical_probes = [LexicalProbe()]

    def _note_lexical_relaxation(self, tier: str | None) -> None:
        """Record ``tier`` if it is deeper than what this probe has seen."""
        for probe in self._lexical_probes:
            if _LEXICAL_RELAX_DEPTH[tier] > _LEXICAL_RELAX_DEPTH[probe.tier]:
                probe.tier = tier

    def _note_lexical_matched(self, matched: int) -> None:
        """Record that the ladder RAN, and how many rows it saw pre-filter.

        ``max`` for the same reason the tier is sticky-deepest: one request can
        evaluate the arm several times (the page and its count, a session
        projection), and "did the words match anything at all" must survive a
        later evaluation that happened to see fewer rows.

        ONLY EVER CALLED BY A SITE THAT ACTUALLY RAN A STATEMENT. Recording a
        zero for a ladder that never ran collapses the three-valued contract
        this method exists to serve: ``None`` means nobody looked, and an arm
        that returns early on text with no word characters looked at nothing.
        Each arm below therefore checks its tier list (or its conjuncts) BEFORE
        the loop and returns without a note when there is nothing to run.
        """
        for probe in self._lexical_probes:
            probe.matched = max(probe.matched or 0, matched)

    def _fts_rows(self, sql: str, params: Sequence, text: str) -> list[sqlite3.Row]:
        """Run one MATCH-bearing SELECT, and be LOUD if it still fails.

        After :func:`fts5_match_query` nothing reaching ``MATCH`` can be a
        syntax error, so a failure here is a lock, a corrupt index, or an FTS5
        that has changed under us. The exception is re-raised UNCHANGED — the
        callers above each have a considered answer to it, and swallowing it
        here would turn "the index is broken" into "there are no matches",
        which is the empty-result-looks-like-success failure this project bans
        by name.
        """
        try:
            return self._c().execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            log.exception(
                "FTS5 MATCH failed for %r (rewritten to %r)",
                text,
                params[0] if params else "",
            )
            raise

    def _fts_ids(self, text: str) -> set[str]:
        """Records-path FTS5 arm. Empty set when the text has no word chars.

        RUNS THE RELAXATION LADDER — strict AND first, and only when a tier
        matches NOTHING the next one: OR, then OR with a prefix on the final
        conjunct. The tier that answered is recorded on the store for the
        output layer to surface; see :func:`fts5_relaxation_tiers` and
        :meth:`lexical_relaxation`. A main user-facing arm, which is why it
        relaxes — the narrow internal helpers (``probe_fts``, the
        ``scope:session`` conjunct intersection) deliberately do not.
        """
        tiers = fts5_relaxation_tiers(fts5_match_conjuncts(text))
        if not tiers:
            # No word characters: no tier ran, so there is nothing to record.
            return set()
        for tier, match_expr in tiers:
            rows = self._fts_rows(
                "SELECT stable_id FROM records_fts WHERE records_fts MATCH ?",
                (match_expr,),
                text,
            )
            if rows:
                self._note_lexical_relaxation(tier)
                self._note_lexical_matched(len(rows))
                return {row["stable_id"] for row in rows}
        self._note_lexical_matched(0)
        return set()

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

    # -- LLM topic tags on records (``aggregator tag``) -------------------
    #
    # The watermark predicate, spelled once. A record NEEDS tagging when it
    # was never tagged (``llm_tags_src_hash IS NULL``) or when its content
    # moved on since the tags were computed (the hashes disagree). The second
    # clause requires ``src_hash IS NOT NULL``: a pre-v4 row with no hash at
    # all cannot "disagree" with anything, so once tagged (the write stores
    # ``''`` for the missing hash) it stays tagged until ingest stamps a real
    # hash — at which point ``'' != hash`` re-queues it, correctly, because
    # the tags describe text the hash does not.
    _LLM_TAGS_NEEDED = (
        "(llm_tags_src_hash IS NULL "
        "OR (src_hash IS NOT NULL AND llm_tags_src_hash != src_hash))"
    )

    def ids_needing_llm_tags(self, sources: Sequence[str]) -> list[str]:
        """Stable ids of records the tag backfill still owes, newest first.

        IDS, NOT ROWS, and that is the streaming story: the CLI snapshots
        this list once per run (a few thousand short strings) and then pulls
        each batch's full rows via :meth:`records_for_tagging`, so bodies are
        only ever held one batch at a time. Snapshotting also gives the run a
        poison-loop guarantee for free — a record whose tagging fails is
        simply not revisited until the NEXT run, rather than re-selected by
        the very predicate its failure left unchanged.

        Newest first because tags exist to make recent material findable; on
        an interrupted first backfill that is the half worth having.
        """
        placeholders = ",".join("?" * len(sources))
        return [
            row["stable_id"]
            for row in self._c().execute(
                f"SELECT stable_id FROM records "  # noqa: S608 - placeholders only
                f"WHERE source IN ({placeholders}) AND {self._LLM_TAGS_NEEDED} "
                f"ORDER BY updated_at DESC, stable_id",
                list(sources),
            )
        ]

    def records_for_tagging(self, ids: Sequence[str]) -> list[sqlite3.Row]:
        """The fields one prompt batch needs, for the given stable ids.

        ``src_hash`` is read HERE, before the LLM call, and the caller hands
        it back to :meth:`write_llm_tags` unchanged — never re-read at write
        time. If ingest rewrites the row while the model is thinking, the
        stored watermark then disagrees with the new ``src_hash`` and the
        record re-queues on the next run, which is the truthful outcome: the
        tags were computed from text the row no longer holds.
        """
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        return self._c().execute(
            f"SELECT stable_id, source, subject, body, src_hash "  # noqa: S608 - placeholders only
            f"FROM records WHERE stable_id IN ({placeholders})",
            list(ids),
        ).fetchall()

    def write_llm_tags(
        self, items: Sequence[tuple[str, Sequence[str], str | None]]
    ) -> int:
        """Write one committed batch of ``(stable_id, tags, src_hash)``.

        ONE COMMIT PER CALL — this is the tag backfill's checkpoint, so a
        kill at any moment costs at most one batch and the watermark (which
        lives in the same rows, same transaction) can never get ahead of the
        tags it describes.

        ``src_hash`` is the value read at SELECT time (see
        :meth:`records_for_tagging`); ``None`` — a pre-v4 row — is stored as
        ``''`` so the row does not re-queue forever, per the predicate note
        above. Touches NOTHING else: not ``tags`` (source-written, pristine),
        not ``src_hash`` — a tag write must never cost the corpus a
        re-scrub. The record's
        ``records_fts`` row is rebuilt (delete + insert, the table's update
        idiom) so the lexical index carries the union immediately.

        Returns how many rows were actually updated; an id that vanished
        between select and write (source rebuild) counts zero and its FTS
        refresh is skipped rather than resurrecting a deleted row's index
        entry.
        """
        if not items:
            return 0
        self._ensure_writable()
        c = self._c()
        written = 0
        for stable_id, tags, src_hash in items:
            cur = c.execute(
                "UPDATE records SET llm_tags = ?, llm_tags_src_hash = ? "
                "WHERE stable_id = ?",
                (json.dumps(list(tags)), src_hash or "", stable_id),
            )
            if cur.rowcount == 0:
                continue
            written += 1
            c.execute(
                "DELETE FROM records_fts WHERE stable_id = ?", (stable_id,)
            )
            c.execute(_RECORDS_FTS_REFRESH_SQL, (stable_id,))
        c.commit()
        return written

    def llm_tag_progress_by_source(self) -> list[dict]:
        """How far the tag backfill has got, PER SOURCE — the truth surface.

        A partially-tagged corpus must never look fully tagged, and one global percentage cannot
        say which sources a ``tag:`` filter can be trusted for. One row per
        source actually present in ``records`` (the population the tagger
        walks), each with ``total``, ``tagged``, ``pending`` and a ``state``
        of ``not_started`` /
        ``in_progress`` / ``complete``. No ``degraded`` here — a failed
        record keeps its NULL watermark and stays honestly in ``pending``;
        the failure itself is reported loudly by the run that hit it.

        Cheap by construction (one grouped count over a ~4.4k-row table), so
        it rides along on ``capabilities()``.
        """
        rows = self._c().execute(
            f"SELECT source, COUNT(*) AS total, "  # noqa: S608 - fixed predicate
            f"SUM(CASE WHEN {self._LLM_TAGS_NEEDED} THEN 0 ELSE 1 END) "
            f"AS tagged FROM records GROUP BY source ORDER BY source"
        ).fetchall()
        out: list[dict] = []
        for row in rows:
            total = int(row["total"])
            tagged = int(row["tagged"] or 0)
            if tagged == 0:
                state = "not_started"
            elif tagged < total:
                state = "in_progress"
            else:
                state = "complete"
            out.append(
                {
                    "source": row["source"],
                    "total": total,
                    "tagged": tagged,
                    "pending": total - tagged,
                    "state": state,
                }
            )
        return out

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
        if ast.tags:
            # Sessions carry no tags, so a ``tag:`` filter can match NO
            # session row — ever. Without this clause a ``tag:``-only query
            # (which union-routes) got its sessions arm back UNFILTERED:
            # every session in the store, drowning the actual tagged records.
            clauses.append("1=0")
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
        if ast.tags:
            # Observations carry no tags either — same invariant as
            # ``_sessions_where``: an item without tags cannot match a
            # ``tag:`` filter.
            clauses.append("1=0")
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
        if ast.text:
            try:
                root_ids, exact_ids = self._text_hit_scope(ast)
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
            and not ast.tags
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
        if ast.text:
            try:
                root_ids, exact_ids = self._text_hit_scope(ast)
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
            and not ast.tags
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

        RUNS THE RELAXATION LADDER, because this is the session-card face of
        the same user-facing search the two arms above serve — a gist query
        that observations rescue and session cards do not would answer the
        same question two ways. The filters ride inside each tier, so a tier
        "matched" only when it matched rows the query can actually see.
        """
        conjuncts = fts5_match_conjuncts(text)
        sql = """
            SELECT DISTINCT o.root_session_id AS root, o.session_id AS sid
            FROM obs_fts f
            JOIN observations o ON o.rowid = f.rowid
            WHERE obs_fts MATCH ?
        """
        extra_params: list = []
        if obs_type:
            sql += " AND o.type = ?"
            extra_params.append(obs_type)
        if provenance:
            clause, extra = self._provenance_clause(provenance, "o.provenance")
            sql += f" AND {clause}"
            extra_params.extend(extra)
        tiers = fts5_relaxation_tiers(conjuncts)
        if not tiers:
            # No word characters: no tier ran, so there is nothing to record.
            return set(), set()
        for tier, match_expr in tiers:
            rows = self._fts_rows(sql, [match_expr, *extra_params], text)
            if rows:
                self._note_lexical_relaxation(tier)
                self._note_lexical_matched(len(rows))
                roots = {r["root"] for r in rows if r["root"]}
                exacts = {r["sid"] for r in rows if r["sid"]}
                return roots, exacts
        self._note_lexical_matched(0)
        return set(), set()

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

        MEMOISED ON A READ-ONLY STORE, AND ONLY THERE — see
        :meth:`_session_scope_cached`. A session-card page calls this once per
        CARD with identical arguments, and ``page_size`` defaults to 200.
        """
        return self._session_scope_cached(
            ("roots", text, obs_type, provenance),
            lambda: self._compute_session_hit_scope(text, obs_type, provenance),
        )

    def _compute_session_hit_scope(
        self, text: str, obs_type: str | None, provenance: str | None
    ) -> tuple[set[str], set[str]]:
        """:meth:`_session_hit_scope` without the memo. Owns the ``MATCH``.

        RECORDS WHAT THE INTERSECTION FOUND, like every other lexical arm, and
        the omission was the ``scope:session`` half of the empty-page lie. This
        probe never relaxes — so it noted no tier — but it is still a lexical
        evaluation, and ``lexical_matches`` staying ``None`` through it left the
        notice with nothing to tell "the terms are not in any one session" from
        "a filter removed the sessions that hold them". It then said the first
        about queries where the second was true. The count is the SESSIONS the
        intersection qualified, measured BEFORE the outer SELECT applies the
        query's row filters — which is exactly the pre-filter reading
        :class:`LexicalProbe` documents.
        """
        conjuncts = fts5_match_conjuncts(text)
        if not conjuncts:
            # No word characters: nothing ran, so nothing is recorded.
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
                self._note_lexical_matched(0)
                return set(), set()
        roots, exacts = roots or set(), exacts or set()
        self._note_lexical_matched(len(roots))
        return roots, exacts

    def _session_scope_cached(self, key: tuple, compute: Callable[[], Any]) -> Any:
        """Memoise one ``scope:session`` answer, on a READ-ONLY store only.

        WHY IT EXISTS. A session-card page computes ``matching_observations``
        per card, so the widening is recomputed once per CARD from identical
        arguments, and ``page_size`` defaults to 200. Measured on the live
        550k-row cache: ``green pull request`` under ``scope:session`` costs
        1.76 s per call against 0.06 s for the unwidened path, so a default-size
        card page would have spent about six minutes recomputing one answer.
        The unwidened path pays the same shape of cost and is left alone —
        changing it is a separate question with its own risk, and the note in
        ``mcp._lexical_session_ids`` already says what the correct fix there is.

        WHY READ-ONLY ONLY. A cache is a promise that the answer has not
        changed, and a store that can write cannot make that promise about
        itself; invalidating from ``upsert_entities`` is machinery no writer
        needs, since no write path runs a ``scope:`` query. The MCP builds a
        fresh ``Store(read_only=True)`` per tool call, so the memo's lifetime is
        one response — exactly the span over which a page and its per-card
        counts have to agree anyway.

        Bounded, and cleared wholesale when full: one request asks one or two
        distinct questions of it, so an eviction policy would be more machinery
        than the thing it manages.
        """
        if not self.read_only:
            return compute()
        hit = self._session_scope_memo.get(key)
        if hit is not None:
            return hit
        value = compute()
        if len(self._session_scope_memo) >= _SESSION_SCOPE_MEMO_MAX:
            self._session_scope_memo.clear()
        self._session_scope_memo[key] = value
        return value

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

    def _fts_obs_ids(self, text: str) -> list[str]:
        """Observations-path FTS5 arm. Empty when the text has no word chars.

        RUNS THE RELAXATION LADDER — the same contract as ``_fts_ids``: one
        tier's rows only, deepest-used tier recorded on the store. This is
        the arm behind ``query_observations``/``count_observations`` and
        ``_scoped_obs_ids``, so every route relaxes identically.
        """
        tiers = fts5_relaxation_tiers(fts5_match_conjuncts(text))
        if not tiers:
            # No word characters: no tier ran, so there is nothing to record.
            return []
        for tier, match_expr in tiers:
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
            if rows:
                self._note_lexical_relaxation(tier)
                self._note_lexical_matched(len(rows))
                return [r["obs_id"] for r in rows]
        self._note_lexical_matched(0)
        return []

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

        THE ROWS IT SEES ARE WHAT ``lexical_matches`` MEANS on this path: this
        is the row set a ``scope:session`` drilldown page is built from, and the
        query's row filters (``from:``/``to:``, the id-scope keys) are applied
        by the SELECT above it, so the count recorded here is the pre-filter one
        the empty-page diagnosis needs. ``roots`` being empty is not a
        no-measurement: the intersection that produced it already recorded its
        own zero, and re-recording here would only repeat it.
        """
        conjuncts = fts5_match_conjuncts(text)
        if not roots or not conjuncts:
            return []
        scope_json = json.dumps(sorted(roots))
        ids: set[str] = set()
        for expr in conjuncts:
            rows = self._fts_rows(
                "SELECT o.obs_id AS obs_id FROM obs_fts f "  # noqa: S608 - placeholders only
                "JOIN observations o ON o.rowid = f.rowid "
                "WHERE obs_fts MATCH ? AND "
                + _json_id_clause("o.root_session_id"),
                (expr, scope_json),
                text,
            )
            ids.update(r["obs_id"] for r in rows)
        self._note_lexical_matched(len(ids))
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
        # Memoised for the same reason and over the same lifetime as the root
        # intersection it builds on: this is the expensive half — measured 1.67
        # of the 1.76 s — and a session-card page asks for it once per card.
        return self._session_scope_cached(
            ("obs", text, ast.obs_type, ast.provenance),
            lambda: self._session_scope_obs_ids(
                text,
                self._session_hit_scope(text, ast.obs_type, ast.provenance)[0],
            ),
        )

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
                "SELECT tags, llm_tags FROM records WHERE source = ?", (s,)
            ):
                # UNION of source and LLM tags, mirroring what ``tag:``
                # actually filters on — an llm-only tag value must show up in
                # the advertised inventory or callers can never discover it.
                # Per-record set, so a tag both arrays carry counts once.
                for t in {
                    *json.loads(row["tags"]),
                    *json.loads(row["llm_tags"] or "[]"),
                }:
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
            # Cheap (one grouped count over records), so it rides the connect
            # path: an agent deciding whether a ``tag:`` filter can be trusted
            # needs this BEFORE it searches.
            "llm_tag_coverage": self.llm_tag_progress_by_source(),
        }


def _row_to_record(row: sqlite3.Row) -> Record:
    # ``llm_tags`` guarded by presence: the column is ensured by migrate(),
    # but a few pure-SQL tests build rows without it, and a missing key must
    # read as "never tagged" rather than raise on the way out of a query.
    # ``.keys()`` is LOAD-BEARING (noqa SIM118): bare ``in`` on sqlite3.Row
    # iterates VALUES, so it would test whether some column holds the string
    # "llm_tags" — a different question that is almost always False.
    llm_tags_raw = row["llm_tags"] if "llm_tags" in row.keys() else None  # noqa: SIM118
    return Record(
        stable_id=row["stable_id"],
        source=row["source"],
        subject=row["subject"],
        body=row["body"],
        tags=json.loads(row["tags"]),
        created_at=_parse_iso(row["created_at"]),
        updated_at=_parse_iso(row["updated_at"]),
        extra=json.loads(row["extra"] or "{}"),
        llm_tags=json.loads(llm_tags_raw) if llm_tags_raw else [],
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
