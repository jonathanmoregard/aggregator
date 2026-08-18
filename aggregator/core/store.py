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
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from aggregator.core.scrub import scrub
from aggregator.sources.base import (
    ObservationRow,
    QueryAST,
    Record,
    SessionEntity,
    SessionRow,
)

log = logging.getLogger(__name__)

SCHEMA_VERSION = 5


class VectorIndexUnavailableError(RuntimeError):
    """Raised when a vector read is attempted and sqlite-vec did not load.

    Its own type, and raised by name, so a caller can tell "the vector arm is
    not installed here" apart from "the query was wrong" — which a bare
    ``sqlite3.OperationalError: no such table: vec_observations`` cannot.
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
    FTS5 or hybrid hit list is routinely bigger than that: the corpus is
    483,193 observations and a common word matches tens of thousands of them.
    ``json_each`` takes the whole list as a single JSON parameter and stays
    index-driven; the plan keeps the id lookup and carries the scope as a LIST
    SUBQUERY. JSON1 has been compiled into SQLite by default since 3.38.

    ``column`` is interpolated and is never caller-supplied: every call site
    passes a literal.
    """
    return f"{column} IN (SELECT value FROM json_each(?))"


def _vec_table_present(c: sqlite3.Connection, table: str) -> bool:
    """Whether ``table`` exists. Probed once per write group, not per row."""
    return (
        c.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (table,)).fetchone()
        is not None
    )


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
    """
    c.execute(f"DELETE FROM {table} WHERE {key} = ?", (row_id,))  # noqa: S608 - fixed literals
    i = 0
    while True:
        cur = c.execute(
            f"DELETE FROM {table} WHERE {key} = ?",  # noqa: S608 - fixed literals
            (f"{row_id}:{i}",),
        )
        if cur.rowcount <= 0:
            return
        i += 1


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
    if not _vec_table_present(c, vec_table):
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
        embedding_state  TEXT
    );
    """,
    "CREATE INDEX IF NOT EXISTS obs_root_ts ON observations(root_session_id, ts);",
    "CREATE INDEX IF NOT EXISTS obs_session_ts ON observations(session_id, ts);",
    "CREATE INDEX IF NOT EXISTS obs_type ON observations(type);",
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
    """
    CREATE TRIGGER IF NOT EXISTS observations_au AFTER UPDATE ON observations
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

#: The ONE opt-in that lets a provenance mismatch DELETE computed vectors.
#:
#: Deliberately not inferred from anything. ``AGGREGATOR_EMBED_BACKEND`` names a
#: loader; it is not consent to discard a month of CPU, and round 2's S1 is
#: exactly that conflation — a stray ``export`` in one shell turning
#: ``aggregator query`` into a demolition, then the pinned timer demolishing
#: what got rebuilt. This variable does nothing at all when the stamp matches,
#: and its name says what it costs.
VECTOR_REINDEX_ENV = "AGGREGATOR_VECTOR_REINDEX"


def reindex_consented() -> bool:
    """Whether this process was explicitly told it may rebuild the index."""
    return os.environ.get(VECTOR_REINDEX_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
    )

_VEC_DDL: list[str] = [
    f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_observations USING vec0(
        obs_id     TEXT PRIMARY KEY,
        embedding  float[{_VEC_DIM}]
    );
    """,
    f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_records USING vec0(
        stable_id  TEXT PRIMARY KEY,
        embedding  float[{_VEC_DIM}]
    );
    """,
]


def vector_provenance() -> tuple[str, int]:
    """``(model id, dimension)`` the vectors in this cache are supposed to be.

    Imported from ``aggregator.core.embed`` LAZILY and deliberately. That
    module owns which model the worker loads, so reading the answer from
    anywhere else is how the stamp ends up naming a model nobody uses — but
    ``store`` is imported by ``aggregator.mcp`` on every editor cold start, so
    the import may not happen at module scope. Importing the module is cheap;
    ``sentence_transformers`` lives inside ``Embedder.__init__``.
    """
    from aggregator.core.embed import configured_model_id

    return configured_model_id(), _VEC_DIM

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
        f"be rebuilt from scratch (weeks of CPU) — say so once, explicitly: "
        f"`{VECTOR_REINDEX_ENV}=1 aggregator embed --catchup --source both`."
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

    def migrate(self) -> None:
        """Create tables + FTS virtual tables + triggers. Idempotent.

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
        for stmt in _DDL:
            c.executescript(stmt)
        if self._vector_available:
            # BEFORE the vec DDL: a foreign table of the wrong width survives
            # ``CREATE VIRTUAL TABLE IF NOT EXISTS`` untouched, so the check
            # that might drop it has to run first.
            self._reconcile_vector_provenance(c)
            if self._vector_quarantine is None:
                for stmt in _VEC_DDL:
                    c.executescript(stmt)
        c.execute(f"PRAGMA user_version = {SCHEMA_VERSION};")
        c.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        c.commit()

    def _reconcile_vector_provenance(self, c: sqlite3.Connection) -> None:
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
        via ``AGGREGATOR_VECTOR_REINDEX=1``, which is the only thing in this
        codebase that may delete computed vectors WHOLESALE.

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
        model, dim = vector_provenance()
        expected = json.dumps({"dim": dim, "model": model}, sort_keys=True)
        row = c.execute(
            "SELECT value FROM meta WHERE key = ?", (VECTOR_PROVENANCE_KEY,)
        ).fetchone()
        if row is not None and row[0] == expected:
            self._adopt_vectors()
            return

        stamped = "no stamp" if row is None else f"stamped {row[0]}"
        # O(n) on a vec0 table, and deliberately paid only here: the matching
        # stamp returns above, so this runs once, on the mismatch path.
        on_disk = 0
        for table in ("vec_observations", "vec_records"):
            exists = c.execute(
                "SELECT 1 FROM sqlite_master WHERE name = ?", (table,)
            ).fetchone()
            if exists:
                on_disk += c.execute(
                    f"SELECT COUNT(*) FROM {table}"  # noqa: S608 - fixed literals
                ).fetchone()[0]

        if on_disk and not reindex_consented():
            self._quarantine_vectors(
                _provenance_refusal(stamped, expected, on_disk)
            )
            log.error("%s", self._vector_quarantine)
            return

        for stmt in _VEC_DROP_ALL:
            c.execute(stmt)
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
        model, dim = vector_provenance()
        expected = json.dumps({"dim": dim, "model": model}, sort_keys=True)
        if row is not None and row[0] == expected:
            self._adopt_vectors()
            return None
        present = c.execute(
            "SELECT 1 FROM sqlite_master WHERE name IN "
            "('vec_observations', 'vec_records')"
        ).fetchone()
        if row is None and present is None:
            self._adopt_vectors()
            return None
        stamped = "no stamp" if row is None else f"stamped {row[0]}"
        self._quarantine_vectors(_provenance_refusal(stamped, expected, None))
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
        # Probed ONCE per group. A cache migrated without sqlite-vec has no vec
        # tables at all, and the watermark reset below must still happen there.
        vec_present = _vec_table_present(c, "vec_observations")

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
                        tool_name, tool_use_id, body, src_hash
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    def upsert_vec_observations(
        self, rows: list[tuple[str, np.ndarray]]
    ) -> int:
        """Upsert ``(obs_id, embedding)`` rows. Returns how many were WRITTEN.

        Delete-then-insert to keep the upsert idempotent: vec0 virtual tables
        do not support ``ON CONFLICT``.

        The return value is what makes the degrade path checkable from outside:
        0 means the vectors were discarded because the extension is missing,
        which is otherwise indistinguishable from success.
        """
        self._ensure_writable()
        if not self._vector_writes_enabled():
            return 0
        c = self._c()
        for obs_id, embedding in rows:
            c.execute("DELETE FROM vec_observations WHERE obs_id = ?", (obs_id,))
            c.execute(
                "INSERT INTO vec_observations(obs_id, embedding) VALUES (?, ?)",
                (obs_id, embedding.astype("float32").tobytes()),
            )
        c.commit()
        return len(rows)

    def upsert_vec_records(self, rows: list[tuple[str, np.ndarray]]) -> int:
        """Upsert ``(stable_id, embedding)`` rows. Returns how many were written."""
        self._ensure_writable()
        if not self._vector_writes_enabled():
            return 0
        c = self._c()
        for stable_id, embedding in rows:
            c.execute("DELETE FROM vec_records WHERE stable_id = ?", (stable_id,))
            c.execute(
                "INSERT INTO vec_records(stable_id, embedding) VALUES (?, ?)",
                (stable_id, embedding.astype("float32").tobytes()),
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

    def select_unembedded(self, kind: str, limit: int = 500) -> list[sqlite3.Row]:
        """Rows whose ``embedding_state IS NULL`` — the embed worker's backlog.

        Newest first: a fresh observation is the one most likely to be searched
        for, so a partially-embedded corpus is useful long before it is
        complete.
        """
        c = self._c()
        if kind == "observations":
            return list(
                c.execute(
                    "SELECT obs_id, body FROM observations "
                    "WHERE embedding_state IS NULL "
                    "ORDER BY ts DESC LIMIT ?",
                    (limit,),
                )
            )
        if kind == "records":
            return list(
                c.execute(
                    "SELECT stable_id, subject, body FROM records "
                    "WHERE embedding_state IS NULL "
                    "ORDER BY updated_at DESC LIMIT ?",
                    (limit,),
                )
            )
        raise ValueError(f"unknown kind: {kind!r}")

    def mark_embedded(self, kind: str, ids: list[str], state: str) -> None:
        """Batch-advance ``embedding_state`` for ``ids``.

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
            return
        if state == "ok":
            self._require_vector()
        c = self._c()
        for page in _chunked(ids, 500):
            placeholders = ",".join("?" * len(page))
            c.execute(
                f"UPDATE {table} SET embedding_state = ? "  # noqa: S608 - allowlisted literals
                f"WHERE {col} IN ({placeholders})",
                (state, *page),
            )
        c.commit()

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

    def claim_embed_row(self, kind: str, row_id: str) -> None:
        """Record — durably — which row is about to be embedded."""
        self._ensure_writable()
        c = self._c()
        c.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            (
                EMBED_CLAIM_KEY,
                json.dumps({"kind": kind, "row_id": row_id}, sort_keys=True),
            ),
        )
        # COMMIT IS THE ENTIRE POINT. An uncommitted claim dies with the
        # process that wrote it, which is precisely the process this is meant
        # to outlive.
        c.commit()

    def release_embed_claim(self) -> None:
        """The claimed row resolved. Clear the claim."""
        self._ensure_writable()
        c = self._c()
        c.execute("DELETE FROM meta WHERE key = ?", (EMBED_CLAIM_KEY,))
        c.commit()

    def pending_embed_claim(self) -> tuple[str, str] | None:
        """``(kind, row_id)`` a previous process died on, or ``None``."""
        row = self._c().execute(
            "SELECT value FROM meta WHERE key = ?", (EMBED_CLAIM_KEY,)
        ).fetchone()
        if row is None:
            return None
        try:
            claim = json.loads(row[0])
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

        ``embedding_state`` is a plain column with an index on it, so the same
        question costs microseconds. It is also the more honest predicate:
        the worker writes vectors BEFORE it marks a row ``'ok'``, so ``'ok'``
        implies a retrievable vector, while ``'skip'`` (nothing embeddable)
        and ``'error'`` correctly do not.

        Raises when the extension is missing, like every other vector read —
        routing has to be able to tell "nothing embedded yet" from "this
        machine cannot run the vector arm at all".
        """
        if kind not in self._VEC_TABLES:
            raise ValueError(f"unknown kind: {kind!r}")
        self._require_vector()
        c = self._c()
        row = c.execute(
            f"SELECT 1 FROM {kind} WHERE embedding_state = 'ok' LIMIT 1"  # noqa: S608 - allowlisted literals
        ).fetchone()
        return row is not None

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
            key = "pending" if row["s"] is None else str(row["s"])
            counts[key] = counts.get(key, 0) + n
        return counts

    def _vec_obs_ids(self, query_embedding: np.ndarray, k: int) -> list[str]:
        """Vector KNN over ``vec_observations``.

        Returns top-K ``obs_id`` ordered by ascending distance — best match
        first, which is the order RRF expects.
        """
        self._require_vector()
        c = self._c()
        rows = c.execute(
            """
            SELECT obs_id
            FROM vec_observations
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
            """,
            (query_embedding.astype("float32").tobytes(), k),
        ).fetchall()
        return [r["obs_id"] for r in rows]

    def _vec_record_ids(self, query_embedding: np.ndarray, k: int) -> list[str]:
        """Vector KNN over ``vec_records``. Same contract as ``_vec_obs_ids``."""
        self._require_vector()
        c = self._c()
        rows = c.execute(
            """
            SELECT stable_id
            FROM vec_records
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
            """,
            (query_embedding.astype("float32").tobytes(), k),
        ).fetchall()
        return [r["stable_id"] for r in rows]

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
            Store._write_one_record(c, r, digest, r.stable_id in stored)
        return unchanged

    @staticmethod
    def _write_one_record(
        c: sqlite3.Connection, r: Record, digest: str, existed: bool = False
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
        if existed and _vec_table_present(c, "vec_records"):
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

    def _fts_ids(self, text: str) -> set[str]:
        c = self._c()
        rows = c.execute(
            "SELECT stable_id FROM records_fts WHERE records_fts MATCH ?",
            (text,),
        ).fetchall()
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
        """Run cheap MATCH probes to surface FTS5 syntax errors.

        Checks both records_fts and obs_fts so a syntactically bad query is
        caught regardless of which index the actual query would hit.
        """
        c = self._c()
        c.execute(
            "SELECT rowid FROM records_fts WHERE records_fts MATCH ? LIMIT 1",
            (text,),
        ).fetchone()
        c.execute(
            "SELECT rowid FROM obs_fts WHERE obs_fts MATCH ? LIMIT 1",
            (text,),
        ).fetchone()

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
                obs_ids = self._fts_obs_ids(ast.text)
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
                obs_ids = self._fts_obs_ids(ast.text)
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
        c = self._c()
        rows = c.execute(
            """
            SELECT DISTINCT o.root_session_id AS root
            FROM obs_fts f
            JOIN observations o ON o.rowid = f.rowid
            WHERE obs_fts MATCH ?
            """,
            (text,),
        ).fetchall()
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
        self, text: str, obs_type: str | None = None
    ) -> tuple[set[str], set[str]]:
        """FTS text (+ optional obs type) → ``(root_ids, exact_ids)``.

        ``root_ids``  — distinct ``root_session_id`` of matching obs; used
        to surface top-level session cards (a hit anywhere under the root
        counts — ``session:`` aggregates subagents).
        ``exact_ids`` — distinct ``session_id`` (composite for subagents);
        used so a subagent card surfaces only when its OWN stream matches.

        Live-model smoke MEDIUM (2026-08-02): the previous root-only
        mapping ignored ``type:`` and surfaced sibling subagents with
        zero own matches.
        """
        c = self._c()
        sql = """
            SELECT DISTINCT o.root_session_id AS root, o.session_id AS sid
            FROM obs_fts f
            JOIN observations o ON o.rowid = f.rowid
            WHERE obs_fts MATCH ?
        """
        params: list = [text]
        if obs_type:
            sql += " AND o.type = ?"
            params.append(obs_type)
        rows = c.execute(sql, params).fetchall()
        roots = {r["root"] for r in rows if r["root"]}
        exacts = {r["sid"] for r in rows if r["sid"]}
        return roots, exacts

    def _obs_id_hit_scope(
        self, obs_ids: Iterable[str], obs_type: str | None = None
    ) -> tuple[set[str], set[str]]:
        """``(root_ids, exact_ids)`` for a set of obs ids — the v5 twin of
        :meth:`_fts_hit_scope`, and it must stay behaviourally identical to it.

        The hybrid retriever fuses obs ids, but the hit list this feeds is
        session CARDS, so the ids have to be projected up the same way FTS
        hits are: a hit anywhere under a root surfaces the top-level card,
        while a subagent card surfaces only on a hit in its own stream.
        Diverging here would make hybrid and FTS5 answer the same question
        with differently-shaped hit lists.

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
            return self._fts_hit_scope(ast.text or "", ast.obs_type)
        roots, exacts = self._obs_id_hit_scope(ast.id_scope, ast.obs_type)
        if ast.text:
            fts_roots, fts_exacts = self._fts_hit_scope(ast.text, ast.obs_type)
            roots &= fts_roots
            exacts &= fts_exacts
        return roots, exacts

    def _fts_obs_ids(self, text: str) -> list[str]:
        c = self._c()
        rows = c.execute(
            """
            SELECT o.obs_id AS obs_id
            FROM obs_fts f
            JOIN observations o ON o.rowid = f.rowid
            WHERE obs_fts MATCH ?
            """,
            (text,),
        ).fetchall()
        return [r["obs_id"] for r in rows]

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
    )


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None
