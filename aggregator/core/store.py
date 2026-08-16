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

from aggregator.core.scrub import scrub
from aggregator.sources.base import (
    ObservationRow,
    QueryAST,
    Record,
    SessionEntity,
    SessionRow,
)

log = logging.getLogger(__name__)

SCHEMA_VERSION = 4

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
        src_hash         TEXT          -- v4; see ``_src_hash``
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
        src_hash   TEXT              -- v4; see ``_src_hash``
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

    # -- connection lifecycle ---------------------------------------------

    def _c(self) -> sqlite3.Connection:
        if self._conn is None:
            if self.read_only:
                # Codex MCP runs under a filesystem sandbox that can read the
                # cache but cannot create SQLite WAL/SHM sidecars. immutable=1
                # keeps recall truly read-only; writable ingest paths use the
                # default Store() connection below.
                uri = f"file:{self.db_path}?mode=ro&immutable=1"
                self._conn = sqlite3.connect(uri, uri=True)
            else:
                self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON;")
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
        """
        self._ensure_writable()
        c = self._c()
        self._ensure_sessions_origin_column(c)
        self._ensure_src_hash_columns(c)
        for stmt in _DDL:
            c.executescript(stmt)
        c.execute(f"PRAGMA user_version = {SCHEMA_VERSION};")
        c.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        c.commit()

    @staticmethod
    def _ensure_sessions_origin_column(c: sqlite3.Connection) -> None:
        """v2 → v3 in-place upgrade: add ``sessions.origin`` when absent.

        Probes ``PRAGMA table_info(sessions)`` rather than ``user_version``
        so a half-applied state (column present, version stale) can't run
        the ALTER twice. No-op on fresh DBs (no sessions table yet — DDL
        creates it with the column) and on already-v3 DBs.
        """
        table_exists = c.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sessions'"
        ).fetchone()
        if not table_exists:
            return
        cols = {row[1] for row in c.execute("PRAGMA table_info(sessions)")}
        if "origin" not in cols:
            c.execute(
                "ALTER TABLE sessions ADD COLUMN origin "
                "TEXT NOT NULL DEFAULT 'claude-code'"
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
            exists = c.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            if not exists:
                continue
            cols = {row[1] for row in c.execute(f"PRAGMA table_info({table})")}
            if "src_hash" not in cols:
                c.execute(f"ALTER TABLE {table} ADD COLUMN src_hash TEXT")  # noqa: S608 - fixed literals

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
                        src_hash        = excluded.src_hash
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
            Store._write_one_record(c, r, digest)
        return unchanged

    @staticmethod
    def _write_one_record(
        c: sqlite3.Connection, r: Record, digest: str
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
        return " AND ".join(clauses), params

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
                root_ids, exact_ids = self._fts_hit_scope(
                    ast.text, ast.obs_type
                )
            except sqlite3.OperationalError as e:
                log.warning("query_sessions FTS5 syntax %r: %s", ast.text, e)
                return []
            if not root_ids and not exact_ids:
                return []
            where += " AND " + self._fts_scope_clause(root_ids, exact_ids)
            params = [*params, *sorted(root_ids), *sorted(exact_ids)]
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
                root_ids, exact_ids = self._fts_hit_scope(
                    ast.text, ast.obs_type
                )
            except sqlite3.OperationalError:
                return 0
            if not root_ids and not exact_ids:
                return 0
            where += " AND " + self._fts_scope_clause(root_ids, exact_ids)
            params = [*params, *sorted(root_ids), *sorted(exact_ids)]
        row = c.execute(
            f"SELECT COUNT(*) AS n FROM sessions WHERE {where}", params
        ).fetchone()
        n = int(row["n"]) if row else 0
        if (
            n == 0
            and ast.top_session_id
            and not ast.text
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
        """Return observation rows matching the AST, ordered by ``ts``."""
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
            placeholders = ",".join("?" * len(obs_ids))
            where += f" AND obs_id IN ({placeholders})"
            params = [*params, *obs_ids]
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
        where, params = self._obs_where(ast)
        c = self._c()
        if ast.text:
            try:
                obs_ids = self._fts_obs_ids(ast.text)
            except sqlite3.OperationalError:
                return 0
            if not obs_ids:
                return 0
            placeholders = ",".join("?" * len(obs_ids))
            where += f" AND obs_id IN ({placeholders})"
            params = [*params, *obs_ids]
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
    def _fts_scope_clause(root_ids: set[str], exact_ids: set[str]) -> str:
        """SQL fragment pairing kinds with their FTS hit scope.

        Top-level cards surface on any hit under their root; subagent
        cards only on hits in their own stream. Caller appends params in
        the same order: sorted(root_ids) then sorted(exact_ids). Empty
        sets render as ``1=0`` (SQL disallows ``IN ()``).
        """
        root_part = (
            "(kind = 'session' AND root_session_id IN ("
            + ",".join("?" * len(root_ids)) + "))"
            if root_ids
            else "1=0"
        )
        exact_part = (
            "(kind = 'subagent' AND session_id IN ("
            + ",".join("?" * len(exact_ids)) + "))"
            if exact_ids
            else "1=0"
        )
        return f"({root_part} OR {exact_part})"

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
