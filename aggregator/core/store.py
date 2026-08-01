"""SQLite + FTS5 store (spec §Storage, plan M2).

Schema is single-table (``records``) plus a mirrored FTS5 virtual table
(``records_fts``). Per-source specialisation lives in the ``extra`` JSON
blob rather than per-source tables — v1 keeps the schema flat so the DSL /
query path is source-agnostic. Individual sources can still specialise their
own retrieval on top of ``extra`` by reading the JSON back at the surface.

Stable-ID discipline (spec constraint 5): ``stable_id`` is the primary key.
Upserting the same ``stable_id`` overwrites the record; a fresh ``stable_id``
inserts a new row. Sources compute ``stable_id`` deterministically from the
external identity (see ``sources/base.py::stable_id_for``), so ``rebuild`` +
re-ingest yields the exact same IDs — round-trip safe.

Every ``upsert`` runs through ``aggregator.core.scrub.scrub`` first (spec
constraint 3, defense in depth: also runs pre-return at MCP/CLI boundary).

FTS5 tokenizer is ``unicode61 remove_diacritics 2`` per spec — Unicode-normal,
tolerant of accented characters, and the same tokenizer used by SQLite's
built-in FTS demos.

FTS5 syntax errors (e.g. unbalanced quotes) are caught and logged; the query
returns ``[]`` rather than raising. The MCP surface (M3) surfaces this as
``ok: false, reason, remediation`` per spec §DSL.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path

from aggregator.core.scrub import scrub
from aggregator.sources.base import QueryAST, Record

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_DDL: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS records (
        stable_id  TEXT PRIMARY KEY,
        source     TEXT NOT NULL,
        subject    TEXT NOT NULL,
        body       TEXT NOT NULL,
        tags       TEXT NOT NULL,       -- JSON array
        created_at TEXT,
        updated_at TEXT,
        extra      TEXT NOT NULL DEFAULT '{}'
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

    Not thread-safe — v1 assumes a single ingest process at a time
    (systemd user timers are serialised per unit). If we later fan out, wrap
    the connection in a lock or move to per-thread connections.
    """

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else _default_db_path()
        self._conn: sqlite3.Connection | None = None

    # -- connection lifecycle -------------------------------------------------

    def _c(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON;")
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- schema ---------------------------------------------------------------

    def migrate(self) -> None:
        """Create tables + FTS virtual table + meta row. Idempotent."""
        c = self._c()
        for stmt in _DDL:
            c.executescript(stmt)
        c.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        c.commit()

    # -- writes ---------------------------------------------------------------

    def upsert(self, records: list[Record]) -> None:
        """Write records to the store, scrubbing every field pre-write.

        Idempotent per ``stable_id``: re-upsert of the same ID overwrites the
        row (INSERT ... ON CONFLICT DO UPDATE); a fresh ID inserts a new row.
        """
        c = self._c()
        for r in records:
            scrubbed_body = scrub(r.body).text
            scrubbed_subject = scrub(r.subject).text
            c.execute(
                """
                INSERT INTO records(
                    stable_id, source, subject, body, tags,
                    created_at, updated_at, extra
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(stable_id) DO UPDATE SET
                    subject    = excluded.subject,
                    body       = excluded.body,
                    tags       = excluded.tags,
                    updated_at = excluded.updated_at,
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
                ),
            )
            # FTS mirror: delete-then-insert avoids stale rows when the
            # underlying record was updated.
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
        c.commit()

    def rebuild(self, source: str) -> None:
        """Drop all rows for one source; caller re-ingests.

        Stable-ID persistence is guaranteed because the source's stable_id
        function is deterministic on its external key (owner/repo:number,
        session_uuid, etc.), so re-ingesting yields the same IDs.
        """
        c = self._c()
        c.execute("DELETE FROM records WHERE source = ?", (source,))
        c.execute("DELETE FROM records_fts WHERE source = ?", (source,))
        c.commit()

    # -- reads ---------------------------------------------------------------

    def _build_where(self, ast: QueryAST) -> tuple[str, list]:
        """Shared WHERE builder for ``query`` and ``count`` (advisor HIGH-3:
        ``total`` must reflect the real match count independent of pagination,
        which requires a COUNT(*) with the same predicates)."""
        clauses = ["1=1"]
        params: list = []
        if ast.source:
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
            # Tags stored as JSON array; look for ``"tag"`` substring. Cheap +
            # good enough for v1; if perf becomes an issue we can promote tags
            # to a side table with an index.
            clauses.append("tags LIKE ?")
            params.append(f'%"{tag}"%')
        return " AND ".join(clauses), params

    def _fts_ids(self, text: str) -> set[str]:
        """Set of stable_ids matching an FTS5 MATCH query.

        Malformed queries raise ``sqlite3.OperationalError``; higher layers
        (MCP) surface that via ``Store.probe_fts`` and translate to a
        structured ``ok: false`` before reaching ``query``. This method
        keeps the exception mode so a stray malformed call is loud rather
        than silently returning ``[]`` — the M3 layer catches at the probe.
        """
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

        Filter order: source → date range → tag substring → FTS MATCH. The
        FTS pass is a set intersection with the pre-filtered ID set.

        Pre-HIGH-3 behaviour hardcoded ``LIMIT 500`` unconditionally, silently
        truncating the caller's view. Post-fix:

        * ``limit=None`` (default) returns every matching row.
        * ``limit=N, offset=M`` returns a page (thin passthrough to SQL).

        On FTS syntax errors we log a warning and return ``[]`` (M3 re-detects
        via ``probe_fts`` and re-surfaces as ``ok: false`` with remediation).
        """
        where, params = self._build_where(ast)
        c = self._c()
        sql = f"SELECT * FROM records WHERE {where} ORDER BY updated_at DESC"
        # SQLite requires LIMIT before OFFSET; use LIMIT -1 (all) when the
        # caller only asks for offset without a limit.
        if limit is not None or offset:
            sql += " LIMIT ? OFFSET ?"
            params = [*params, (-1 if limit is None else int(limit)), int(offset)]
        try:
            rows = list(c.execute(sql, params))
        except sqlite3.OperationalError as e:
            # Base query itself failing is unexpected; return [] rather than
            # bubble up. FTS syntax errors are handled below.
            log.warning("store.query base SELECT failed for ast=%r: %s", ast, e)
            return []
        if ast.text:
            try:
                fts_ids = self._fts_ids(ast.text)
            except sqlite3.OperationalError as e:
                # Malformed FTS5 syntax (unbalanced quote, dangling operator).
                # Surface layer (M3) turns this into ok:false + remediation.
                log.warning("FTS5 syntax error for query %r: %s", ast.text, e)
                return []
            allowed_ids = {row["stable_id"] for row in rows}
            keep = allowed_ids & fts_ids
            rows = [row for row in rows if row["stable_id"] in keep]
        return [_row_to_record(row) for row in rows]

    def count(self, ast: QueryAST) -> int:
        """Return the total number of records matching ``ast``.

        Used by MCP's ``aggregator_query`` to populate ``total`` accurately
        even when the caller is paginating (advisor HIGH-3: pre-fix, ``total``
        was capped at 500 because ``query`` was capped at 500).

        For text queries the count still requires intersection with the FTS
        result set, so we cannot avoid loading the FTS ID set — but we skip
        loading full rows.
        """
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

    def probe_fts(self, text: str) -> None:
        """Run a cheap MATCH probe to surface FTS5 syntax errors.

        Public replacement for MCP's private ``_probe_fts_syntax`` that used
        to reach into ``store._c()`` (advisor MEDIUM). Raises
        ``sqlite3.OperationalError`` on syntax errors; caller converts to a
        structured MCP error.
        """
        c = self._c()
        c.execute(
            "SELECT rowid FROM records_fts WHERE records_fts MATCH ? LIMIT 1",
            (text,),
        ).fetchone()

    def capabilities(self) -> dict:
        """Return a summary of what the store currently holds.

        Used by M3's ``aggregator_capabilities`` MCP tool + M5's CLI ``--help``
        to feed ``format_help`` with real cached inventory.
        """
        c = self._c()
        sources = [r["source"] for r in c.execute(
            "SELECT DISTINCT source FROM records"
        )]
        freshness: dict[str, str | None] = {}
        tags_by_source: dict[str, list[str]] = {}
        for s in sources:
            row = c.execute(
                "SELECT MAX(updated_at) AS m FROM records WHERE source = ?", (s,)
            ).fetchone()
            freshness[s] = row["m"] if row else None
            # Top-20 tag frequency from JSON blobs. Cheap for v1 volumes.
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
        row = c.execute(
            "SELECT MIN(created_at) AS lo, MAX(updated_at) AS hi FROM records"
        ).fetchone()
        date_range: tuple[str, str] | None = None
        if row and row["lo"] and row["hi"]:
            date_range = (row["lo"][:10], row["hi"][:10])
        return {
            "sources": sources,
            "freshness": freshness,
            "tags_by_source": tags_by_source,
            "date_range": date_range,
            "cache_path": str(self.db_path),
            "schema_version": SCHEMA_VERSION,
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


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None
