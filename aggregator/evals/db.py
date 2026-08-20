"""Baselines, run history, and the zero-result log — in their own database.

SCHEMA SHAPE FROM ``shawnhack/exocortex``, as the research report §8 describes
it: ``retrieval_regression_baselines`` (golden query -> baseline result ids),
``retrieval_regression_runs`` (run history + drift metrics), ``search_misses``
(zero-result query log), driven by one CLI command.

A SEPARATE FILE FROM ``cache.db``, AND THE CONSTRUCTOR REFUSES THAT NAME. Two
reasons, both load-bearing. First, the live cache is 1.2 GB, WAL-hot, and has
an ingest timer writing to it every 30 minutes; adding eval bookkeeping tables
to it would be a schema migration on the artifact under measurement, applied
by the tool whose job is to notice when that artifact changes. Second, a
baseline has to survive things the cache does not — a re-ingest, a vector
rebuild, a model swap — and the whole point of a baseline is that it does not
move when the thing it measures moves.

NOTHING HERE SWALLOWS AN ERROR. A miss that cannot be logged starves the
golden set of exactly the queries worth freezing, and silence would make an
empty log indistinguishable from a healthy one.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS retrieval_regression_baselines (
    query_id    TEXT NOT NULL,
    mode        TEXT NOT NULL,
    rank        INTEGER NOT NULL,
    result_id   TEXT NOT NULL,
    query_text  TEXT NOT NULL,
    frozen_at   TEXT NOT NULL,
    PRIMARY KEY (query_id, mode, rank)
);

-- An abstaining query has no rows in the table above, which is
-- indistinguishable from "never frozen". This table records the freeze
-- itself, so a negative query's empty baseline is a fact rather than a gap.
CREATE TABLE IF NOT EXISTS retrieval_regression_frozen (
    query_id    TEXT NOT NULL,
    mode        TEXT NOT NULL,
    query_text  TEXT NOT NULL,
    frozen_at   TEXT NOT NULL,
    PRIMARY KEY (query_id, mode)
);

CREATE TABLE IF NOT EXISTS retrieval_regression_runs (
    run_id                TEXT PRIMARY KEY,
    mode                  TEXT NOT NULL,
    started_at            TEXT NOT NULL,
    label                 TEXT,
    queries               INTEGER NOT NULL,
    mean_drift            REAL NOT NULL,
    max_drift             REAL NOT NULL,
    drifted_queries       INTEGER NOT NULL,
    top1_changes          INTEGER NOT NULL,
    abstention_violations INTEGER NOT NULL,
    labelled_queries      INTEGER NOT NULL,
    -- NULL, never 0.0. "Nobody labelled this" and "scored zero" are opposite
    -- facts and must not share a representation.
    ndcg_at_10            REAL,
    recall_at_50          REAL,
    mrr_at_10             REAL,
    detail_json           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS search_misses (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text   TEXT NOT NULL,
    mode         TEXT,
    result_count INTEGER NOT NULL DEFAULT 0,
    observed_at  TEXT NOT NULL,
    note         TEXT
);
CREATE INDEX IF NOT EXISTS idx_search_misses_text ON search_misses(query_text);
"""


class EvalStoreError(RuntimeError):
    """Raised when the eval database cannot be opened or written."""


def default_eval_db_path() -> Path:
    """``$XDG_DATA_HOME/aggregator/retrieval_eval.db`` (creating parents)."""
    root = Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))
    path = root / "aggregator" / "retrieval_eval.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _iso(value: datetime | None) -> str:
    return (value or datetime.now(UTC)).isoformat()


class EvalStore:
    """The eval database. One connection, created on construction."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else default_eval_db_path()
        if self.db_path.name == "cache.db":
            raise EvalStoreError(
                f"refusing to use {self.db_path} as the eval database: cache.db is "
                "the live aggregator cache, and the harness must never migrate or "
                "write to the artifact it measures. Use retrieval_eval.db."
            )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- lifecycle ---------------------------------------------------------

    def _c(self) -> sqlite3.Connection:
        if self._conn is None:
            raise EvalStoreError(
                f"eval store at {self.db_path} is closed; reopen it before writing. "
                "Dropping the write instead would make a dead log look like an "
                "empty one."
            )
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- baselines ---------------------------------------------------------

    def freeze_baseline(
        self,
        results: Mapping[str, Sequence[str]],
        *,
        mode: str,
        query_texts: Mapping[str, str],
        frozen_at: datetime | None = None,
    ) -> int:
        """Replace this mode's baseline with ``{query_id: ranked result ids}``.

        REPLACE, not merge: a baseline is a snapshot of one moment, and half of
        yesterday mixed with half of today is a baseline of nothing.
        """
        c = self._c()
        stamp = _iso(frozen_at)
        with c:
            c.execute(
                "DELETE FROM retrieval_regression_baselines WHERE mode = ?", (mode,)
            )
            c.execute("DELETE FROM retrieval_regression_frozen WHERE mode = ?", (mode,))
            for query_id, ids in results.items():
                text = query_texts.get(query_id, "")
                c.execute(
                    "INSERT INTO retrieval_regression_frozen"
                    "(query_id, mode, query_text, frozen_at) VALUES (?, ?, ?, ?)",
                    (query_id, mode, text, stamp),
                )
                for rank, result_id in enumerate(ids, start=1):
                    c.execute(
                        "INSERT INTO retrieval_regression_baselines"
                        "(query_id, mode, rank, result_id, query_text, frozen_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (query_id, mode, rank, result_id, text, stamp),
                    )
        return len(results)

    def baseline(self, mode: str) -> dict[str, list[str]]:
        """This mode's frozen baseline, including the empty (abstaining) ones."""
        c = self._c()
        frozen = {
            row["query_id"]: []
            for row in c.execute(
                "SELECT query_id FROM retrieval_regression_frozen WHERE mode = ?",
                (mode,),
            )
        }
        for row in c.execute(
            "SELECT query_id, result_id FROM retrieval_regression_baselines "
            "WHERE mode = ? ORDER BY query_id, rank",
            (mode,),
        ):
            frozen.setdefault(row["query_id"], []).append(row["result_id"])
        return frozen

    def baseline_query_texts(self, mode: str) -> dict[str, str]:
        """The query text each baseline was frozen against.

        Kept so a golden query edited after freezing is visible as a mismatch
        rather than silently compared against a baseline for different words.
        """
        return {
            row["query_id"]: row["query_text"]
            for row in self._c().execute(
                "SELECT query_id, query_text FROM retrieval_regression_frozen "
                "WHERE mode = ?",
                (mode,),
            )
        }

    def has_baseline(self, mode: str) -> bool:
        row = self._c().execute(
            "SELECT 1 FROM retrieval_regression_frozen WHERE mode = ? LIMIT 1",
            (mode,),
        ).fetchone()
        return row is not None

    # -- run history -------------------------------------------------------

    def record_run(
        self,
        *,
        run_id: str,
        mode: str,
        started_at: datetime | None,
        label: str | None,
        queries: int,
        mean_drift: float,
        max_drift: float,
        drifted_queries: int,
        top1_changes: int,
        abstention_violations: int,
        labelled_queries: int,
        ndcg_at_10: float | None,
        recall_at_50: float | None,
        mrr_at_10: float | None,
        detail: Mapping[str, object],
    ) -> None:
        c = self._c()
        with c:
            c.execute(
                "INSERT INTO retrieval_regression_runs("
                "run_id, mode, started_at, label, queries, mean_drift, max_drift, "
                "drifted_queries, top1_changes, abstention_violations, "
                "labelled_queries, ndcg_at_10, recall_at_50, mrr_at_10, detail_json"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    mode,
                    _iso(started_at),
                    label,
                    int(queries),
                    float(mean_drift),
                    float(max_drift),
                    int(drifted_queries),
                    int(top1_changes),
                    int(abstention_violations),
                    int(labelled_queries),
                    ndcg_at_10,
                    recall_at_50,
                    mrr_at_10,
                    json.dumps(dict(detail), sort_keys=True),
                ),
            )

    def runs(self, limit: int | None = None) -> list[dict]:
        """Run history, newest first."""
        sql = "SELECT * FROM retrieval_regression_runs ORDER BY started_at DESC"
        params: list[object] = []
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        out = []
        for row in self._c().execute(sql, params):
            item = dict(row)
            item["detail"] = json.loads(item.pop("detail_json"))
            out.append(item)
        return out

    # -- the zero-result log ----------------------------------------------

    def record_search_miss(
        self,
        query_text: str,
        *,
        mode: str | None = None,
        result_count: int = 0,
        observed_at: datetime | None = None,
        note: str | None = None,
    ) -> None:
        """Log a query that came back with nothing.

        APPEND-ONLY AND NOT DEDUPLICATED: how often a query misses is the
        ranking signal for which misses are worth freezing into the golden set.
        Collapsing repeats would throw that away.

        Called from the query path by criterion D.
        """
        c = self._c()
        with c:
            c.execute(
                "INSERT INTO search_misses"
                "(query_text, mode, result_count, observed_at, note) "
                "VALUES (?, ?, ?, ?, ?)",
                (query_text, mode, int(result_count), _iso(observed_at), note),
            )

    def search_misses(self, limit: int | None = None) -> list[dict]:
        """The zero-result log, oldest first (first-seen order is the useful one)."""
        sql = "SELECT * FROM search_misses ORDER BY id ASC"
        params: list[object] = []
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        return [dict(row) for row in self._c().execute(sql, params)]
