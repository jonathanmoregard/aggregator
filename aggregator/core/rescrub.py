"""Re-apply the SECRET layers of ``scrub`` to rows already in the store.

WHY THIS EXISTS INSTEAD OF A ``SCRUB_FINGERPRINT`` BUMP. ``core/store.py``'s
fingerprint is the documented way to carry a scrubber change to stored rows:
bump it and every row re-scrubs on the next ingest. For a change that only
ADDS secret detection that is the wrong tool, two ways over:

* it re-runs Presidio on every row — ~11 hours at the measured 827 rows/min —
  to re-derive PII redactions that are already in the stored text;
* it only reaches rows whose source still exists on disk.

``scrub`` is idempotent, so running its secret layers over the STORED text
redacts exactly what the new detectors find there and leaves every other
byte alone. Only rows whose text actually changes are written. A change to
the PII layers, or one that REMOVES a redaction, still needs the fingerprint
bump.

WHAT A WRITE KEEPS CONSISTENT, per row that changed:

* ``observations.body`` — ``obs_fts`` follows by the ``observations_au``
  trigger (``AFTER UPDATE OF body``), exactly as on the ingest UPSERT.
* ``records.subject`` / ``records.body`` — ``records_fts`` is not
  external-content, so its row is deleted and rebuilt from the stored row,
  as ``Store._write_one_record`` does.
* ``src_hash`` is NOT touched. It fingerprints the SOURCE's row, which did
  not change; leaving it means the next ingest still skips the row instead
  of re-scrubbing it from scratch.

``--dry-run`` reads only and reports counts by rule. Resumable by
construction: a re-run finds nothing left to change in rows it already did.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from aggregator.core.scrub import scrub_secrets_many
from aggregator.core.store import _RECORDS_FTS_REFRESH_SQL, Store

#: Rows per chunk. Each chunk is one read, one secret-redact call and, when not a
#: dry run, one committed write — so this bounds what a SIGTERM costs, not
#: the throughput.
DEFAULT_CHUNK = 2000

#: Chunks scanned concurrently, each by its own secret-redact process (the GIL is
#: released while it runs, and the scrubber caps it at 2 threads). Writes
#: stay serial.
DEFAULT_JOBS = 4


@dataclass
class RescrubReport:
    scanned: Counter = field(default_factory=Counter)  # rows read, per table
    changed: Counter = field(default_factory=Counter)  # rows whose text moved
    by_rule: Counter = field(default_factory=Counter)  # redactions, per rule
    interrupted: bool = False


_SELECT = {
    "observations": (
        "SELECT rowid, obs_id, body FROM observations "
        "WHERE rowid > ? ORDER BY rowid LIMIT ?"
    ),
    "records": (
        "SELECT rowid, stable_id, subject, body FROM records "
        "WHERE rowid > ? ORDER BY rowid LIMIT ?"
    ),
}


def rescrub(
    store: Store,
    *,
    dry_run: bool,
    chunk_size: int = DEFAULT_CHUNK,
    jobs: int = DEFAULT_JOBS,
    stop: Callable[[], bool] | None = None,
    progress: Callable[[RescrubReport], None] | None = None,
) -> RescrubReport:
    """Re-scrub ``observations.body`` and ``records.subject``/``body``.

    Raises ``ScrubUnavailableError`` (nothing further written) when the secret
    scanner cannot run. Each chunk commits on its own, so a raise or a stop
    loses at most the chunks in flight.
    """
    report = RescrubReport()
    jobs = max(jobs, 1)
    scan = {"observations": _scan_observations, "records": _scan_records}
    apply = {"observations": _apply_observations, "records": _apply_records}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for table in ("observations", "records"):
            last = 0
            while True:
                if stop is not None and stop():
                    report.interrupted = True
                    return report
                chunks, last = _next_chunks(store, table, last, chunk_size, jobs)
                if not chunks:
                    break
                for rows, changed in zip(chunks, pool.map(scan[table], chunks), strict=True):
                    report.scanned[table] += len(rows)
                    report.changed[table] += len(changed)
                    for _row, results in changed:
                        for res in results:
                            report.by_rule.update(res.counts)
                    if changed and not dry_run:
                        apply[table](store, changed)
                if progress is not None:
                    progress(report)
    return report


def _next_chunks(
    store: Store, table: str, last: int, chunk_size: int, jobs: int
) -> tuple[list[list], int]:
    """Up to ``jobs`` chunks of rows after rowid ``last``, and the new ``last``."""
    chunks = []
    for _ in range(jobs):
        rows = store._c().execute(_SELECT[table], (last, chunk_size)).fetchall()
        if not rows:
            break
        last = rows[-1][0]
        chunks.append(rows)
    return chunks, last


def _scan_observations(rows) -> list[tuple]:
    """``(row, [body result])`` for each row whose body changes."""
    results = scrub_secrets_many([r[2] or "" for r in rows])
    return [
        (r, [res]) for r, res in zip(rows, results, strict=True)
        if res.text != (r[2] or "")
    ]


def _apply_observations(store: Store, changed) -> None:
    # Queue for the cross-process writer slot only now, after the scan: the
    # scan is the slow part and must not hold the ingest timer off.
    store._ensure_writable()
    c = store._c()
    try:
        for (rowid, _obs_id, body), (res,) in changed:
            # COMPARE-AND-SWAP on the body that was scanned. An ingest that
            # rewrote the row between our read and our lock stored text the
            # current scrubber already cleaned; overwriting it with a rescrub
            # of the OLD text would put stale content back.
            c.execute(
                "UPDATE observations SET body = ? WHERE rowid = ? AND body = ?",
                (res.text, rowid, body),
            )
    except BaseException:
        c.rollback()
        raise
    c.commit()


def _scan_records(rows) -> list[tuple]:
    """``(row, [subject result, body result])`` for each row that changes."""
    n = len(rows)
    results = scrub_secrets_many(
        [r[2] or "" for r in rows] + [r[3] or "" for r in rows]
    )
    changed = []
    for i, r in enumerate(rows):
        subj, body = results[i], results[n + i]
        if subj.text != (r[2] or "") or body.text != (r[3] or ""):
            changed.append((r, [subj, body]))
    return changed


def _apply_records(store: Store, changed) -> None:
    store._ensure_writable()
    c = store._c()
    try:
        for (rowid, stable_id, old_subject, old_body), (subj, body) in changed:
            # Compare-and-swap, for the reason given in _apply_observations.
            cur = c.execute(
                "UPDATE records SET subject = ?, body = ? "
                "WHERE rowid = ? AND subject = ? AND body = ?",
                (subj.text, body.text, rowid, old_subject, old_body),
            )
            if not cur.rowcount:
                continue
            c.execute("DELETE FROM records_fts WHERE stable_id = ?", (stable_id,))
            c.execute(_RECORDS_FTS_REFRESH_SQL, (stable_id,))
    except BaseException:
        c.rollback()
        raise
    c.commit()
