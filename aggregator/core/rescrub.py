"""Re-apply the SECRET layers of ``scrub`` to rows already in the store.

WHY THIS EXISTS INSTEAD OF A ``SCRUB_FINGERPRINT`` BUMP. ``core/store.py``'s
fingerprint is the documented way to carry a scrubber change to stored rows:
bump it and every row re-scrubs on the next ingest. For a change that only
ADDS secret detection that is the wrong tool, three ways over:

* it re-runs Presidio on every row — ~11 hours at the measured 827 rows/min —
  to re-derive PII redactions that are already in the stored text;
* the ingest UPSERT resets ``embedding_state`` and drops the vectors of every
  row it rewrites, so the whole vector arm (weeks of CPU) would be discarded
  to change a few hundred rows;
* it only reaches rows whose source still exists on disk.

``scrub`` is idempotent, so running its secret layers over the STORED text
redacts exactly what the new detectors find there and leaves every other
byte alone. Only rows whose text actually changes are written, and only
those lose their vectors.

WHAT A WRITE KEEPS CONSISTENT, per row that changed:

* ``observations.body`` — ``obs_fts`` follows by the ``observations_au``
  trigger (``AFTER UPDATE OF body``), exactly as on the ingest UPSERT.
* ``records.subject`` / ``records.body`` — ``records_fts`` is not
  external-content, so its row is deleted and rebuilt from the stored row,
  as ``Store._write_one_record`` does.
* the vector arm — ``embedding_state = NULL`` puts the row back in the embed
  backlog and its vectors (and ``chunk_embeddings`` rows) are dropped, so no
  vector keeps describing text that is no longer there.
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
from aggregator.core.store import (
    _RECORDS_FTS_REFRESH_SQL,
    Store,
    _drop_row_vectors,
    _vec_table_usable,
)

#: Rows per chunk. Each chunk is one read, one or a few gitleaks calls (the
#: scanner splits at ~4 MB) and, when not a dry run, one committed write —
#: so this bounds what a SIGTERM costs, not the throughput.
DEFAULT_CHUNK = 2000

#: Chunks scanned concurrently. The scan is gitleaks subprocesses (the GIL is
#: released while they run), measured ~0.5 MB/s each; the store holds ~1.1 GB
#: of text, so one scanner is ~40 min and four are ~10. Writes stay serial.
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
    c = store._c()
    scan = {"observations": _scan_observations, "records": _scan_records}
    apply = {"observations": _apply_observations, "records": _apply_records}
    with ThreadPoolExecutor(max_workers=max(jobs, 1)) as pool:
        for table in ("observations", "records"):
            last = 0
            while True:
                if stop is not None and stop():
                    report.interrupted = True
                    return report
                chunks = []
                for _ in range(max(jobs, 1)):
                    rows = c.execute(_SELECT[table], (last, chunk_size)).fetchall()
                    if not rows:
                        break
                    last = rows[-1][0]
                    chunks.append(rows)
                if not chunks:
                    break
                for rows, changed in zip(
                    chunks, pool.map(scan[table], chunks), strict=True
                ):
                    report.scanned[table] += len(rows)
                    report.changed[table] += len(changed)
                    for item in changed:
                        for res in item[1:]:
                            report.by_rule.update(res.counts)
                    if not dry_run and changed:
                        apply[table](store, changed)
                if progress is not None:
                    progress(report)
    return report


def _scan_observations(rows) -> list[tuple]:
    results = scrub_secrets_many([r[2] or "" for r in rows])
    return [
        (r, res) for r, res in zip(rows, results, strict=True)
        if res.text != (r[2] or "")
    ]


def _apply_observations(store: Store, changed) -> None:
    # Queue for the cross-process writer slot only now, after the scan: the
    # scan is the slow part and must not hold the ingest timer off.
    store._ensure_writable()
    c = store._c()
    vec = _vec_table_usable(c, "vec_observations")
    try:
        for (rowid, obs_id, body), res in changed:
            # COMPARE-AND-SWAP on the body that was scanned. An ingest that
            # rewrote the row between our read and our lock stored text the
            # current scrubber already cleaned; overwriting it with a rescrub
            # of the OLD text would put stale content back.
            cur = c.execute(
                "UPDATE observations SET body = ?, embedding_state = NULL "
                "WHERE rowid = ? AND body = ?",
                (res.text, rowid, body),
            )
            if cur.rowcount and vec:
                _drop_row_vectors(c, "vec_observations", "obs_id", obs_id)
    except BaseException:
        c.rollback()
        raise
    c.commit()


def _scan_records(rows) -> list[tuple]:
    n = len(rows)
    results = scrub_secrets_many(
        [r[2] or "" for r in rows] + [r[3] or "" for r in rows]
    )
    changed = []
    for i, r in enumerate(rows):
        subj, body = results[i], results[n + i]
        if subj.text != (r[2] or "") or body.text != (r[3] or ""):
            changed.append((r, subj, body))
    return changed


def _apply_records(store: Store, changed) -> None:
    store._ensure_writable()
    c = store._c()
    vec = _vec_table_usable(c, "vec_records")
    try:
        for (rowid, stable_id, old_subject, old_body), subj, body in changed:
            # Compare-and-swap, for the reason given in _apply_observations.
            cur = c.execute(
                "UPDATE records SET subject = ?, body = ?, embedding_state = NULL "
                "WHERE rowid = ? AND subject = ? AND body = ?",
                (subj.text, body.text, rowid, old_subject, old_body),
            )
            if not cur.rowcount:
                continue
            if vec:
                _drop_row_vectors(c, "vec_records", "stable_id", stable_id)
            c.execute("DELETE FROM records_fts WHERE stable_id = ?", (stable_id,))
            c.execute(_RECORDS_FTS_REFRESH_SQL, (stable_id,))
    except BaseException:
        c.rollback()
        raise
    c.commit()
