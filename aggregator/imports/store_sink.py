"""The real sink: an ``ImportSink`` backed by the SQLite store.

Dispatches on the item's concrete type rather than asking adapters to
declare a shape, so one batch can carry both ontologies and neither has to
pretend to be the other (``core/store.py``: a PR is not naturally a session).

Counts are established BEFORE the write, by probing which primary keys the
store already holds. Both write paths are upserts, so afterwards there is no
way to tell an insert from an overwrite — and a summary that always reports
``added=<batch size>`` (which is what ``cli.py`` prints today) is worse than
no summary, because it looks like progress on a run that changed nothing.
"""
from __future__ import annotations

from collections.abc import Sequence

from aggregator.core.store import Store
from aggregator.imports.port import ImportItem, WriteCounts
from aggregator.sources.base import ObservationRow, Record, SessionRow


def count_writes(store: Store, table: str, ids: Sequence[str]) -> WriteCounts:
    """Probe which of ``ids`` ``table`` already holds, ahead of the write.

    MUST be called before the upsert. Afterwards every id is present and the
    add/update distinction is gone for good — which is precisely how
    ``cli.py`` ended up printing ``added=<batch size> updated=0`` on every
    run, including runs that changed nothing.

    Repeats within one batch address one row, so the first occurrence is the
    add (or update) and any later one is an update.

    Shared with ``cli.py`` deliberately: one definition of "added", so the
    hand-run CLI summary and the runner's report can never disagree about
    what happened.
    """
    existing = store.existing_ids(table, ids)
    seen: set[str] = set()
    added = updated = 0
    for i in ids:
        if i in seen or i in existing:
            updated += 1
        else:
            added += 1
        seen.add(i)
    return WriteCounts(added=added, updated=updated)


class StoreSink:
    """Write import items to a ``Store``, reporting what actually changed.

    Synchronous (per ``ImportSink``): SQLite writes are best serialised, and
    a sync call can't be suspended by the event loop, so adapters running
    concurrently can share one instance without a lock.

    ``skipped`` is always 0 here — every item this sink accepts is either an
    insert or an overwrite, and an unsupported type raises rather than being
    quietly dropped. The field stays in ``WriteCounts`` for sinks that do
    filter (a dry-run sink, a future no-change detector).
    """

    def __init__(self, store: Store) -> None:
        self._store = store

    def write(self, items: Sequence[ImportItem]) -> WriteCounts:
        records: list[Record] = []
        sessions: list[SessionRow] = []
        observations: list[ObservationRow] = []
        for item in items:
            if isinstance(item, Record):
                records.append(item)
            elif isinstance(item, SessionRow):
                sessions.append(item)
            elif isinstance(item, ObservationRow):
                observations.append(item)
            else:
                raise TypeError(
                    f"unsupported import item: {type(item).__name__}"
                )

        # VALIDATE THE WHOLE BATCH BEFORE WRITING ANY OF IT. This used to sit
        # inside the entity path, i.e. after the records had already landed —
        # and the raise discards this method's return value, so the counts for
        # the rows that DID land went with it. Rows in the store that no
        # report ever counted is precisely the failure this file's counting
        # exists to prevent.
        self._refuse_orphan_observations(sessions, observations)

        # ONE INVARIANT ACROSS BOTH PATHS: ``added + updated == len(items)``,
        # and ``skipped`` is always 0 here. Neither path de-duplicates when
        # counting, so a repeated id is one add and one update whether the
        # repeat sits inside a batch or straddles a flush — i.e. the totals do
        # not move when ``batch_size`` does.
        #
        # A MIXED BATCH IS NOT ATOMIC, and that is the one gap left. If the
        # entity write raises after the record write committed, this method's
        # return value goes with the exception and the landed records are
        # counted by nobody. Each store call rolls its OWN batch back, so
        # nothing is half-written, but the two are separate transactions.
        # Unreachable today — ``SyncSourceAdapter`` binds one source to one of
        # ``iter_entities`` / ``iter_records``, so no shipped adapter can emit
        # both shapes, and the runner batches one adapter at a time — and the
        # run is already reporting the exception, so it cannot look like
        # success. An adapter that DOES mix shapes must not rely on the counts
        # of a failing write; closing it properly needs one transaction
        # spanning both store calls, which is a change to ``Store``'s surface,
        # not to this file.
        counts = WriteCounts()
        if records:
            counts = counts + self._write_records(records)
        if sessions or observations:
            counts = counts + self._write_entities(sessions, observations)
        return counts

    def _write_records(self, records: list[Record]) -> WriteCounts:
        # De-duplicates within the batch too: two items with the same id in
        # one batch are one add, not two.
        counts = count_writes(
            self._store, "records", [r.stable_id for r in records]
        )
        self._store.upsert(records)
        return counts

    def _write_entities(
        self,
        sessions: list[SessionRow],
        observations: list[ObservationRow],
    ) -> WriteCounts:
        # COUNTED EXACTLY LIKE THE RECORDS PATH: every item is one add or one
        # update, so ``added + updated == len(items)`` holds on both paths and
        # a caller can read the two summaries the same way.
        #
        # This used to de-duplicate within the batch, which made the entity
        # counts disagree with the record counts about a repeat AND made the
        # run's totals depend on ``batch_size``: the same four-item stream
        # reported added=2 updated=0 in one batch and added=2 updated=2 in two,
        # because a repeat spanning a flush boundary was already stored by the
        # time the second batch was counted and no dedupe could reach across.
        # A number that changes with an unrelated tuning knob is not a report.
        counts = count_writes(
            self._store, "sessions", [s.session_id for s in sessions]
        ) + count_writes(
            self._store, "observations", [o.obs_id for o in observations]
        )
        # Sessions before observations WITHIN THE BATCH, because
        # ``observations.session_id`` is a real FK and ``PRAGMA foreign_keys``
        # is ON. This reordering is per-batch and cannot be anything else — the
        # sink sees one batch at a time and has no memory between calls. The
        # contract on the adapter is therefore the stronger one, checked in
        # ``write`` before anything at all is written.
        self._store.upsert_entities([*sessions, *observations])
        return counts

    def _refuse_orphan_observations(
        self,
        sessions: list[SessionRow],
        observations: list[ObservationRow],
    ) -> None:
        """Reject observations whose session is neither in this batch nor stored.

        THE ADAPTER CONTRACT: a session row must be yielded before any of its
        observations, in stream order. This used to be documented as "an
        adapter is allowed to yield an observation before its session row
        (batch boundaries can split a stream anywhere)", which was true only
        inside one batch and is precisely backwards about batch boundaries —
        the runner flushes every 500 items, and an observation landing in an
        earlier batch than its session hits the FK. That shape is invisible in
        a small test (one batch, reordered, green) and aborts the adapter on
        real volume.

        Checked here rather than left to SQLite so the failure names what to
        change. ``FOREIGN KEY constraint failed`` says nothing about which
        observation, which session, or that an ordering rule exists at all.
        """
        if not observations:
            return
        in_batch = {s.session_id for s in sessions}
        wanted = {o.session_id for o in observations if o.session_id not in in_batch}
        if not wanted:
            return
        missing = wanted - self._store.existing_ids("sessions", sorted(wanted))
        if not missing:
            return
        example = next(
            o for o in observations if o.session_id in missing
        )
        raise ValueError(
            f"observation {example.obs_id!r} references session "
            f"{example.session_id!r}, which is neither in this batch nor "
            f"already stored ({len(missing)} session id(s) affected). An "
            f"adapter must yield a session row BEFORE any of its observations "
            f"— the runner flushes the stream in batches and the sink can only "
            f"reorder within one batch, so an observation that precedes its "
            f"session in the stream hits the sessions foreign key."
        )


__all__ = ["StoreSink", "count_writes"]
