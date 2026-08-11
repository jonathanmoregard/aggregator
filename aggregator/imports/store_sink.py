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

        counts = WriteCounts()
        if records:
            counts = counts + self._write_records(records)
        if sessions or observations:
            counts = counts + self._write_entities(sessions, observations)
        return counts

    def _write_records(self, records: list[Record]) -> WriteCounts:
        existing = self._store.existing_ids(
            "records", [r.stable_id for r in records]
        )
        # De-duplicate within the batch too: two items with the same id in
        # one batch are one add, not two.
        seen: set[str] = set()
        added = updated = 0
        for r in records:
            if r.stable_id in seen:
                updated += 1
                continue
            seen.add(r.stable_id)
            if r.stable_id in existing:
                updated += 1
            else:
                added += 1
        self._store.upsert(records)
        return WriteCounts(added=added, updated=updated)

    def _write_entities(
        self,
        sessions: list[SessionRow],
        observations: list[ObservationRow],
    ) -> WriteCounts:
        existing_sessions = self._store.existing_ids(
            "sessions", [s.session_id for s in sessions]
        )
        existing_obs = self._store.existing_ids(
            "observations", [o.obs_id for o in observations]
        )
        added = updated = 0
        for sid in _unique([s.session_id for s in sessions]):
            updated += 1 if sid in existing_sessions else 0
            added += 0 if sid in existing_sessions else 1
        for oid in _unique([o.obs_id for o in observations]):
            updated += 1 if oid in existing_obs else 0
            added += 0 if oid in existing_obs else 1
        # Sessions first: ``observations.session_id`` is a real FK, and an
        # adapter is allowed to yield an observation before its session row
        # (batch boundaries can split a stream anywhere).
        self._store.upsert_entities([*sessions, *observations])
        return WriteCounts(added=added, updated=updated)


def _unique(ids: list[str]) -> list[str]:
    """Order-preserving dedupe — repeats within one batch are one row."""
    seen: set[str] = set()
    out: list[str] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


__all__ = ["StoreSink"]
