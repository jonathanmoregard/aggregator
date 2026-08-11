"""The unified runner: N adapters, one pass, one report.

Why this exists: exactly one source auto-imports today (github, via a
systemd user timer); the rest are hand-run and drift days to weeks stale.
One runner means one timer can drive every source, and the
notify-on-failure wiring is written once instead of once per source.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from aggregator.imports.port import (
    ImportAdapter,
    ImportItem,
    ImportSink,
    SupportsInputFreshness,
    SupportsNonFatalErrors,
)

# Items buffered before a sink write. Bounded so a 359k-observation source
# streams through in constant memory instead of materialising.
DEFAULT_BATCH_SIZE = 500


@dataclass
class AdapterReport:
    """One adapter's outcome. Counts come from the sink, never from len()."""

    name: str
    added: int = 0
    updated: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)
    # Newest timestamp among the inputs this adapter read, when it can say
    # (``SupportsInputFreshness``). None = didn't offer / doesn't know.
    #
    # Read by the notify hook, which is why the hook now runs on every run:
    # gated on failure, this field was unreachable on exactly the runs it
    # exists to describe — the clean ones that imported nothing because the
    # input is stale.
    input_newest_at: datetime | None = None

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class RunReport:
    """Whole-run outcome, keyed by adapter name."""

    adapters: dict[str, AdapterReport] = field(default_factory=dict)
    # Faults belonging to the run itself rather than to any one adapter —
    # currently just a notification hook that blew up.
    run_errors: list[str] = field(default_factory=list)

    @property
    def added(self) -> int:
        return sum(a.added for a in self.adapters.values())

    @property
    def updated(self) -> int:
        return sum(a.updated for a in self.adapters.values())

    @property
    def skipped(self) -> int:
        return sum(a.skipped for a in self.adapters.values())

    @property
    def errors(self) -> list[str]:
        return [
            *(f"{a.name}: {e}" for a in self.adapters.values() for e in a.errors),
            *self.run_errors,
        ]

    @property
    def failed_adapters(self) -> list[str]:
        return [name for name, a in self.adapters.items() if not a.ok]

    @property
    def ok(self) -> bool:
        """False when ANY adapter errored — what the CLI turns into an exit
        code (that wiring is a separate task; the runner only exposes it)."""
        return not self.failed_adapters and not self.run_errors


NotifyHook = Callable[[RunReport], None]


def _no_notification(report: RunReport) -> None:
    """Default hook: do nothing.

    Library code must not shell out to ``notify-send``. The CLI / systemd
    layer injects the real notifier, which keeps the failure path testable
    and keeps the desktop dependency out of the import path.
    """


async def _run_one(
    adapter: ImportAdapter,
    sink: ImportSink,
    batch_size: int,
) -> AdapterReport:
    report = AdapterReport(name=adapter.name)
    batch: list[ImportItem] = []

    def flush() -> None:
        if not batch:
            return
        # Detach before writing: if the sink raises, the batch is already
        # out of the buffer, so the recovery flush below can't retry the
        # same doomed write forever.
        pending = list(batch)
        batch.clear()
        counts = sink.write(pending)
        report.added += counts.added
        report.updated += counts.updated
        report.skipped += counts.skipped

    try:
        async for item in adapter.get_data():
            batch.append(item)
            if len(batch) >= batch_size:
                flush()
        flush()
    except Exception as e:  # noqa: BLE001 -- isolation boundary, see below
        # PER-ADAPTER FAILURE ISOLATION. One source dying (expired token,
        # unreachable dir, locked DB) must not deny the other seven their
        # run. The exception is recorded with its type so the notification
        # and the CLI can say what broke — it is contained here, never
        # swallowed: ``report.ok`` goes False and stays False.
        # BaseException (CancelledError, KeyboardInterrupt) deliberately
        # propagates — those mean the whole run is being torn down.
        report.errors.append(f"{type(e).__name__}: {e}")
        try:
            # Partial ingest beats total loss: whatever arrived before the
            # crash still gets written.
            flush()
        except Exception as e2:  # noqa: BLE001
            report.errors.append(
                f"flush after failure: {type(e2).__name__}: {e2}"
            )

    # Optional protocols, checked structurally — an adapter opts in simply by
    # having the method. Both run after the stream ends, including when it
    # ended by raising: a source that logged five unreadable files before
    # dying must still surface all five.
    if isinstance(adapter, SupportsNonFatalErrors):
        try:
            report.errors.extend(adapter.drain_errors())
        except Exception as e:  # noqa: BLE001
            report.errors.append(f"drain_errors failed: {type(e).__name__}: {e}")
    if isinstance(adapter, SupportsInputFreshness):
        try:
            report.input_newest_at = adapter.input_freshness()
        except Exception as e:  # noqa: BLE001
            report.errors.append(f"input_freshness failed: {type(e).__name__}: {e}")
    return report


async def run_imports(
    adapters: Iterable[ImportAdapter],
    sink: ImportSink,
    *,
    notify: NotifyHook = _no_notification,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> RunReport:
    """Drive every adapter concurrently and return the aggregated report.

    ``notify`` fires on EVERY run, not only failing ones, and the hook decides
    what is worth telling a human. The runner has no way to know: a clean run
    that imported nothing because the export archive is 31 days old is not an
    error — nothing broke, a person just has not exported lately — yet it is
    exactly what an operator needs told, and it is indistinguishable from a
    healthy no-op in every other channel. ``AdapterReport.input_newest_at``
    exists for a notifier to weigh, which it could not do while notification
    was gated on ``not report.ok``: on a clean run no notifier ever ran, so
    the value was unreachable by construction. The default hook is a no-op,
    so "fires on every run" costs a caller who wants nothing exactly nothing.
    """
    adapter_list: Sequence[ImportAdapter] = list(adapters)
    _refuse_duplicate_names(adapter_list)
    results = await asyncio.gather(
        *(_run_one(a, sink, batch_size) for a in adapter_list)
    )
    report = RunReport(adapters={r.name: r for r in results})
    try:
        notify(report)
    except Exception as e:  # noqa: BLE001
        # A missing notify-send must not cost us the report — the caller
        # still has to be able to print WHICH adapter failed. Recorded,
        # not swallowed: a notifier that cannot notify is itself a fault,
        # and on an otherwise-clean run this is the only thing that says so.
        report.run_errors.append(f"notify hook failed: {type(e).__name__}: {e}")
    return report


def _refuse_duplicate_names(adapters: Sequence[ImportAdapter]) -> None:
    """The report is keyed by adapter name, so a collision would silently
    drop one source's entire outcome. Refuse before any work is done."""
    seen: set[str] = set()
    for a in adapters:
        if a.name in seen:
            raise ValueError(f"duplicate adapter name: {a.name!r}")
        seen.add(a.name)


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "AdapterReport",
    "NotifyHook",
    "RunReport",
    "run_imports",
]
