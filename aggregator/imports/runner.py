"""The unified runner: N adapters, one pass, one report.

Why this exists: exactly one source auto-imports today (github, via a
systemd user timer); the rest are hand-run and drift days to weeks stale.
One runner means one timer can drive every source, and the
notify-on-failure wiring is written once instead of once per source.

HOW A RUN ADVANCES, AND WHY IT IS SHAPED THIS WAY
=================================================
The 2026-08-15 live run had no unit of work that was ever durably
acknowledged: ``since`` was always ``None``, so every 30-minute tick re-ingested
all 372k observations from scratch, and being SIGTERMed at 44% threw all of it
away. Four properties fix that, and each one is load-bearing:

* **Chunked.** Every ``batch_size`` items are written and COMMITTED. A kill at
  any moment therefore costs at most one chunk of writing, never the run.
* **Checkpointed at end of stream.** The source's high-water mark rides the
  FINAL chunk, in the same transaction. It cannot ride an earlier one, because
  nothing here yields items in cursor order — the sessions source walks a
  directory tree, a chat export is grouped by conversation — so a mid-stream
  maximum says nothing about what is still to come, and advancing to it would
  skip every later record below that value, permanently. See ``port.Checkpoint``.
* **Cheap to redo.** What makes "lose one chunk, re-read the window" tolerable
  is that re-writing an unchanged row costs no scrub and no page write
  (``core/store.py``'s ``src_hash`` guard). Without it, chunking would trade a
  doom loop for a slow bleed.
* **Loud.** Every chunk logs, and a heartbeat logs even when no chunk does.
  The two-hour silence that started this happened because the leg doing all
  the work said nothing at all.
"""
from __future__ import annotations

import asyncio
import contextlib
import signal
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from aggregator.imports.ingest_state import (
    STALE_INPUTS,
    FaultVerdict,
    HeldRecord,
    IngestMarkers,
    KnownFault,
    PoisonLedger,
    SourcePlan,
    Watermarks,
    full_jitter,
)
from aggregator.imports.port import (
    Checkpoint,
    Delivery,
    ImportAdapter,
    ImportItem,
    ImportSink,
    SupportsCheckpoint,
    SupportsInputFreshness,
    SupportsNonFatalErrors,
    SupportsPermanentFaults,
    SupportsWriteBarrier,
    WriteCounts,
    is_report_gating,
)
from aggregator.imports.progress import RunProgress
from aggregator.sources.base import (
    ObservationRow,
    PermanentFault,
    Record,
    SessionRow,
)

# Items buffered before a sink write. Bounded so a 359k-observation source
# streams through in constant memory instead of materialising.
#
# ALSO THE UNIT OF LOSS. A chunk is what a SIGTERM can cost, so it wants to be
# small; it is also one transaction, so it wants to be big enough that commit
# overhead is noise. 500 rows of this pipeline's work is well under a second,
# which is comfortably inside the unit's ``TimeoutStopSec`` and fine-grained
# enough that the progress log is a pulse rather than an occasional event.
DEFAULT_BATCH_SIZE = 500

# How many times a chunk write is retried before its items are examined one by
# one. This covers the TRANSIENT class only — ``database is locked`` under a
# concurrent writer is the one that actually happens — and the delays are
# jittered because un-jittered retries re-synchronise into clusters.
CHUNK_RETRY_ATTEMPTS = 3
CHUNK_RETRY_BASE = 0.2
CHUNK_RETRY_CAP = 5.0


@dataclass
class AdapterReport:
    """One adapter's outcome. Counts come from the sink, never from len()."""

    name: str
    added: int = 0
    updated: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)
    # How many of ``updated`` were already stored byte-identically, so cost no
    # scrub and no page write. THE number that tells a re-run from the doom
    # loop: both report ~372k updates, only one of them did nothing expensive.
    unchanged: int = 0
    # Records that could not be written and were set aside rather than allowed
    # to abort the run. Non-zero is loud (it lands in ``errors``), because a
    # dropped record that nobody counts is a gap that reads as full coverage.
    quarantined: int = 0
    # The window this source was given, in words — "since <ts>", "FULL SCAN
    # (first run)", "FULL SCAN (no usable cursor): ...". Carried into the report
    # so a journal entry read six hours later can answer "was this run
    # incremental?" without anyone opening the code, and so a source that
    # CANNOT be windowed says so instead of looking like one that can.
    window: str = ""
    cursor_kind: str = ""
    # Where the mark ended up, or None when it did not move. It does not move
    # after an interrupted or failed pass, ON PURPOSE: the next run re-reads
    # the same window, which is cheap, and nothing in it can be lost.
    advanced_to: datetime | None = None
    # SIGTERM arrived and this source stopped at a chunk boundary. Not an
    # error — it is the designed behaviour — but it is why the mark did not
    # move, and a report that did not say so would look like a clean run that
    # found less than it should have.
    interrupted: bool = False
    # The source was not run at all this tick because it is resting after
    # repeated failures. Distinct from "ran and found nothing", which prints
    # the same zeros.
    skipped_for_backoff: bool = False
    # Operator-facing, non-error notes: the backoff line, a refused watermark
    # advance. Deliberately not ``errors`` — nothing failed on THIS run — and
    # deliberately not ``warnings`` either, which are notified and deduplicated
    # per episode.
    notes: list[str] = field(default_factory=list)
    # Newest timestamp among the inputs this adapter read, when it can say
    # (``SupportsInputFreshness``). None = didn't offer / doesn't know.
    #
    # Read by the notify hook, which is why the hook now runs on every run:
    # gated on failure, this field was unreachable on exactly the runs it
    # exists to describe — the clean ones that imported nothing because the
    # input is stale.
    input_newest_at: datetime | None = None
    # Whether the adapter OPTED IN to ``SupportsReportBarrier``
    # (``port.is_report_gating``), i.e. whether any of this adapter's error lines
    # GATES a receipt. Recorded because the notifier composes a size-limited
    # payload and has to put those lines in it first: a line that is elided is a
    # line that was not delivered, and an adapter that cannot show its whole
    # report got through keeps repeating it.
    #
    # Opt-in rather than structural because this field is only useful when it
    # discriminates. Round 8: ``isinstance`` against the runtime-checkable
    # protocol answered yes for all nine sources — ``SyncSourceAdapter`` defines
    # the forwarding method for every one of them — so ``gating_errors`` was the
    # whole error list and the prioritisation below did nothing at all.
    holds_report_barrier: bool = False
    # Permanently-bad input this adapter reported that the ledger had ALREADY
    # been told about. Their error lines are removed from ``errors`` above and
    # rendered as notes instead, which is what lets a run whose only faults are
    # known ones exit 0 and stop notifying. Never dropped silently: the count is
    # in the run summary and every one of them is listed by ``aggregator
    # status``.
    known_faults: list[KnownFault] = field(default_factory=list)
    # Permanently-bad input NOBODY has been told about yet. These stay in
    # ``errors`` — the run fails, the notification fires — and they are written
    # into the ledger only once a channel declares it carried their line. See
    # ``commit_fault_receipts``.
    new_faults: list[PermanentFault] = field(default_factory=list)
    # Whether the adapter implements ``SupportsInputFreshness`` at all.
    # Without it, ``input_newest_at is None`` conflates two opposite things:
    # "this source has no export ritual and never goes stale" (github reads a
    # live API) and "the export archive is missing entirely", which is the
    # loudest version of the problem — every count is zero and it looks
    # exactly like a healthy no-op.
    offers_input_freshness: bool = False

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
    # Non-fatal, operator-actionable notices: a hand-refreshed export that has
    # gone stale, or one that is missing entirely. Deliberately NOT errors —
    # nothing failed, a human just has not exported lately, and it is fixed by
    # a different action than a crashed adapter — so these do not touch ``ok``
    # and cannot change the exit code. They live ON the report rather than
    # only on the caller's stderr because otherwise the notify hook, the one
    # thing that can reach a human, cannot see them.
    #
    # ONLY THE UN-SUPPRESSED ONES. A source is warned about once per staleness
    # EPISODE and then goes quiet — see :func:`plan_staleness_report`.
    warnings: list[str] = field(default_factory=list)
    # This run's staleness verdict and the marker write that has to wait for
    # delivery, or None when the caller set no staleness policy. Carried on the
    # report because the CLI has a channel the runner cannot see (a watched
    # terminal, printed after ``run_imports`` returned) and must be able to
    # commit against it too — the same reason ``commit_report_barriers`` is
    # public. See :func:`commit_staleness_receipts`.
    stale_episodes: StalenessEpisodes | None = None
    # Correlates every progress line in the journal with this run. Cheap, and
    # the difference between "these two lines are from the same run" being a
    # fact and being a guess when a timer fires every 30 minutes.
    run_id: str = ""
    # SIGTERM arrived and at least one source stopped at a chunk boundary. NOT
    # a failure — every committed chunk is durable and the next run resumes —
    # but the run did less than a full pass and the summary must say so, or a
    # deliberately shortened run reads as a suspiciously quiet one.
    interrupted: bool = False

    @property
    def added(self) -> int:
        return sum(a.added for a in self.adapters.values())

    @property
    def unchanged(self) -> int:
        return sum(a.unchanged for a in self.adapters.values())

    @property
    def quarantined(self) -> int:
        return sum(a.quarantined for a in self.adapters.values())

    @property
    def known_faults(self) -> list[KnownFault]:
        """Every permanently-bad input this run went quiet about."""
        return [f for a in self.adapters.values() for f in a.known_faults]

    @property
    def new_faults(self) -> list[PermanentFault]:
        """Every permanently-bad input this run reported for the FIRST time."""
        return [f for a in self.adapters.values() for f in a.new_faults]

    @property
    def quarantined_records(self) -> int:
        """How many records the known faults cost the index, this run's view.

        The number an operator actually wants when a run reports zero errors
        over a file that has been unparseable since March. Printed in the run
        summary; ``aggregator status`` prints the whole ledger, including the
        faults from sources this run never reached.
        """
        return sum(f.count for f in self.known_faults)

    @property
    def updated(self) -> int:
        return sum(a.updated for a in self.adapters.values())

    @property
    def skipped(self) -> int:
        return sum(a.skipped for a in self.adapters.values())

    def errors_from(self, name: str) -> list[str]:
        """This adapter's error lines, spelled exactly as the report shows them.

        The one place the ``"{adapter}: {error}"`` rendering lives, because two
        renderings would silently disagree: a report barrier fires only when a
        channel's payload contained every line ITS adapter reported, and that
        comparison is WHOLE-LINE identity (``Delivery.accepted``). A second copy
        of this format that drifted would make coverage unachievable — safe, but
        permanently loud for no reason a reader could find.

        The prefix is also what keeps two adapters from ever rendering the same
        line, which is why ``Delivery`` can treat identical text as one
        sentence. It is NOT what keeps one adapter's line from being delivered
        by another's: names are not validated against being suffixes of each
        other, and under R8's substring matching ``box: file unreadable`` was
        covered by ``dropbox: file unreadable``.
        """
        entry = self.adapters.get(name)
        return [] if entry is None else [f"{entry.name}: {e}" for e in entry.errors]

    @property
    def errors(self) -> list[str]:
        return [
            *(line for name in self.adapters for line in self.errors_from(name)),
            *self.run_errors,
        ]

    @property
    def reported(self) -> list[str]:
        """EVERYTHING this run said out loud — errors AND warnings.

        What a channel declares delivery against (``Delivery.accepted(payload,
        report.reported)``). Errors alone was right while a receipt could only
        ever suppress an error line; a staleness warning is now suppressed by
        the same mechanism, and a line that is not in ``reported`` can never be
        in the delivered set, so its marker could never be earned and the
        warning would repeat on every 30-minute tick forever.

        Warnings LAST, deliberately: a channel that takes a prefix of the
        payload keeps spending its budget on failures first, and a staleness
        warning that gets cut is merely repeated next run.
        """
        return [*self.errors, *self.warnings]

    @property
    def gating_errors(self) -> list[str]:
        """The error lines that gate a receipt, i.e. that a payload must carry.

        A notification has a size budget and this is what gets spent first. An
        adapter without a report barrier repeats its errors next run regardless,
        so eliding one of those costs an operator a line in one toast; eliding a
        gating one used to cost them the alert entirely, and now costs them the
        suppression that keeps a known, already-reported gap from re-alarming on
        every timer tick.

        Only worth anything while it is a MINORITY of the errors, which is what
        ``port.is_report_gating``'s explicit opt-in buys: when every adapter
        counted as gating this returned the whole list and prioritising by it
        reordered nothing.

        A NEVER-SEEN PERMANENT FAULT GATES TOO, and for exactly the same reason
        a report barrier does: its line is what buys the entry in the poison
        ledger, so a line elided from the toast is a fault that will be
        reported all over again next run. With eight of them and a five-line
        budget the run converges in two ticks instead of one; without the
        priority a chatty unrelated source could keep one of them out of the
        payload indefinitely, which is the round-7 starvation in a new costume.
        """
        gating = {
            *(
                line
                for name, a in self.adapters.items()
                if a.holds_report_barrier
                for line in self.errors_from(name)
            ),
            *(
                f"{name}: {fault.line}"
                for name, a in self.adapters.items()
                for fault in a.new_faults
            ),
        }
        # Filtered out of ``errors`` rather than assembled independently, so the
        # lines keep the report's order and a line that is somehow not in the
        # report cannot be prioritised into the payload.
        return [line for line in self.errors if line in gating]

    @property
    def failed_adapters(self) -> list[str]:
        return [name for name, a in self.adapters.items() if not a.ok]

    @property
    def ok(self) -> bool:
        """False when ANY adapter errored — what the CLI turns into an exit
        code (that wiring is a separate task; the runner only exposes it)."""
        return not self.failed_adapters and not self.run_errors


# A hook DECLARES what it delivered by returning ``Delivery.accepted(payload,
# report.errors)``. ``None`` is in the type because that is what a hook that only
# prints, only logs, or does nothing at all returns, and every one of those
# carried nothing to anybody — see ``port.Delivery``. The runner checks the TYPE,
# so no other return value can pass for a declaration either.
NotifyHook = Callable[[RunReport], Delivery | None]


def _no_notification(report: RunReport) -> None:
    """Default hook: do nothing. CANNOT count as delivery, by construction.

    Library code must not shell out to ``notify-send``. The CLI / systemd
    layer injects the real notifier, which keeps the failure path testable
    and keeps the desktop dependency out of the import path.

    Annotated ``-> None`` and it means it: this function sends no payload, so it
    has nothing to build a ``Delivery`` out of and a run with nothing configured
    cannot stamp a report barrier's receipt. It used to — the receipt was gated
    on ``notify``
    returning without raising, and doing nothing returns without raising, so
    every default-configured run recorded that a human had been told by a
    channel that does not exist. Making the default hook's silence unable to
    speak is the fix; a gate saying ``if notify is not _no_notification`` would
    have been one an alternative silent hook walks straight past.
    """


def item_cursor(item: ImportItem) -> datetime | None:
    """The timestamp a source's ``since`` is compared against, per item shape.

    MUST MATCH WHAT THE SOURCE FILTERS ON, or the mark is measured against one
    quantity and applied to another and rows are dropped in the gap. Verified
    per source, not inferred from the shape:

    * ``SessionRow.last_ts`` — sessions, chatgpt and claude-web all skip a
      conversation whose ``last_ts`` is below ``since``.
    * ``ObservationRow.ts`` — always <= its session's ``last_ts``, so taking the
      maximum over a mixed stream is still the maximum ``last_ts``.
    * ``Record.updated_at`` — every records-shaped source writes the very value
      it filters on into this field: file mtime for dropbox / research /
      sota-watch, zip-member mtime for substack, the API's ``updated_at`` for
      github. TickTick is the exception and is why this had to be checked one
      source at a time: its ``since`` filters BACKUP FILES by mtime while its
      records carry the TASK's completion time. It is declared
      ``CursorKind.NONE`` for exactly that reason and never gets a window.

    ``None`` means "this item cannot be placed on the cursor", which is treated
    as poison for the WHOLE pass's mark — see ``_run_one``.
    """
    if isinstance(item, SessionRow):
        return item.last_ts
    if isinstance(item, ObservationRow):
        return item.ts
    if isinstance(item, Record):
        return item.updated_at
    return None


def item_key(item: ImportItem) -> str:
    """The stable identity a held record is remembered by."""
    if isinstance(item, SessionRow):
        return item.session_id
    if isinstance(item, ObservationRow):
        return item.obs_id
    if isinstance(item, Record):
        return item.stable_id
    return repr(item)


def _sink_write(
    sink: ImportSink, items: Sequence[ImportItem], checkpoint: Checkpoint | None
) -> WriteCounts:
    """One chunk through the sink, with its mark when there is one.

    A sink that cannot checkpoint (a counting stub, a dry-run sink, a test
    double) gets the plain write and the source simply keeps the mark it had —
    which costs a re-read and never a dropped row.
    """
    if checkpoint is not None and isinstance(sink, SupportsCheckpoint):
        return sink.write_checkpoint(items, checkpoint)
    return sink.write(items)


async def _write_chunk(
    sink: ImportSink,
    items: Sequence[ImportItem],
    checkpoint: Checkpoint | None,
    *,
    source: str,
    poison: PoisonLedger | None,
    held: dict[str, HeldRecord],
    report: AdapterReport,
) -> WriteCounts:
    """Commit one chunk: retry the transient, isolate the poisonous, keep going.

    THREE FAILURE CLASSES, THREE ANSWERS, and conflating any two of them is a
    bug this pipeline has already paid for:

    * **Transient** — ``database is locked`` under a concurrent writer, an
      fsync that hiccuped. Retried, with capped exponential backoff and full
      jitter, then re-classified as one of the others.
    * **One bad record** — a malformed row, an observation whose session never
      arrived. Isolated: the chunk is retried item by item, the good rows land,
      the offender is set aside with an attempt count so it is neither lost nor
      retried forever. One bad row must not cost a 372k-row stream.
    * **A broken sink** — the disk is full, the schema is wrong, the store will
      not open. Indistinguishable from poison for ONE record, but not for all
      of them: if every item in the chunk fails individually, the sink is what
      is broken, so the original error is re-raised and the adapter reports it.
      That distinction is the whole reason the isolation pass counts successes.

    THE MARK IS NOT ADVANCED BY AN ISOLATED CHUNK. If anything was set aside,
    the checkpoint is dropped and this pass ends with the old mark — the next
    run re-reads the window, and a record that was merely unlucky gets another
    go from a run that can still advance.
    """
    last_error: Exception | None = None
    for attempt in range(CHUNK_RETRY_ATTEMPTS):
        try:
            return _sink_write(sink, items, checkpoint)
        except Exception as e:  # noqa: BLE001 -- classified below, never swallowed
            last_error = e
            if attempt < CHUNK_RETRY_ATTEMPTS - 1:
                delay = full_jitter(
                    attempt,
                    base=_seconds(CHUNK_RETRY_BASE),
                    cap=_seconds(CHUNK_RETRY_CAP),
                )
                await asyncio.sleep(delay.total_seconds())

    counts = WriteCounts()
    survived = 0
    # NOTHING IS RECORDED UNTIL THE PASS IS OVER. Whether a failure is "one bad
    # record" or "a broken sink" is not knowable per item — it is decided by
    # whether ANYTHING survived — so the verdict is collected first and written
    # only once it is known. Writing as we went would leave a chunk's worth of
    # perfectly good rows condemned in the holding table by an outage.
    failures: list[tuple[ImportItem, Exception]] = []
    for item in items:
        try:
            # No checkpoint on the isolation pass: a chunk that needed
            # isolating has not been fully stored, so nothing about it may
            # move the mark.
            counts = counts + _sink_write(sink, [item], None)
        except Exception as e:  # noqa: BLE001 -- per-record isolation boundary
            failures.append((item, e))
        else:
            survived += 1
            if poison is not None and item_key(item) in held:
                poison.release(source, item_key(item))
    if survived == 0 and last_error is not None:
        # Everything failed, so this is not a bad record — it is a bad sink.
        # The adapter is about to report the real fault; setting the whole
        # chunk aside would turn a total outage into a quiet "some records
        # were odd" line, which is the opposite of loud.
        raise last_error
    for item, error in failures:
        key = item_key(item)
        report.quarantined += 1
        report.errors.append(
            f"record {key!r} could not be written and was set aside "
            f"({type(error).__name__}: {error})"
        )
        if poison is not None:
            poison.hold(source, key, error, previous=held.get(key))
    return counts


def _seconds(value: float) -> timedelta:
    return timedelta(seconds=value)


def _final_checkpoint(
    report: AdapterReport,
    *,
    plan: SourcePlan | None,
    high: datetime | None,
    interrupted: bool,
    unplaceable: str | None,
    enabled: bool,
) -> Checkpoint | None:
    """Where this source's mark ends up, and — when it does not move — why.

    FOUR REASONS THE MARK STAYS PUT, all of which still record a successful
    pass (stamping the run time and clearing the failure counter), because a
    pass that ran and simply had nothing to advance to is not a failure:

    * the source declares no usable cursor (``CursorKind.NONE`` — TickTick),
      so there is nothing to advance and the report says FULL SCAN;
    * SIGTERM arrived, so the stream is incomplete and the maximum seen is not
      a high-water mark;
    * an item carried no cursor timestamp at all, so there is no way to say
      what "past it" means and advancing would skip it forever;
    * the sink declined rows (``skipped``), so the mark would describe records
      no store received.

    The cost of standing still is one re-read of the window, which the
    ``src_hash`` guard makes almost free. The cost of advancing wrongly is the
    records in between, permanently. That asymmetry decides every branch here.
    """
    if not enabled or plan is None:
        return None
    if not plan.cursor.is_incremental:
        return Checkpoint(source=report.name, cursor_value=None, rows=report.added)
    reason: str | None = None
    if interrupted:
        reason = (
            "shutdown requested mid-stream, so the watermark was left where it "
            "was; the next run re-reads this window and re-writing an unchanged "
            "row costs nothing"
        )
    elif unplaceable is not None:
        reason = (
            f"item {unplaceable!r} carried no timestamp on this source's cursor "
            f"({plan.cursor.field}), so the watermark could not be advanced "
            f"safely and this source full-scans again next run"
        )
    elif report.skipped:
        reason = (
            f"the sink declined {report.skipped} row(s), so the watermark would "
            f"describe records no store received"
        )
    if reason is not None:
        report.notes.append(f"{report.name}: {reason}")
        return Checkpoint(source=report.name, cursor_value=None, rows=report.added)
    report.advanced_to = high
    return Checkpoint(source=report.name, cursor_value=high, rows=report.added)


async def _run_one(
    adapter: ImportAdapter,
    sink: ImportSink,
    batch_size: int,
    *,
    plan: SourcePlan | None = None,
    watermarks: Watermarks | None = None,
    poison: PoisonLedger | None = None,
    stop: Callable[[], bool] | None = None,
    progress: RunProgress | None = None,
) -> AdapterReport:
    report = AdapterReport(name=adapter.name)
    tracker = progress or RunProgress()
    if plan is not None:
        report.window = plan.window_description
        report.cursor_kind = plan.cursor.kind
        if plan.skip_reason is not None:
            # RESTED, NOT RUN, AND SAID OUT LOUD. A source missing from the
            # output is indistinguishable from a source with nothing new, and
            # this is the case where that difference matters most: a source
            # backing off is a source nobody is watching.
            report.skipped_for_backoff = True
            report.notes.append(plan.skip_reason)
            tracker.skipped(adapter.name, plan.skip_reason)
            return report

    entry = tracker.begin(adapter.name, report.window or "(no watermark policy)")
    held = poison.held(adapter.name) if poison is not None else {}
    batch: list[ImportItem] = []
    high: datetime | None = None
    unplaceable: str | None = None
    stop_requested = False
    held_skipped = 0

    async def flush(checkpoint: Checkpoint | None = None) -> None:
        # Detach before writing: if the sink raises, the batch is already
        # out of the buffer, so the recovery flush below can't retry the
        # same doomed write forever.
        pending, batch[:] = list(batch), []
        if not pending and checkpoint is None:
            return
        counts = await _write_chunk(
            sink,
            pending,
            checkpoint,
            source=adapter.name,
            poison=poison,
            held=held,
            report=report,
        )
        report.added += counts.added
        report.updated += counts.updated
        report.skipped += counts.skipped
        report.unchanged += counts.unchanged
        # THE FALLBACK, AND ITS ORDER IS THE POINT. A sink that cannot
        # checkpoint has already committed this chunk, so recording the mark
        # now is the SAFE half of the two-write problem: data first, mark
        # second, and a crash in between costs a re-read rather than the
        # records. Without it a run through such a sink would never stamp its
        # run time or clear its failure counter, and a source that failed once
        # could rest forever.
        if (
            checkpoint is not None
            and watermarks is not None
            and not isinstance(sink, SupportsCheckpoint)
        ):
            watermarks.advance(
                adapter.name, checkpoint.cursor_value, rows=checkpoint.rows
            )
        tracker.chunk(
            entry,
            rows=len(pending),
            added=counts.added,
            updated=counts.updated,
            unchanged=counts.unchanged,
            quarantined=report.quarantined,
        )

    wrote_everything = False
    stream_failed = False
    try:
        async for item in adapter.get_data():
            key = item_key(item)
            if key in held:
                # Known bad, not yet due for another attempt. Skipping it is
                # what stops one poison record from being scrubbed and retried
                # on every tick forever; it is still counted, still reported,
                # and still in the holding table where a human can see it.
                #
                # A held SESSION row cascades: its observations then reference a
                # session the sink has never seen, so they fail the foreign-key
                # check and are set aside too. That is the correct outcome —
                # they genuinely cannot be written — and it is noisy rather than
                # wrong, so it is not special-cased.
                report.quarantined += 1
                held_skipped += 1
                continue
            at = item_cursor(item)
            if at is None:
                # An item that cannot be placed on the cursor makes the WHOLE
                # pass's mark unsafe: advancing past it would skip it forever,
                # and there is no way to tell where "past it" is. Refuse to
                # advance and say why. Costs one full re-read; the alternative
                # costs the record.
                unplaceable = unplaceable or key
            elif high is None or at > high:
                high = at
            batch.append(item)
            if len(batch) >= batch_size:
                await flush()
            # BETWEEN ITEMS, NEVER INSIDE A CHUNK'S TRANSACTION. Per item
            # rather than per chunk so a stop is answered promptly even when a
            # chunk is slow to fill; the buffered remainder is then committed by
            # the flush below, so the stop still lands on a transaction
            # boundary and nothing is left half-written.
            if stop is not None and stop():
                stop_requested = True
                break
        # END OF STREAM — the only flush that may carry the mark. Every earlier
        # chunk committed its rows and nothing else, because until the stream
        # is exhausted the maximum cursor seen is not a high-water mark: these
        # streams are not ordered by cursor, so a later chunk can legitimately
        # carry an earlier timestamp, and a mark advanced at chunk 3 would skip
        # it forever. See ``port.Checkpoint``.
        await flush(
            _final_checkpoint(
                report,
                plan=plan,
                high=high,
                interrupted=stop_requested,
                unplaceable=unplaceable,
                enabled=watermarks is not None,
            )
        )
        wrote_everything = not stop_requested
    except Exception as e:  # noqa: BLE001 -- isolation boundary, see below
        # PER-ADAPTER FAILURE ISOLATION. One source dying (expired token,
        # unreachable dir, locked DB) must not deny the other seven their
        # run. The exception is recorded with its type so the notification
        # and the CLI can say what broke — it is contained here, never
        # swallowed: ``report.ok`` goes False and stays False.
        # BaseException (CancelledError, KeyboardInterrupt) deliberately
        # propagates — those mean the whole run is being torn down.
        report.errors.append(f"{type(e).__name__}: {e}")
        stream_failed = True
        # The final flush may have already decided where the mark WOULD go
        # before raising. It did not get there, so the report must not say it
        # did — a summary claiming an advance that never committed is exactly
        # the kind of number that makes the next investigation start from a
        # false premise.
        report.advanced_to = None
        try:
            # Partial ingest beats total loss: whatever arrived before the
            # crash still gets written. NEVER with a checkpoint — a failed pass
            # must leave the mark exactly where it was, so the next run
            # re-reads the same window and nothing in it can be lost.
            await flush()
        except Exception as e2:  # noqa: BLE001
            report.errors.append(
                f"flush after failure: {type(e2).__name__}: {e2}"
            )

    report.interrupted = stop_requested
    if watermarks is not None and stream_failed:
        # Counted, so a source that keeps dying gets rested rather than
        # hammered every 30 minutes. Deliberately NOT counted for the per-file
        # errors ``drain_errors`` is about to add: those are by design (partial
        # ingest beats total loss) and a source that reports one unreadable PDF
        # per run is healthy, not flapping.
        try:
            watermarks.record_failure(adapter.name, report.errors[0])
        except Exception as e:  # noqa: BLE001 -- bookkeeping must not fail a run
            report.errors.append(
                f"ingest state could not be updated: {type(e).__name__}: {e}"
            )

    # The write barrier, and it is the ONE optional protocol that must not run
    # after a failure: it exists so an adapter's own state cannot get ahead of
    # the data it implies. A partial run leaves that state untouched and
    # re-derives it next time. See ``SupportsWriteBarrier``.
    #
    # ``report.skipped`` is half the condition, not decoration. "The stream
    # ended without raising" is NOT "the items were persisted": ``skipped`` is
    # the sink saying it declined to write, which ``WriteCounts`` explicitly
    # anticipates for a filtering or dry-run sink. Firing the barrier then
    # advances a baseline for records that reached no store at all — and for
    # TickTick that is permanent, because the Open API reports a completion
    # exactly once. Declining to fire costs one poll's inference; firing costs
    # the inference forever. The asymmetry decides it.
    report.holds_report_barrier = is_report_gating(adapter)
    persisted_everything = wrote_everything and report.skipped == 0
    if persisted_everything and isinstance(adapter, SupportsWriteBarrier):
        try:
            adapter.commit_after_write()
        except Exception as e:  # noqa: BLE001
            # The items ARE written, so this is not an ingest failure — but an
            # adapter whose state never advances silently stops noticing what
            # it exists to notice, and this is the only thing that says so.
            report.errors.append(
                f"commit_after_write failed: {type(e).__name__}: {e}"
            )

    # Optional protocols, checked structurally — an adapter opts in simply by
    # having the method. Both run after the stream ends, including when it
    # ended by raising: a source that logged five unreadable files before
    # dying must still surface all five.
    if isinstance(adapter, SupportsNonFatalErrors):
        try:
            report.errors.extend(_validated_errors(adapter.drain_errors()))
        except Exception as e:  # noqa: BLE001
            report.errors.append(f"drain_errors failed: {type(e).__name__}: {e}")
    if poison is not None and isinstance(adapter, SupportsPermanentFaults):
        # AFTER ``drain_errors`` AND NEVER BEFORE IT. Every fault names a line
        # that list must already contain, because what this does is move a line
        # from "error" to "note"; run the other way round and the line is
        # re-added by the drain and the run fails on a fault it just excused.
        try:
            verdict = poison.reconcile_faults(
                adapter.name, _validated_faults(adapter.drain_faults())
            )
        except Exception as e:  # noqa: BLE001 -- bookkeeping must not fail a run
            # LOUD, and in the direction that costs noise rather than silence:
            # a ledger that cannot be read, or an adapter whose faults cannot
            # be trusted, excuses nothing — every line this adapter reported
            # stays in ``errors`` and the run fails exactly as it did before
            # this mechanism existed.
            report.errors.append(
                f"permanent faults could not be reconciled, so nothing was "
                f"held quiet this run: {type(e).__name__}: {e}"
            )
        else:
            _apply_fault_verdict(report, verdict)
    if isinstance(adapter, SupportsInputFreshness):
        report.offers_input_freshness = True
        try:
            report.input_newest_at = _as_utc(adapter.input_freshness())
        except Exception as e:  # noqa: BLE001
            report.errors.append(f"input_freshness failed: {type(e).__name__}: {e}")
    if held_skipped:
        # LOUD, ONCE, WITH A COUNT. These records produce no per-record line of
        # their own — they were not attempted — so without this they would be
        # an invisible gap in the index that looks exactly like full coverage.
        # One line rather than one per record: a source with 300 bad rows must
        # not starve the notification budget of everything else in the run.
        report.errors.append(
            f"{held_skipped} record(s) were skipped: they failed to write on an "
            f"earlier run and are not due for another attempt. "
            f"`aggregator status` lists them"
        )
    tracker.end(
        entry,
        status=(
            "failed"
            if not report.ok
            else "interrupted"
            if report.interrupted
            else "finished"
        ),
        mark=(
            report.advanced_to.isoformat()
            if report.advanced_to is not None
            else "unchanged"
        ),
    )
    return report


def _apply_fault_verdict(report: AdapterReport, verdict: FaultVerdict) -> None:
    """Turn a known fault's error line into a note, and say what changed.

    THE ONE PLACE A LINE GOES QUIET, and it goes quiet by IDENTITY: only lines
    that a fault in ``verdict.known`` names are removed, and a fault is only
    ``known`` when its exact ``source + file + reason + record list`` hash is
    already in the ledger. No count threshold, no substring, no "the sessions
    source is noisy so mute it" — a different bad line in the same file hashes
    differently, is new, and fails the run.

    Every removed line reappears as a NOTE, which the CLI prints under its
    source and which changes no exit code. That is the whole bargain: the run
    stops re-failing over a file that has been broken since March, and the file
    is still named on every single run, plus in ``aggregator status``, plus in
    the ledger with the date it was first seen.
    """
    report.known_faults = list(verdict.known)
    report.new_faults = list(verdict.new)
    quiet = {fault.line for fault in verdict.known}
    report.errors = [line for line in report.errors if line not in quiet]
    for fault in verdict.known:
        since = (
            fault.first_seen_at.isoformat()
            if fault.first_seen_at is not None
            else "an earlier run"
        )
        report.notes.append(
            f"{report.name}: KNOWN BAD INPUT, {fault.count} record(s) still "
            f"missing from the index, first reported {since} and not repeated "
            f"as a failure since — {fault.line}"
        )
    for fault in verdict.released:
        report.notes.append(
            f"{report.name}: {fault.scope} parses again ({fault.count} "
            f"record(s) recovered); it is no longer held in the poison ledger"
        )


def _validated_faults(drained: object) -> list[PermanentFault]:
    """Check what ``drain_faults`` actually returned, at the boundary.

    Same reasoning as :func:`_validated_errors`, with more at stake: a fault is
    a REQUEST TO GO QUIET, so a malformed one must never be honoured. The
    protocol is ``runtime_checkable`` and therefore gates on method presence
    alone, so an adapter whose ``drain_faults`` returns strings, dicts or None
    reaches here looking perfectly conformant.

    Anything that is not a ``PermanentFault`` is DROPPED rather than coerced —
    dropping it costs the adapter its suppression and nothing else, so the line
    it was about stays in ``errors`` and keeps failing the run. Guessing at an
    identity instead would silence a line on the strength of a hash of
    something nobody designed. A non-iterable answer raises into the caller's
    handler, which reports it and excuses nothing.
    """
    if not isinstance(drained, Iterable) or isinstance(drained, str | bytes):
        raise TypeError(
            f"drain_faults returned {type(drained).__name__}, not "
            f"list[PermanentFault] (adapter contract); nothing was held quiet"
        )
    return [fault for fault in drained if isinstance(fault, PermanentFault)]


def _validated_errors(drained: object) -> list[str]:
    """Check what ``drain_errors`` actually returned, at the boundary.

    ``SupportsNonFatalErrors`` is ``runtime_checkable``, and that gates on
    method PRESENCE only — the return annotation is not enforced anywhere. An
    adapter returning a bare ``str`` therefore passed ``isinstance`` and then
    ``report.errors.extend("ticktick token expired")`` appended one entry PER
    CHARACTER: 22 entries reading 't', 'i', 'c', ... The count is what the CLI
    prints and what the notifier's summary counts, and the message — the only
    operator-facing diagnostic a non-fatal failure has — is destroyed.

    A ``str`` is kept whole (the message is the valuable part) and labelled, so
    the adapter bug is visible without costing the operator the diagnostic.
    Non-str entries are rendered rather than dropped for the same reason.
    Anything not iterable at all raises out of here into the caller's handler,
    which already reports it as ``drain_errors failed: ...``.
    """
    if isinstance(drained, str):
        return [
            f"drain_errors returned a bare str, not list[str] (adapter "
            f"contract); the message was: {drained}"
        ]
    return [e if isinstance(e, str) else f"{type(e).__name__}: {e}" for e in drained]


def _as_utc(value: datetime | None) -> datetime | None:
    """Stamp a naive input timestamp as UTC. Normalisation, not a guess.

    ``input_freshness`` is annotated ``datetime | None`` and a NAIVE datetime
    satisfies that annotation, so an adapter returning one is not a type error
    — it is a runtime error, and it lands nowhere near the adapter that caused
    it. Every consumer does aware-datetime arithmetic against ``now(UTC)``,
    and subtracting a naive datetime from an aware one raises TypeError. That
    exception escaped the staleness pass, which runs once for the WHOLE run
    after every adapter has finished: one source's naive timestamp took down
    the run report, the error listing for all nine sources, and the exit-3
    that tells the timer anything failed. Measured, not theoretical.

    UTC because it is what every other timestamp in this codebase means and
    what the whole comparison is against. The alternative — refusing the value
    — would let one adapter's tz sloppiness cost the run its freshness signal,
    which is the signal that exists to stop silent staleness in the first
    place.
    """
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=UTC)


def _staleness_line(
    name: str, entry: AdapterReport, *, max_age_days: int, moment: datetime
) -> str | None:
    """The one line this source's input earns right now, or None if it is fine.

    THE ONLY PLACE THIS SENTENCE IS SPELLED, for the same reason
    ``RunReport.errors_from`` is the only place an error line is: a marker is
    stamped only when a channel carried the line, and that comparison is
    WHOLE-LINE identity (``Delivery.accepted``). A second copy of the wording
    that drifted by one character would make the delivery unachievable — safe,
    but permanently loud with nothing a reader could point at.
    """
    # ``_as_utc`` again, not only in ``_run_one``: the callers are public and
    # can be handed a report assembled elsewhere. A TypeError here would escape
    # the whole-run staleness pass and take the report, every adapter's error
    # listing and the exit code with it.
    newest = _as_utc(entry.input_newest_at)
    if newest is None:
        return (
            f"{name}: no input found — this source imported "
            f"nothing, which looks identical to having nothing new. Drop "
            f"a fresh export where {name} looks for it."
        )
    age_days = (moment - newest).days
    if age_days > max_age_days:
        return (
            f"{name}: input is {age_days} days stale "
            f"(threshold {max_age_days}). Nothing on this machine "
            f"refreshes it — export a new one, or raise "
            f"--stale-after-days if that is the intended cadence."
        )
    return None


def staleness_warnings(
    report: RunReport,
    *,
    max_age_days: int,
    now: datetime | None = None,
) -> list[str]:
    """One line per source whose hand-refreshed input has gone stale.

    THE UNDEDUPLICATED VIEW: every source that is stale right now, whether or
    not it has already been reported. ``run_imports`` uses
    :func:`plan_staleness_report` instead, which is this filtered by what a
    human has already been told; this remains the answer to "what is stale?"
    for a caller that wants to look rather than to notify.

    Only adapters that offered ``input_freshness`` are considered — a live API
    or a continuously-synced directory has no export ritual to forget, and
    inventing an age for those would put a number in the report nobody can act
    on and train the operator to ignore the warnings that matter.

    A missing input warns too, and is the more dangerous case: the source
    imported nothing, every count is zero, and that is indistinguishable from
    a healthy run with nothing new.

    Deliberately NOT errors, so the exit code keeps meaning exactly one thing
    ("this run failed or dropped data"). A stale zip is not a failure —
    nothing broke, a human just has not exported lately — and it is fixed by a
    different action than a crashed adapter. That is also exactly why
    deduplicating it matters: the notification IS the channel, so a warning
    that repeats is a CRITICAL toast every 30 minutes for as long as the export
    stays old, and an alarm that always fires is one an operator learns to
    dismiss unread — which costs the next real failure its audience.
    """
    moment = _as_utc(now) or datetime.now(UTC)
    warnings: list[str] = []
    for name, entry in report.adapters.items():
        if not entry.offers_input_freshness:
            continue
        line = _staleness_line(name, entry, max_age_days=max_age_days, moment=moment)
        if line is not None:
            warnings.append(line)
    return warnings


def _episode(entry: AdapterReport, *, max_age_days: int) -> dict[str, object]:
    """WHICH staleness episode this source is in, as a comparable value.

    An episode is one continuous period during which a source's input is older
    than the threshold, and what ends it is a human dropping a fresh export.
    So the identity of an episode is THE INPUT ITSELF — the mtime the adapter
    reported — and not the warning text, which changes every day as the age
    ticks up, nor the run, which is what repeating once per run means.

    ``stale_after_days`` rides along because the operator can move the goalposts
    (see :func:`_silences`). ``None`` is the missing-input case, which no
    threshold makes more or less true.
    """
    newest = _as_utc(entry.input_newest_at)
    return {
        # "" for "there is no input at all", which is a real and distinct
        # episode: an archive that later APPEARS and is already old is a
        # different fact and gets said out loud again.
        "input_newest_at": "" if newest is None else newest.isoformat(),
        "stale_after_days": None if newest is None else max_age_days,
    }


def _same_episode(mark: object, episode: dict[str, object]) -> bool:
    """Is this marker still about the input in front of us?

    Identity is the input timestamp. A marker whose input differs is about an
    export that has since been replaced — the episode it recorded is over — so
    it neither suppresses nor survives the run. Anything that is not a marker
    at all (a hand-edited fragment, a null, a bare string) answers False, which
    is the loud direction.
    """
    return (
        isinstance(mark, dict)
        and mark.get("input_newest_at") == episode["input_newest_at"]
    )


def _silences(mark: object, episode: dict[str, object]) -> bool:
    """May this marker keep the warning quiet? Only for a threshold as loose.

    THE THRESHOLD RULE, stated: a marker suppresses only a warning raised at a
    threshold AT LEAST AS LOOSE as the one it was earned at. Lowering
    ``--stale-after-days`` is the operator saying the old cadence was too
    generous, and a source they were told about at 14 days is a source they
    have NOT been told about at 7 — the fact is stricter now — so it warns
    again and earns a marker at the new threshold. Raising it (or leaving it)
    reports nothing new: the same input, already heard about, judged by a
    kinder rule.

    A marker with no recorded threshold is the missing-input case and silences
    a missing input at any threshold — nothing about "no export exists" is a
    matter of degree. A marker whose recorded threshold is not an integer is
    garbled, and a garbled marker never buys silence.
    """
    if not isinstance(mark, dict) or not _same_episode(mark, episode):
        return False
    recorded = mark.get("stale_after_days")
    if recorded is None:
        return episode["stale_after_days"] is None
    if not isinstance(recorded, int) or isinstance(recorded, bool):
        return False
    threshold = episode["stale_after_days"]
    return isinstance(threshold, int) and threshold >= recorded


@dataclass(frozen=True)
class StalenessEpisodes:
    """What this run has to SAY about stale inputs, and the write that waits.

    ``warnings`` is the deduplicated list — the sources whose current episode no
    human has been told about yet. ``suppressed`` is the other half, the ones
    that were held back and the marker that held them, kept so a caller (and a
    test) can see that quiet is not the same as unnoticed.

    ``commit`` takes the run's :class:`~aggregator.imports.port.Delivery` and
    writes the markers for the lines a channel actually carried. Named rather
    than returned as a bare callable for the reason ``OpenTaskReconcile`` gives:
    a commit that runs at the wrong moment buys silence for something nobody
    heard, which is the whole defect class this mechanism lives in.
    """

    warnings: list[str]
    suppressed: dict[str, dict]
    commit: Callable[[Delivery], None]


def plan_staleness_report(
    report: RunReport,
    *,
    max_age_days: int,
    now: datetime | None = None,
    markers: IngestMarkers | None = None,
) -> StalenessEpisodes:
    """Say each stale source ONCE PER EPISODE, and hand back the receipt write.

    THE DEFECT THIS EXISTS FOR. ``staleness_warnings`` is correct and was
    reported on every run: ``substack``'s export is 31 days old and ``chatgpt``
    has never been ingested at all, so under the 30-minute timer that is a
    desktop notification every half hour, forever, until a human downloads a
    fresh export. It is the same alarm fatigue this branch spent four rounds
    eliminating for a vanished TickTick project — a permanently-red signal
    trains an operator to ignore the one that matters — and staleness simply
    never got the treatment.

    SAME SHAPE AS THE TICKTICK RECEIPT, deliberately, down to the two-phase
    split: the diff happens now (it needs the run), and the write that buys the
    silence waits for a human to have been TOLD. Delivery is
    ``port.Delivery`` and nothing else — the set of lines a channel declares it
    carried, read out of the payload it sent — so every way a report can miss
    its audience is already covered: no notifier configured, a hook that only
    logs, a hook that raised, a toast that truncated, an unwatched journal. Each
    of those leaves the marker unwritten and the source loud next run, which is
    the cheap direction.

    RE-ARMING IS THE POINT OF AN EPISODE. A marker survives only while it is
    still about the input in front of it (:func:`_same_episode`), so:

    * the export is refreshed and the source is fresh -> no warning, and the
      marker is DROPPED, because a suppressed state that is no longer true
      would make ``aggregator status`` lie about it;
    * the refreshed export later goes stale -> different mtime, no marker
      matches, and the source is loud again.

    PER SOURCE, NEVER A GLOBAL MUTE: markers are keyed by adapter name, so a
    second source going stale is reported while the first stays quiet.

    A missing or unreadable marker file resolves to "nothing is suppressed" —
    see ``ingest_state``. One toast too many is a cost; one alert too few is the
    failure.
    """
    store = markers if markers is not None else IngestMarkers()
    moment = _as_utc(now) or datetime.now(UTC)
    stored = store.load(STALE_INPUTS)

    # Only the sources this run can speak for. Every other key in the section is
    # carried through untouched: a partial run (a test, a future single-source
    # driver) must not prune markers for sources it never looked at.
    managed: list[str] = []
    episodes: dict[str, dict[str, object]] = {}
    warnings: list[str] = []
    suppressed: dict[str, dict] = {}
    pending: dict[str, tuple[str, dict]] = {}

    for name, entry in report.adapters.items():
        if not entry.offers_input_freshness:
            continue
        managed.append(name)
        line = _staleness_line(name, entry, max_age_days=max_age_days, moment=moment)
        if line is None:
            # Fresh. Not in ``episodes``, so any marker it still had is dropped
            # by the commit — this is the re-arm.
            continue
        episode = _episode(entry, max_age_days=max_age_days)
        episodes[name] = episode
        held = stored.get(name)
        if _silences(held, episode):
            suppressed[name] = held
            continue
        warnings.append(line)
        pending[name] = (line, {**episode, "first_reported": moment.isoformat()})

    def commit(delivered: Delivery) -> None:
        """Write the markers for the warnings a channel actually carried.

        READ-MODIFY-WRITE against what is on disk NOW, not a replay of the plan,
        because this can run twice for one run: once against what the notify
        hook declared and once against what the CLI printed to a terminal
        somebody was watching. Rebuilding from the plan alone, the second call
        would erase the marker the first one earned.

        Writes only when the section actually changes, so the steady state —
        every source fresh, or every stale one already reported — costs no write
        at all.
        """
        current = store.load(STALE_INPUTS)
        updated = {name: mark for name, mark in current.items() if name not in managed}
        for name, episode in episodes.items():
            held = current.get(name)
            if _same_episode(held, episode):
                updated[name] = held
        for name, (line, mark) in pending.items():
            if line in delivered.lines:
                updated[name] = mark
        if updated != current:
            store.save(STALE_INPUTS, updated)

    return StalenessEpisodes(warnings=warnings, suppressed=suppressed, commit=commit)


def commit_staleness_receipts(report: RunReport, delivered: Delivery) -> None:
    """Let this run record which staleness warnings actually reached a human.

    The staleness half of ``commit_report_barriers``, and PUBLIC for the same
    reason: ``ingest --all`` prints the run's warnings to a terminal AFTER
    ``run_imports`` has returned, and a person watching that terminal is as much
    an audience as a toast is. Calling it twice is safe — the second call reads
    the file the first one wrote and can only add to it — and forgetting to call
    it is the loud direction.

    A no-op for a caller with no staleness policy: nothing was evaluated, so
    there is nothing to suppress and, crucially, no marker is pruned on the
    strength of a run that never asked the question.

    A failed write is recorded rather than raised. The ingest succeeded and the
    warning was delivered; all that is lost is the silence, so the next run says
    the same thing once more. It is still reported, because a marker that
    silently never lands turns "reported once" back into "reported forever" and
    nothing else in the run would say so.
    """
    plan = report.stale_episodes
    if plan is None:
        return
    try:
        plan.commit(delivered)
    except Exception as e:  # noqa: BLE001 -- reported, never fatal to the ingest
        report.run_errors.append(
            f"staleness markers could not be written: {type(e).__name__}: {e}"
        )


async def run_imports(
    adapters: Iterable[ImportAdapter],
    sink: ImportSink,
    *,
    notify: NotifyHook = _no_notification,
    batch_size: int = DEFAULT_BATCH_SIZE,
    stale_after_days: int | None = None,
    now: datetime | None = None,
    markers: IngestMarkers | None = None,
    watermarks: Watermarks | None = None,
    poison: PoisonLedger | None = None,
    stop: Callable[[], bool] | None = None,
    progress: RunProgress | None = None,
) -> RunReport:
    """Drive every adapter concurrently and return the aggregated report.

    ``watermarks`` is what makes a run INCREMENTAL, and passing ``None`` is a
    full scan of every source — which is what this command did on every
    30-minute tick until 2026-08-16 and why it could never finish. It is a
    parameter rather than a global for the same reason ``markers`` is: the
    state belongs to a database a caller chose, and a test must be able to
    point it somewhere else.

    Concurrency is unchanged and deliberately so: the ~9 sources are
    independent and mostly network- or file-bound, so overlapping their
    acquisition is the single biggest wall-clock win available and costs
    nothing in correctness. The WRITE leg is where concurrency would hurt —
    SQLite allows one write transaction at a time — and it is already
    serialised by ``ImportSink`` being synchronous, so no coroutine can be
    suspended holding the write lock.

    ``stop`` is polled at chunk boundaries only. See :func:`graceful_shutdown`.

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

    A hook that actually reached a human RETURNS A ``Delivery`` NAMING WHAT IT
    CARRIED — ``Delivery.accepted(payload, report.errors)``, read out of the text
    it sent — and an adapter's ``SupportsReportBarrier`` fires only when that set
    covers every line the adapter reported. Returning nothing is the default and
    means nothing was delivered, which is what the no-op above returns and what a
    hook that merely logs returns. That is the whole reason the hook has a return
    type: a report barrier lets a source go QUIET about something, and a source
    may only go quiet because somebody heard THAT THING — never because nothing
    went wrong, and never because a different line got through in its place.

    ``stale_after_days`` turns the collected freshness values into
    ``report.warnings`` BEFORE ``notify`` fires, which is the whole point of
    evaluating them here rather than in the caller: a warning the notifier
    cannot see is stderr text on an exit-0 run, and no timer reads stderr.
    ``None`` skips the evaluation (a caller with no staleness policy), and
    skipping it touches no marker — a run that never asked whether an export
    was stale has no business re-arming or suppressing anything.

    THOSE WARNINGS ARE DEDUPLICATED PER EPISODE and go quiet once a human has
    been told, on exactly the ``Delivery`` the report barriers use. See
    :func:`plan_staleness_report`; ``markers`` is the file they live in, and is
    a parameter for the same reason ``TickTickSource`` takes a ``state_file``.
    """
    adapter_list: Sequence[ImportAdapter] = list(adapters)
    _refuse_duplicate_names(adapter_list)
    tracker = progress or RunProgress()
    plans = (
        {a.name: watermarks.plan(a.name, now=now) for a in adapter_list}
        if watermarks is not None
        else {}
    )
    # THE HEARTBEAT RUNS BESIDE THE WORK, NOT INSIDE IT, because the condition
    # it exists to report — a leg producing no chunks — is precisely the one in
    # which the chunk loop is not running. Two hours of silence is what started
    # this; a task on a timer is what makes it impossible.
    beat = asyncio.create_task(tracker.beat())
    try:
        results = await asyncio.gather(
            *(
                _run_one(
                    a,
                    sink,
                    batch_size,
                    plan=plans.get(a.name),
                    watermarks=watermarks,
                    poison=poison,
                    stop=stop,
                    progress=tracker,
                )
                for a in adapter_list
            )
        )
    finally:
        beat.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await beat
    report = RunReport(adapters={r.name: r for r in results})
    report.run_id = tracker.run_id
    report.interrupted = any(r.interrupted for r in results)
    if stale_after_days is not None:
        report.stale_episodes = plan_staleness_report(
            report, max_age_days=stale_after_days, now=now, markers=markers
        )
        report.warnings.extend(report.stale_episodes.warnings)
    # NOTHING DELIVERED UNTIL A CHANNEL SHOWS WHAT IT CARRIED. The initial value
    # is the answer for every run that has no channel — no notifier configured, a
    # hook that only logs, a hook that raised — and the only thing that changes
    # it is a hook returning a ``Delivery`` built from the payload it sent. See
    # ``port.Delivery`` for why this is a set of lines rather than a yes/no.
    delivered = Delivery()
    try:
        answer = notify(report)
    except Exception as e:  # noqa: BLE001
        # A missing notify-send must not cost us the report — the caller
        # still has to be able to print WHICH adapter failed. Recorded,
        # not swallowed: a notifier that cannot notify is itself a fault,
        # and on an otherwise-clean run this is the only thing that says so.
        report.run_errors.append(f"notify hook failed: {type(e).__name__}: {e}")
    else:
        # TYPE, not truthiness. The obvious hook — ``lambda report:
        # subprocess.run(...)`` — returns a truthy ``CompletedProcess`` even when
        # the notifier exited non-zero, and ``None`` is what a hook that only
        # logs returns. Neither is a statement about what reached anybody.
        if isinstance(answer, Delivery):
            delivered = answer
    commit_report_barriers(adapter_list, report, delivered)
    # The same declaration, spent on the other thing a run may go quiet about.
    # One notion of "delivered", one place it is computed; a staleness marker
    # that used its own idea of "we told them" would be the fifth round of a bug
    # this branch has already fixed four times.
    commit_staleness_receipts(report, delivered)
    # And the third thing, on the same declaration for the same reason. A
    # permanent fault is written into the ledger — which is what stops it being
    # re-reported — only for lines this channel actually carried.
    commit_fault_receipts(report, delivered, poison)
    return report


def commit_fault_receipts(
    report: RunReport, delivered: Delivery, poison: PoisonLedger | None
) -> None:
    """Record the permanent faults a channel actually carried, and only those.

    THE RECEIPT, AND WHY IT IS NOT OPTIONAL. Writing a fault into the ledger
    buys permanent silence for it: no future run fails on that identity again.
    Buying that silence on the strength of "the run reported it" is the exact
    defect ``port.Delivery`` documents four rounds of — reported is not heard.
    A run with no notifier, a hook that only logs, a hook that raised, a toast
    whose five-line budget cut this line: each of those leaves the ledger
    untouched and the fault loud again next run, which costs one duplicate line
    and never an unheard alert.

    PER LINE, not per adapter and not per run, unlike ``commit_report_barriers``
    — which fires an adapter's barrier only when EVERY line it reported got
    through. A fault is its own sentence and its own ledger row, so eight new
    faults against a five-line toast record five now and the remaining three on
    the next tick. Whole-adapter coverage here would mean a source that also
    happens to have an unrelated transient error can never record anything.

    Called with the ledger the run used, or ``None`` for a caller running
    without one — in which case nothing was ever held quiet and there is
    nothing to record. Calling it twice is safe: a recorded fault is dropped
    from ``new_faults``, so the second call (the CLI's, against a terminal a
    human was watching) can only ever add.

    A failed write is recorded rather than raised, like the staleness commit:
    the ingest succeeded and the operator was told; all that is lost is the
    silence, so next run says the same thing again.
    """
    if poison is None:
        return
    for name, entry in report.adapters.items():
        carried = [
            fault
            for fault in entry.new_faults
            if f"{name}: {fault.line}" in delivered.lines
        ]
        if not carried:
            continue
        try:
            for fault in carried:
                poison.record_fault(name, fault)
        except Exception as e:  # noqa: BLE001 -- reported, never fatal to the ingest
            report.run_errors.append(
                f"{name}: the permanent-fault ledger could not be written, so "
                f"this input will be reported again next run: "
                f"{type(e).__name__}: {e}"
            )
        else:
            entry.new_faults = [f for f in entry.new_faults if f not in carried]


def commit_report_barriers(
    adapters: Sequence[ImportAdapter], report: RunReport, delivered: Delivery
) -> None:
    """Let adapters record that THEIR OWN report actually reached somebody.

    PER ADAPTER, AGAINST WHAT THE CHANNEL CARRIED. An adapter's receipt
    suppresses a repeat of a line that adapter reported, so it may only be
    stamped when every line that adapter reported this run is in ``delivered``.
    Round 7 is what the granularity is for: the toast carried
    ``report.errors[:5]``, delivery was declared for the whole run, and five
    unrelated failures elsewhere bought permanent silence for a line that never
    left the process. ``Delivery.covers`` also refuses an adapter that reported
    nothing — see there for why that emptiness is not vacuously true.

    Called UNCONDITIONALLY, with an empty ``Delivery`` when nothing was heard,
    rather than behind an ``if delivered:`` at the call site. The gate lives in
    one place, inside the loop, where it is the same expression for every
    channel; an ``if`` at each call site is the shape the first four rounds of
    this bug all had.

    PUBLIC, because the CLI has a channel the runner cannot see: ``ingest --all``
    prints the run's errors to a terminal AFTER ``run_imports`` has returned, and
    a person watching that terminal is as much an audience as a toast is. Calling
    it a second time is safe — the adapters' pending receipts are cleared when
    they fire — and forgetting to call it is the loud direction, which is what
    separates this from the four rounds where forgetting bought silence.

    Not gated on ``report.ok``: a run that failed and said so is exactly the
    shape that must be allowed to go quiet, or the alarm repeats every 30
    minutes and the operator learns to dismiss it unread.

    A barrier that raises is recorded rather than fatal. The records are
    written, the report is delivered; all that is lost is the suppression, so
    the next run reports the same thing once more. That is the harmless
    direction, and it is still not silent.
    """
    for adapter in adapters:
        if not is_report_gating(adapter):
            continue
        if not delivered.covers(report.errors_from(adapter.name)):
            continue
        try:
            adapter.commit_after_report()
        except Exception as e:  # noqa: BLE001 -- one adapter must not stop the rest
            report.run_errors.append(
                f"{adapter.name}: commit_after_report failed: {type(e).__name__}: {e}"
            )


@contextmanager
def graceful_shutdown(
    signals: Sequence[int] = (signal.SIGTERM, signal.SIGINT),
) -> Iterator[Callable[[], bool]]:
    """Turn SIGTERM into "stop at the next chunk boundary", not "die".

    THE SHAPE THE UNIT NEEDS. ``TimeoutStartSec`` fires SIGTERM at a fixed wall
    clock; before this branch that killed a run at ~44% and threw all of it
    away. Now the flag is set, the chunk in flight finishes and commits, every
    earlier chunk is already durable, and the process exits cleanly — so the
    worst a timeout can cost is one chunk of re-work, and the next run picks up
    from a window it can walk almost for free.

    NO WORK IN THE HANDLER. It runs at an arbitrary point, possibly mid-
    transaction; all it may do is set a flag that the chunk loop reads at a
    boundary it chose. ``TimeoutStopSec`` then only has to exceed one chunk's
    duration — at 500 rows that is well under a second, against the unit's
    90 seconds.

    ``add_signal_handler`` rather than ``signal.signal`` because it wakes the
    event loop; the ``signal.signal`` fallback covers a non-Linux loop, and
    both degrade to "no handler" rather than raising, because a caller that
    cannot install one (a worker thread, a nested loop) must still be able to
    run an ingest.
    """
    stopped = False

    def request_stop(*_: object) -> None:
        nonlocal stopped
        stopped = True

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # pragma: no cover - called outside a loop
        loop = None
    installed: list[int] = []
    previous: dict[int, object] = {}
    for sig in signals:
        installed_here = False
        if loop is not None:
            try:
                loop.add_signal_handler(sig, request_stop)
            except (NotImplementedError, RuntimeError, ValueError):
                installed_here = False
            else:
                installed.append(sig)
                installed_here = True
        if not installed_here:
            try:
                previous[sig] = signal.signal(sig, request_stop)
            except (OSError, ValueError):  # pragma: no cover - not main thread
                continue
    try:
        yield lambda: stopped
    finally:
        for sig in installed:
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                loop.remove_signal_handler(sig)  # type: ignore[union-attr]
        for sig, handler in previous.items():
            with contextlib.suppress(OSError, ValueError, TypeError):
                signal.signal(sig, handler)  # type: ignore[arg-type]


def _refuse_duplicate_names(adapters: Sequence[ImportAdapter]) -> None:
    """The report is keyed by adapter name, so a collision would silently
    drop one source's entire outcome. Refuse before any work is done."""
    seen: set[str] = set()
    for a in adapters:
        if a.name in seen:
            raise ValueError(f"duplicate adapter name: {a.name!r}")
        seen.add(a.name)


__all__ = [
    "CHUNK_RETRY_ATTEMPTS",
    "DEFAULT_BATCH_SIZE",
    "AdapterReport",
    # Re-exported: a notify hook is written against ``run_imports``, and the
    # value it has to return to unlock a report barrier should not require
    # finding a second module first.
    "Delivery",
    "NotifyHook",
    "RunReport",
    "StalenessEpisodes",
    # Public because the CLI declares a channel the runner never sees: the
    # errors and warnings it prints to a terminal a human is watching, AFTER
    # run_imports has returned. See ``commit_report_barriers``.
    "commit_fault_receipts",
    "commit_report_barriers",
    "commit_staleness_receipts",
    # The CLI installs this around ``run_imports`` — the process owns its
    # signals, library code merely reads the flag.
    "graceful_shutdown",
    "item_cursor",
    "item_key",
    "plan_staleness_report",
    "run_imports",
    "staleness_warnings",
]
