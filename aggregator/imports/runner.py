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
from datetime import UTC, datetime

from aggregator.imports.ingest_state import STALE_INPUTS, IngestMarkers
from aggregator.imports.port import (
    Delivery,
    ImportAdapter,
    ImportItem,
    ImportSink,
    SupportsInputFreshness,
    SupportsNonFatalErrors,
    SupportsWriteBarrier,
    is_report_gating,
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

    @property
    def added(self) -> int:
        return sum(a.added for a in self.adapters.values())

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
        """
        return [
            line
            for name, a in self.adapters.items()
            if a.holds_report_barrier
            for line in self.errors_from(name)
        ]

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

    wrote_everything = False
    try:
        async for item in adapter.get_data():
            batch.append(item)
            if len(batch) >= batch_size:
                flush()
        flush()
        wrote_everything = True
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
    if isinstance(adapter, SupportsInputFreshness):
        report.offers_input_freshness = True
        try:
            report.input_newest_at = _as_utc(adapter.input_freshness())
        except Exception as e:  # noqa: BLE001
            report.errors.append(f"input_freshness failed: {type(e).__name__}: {e}")
    return report


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
    results = await asyncio.gather(
        *(_run_one(a, sink, batch_size) for a in adapter_list)
    )
    report = RunReport(adapters={r.name: r for r in results})
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
    return report


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
    "commit_report_barriers",
    "commit_staleness_receipts",
    "plan_staleness_report",
    "run_imports",
    "staleness_warnings",
]
