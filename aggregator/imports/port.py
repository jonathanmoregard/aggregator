"""The import port: one structural seam every source is adapted onto.

Ports and adapters, the pythonic way — ``typing.Protocol`` for structural
typing, no ABCs, no factories, no DI container, no entry-point registry. An
adapter is any object with a ``name`` and a ``get_data``; conformance is
checked by shape, not by inheritance or registration.

``get_data`` returns an ``AsyncIterator``, i.e. implementations are
``async def get_data(self)`` containing ``yield``. Deliberately NOT a list:
the sessions source alone emits ~359k observations and materialising that
per adapter would hold the whole index in memory while the runner writes.
Streaming also lets the runner flush in batches, so a crash mid-source keeps
whatever was already written.

Naming: snake_case ``get_data``, not ``getData`` — PEP 8 wins in Python,
and the user licensed "the pythonic way of your chosen scripting language".

No ``since`` parameter on the port. Sources that support incremental
windows take ``since`` at construction time (the adapter closes over it), so
the port stays the single-verb interface it is meant to be. Adding a
parameter here would push per-source acquisition knobs into the seam every
future source has to implement.

THE ORDER A CALLER MUST DRIVE THESE IN
--------------------------------------
There are five protocols here and each one documents its own timing, which
between them did NOT add up to a stated sequence — a writing caller had to
reconstruct it from ``runner._run_one``. It is:

  1. ``ImportAdapter.get_data``      stream, flushing to the sink in batches
  2. ``SupportsWriteBarrier``        ``commit_after_write`` — ONLY if the
                                     stream ended without raising AND the sink
                                     skipped nothing
  3. ``SupportsNonFatalErrors``      ``drain_errors`` — always, INCLUDING after
                                     a raise
  3b. ``SupportsPermanentFaults``    ``drain_faults`` — always, and AFTER
                                     ``drain_errors``, because every fault it
                                     returns names a line that list must
                                     already contain
  4. ``SupportsInputFreshness``      ``input_freshness`` — always
  --- every adapter has now finished; the run's report is assembled ---
  5. some of the report reaches a human, or none of it does (:class:`Delivery`)
  6. ``SupportsReportBarrier``       ``commit_after_report`` — ONLY for an
                                     adapter that DECLARED ``gates_report``
                                     (:func:`is_report_gating`) and whose EVERY
                                     reported line is in what step 5 declared
                                     delivered
  7. the run's own staleness markers — ``runner.commit_staleness_receipts``,
                                     against the SAME :class:`Delivery`. Not an
                                     adapter protocol: the warning belongs to
                                     the run (see ``SupportsInputFreshness``),
                                     so no adapter has to implement or call it.
  7b. the run's own poison ledger  — ``runner.commit_fault_receipts``, against
                                     the same :class:`Delivery` again, for the
                                     same reason: a permanent fault may only go
                                     quiet once a channel carried the line that
                                     announced it.

Steps 1-4 are matched STRUCTURALLY: having the method is opting in, because
over-inclusion is free or safe for each of them. Step 6 is the exception and is
opted into by name — see :func:`is_report_gating` for what its over-inclusion
cost. Steps 3 and 4 are order-independent of each other and of 2; they collect,
they decide nothing. The load-bearing edge is 2 BEFORE 6, and the gap between
them is the point: the write barrier answers "did the records land?" and can
be answered immediately, while the report barrier answers "was a human told?",
which is not knowable until every adapter has finished and a channel has
reported success.

WHAT GOING WRONG COSTS, since the two fail in opposite directions:

* 6 before 2 — measured on the one adapter that implements both: the receipt
  lands, and the advance behind it is then REFUSED, because writing the receipt
  moved the file the advance was going to compare-and-swap against. It raises,
  the caller records it, the run exits 3, and the next poll re-derives the
  advance. Loud and recoverable rather than silent — but it costs a poll, so
  the order is not a matter of taste.
* 2 without 6 — the normal state of any run nobody was told about. Cost: one
  more report next run. This is the DEFAULT and it is deliberate.
* 6 without 2 — a run whose write failed but whose report was heard. Legal and
  intended: the human did hear it, and the baseline is re-derived next run
  identically. Only the report is suppressed, never the retention.
* NEVER calling 2 — the one unbounded failure. The adapter's state freezes, so
  a task created after the freeze is never in the baseline and its later
  disappearance is invisible to every future poll. "Safe to skip" applies to
  one run, never to the contract.

``tests/imports/test_write_barrier_contract.py`` drives every write path in
this repo against a probe adapter, and ``tests/test_delivery_contract.py``
pins step 5-6 end to end, so a new call site that gets this wrong fails there
rather than in production six months later.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from aggregator.sources.base import (
    ObservationRow,
    PermanentFault,
    Record,
    SessionRow,
)

# The store has two intentionally-distinct write paths and the port carries
# both rather than collapsing them (see ``core/store.py`` module docstring —
# a PR is not naturally a session):
#
#   * ``Record``                        -> ``Store.upsert`` (records table,
#                                          ON CONFLICT(stable_id) DO UPDATE)
#   * ``SessionRow`` | ``ObservationRow`` -> ``Store.upsert_entities``
#
# Sinks dispatch on the concrete type; adapters never have to know which
# table they land in.
ImportItem = Record | SessionRow | ObservationRow


@runtime_checkable
class ImportAdapter(Protocol):
    """What the runner needs from a source. Nothing else.

    ``runtime_checkable`` so tests (and a future ``status`` surface) can
    duck-type a registry entry with ``isinstance``. Note that
    ``issubclass`` is not available for protocols carrying non-method
    members like ``name`` — that is a CPython restriction, not a design
    intent.

    STREAM ORDER MATTERS for the entity shapes: yield a ``SessionRow`` BEFORE
    any ``ObservationRow`` that names it. ``observations.session_id`` is a real
    foreign key with ``PRAGMA foreign_keys`` ON, and the runner flushes the
    stream to the sink in batches — the sink can reorder within one batch and
    nowhere else, so an observation that precedes its session in the stream
    lands in an earlier batch and aborts the adapter. Small tests never see it
    (one batch, reordered, green); real volume always does. The sink refuses
    such a batch with a message naming the offending pair.
    """

    name: str

    def get_data(self) -> AsyncIterator[ImportItem]: ...


@dataclass(frozen=True)
class WriteCounts:
    """What a write actually did. Addable so the runner can fold batches.

    ``added`` — the item's primary key was not in the store before.
    ``updated`` — it was, and the row was reconciled against the store.
    ``skipped`` — the sink declined to write it (unknown shape, filtered).
    Load-bearing beyond the summary: a nonzero ``skipped`` withholds the
    ``SupportsWriteBarrier`` call, because an adapter must not advance state
    that implies rows the sink says it did not write.

    ``unchanged`` — a SUB-COUNT OF ``updated``, not a fourth bucket: the row
    was already stored byte-identically, so it cost no scrub, no page write and
    no rowid. Deliberately not folded into ``skipped``, which means "the sink
    declined" and gates the write barrier; an unchanged row was fully
    reconciled and must not withhold anything. ``added + updated == len(items)``
    therefore still holds, which is what keeps every existing reading of these
    numbers correct.

    Why it is worth a field at all: it is the direct evidence that a re-run
    costs what a re-run should cost. A run reporting ``updated=372450
    unchanged=372450`` did nothing expensive; the same line without the second
    number is indistinguishable from the doom loop, which also reported
    ~372k updates every 30 minutes.

    These have to come back FROM the write. ``cli.py`` reports
    ``added=len(records) updated=0`` for every run, so its summary is the
    same three numbers whatever happened; that bug is not repeated here.
    """

    added: int = 0
    updated: int = 0
    skipped: int = 0
    unchanged: int = 0

    def __add__(self, other: WriteCounts) -> WriteCounts:
        return WriteCounts(
            added=self.added + other.added,
            updated=self.updated + other.updated,
            skipped=self.skipped + other.skipped,
            unchanged=self.unchanged + other.unchanged,
        )


@runtime_checkable
class ImportSink(Protocol):
    """Where the runner puts what an adapter yields.

    Synchronous on purpose. The real sink is SQLite, whose writes are best
    serialised anyway, and a sync call cannot be suspended mid-way by the
    event loop — so concurrent adapters can share one sink without a lock
    and without half-written batches interleaving.
    """

    def write(self, items: Sequence[ImportItem]) -> WriteCounts: ...


@dataclass(frozen=True)
class Checkpoint:
    """One source's high-water mark, handed to the sink WITH the chunk it describes.

    THE UNIT OF LOSS IS THE UNIT OF TRANSACTION. A chunk's rows and the mark
    that says "everything up to here is stored" have to land together or a
    crash between them means one of two things, and only one of them is
    survivable: mark-first leaves the mark ahead of unprocessed records, which
    is silent permanent loss that nothing ever reports; data-first merely
    re-reads, which an idempotent apply makes free. Committing both in one
    transaction removes the choice.

    ``cursor_value`` is ``None`` for a pass that emitted nothing, and that is
    the normal state of a quiet source rather than an error — it stamps the run
    time and clears the failure counter WITHOUT writing NULL over a live mark.

    WHY THE MARK ONLY EVER RIDES THE FINAL CHUNK. Nothing in this pipeline
    yields items in cursor order: the sessions source walks
    ``~/.claude/projects`` in path order, dropbox walks a directory tree, and a
    chat export is grouped by conversation. So the maximum timestamp seen after
    three chunks says nothing about the fourth, and advancing the mark to it
    mid-stream would skip every record still to come below that value —
    permanently. A mid-stream checkpoint is only safe for a stream that DECLARES
    itself sorted by its cursor, which none of these are; the honest version of
    "checkpoint often" here is that every chunk's DATA is committed as it goes
    (so a kill loses at most one chunk of work) while the mark waits for the
    end of the stream, and the ``src_hash`` guard is what makes the re-read
    after an interrupted run cost almost nothing.
    """

    source: str
    cursor_value: datetime | None
    rows: int = 0


@runtime_checkable
class SupportsCheckpoint(Protocol):
    """Optional: a sink that can store a chunk and its watermark ATOMICALLY.

    Optional, and matched structurally like the other collect-only protocols,
    because over-inclusion is harmless: a sink that cannot checkpoint (a
    counting stub, a dry-run sink, a test double) simply gets ``write`` and the
    source keeps whatever mark it already had, which costs a re-read and never
    a dropped row. The real sink — ``store_sink.StoreSink`` — implements it,
    which is what makes the shipped pipeline incremental.

    Deliberately a SECOND method rather than a keyword on ``write``: ``write``
    is the one verb every sink in this repo and its tests implements, and
    growing its signature would make the checkpoint something every sink has to
    know about in order to keep not caring about it.
    """

    def write_checkpoint(
        self, items: Sequence[ImportItem], checkpoint: Checkpoint
    ) -> WriteCounts: ...


@runtime_checkable
class SupportsNonFatalErrors(Protocol):
    """Optional: an adapter that survives per-item failures reports them here.

    Existing sources append per-file problems to an ``errors`` list and keep
    going (partial ingest beats total loss). That policy must not silently
    lose its output on the new path: the session constraint is that a run
    ending with a non-empty ``errors`` list still notifies. The runner drains
    this after the stream ends — including when it ended by raising — and
    folds the result into the adapter's report.
    """

    def drain_errors(self) -> list[str]: ...


@runtime_checkable
class SupportsPermanentFaults(Protocol):
    """Optional: which of this adapter's errors will NEVER fix themselves.

    THE SUBSET OF ``drain_errors`` THAT MAY GO QUIET, and the split is the whole
    point. ``SupportsNonFatalErrors`` carries everything a source survived: a
    locked database, an expired token, an unreadable directory AND two lines of
    JSONL that no parser will ever accept. The first three are transient — the
    next run might succeed, so they must be reported until somebody fixes them.
    The last one is permanent, and reporting it as a fresh failure on every
    30-minute tick is a permanently-red alarm, which is the alarm an operator
    learns to dismiss unread. On 2026-08-16 that was four runs in six hours, the
    identical eight errors, four CRITICAL toasts, none of them new information.

    So a fault is DECLARED here, one at a time, with an identity the ledger can
    remember (``sources.base.PermanentFault``); the runner reports a never-seen
    identity loudly exactly once and quietly thereafter, and everything NOT
    declared here stays loud on every run forever. Forgetting to declare is
    therefore the safe direction — it costs noise, never silence — which is
    what makes structural matching affordable on this protocol where
    ``SupportsReportBarrier`` had to be an explicit opt-in: over-inclusion buys
    an adapter nothing, because the declaration is per FAULT and an adapter
    with none returns an empty list.

    Drained AFTER the stream ends, alongside ``drain_errors``, and every fault
    it returns must carry the exact ``line`` its ``drain_errors`` reported — the
    runner moves THAT line out of the run's errors and nothing else.
    """

    def drain_faults(self) -> list[PermanentFault]: ...


@runtime_checkable
class SupportsWriteBarrier(Protocol):
    """Optional: an adapter with state it may only advance once data landed.

    Some acquisition is DESTRUCTIVE OF ITS OWN EVIDENCE. TickTick's Open API
    serves open tasks only, so a completion appears exactly once — as a
    disappearance between two polls — and the poll that notices it also
    overwrites the baseline that made noticing possible. Advancing that
    baseline before the inferred records are in the store turns any later sink
    failure into permanent loss.

    So the adapter keeps the advance pending and the runner calls this once
    the stream has been PERSISTED: after the final flush, only when nothing
    raised, and only when the sink reported nothing skipped. A partial run
    leaves the state untouched and re-derives it next time, which is why the
    barrier is not called on the failure path.

    "Nothing skipped" is not belt-and-braces. ``WriteCounts.skipped`` is the
    sink saying it declined to write — a dry-run or filtering sink, which the
    field exists for — and that stream also ends cleanly. Gating on the stream
    alone let such a sink advance a baseline for records no store ever
    received.

    Not a transaction and not a rollback — the sink's writes are already
    committed by then. It is the narrower guarantee that the adapter's own
    state cannot get ahead of them.

    A WRITING CALLER MUST CALL IT. Skipping one call is cheap — the next run
    re-derives the same advance — but a caller that never calls it freezes the
    adapter's state, and for a source whose acquisition destroys its own
    evidence that is unbounded loss, not one run's worth: TickTick's baseline
    stops growing, so a task created after the freeze is never in it and its
    later disappearance is invisible to every future poll. "Safe to skip" is
    about one run, never about the contract.

    The rules are asserted, not just described, in
    ``tests/imports/test_write_barrier_contract.py``: every write path in this
    repo is driven against a probe adapter that implements nothing but this
    protocol, so a source that grows a barrier inherits the guarantees and a
    regression in any call site fails there rather than in production.
    """

    def commit_after_write(self) -> None: ...


def _carried_whole(carried: Sequence[str], wanted: Sequence[str]) -> bool:
    """Did ``carried`` contain ``wanted`` as a run of COMPLETE lines?

    The identity test behind :meth:`Delivery.accepted`, factored out because it
    is the entire difference between a receipt that means something and one that
    a longer sentence elsewhere in the payload can forge. ``wanted`` is normally
    one line and this is then set membership spelled as a scan; it is a scan
    because a reported line may legitimately contain newlines (exception text),
    and such a line is delivered only when the channel carried its lines
    adjacent and in order — never assembled out of pieces of other reports.
    """
    if not wanted:
        return False
    span = len(wanted)
    return any(
        list(carried[i : i + span]) == list(wanted)
        for i in range(len(carried) - span + 1)
    )


@dataclass(frozen=True)
class Delivery:
    """WHICH of this run's reported lines reached a human. Never merely THAT.

    A notify hook returns this, and it is a SET rather than a yes/no because
    four rounds of this bug were all the same mismatch: delivery was asserted
    at a coarser grain than suppression, and something in between dropped the
    one line that mattered.

      R4: stamped when the report was merely EMITTED.
      R5: stamped when ``notify`` RETURNED WITHOUT RAISING.
      R6: ...and the shipped default notifier does nothing, which returns
          without raising — so runs with no channel at all stamped receipts.
      R7: ``notify`` returned a run-scoped ``DELIVERED`` while the payload it
          actually sent was ``report.errors[:5]``. Five failures from other
          sources pushed TickTick's uncovered-project line out of the toast,
          and its receipt — which suppresses THAT LINE and nothing else — was
          stamped for a sentence no human ever saw.
      R8: the set was right and its membership test was ``line in payload``,
          i.e. SUBSTRING. A line was then delivered by any longer line that
          contained it, so the same "stamped for something never sent" arrived
          by a fifth route. See :meth:`accepted`.

    THE UNIT OF DELIVERY IS NOW THE UNIT OF SUPPRESSION. A receipt suppresses
    one adapter's repeat of one report, so delivery is tracked per reported
    line and ``runner.commit_report_barriers`` fires an adapter's barrier only
    when EVERY line that adapter reported this run is in here.

    DERIVED FROM THE PAYLOAD, NOT ASSERTED ALONGSIDE IT — see :meth:`accepted`.
    There is deliberately no ``Delivery.DELIVERED`` constant any more: no value
    in this module means "all of it got through" without naming what "it" was.
    That is what makes a fifth layer structurally different from the first four.
    Every one of those was an OMISSION at a call site (a check nobody wrote, a
    default nobody considered, a slice nobody re-read), and an omission now
    fails safe: truncate the payload, reformat it, drop a line, send nothing at
    all, and the delivered set shrinks by itself because it is computed from the
    bytes handed over. Over-claiming is no longer reachable by forgetting
    something; it takes affirmatively passing a payload that was never sent.

    THE FAILURE DIRECTION IS LOUD. Lines that did not make it are reported
    again next run, and the next, until a channel carries them. Reporting twice
    costs an operator a duplicate line; not reporting costs them the only alert
    the mechanism exists to raise, permanently.
    """

    # Empty is the default and the answer for every run with no channel: no
    # notifier configured, a hook that only logs, a hook that raised, a hook
    # that returned some stray value the runner refused.
    lines: frozenset[str] = frozenset()

    @classmethod
    def accepted(cls, payload: str, reported: Iterable[str]) -> Delivery:
        """The lines of ``reported`` that ``payload`` carried AS WHOLE LINES.

        The only constructor a channel should use, and the reason a truncating
        channel cannot lie: it hands over the text it sent and this reads the
        answer out of it. ``_desktop_notification`` sends at most five error
        lines; the sixth is not in ``payload``, so it is not in the result, so
        the adapter that reported it does not go quiet. No check to forget.

        WHOLE-LINE IDENTITY, NOT SUBSTRING. Round 8: this asked ``line in
        payload``, which is substring containment, so a line was "delivered" by
        any longer line that happened to contain it. One adapter reporting
        ``box: file unreadable`` and another reporting ``dropbox: file
        unreadable`` is the whole bug — send only the second and the first is
        marked delivered, and its receipt (which is the thing that buys silence)
        is stamped for a sentence no human ever saw. Adapter names are not
        validated against being suffixes of each other and two lines from ONE
        adapter need no such coincidence at all: ``A`` and ``A plus detail``
        does it. So ``payload`` is split into the lines the channel actually
        took and compared as complete strings.

        THE NEIGHBOURS, all decided in the fail-safe direction:

        * ``splitlines`` rather than ``split("\\n")``, so a CRLF channel and a
          payload with a trailing newline both answer correctly. Getting these
          wrong is merely loud, but loud-forever is the round-4 alarm fatigue
          this whole mechanism is trying not to cause.
        * Trailing or leading whitespace is now a MISMATCH. A channel that pads
          or re-indents is a channel that changed the text, and this class only
          ever gets to be honest by refusing to guess how much change is still
          "the same line". No shipped channel pads — both of them join the raw
          report lines with ``\\n`` — so this costs nothing today and stays loud
          rather than silent if one ever starts.
        * A reported line that itself spans several lines is matched as a
          CONTIGUOUS BLOCK. Exception text with an embedded newline is real
          (``f"{type(e).__name__}: {e}"`` over a subprocess error), the channel
          does carry it verbatim, and refusing it would keep an adapter loud
          forever over a line the human demonstrably read.
        * DUPLICATE IDENTICAL LINES COUNT AS DELIVERED FOR EVERY OCCURRENCE.
          This is a set of TEXT, and a receipt suppresses a repeat of that text;
          a human who read the sentence once has read it, so a second identical
          copy in the report cannot be a sentence they missed. Fail-safe because
          it can never silence anything unread — the only thing it over-covers
          is a verbatim duplicate of a line that WAS on screen. (Counting
          multiplicity instead would make an adapter that stuttered stay loud
          forever with nothing an operator could do about it, and adapter-name
          prefixing means two different adapters can never render the same line
          anyway.)

        Blank and whitespace-only report lines are dropped: they carry nothing
        to a human, so nothing they might gate should go quiet on their account.
        """
        carried = payload.splitlines()
        return cls(
            frozenset(
                line
                for line in reported
                if line.strip() and _carried_whole(carried, line.splitlines())
            )
        )

    def covers(self, reported: Sequence[str]) -> bool:
        """Did every one of ``reported`` reach a human?

        THE EMPTINESS CHECK IS IN HERE, not at the call sites, because "all of
        an empty list was delivered" is vacuously true and that is precisely the
        shape of reasoning this class exists to refuse — R4 through R6 were each
        a version of "nothing went wrong, so somebody must have heard". An
        adapter that reported nothing has nothing to suppress, so it has no
        business stamping a receipt on the strength of a channel that carried
        none of its words.
        """
        return bool(reported) and all(line in self.lines for line in reported)

    def __or__(self, other: Delivery) -> Delivery:
        """Two channels on one run — a terminal somebody watched AND a toast.

        Union, because delivery is a property of the line and the human, not of
        the channel: a line that made it into either one was heard.
        """
        return Delivery(self.lines | other.lines)


class SupportsReportBarrier(Protocol):
    """Optional: an adapter that may only go quiet once a human was TOLD.

    DELIBERATELY NOT ``runtime_checkable`` — see :func:`is_report_gating`. It is
    the one protocol here whose over-inclusion has a cost, and ``isinstance``
    against a runtime-checkable Protocol tests METHOD PRESENCE ONLY.

    The second barrier, and it exists because the first one fires too early to
    answer a different question. ``SupportsWriteBarrier`` asks "did the records
    land?"; this asks "did the run's report reach anybody?" — and the answer to
    that is not known until every adapter has finished and ``notify`` has run.

    TickTick is again the case. A baseline task whose project the poll could not
    cover is retained and REPORTED, and the report is deliberately made once per
    disappearance rather than once per poll (an error on every 30-minute tick is
    an alarm an operator learns to ignore, which costs the next real failure its
    audience). The suppression is a receipt written into the baseline — and
    written by the write barrier it recorded "we tried to tell them", not "they
    were told": a notify hook that could not run left the receipt behind anyway,
    so every later poll stayed quiet about a disappearance nobody ever heard
    about. One alert, suppressed by a record claiming it was delivered.

    So the receipt waits for this, and this waits for a DECLARATION of delivery
    — see :class:`Delivery`. Every caller fires it for an adapter only when the
    declaration COVERS EVERY LINE THAT ADAPTER REPORTED THIS RUN: the runner
    against what its notify hook's payload carried, the CLI against what it
    printed to a stderr somebody was watching. Under a systemd timer stderr is
    the journal, which nobody reads unprompted, so that path declares nothing
    and the report stands.

    WHOLE-ADAPTER COVERAGE, not "the report was delivered". Round 7: the toast
    carried ``report.errors[:5]`` and delivery was declared for the run, so five
    unrelated failures elsewhere silenced the one line this barrier exists to
    repeat. An adapter that cannot show that everything it said got through has
    no basis for deciding which of its own sentences to stop saying, so it says
    them all again — which costs one duplicate line and never an alert.

    THERE IS NO "PROBABLY DELIVERED". A hook that raised, a hook that did
    nothing, a run with no notifier configured at all, an unattended stderr, and
    a payload that dropped the line are the same answer: not delivered, nothing
    stamped, reported again next run. That is deliberately loud forever, and it
    is bounded by configuring a channel — which is the action the noise is
    asking for.

    NOT CALLING IT IS THE SAFE DIRECTION, unlike the write barrier: the cost is
    one more report of a disappearance already reported, never a lost one. The
    two barriers therefore fail opposite ways on purpose. Which is also why
    forgetting ``gates_report`` costs an adapter its suppression rather than its
    alert — the opt-in below fails the same way the barrier does.
    """

    # THE OPT-IN, and it is a value rather than a shape on purpose. See
    # ``is_report_gating``.
    gates_report: bool

    def commit_after_report(self) -> None: ...


def is_report_gating(adapter: object) -> bool:
    """Does this adapter OPT IN to receipt gating? A matching shape never does.

    ROUND 8 MEDIUM, and the third defect in one review traceable to the same
    cause: ``isinstance`` against a ``runtime_checkable`` Protocol checks METHOD
    PRESENCE and nothing else. ``SyncSourceAdapter`` defines
    ``commit_after_report`` unconditionally — it FORWARDS it, and forwarding to a
    source that has none is a no-op, so defining it always was the simple thing
    — and that made every one of the nine sources satisfy
    ``SupportsReportBarrier``.

    What that cost is not academic. ``RunReport.gating_errors`` is what the
    notifier spends its five-line budget on FIRST, precisely so a chronically
    noisy source cannot push the one receipt-gating line out of the toast. With
    every adapter "gating", that list is just ``report.errors``, the ordering is
    a no-op, and five chronic errors from any other source starve TickTick's
    uncovered-project line out of the payload — on every run, forever. That is
    the exact starvation ``gating_errors`` was added to prevent, reintroduced by
    the membership test underneath it.

    So gating is DECLARED, in the adapter's own code, and a shape cannot supply
    the declaration. ``SupportsReportBarrier`` is not ``runtime_checkable``, so
    the accidental spelling (``isinstance(adapter, SupportsReportBarrier)``)
    raises TypeError instead of quietly answering yes; this function is the only
    way to ask. ``is True`` rather than truthiness, so a half-written marker — a
    string, a stray object — is refused rather than promoted.

    FORGETTING THE FLAG IS THE SAFE DIRECTION, which is what makes an explicit
    opt-in affordable here and not on the write barrier: an adapter that gates
    nothing simply repeats its report next run. The write barrier is the
    opposite (never calling it is unbounded loss), so that one stays structural
    and over-inclusive on purpose.
    """
    return getattr(adapter, "gates_report", False) is True and callable(
        getattr(adapter, "commit_after_report", None)
    )


@runtime_checkable
class SupportsInputFreshness(Protocol):
    """Optional: when was the newest input this adapter reads last touched?

    Sources differ on ACQUISITION, not parsing. ``research`` / ``sota-watch``
    / ``dropbox`` read local dirs that other tooling refreshes; the chat
    exports (``claude-web``, ``chatgpt``, ``substack``) read a
    manually-downloaded archive that nothing on this machine refreshes, so a
    timer would happily re-import the same stale zip forever and report
    success. Returning the newest input timestamp lets a later task say
    "substack input is 31 days stale" instead. ``None`` means unknown.

    REPORTED ONCE PER EPISODE, where an episode is one continuous period during
    which this adapter's input is older than the run's threshold. The warning is
    not an error and deliberately does NOT change the exit code — systemd would
    mark the unit failed on every tick for weeks — so the NOTIFICATION IS THE
    CHANNEL, and a notification that repeats every 30 minutes forever is an
    alarm an operator learns to dismiss unread. So the runner suppresses the
    repeat, and the marker that buys the silence waits for the very
    :class:`Delivery` step 6 waits for. What re-arms it is a human dropping a
    fresh export: the value returned here IS the episode's identity, which is
    the second reason it must describe the input rather than the run.

    RETURN AN AWARE DATETIME. A naive one satisfies the annotation and is
    therefore not a type error, but every consumer compares it against
    ``datetime.now(UTC)`` and subtracting a naive datetime from an aware one
    raises TypeError — from the whole-run staleness pass, i.e. one adapter's
    naive timestamp used to cost the run its report, every adapter's error
    listing and its exit code. The runner now normalises a naive value to UTC
    at the boundary rather than trusting this note, but "UTC" is a guess it
    should not have to make: an adapter reading local files should stamp what
    the filesystem actually means (``datetime.fromtimestamp(mtime, tz=UTC)``,
    as the shipped ones do).

    Implementing this is the WHOLE opt-in: the threshold, the wording, the
    dedup and the marker file all live in ``imports/runner.py`` and
    ``imports/ingest_state.py``, because they belong to the run rather than to
    any one source. An adapter answers when its input was last touched and
    nothing else — which is also why growing this protocol did not make four
    more adapters ``gates_report`` and starve the toast of TickTick's line.
    """

    def input_freshness(self) -> datetime | None: ...
