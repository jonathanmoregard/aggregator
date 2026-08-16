"""State that belongs to an ingest RUN rather than to any one source.

TWO STORES, AND THE SPLIT IS DELIBERATE
=======================================
This module holds both kinds of ingest-level state, and they live in different
places for reasons that are the opposite of each other.

* **Staleness markers** — a JSON document under ``$XDG_STATE_HOME``. Losing it
  costs one repeated warning. Reading it fails safe, in the loud direction.
  Everything below the "ONE FILE, SECTIONED" heading is about that.
* **High-water marks** — a table in ``cache.db``, beside the data. Losing one
  costs a full re-scan; getting one AHEAD of its data costs records, silently
  and permanently. So it is stored where it can be advanced in the SAME
  TRANSACTION as the chunk it describes, and no other placement is acceptable.

Two artifacts that must agree, updated by two separate ``write()`` calls,
cannot be made to agree under SIGTERM. You get to pick which failure you have:
mark-first means a mark advanced past unprocessed records — silent, permanent
loss that nothing ever reports — and data-first means re-processing, which is
harmless only while the apply is idempotent. Having a real database, we pick
neither: one row and one batch inside one transaction.

WHAT THE HIGH-WATER MARK IS FOR
-------------------------------
On 2026-08-15 ``ingest --all`` never computed one. ``cli.py`` set ``since``
only from an explicit ``--since`` flag, so ``default_adapters(since=None)``
made every 30-minute timer run re-ingest all 372k observations from scratch.
Measured 827 rows/min, ETA 04:55, against a ``TimeoutStartSec=4h`` that fired
at 01:58:58 — SIGTERM at ~44%, the work discarded, the timer refiring 30
minutes later, forever. A doom loop that looked like a hang.

ONE FILE, SECTIONED, not one file per source. The first tenant is the
staleness episode marker: a source whose hand-downloaded export has gone stale
must be warned about ONCE and then go quiet, and something on disk has to
remember that it was. Four of the nine sources can go stale (``chatgpt``,
``claude-web``, ``substack``, and ticktick's CSV leg) and only TickTick has a
state file of its own — giving the other three one apiece would put four
files, four load/save pairs and four sets of permission and atomicity rules
where one will do, and the next ingest-level marker would make it five.

WHERE, AND WHY IT IS STATE. ``$XDG_STATE_HOME/aggregator/ingest/markers.json``,
beside ``$XDG_STATE_HOME/aggregator/ticktick/open_tasks.json`` and for the same
reason: it is regenerable — losing it costs one extra warning — but nothing
else can reconstruct it, so it is not a cache. An unset *or empty* variable
takes the spec default; reading an empty one literally yields a RELATIVE path,
so the markers would be written wherever the timer happened to start and the
next run, started elsewhere, would find none and warn all over again.

0600 AND A DURABLE RENAME, both copied from ``ticktick_api._write_state``. The
mode is set on the scratch fd BEFORE any bytes are written, so there is no
window at 0644, and applied explicitly rather than through ``O_CREAT``'s mode
argument, which does nothing when a scratch file from an earlier run already
exists. The content is a list of the user's source names and export dates —
not a credential, but not something a state file should publish either.

"Durable" rather than merely "atomic" because the rename alone only guarantees
that no READER sees a half-written document; the bytes behind it can still be
in the page cache when the power goes, leaving a zero-length markers file. Here
that is the mildest of the three write paths that share this recipe — reading
fails safe, so a lost file costs one repeated warning — but it is the same
recipe, and a copy that quietly drops a step is how the recipe rots. See
``core/durable.py``.

READING FAILS SAFE, WHICH IS THE OPPOSITE OF THE TICKTICK BASELINE. There the
absent file and the broken file are opposite answers and a broken one RAISES,
because the caller's next act would overwrite unrecoverable completions. Here
the only thing a marker buys is silence, so an unreadable file must resolve to
"nothing is suppressed" — one toast too many, never one too few — and may be
replaced by the next write. A broken file is logged at WARNING (nothing under
``aggregator/`` configures logging, so ``logging.lastResort`` is what prints
and anything quieter reaches nobody) rather than raised: the dedup being dead
is worth saying, and it must not cost the run its ingest.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import random
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from aggregator.core.durable import flush_to_disk, replace_durably
from aggregator.sources.base import PermanentFault, fault_stamp

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance only
    from aggregator.core.store import Store

log = logging.getLogger(__name__)

# The section holding one marker per source whose input has gone stale. Keyed
# by adapter name; see ``runner.plan_staleness_report`` for what a value means.
STALE_INPUTS = "stale_inputs"


class CursorKind:
    """How a source's change-detection works. Three kinds, all in use.

    Not an enum: these strings are written into ``ingest_state.cursor_kind``
    and read back by ``aggregator status``, so they are a stored vocabulary and
    a plain constant keeps the SQL, the report and the tests spelling them the
    same way.
    """

    #: A mutable timestamp the source filters on and that MOVES when a row is
    #: revised — a file mtime, a conversation's ``last_ts``, a PR's
    #: ``updated_at``. The default and the common case.
    MODIFIED_TIME = "modified-time"
    #: A row, once emitted, is never revised. The cursor still walks forward,
    #: but re-reading behind it corrects nothing, so it needs only enough
    #: margin to cover the boundary tie.
    APPEND_ONLY = "append-only"
    #: No usable cursor at all. This source full-scans on every run BY DESIGN
    #: and says so; it must never be made to look incremental.
    NONE = "none"


# THE OVERLAP WINDOW — the whole late-arriving-data mitigation, in one number.
#
# A naive ``max(timestamp)`` mark silently drops records for three compounding
# reasons: a write that COMMITTED after the extract read past its timestamp;
# clocks that disagree between whatever stamped the rows; and the tie at the
# boundary itself, where ``>`` drops and ``>=`` double-loads. None of the three
# is fixable at the reader.
#
# The fix is not per-partition frontiers, a global low watermark, idle-partition
# timeouts or a side-output ledger for late events — that machinery is for a
# multi-partition distributed stream, and this is nine independent sources on
# one laptop with no cross-source window to seal. The fix is: read from
# ``mark - margin`` and make the re-read free. Production guidance for the
# margin lands at 5 minutes typical, 30 minutes for heavy workloads, an hour as
# a worst case. On a 30-minute timer over personal data the re-read cost is
# bounded by the change rate, which is tiny, so a flat hour buys the worst case
# for nothing and removes any per-source number to get wrong.
#
# "Free" is not an assumption: it is enforced by the ``src_hash`` guard in
# ``core/store.py``, which makes re-writing an unchanged row cost no scrub and
# no page write. Without that guard this margin would rewrite every row in the
# window on all 48 daily runs — a doom loop traded for a slow bleed.
MODIFIED_TIME_OVERLAP = timedelta(hours=1)

# An append-only source needs no correction window: its rows are never revised,
# so there is nothing behind the mark that could have changed. What is left is
# the boundary tie and a file caught between two writes, which minutes covers.
APPEND_ONLY_OVERLAP = timedelta(minutes=5)

# How many consecutive failures before a source is rested, and how long for.
#
# This is the entire circuit breaker, and it is a counter rather than a state
# machine on purpose. Amazon's own guidance argues against a real breaker even
# at scale — it introduces modal behaviour that is hard to test and slow to
# recover from, and a local retry limiter does the same job — and at nine
# source calls per run there is no herd to thunder. A flapping source degrades
# to "checked less often" and never blocks the other eight.
BACKOFF_AFTER_FAILURES = 5
BACKOFF_BASE = timedelta(minutes=1)
# Capped so a backoff is always a pause and never a disablement: whatever the
# failure count, the source is retried within six hours. A source held back
# indefinitely by a counter nobody is watching is the silent-rot failure this
# repo keeps ruling out.
BACKOFF_CAP = timedelta(hours=6)


def full_jitter(attempt: int, *, base: timedelta, cap: timedelta) -> timedelta:
    """Capped exponential backoff with FULL jitter: ``uniform(0, min(cap, b*2^n))``.

    Jitter rather than plain exponential backoff because un-jittered retries
    re-synchronise into clusters — the measured comparison has full jitter doing
    the least total work of the three published variants, and no-jitter as the
    clear loser. At this scale the clustering barely matters; using the
    known-good formula rather than a hand-rolled one costs a line.
    """
    ceiling = min(cap.total_seconds(), base.total_seconds() * (2**attempt))
    return timedelta(seconds=random.uniform(0, ceiling))  # noqa: S311 - backoff, not crypto


@dataclass(frozen=True)
class SourceCursor:
    """What one source's watermark MEANS, declared next to why.

    ``field`` names what the source actually compares ``since`` against, in the
    source's own vocabulary. It is not decoration: a mark taken from one field
    and applied to another is a silent drop, and TickTick is the live example —
    its ``since`` filters CSV BACKUP FILES by mtime while every record it emits
    carries the task's own completion time. Writing both down is what made that
    mismatch visible instead of shipping it.
    """

    kind: str
    field: str
    note: str

    @property
    def overlap(self) -> timedelta:
        """How far behind the stored mark this source reads. See the constants."""
        if self.kind == CursorKind.MODIFIED_TIME:
            return MODIFIED_TIME_OVERLAP
        if self.kind == CursorKind.APPEND_ONLY:
            return APPEND_ONLY_OVERLAP
        return timedelta(0)

    @property
    def is_incremental(self) -> bool:
        return self.kind != CursorKind.NONE


# EVERY SOURCE, WITH THE KIND VERIFIED AGAINST ITS CODE — not assumed from its
# shape. A missing entry is refused by a test, because a source that fell out of
# this table would silently full-scan on every tick and look exactly like a
# source with nothing new: the bug, reintroduced one source at a time.
SOURCE_CURSORS: dict[str, SourceCursor] = {
    "sessions": SourceCursor(
        kind=CursorKind.MODIFIED_TIME,
        field="SessionRow.last_ts",
        note=(
            "sessions.py skips a session whose last_ts is below `since`, and "
            "last_ts moves every time the JSONL grows, so the cursor is a "
            "mutable modified-time"
        ),
    ),
    "github": SourceCursor(
        kind=CursorKind.MODIFIED_TIME,
        field="Record.updated_at (GitHub search `updated:>=`)",
        note=(
            "the search qualifier is DAY-granular, so the window is always "
            "rounded down to midnight UTC and re-reads the whole day; the "
            "idempotent upsert is what makes that free"
        ),
    ),
    "chatgpt": SourceCursor(
        kind=CursorKind.MODIFIED_TIME,
        field="SessionRow.last_ts (conversation update_time)",
        note="a conversation continued later carries a later update_time",
    ),
    "claude-web": SourceCursor(
        kind=CursorKind.MODIFIED_TIME,
        field="SessionRow.last_ts (conversation updated_at)",
        note="a conversation continued later carries a later updated_at",
    ),
    "research": SourceCursor(
        kind=CursorKind.APPEND_ONLY,
        field="Record.updated_at (report file mtime)",
        note=(
            "report filenames are the request's content id and the agent "
            "writes each file once — a new report is a new file, existing "
            "ones are not revised, so re-reading behind the mark corrects "
            "nothing"
        ),
    ),
    "sota-watch": SourceCursor(
        kind=CursorKind.MODIFIED_TIME,
        field="Record.updated_at (proposal file mtime)",
        note="proposals are edited in place, unlike research reports",
    ),
    "substack": SourceCursor(
        kind=CursorKind.MODIFIED_TIME,
        field="Record.updated_at (zip member mtime)",
        note=(
            "a re-exported archive restamps its members, so the same post can "
            "legitimately arrive with a newer mtime"
        ),
    ),
    "dropbox": SourceCursor(
        kind=CursorKind.MODIFIED_TIME,
        field="Record.updated_at (file mtime)",
        note=(
            "a synced file is rewritten in place; the mtime filter also runs "
            "BEFORE extraction, so an incremental run costs a stat per file "
            "rather than a PDF parse"
        ),
    ),
    "ticktick": SourceCursor(
        kind=CursorKind.NONE,
        field="(none — see note)",
        note=(
            "two legs, neither windowable. `since` filters the CSV BACKUP "
            "FILES by their mtime, while every record carries the TASK's own "
            "completion or creation time, so a mark taken from the items "
            "would be compared against a different quantity and drop rows. "
            "And the Open API leg serves OPEN tasks only: a completion is "
            "visible solely as a DISAPPEARANCE between two full polls, so a "
            "narrowed poll would infer completions that never happened. This "
            "source full-scans on every run and the run report says so"
        ),
    ),
}


def cursor_for(source: str) -> SourceCursor:
    """This source's declared cursor, or the honest "we do not know" answer.

    An unknown source gets ``CursorKind.NONE`` rather than a guessed window:
    the cost is a full scan that is REPORTED as a full scan, and the
    alternative — inventing a modified-time window for a source nobody
    classified — is a silent drop.
    """
    known = SOURCE_CURSORS.get(source)
    if known is not None:
        return known
    return SourceCursor(
        kind=CursorKind.NONE,
        field="(unknown)",
        note=(
            f"{source!r} is not in SOURCE_CURSORS, so nothing here knows what "
            f"its `since` is compared against; it full-scans until someone "
            f"classifies it"
        ),
    )


@dataclass(frozen=True)
class SourcePlan:
    """What this run intends to do with one source, and why.

    Carried into the report so a journal entry read after the fact can
    distinguish "nothing new" from "not looked at" from "cannot be windowed" —
    three outcomes that otherwise print the same three zeros.
    """

    source: str
    cursor: SourceCursor
    cursor_value: datetime | None
    since: datetime | None
    consecutive_failures: int
    last_run_at: datetime | None
    last_error: str | None
    skip_reason: str | None
    override: bool = False

    @property
    def window_description(self) -> str:
        """One line an operator can read without opening the code."""
        if self.override and self.since is not None:
            return f"since {self.since.isoformat()} (--since, typed by hand)"
        if not self.cursor.is_incremental:
            return f"FULL SCAN (no usable cursor): {self.cursor.note}"
        if self.since is None:
            return f"FULL SCAN (first run; cursor is {self.cursor.field})"
        return f"since {self.since.isoformat()} (cursor {self.cursor.field})"


class Watermarks:
    """The per-source high-water marks, read from and written to ``cache.db``.

    Thin over ``Store`` on purpose: the SQL and its monotonicity guard live in
    ``core/store.py`` beside the data they must commit with, and the POLICY —
    which overlap, when to back off, how to say "this source cannot be
    windowed" — lives here, beside the table that declares it. Splitting them
    the other way round would put the reason a window is an hour wide in a file
    that only knows how to run statements.
    """

    def __init__(self, store: Store, *, override: datetime | None = None) -> None:
        self._store = store
        # ``--since`` typed by a human, applied to every source. Held on the
        # INSTANCE rather than passed per call so the two places that ask —
        # ``registry.default_adapters``, which builds the adapters, and the
        # runner, which describes the window in the report — cannot disagree
        # about what window this run actually used. A report that says
        # "since <watermark>" while the adapter was handed something else is
        # worse than no report.
        self._override = override

    def plan(self, source: str, *, now: datetime | None = None) -> SourcePlan:
        """What to pass this source as ``since``, and whether to run it at all."""
        moment = now or datetime.now(UTC)
        cursor = cursor_for(source)
        state = self._store.read_ingest_state(source)
        stored = _parse(state.get("cursor_value"))
        failures = int(state.get("consecutive_failures") or 0)
        last_run = _parse(state.get("last_run_at"))
        since: datetime | None = None
        if self._override is not None:
            # AN OVERRIDE APPLIES EVEN TO A SOURCE WITH NO CURSOR. "--since" is
            # a human narrowing this run by hand, and every source that can
            # honour a window should honour the one they typed; the mark is
            # still advanced afterwards, so a hand-run does not leave the
            # timer's state behind.
            since = self._override
        elif cursor.is_incremental and stored is not None:
            since = stored - cursor.overlap
        return SourcePlan(
            override=self._override is not None,
            source=source,
            cursor=cursor,
            cursor_value=stored,
            since=since,
            consecutive_failures=failures,
            last_run_at=last_run,
            last_error=state.get("last_error"),
            skip_reason=_backoff_reason(source, failures, _parse(state.get("next_attempt_at")), moment),
        )

    def advance(
        self,
        source: str,
        cursor_value: datetime | None,
        *,
        rows: int,
        now: datetime | None = None,
        commit: bool = True,
    ) -> None:
        """Record a SUCCESSFUL pass over ``source``, moving the mark forward.

        ``cursor_value=None`` is the quiet run — the normal state of this
        pipeline, since most ticks have nothing new — and it must never write
        NULL over a live mark. That exact defect is on record in another
        implementation of this pattern (an empty sync nuking state to ``{}``),
        and here it would full-scan 372k rows on the very next tick.

        Clears ``consecutive_failures``: one clean pass is the definition of
        the source being healthy again.

        ``commit=False`` leaves the transaction open so the caller can make
        this advance and the chunk it describes ONE atomic unit — which is the
        entire reason the mark lives in this database. See
        ``store_sink.StoreSink.write_checkpoint``.
        """
        self._store.advance_ingest_cursor(
            source,
            cursor_kind=cursor_for(source).kind,
            cursor_value=cursor_value.isoformat() if cursor_value else None,
            rows=rows,
            at=(now or datetime.now(UTC)).isoformat(),
            _commit=commit,
        )

    def record_failure(
        self, source: str, error: str, *, now: datetime | None = None
    ) -> None:
        """Count a failed pass, and decide once when to try again.

        THE MARK IS NEVER TOUCHED. A failed run's mark must stay exactly where
        it was: the next run re-reads the same window, which the idempotent
        apply makes cheap, and nothing between the old mark and wherever this
        run got to can be lost.

        The next-attempt time is computed HERE and stored, rather than derived
        on read from the failure count. The delay is jittered, so deriving it
        would re-roll the dice on every read and let two calls inside one run
        disagree about whether this source runs — a decision that has to be
        made once and then stay made.
        """
        moment = now or datetime.now(UTC)
        failures = int(
            self._store.read_ingest_state(source).get("consecutive_failures") or 0
        ) + 1
        next_attempt: datetime | None = None
        if failures >= BACKOFF_AFTER_FAILURES:
            next_attempt = moment + full_jitter(
                failures, base=BACKOFF_BASE, cap=BACKOFF_CAP
            )
        self._store.record_ingest_failure(
            source,
            cursor_kind=cursor_for(source).kind,
            error=error,
            at=moment.isoformat(),
            next_attempt_at=next_attempt.isoformat() if next_attempt else None,
        )


#: How many times a record that cannot be written is retried before it is
#: declared terminal. Three is the usual range's floor and the right end of it
#: here: a record that failed three ingests a run apart is not going to start
#: working, and every extra attempt is a scrub and a write spent on a row that
#: will fail again.
POISON_MAX_ATTEMPTS = 3

#: Backoff base for a held record's next attempt. Wider than the source-level
#: one because the thing being waited on is usually a human fixing a file.
POISON_RETRY_BASE = timedelta(minutes=15)
POISON_RETRY_CAP = timedelta(days=1)


@dataclass(frozen=True)
class HeldRecord:
    """One record that could not be written, and what happens to it next.

    ``next_retry_at is None`` is TERMINAL: never attempted again. The row is
    not deleted, because a failure nobody can count is a gap in the index that
    reads as full coverage — which is the one failure mode this repo keeps
    ruling out. ``aggregator status`` lists them.
    """

    source: str
    record_key: str
    error_type: str
    error_detail: str | None
    attempts: int
    next_retry_at: datetime | None

    @property
    def terminal(self) -> bool:
        return self.next_retry_at is None


@dataclass(frozen=True)
class KnownFault:
    """One permanent fault AS STORED — i.e. one a human has already been told about.

    Distinct from :class:`~aggregator.sources.base.PermanentFault`, which is
    what a source reports THIS run, because the two carry different facts and
    only one of them can be re-derived. ``first_seen_at`` is the number that
    decides whether anybody cares and exists only in the ledger; ``stamp`` is
    what the artifact looked like when we last saw the fault, and comparing it
    against the file now is how a fault that has been fixed stops being
    reported as quarantined.
    """

    source: str
    key: str
    scope: str
    reason: str
    detail: str
    count: int
    line: str
    first_seen_at: datetime | None
    last_seen_at: datetime | None
    stamp: str = ""


@dataclass(frozen=True)
class FaultVerdict:
    """What this run's permanent faults mean, split three ways.

    ``new`` is loud (it stays in the run's errors and fails the run), ``known``
    is quiet (it becomes a note), and ``released`` is a row that has left the
    ledger because the input parses again. A run with all three is normal.
    """

    known: list[KnownFault]
    new: list[PermanentFault]
    released: list[KnownFault]

    @property
    def quarantined_records(self) -> int:
        """How many records are currently being held quiet by this source."""
        return sum(fault.count for fault in self.known)


class PoisonLedger:
    """Per-record failures, held in ``cache.db`` beside the data.

    THE TWO FAILURES THIS AVOIDS SIMULTANEOUSLY. One bad record must not abort
    a run — the sessions leg emits ~372k rows and losing all of them to one
    malformed observation is absurd — and it must not be retried forever
    either, because an unbounded retry is a run that never finishes and an
    alert that never stops.

    A table rather than a broker or a separate dead-letter process: for one
    user on one machine the table is strictly better, because it is
    transactional with the data writes and a broker can never be. The
    distributed machinery exists because SQS consumers are stateless and
    spread out; this one is neither.
    """

    def __init__(self, store: Store) -> None:
        self._store = store

    def held(self, source: str, *, now: datetime | None = None) -> dict[str, HeldRecord]:
        """The records this run must NOT attempt, keyed by record key.

        Terminal rows and rows whose retry time has not arrived. A row that IS
        due is deliberately absent, so it gets attempted and either succeeds
        (and is released) or is held again with a longer delay.
        """
        moment = now or datetime.now(UTC)
        out: dict[str, HeldRecord] = {}
        for row in self._store.read_poison(source):
            entry = _held_from_row(row)
            due = entry.next_retry_at
            if due is None or due > moment:
                out[entry.record_key] = entry
        return out

    def hold(
        self,
        source: str,
        record_key: str,
        error: BaseException,
        *,
        previous: HeldRecord | None = None,
        now: datetime | None = None,
    ) -> HeldRecord:
        """Record one failed record and schedule (or refuse) its next attempt."""
        moment = now or datetime.now(UTC)
        attempts = (previous.attempts if previous else 0) + 1
        if attempts >= POISON_MAX_ATTEMPTS:
            next_retry = None
        else:
            next_retry = moment + full_jitter(
                attempts, base=POISON_RETRY_BASE, cap=POISON_RETRY_CAP
            )
        detail = f"{type(error).__name__}: {error}"
        self._store.hold_poison(
            source,
            record_key,
            error_type=type(error).__name__,
            error_detail=detail,
            at=moment.isoformat(),
            next_retry_at=next_retry.isoformat() if next_retry else None,
        )
        return HeldRecord(
            source=source,
            record_key=record_key,
            error_type=type(error).__name__,
            error_detail=detail,
            attempts=attempts,
            next_retry_at=next_retry,
        )

    def release(self, source: str, record_key: str) -> None:
        """A previously-poison record wrote cleanly. Stop holding it."""
        self._store.release_poison(source, record_key)

    def summary(self) -> list[dict[str, object]]:
        """``(source, error_type, count, terminal)`` — the whole "what is broken" report."""
        return self._store.poison_summary()

    # -- the other half: input that will never parse ----------------------

    def reconcile_faults(
        self,
        source: str,
        reported: Sequence[PermanentFault],
        *,
        now: datetime | None = None,
    ) -> FaultVerdict:
        """Sort this run's permanent faults into KNOWN, NEW and GONE.

        THE THREE ANSWERS, and the whole alert-fatigue fix is the difference
        between the first two:

        * **known** — this exact identity is already in the ledger, so a human
          has already been told about it and told once is the contract. Its
          error line comes OUT of the run's errors and becomes a note; the run
          can then exit 0 and the timer stops notifying about a file that has
          been broken since March.
        * **new** — never seen. Stays in the errors list, fails the run, fires
          the notification. Nothing about this path is softened, and that is
          the point: "fail loudly" means the operator learns of each new
          problem exactly once, not that the same permanent problem shouts
          twice an hour.
        * **released** — a stored fault whose artifact has CHANGED since it was
          recorded and which this run did not re-report. The file was rewritten
          and now parses, so the records are in the index and the row would
          otherwise have ``aggregator status`` overstating the damage forever.

        WHY "CHANGED **AND** NOT RE-REPORTED", both halves. Changed alone is
        wrong: ``~/.claude/projects`` files are APPENDED TO by every resumed
        session, so a live file's stamp moves constantly while its corrupt line
        at 318 sits exactly where it was — dropping the row on a stamp change
        would re-alarm that file on every resume, i.e. the original bug with
        extra steps. Not-re-reported alone is wrong the other way: an
        incremental run legitimately skips files below its watermark, and
        forgetting a fault merely because this pass did not look at the file
        would re-alarm it the moment the file is next read.

        A ledger write happens here for the known and released cases only.
        Recording a NEW fault waits for :func:`runner.commit_fault_receipts`,
        because a fault reported to nobody must not buy any silence.
        """
        moment = now or datetime.now(UTC)
        stored = {
            str(row["fault_key"]): _known_from_row(row)
            for row in self._store.read_faults(source)
        }
        seen = {fault.key for fault in reported}
        known: list[KnownFault] = []
        fresh: list[PermanentFault] = []
        for fault in reported:
            held = stored.get(fault.key)
            if held is None:
                fresh.append(fault)
                continue
            # Still true, and still the same fault. Refresh what MOVES (the
            # stamp, the sighting time) and never ``first_seen_at``.
            self.record_fault(source, fault, now=moment)
            known.append(
                KnownFault(
                    source=source,
                    key=fault.key,
                    scope=fault.scope,
                    reason=fault.reason,
                    detail=fault.detail,
                    count=fault.count,
                    line=fault.line,
                    first_seen_at=held.first_seen_at,
                    last_seen_at=moment,
                )
            )
        released = [
            held
            for key, held in stored.items()
            if key not in seen and fault_stamp(held.scope) != held.stamp
        ]
        for held in released:
            self._store.forget_fault(source, held.key)
        return FaultVerdict(known=known, new=fresh, released=released)

    def record_fault(
        self, source: str, fault: PermanentFault, *, now: datetime | None = None
    ) -> None:
        """Write one permanent fault into the ledger. THE RECEIPT, in one call.

        Called for a NEW fault only once a channel has declared it carried the
        line — see :func:`runner.commit_fault_receipts` — and for a known one on
        every sighting, to refresh the stamp.
        """
        moment = (now or datetime.now(UTC)).isoformat()
        self._store.record_fault(
            source,
            fault.key,
            scope=fault.scope,
            scope_stamp=fault.stamp,
            reason=fault.reason,
            detail=fault.detail,
            record_count=fault.count,
            line=fault.line,
            at=moment,
        )

    def fault_summary(self) -> list[dict[str, object]]:
        """Every permanently-bad input this ledger is holding quiet.

        THE VISIBLE HALF OF THE BARGAIN, and it is the reason going quiet is
        defensible at all: the same rule the TickTick uncovered-project report
        and the staleness markers already live under. A suppressed alarm that
        nothing can show you on demand has not been suppressed, it has been
        lost. ``aggregator status`` prints this.
        """
        return self._store.fault_summary()


def _known_from_row(row: Mapping[str, object]) -> KnownFault:
    return KnownFault(
        source=str(row["source"]),
        key=str(row["fault_key"]),
        scope=str(row["scope"]),
        reason=str(row["reason"]),
        detail=str(row["detail"]),
        count=int(row["record_count"] or 0),
        line=str(row["line"]),
        first_seen_at=_parse(row.get("first_seen_at")),
        last_seen_at=_parse(row.get("last_seen_at")),
        stamp=str(row.get("scope_stamp") or ""),
    )


def _held_from_row(row: Mapping[str, object]) -> HeldRecord:
    return HeldRecord(
        source=str(row["source"]),
        record_key=str(row["record_key"]),
        error_type=str(row["error_type"]),
        error_detail=(
            str(row["error_detail"]) if row.get("error_detail") is not None else None
        ),
        attempts=int(row["attempts"] or 0),
        next_retry_at=_parse(row.get("next_retry_at")),
    )


def _backoff_reason(
    source: str, failures: int, next_attempt: datetime | None, moment: datetime
) -> str | None:
    """Whether to rest this source this run, read off a stored decision.

    A comparison rather than a computation, so the same run always gets the
    same answer and an operator reading ``aggregator status`` sees the very
    time the runner is honouring.
    """
    if next_attempt is None or moment >= next_attempt:
        return None
    return (
        f"{source}: skipped this run — {failures} consecutive failures, "
        f"resting until {next_attempt.isoformat()}. The other sources are "
        f"unaffected, and the mark is untouched, so nothing is lost by waiting"
    )


def _parse(value: object) -> datetime | None:
    """Read a stored ISO timestamp back, refusing to guess at a broken one."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        log.warning(
            "ingest_state holds an unparseable timestamp %r; treating it as "
            "absent, which costs one full scan and never a dropped row",
            value,
        )
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def default_marker_path() -> Path:
    """Where ingest-level markers live when the caller names no path.

    Same rule as ``ticktick_api.default_state_path``, including the empty-value
    trap: ``XDG_STATE_HOME=`` resolves to a relative path, which would scatter
    the markers across whatever directory each run started in.
    """
    base = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
    return Path(base) / "aggregator" / "ingest" / "markers.json"


@contextmanager
def _marker_lock(path: Path) -> Iterator[None]:
    """Serialise marker writes across processes, via a sidecar lock file.

    A sidecar rather than the file itself, because the file is replaced by
    ``rename`` on every save: a lock held on the old inode says nothing about
    the new one. Ingests DO overlap (a timer firing over a manual run), and the
    two would otherwise share one ``.tmp`` scratch path and interleave their
    bytes in it.

    No compare-and-swap, unlike the open-task baseline. There the loser of a
    race destroys completions the Open API can never re-serve; here it costs
    one duplicated warning, which the next run corrects by itself.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


@dataclass(frozen=True)
class IngestMarkers:
    """The marker document as one JSON file. Sectioned, so it stays one file.

    Thin on purpose: it knows about bytes and permissions and nothing about
    what a marker means. The episode rules live in ``imports/runner.py``, with
    the code that computes the warning a marker suppresses, so "stale" cannot
    come to mean one thing where the warning is raised and another where it is
    silenced.
    """

    path: Path = field(default_factory=default_marker_path)

    def _read(self) -> dict[str, object]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
            log.warning(
                "aggregator ingest markers at %s could not be read (%s: %s); "
                "treating every marker as absent, so anything they were "
                "suppressing is reported again. The next run that has "
                "something to record replaces the file",
                self.path,
                type(e).__name__,
                e,
            )
            return {}
        if not isinstance(data, dict):
            log.warning(
                "aggregator ingest markers at %s hold a %s, not a section map; "
                "treating every marker as absent",
                self.path,
                type(data).__name__,
            )
            return {}
        return data

    def load(self, section: str) -> dict[str, dict]:
        """One section's markers, keyed by source. ``{}`` when there are none.

        NEVER RAISES, and every failure answers ``{}`` — which means "nothing is
        suppressed", i.e. warn. See the module docstring for why that direction
        is not negotiable.
        """
        value = self._read().get(section)
        if not isinstance(value, dict):
            return {}
        return {name: mark for name, mark in value.items() if isinstance(name, str)}

    def save(self, section: str, markers: Mapping[str, dict]) -> None:
        """Replace one section, leaving every other section alone.

        Read-modify-write INSIDE the lock: a second section (a future
        ingest-level marker) must not be lost to a staleness write that read the
        document before it was added.

        Raises ``OSError`` when the file cannot be written. The caller reports
        it rather than failing the run — the ingest itself succeeded and the
        cost is one repeated warning — but a marker that silently never lands
        turns "reported once" back into "reported every 30 minutes".
        """
        with _marker_lock(self.path):
            document = self._read()
            document[section] = dict(markers)
            scratch = self.path.with_name(self.path.name + ".tmp")
            fd = os.open(scratch, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    os.fchmod(handle.fileno(), 0o600)
                    handle.write(json.dumps(document))
                    flush_to_disk(handle)
            except BaseException:
                scratch.unlink(missing_ok=True)
                raise
            replace_durably(scratch, self.path)


def stale_input_markers(path: Path | None = None) -> dict[str, dict]:
    """Every source whose staleness warning is currently suppressed.

    THE VISIBLE HALF of the report-once rule, exactly as
    ``ticktick_api.uncovered_projects`` is for a vanished project: suppressing a
    repeat is only defensible while the suppressed state is somewhere a human
    can go and look, or "quiet" has become "forgotten". ``aggregator status``
    prints this.

    Never raises, for the same reason that one does not — a read-only report
    that cannot show anything at all is worse than one showing nothing.
    """
    return IngestMarkers(path or default_marker_path()).load(STALE_INPUTS)


__all__ = [
    "APPEND_ONLY_OVERLAP",
    "BACKOFF_AFTER_FAILURES",
    "BACKOFF_BASE",
    "BACKOFF_CAP",
    "MODIFIED_TIME_OVERLAP",
    "POISON_MAX_ATTEMPTS",
    "POISON_RETRY_BASE",
    "POISON_RETRY_CAP",
    "SOURCE_CURSORS",
    "STALE_INPUTS",
    "CursorKind",
    "FaultVerdict",
    "HeldRecord",
    "IngestMarkers",
    "KnownFault",
    "PoisonLedger",
    "SourceCursor",
    "SourcePlan",
    "Watermarks",
    "cursor_for",
    "default_marker_path",
    "full_jitter",
    "stale_input_markers",
]
