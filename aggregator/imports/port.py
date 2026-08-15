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
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from aggregator.sources.base import ObservationRow, Record, SessionRow

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
    ``updated`` — it was, and the row was overwritten.
    ``skipped`` — the sink declined to write it (unknown shape, filtered).
    Load-bearing beyond the summary: a nonzero ``skipped`` withholds the
    ``SupportsWriteBarrier`` call, because an adapter must not advance state
    that implies rows the sink says it did not write.

    These have to come back FROM the write. ``cli.py`` reports
    ``added=len(records) updated=0`` for every run, so its summary is the
    same three numbers whatever happened; that bug is not repeated here.
    """

    added: int = 0
    updated: int = 0
    skipped: int = 0

    def __add__(self, other: WriteCounts) -> WriteCounts:
        return WriteCounts(
            added=self.added + other.added,
            updated=self.updated + other.updated,
            skipped=self.skipped + other.skipped,
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


@runtime_checkable
class SupportsReportBarrier(Protocol):
    """Optional: an adapter that may only go quiet once a human was TOLD.

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

    So the receipt waits for this. The runner calls it after ``notify`` returns
    WITHOUT raising; the single-source CLI path calls it after printing the run's
    errors, which is that path's only channel. A hook that raised means the
    report reached nobody, so nothing is stamped and the next run reports again.

    NOT CALLING IT IS THE SAFE DIRECTION, unlike the write barrier: the cost is
    one more report of a disappearance already reported, never a lost one. The
    two barriers therefore fail opposite ways on purpose.
    """

    def commit_after_report(self) -> None: ...


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

    Not built out yet — this is the seam so it can be, without reopening the
    port.
    """

    def input_freshness(self) -> datetime | None: ...
