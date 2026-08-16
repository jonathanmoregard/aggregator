"""Make "silent for two hours" structurally impossible.

THE INCIDENT, RESTATED AS A LOGGING PROBLEM. On 2026-08-15 the first live timer
run printed 102 lines of pypdf noise and then said nothing for two hours. The
sessions/observations leg — which was doing all of the work, and doing it
wrong — logged NOTHING AT ALL, so a library's per-object warnings were the only
voice in the journal. Its absence read as "pypdf is hung"; the dropbox leg had
in fact finished at 22:17 and the fault was a missing watermark somewhere else
entirely. Two hours and a disproven hypothesis went into that gap.

Two rules follow, and they are different rules:

1. **Every chunk gets a line.** Chunk boundaries are where the useful facts
   are — rows so far, rate, what the mark will become — and at a few hundred
   items per chunk they arrive often enough to be a pulse.
2. **Silence gets a line too.** Rule 1 alone is not enough: a leg that is
   parsing one enormous file, blocked on a socket, or stuck produces no chunk
   boundaries, which is exactly the state that looked like a hang. So a
   heartbeat runs on the clock rather than on the work, and says how long the
   quiet has lasted. A source that has gone quiet is then a visible fact
   instead of an inference from missing output.

WHY A LOGGER AND NOT ``print``. This is telemetry, not the run's report: the
report is printed by ``cli.py`` and is what a human reads on purpose, while
these lines are what a journal holds for the one time somebody goes looking.
Keeping them on a logger also means a library caller of ``run_imports`` gets
them only if it asks.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field

log = logging.getLogger("aggregator.ingest")

# How long a source may say nothing before the heartbeat says it for them.
#
# Short enough that a stall is visible while an operator still has the context
# to care, long enough that a healthy fast source is described by its chunk
# lines rather than by this. The absolute rule it enforces is the one the
# incident produced: never more than a minute of silence, ever.
HEARTBEAT_SECONDS = 30.0


@dataclass
class SourceProgress:
    """One source's live counters. Mutated in place by the runner's chunk loop."""

    source: str
    window: str
    started_at: float
    phase: str = "streaming"
    rows: int = 0
    chunks: int = 0
    written: int = 0
    unchanged: int = 0
    quarantined: int = 0
    last_line_at: float = 0.0
    finished: bool = False
    # Rate is windowed over the last few chunks rather than over the run,
    # because a run-lifetime average lies badly when one source parses PDFs and
    # another reads JSON: it reports a number that was never true at any moment.
    recent: list[tuple[float, int]] = field(default_factory=list)

    def rate_per_s(self) -> float:
        if len(self.recent) < 2:
            return 0.0
        (t0, r0), (t1, r1) = self.recent[0], self.recent[-1]
        span = t1 - t0
        return (r1 - r0) / span if span > 0 else 0.0


class RunProgress:
    """Per-run progress logging, with a heartbeat that fires when nothing does.

    ``clock`` is injectable so a test can assert the heartbeat's cadence
    without sleeping through it — a timing rule verified by waiting is a
    timing rule that gets marked flaky and then deleted.
    """

    #: How many chunk samples the windowed rate is computed over.
    RATE_WINDOW = 8

    def __init__(
        self,
        *,
        run_id: str | None = None,
        heartbeat_seconds: float = HEARTBEAT_SECONDS,
        clock=time.monotonic,
    ) -> None:
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.heartbeat_seconds = heartbeat_seconds
        self._clock = clock
        self.sources: dict[str, SourceProgress] = {}

    # -- lifecycle ---------------------------------------------------------

    def begin(self, source: str, window: str) -> SourceProgress:
        """A source is starting. Says WHICH WINDOW, which is the new fact.

        "sessions: starting" and "sessions: starting, since 2026-08-16T09:00"
        are the difference between a journal that can and cannot answer "was
        this run incremental?" six hours after the fact.
        """
        now = self._clock()
        entry = SourceProgress(
            source=source, window=window, started_at=now, last_line_at=now
        )
        self.sources[source] = entry
        log.info(
            "run=%s source=%s phase=begin window=%s",
            self.run_id,
            source,
            window,
        )
        return entry

    def chunk(
        self,
        entry: SourceProgress,
        *,
        rows: int,
        added: int,
        updated: int,
        unchanged: int,
        quarantined: int = 0,
    ) -> None:
        """One committed chunk. The pulse."""
        now = self._clock()
        entry.rows += rows
        entry.chunks += 1
        entry.written += added + updated - unchanged
        entry.unchanged += unchanged
        entry.quarantined += quarantined
        entry.last_line_at = now
        entry.recent.append((now, entry.rows))
        if len(entry.recent) > self.RATE_WINDOW:
            entry.recent.pop(0)
        log.info(
            "run=%s source=%s phase=chunk rows=%d chunks=%d written=%d "
            "unchanged=%d quarantined=%d rate=%.1f/s elapsed=%.1fs",
            self.run_id,
            entry.source,
            entry.rows,
            entry.chunks,
            entry.written,
            entry.unchanged,
            entry.quarantined,
            entry.rate_per_s(),
            now - entry.started_at,
        )

    def end(self, entry: SourceProgress, *, status: str, mark: str) -> None:
        """A source is done, and the line says WHERE THE MARK ENDED UP.

        ``status`` distinguishes the outcomes that otherwise print the same
        zeros: finished / interrupted / failed / skipped. Without it a run that
        was SIGTERMed at 44% and a run with nothing to do are the same entry.
        """
        entry.finished = True
        entry.phase = status
        now = self._clock()
        entry.last_line_at = now
        log.info(
            "run=%s source=%s phase=end status=%s rows=%d written=%d "
            "unchanged=%d quarantined=%d mark=%s elapsed=%.1fs",
            self.run_id,
            entry.source,
            status,
            entry.rows,
            entry.written,
            entry.unchanged,
            entry.quarantined,
            mark,
            now - entry.started_at,
        )

    def skipped(self, source: str, reason: str) -> None:
        """A source that was not run at all. SAID OUT LOUD, never merely absent.

        A source missing from the output is indistinguishable from a source
        with nothing to do, and this is the one case where the difference
        matters most: a source resting on backoff is not being watched.
        """
        log.info(
            "run=%s source=%s phase=skipped reason=%s", self.run_id, source, reason
        )

    # -- the heartbeat -----------------------------------------------------

    def emit_heartbeat(self) -> list[str]:
        """Log one line per source that has gone quiet. Returns what it said.

        Returning the lines is not decoration: it is how a test asserts the
        rule without waiting for wall-clock seconds to pass.
        """
        now = self._clock()
        said: list[str] = []
        for entry in self.sources.values():
            if entry.finished:
                continue
            quiet = now - entry.last_line_at
            if quiet < self.heartbeat_seconds:
                continue
            entry.last_line_at = now
            line = (
                f"run={self.run_id} source={entry.source} phase=heartbeat "
                f"rows={entry.rows} quiet_for={quiet:.0f}s "
                f"elapsed={now - entry.started_at:.0f}s window={entry.window}"
            )
            said.append(line)
            log.warning(
                "%s -- this source has produced no chunk in %.0fs; it is "
                "working or blocked, but it is NOT forgotten",
                line,
                quiet,
            )
        return said

    async def beat(self) -> None:
        """Drive :meth:`emit_heartbeat` forever. Cancelled by the runner.

        Deliberately a task rather than a check inside the chunk loop: the
        condition it exists to report is precisely the one in which the chunk
        loop is not running.
        """
        try:
            while True:
                await asyncio.sleep(self.heartbeat_seconds / 2)
                self.emit_heartbeat()
        except asyncio.CancelledError:  # pragma: no cover - normal teardown
            return


__all__ = ["HEARTBEAT_SECONDS", "RunProgress", "SourceProgress", "log"]
