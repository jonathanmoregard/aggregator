"""The TickTick source on the import port.

Same construction as ``imports/research.py`` — subclass ``SyncSourceAdapter``,
let the bridge handle the thread hop and the ``drain_errors`` plumbing, keep
everything TickTick-specific in ``sources/ticktick.py``. Adapting a source is
meant to cost a constructor, not a rewrite, and TickTick is the case that
proves it: two legs, a urllib client hardened GET-only/https-only, and a
credential owned by another integration, none of which the seam has to know.

What this adds over the reference adapter is ``input_freshness``. TickTick's
completed-task history exists only in a CSV a human exports by hand and drops
in ~/Downloads; nothing on this machine refreshes it. Without a freshness
signal a timer re-imports the same months-old export forever and reports
success every time, which is indistinguishable from an index that is current.
"""
from __future__ import annotations

import os
from datetime import datetime

from aggregator.imports.sync_bridge import DEFAULT_CHUNK_SIZE, SyncSourceAdapter
from aggregator.sources.ticktick import TickTickSource


class TickTickAdapter(SyncSourceAdapter):
    """``TickTickSource`` as an ``ImportAdapter``.

    Implements both optional port protocols:

    * ``SupportsNonFatalErrors`` (inherited) — a broken credential, a failed
      project fetch or an unparseable backup is recorded and the run continues
      on whatever leg still works. Draining those into the run report is what
      keeps "degraded to CSV-only" from looking exactly like a healthy run.
    * ``SupportsInputFreshness`` — the age of the newest backup CSV.
    """

    def __init__(
        self,
        *,
        backup_dir: str | os.PathLike[str] | None = None,
        token: str | None = None,
        token_file: str | None = None,
        state_file: str | os.PathLike[str] | None = None,
        archive_dir: str | os.PathLike[str] | None = None,
        source: TickTickSource | None = None,
        since: datetime | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        super().__init__(
            source
            or TickTickSource(
                backup_dir=backup_dir,
                token=token,
                token_file=token_file,
                state_file=state_file,
                archive_dir=archive_dir,
            ),
            since=since,
            chunk_size=chunk_size,
        )

    def input_freshness(self) -> datetime | None:
        """When the newest TickTick backup CSV was last written, or None.

        Deliberately the *input's* age and not the run's: an ingest that reads
        a 31-day-old export is not 31 days behind because it failed — it is
        behind because nobody exported a new one, and only a human can fix
        that. ``None`` means no backup was found at all, which is unknown
        rather than fresh.
        """
        return self._source.newest_backup_mtime()


__all__ = ["TickTickAdapter"]
