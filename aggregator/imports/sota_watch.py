"""The sota-watch source on the import port.

Same construction as ``imports/research.py``: subclass ``SyncSourceAdapter``,
inherit the thread hop and the ``drain_errors`` plumbing, keep the file walk
in ``sources/sota_watch.py``.

No ``input_freshness`` here, deliberately. The proposals dir is written by the
sota-watch tooling itself, so its age is a signal about that tool, not about a
human forgetting an export ritual — the case ``SupportsInputFreshness`` exists
for. Claiming freshness knowledge we do not have would put a number in the run
report that nobody can act on.
"""
from __future__ import annotations

import os
from datetime import datetime

from aggregator.imports.sync_bridge import DEFAULT_CHUNK_SIZE, SyncSourceAdapter
from aggregator.sources.sota_watch import SotaWatchSource


class SotaWatchAdapter(SyncSourceAdapter):
    """``SotaWatchSource`` as an ``ImportAdapter``."""

    def __init__(
        self,
        *,
        proposals_dir: str | os.PathLike[str] | None = None,
        source: SotaWatchSource | None = None,
        since: datetime | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        super().__init__(
            source or SotaWatchSource(proposals_dir=proposals_dir),
            since=since,
            chunk_size=chunk_size,
        )


__all__ = ["SotaWatchAdapter"]
