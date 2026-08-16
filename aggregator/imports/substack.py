"""The Substack export source on the import port.

Same construction and the same freshness story as ``imports/chatgpt.py``. The
input is a zip produced from Settings → Exports and downloaded by hand; the
newest one on disk is 31 days old as of this migration, which is the concrete
case that motivated ``SupportsInputFreshness`` in the first place.

The zip's own mtime is the signal, not its members': the posts inside carry
publication dates going back years, while the question a staleness warning
answers is when a human last fetched an export.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from aggregator.imports.sync_bridge import DEFAULT_CHUNK_SIZE, SyncSourceAdapter
from aggregator.sources.exportdrops import downloads_dir, newest_export_mtime
from aggregator.sources.substack import SubstackSource


class SubstackAdapter(SyncSourceAdapter):
    """``SubstackSource`` as an ``ImportAdapter``.

    Implements ``SupportsNonFatalErrors`` (inherited) and
    ``SupportsInputFreshness``.
    """

    def __init__(
        self,
        *,
        drops_dir: str | os.PathLike[str] | None = None,
        source: SubstackSource | None = None,
        since: datetime | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        super().__init__(
            source or SubstackSource(drops_dir=str(drops_dir) if drops_dir else None),
            since=since,
            chunk_size=chunk_size,
        )

    def input_freshness(self) -> datetime | None:
        """When the newest Substack export zip was last written, or None."""
        return newest_export_mtime(
            "substack", dirs=[Path(self._source.drops_dir), downloads_dir()]
        )


__all__ = ["SubstackAdapter"]
