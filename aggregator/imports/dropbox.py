"""The Dropbox source on the import port.

Same construction as ``imports/research.py``. Nothing Dropbox-specific moves
here: the walk, the exclude patterns, the size caps and the no-OCR policy all
stay in ``sources/dropbox.py``.

No ``input_freshness``. Dropbox's own client syncs this tree continuously, so
the input has an owner that keeps it current; the staleness seam is for inputs
that only a human refreshes. A source whose files are old because the user has
not written anything lately is not stale, and reporting it as such would train
the operator to ignore the warning that matters.
"""
from __future__ import annotations

import os
from datetime import datetime

from aggregator.imports.sync_bridge import DEFAULT_CHUNK_SIZE, SyncSourceAdapter
from aggregator.sources.dropbox import DropboxSource


class DropboxAdapter(SyncSourceAdapter):
    """``DropboxSource`` as an ``ImportAdapter``."""

    def __init__(
        self,
        *,
        root: str | os.PathLike[str] | None = None,
        exclude: str | None = None,
        source: DropboxSource | None = None,
        since: datetime | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        super().__init__(
            source or DropboxSource(root=root, exclude=exclude),
            since=since,
            chunk_size=chunk_size,
        )


__all__ = ["DropboxAdapter"]
