"""The claude.ai export source on the import port.

Same construction and the same freshness story as ``imports/chatgpt.py``:
a data-export archive that only a human refreshes, so the run report carries
the input's age rather than reporting success on a months-old zip.

An automated fetch for this source is still an open decision in
``tasks/mission.md`` — and whatever lands there has to fail loudly on an
expired session rather than parse a Cloudflare challenge page. Until it
exists, the age of the last hand-dropped export is the honest signal, and it
is the one thing that tells the operator the index has quietly stopped moving.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from aggregator.imports.sync_bridge import DEFAULT_CHUNK_SIZE, SyncSourceAdapter
from aggregator.sources.claude_web import ClaudeWebSource
from aggregator.sources.exportdrops import downloads_dir, newest_export_mtime


class ClaudeWebAdapter(SyncSourceAdapter):
    """``ClaudeWebSource`` as an ``ImportAdapter``.

    Implements ``SupportsNonFatalErrors`` (inherited) and
    ``SupportsInputFreshness``.
    """

    def __init__(
        self,
        *,
        drops_dir: str | os.PathLike[str] | None = None,
        source: ClaudeWebSource | None = None,
        since: datetime | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        super().__init__(
            source or ClaudeWebSource(drops_dir=str(drops_dir) if drops_dir else None),
            since=since,
            chunk_size=chunk_size,
        )

    def input_freshness(self) -> datetime | None:
        """When the newest claude.ai export was last written, or None.

        Vendor classification is by content, so a fresh ChatGPT drop sitting in
        the same directory does not make this source look current — which is
        exactly the staleness the warning exists to catch.
        """
        return newest_export_mtime(
            "claude-web", dirs=[Path(self._source.drops_dir), downloads_dir()]
        )


__all__ = ["ClaudeWebAdapter"]
