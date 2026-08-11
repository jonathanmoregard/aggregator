"""The ChatGPT export source on the import port.

Same construction as ``imports/research.py`` for the acquisition plumbing; the
parsing stays in ``sources/chatgpt.py``. What it adds is ``input_freshness``,
and for this source that is the entire reason the adapter is interesting.

ChatGPT has no consumer history API, and the session constraint of 2026-08-11
is explicit that it gets NO scraper: request-and-wait export, up to seven days,
a link that expires in 24 hours, and a login behind Cloudflare — it loses on
low upkeep, low manual work and robustness at once. "Do NOT build the fragile
one and call it done — build the dull one and make the gap loud" is what this
method is. The zip in ~/Downloads only moves when a human moves it, so the
run report has to say how old it is rather than reporting a cheerful success
on an export from July.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from aggregator.imports.sync_bridge import DEFAULT_CHUNK_SIZE, SyncSourceAdapter
from aggregator.sources.chatgpt import ChatGPTSource
from aggregator.sources.exportdrops import downloads_dir, newest_export_mtime


class ChatGPTAdapter(SyncSourceAdapter):
    """``ChatGPTSource`` as an ``ImportAdapter``.

    Implements both optional port protocols: ``SupportsNonFatalErrors``
    (inherited — a corrupt drop is recorded and the rest of the export still
    imports) and ``SupportsInputFreshness``.
    """

    def __init__(
        self,
        *,
        drops_dir: str | os.PathLike[str] | None = None,
        source: ChatGPTSource | None = None,
        since: datetime | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        super().__init__(
            source or ChatGPTSource(drops_dir=str(drops_dir) if drops_dir else None),
            since=since,
            chunk_size=chunk_size,
        )

    def input_freshness(self) -> datetime | None:
        """When the newest ChatGPT export was last written, or None.

        Scans the same two dirs the ingest does (the source's drops dir plus
        the live ~/Downloads), so the answer describes the input this adapter
        would actually read. ``None`` means no export was found — unknown,
        not fresh.
        """
        return newest_export_mtime(
            "chatgpt", dirs=[Path(self._source.drops_dir), downloads_dir()]
        )


__all__ = ["ChatGPTAdapter"]
