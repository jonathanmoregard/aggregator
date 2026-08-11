"""The Claude Code sessions source on the import port.

Same construction as ``imports/research.py``, but this is the entity-shaped
one: ``SyncSourceAdapter`` finds ``iter_entities`` first, so the adapter
yields ``SessionRow`` / ``ObservationRow`` and ``StoreSink`` routes them to
``upsert_entities`` — dispatch by type, never by the adapter declaring a
table.

This is also the source the port's streaming contract exists for: ~359k
observations, which a list-returning ``get_data`` would hold in memory in
full while the sink writes. Chunked hand-off through the bridge keeps that
bounded.

No ``input_freshness``: ``~/.claude/projects`` is appended to by Claude Code
every session. There is no export ritual to forget, and the source already
skips files touched in the last five minutes precisely because they are being
written right now.

``--rebuild`` is NOT reachable from here, deliberately. The CLI's entity
rebuild replaces the whole sessions + observations tables and would wipe the
chat-export origins; the runner path is upsert-only, which is idempotent per
session/observation id.
"""
from __future__ import annotations

import os
from datetime import datetime

from aggregator.imports.sync_bridge import DEFAULT_CHUNK_SIZE, SyncSourceAdapter
from aggregator.sources.sessions import SessionsSource


class SessionsAdapter(SyncSourceAdapter):
    """``SessionsSource`` as an ``ImportAdapter``."""

    def __init__(
        self,
        *,
        projects_root: str | os.PathLike[str] | None = None,
        source: SessionsSource | None = None,
        since: datetime | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        super().__init__(
            source
            or SessionsSource(
                projects_root=str(projects_root) if projects_root else None
            ),
            since=since,
            chunk_size=chunk_size,
        )


__all__ = ["SessionsAdapter"]
