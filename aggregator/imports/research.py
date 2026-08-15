"""Reference adapter: the research-reports source on the import port.

Picked as the proof of the seam because it is the simplest real source — a
local directory of markdown, record-shaped, already idempotent per
stable_id, no credentials, no export ritual. Everything specific to it lives
in ``sources/research_reports.py`` still; this module only says how it is
acquired.

Additive by design: ``cli.py::_default_sources()`` keeps registering the
same ``ResearchReportsSource`` and the existing ``aggregator ingest
research`` path is untouched. Migrating the remaining sources, and the CLI
itself, are later tasks.
"""
from __future__ import annotations

import os
from datetime import datetime

from aggregator.imports.sync_bridge import DEFAULT_CHUNK_SIZE, SyncSourceAdapter
from aggregator.sources.research_reports import ResearchReportsSource


class ResearchReportsAdapter(SyncSourceAdapter):
    """``ResearchReportsSource`` as an ``ImportAdapter``.

    Subclassing ``SyncSourceAdapter`` rather than re-implementing the bridge:
    the file walk stays synchronous (and stays in the source module), the
    thread hop and the ``drain_errors`` plumbing are inherited. That is the
    whole point of the bridge — a source becomes an adapter by construction,
    not by rewrite.
    """

    def __init__(
        self,
        *,
        reports_dir: str | os.PathLike[str] | None = None,
        source: ResearchReportsSource | None = None,
        since: datetime | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        super().__init__(
            source or ResearchReportsSource(reports_dir=reports_dir),
            since=since,
            chunk_size=chunk_size,
        )


__all__ = ["ResearchReportsAdapter"]
