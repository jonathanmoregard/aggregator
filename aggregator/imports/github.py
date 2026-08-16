"""The GitHub source on the import port.

github is the only source that already auto-imports (a systemd user timer runs
``aggregator ingest github``), so this adapter's job is to change nothing
observable: same ``iter_records`` walk, same fail-closed scope check, same
per-endpoint error policy. It exists so the one timer this repo is heading
towards can drive github alongside the other seven instead of beside them.

``source`` is the injection seam. ``GitHubSource`` already carries its own
``_scope_fetcher`` / ``_api_fetcher`` / ``_gh_token_fetcher`` seams, and
re-exporting all three through the adapter would duplicate a surface that is
going to drift. Tests build the source and hand it over.

No ``input_freshness``: github is a live API. Its records are as current as
the last successful call, so there is no manually-refreshed input to go stale
— a failed call is already an error, which is a louder signal than an age.
"""
from __future__ import annotations

from datetime import datetime

from aggregator.imports.sync_bridge import DEFAULT_CHUNK_SIZE, SyncSourceAdapter
from aggregator.sources.github import GitHubSource


class GitHubAdapter(SyncSourceAdapter):
    """``GitHubSource`` as an ``ImportAdapter``."""

    def __init__(
        self,
        *,
        source: GitHubSource | None = None,
        since: datetime | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        super().__init__(
            source or GitHubSource(),
            since=since,
            chunk_size=chunk_size,
        )


__all__ = ["GitHubAdapter"]
