"""Every source, as adapters, in one list.

This is the thing that makes ONE timer possible. Today exactly one source
auto-imports (github) and the rest are hand-run and days to weeks stale; the
whole point of the port was that the runner could drive all of them and the
notify-on-failure wiring got written once instead of eight times. This module
is where "all of them" is spelled out.

A flat function, not a plugin registry or entry points: the set of sources is
known at import time, changes about twice a year, and a decorator-populated
dict would only add a way for a source to go missing without anyone noticing.

Coverage against ``cli.py::_default_sources()`` is asserted by a test. A
source that exists there but not here would never be imported by the run-all
path and would look exactly like a source with nothing new — the failure mode
this repo's fail-loudly constraint exists to rule out.

Construction is side-effect-free (env and path resolution only). No
filesystem walk, no network, no credential read happens until the runner pulls
``get_data()``, so building the full list to print a status is cheap and safe.
"""
from __future__ import annotations

from datetime import datetime

from aggregator.imports.chatgpt import ChatGPTAdapter
from aggregator.imports.claude_web import ClaudeWebAdapter
from aggregator.imports.dropbox import DropboxAdapter
from aggregator.imports.github import GitHubAdapter
from aggregator.imports.port import ImportAdapter
from aggregator.imports.research import ResearchReportsAdapter
from aggregator.imports.sessions import SessionsAdapter
from aggregator.imports.sota_watch import SotaWatchAdapter
from aggregator.imports.substack import SubstackAdapter
from aggregator.imports.ticktick import TickTickAdapter


def default_adapters(since: datetime | None = None) -> list[ImportAdapter]:
    """Build one adapter per source, in a stable order.

    ``since`` is captured at construction because the port is a single-verb
    interface — per-source acquisition knobs belong to the adapter instance,
    not to ``get_data()``.
    """
    return [
        SessionsAdapter(since=since),
        GitHubAdapter(since=since),
        ChatGPTAdapter(since=since),
        ClaudeWebAdapter(since=since),
        ResearchReportsAdapter(since=since),
        SotaWatchAdapter(since=since),
        SubstackAdapter(since=since),
        DropboxAdapter(since=since),
        TickTickAdapter(since=since),
    ]


__all__ = ["default_adapters"]
