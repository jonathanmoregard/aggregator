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

from collections.abc import Callable
from datetime import datetime

from aggregator.imports.chatgpt import ChatGPTAdapter
from aggregator.imports.claude_web import ClaudeWebAdapter
from aggregator.imports.dropbox import DropboxAdapter
from aggregator.imports.github import GitHubAdapter
from aggregator.imports.ingest_state import Watermarks
from aggregator.imports.port import ImportAdapter
from aggregator.imports.research import ResearchReportsAdapter
from aggregator.imports.sessions import SessionsAdapter
from aggregator.imports.sota_watch import SotaWatchAdapter
from aggregator.imports.substack import SubstackAdapter
from aggregator.imports.ticktick import TickTickAdapter


class _UnbuildableAdapter:
    """Stands in for a source whose adapter could not be CONSTRUCTED.

    The runner's failure isolation begins at ``get_data``, so it protected
    every source from every other source's acquisition failing — but not from
    one of them failing to exist. Construction happens in one unprotected list
    in this module, called from ``cli.main``, so a single constructor raising
    (a path that resolved to a file, an env var read at ``__init__`` time)
    killed the whole ``--all`` run before the runner ever saw it: no report, no
    per-source errors, no exit 3, just a traceback. The timer then reports a
    unit failure with nothing saying which of the nine sources was at fault or
    that the other eight never ran.

    Construction here is documented as side-effect-free (env and path
    resolution only), which is exactly why an environment-dependent raise is
    the plausible failure and exactly why it must not be fatal to the run.

    Carrying the intended ``name`` matters: the report is keyed by it, so the
    failure lands on the right source's line instead of appearing as a source
    that silently vanished from the run.
    """

    def __init__(self, name: str, error: BaseException) -> None:
        self.name = name
        self._error = error

    async def get_data(self):
        """Re-raise the construction failure where the runner can contain it."""
        raise RuntimeError(
            f"{self.name}: adapter could not be constructed: "
            f"{type(self._error).__name__}: {self._error}"
        ) from self._error
        yield  # pragma: no cover - unreachable; makes this an async generator


def _build(name: str, factory: Callable[[], ImportAdapter]) -> ImportAdapter:
    try:
        return factory()
    except Exception as e:  # noqa: BLE001 -- isolation boundary, see the class
        return _UnbuildableAdapter(name, e)


def default_adapters(
    since: datetime | None = None,
    *,
    watermarks: Watermarks | None = None,
    now: datetime | None = None,
) -> list[ImportAdapter]:
    """Build one adapter per source, in a stable order.

    ``since`` is captured at construction because the port is a single-verb
    interface — per-source acquisition knobs belong to the adapter instance,
    not to ``get_data()``.

    ``watermarks`` IS THE FIX FOR THE DOOM LOOP, and it is why ``since`` is now
    computed PER SOURCE rather than shared. Until 2026-08-16 this function took
    one ``since`` that ``cli.py`` set only from an explicit ``--since`` flag, so
    every unattended run passed ``None`` to all nine sources and re-ingested the
    entire corpus — 372k observations, ~5 hours, against a 4-hour timeout, every
    30 minutes, forever. Each source now gets its own window, read from its own
    high-water mark and widened by its own overlap; a source whose cursor cannot
    support a window gets ``None`` and the run report says so out loud.

    ``since`` still wins when given: it is a human narrowing this run by hand,
    and ``Watermarks`` carries the same override so the report describes the
    window that was actually used rather than the one on file.

    ``watermarks=None`` means no persistence at all — a full scan of everything,
    which is the right answer for a caller that has no database to record
    against, and the wrong answer for the timer.

    A constructor that raises yields an ``_UnbuildableAdapter`` under the same
    name rather than propagating, so the run-all path keeps the property it
    exists for: one broken source costs its own line in the report and nothing
    else. The name is spelled here rather than read off the built adapter
    because in the failing case there is no adapter to read it from.
    """

    def window(name: str) -> datetime | None:
        if watermarks is None:
            return since
        return watermarks.plan(name, now=now).since

    factories: list[tuple[str, Callable[[], ImportAdapter]]] = [
        ("sessions", lambda: SessionsAdapter(since=window("sessions"))),
        ("github", lambda: GitHubAdapter(since=window("github"))),
        ("chatgpt", lambda: ChatGPTAdapter(since=window("chatgpt"))),
        ("claude-web", lambda: ClaudeWebAdapter(since=window("claude-web"))),
        ("research", lambda: ResearchReportsAdapter(since=window("research"))),
        ("sota-watch", lambda: SotaWatchAdapter(since=window("sota-watch"))),
        ("substack", lambda: SubstackAdapter(since=window("substack"))),
        ("dropbox", lambda: DropboxAdapter(since=window("dropbox"))),
        ("ticktick", lambda: TickTickAdapter(since=window("ticktick"))),
    ]
    return [_build(name, factory) for name, factory in factories]


__all__ = ["default_adapters"]
