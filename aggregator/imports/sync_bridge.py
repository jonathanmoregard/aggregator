"""Adapt the existing SYNCHRONOUS sources onto the async import port.

Why a bridge instead of async sources: ``ticktick_api.py`` is deliberately
stdlib ``urllib`` with GET-only + https-only + unredirected-Authorization
hardening, and the other sources are file walkers. Rewriting eight working,
tested sources onto httpx/aiohttp to satisfy an async seam would trade real
hardening for interface purity. Instead the sync internals stay exactly as
they are and get adapted at the boundary: the blocking work runs in a worker
thread via ``asyncio.to_thread`` so it can't stall the adapters running
concurrently beside it, and items are handed over in chunks as they are
produced rather than materialised first.

Thread caveat: consecutive chunks may be pulled on different pool threads
(never concurrently — each is awaited before the next is requested). That is
fine for file and HTTP sources. A sync iterator holding a thread-affine
resource — an open ``sqlite3`` connection is the one that bites in this
codebase — must not be wrapped here.
"""
from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from datetime import datetime
from typing import Any, TypeVar

from aggregator.imports.port import ImportItem

T = TypeVar("T")

# Items pulled per thread hop. Big enough to amortise the hop over a source
# yielding hundreds of thousands of rows, small enough that the consumer
# still sees data early and memory stays bounded.
DEFAULT_CHUNK_SIZE = 256


def _next_chunk(
    iterator: Iterator[T], size: int
) -> tuple[list[T], Exception | None]:
    """Pull up to ``size`` items. Runs in the worker thread.

    Returns whatever was pulled ALONGSIDE the failure, rather than letting the
    failure discard it. A source that dies part-way through a chunk had already
    produced up to ``size - 1`` items, and raising from here threw every one of
    them away before the consumer saw them — the runner's "whatever arrived
    before the crash still gets written" only ever applied to whole chunks that
    had already crossed the thread boundary. At the default chunk size that is
    up to 255 items lost per crash.

    ``Exception``, not ``BaseException``: CancelledError and KeyboardInterrupt
    mean the whole run is being torn down and must keep propagating, the same
    rule the runner's isolation boundary follows.
    """
    chunk: list[T] = []
    for _ in range(size):
        try:
            chunk.append(next(iterator))
        except StopIteration:
            break
        except Exception as e:  # noqa: BLE001 -- handed back, never swallowed
            return chunk, e
    return chunk, None


async def aiter_in_thread(
    make_iterator: Callable[[], Iterable[T]],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> AsyncIterator[T]:
    """Run a sync iterable in a worker thread, yielding as it goes.

    ``make_iterator`` is a zero-arg callable so construction (which for a
    generator function is where a bad signature blows up) also happens off
    the loop. Exceptions raised by the sync side propagate to the caller
    unchanged — the runner turns them into a per-adapter error, and nothing
    is silently dropped.

    Items pulled BEFORE the failure are yielded first, then the exception is
    re-raised. Partial ingest beats total loss is the policy everywhere else in
    this pipeline, and it did not hold here: a raise mid-chunk discarded up to
    ``chunk_size - 1`` items the source had already produced.
    """
    iterator = await asyncio.to_thread(lambda: iter(make_iterator()))
    while True:
        chunk, error = await asyncio.to_thread(_next_chunk, iterator, chunk_size)
        for item in chunk:
            yield item
        if error is not None:
            raise error
        if not chunk:
            return


def accepts_errors_kwarg(fn: Callable[..., Any]) -> bool:
    """True when ``fn`` takes an ``errors`` sink.

    Signature inspection rather than a ``try``/``except TypeError`` around the
    call: that pattern also catches a genuine TypeError raised from inside the
    source, which looks exactly like an old signature and silently re-runs the
    iteration. Public, and imported by ``cli.py``, so both ingest surfaces
    decide this the same way — the CLI carried the try/except version until
    round 2, where a re-poll cost TickTick a whole poll's inferred completions.
    """
    try:
        return "errors" in inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins/C callables
        return False


class SyncSourceAdapter:
    """Wrap an existing sync ``Source`` as an ``ImportAdapter``.

    Handles both store shapes: entity-shaped sources (``iter_entities`` →
    sessions/observations) are preferred when present, else record-shaped
    (``iter_records`` → records), mirroring the CLI's dispatch order so the
    two paths can't disagree about a source that grows both.

    ``since`` is captured here, not passed to ``get_data()`` — the port is a
    single-verb interface and per-source acquisition knobs belong to the
    adapter instance.
    """

    def __init__(
        self,
        source: Any,
        *,
        since: datetime | None = None,
        name: str | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        iter_fn = getattr(source, "iter_entities", None) or getattr(
            source, "iter_records", None
        )
        if iter_fn is None:
            raise TypeError(
                f"{type(source).__name__} exposes neither iter_entities nor "
                f"iter_records; it cannot be adapted to the import port"
            )
        self._source = source
        self._iter_fn = iter_fn
        self._since = since
        self._chunk_size = chunk_size
        self.name: str = name or getattr(source, "name", type(source).__name__)
        self._errors: list[str] = []

    def get_data(self) -> AsyncIterator[ImportItem]:
        """Stream the wrapped source's items off the event-loop thread."""
        self._errors = []

        def make_iterator() -> Iterable[ImportItem]:
            if accepts_errors_kwarg(self._iter_fn):
                return self._iter_fn(self._since, errors=self._errors)
            return self._iter_fn(self._since)

        return aiter_in_thread(make_iterator, chunk_size=self._chunk_size)

    def drain_errors(self) -> list[str]:
        """Hand the source's per-item failures to the runner, then forget them.

        Sources append unreadable files here and keep going (partial ingest
        beats total loss). Draining them into the run report is what keeps
        that from becoming a silent gap: a run ending with a non-empty errors
        list still trips the failure notification.
        """
        errors, self._errors = self._errors, []
        return errors


__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "SyncSourceAdapter",
    "accepts_errors_kwarg",
    "aiter_in_thread",
]
