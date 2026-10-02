"""Retry a network call through a transient upstream failure.

A single HTTP 500 from one TickTick project failed two whole ingest runs on
2026-10-02 and woke the failure notifier for an outage that healed by the next
request. Each source owns the judgement of what is transient (it knows its own
transport's error shapes); this module owns the schedule, so every source
backs off the same way.

The schedule is short on purpose. It rides out a blip, not an outage: a source
that is still failing after the last attempt raises exactly as it did before,
and the run reports it.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")

#: Seconds to wait before each retry. Two retries, so three attempts in all.
DEFAULT_DELAYS: tuple[float, ...] = (2.0, 8.0)

# Module attribute rather than a direct ``time.sleep`` call so tests can
# replace it without patching the global ``time`` module.
_sleep = time.sleep


def call_with_retry(
    fn: Callable[[], T],
    *,
    is_transient: Callable[[BaseException], bool],
    what: str,
    delays: tuple[float, ...] = DEFAULT_DELAYS,
) -> T:
    """Return ``fn()``, retrying after each transient failure in ``delays``.

    A failure ``is_transient`` rejects propagates at once; the last transient
    failure propagates once ``delays`` is spent. ``what`` names the call in the
    retry log line and must not carry a credential.
    """
    for delay in delays:
        try:
            return fn()
        except Exception as e:
            if not is_transient(e):
                raise
            log.warning("%s: transient failure (%s); retrying in %.0fs", what, e, delay)
            _sleep(delay)
    return fn()
