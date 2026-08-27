"""The embedder honours a thread cap the environment asked for.

THE FAN IS THE SYMPTOM AND THE CGROUP IS NOT THE WHOLE FIX. Measured from
``aggregator-embed.service``'s own accounting on 2026-08-27: **1d 7h 51min of
CPU time over 4h 3s of wall clock**, a sustained ~8x parallelism on a 12-core
laptop. ``Nice=19`` was already set and did nothing about it — nice is
PRIORITY, not QUANTITY, so every core still ran flat out and the machine still
had to move that heat.

A cgroup quota alone is not the whole fix either, and the direction it fails
in is the counter-intuitive one: with ``CPUQuota=400%`` and twelve torch
threads, all twelve threads still get scheduled and are then throttled
together, so the process pays full context-switching and cache-thrash for a
third of the throughput. The thread pool has to shrink WITH the quota, not
under it.

So the cap is expressed twice — ``OMP_NUM_THREADS`` in the unit's
``Environment=`` (read by torch's OpenMP pool at import) plus ``CPUQuota=`` as
the bound that survives anything the process does to itself — and this module
pins the half that lives in Python: torch's own intra-op pool, which is a
SEPARATE knob from the OpenMP one and is not set by that variable on every
build.

Nothing here loads a model. ``_pin_thread_pools`` is a pure function of the
environment and a setter, so it is tested against a fake setter.
"""

from __future__ import annotations

import pytest

from aggregator.core.embed import _pin_thread_pools


class _Recorder:
    """Stands in for ``torch.set_num_threads``."""

    def __init__(self):
        self.calls: list[int] = []

    def __call__(self, n: int) -> None:
        self.calls.append(n)


def test_it_pins_the_pool_to_what_the_environment_asked_for(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    setter = _Recorder()

    assert _pin_thread_pools(setter) == 4
    assert setter.calls == [4]


def test_an_unset_variable_leaves_the_pool_alone(monkeypatch):
    """No cap asked for, no cap applied — and NOT a default of 1.

    The MCP server builds an ``Embedder`` too, and a query there is one short
    encode a human is waiting on. Pinning it to the background worker's cap
    would trade a fan nobody hears during a search for latency they do. The
    unit that wants the cap is the unit that sets the variable.
    """
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    setter = _Recorder()

    assert _pin_thread_pools(setter) is None
    assert setter.calls == []


@pytest.mark.parametrize("value", ["", "   ", "many", "4.5", "0", "-2"])
def test_a_value_that_is_not_a_positive_count_is_ignored(value, monkeypatch):
    """Garbage in the environment must not crash the worker or pin it to zero.

    ``torch.set_num_threads(0)`` is not a no-op — it is undefined behaviour
    that has been observed to abort. This runs unattended from a timer, so the
    failure mode of a typo in a unit file has to be "the cap is not applied",
    never "the backfill dies twice an hour".
    """
    monkeypatch.setenv("OMP_NUM_THREADS", value)
    setter = _Recorder()

    assert _pin_thread_pools(setter) is None
    assert setter.calls == []


def test_a_setter_that_raises_does_not_escape(monkeypatch):
    """A torch build without the setter must not take the run down with it.

    The cap is an optimisation for the operator's comfort. It is never worth
    more than the backfill it is throttling, so it fails open — loudly in the
    log, quietly in the process.
    """
    monkeypatch.setenv("OMP_NUM_THREADS", "4")

    def explode(_n: int) -> None:
        raise RuntimeError("this build has no intra-op pool")

    assert _pin_thread_pools(explode) is None
