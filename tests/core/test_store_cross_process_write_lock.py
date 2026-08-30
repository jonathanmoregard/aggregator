"""Two aggregator processes must not be able to make each other fail.

OBSERVED IN PRODUCTION, 2026-08-30, on EVERY tick for at least five hours.
``aggregator-embed.service`` died with ``sqlite3.OperationalError: database is
locked`` ten runs out of ten between 19:15 and 23:47, each failure timestamped
inside a window in which ``aggregator-ingest.service`` was running. Twice the
death was the unhandled traceback at startup — ``cli._cmd_embed`` ->
``Store.migrate`` -> the ``embedding_versions`` INSERT — 33 seconds after the
run began, which is the 30s ``busy_timeout`` expiring plus the statement that
gave up. Eight times it was the batch write, after up to 47 minutes of encoder
work that then had nowhere to land.

WHY THE PREVIOUS DESIGN COULD NOT HOLD. ``nix/aggregator.nix`` stated the
correctness story as "WAL journal mode plus a 30s busy_timeout, and on the
worker side one short write transaction per batch rather than one long one per
run". Both halves are true of the embed worker and the second half is simply
false of ingest: ``cli._ingest_entities`` calls ``store.upsert_entities(
entities)`` once for a whole source, and ``_do_write_entities`` chunks only the
hash PROBE, not the commit. One transaction covers every changed row.

Measured on the live cache 2026-08-31 by sampling ``BEGIN IMMEDIATE`` at 1Hz
through an ordinary (light, five-minute) ingest run: the write lock was held in
contiguous blocks of 49s, 29s, 139s, 25s and 23s. A 30s timeout against a 139s
transaction is not a near miss that a bigger number would fix — and the corpus
only grows, so every number is the wrong number eventually.

WHAT THESE TESTS PIN is therefore not a duration. It is that the two writers
take an OS-level lock BEFORE either asks SQLite for anything, so the losing
writer waits on a lock designed for waiting instead of racing one designed for
failing. The timeout is turned down to milliseconds here precisely so that a
test which passes because it was lucky cannot exist: at 200ms, a two-second
transaction in another process is fatal to any writer not holding the baton.
"""

import os
import subprocess
import sys
import textwrap
import time

import pytest

from aggregator.core import store as store_mod
from aggregator.core.store import Store

#: How long the stand-in for ingest keeps its transaction open.
#:
#: Two seconds is 10x the shortened ``busy_timeout`` below and 1/70th of the
#: 139s the real thing was measured holding. Big enough that no retry schedule
#: can accidentally survive it, small enough to stay a unit test.
_HOLD_SECONDS = 2.0

#: What the writer's own busy handler is allowed to absorb during these tests.
#:
#: DELIBERATELY FAR TOO SMALL to survive ``_HOLD_SECONDS``. If a change ever
#: makes these tests pass by waiting SQLite out rather than by holding the
#: cross-process lock, that change has reintroduced the production bug with a
#: bigger number in front of it, and this value is what refuses to let it look
#: green.
_TEST_BUSY_TIMEOUT_MS = 200


def _holder_script(db, ready, hold_seconds=_HOLD_SECONDS):
    """A separate PROCESS that writes like ingest and holds like ingest.

    A THREAD WOULD PROVE NOTHING. ``flock`` is held per open file description,
    and the store keeps one per process, so two threads share the baton and
    never contend for it. The bug is between processes and so is the test.

    The write is ``advance_ingest_cursor(..., _commit=False)`` because that is
    literally the shape ingest holds open: a real write on the real store, with
    the commit deferred, which is the state ``upsert_entities`` sits in for
    every one of the minutes it spends walking a source.
    """
    return textwrap.dedent(
        f"""
        import time
        from aggregator.core.store import Store

        s = Store(db_path={str(db)!r})
        s.migrate()
        s.advance_ingest_cursor(
            "sessions",
            cursor_kind="ts",
            cursor_value="2026-01-01T00:00:00+00:00",
            rows=1,
            at="2026-01-01T00:00:00+00:00",
            _commit=False,
        )
        # Only now: the transaction is open and the write lock is genuinely
        # held. Announcing earlier would let the test race the setup rather
        # than the lock.
        open({str(ready)!r}, "w").write("held")
        time.sleep({hold_seconds})
        s.commit()
        s.close()
        """
    )


def _spawn(script):
    env = dict(os.environ)
    # The worktree, not whatever `aggregator` is installed globally — these
    # tests are meaningless run against a build that predates the lock.
    env["PYTHONPATH"] = str(
        os.path.dirname(os.path.dirname(os.path.dirname(store_mod.__file__)))
    )
    return subprocess.Popen(
        [sys.executable, "-c", script],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


#: How long to let the holder process get as far as opening its transaction.
#:
#: GENEROUS ON PURPOSE, because this is setup and not the property under test.
#: The holder pays for a fresh interpreter, the ``aggregator.core.store`` import
#: chain (which reaches Presidio) and a ``migrate()`` before it can announce
#: anything, and these tests run on a machine whose whole point is a background
#: embed worker at ``Nice=19`` eating every core. At 30s this was observed
#: failing in a full-suite run and passing standalone — i.e. measuring the load
#: on the box rather than the lock. What the lock does is asserted afterwards,
#: against ``_HOLD_SECONDS``, and no amount of slow setup can make that pass.
_READY_TIMEOUT = 120.0


def _await_ready(proc, ready, timeout=_READY_TIMEOUT):
    """Block until the holder says its transaction is open, or explain why not."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if ready.exists():
            return
        if proc.poll() is not None:
            out, err = proc.communicate()
            pytest.fail(f"holder process died before holding:\n{out}\n{err}")
        time.sleep(0.02)
    proc.kill()
    pytest.fail("holder process never opened its transaction")


@pytest.fixture
def cache(tmp_path):
    db = tmp_path / "cache.db"
    s = Store(db_path=db)
    s.migrate()
    s.close()
    return db


@pytest.fixture
def impatient(monkeypatch):
    """Give the writer under test a busy handler that cannot save it."""
    monkeypatch.setattr(store_mod, "_BUSY_TIMEOUT_MS", _TEST_BUSY_TIMEOUT_MS)


def test_a_write_waits_out_another_process_instead_of_failing_locked(
    cache, impatient, tmp_path
):
    """THE PRODUCTION REGRESSION, at the smallest layer that carries it.

    Ingest holds a long write transaction; embed writes into it. Before the
    cross-process lock this raised ``database is locked`` — ten times out of
    ten on the deployed machine, and deterministically here, because 200ms of
    busy handler cannot outlast a two-second transaction.

    The assertion is that the write SUCCEEDS, not that it is fast. Waiting is
    the correct behaviour and the embed unit is built for it
    (``TimeoutStartSec=infinity``, a watermark it resumes from). What is not
    correct is throwing away up to 47 minutes of encoder work because another
    process was mid-commit.
    """
    ready = tmp_path / "held"
    holder = _spawn(_holder_script(cache, ready))
    try:
        _await_ready(holder, ready)

        started = time.monotonic()
        s = Store(db_path=cache)
        try:
            # The embed worker's own first write is ``migrate()``'s INSERT into
            # embedding_versions — store.py:1386 in the production traceback.
            s.migrate()
        finally:
            s.close()
        waited = time.monotonic() - started
    finally:
        holder.kill()
        holder.wait()

    assert waited >= _HOLD_SECONDS * 0.5, (
        f"the write returned in {waited:.2f}s, faster than the {_HOLD_SECONDS}s "
        f"transaction it was supposed to be queued behind. It cannot have "
        f"waited for the baton, so this test is no longer observing the thing "
        f"it was written to observe."
    )


def test_the_baton_is_handed_back_when_the_transaction_ends(cache, impatient, tmp_path):
    """A lock that is never released is not a fix, it is a nicer-looking outage.

    The failure this guards is the easy way to make the test above pass: hold
    the lock for the life of the process. That would serialise the two units
    down to one, turning a 30-minute embed tick into a 30-minute block on
    ingest — which is the same starvation with the blame moved.

    So: after the holder commits and exits, an unrelated process must be able
    to take the baton immediately. Sequential, not concurrent, on purpose —
    concurrency is the previous test's job and would only blur this one.
    """
    ready = tmp_path / "held"
    first = _spawn(_holder_script(cache, ready, hold_seconds=0.1))
    out, err = first.communicate(timeout=_READY_TIMEOUT)
    assert first.returncode == 0, (
        f"the first holder failed to commit and exit:\n{out}\n{err}"
    )
    assert ready.exists(), "the first holder never opened a transaction to release"

    # A SECOND PROCESS, not a second Store in this one: an in-process caller
    # shares the descriptor and would be handed the baton by the holder count
    # whether or not the kernel lock was ever dropped.
    ready2 = tmp_path / "held2"
    second = _spawn(_holder_script(cache, ready2, hold_seconds=0.1))
    out, err = second.communicate(timeout=_READY_TIMEOUT)
    assert second.returncode == 0, (
        f"a second process could not take the baton after the first committed. "
        f"A lock held past its transaction serialises the two units down to "
        f"one, which is the outage again with the blame moved:\n{out}\n{err}"
    )


def test_a_reader_is_never_made_to_wait_for_the_writer(cache, tmp_path):
    """WAL's whole bargain must survive the lock.

    The baton is taken by ``_ensure_writable``, which read-only stores never
    reach — so ``aggregator_search_memory`` answering out of the MCP must stay
    exactly as available during a long ingest as it was before. If this ever
    goes red, the lock has been moved somewhere that costs reads, and the fix
    for a background backfill has been paid for by the interactive path.
    """
    ready = tmp_path / "held"
    holder = _spawn(_holder_script(cache, ready))
    try:
        _await_ready(holder, ready)
        started = time.monotonic()
        r = Store(db_path=cache, read_only=True)
        try:
            r._c().execute("SELECT count(*) FROM sessions").fetchone()
        finally:
            r.close()
        assert time.monotonic() - started < _HOLD_SECONDS / 2, (
            "a read blocked behind a write transaction"
        )
    finally:
        holder.kill()
        holder.wait()
