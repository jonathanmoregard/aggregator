"""The ``SupportsWriteBarrier`` contract, as a test instead of a docstring.

Round 3 W4: the barrier's rules lived only in prose. Three write paths obey
them — ``imports/runner._run_one``, ``cli._cmd_ingest`` (records) and
``cli._cmd_ingest_entities`` — and a regression in any one of them is silent
data loss for whichever source is holding a pending advance, because the
barrier's whole job is to stop an adapter's state getting ahead of the store.

Everything here drives a PROBE, not TickTick. The rules belong to the port, so
they are asserted against an anonymous adapter that does nothing but implement
the protocol; a "source ten" that grows a barrier tomorrow inherits the same
guarantees without adding a test. The one TickTick-specific check at the bottom
is the reverse direction: that the shipped source really is one of these.
"""
from __future__ import annotations

import asyncio
import contextlib
import sqlite3
from datetime import UTC, datetime

import pytest

from aggregator import cli
from aggregator.core.store import Store
from aggregator.imports.port import SupportsWriteBarrier, WriteCounts
from aggregator.imports.runner import run_imports
from aggregator.imports.sync_bridge import SyncSourceAdapter
from aggregator.imports.ticktick import TickTickAdapter
from aggregator.sources.base import ObservationRow, Record, SessionRow
from aggregator.sources.ticktick import TickTickSource

_TS = datetime(2026, 8, 11, tzinfo=UTC)

# One shared event log per drive: "write" for each batch the sink accepted,
# "commit" for each barrier call. The contract is a statement about this
# sequence, so asserting on it is asserting on the contract itself.


def _record(i: int) -> Record:
    return Record(stable_id=f"probe:{i}", source="probe", subject=f"r{i}", body="b")


def _session(i: int) -> SessionRow:
    return SessionRow(
        session_id=f"probe-s{i}",
        root_session_id=f"probe-s{i}",
        parent_session_id=None,
        kind="session",
        agent_id=None,
        agent_type=None,
        spawned_by_tool_use_id=None,
        cwd=None,
        git_branch=None,
        first_ts=_TS,
        last_ts=_TS,
        jsonl_path=f"/probe/{i}.jsonl",
        origin="probe",
    )


def _observation(i: int) -> ObservationRow:
    return ObservationRow(
        obs_id=f"probe-o{i}",
        session_id=f"probe-s{i}",
        root_session_id=f"probe-s{i}",
        parent_obs_id=None,
        type="user",
        ts=_TS,
        model=None,
        input_tokens=None,
        output_tokens=None,
        tool_name=None,
        tool_use_id=None,
        body="t",
    )


class _Probe:
    """A source/adapter wearing nothing but the barrier protocol."""

    name = "probe"

    def __init__(self, items, *, entities: bool = False, raise_at: int | None = None):
        self.events: list[str] = []
        self._items = list(items)
        self._raise_at = raise_at
        if entities:
            self.iter_entities = self._iterate
        else:
            self.iter_records = self._iterate

    def _iterate(self, since, errors=None):
        for i, item in enumerate(self._items):
            if i == self._raise_at:
                raise RuntimeError("source died mid-stream")
            yield item

    def commit_after_write(self) -> None:
        self.events.append("commit")


class _RecordingSink:
    def __init__(self, probe: _Probe, *, counts=None, boom: bool = False):
        self._probe = probe
        self._counts = counts
        self._boom = boom

    def write(self, items) -> WriteCounts:
        if self._boom:
            raise sqlite3.OperationalError("database is locked")
        self._probe.events.append("write")
        if self._counts is not None:
            return self._counts(items)
        return WriteCounts(added=len(items))


class _RecordingStore(Store):
    """The CLI writes through the store, so that is where its writes show up."""

    probe: _Probe

    def upsert(self, records):
        self.probe.events.append("write")
        return super().upsert(records)

    def upsert_entities(self, entities):
        self.probe.events.append("write")
        return super().upsert_entities(entities)


# -- the three drives, one per write path ---------------------------------


def _drive_runner(probe: _Probe, tmp_path, **kw) -> None:
    # Through the real bridge, because that is what every shipped source is
    # wrapped in and it is the thing that forwards the barrier.
    adapter = SyncSourceAdapter(probe)
    asyncio.run(run_imports([adapter], _RecordingSink(probe, **kw), batch_size=2))


def _drive_cli(probe: _Probe, tmp_path, *, boom: bool = False, **kw) -> None:
    store = _RecordingStore(db_path=tmp_path / "cache.db")
    store.migrate()
    store.probe = probe
    if boom:
        store.upsert = store.upsert_entities = _explode  # type: ignore[method-assign]
    # The CLI lets a store failure propagate, which is itself the contract: it
    # returns before the barrier rather than reporting and carrying on.
    with contextlib.suppress(sqlite3.OperationalError):
        cli.main(["ingest", "probe"], _store=store, _sources={"probe": probe})


def _explode(*a, **kw):
    raise sqlite3.OperationalError("database is locked")


def _drive_cli_entities(probe: _Probe, tmp_path, **kw) -> None:
    _drive_cli(probe, tmp_path, **kw)


ALL_DRIVES = pytest.mark.parametrize(
    ("drive", "entities"),
    [
        (_drive_runner, False),
        (_drive_cli, False),
        (_drive_cli_entities, True),
    ],
    ids=["runner", "cli-records", "cli-entities"],
)


def _items(entities: bool, n: int = 5):
    if entities:
        out: list = []
        for i in range(n):
            out.extend([_session(i), _observation(i)])
        return out
    return [_record(i) for i in range(n)]


# -- the contract ---------------------------------------------------------


@ALL_DRIVES
def test_the_barrier_fires_once_and_only_after_every_write(
    drive, entities, tmp_path
):
    """Rule 1. Once — a barrier run twice saves twice — and last, because
    "the records have landed" is the only thing it may assume."""
    probe = _Probe(_items(entities), entities=entities)

    drive(probe, tmp_path)

    assert probe.events.count("commit") == 1
    assert probe.events[-1] == "commit", (
        f"the barrier must follow every write, got {probe.events}"
    )
    assert "write" in probe.events


@ALL_DRIVES
def test_a_failed_write_does_not_fire_the_barrier(drive, entities, tmp_path):
    """Rule 2, and the reason the protocol exists. The adapter's state must
    not get ahead of a store that never received the rows."""
    probe = _Probe(_items(entities), entities=entities)

    drive(probe, tmp_path, boom=True)

    assert "commit" not in probe.events


@ALL_DRIVES
def test_a_source_that_dies_mid_stream_does_not_fire_the_barrier(
    drive, entities, tmp_path
):
    """Rule 3. A partial run leaves the state untouched and re-derives it
    next time — the items that never arrived are exactly the ones whose
    state advance would be a lie."""
    probe = _Probe(_items(entities), entities=entities, raise_at=4)

    drive(probe, tmp_path)

    assert "commit" not in probe.events


def test_a_sink_that_persisted_nothing_does_not_fire_the_barrier(tmp_path):
    """Rule 4, the runner's alone — the CLI writes through the store, which
    cannot decline. ``WriteCounts.skipped`` is the sink saying it did not
    write, and a stream that ends cleanly having stored nothing must not
    advance anything."""
    probe = _Probe(_items(False), entities=False)

    _drive_runner(probe, tmp_path, counts=lambda items: WriteCounts(skipped=len(items)))

    assert "commit" not in probe.events


def test_an_empty_stream_still_fires_the_barrier(tmp_path):
    """The boundary that must NOT be tightened: a poll that legitimately
    found nothing has a real advance to commit (everything it used to hold
    disappeared), and refusing it would freeze the baseline forever."""
    probe = _Probe([])

    _drive_runner(probe, tmp_path)

    assert probe.events == ["commit"]


def test_a_barrier_that_raises_is_reported_not_swallowed(tmp_path):
    """The records ARE written, so this is not an ingest failure — but an
    adapter whose state never advances silently stops noticing what it
    exists to notice, and nothing else in the run would say so."""

    class _Angry(_Probe):
        def commit_after_write(self) -> None:
            raise OSError("read-only file system")

    probe = _Angry([_record(0)])
    report = asyncio.run(
        run_imports([SyncSourceAdapter(probe)], _RecordingSink(probe))
    )

    assert report.ok is False
    assert any("commit_after_write" in e for e in report.errors)


# -- the shipped source really is one of these ----------------------------


def test_ticktick_is_a_write_barrier_adapter_at_both_seams():
    """The reverse direction: the contract above is worth nothing if the one
    source that needs it does not actually reach either call site. The
    adapter is checked structurally (the runner's test) and the source by
    attribute (``cli._commit_after_write`` does a getattr)."""
    adapter = TickTickAdapter(source=TickTickSource())
    assert isinstance(adapter, SupportsWriteBarrier)
    assert callable(getattr(TickTickSource(), "commit_after_write", None))
