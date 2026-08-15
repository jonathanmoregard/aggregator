"""Tests for aggregator.imports.sync_bridge.

The existing sources are synchronous on purpose — ``ticktick_api.py`` is
deliberately stdlib ``urllib`` with GET-only + https-only + unredirected
Authorization hardening, and rewriting eight sources onto httpx/aiohttp to
satisfy an async port would be the tail wagging the dog. So the sync
internals stay and get adapted at the boundary, in a worker thread, yielding
as they go.
"""
from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest

from aggregator.imports.port import ImportAdapter, SupportsNonFatalErrors
from aggregator.imports.sync_bridge import SyncSourceAdapter, aiter_in_thread
from aggregator.sources.base import ObservationRow, Record, SessionRow


def _rec(n: str) -> Record:
    return Record(stable_id=f"t:{n}", source="t", subject=n, body=n)


async def _drain(aiterator) -> list:
    return [item async for item in aiterator]


def test_aiter_in_thread_yields_every_item_in_order():
    items = asyncio.run(_drain(aiter_in_thread(lambda: iter([1, 2, 3, 4, 5]))))
    assert items == [1, 2, 3, 4, 5]


def test_sync_work_happens_off_the_event_loop_thread():
    """The point of ``asyncio.to_thread``: a blocking urllib GET or a big
    file read must not stall the other adapters running concurrently."""
    main = threading.current_thread()
    seen: list[threading.Thread] = []

    def produce() -> Iterator[int]:
        for i in range(3):
            seen.append(threading.current_thread())
            yield i

    asyncio.run(_drain(aiter_in_thread(produce)))
    assert seen, "generator never ran"
    assert all(t is not main for t in seen)


def test_aiter_in_thread_streams_rather_than_draining_first():
    produced = 0
    consumed_at: list[int] = []

    def produce() -> Iterator[int]:
        nonlocal produced
        for i in range(6):
            produced += 1
            yield i

    async def consume() -> None:
        async for _ in aiter_in_thread(produce, chunk_size=2):
            consumed_at.append(produced)

    asyncio.run(consume())
    # With chunk_size=2 the first two items are consumed while only 3 have
    # been produced (the third is the one that fills the next chunk request),
    # never all 6 up front.
    assert consumed_at[0] < 6
    assert consumed_at == sorted(consumed_at)


def test_exception_from_the_sync_iterator_propagates():
    def produce() -> Iterator[int]:
        yield 1
        raise OSError("disk gone")

    with pytest.raises(OSError, match="disk gone"):
        asyncio.run(_drain(aiter_in_thread(produce, chunk_size=1)))


def test_items_pulled_before_a_mid_chunk_crash_still_reach_the_consumer():
    """Round-1 LOW. ``_next_chunk`` raised from inside the pull loop, throwing
    away everything it had already collected — up to 255 items at the default
    chunk size. "Partial ingest beats total loss" is the policy everywhere else
    in this pipeline and did not hold here."""
    seen: list[int] = []

    def produce() -> Iterator[int]:
        yield from range(7)
        raise OSError("disk gone")

    async def drive() -> None:
        async for item in aiter_in_thread(produce, chunk_size=256):
            seen.append(item)

    with pytest.raises(OSError, match="disk gone"):
        asyncio.run(drive())

    assert seen == list(range(7))


def test_a_crash_on_a_later_chunk_keeps_the_earlier_ones_and_its_own_partial():
    seen: list[int] = []

    def produce() -> Iterator[int]:
        yield from range(5)
        raise RuntimeError("upstream 500")

    async def drive() -> None:
        async for item in aiter_in_thread(produce, chunk_size=2):
            seen.append(item)

    with pytest.raises(RuntimeError, match="upstream 500"):
        asyncio.run(drive())

    # 0,1 | 2,3 | 4 then the raise — the partial final chunk is not lost.
    assert seen == [0, 1, 2, 3, 4]


class FakeRecordSource:
    """Shaped like research/sota-watch: iter_records(since, errors=...)."""

    name = "faker"

    def __init__(self, records: list[Record]) -> None:
        self._records = records
        self.seen_since: object = "unset"

    def iter_records(
        self, since: datetime | None, errors: list[str] | None = None
    ) -> Iterator[Record]:
        self.seen_since = since
        yield from self._records
        if errors is not None:
            errors.append("one.md: read failed")


class FakeEntitySource:
    """Shaped like sessions/chatgpt: iter_entities(since, errors=...)."""

    name = "entities"

    def __init__(self, entities: list) -> None:
        self._entities = entities

    def iter_entities(
        self, since: datetime | None, errors: list[str] | None = None
    ) -> Iterator:
        yield from self._entities


class LegacyRecordSource:
    """Older signature with no ``errors`` sink — must still adapt."""

    name = "legacy"

    def iter_records(self, since: datetime | None) -> Iterator[Record]:
        yield _rec("legacy1")


def test_adapts_a_record_shaped_sync_source():
    source = FakeRecordSource([_rec("a"), _rec("b")])
    adapter = SyncSourceAdapter(source)

    items = asyncio.run(_drain(adapter.get_data()))

    assert isinstance(adapter, ImportAdapter)
    assert adapter.name == "faker"
    assert [r.stable_id for r in items] == ["t:a", "t:b"]


def test_adapts_an_entity_shaped_sync_source():
    now = datetime(2026, 8, 11, tzinfo=UTC)
    session = SessionRow(
        session_id="s1",
        root_session_id="s1",
        parent_session_id=None,
        kind="session",
        agent_id=None,
        agent_type=None,
        spawned_by_tool_use_id=None,
        cwd=None,
        git_branch=None,
        first_ts=now,
        last_ts=now,
        jsonl_path="/tmp/s1.jsonl",
    )
    obs = ObservationRow(
        obs_id="o1",
        session_id="s1",
        root_session_id="s1",
        parent_obs_id=None,
        type="user",
        ts=now,
        model=None,
        input_tokens=None,
        output_tokens=None,
        tool_name=None,
        tool_use_id=None,
        body="hi",
    )
    adapter = SyncSourceAdapter(FakeEntitySource([session, obs]))

    items = asyncio.run(_drain(adapter.get_data()))

    assert items == [session, obs]


def test_since_is_captured_at_construction_not_passed_through_get_data():
    """The port is a single-verb interface; per-source acquisition knobs are
    constructor arguments, so ``get_data()`` stays parameterless."""
    since = datetime(2026, 8, 1, tzinfo=UTC)
    source = FakeRecordSource([_rec("a")])
    adapter = SyncSourceAdapter(source, since=since)

    asyncio.run(_drain(adapter.get_data()))

    assert source.seen_since == since


def test_per_file_errors_are_exposed_through_drain_errors():
    adapter = SyncSourceAdapter(FakeRecordSource([_rec("a")]))
    assert isinstance(adapter, SupportsNonFatalErrors)

    asyncio.run(_drain(adapter.get_data()))

    assert adapter.drain_errors() == ["one.md: read failed"]
    # Drained means drained — a second run starts from a clean sheet.
    assert adapter.drain_errors() == []


def test_source_without_an_errors_kwarg_still_adapts():
    adapter = SyncSourceAdapter(LegacyRecordSource())
    items = asyncio.run(_drain(adapter.get_data()))
    assert [r.stable_id for r in items] == ["t:legacy1"]


def test_falling_back_to_an_old_signature_is_itself_reported():
    """Round-5 inventory. The fallback silently un-wires ``self._errors``.

    ``drain_errors`` is how a source's per-item failures reach the run report,
    and this branch calls the source without the sink at all — so anything it
    soft-skips internally has nowhere to land and the adapter reports a clean
    run. Nothing logged that the fallback had fired, either.

    Latent today (all nine registered sources declare ``errors``), so this is
    a regression guard rather than a live loss.
    """
    adapter = SyncSourceAdapter(LegacyRecordSource())

    asyncio.run(_drain(adapter.get_data()))

    reported = adapter.drain_errors()
    assert len(reported) == 1, "the un-wired sink was not reported at all"
    assert "errors" in reported[0]
    assert adapter.name in reported[0]
    # Drained means drained, same as every other error this adapter carries.
    assert adapter.drain_errors() == []


def test_source_with_neither_iter_method_fails_loudly_at_construction():
    class NotASource:
        name = "nope"

    with pytest.raises(TypeError, match="iter_entities|iter_records"):
        SyncSourceAdapter(NotASource())


def test_name_can_be_overridden_for_registry_keys_that_differ_from_source_name():
    adapter = SyncSourceAdapter(FakeRecordSource([]), name="research-reports")
    assert adapter.name == "research-reports"
