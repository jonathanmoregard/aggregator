"""Tests for the import port (aggregator.imports.port).

The port is the seam every source is adapted onto: one structural Protocol,
``get_data()`` returning an AsyncIterator of store-shaped items. These tests
pin the shape, not an implementation — a plain object with the right members
must satisfy it, with no base class and no registration step.

Async is driven with ``asyncio.run()`` inside sync tests on purpose: the
venv has no pytest-asyncio, and ``anyio`` is only present transitively (via
fastmcp), so leaning on its pytest plugin would make the suite depend on a
dependency this project never declared.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from aggregator.imports.port import (
    ImportAdapter,
    ImportItem,
    ImportSink,
    SupportsInputFreshness,
    SupportsNonFatalErrors,
    WriteCounts,
)
from aggregator.sources.base import ObservationRow, Record, SessionRow


class _MinimalAdapter:
    """Structural conformance only — no inheritance, no registration."""

    name = "minimal"

    async def get_data(self) -> AsyncIterator[ImportItem]:
        yield Record(
            stable_id="minimal:1",
            source="minimal",
            subject="s",
            body="b",
        )


async def _drain(adapter) -> list:
    return [item async for item in adapter.get_data()]


def test_plain_object_satisfies_ImportAdapter_structurally():
    assert isinstance(_MinimalAdapter(), ImportAdapter)


def test_object_missing_get_data_does_not_satisfy_ImportAdapter():
    class NoGetData:
        name = "nope"

    assert not isinstance(NoGetData(), ImportAdapter)


def test_get_data_is_an_async_iterator_not_a_list():
    items = asyncio.run(_drain(_MinimalAdapter()))
    assert [r.stable_id for r in items] == ["minimal:1"]


def test_ImportItem_covers_both_existing_write_shapes():
    """Record-shaped (store.upsert) and entity-shaped (store.upsert_entities).

    The two ontologies are deliberately not collapsed (see store.py docstring);
    the port's union has to carry both.
    """
    now = datetime(2026, 8, 11, tzinfo=UTC)
    values: list[ImportItem] = [
        Record(stable_id="x:1", source="x", subject="s", body="b"),
        SessionRow(
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
        ),
        ObservationRow(
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
        ),
    ]
    assert all(isinstance(v, Record | SessionRow | ObservationRow) for v in values)


def test_WriteCounts_adds_componentwise():
    """The runner sums per-batch counts; a report must carry REAL numbers.

    ``cli.py`` currently prints ``added=len(records) updated=0`` regardless of
    outcome. Counts only stay honest if they come back from the write itself,
    so they have to be addable.
    """
    total = WriteCounts(added=2, updated=1, skipped=0) + WriteCounts(
        added=3, updated=0, skipped=4
    )
    assert (total.added, total.updated, total.skipped) == (5, 1, 4)


def test_plain_object_satisfies_ImportSink_structurally():
    class Sink:
        def write(self, items):
            return WriteCounts(added=len(list(items)))

    assert isinstance(Sink(), ImportSink)
    assert not isinstance(object(), ImportSink)


def test_optional_protocols_are_opt_in():
    """Freshness + non-fatal errors are structural extras, not port members.

    Keeping them off ``ImportAdapter`` means a trivial adapter stays trivial;
    the runner checks for them with ``isinstance`` and skips what is absent.
    """

    class Extras:
        name = "extras"

        async def get_data(self) -> AsyncIterator[ImportItem]:  # pragma: no cover
            yield Record(stable_id="e:1", source="e", subject="s", body="b")

        def drain_errors(self) -> list[str]:
            return ["one file failed"]

        def input_freshness(self) -> datetime | None:
            return datetime(2026, 7, 11, tzinfo=UTC)

    extras = Extras()
    assert isinstance(extras, SupportsNonFatalErrors)
    assert isinstance(extras, SupportsInputFreshness)
    assert not isinstance(_MinimalAdapter(), SupportsNonFatalErrors)
    assert not isinstance(_MinimalAdapter(), SupportsInputFreshness)
