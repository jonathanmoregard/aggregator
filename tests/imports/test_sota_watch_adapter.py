"""Tests for aggregator.imports.sota_watch.

sota-watch reads a local proposals dir that other tooling refreshes, so it
gets ``SupportsNonFatalErrors`` (unreadable proposals must still reach the
run report) but deliberately NOT ``SupportsInputFreshness`` — there is no
manual export ritual here to go stale.
"""
from __future__ import annotations

import asyncio
import os

import pytest

from aggregator.cli import _default_sources
from aggregator.imports.port import (
    ImportAdapter,
    SupportsInputFreshness,
    SupportsNonFatalErrors,
)
from aggregator.imports.sota_watch import SotaWatchAdapter
from aggregator.sources.base import Record
from aggregator.sources.sota_watch import SotaWatchSource


def _write_proposals(tmp_path):
    proposals = tmp_path / "proposals"
    proposals.mkdir()
    (proposals / "aaa.md").write_text("# First proposal\n\nbody one\n")
    (proposals / "bbb.md").write_text("# Second proposal\n\nbody two\n")
    return proposals


async def _drain(adapter):
    return [item async for item in adapter.get_data()]


def test_adapter_yields_the_same_records_the_source_produces(tmp_path):
    proposals = _write_proposals(tmp_path)
    adapter = SotaWatchAdapter(proposals_dir=proposals)

    items = asyncio.run(_drain(adapter))

    assert all(isinstance(i, Record) for i in items)
    assert [i.stable_id for i in items] == ["sota-watch:aaa", "sota-watch:bbb"]
    # Byte-identical to the old path — the adapter changes acquisition
    # plumbing, not parsing.
    direct = list(SotaWatchSource(proposals_dir=proposals).iter_records(None))
    assert items == direct


def test_adapter_conforms_to_the_port(tmp_path):
    """Regression guard (passes on arrival once the adapter exists)."""
    adapter = SotaWatchAdapter(proposals_dir=_write_proposals(tmp_path))
    assert isinstance(adapter, ImportAdapter)
    assert isinstance(adapter, SupportsNonFatalErrors)
    assert not isinstance(adapter, SupportsInputFreshness)
    assert adapter.name == "sota-watch"


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
def test_unreadable_proposal_surfaces_through_drain_errors(tmp_path):
    proposals = _write_proposals(tmp_path)
    broken = proposals / "ccc.md"
    broken.write_text("# Third\n")
    broken.chmod(0o000)
    try:
        adapter = SotaWatchAdapter(proposals_dir=proposals)
        items = asyncio.run(_drain(adapter))
    finally:
        broken.chmod(0o644)

    assert [i.stable_id for i in items] == ["sota-watch:aaa", "sota-watch:bbb"]
    errors = adapter.drain_errors()
    assert len(errors) == 1
    assert "ccc.md" in errors[0]


def test_cli_registry_registers_the_same_source():
    """Regression guard: this task adds a seam, it does not remove the one in
    use. If the registry swaps the source out, the adapter follows it."""
    assert isinstance(_default_sources()["sota-watch"], SotaWatchSource)
    assert SotaWatchAdapter().name == _default_sources()["sota-watch"].name
