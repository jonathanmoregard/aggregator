"""Tests for aggregator.imports.research — the reference adapter.

``research`` was picked to prove the seam because it is the simplest real
source: a local directory, record-shaped, already idempotent per stable_id,
and owned by nobody else right now. The old ``cli.py`` path over the same
source is left working untouched — this adapter is additive.
"""
from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta

import pytest

from aggregator.cli import _default_sources
from aggregator.core.store import Store
from aggregator.imports.port import (
    ImportAdapter,
    SupportsNonFatalErrors,
)
from aggregator.imports.research import ResearchReportsAdapter
from aggregator.imports.runner import run_imports
from aggregator.imports.store_sink import StoreSink
from aggregator.sources.base import Record
from aggregator.sources.research_reports import ResearchReportsSource


def _write_reports(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "aaa111.md").write_text("# First report\n\nbody one\n")
    (reports / "bbb222.md").write_text("# Second report\n\nbody two\n")
    return reports


async def _drain(adapter):
    return [item async for item in adapter.get_data()]


def test_adapter_conforms_to_the_port(tmp_path):
    adapter = ResearchReportsAdapter(reports_dir=_write_reports(tmp_path))
    assert isinstance(adapter, ImportAdapter)
    assert isinstance(adapter, SupportsNonFatalErrors)
    assert adapter.name == "research"


def test_adapter_yields_the_same_records_the_source_produces(tmp_path):
    reports = _write_reports(tmp_path)
    adapter = ResearchReportsAdapter(reports_dir=reports)

    items = asyncio.run(_drain(adapter))

    assert all(isinstance(i, Record) for i in items)
    assert [i.stable_id for i in items] == ["research:aaa111", "research:bbb222"]
    assert [i.subject for i in items] == ["First report", "Second report"]
    # Byte-identical to the old path — the adapter changes acquisition
    # plumbing, not parsing.
    direct = list(ResearchReportsSource(reports_dir=reports).iter_records(None))
    assert items == direct


def test_since_is_honoured(tmp_path):
    reports = _write_reports(tmp_path)
    future = datetime.now(UTC) + timedelta(days=1)
    adapter = ResearchReportsAdapter(reports_dir=reports, since=future)

    assert asyncio.run(_drain(adapter)) == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
def test_unreadable_report_surfaces_through_drain_errors_without_aborting(
    tmp_path,
):
    """Per-file failures stay non-fatal (partial ingest beats total loss) but
    they must not vanish — a run ending with errors has to be able to notify."""
    reports = _write_reports(tmp_path)
    broken = reports / "ccc333.md"
    broken.write_text("# Third\n")
    broken.chmod(0o000)
    try:
        adapter = ResearchReportsAdapter(reports_dir=reports)
        items = asyncio.run(_drain(adapter))
    finally:
        broken.chmod(0o644)

    assert [i.stable_id for i in items] == ["research:aaa111", "research:bbb222"]
    errors = adapter.drain_errors()
    assert len(errors) == 1
    assert "ccc333.md" in errors[0]


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
def test_errors_reach_the_run_report_and_flip_ok_false(tmp_path, tmp_data_home):
    reports = _write_reports(tmp_path)
    broken = reports / "ccc333.md"
    broken.write_text("# Third\n")
    broken.chmod(0o000)
    store = Store()
    store.migrate()
    try:
        report = asyncio.run(
            run_imports(
                [ResearchReportsAdapter(reports_dir=reports)], StoreSink(store)
            )
        )
    finally:
        broken.chmod(0o644)

    assert report.adapters["research"].added == 2
    assert report.ok is False
    assert any("ccc333.md" in e for e in report.errors)


def test_runner_writes_research_records_into_the_real_store(tmp_path, tmp_data_home):
    store = Store()
    store.migrate()
    reports = _write_reports(tmp_path)

    report = asyncio.run(
        run_imports([ResearchReportsAdapter(reports_dir=reports)], StoreSink(store))
    )

    assert store.count_by_source("research") == 2
    assert (report.added, report.updated, report.skipped) == (2, 0, 0)
    assert report.ok is True

    # Idempotent: a second pass updates, never re-adds.
    again = asyncio.run(
        run_imports([ResearchReportsAdapter(reports_dir=reports)], StoreSink(store))
    )
    assert (again.added, again.updated) == (0, 2)
    assert store.count_by_source("research") == 2


def test_old_cli_path_still_registers_the_same_source(tmp_path):
    """This task adds a seam; it does not remove the one in use. If the CLI
    registry ever swaps the research source out, this fails and the adapter
    gets updated with it."""
    assert isinstance(_default_sources()["research"], ResearchReportsSource)
    assert ResearchReportsAdapter().name == _default_sources()["research"].name
