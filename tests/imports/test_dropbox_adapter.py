"""Tests for aggregator.imports.dropbox.

Dropbox syncs to local disk continuously, so this source has an acquisition
story already — no manual export ritual, hence no ``SupportsInputFreshness``.
It does have a busy ``errors`` sink (one corrupt PDF in 1600 files), so
``SupportsNonFatalErrors`` has to keep working through the port.
"""
from __future__ import annotations

import asyncio
import os

import pytest

from aggregator.cli import _default_sources
from aggregator.imports.dropbox import DropboxAdapter
from aggregator.imports.port import (
    ImportAdapter,
    SupportsInputFreshness,
    SupportsNonFatalErrors,
    WriteCounts,
)
from aggregator.imports.runner import run_imports
from aggregator.sources.base import Record
from aggregator.sources.dropbox import DropboxSource


def _write_tree(tmp_path):
    root = tmp_path / "Dropbox"
    (root / "notes").mkdir(parents=True)
    (root / "notes" / "one.md").write_text("# One\n\nbody one\n")
    (root / "notes" / "two.md").write_text("# Two\n\nbody two\n")
    return root


async def _drain(adapter):
    return [item async for item in adapter.get_data()]


def test_adapter_yields_the_same_records_the_source_produces(tmp_path):
    root = _write_tree(tmp_path)
    adapter = DropboxAdapter(root=root, exclude="")

    items = asyncio.run(_drain(adapter))

    assert all(isinstance(i, Record) for i in items)
    assert [i.extra["relpath"] for i in items] == [
        "notes/one.md",
        "notes/two.md",
    ]
    direct = list(DropboxSource(root=root, exclude="").iter_records(None))
    assert items == direct


def test_adapter_conforms_to_the_port(tmp_path):
    """Regression guard (passes on arrival once the adapter exists)."""
    adapter = DropboxAdapter(root=_write_tree(tmp_path), exclude="")
    assert isinstance(adapter, ImportAdapter)
    assert isinstance(adapter, SupportsNonFatalErrors)
    assert not isinstance(adapter, SupportsInputFreshness)
    assert adapter.name == "dropbox"


def test_exclude_patterns_reach_the_wrapped_source(tmp_path):
    root = _write_tree(tmp_path)
    (root / "private").mkdir()
    (root / "private" / "contract.md").write_text("# Contract\n\nsecret\n")

    items = asyncio.run(_drain(DropboxAdapter(root=root, exclude="private")))

    assert [i.extra["relpath"] for i in items] == [
        "notes/one.md",
        "notes/two.md",
    ]


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
def test_unreadable_file_surfaces_through_drain_errors(tmp_path):
    root = _write_tree(tmp_path)
    broken = root / "notes" / "zzz.md"
    broken.write_text("# Third\n")
    broken.chmod(0o000)
    try:
        adapter = DropboxAdapter(root=root, exclude="")
        items = asyncio.run(_drain(adapter))
    finally:
        broken.chmod(0o644)

    assert [i.extra["relpath"] for i in items] == [
        "notes/one.md",
        "notes/two.md",
    ]
    errors = adapter.drain_errors()
    assert len(errors) == 1
    assert "zzz.md" in errors[0]


class _CountingSink:
    def write(self, items):
        return WriteCounts(added=len(list(items)), updated=0, skipped=0)


def test_unmounted_root_lands_in_the_run_report_instead_of_a_clean_zero(tmp_path):
    """The unattended path, concretely: the 03:00 timer must not report success.

    ``DropboxRootUnavailableError`` is raised by the source, caught by the
    runner's per-adapter isolation boundary and recorded — so an unmounted
    Dropbox produces a line a human can read and ``ok is False``, where it used
    to produce ``added=0 errors=0`` and a green run forever.
    """
    adapter = DropboxAdapter(root=tmp_path / "not-mounted", exclude="")

    report = asyncio.run(run_imports([adapter], _CountingSink()))

    assert report.ok is False
    assert any("not-mounted" in e for e in report.errors), report.errors
    assert any("DropboxRootUnavailableError" in e for e in report.errors)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
def test_unreadable_subtree_surfaces_through_drain_errors(tmp_path):
    """A subtree the walk cannot list is a per-item fault, not a hard failure."""
    root = _write_tree(tmp_path)
    blocked = root / "blocked"
    blocked.mkdir()
    (blocked / "hidden.md").write_text("# Hidden\n")
    blocked.chmod(0o000)
    try:
        adapter = DropboxAdapter(root=root, exclude="")
        items = asyncio.run(_drain(adapter))
    finally:
        blocked.chmod(0o755)

    assert [i.extra["relpath"] for i in items] == ["notes/one.md", "notes/two.md"]
    errors = adapter.drain_errors()
    assert len(errors) == 1
    assert "blocked" in errors[0]


def test_cli_registry_registers_the_same_source():
    """Regression guard: the old per-source path keeps its source."""
    assert isinstance(_default_sources()["dropbox"], DropboxSource)
    assert DropboxAdapter().name == _default_sources()["dropbox"].name
