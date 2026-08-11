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
)
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


def test_cli_registry_registers_the_same_source():
    """Regression guard: the old per-source path keeps its source."""
    assert isinstance(_default_sources()["dropbox"], DropboxSource)
    assert DropboxAdapter().name == _default_sources()["dropbox"].name
