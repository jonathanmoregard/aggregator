"""Tests for aggregator.imports.substack.

Substack exports are produced from Settings → Exports and downloaded by hand.
Nothing on this machine refreshes them, which is exactly why the run report
has to be able to say "substack input is 31 days stale" instead of reporting a
clean import of a zip from July.
"""
from __future__ import annotations

import asyncio
import os
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from aggregator.cli import _default_sources
from aggregator.imports.port import (
    ImportAdapter,
    SupportsInputFreshness,
    SupportsNonFatalErrors,
)
from aggregator.imports.substack import SubstackAdapter
from aggregator.sources.base import Record
from aggregator.sources.substack import SubstackSource


def _build_zip(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("posts.csv", "post_id,title\n1,Hello\n")
        zf.writestr(
            "posts/1.hello-world.html",
            "<html><body><h1>Hello world</h1><p>first post</p></body></html>",
        )
    return path


def _drops(tmp_path, *, name="substack-export.zip"):
    drops = tmp_path / "drops"
    drops.mkdir(exist_ok=True)
    return drops, _build_zip(drops / name)


async def _drain(adapter):
    return [item async for item in adapter.get_data()]


def test_input_freshness_reports_the_manual_export_going_stale(tmp_path):
    drops, path = _drops(tmp_path)
    stale = (datetime.now(UTC) - timedelta(days=31)).timestamp()
    os.utime(path, (stale, stale))

    freshness = SubstackAdapter(drops_dir=drops).input_freshness()

    assert freshness is not None
    assert (datetime.now(UTC) - freshness).days == 31


def test_input_freshness_is_none_when_no_export_has_ever_been_dropped(tmp_path):
    """Unknown, not fresh."""
    empty = tmp_path / "empty"
    empty.mkdir()
    assert SubstackAdapter(drops_dir=empty).input_freshness() is None


def test_input_freshness_is_the_download_time_not_the_posts_dates(tmp_path):
    """The zip members carry publication dates from years ago; the question
    the warning answers is when a human last fetched an export."""
    drops, path = _drops(tmp_path)
    downloaded = (datetime.now(UTC) - timedelta(days=3)).timestamp()
    os.utime(path, (downloaded, downloaded))

    freshness = SubstackAdapter(drops_dir=drops).input_freshness()

    assert freshness is not None
    assert (datetime.now(UTC) - freshness).days == 3


def test_adapter_yields_the_same_records_the_source_produces(tmp_path):
    drops, _ = _drops(tmp_path)
    items = asyncio.run(_drain(SubstackAdapter(drops_dir=drops)))

    assert all(isinstance(i, Record) for i in items)
    assert [i.stable_id for i in items] == ["substack:1"]
    direct = list(SubstackSource(drops_dir=str(drops)).iter_records(None))
    assert items == direct


def test_adapter_conforms_to_the_port(tmp_path):
    """Regression guard (passes on arrival once the adapter exists)."""
    drops, _ = _drops(tmp_path)
    adapter = SubstackAdapter(drops_dir=drops)
    assert isinstance(adapter, ImportAdapter)
    assert isinstance(adapter, SupportsNonFatalErrors)
    assert isinstance(adapter, SupportsInputFreshness)
    assert adapter.name == "substack"


def test_cli_registry_registers_the_same_source():
    """Regression guard: the old per-source path keeps its source."""
    assert isinstance(_default_sources()["substack"], SubstackSource)
    assert SubstackAdapter().name == _default_sources()["substack"].name
