"""Tests for aggregator.imports.chatgpt.

ChatGPT has no consumer history API and no scraper (session constraint,
2026-08-11: it fails low-upkeep, low-manual-work and robustness on all three
counts, so it deliberately gets none). Its input is a zip a human requests,
waits up to seven days for, and downloads by hand.

That makes ``SupportsInputFreshness`` the whole point of this adapter. Without
it a timer re-imports the same July export every night and reports success
every night, which is indistinguishable from an index that is current.
"""
from __future__ import annotations

import asyncio
import json
import os
import zipfile
from datetime import UTC, datetime, timedelta

from aggregator.cli import _default_sources
from aggregator.imports.chatgpt import ChatGPTAdapter
from aggregator.imports.port import (
    ImportAdapter,
    SupportsInputFreshness,
    SupportsNonFatalErrors,
)
from aggregator.sources.base import ObservationRow, SessionRow
from aggregator.sources.chatgpt import ChatGPTSource

_CONVERSATIONS = [
    {
        "conversation_id": "conv-port-1",
        "title": "porting the sources",
        "create_time": 1750000000.0,
        "update_time": 1750003600.0,
        "mapping": {
            "n1": {
                "id": "n1",
                "parent": None,
                "message": {
                    "author": {"role": "user"},
                    "create_time": 1750000000.0,
                    "content": {"parts": ["how does the import port work?"]},
                },
            }
        },
    }
]


def _drops(tmp_path, *, name="conversations.json"):
    drops = tmp_path / "drops"
    drops.mkdir(exist_ok=True)
    path = drops / name
    path.write_text(json.dumps(_CONVERSATIONS))
    return drops, path


async def _drain(adapter):
    return [item async for item in adapter.get_data()]


def test_input_freshness_reports_the_manual_export_going_stale(tmp_path):
    """The signal that turns a cheerful success into "chatgpt input is 31 days
    stale". Nothing on this machine refreshes this file."""
    drops, path = _drops(tmp_path)
    stale = (datetime.now(UTC) - timedelta(days=31)).timestamp()
    os.utime(path, (stale, stale))

    freshness = ChatGPTAdapter(drops_dir=drops).input_freshness()

    assert freshness is not None
    assert (datetime.now(UTC) - freshness).days == 31


def test_input_freshness_is_none_when_no_export_has_ever_been_dropped(tmp_path):
    """Unknown, not fresh. An empty drops dir must not read as up to date."""
    empty = tmp_path / "empty"
    empty.mkdir()
    assert ChatGPTAdapter(drops_dir=empty).input_freshness() is None


def test_input_freshness_uses_the_newest_of_several_exports(tmp_path):
    drops, old_path = _drops(tmp_path)
    new_path = drops / "conversations-2.json"
    new_path.write_text(json.dumps(_CONVERSATIONS))
    old = (datetime.now(UTC) - timedelta(days=90)).timestamp()
    os.utime(old_path, (old, old))
    recent = (datetime.now(UTC) - timedelta(days=2)).timestamp()
    os.utime(new_path, (recent, recent))

    freshness = ChatGPTAdapter(drops_dir=drops).input_freshness()

    assert freshness is not None
    assert (datetime.now(UTC) - freshness).days == 2


def test_input_freshness_reads_the_zip_not_its_members(tmp_path):
    """A vendor zip's member timestamps are whenever the vendor built it; the
    acquisition date this warning is about is when the human downloaded it."""
    drops = tmp_path / "drops"
    drops.mkdir()
    zip_path = drops / "chatgpt-export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("conversations.json", json.dumps(_CONVERSATIONS))
    downloaded = (datetime.now(UTC) - timedelta(days=5)).timestamp()
    os.utime(zip_path, (downloaded, downloaded))

    freshness = ChatGPTAdapter(drops_dir=drops).input_freshness()

    assert freshness is not None
    assert (datetime.now(UTC) - freshness).days == 5


def test_adapter_yields_the_same_entities_the_source_produces(tmp_path):
    drops, _ = _drops(tmp_path)
    items = asyncio.run(_drain(ChatGPTAdapter(drops_dir=drops)))

    assert [type(i) for i in items] == [SessionRow, ObservationRow]
    assert items[0].session_id == "chatgpt:conv-port-1"
    direct = list(ChatGPTSource(drops_dir=str(drops)).iter_entities(None))
    assert items == direct


def test_adapter_conforms_to_the_port(tmp_path):
    """Regression guard (passes on arrival once the adapter exists)."""
    drops, _ = _drops(tmp_path)
    adapter = ChatGPTAdapter(drops_dir=drops)
    assert isinstance(adapter, ImportAdapter)
    assert isinstance(adapter, SupportsNonFatalErrors)
    assert isinstance(adapter, SupportsInputFreshness)
    assert adapter.name == "chatgpt"


def test_corrupt_export_surfaces_through_drain_errors(tmp_path):
    drops, _ = _drops(tmp_path)
    (drops / "conversations-broken.json").write_text("{not json")

    adapter = ChatGPTAdapter(drops_dir=drops)
    items = asyncio.run(_drain(adapter))

    assert [i.session_id for i in items if isinstance(i, SessionRow)] == [
        "chatgpt:conv-port-1"
    ]
    assert any("conversations-broken.json" in e for e in adapter.drain_errors())


def test_cli_registry_registers_the_same_source():
    """Regression guard: the old per-source path keeps its source."""
    assert isinstance(_default_sources()["chatgpt"], ChatGPTSource)
    assert ChatGPTAdapter().name == _default_sources()["chatgpt"].name
