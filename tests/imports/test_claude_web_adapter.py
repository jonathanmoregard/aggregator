"""Tests for aggregator.imports.claude_web.

claude-web reads the same manually-downloaded archive shape as chatgpt, so it
gets the same ``SupportsInputFreshness`` treatment. A fetch adapter for this
source is a later decision (mission.md); until one exists and is proven to
fail loudly on an expired session, the honest signal is the age of the last
export a human dropped.
"""
from __future__ import annotations

import asyncio
import json
import os
import zipfile
from datetime import UTC, datetime, timedelta

from aggregator.cli import _default_sources
from aggregator.imports.claude_web import ClaudeWebAdapter
from aggregator.imports.port import (
    ImportAdapter,
    SupportsInputFreshness,
    SupportsNonFatalErrors,
)
from aggregator.sources.base import ObservationRow, SessionRow
from aggregator.sources.claude_web import ClaudeWebSource

_CONVERSATIONS = [
    {
        "uuid": "conv-port-1",
        "name": "porting the sources",
        "created_at": "2026-08-01T10:00:00Z",
        "updated_at": "2026-08-01T11:00:00Z",
        "chat_messages": [
            {
                "uuid": "msg-1",
                "parent_message_uuid": "00000000-0000-4000-8000-000000000000",
                "sender": "human",
                "created_at": "2026-08-01T10:00:00Z",
                "content": [{"type": "text", "text": "how does the port work?"}],
            }
        ],
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
    drops, path = _drops(tmp_path)
    stale = (datetime.now(UTC) - timedelta(days=31)).timestamp()
    os.utime(path, (stale, stale))

    freshness = ClaudeWebAdapter(drops_dir=drops).input_freshness()

    assert freshness is not None
    assert (datetime.now(UTC) - freshness).days == 31


def test_input_freshness_is_none_when_no_export_has_ever_been_dropped(tmp_path):
    """Unknown, not fresh."""
    empty = tmp_path / "empty"
    empty.mkdir()
    assert ClaudeWebAdapter(drops_dir=empty).input_freshness() is None


def test_input_freshness_reads_the_zip_not_its_members(tmp_path):
    drops = tmp_path / "drops"
    drops.mkdir()
    zip_path = drops / "claude-export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("conversations.json", json.dumps(_CONVERSATIONS))
    downloaded = (datetime.now(UTC) - timedelta(days=5)).timestamp()
    os.utime(zip_path, (downloaded, downloaded))

    freshness = ClaudeWebAdapter(drops_dir=drops).input_freshness()

    assert freshness is not None
    assert (datetime.now(UTC) - freshness).days == 5


def test_a_chatgpt_export_does_not_count_as_claude_web_freshness(tmp_path):
    """Vendor classification is by content. A fresh ChatGPT drop must not make
    claude-web look current — that would hide the exact staleness this is for.
    """
    drops = tmp_path / "drops"
    drops.mkdir()
    (drops / "conversations.json").write_text(
        json.dumps([{"conversation_id": "c1", "mapping": {}}])
    )
    assert ClaudeWebAdapter(drops_dir=drops).input_freshness() is None


def test_adapter_yields_the_same_entities_the_source_produces(tmp_path):
    drops, _ = _drops(tmp_path)
    items = asyncio.run(_drain(ClaudeWebAdapter(drops_dir=drops)))

    assert [type(i) for i in items] == [SessionRow, ObservationRow]
    assert items[0].session_id == "claude-web:conv-port-1"
    direct = list(ClaudeWebSource(drops_dir=str(drops)).iter_entities(None))
    assert items == direct


def test_adapter_conforms_to_the_port(tmp_path):
    """Regression guard (passes on arrival once the adapter exists)."""
    drops, _ = _drops(tmp_path)
    adapter = ClaudeWebAdapter(drops_dir=drops)
    assert isinstance(adapter, ImportAdapter)
    assert isinstance(adapter, SupportsNonFatalErrors)
    assert isinstance(adapter, SupportsInputFreshness)
    assert adapter.name == "claude-web"


def test_corrupt_export_surfaces_through_drain_errors(tmp_path):
    drops, _ = _drops(tmp_path)
    (drops / "conversations-broken.json").write_text("{not json")

    adapter = ClaudeWebAdapter(drops_dir=drops)
    items = asyncio.run(_drain(adapter))

    assert [i.session_id for i in items if isinstance(i, SessionRow)] == [
        "claude-web:conv-port-1"
    ]
    assert any("conversations-broken.json" in e for e in adapter.drain_errors())


def test_cli_registry_registers_the_same_source():
    """Regression guard: the old per-source path keeps its source."""
    assert isinstance(_default_sources()["claude-web"], ClaudeWebSource)
    assert ClaudeWebAdapter().name == _default_sources()["claude-web"].name
