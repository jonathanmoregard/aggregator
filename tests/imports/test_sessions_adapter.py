"""Tests for aggregator.imports.sessions.

The sessions source is the entity-shaped one and by far the largest — ~359k
observations — which is why the port streams rather than returning a list.
This adapter is the proof that ``SyncSourceAdapter`` picks ``iter_entities``
over ``iter_records`` and that the store sink routes SessionRow /
ObservationRow to ``upsert_entities`` without the adapter knowing.

``~/.claude/projects`` is written continuously by Claude Code itself, so there
is no manual export to go stale: no ``SupportsInputFreshness``.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from aggregator.cli import _default_sources
from aggregator.core.store import Store
from aggregator.imports.port import (
    ImportAdapter,
    SupportsInputFreshness,
    SupportsNonFatalErrors,
)
from aggregator.imports.runner import run_imports
from aggregator.imports.sessions import SessionsAdapter
from aggregator.imports.store_sink import StoreSink
from aggregator.sources.base import ObservationRow, SessionRow
from aggregator.sources.sessions import SessionsSource

_LINE = {
    "parentUuid": None,
    "isSidechain": False,
    "promptId": "p1",
    "type": "user",
    "message": {"role": "user", "content": "hello from the port"},
    "uuid": "u-1",
    "timestamp": "2026-07-25T10:00:01.000Z",
    "cwd": "/home/u/proj",
    "sessionId": "sess-port-1",
    "version": "2.1.92",
    "gitBranch": "main",
}


def _backdate(path: Path) -> None:
    """Past the source's 5-minute live-file window, or it is skipped."""
    old = time.time() - 24 * 60 * 60
    os.utime(path, (old, old))


def _projects_root(tmp_path) -> Path:
    root = tmp_path / "projects" / "proj-slug"
    root.mkdir(parents=True)
    path = root / "sess-port-1.jsonl"
    path.write_text(json.dumps(_LINE) + "\n")
    _backdate(path)
    return tmp_path / "projects"


async def _drain(adapter):
    return [item async for item in adapter.get_data()]


def test_adapter_yields_the_same_entities_the_source_produces(tmp_path):
    root = _projects_root(tmp_path)
    items = asyncio.run(_drain(SessionsAdapter(projects_root=root)))

    assert [type(i) for i in items] == [SessionRow, ObservationRow]
    assert items[0].session_id == "sess-port-1"
    direct = list(SessionsSource(projects_root=str(root)).iter_entities(None))
    assert items == direct


def test_adapter_conforms_to_the_port(tmp_path):
    """Regression guard (passes on arrival once the adapter exists)."""
    adapter = SessionsAdapter(projects_root=_projects_root(tmp_path))
    assert isinstance(adapter, ImportAdapter)
    assert isinstance(adapter, SupportsNonFatalErrors)
    assert not isinstance(adapter, SupportsInputFreshness)
    assert adapter.name == "sessions"


def test_corrupt_line_surfaces_through_drain_errors_without_aborting(tmp_path):
    root = _projects_root(tmp_path)
    bad = root / "proj-slug" / "broken.jsonl"
    bad.write_text("{not json at all\n")
    _backdate(bad)

    adapter = SessionsAdapter(projects_root=root)
    items = asyncio.run(_drain(adapter))

    assert [i.session_id for i in items if isinstance(i, SessionRow)] == [
        "sess-port-1"
    ]
    errors = adapter.drain_errors()
    assert any("broken.jsonl" in e for e in errors)


def test_runner_writes_sessions_and_observations_into_the_real_store(
    tmp_path, tmp_data_home
):
    """Entity-shaped items land on the store's entity path, not the records
    table — the sink dispatches on type, the adapter never says which."""
    store = Store()
    store.migrate()
    root = _projects_root(tmp_path)

    report = asyncio.run(
        run_imports([SessionsAdapter(projects_root=root)], StoreSink(store))
    )

    assert (report.added, report.updated, report.skipped) == (2, 0, 0)
    assert report.ok is True

    again = asyncio.run(
        run_imports([SessionsAdapter(projects_root=root)], StoreSink(store))
    )
    assert (again.added, again.updated) == (0, 2)


def test_cli_registry_registers_the_same_source():
    """Regression guard: the old per-source path keeps its source."""
    assert isinstance(_default_sources()["sessions"], SessionsSource)
    assert SessionsAdapter().name == _default_sources()["sessions"].name


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
def test_unreadable_jsonl_surfaces_through_drain_errors(tmp_path):
    root = _projects_root(tmp_path)
    locked = root / "proj-slug" / "locked.jsonl"
    locked.write_text(json.dumps(_LINE) + "\n")
    _backdate(locked)
    locked.chmod(0o000)
    try:
        adapter = SessionsAdapter(projects_root=root)
        asyncio.run(_drain(adapter))
    finally:
        locked.chmod(0o644)

    assert any("locked.jsonl" in e for e in adapter.drain_errors())
