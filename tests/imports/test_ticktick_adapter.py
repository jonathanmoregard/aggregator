"""Tests for aggregator.imports.ticktick — the TickTick adapter.

``research`` proved the seam with the simplest possible source. TickTick is
the first one that is genuinely hard: two legs, a credential that expires, and
an input nothing on this machine refreshes. It is therefore also the first
adapter to need both optional protocols — ``SupportsNonFatalErrors`` so a dead
API leg still notifies, and ``SupportsInputFreshness`` so a manual export
going stale can be reported instead of silently re-imported forever.
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
    SupportsInputFreshness,
    SupportsNonFatalErrors,
)
from aggregator.imports.runner import run_imports
from aggregator.imports.store_sink import StoreSink
from aggregator.imports.ticktick import TickTickAdapter
from aggregator.sources import ticktick_api
from aggregator.sources.base import Record
from aggregator.sources.ticktick import TickTickSource
from tests.sources.test_ticktick import _backup, _row


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """No test in this file may reach the real TickTick API."""

    def _forbidden(*args, **kwargs):
        raise AssertionError("test attempted a real network call")

    monkeypatch.setattr(ticktick_api, "_open", _forbidden)


@pytest.fixture(autouse=True)
def _no_real_credentials(monkeypatch, tmp_path):
    """No test in this file may read the developer's own TickTick token."""
    monkeypatch.setattr(ticktick_api, "DEFAULT_ENV_FILE", str(tmp_path / "no-such-env"))
    for var in (
        "TICKTICK_ACCESS_TOKEN",
        "TICKTICK_TOKEN_EXPIRES_AT",
        "AGGREGATOR_TICKTICK_TOKEN",
        "AGGREGATOR_TICKTICK_TOKEN_FILE",
        "AGGREGATOR_TICKTICK_DIR",
    ):
        monkeypatch.delenv(var, raising=False)


async def _drain(adapter):
    return [item async for item in adapter.get_data()]


def _adapter(tmp_path, **kw):
    return TickTickAdapter(
        backup_dir=tmp_path / "downloads",
        archive_dir=tmp_path / "archive",
        state_file=tmp_path / "state.json",
        **kw,
    )


def test_adapter_conforms_to_the_port(tmp_path):
    _backup(tmp_path / "downloads" / "TickTick.csv", [_row()])
    adapter = _adapter(tmp_path)
    assert isinstance(adapter, ImportAdapter)
    assert isinstance(adapter, SupportsNonFatalErrors)
    assert isinstance(adapter, SupportsInputFreshness)
    assert adapter.name == "ticktick"


def test_input_freshness_reports_the_manual_export_going_stale(tmp_path):
    """The signal a later surface turns into "ticktick input is 31 days stale".

    Nothing on this machine refreshes this file, so a timer would otherwise
    re-import the same export forever and report success every single run.
    """
    path = _backup(tmp_path / "downloads" / "TickTick.csv", [_row()])
    stale = (datetime.now(UTC) - timedelta(days=31)).timestamp()
    os.utime(path, (stale, stale))

    freshness = _adapter(tmp_path).input_freshness()

    assert freshness is not None
    assert (datetime.now(UTC) - freshness).days == 31


def test_adapter_yields_the_same_records_the_source_produces(tmp_path):
    """The adapter changes acquisition plumbing, not parsing."""
    _backup(tmp_path / "downloads" / "TickTick.csv", [_row(), _row(task_id="def456")])
    items = asyncio.run(_drain(_adapter(tmp_path)))

    assert all(isinstance(i, Record) for i in items)
    assert [i.stable_id for i in items] == ["ticktick:abc123", "ticktick:def456"]
    direct = list(
        TickTickSource(
            backup_dir=tmp_path / "downloads",
            archive_dir=tmp_path / "archive",
            state_file=tmp_path / "state.json",
        ).iter_records(None)
    )
    assert items == direct


def test_expired_token_degrades_the_run_to_csv_only_and_still_notifies(
    tmp_path, tmp_data_home, monkeypatch
):
    """The whole invariant, end to end through the real runner and store.

    An expired token in the shared store is the failure this source will
    actually hit — the todo backend refreshes that token, and when the refresh
    itself has lapsed nothing re-authorizes it unattended. The run must still
    write the CSV archive, and must still come back ``ok=False`` naming the
    command a human has to run, because a timer that reported success here
    would let the index rot with nobody told.
    """
    env_file = tmp_path / "env"
    env_file.write_text(
        'TICKTICK_ACCESS_TOKEN="stale"\nTICKTICK_TOKEN_EXPIRES_AT="1000000000"\n',
        encoding="utf-8",
    )
    # Overrides the autouse credential guard: this test WANTS a token in the
    # shared store, just a dead one.
    monkeypatch.setattr(ticktick_api, "DEFAULT_ENV_FILE", str(env_file))
    _backup(tmp_path / "downloads" / "TickTick.csv", [_row()])
    store = Store()
    store.migrate()

    report = asyncio.run(run_imports([_adapter(tmp_path)], StoreSink(store)))

    assert store.count_by_source("ticktick") == 1
    assert report.adapters["ticktick"].added == 1
    assert report.ok is False
    assert any(ticktick_api.RELOGIN_COMMAND in e for e in report.errors)


def test_runner_writes_ticktick_records_into_the_real_store(tmp_path, tmp_data_home):
    store = Store()
    store.migrate()
    _backup(tmp_path / "downloads" / "TickTick.csv", [_row()])

    report = asyncio.run(run_imports([_adapter(tmp_path)], StoreSink(store)))

    assert store.count_by_source("ticktick") == 1
    assert (report.added, report.updated, report.skipped) == (1, 0, 0)
    assert report.ok is True

    # Idempotent: a second pass updates, never re-adds. The merge is keyed by
    # stable_id, so re-reading the same backup cannot fork a task into two rows.
    again = asyncio.run(run_imports([_adapter(tmp_path)], StoreSink(store)))
    assert (again.added, again.updated) == (0, 1)
    assert store.count_by_source("ticktick") == 1


def test_cli_registry_registers_the_same_source():
    """This task adds a seam; it does not remove the one in use. If the CLI
    registry ever swaps the ticktick source out, this fails and the adapter
    gets updated with it."""
    assert isinstance(_default_sources()["ticktick"], TickTickSource)
    assert TickTickAdapter().name == _default_sources()["ticktick"].name
