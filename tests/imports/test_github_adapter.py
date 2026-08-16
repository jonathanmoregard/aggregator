"""Tests for aggregator.imports.github.

github is the one source that already auto-imports (a systemd user timer), so
its migration must be provably behaviour-preserving: same records, same
per-endpoint error policy. It reads a live API rather than a hand-dropped
archive, so it does NOT report input freshness.

No test here touches the network: every ``GitHubSource`` is constructed with
stub ``_scope_fetcher`` / ``_api_fetcher`` seams.
"""
from __future__ import annotations

import asyncio

from aggregator.cli import _default_sources
from aggregator.imports.github import GitHubAdapter
from aggregator.imports.port import (
    ImportAdapter,
    SupportsInputFreshness,
    SupportsNonFatalErrors,
)
from aggregator.sources.base import Record
from aggregator.sources.github import GitHubSource

_PR_ROW = {
    "number": 7,
    "title": "add the import port",
    "body": "one seam for every source",
    "state": "open",
    "repository_url": "https://api.github.com/repos/jonathan/aggregator",
    "html_url": "https://github.com/jonathan/aggregator/pull/7",
    "user": {"login": "jonathan"},
    "created_at": "2026-08-01T10:00:00Z",
    "updated_at": "2026-08-02T10:00:00Z",
}


def _source(api_fetcher):
    return GitHubSource(
        _scope_fetcher=lambda: ["public_repo"],
        _api_fetcher=api_fetcher,
        _gh_token_fetcher=lambda: None,
    )


async def _drain(adapter):
    return [item async for item in adapter.get_data()]


def test_adapter_yields_the_same_records_the_source_produces(monkeypatch):
    monkeypatch.delenv("AGGREGATOR_ALLOW_WRITE_TOKEN", raising=False)

    def api(path: str) -> list[dict]:
        return [_PR_ROW] if "is:pr+author" in path else []

    items = asyncio.run(_drain(GitHubAdapter(source=_source(api))))

    assert all(isinstance(i, Record) for i in items)
    assert [i.stable_id for i in items] == ["github:jonathan/aggregator:7"]
    direct = list(_source(api).iter_records(None))
    assert items == direct


def test_adapter_conforms_to_the_port():
    """Regression guard (passes on arrival once the adapter exists)."""
    adapter = GitHubAdapter(source=_source(lambda p: []))
    assert isinstance(adapter, ImportAdapter)
    assert isinstance(adapter, SupportsNonFatalErrors)
    assert not isinstance(adapter, SupportsInputFreshness)
    assert adapter.name == "github"


def test_endpoint_failure_surfaces_through_drain_errors_without_aborting(
    monkeypatch,
):
    """Partial ingest beats total loss, but the loss must still be reported:
    three of four endpoints answering is not a clean run."""
    monkeypatch.delenv("AGGREGATOR_ALLOW_WRITE_TOKEN", raising=False)

    def api(path: str) -> list[dict]:
        if "review-requested" in path:
            raise RuntimeError("gh api: 502 Bad Gateway")
        return [_PR_ROW] if "is:pr+author" in path else []

    adapter = GitHubAdapter(source=_source(api))
    items = asyncio.run(_drain(adapter))

    assert [i.stable_id for i in items] == ["github:jonathan/aggregator:7"]
    errors = adapter.drain_errors()
    assert len(errors) == 1
    assert "502 Bad Gateway" in errors[0]


def test_cli_registry_registers_the_same_source():
    """Regression guard: the old per-source path keeps its source."""
    assert isinstance(_default_sources()["github"], GitHubSource)
    assert GitHubAdapter().name == _default_sources()["github"].name
