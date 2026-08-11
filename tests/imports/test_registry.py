"""Tests for aggregator.imports.registry.

The registry is what lets ONE timer drive every source. Its only real
invariant is coverage: a source that exists in the CLI's ``_default_sources()``
but not here would silently never be imported by the run-all path, which is
precisely the "looks like it is working" failure the fail-loudly constraint
exists to prevent.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from aggregator.cli import _default_sources
from aggregator.imports.port import ImportAdapter
from aggregator.imports.registry import default_adapters


def test_every_cli_source_has_an_adapter():
    """The coverage invariant. Add a source to _default_sources() without an
    adapter and this fails — better than the run-all path quietly skipping it.
    """
    assert {a.name for a in default_adapters()} == set(_default_sources())


def test_every_entry_conforms_to_the_port():
    for adapter in default_adapters():
        assert isinstance(adapter, ImportAdapter), adapter


def test_names_are_unique_so_the_run_report_cannot_drop_a_source():
    """RunReport is keyed by adapter name; a collision would silently discard
    one source's entire outcome. The runner refuses duplicates, but the
    registry must not hand it any."""
    names = [a.name for a in default_adapters()]
    assert len(names) == len(set(names))


def test_since_is_passed_through_to_every_adapter():
    since = datetime(2026, 8, 1, tzinfo=UTC)
    for adapter in default_adapters(since=since):
        assert adapter._since == since, adapter.name


@pytest.mark.parametrize(
    "name", ["chatgpt", "claude-web", "substack", "ticktick"]
)
def test_manual_export_sources_report_input_freshness(name):
    """These four read an archive only a human refreshes. Losing the freshness
    signal on any of them means a timer reports success on a stale zip."""
    adapter = next(a for a in default_adapters() if a.name == name)
    assert hasattr(adapter, "input_freshness")


@pytest.mark.parametrize(
    "name", ["sessions", "github", "research", "sota-watch", "dropbox"]
)
def test_live_input_sources_do_not_claim_a_freshness_signal(name):
    """A live API or a continuously-synced dir has no export ritual to forget.
    Reporting an age here would put a number in the report nobody can act on,
    and train the operator to ignore the warnings that matter."""
    adapter = next(a for a in default_adapters() if a.name == name)
    assert not hasattr(adapter, "input_freshness")
