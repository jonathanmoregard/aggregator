"""Tests for aggregator.imports.registry.

The registry is what lets ONE timer drive every source. Its only real
invariant is coverage: a source that exists in the CLI's ``_default_sources()``
but not here would silently never be imported by the run-all path, which is
precisely the "looks like it is working" failure the fail-loudly constraint
exists to prevent.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from aggregator.cli import _default_sources
from aggregator.imports.port import ImportAdapter, WriteCounts
from aggregator.imports.registry import _UnbuildableAdapter, default_adapters
from aggregator.imports.runner import run_imports
from aggregator.sources.base import Record


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


# -- failure isolation has to start at CONSTRUCTION ------------------------
#
# Round-1 MEDIUM: the runner isolates per adapter from ``get_data`` onward, but
# all nine adapters were constructed in one unprotected list expression called
# from ``cli.main``. One constructor raising killed the whole --all run before
# the runner saw it: traceback, no report, no per-source errors, no exit 3, and
# nothing saying which source was at fault or that the other eight never ran.


def test_a_constructor_that_raises_does_not_kill_the_registry(monkeypatch):
    class Unbuildable:
        def __init__(self, *a, **kw):
            raise RuntimeError("XDG_DATA_HOME points at a file, not a directory")

    monkeypatch.setattr(
        "aggregator.imports.registry.TickTickAdapter", Unbuildable
    )

    adapters = default_adapters()

    assert {a.name for a in adapters} == set(_default_sources()), (
        "the broken source must still occupy its slot, under its own name"
    )


def test_the_construction_failure_surfaces_as_that_source_s_error():
    """It has to arrive where a failure normally arrives — on the source's own
    line in the report — not vanish."""

    class Sink:
        def write(self, items):
            return WriteCounts()

    broken = _UnbuildableAdapter("ticktick", RuntimeError("no such directory"))
    report = asyncio.run(run_imports([broken], Sink()))

    assert report.ok is False
    assert report.failed_adapters == ["ticktick"]
    errors = report.adapters["ticktick"].errors
    assert any("could not be constructed" in e for e in errors), errors
    assert any("no such directory" in e for e in errors), errors


def test_the_other_sources_still_run(monkeypatch):
    """The whole point of the run-all path: one broken source costs its own
    line and nothing else."""

    class Unbuildable:
        def __init__(self, *a, **kw):
            raise RuntimeError("boom")

    class Sink:
        def __init__(self):
            self.calls = 0

        def write(self, items):
            self.calls += 1
            return WriteCounts()

    class Fine:
        name = "fine"

        async def get_data(self):
            yield Record(stable_id="research:1", source="research", subject="s", body="b")

    monkeypatch.setattr(
        "aggregator.imports.registry.TickTickAdapter", Unbuildable
    )
    broken = next(a for a in default_adapters() if a.name == "ticktick")
    sink = Sink()

    report = asyncio.run(run_imports([broken, Fine()], sink))

    assert report.failed_adapters == ["ticktick"]
    assert report.adapters["fine"].ok is True
    assert sink.calls == 1


def test_an_unbuildable_entry_still_conforms_to_the_port():
    broken = _UnbuildableAdapter("ticktick", RuntimeError("boom"))
    assert isinstance(broken, ImportAdapter)
