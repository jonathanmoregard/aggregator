"""Round-2 HIGH-1: the source registry is built before command dispatch.

``main`` used to call ``_default_sources()`` unconditionally, on the line after
``store.migrate()`` and before it looked at ``args.cmd``. Nine real
constructors therefore ran on EVERY invocation, including ``query`` and
``status``, which need no source at all, and including ``ingest --all``, which
uses the adapter registry instead.

That defeated the round-1 ``_UnbuildableAdapter`` isolation: that guard sits in
``imports/registry.py``, downstream of this call, so a single constructor
raising (an env var read at ``__init__`` time, an XDG path that resolved to a
file) killed the process with a bare traceback before the runner existed — no
report, no notify, no exit 3, and the other eight sources never ran.
"""
from __future__ import annotations

from aggregator.cli import main
from aggregator.core.store import Store
from aggregator.sources.base import Record


class _BoomSource:
    """A source whose construction fails the way a real one would.

    Construction is documented as side-effect-free (env + path resolution
    only), which is exactly why an environment-dependent raise is the
    plausible failure mode.
    """

    def __init__(self, *a, **kw) -> None:
        raise RuntimeError("XDG_DATA_HOME points at a file")


class _StubResearch:
    name = "research"

    def iter_records(self, since, errors=None):
        yield Record(
            stable_id="research:1",
            source="research",
            subject="a report",
            body="body",
        )


class _FakeAdapter:
    def __init__(self, name: str) -> None:
        self.name = name

    async def get_data(self):
        return
        yield  # pragma: no cover - makes this an async generator


def _store(tmp_path) -> Store:
    store = Store(db_path=tmp_path / "cache.db")
    store.migrate()
    return store


def test_status_survives_a_broken_source_constructor(
    tmp_path, monkeypatch, capsys
):
    """``status`` reads the store's capabilities and touches no source."""
    monkeypatch.setattr("aggregator.cli.GitHubSource", _BoomSource)

    rc = main(["status"], _store=_store(tmp_path))

    assert rc == 0
    assert "cache_path" in capsys.readouterr().out


def test_query_survives_a_broken_source_constructor(tmp_path, monkeypatch):
    """Same for ``query`` — it runs entirely against the SQLite index."""
    monkeypatch.setattr("aggregator.cli.GitHubSource", _BoomSource)

    rc = main(["query", "source:sessions"], _store=_store(tmp_path))

    assert rc == 0


def test_ingest_all_reaches_the_runners_isolation(tmp_path, monkeypatch):
    """THE finding. ``--all`` never consults this registry, yet a constructor
    in it aborted the run before the adapter-level isolation could contain
    anything."""
    monkeypatch.setattr("aggregator.cli.GitHubSource", _BoomSource)

    rc = main(
        ["ingest", "--all"],
        _store=_store(tmp_path),
        _adapters=[_FakeAdapter("alpha")],
    )

    assert rc == 0


def test_one_broken_source_does_not_block_ingesting_another(
    tmp_path, monkeypatch
):
    """Per-source isolation has to cover CONSTRUCTION, not just acquisition."""
    monkeypatch.setattr("aggregator.cli.GitHubSource", _BoomSource)
    monkeypatch.setattr("aggregator.cli.ResearchReportsSource", _StubResearch)
    store = _store(tmp_path)

    rc = main(["ingest", "research"], _store=store)

    assert rc == 0
    assert store.count_by_source("research") == 1


def test_ingesting_the_broken_source_itself_fails_loudly_not_fatally(
    tmp_path, monkeypatch, capsys
):
    """A traceback is not a report. The failure must name the source and come
    back as an exit code the timer wrapper already understands."""
    monkeypatch.setattr("aggregator.cli.GitHubSource", _BoomSource)

    rc = main(["ingest", "github"], _store=_store(tmp_path))

    assert rc == 1
    err = capsys.readouterr().err
    assert "github" in err
    assert "XDG_DATA_HOME points at a file" in err
