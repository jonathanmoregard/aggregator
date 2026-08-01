"""M5: aggregator CLI. RED tests first.

Covers the five subcommand behaviours specified in the M5 plan:

* ``query`` — human-readable wrapped output (default mode)
* ``query --json`` — machine-readable JSON envelope
* ``status`` — capabilities dump (sources, freshness, cache path)
* ``ingest <source>`` — dispatches to the real source's ``ingest()``
* Bad DSL surfaces a non-zero exit and a structured error on stderr

All tests use the ``tmp_data_home`` fixture (conftest.py) so cache.db lives
in a per-test tmpdir, and inject dependencies via the ``_store`` / ``_sources``
seams so unit tests never hit the real gh CLI or ``~/.claude/projects``.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime

from aggregator import cli
from aggregator.core.store import Store
from aggregator.sources.base import IngestResult, Record


def _seed(store: Store) -> None:
    store.migrate()
    store.upsert(
        [
            Record(
                stable_id="sessions:a",
                source="sessions",
                subject="hi",
                body="refactor foo.py",
                tags=["proj-alpha"],
                created_at=datetime(2026, 7, 25, tzinfo=UTC),
                updated_at=datetime(2026, 7, 25, tzinfo=UTC),
            ),
        ]
    )


def test_query_command_prints_wrapped_content(tmp_data_home, capsys):
    store = Store()
    _seed(store)
    rc = cli.main(["query", "source:sessions", "--fields", "full"], _store=store)
    assert rc == 0
    out = capsys.readouterr().out
    assert '<ExternalContent source="sessions:a">' in out
    assert "refactor foo.py" in out


def test_status_command_prints_capabilities(tmp_data_home, capsys):
    store = Store()
    _seed(store)
    rc = cli.main(["status"], _store=store)
    assert rc == 0
    out = capsys.readouterr().out
    assert "sessions" in out
    assert "cache_path" in out or "cache.db" in out


def test_ingest_command_dispatches(tmp_data_home, capsys):
    store = Store()
    store.migrate()
    called: dict[str, bool] = {}

    class StubSource:
        name = "sessions"

        def ingest(self, since):
            called["yes"] = True
            return IngestResult(added=1, updated=0, skipped=0)

    rc = cli.main(
        ["ingest", "sessions"], _store=store, _sources={"sessions": StubSource()}
    )
    assert rc == 0
    assert called.get("yes")
    out = capsys.readouterr().out
    assert "added=1" in out


def test_bad_dsl_returns_nonzero(tmp_data_home, capsys):
    store = Store()
    store.migrate()
    rc = cli.main(["query", "from:not-a-date"], _store=store)
    assert rc != 0
    err = capsys.readouterr().err
    assert "reason" in err.lower() or "date" in err.lower()


def test_query_json_output(tmp_data_home, capsys):
    store = Store()
    _seed(store)
    rc = cli.main(["query", "source:sessions", "--json"], _store=store)
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["ok"] is True
    assert data["records"][0]["source"] == "sessions"


def test_unknown_source_returns_nonzero(tmp_data_home, capsys):
    """Ingest against an unregistered source is a user-input error, not a crash."""
    store = Store()
    store.migrate()
    rc = cli.main(
        ["ingest", "nope"], _store=store, _sources={"sessions": object()}
    )
    assert rc != 0
    err = capsys.readouterr().err
    assert "nope" in err
