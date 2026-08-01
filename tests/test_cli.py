"""M5: aggregator CLI (v2, Schema B).

Covers:

* ``query`` — human-readable wrapped output for records + session hit lists.
* ``query --json`` — machine-readable JSON envelope.
* ``status`` — capabilities dump.
* ``ingest <source>`` — dispatches to the real source's ingest.
* Bad DSL surfaces a non-zero exit and a structured error on stderr.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime

from aggregator import cli
from aggregator.core.store import Store
from aggregator.sources.base import IngestResult, Record


def _seed_records(store: Store) -> None:
    store.migrate()
    store.upsert(
        [
            Record(
                stable_id="github:acme/api:1",
                source="github",
                subject="hi",
                body="refactor foo.py",
                tags=["pr"],
                created_at=datetime(2026, 7, 25, tzinfo=UTC),
                updated_at=datetime(2026, 7, 25, tzinfo=UTC),
            ),
        ]
    )


def test_query_command_prints_wrapped_content(tmp_data_home, capsys):
    store = Store()
    _seed_records(store)
    rc = cli.main(["query", "source:github", "--fields", "full"], _store=store)
    assert rc == 0
    out = capsys.readouterr().out
    assert '<ExternalContent source="github:acme/api:1">' in out
    assert "refactor foo.py" in out


def test_status_command_prints_capabilities(tmp_data_home, capsys):
    store = Store()
    _seed_records(store)
    rc = cli.main(["status"], _store=store)
    assert rc == 0
    out = capsys.readouterr().out
    assert "github" in out
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
    _seed_records(store)
    rc = cli.main(["query", "source:github", "--json"], _store=store)
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["ok"] is True
    assert data["records"][0]["source"] == "github"


def test_unknown_source_returns_nonzero(tmp_data_home, capsys):
    store = Store()
    store.migrate()
    rc = cli.main(
        ["ingest", "nope"], _store=store, _sources={"sessions": object()}
    )
    assert rc != 0
    err = capsys.readouterr().err
    assert "nope" in err


def test_query_drilldown_flag_present(tmp_data_home, capsys):
    """--drilldown is accepted; on sessions path returns observation rows."""
    store = Store()
    store.migrate()
    from aggregator.sources.base import ObservationRow, SessionRow
    store.upsert_entities(
        [
            SessionRow(
                session_id="sess-a",
                root_session_id="sess-a",
                parent_session_id=None,
                kind="session",
                agent_id=None,
                agent_type=None,
                spawned_by_tool_use_id=None,
                cwd=None,
                git_branch=None,
                first_ts=datetime(2026, 7, 25, tzinfo=UTC),
                last_ts=datetime(2026, 7, 25, tzinfo=UTC),
                jsonl_path="/tmp/x",
            ),
            ObservationRow(
                obs_id="o1",
                session_id="sess-a",
                root_session_id="sess-a",
                parent_obs_id=None,
                type="user",
                ts=datetime(2026, 7, 25, tzinfo=UTC),
                model=None,
                input_tokens=None,
                output_tokens=None,
                tool_name=None,
                tool_use_id=None,
                body="hello",
            ),
        ]
    )
    rc = cli.main(
        ["query", "source:sessions", "--drilldown", "--json"], _store=store
    )
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["mode"] == "observations"
