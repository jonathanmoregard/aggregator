"""Round-1 MEDIUM: a refused ``--rebuild`` had already consumed the source.

``_cmd_ingest`` listed the source's iterator FIRST and only then asked whether
``--rebuild`` was permitted for that source, or whether a shrink guard refused
this particular run's numbers. Iterating a source is not a read-only act:
TickTick's poll diffs the live open-task set against the previous baseline and
then makes this poll the new baseline, on disk, during iteration. A run that
consumed the baseline and exited without writing took every completion it had
just inferred with it — the Open API only ever serves OPEN tasks, so a task
that disappears between two polls is reported exactly once.

Fix: ``_rebuild_refusal`` decides before anything is iterated.
"""
from __future__ import annotations

from datetime import UTC, datetime

from aggregator import cli
from aggregator.core.store import Store
from aggregator.sources.base import Record, SessionRow


def _store(tmp_path) -> Store:
    store = Store(db_path=tmp_path / "cache.db")
    store.migrate()
    return store


class _StatefulEntitySource:
    """An entity source whose iteration has a side effect it cannot undo."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.iterated = 0

    def iter_entities(self, since, errors=None):
        self.iterated += 1
        yield SessionRow(
            session_id="s1",
            root_session_id="s1",
            parent_session_id=None,
            kind="session",
            agent_id=None,
            agent_type=None,
            spawned_by_tool_use_id=None,
            cwd=None,
            git_branch=None,
            first_ts=datetime(2026, 7, 1, tzinfo=UTC),
            last_ts=datetime(2026, 7, 1, tzinfo=UTC),
            jsonl_path="/x/s1.jsonl",
            origin=name_to_origin(self.name),
        )


def name_to_origin(name: str) -> str:
    return {"chatgpt": "chatgpt", "claude-web": "claude-web"}.get(name, "claude-code")


class _StatefulRecordSource:
    def __init__(self, name: str) -> None:
        self.name = name
        self.iterated = 0

    def iter_records(self, since, errors=None):
        self.iterated += 1
        yield Record(stable_id=f"{self.name}:1", source=self.name, subject="s", body="b")


def test_an_unsupported_rebuild_is_refused_before_the_source_is_iterated(
    tmp_path, capsys
):
    """THE ordering finding. The refusal was printed after the iterator had
    already run, so whatever that run consumed was gone regardless."""
    store = _store(tmp_path)
    src = _StatefulEntitySource("chatgpt")

    rc = cli.main(
        ["ingest", "chatgpt", "--rebuild"],
        _store=store,
        _sources={"chatgpt": src},
    )

    assert rc == 2
    assert src.iterated == 0, (
        "a run that is going to refuse must not consume the source first"
    )
    assert "--rebuild" in capsys.readouterr().err


def test_a_permitted_rebuild_still_iterates(tmp_path):
    """The ordering fix must not turn a legitimate rebuild into a no-op."""
    store = _store(tmp_path)
    src = _StatefulEntitySource("sessions")

    rc = cli.main(
        ["ingest", "sessions", "--rebuild"],
        _store=store,
        _sources={"sessions": src},
    )

    assert rc == 0
    assert src.iterated == 1
    assert store.count_sessions_by_origin(("claude-code",)) == 1


def test_a_plain_ingest_of_a_rebuild_refused_source_still_works(tmp_path):
    """Only --rebuild is refused. The everyday idempotent upsert is the whole
    point of refusing it."""
    store = _store(tmp_path)
    src = _StatefulEntitySource("chatgpt")

    rc = cli.main(
        ["ingest", "chatgpt"], _store=store, _sources={"chatgpt": src}
    )

    assert rc == 0
    assert src.iterated == 1
    assert store.count_sessions_by_origin(("chatgpt",)) == 1


def test_a_record_source_without_a_refusal_is_unaffected(tmp_path):
    store = _store(tmp_path)
    src = _StatefulRecordSource("github")

    rc = cli.main(
        ["ingest", "github", "--rebuild"], _store=store, _sources={"github": src}
    )

    assert rc == 0
    assert src.iterated == 1
