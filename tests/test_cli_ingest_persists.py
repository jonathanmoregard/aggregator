"""Regression: `aggregator ingest` must actually persist records to the store.

Covers both the Record path (github) and the v2 entity path (sessions),
plus the Chunk-4 wiring of the three new sources (chatgpt, claude-web,
research) through the default registry.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from aggregator import cli
from aggregator.core.store import Store
from aggregator.sources.base import (
    IngestResult,
    ObservationRow,
    QueryAST,
    Record,
    SessionRow,
)

# --- Record path (github) -------------------------------------------------


class _StubRecordSource:
    """Yields a fixed Record set via ``iter_records``. Github-shaped source."""

    name = "github"

    def __init__(self, records: list[Record]):
        self._records = records

    def iter_records(self, since):
        for r in self._records:
            if since and r.updated_at and r.updated_at < since:
                continue
            yield r

    def ingest(self, since):
        return IngestResult(
            added=sum(1 for _ in self.iter_records(since)),
            updated=0,
            skipped=0,
        )

    def search(self, ast):  # pragma: no cover
        return []

    def record_shape(self):
        return {}


def _mk_records() -> list[Record]:
    return [
        Record(
            stable_id="github:acme/api:1",
            source="github",
            subject="first",
            body="body one",
            tags=["pr"],
            created_at=datetime(2026, 7, 25, tzinfo=UTC),
            updated_at=datetime(2026, 7, 25, tzinfo=UTC),
        ),
        Record(
            stable_id="github:acme/api:2",
            source="github",
            subject="second",
            body="body two",
            tags=["pr"],
            created_at=datetime(2026, 7, 26, tzinfo=UTC),
            updated_at=datetime(2026, 7, 26, tzinfo=UTC),
        ),
    ]


def test_ingest_records_persists_to_store(tmp_data_home):
    store = Store()
    store.migrate()
    source = _StubRecordSource(_mk_records())

    rc = cli.main(
        ["ingest", "github"], _store=store, _sources={"github": source}
    )
    assert rc == 0

    stored = store.query(QueryAST(source="github"))
    stored_ids = {r.stable_id for r in stored}
    assert stored_ids == {"github:acme/api:1", "github:acme/api:2"}


def test_ingest_records_rebuild_drops_before_persist(tmp_data_home):
    store = Store()
    store.migrate()
    stale = Record(
        stable_id="github:acme/api:stale",
        source="github",
        subject="stale",
        body="stale body",
        created_at=datetime(2026, 7, 1, tzinfo=UTC),
        updated_at=datetime(2026, 7, 1, tzinfo=UTC),
    )
    store.upsert([stale])

    source = _StubRecordSource(_mk_records())
    rc = cli.main(
        ["ingest", "github", "--rebuild"],
        _store=store,
        _sources={"github": source},
    )
    assert rc == 0
    stored_ids = {r.stable_id for r in store.query(QueryAST(source="github"))}
    assert "github:acme/api:stale" not in stored_ids
    assert stored_ids == {"github:acme/api:1", "github:acme/api:2"}


# --- v2 entity path (sessions) --------------------------------------------


class _StubEntitySource:
    """Yields SessionRow + ObservationRow entities via iter_entities."""

    name = "sessions"

    def __init__(self, entities):
        self._entities = list(entities)

    def iter_entities(self, since, errors=None):
        yield from self._entities

    def ingest(self, since):  # pragma: no cover
        return IngestResult(added=len(self._entities), updated=0, skipped=0)

    def record_shape(self):
        return {}


def _mk_entities():
    now = datetime(2026, 7, 25, 10, 0, tzinfo=UTC)
    return [
        SessionRow(
            session_id="sess-a",
            root_session_id="sess-a",
            parent_session_id=None,
            kind="session",
            agent_id=None,
            agent_type=None,
            spawned_by_tool_use_id=None,
            cwd="/x",
            git_branch="main",
            first_ts=now,
            last_ts=now,
            jsonl_path="/tmp/a.jsonl",
        ),
        ObservationRow(
            obs_id="o-a-1",
            session_id="sess-a",
            root_session_id="sess-a",
            parent_obs_id=None,
            type="user",
            ts=now,
            model=None,
            input_tokens=None,
            output_tokens=None,
            tool_name=None,
            tool_use_id=None,
            body="hello world",
        ),
    ]


def test_ingest_entities_persists_to_store(tmp_data_home):
    store = Store()
    store.migrate()
    source = _StubEntitySource(_mk_entities())

    rc = cli.main(
        ["ingest", "sessions"], _store=store, _sources={"sessions": source}
    )
    assert rc == 0
    sessions = store.query_sessions(QueryAST())
    assert {s.session_id for s in sessions} == {"sess-a"}
    obs = store.query_observations(QueryAST(top_session_id="sess-a"))
    assert {o.obs_id for o in obs} == {"o-a-1"}


def test_ingest_entities_rebuild_drops_before_persist(tmp_data_home):
    store = Store()
    store.migrate()
    stale_session = SessionRow(
        session_id="sess-stale",
        root_session_id="sess-stale",
        parent_session_id=None,
        kind="session",
        agent_id=None,
        agent_type=None,
        spawned_by_tool_use_id=None,
        cwd=None,
        git_branch=None,
        first_ts=datetime(2026, 7, 1, tzinfo=UTC),
        last_ts=datetime(2026, 7, 1, tzinfo=UTC),
        jsonl_path="/tmp/stale.jsonl",
    )
    store.upsert_entities([stale_session])

    source = _StubEntitySource(_mk_entities())
    rc = cli.main(
        ["ingest", "sessions", "--rebuild"],
        _store=store,
        _sources={"sessions": source},
    )
    assert rc == 0
    ids = {s.session_id for s in store.query_sessions(QueryAST())}
    assert ids == {"sess-a"}, ids


# --- Chunk 4: chatgpt / claude-web / research wired into the CLI -----------


def test_default_sources_registry_includes_new_sources():
    names = set(cli._default_sources())
    assert {
        "sessions",
        "github",
        "chatgpt",
        "claude-web",
        "research",
        "sota-watch",
    } <= names


def test_ingest_research_persists_records(tmp_data_home, tmp_path, monkeypatch):
    """`aggregator ingest research` through the DEFAULT registry lands
    Records in the store (records path, like github)."""
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "abc123.md").write_text(
        "# Export formats survey\n\nreport body text\n", encoding="utf-8"
    )
    monkeypatch.setenv("AGGREGATOR_RESEARCH_REPORTS_DIR", str(reports))
    store = Store()
    store.migrate()
    rc = cli.main(["ingest", "research"], _store=store)
    assert rc == 0
    stored = store.query(QueryAST(source="research"))
    assert [r.stable_id for r in stored] == ["research:abc123"]
    assert stored[0].subject == "Export formats survey"
    assert stored[0].tags == ["research"]


def test_ingest_sota_watch_persists_records(tmp_data_home, tmp_path, monkeypatch):
    """`aggregator ingest sota-watch` through the DEFAULT registry lands
    Records in the store (records path, like github + research)."""
    proposals = tmp_path / "proposals"
    proposals.mkdir()
    (proposals / "2026-07-31-tts.md").write_text(
        "# TTS proposal\n\nproposal body\n", encoding="utf-8"
    )
    monkeypatch.setenv("AGGREGATOR_SOTA_WATCH_DIR", str(proposals))
    store = Store()
    store.migrate()
    rc = cli.main(["ingest", "sota-watch"], _store=store)
    assert rc == 0
    stored = store.query(QueryAST(source="sota-watch"))
    assert [r.stable_id for r in stored] == ["sota-watch:2026-07-31-tts"]
    assert stored[0].subject == "TTS proposal"
    assert stored[0].tags == ["sota-watch"]


def test_ingest_chatgpt_persists_sessions(tmp_data_home, repo_root, monkeypatch):
    """`aggregator ingest chatgpt` through the DEFAULT registry lands
    SessionRows (origin=chatgpt) + ObservationRows via the entity path."""
    fixtures = Path(repo_root) / "tests" / "fixtures" / "chatgpt"
    monkeypatch.setenv("AGGREGATOR_DROPS_DIR", str(fixtures))
    store = Store()
    store.migrate()
    rc = cli.main(["ingest", "chatgpt"], _store=store)
    assert rc == 0
    sessions = store.query_sessions(QueryAST(source="chatgpt"))
    assert {s.session_id for s in sessions} == {
        "chatgpt:conv-uuid-1",
        "chatgpt:conv-uuid-2",
    }
    obs = store.query_observations(QueryAST(source="chatgpt"))
    assert any(o.obs_id == "chatgpt:node-u1" for o in obs)


def test_ingest_claude_web_persists_sessions(tmp_data_home, repo_root, monkeypatch):
    fixtures = Path(repo_root) / "tests" / "fixtures" / "claude-web"
    monkeypatch.setenv("AGGREGATOR_DROPS_DIR", str(fixtures))
    store = Store()
    store.migrate()
    rc = cli.main(["ingest", "claude-web"], _store=store)
    assert rc == 0
    sessions = store.query_sessions(QueryAST(source="claude-web"))
    assert len(sessions) == 2
    assert all(s.origin == "claude-web" for s in sessions)


def test_ingest_chatgpt_with_empty_drops_exits_zero(tmp_data_home, tmp_path,
                                                    monkeypatch, capsys):
    """No drops present → clean exit with 0 found, never a crash."""
    empty = tmp_path / "empty-drops"
    empty.mkdir()
    monkeypatch.setenv("AGGREGATOR_DROPS_DIR", str(empty))
    store = Store()
    store.migrate()
    rc = cli.main(["ingest", "chatgpt"], _store=store)
    assert rc == 0
    out = capsys.readouterr().out
    assert "sessions=0" in out


def test_ingest_chat_source_rebuild_refused(tmp_data_home, repo_root,
                                            monkeypatch, capsys):
    """--rebuild on a chat-export source must refuse: the entity rebuild
    path replaces the ENTIRE sessions/observations tables, which would wipe
    every other origin's rows (claude-code sessions included)."""
    store = Store()
    store.migrate()
    store.upsert_entities([_mk_entities()[0]])  # a claude-code session
    fixtures = Path(repo_root) / "tests" / "fixtures" / "chatgpt"
    monkeypatch.setenv("AGGREGATOR_DROPS_DIR", str(fixtures))
    rc = cli.main(["ingest", "chatgpt", "--rebuild"], _store=store)
    assert rc != 0
    err = capsys.readouterr().err
    assert "rebuild" in err.lower()
    # The pre-existing claude-code session is untouched.
    ids = {s.session_id for s in store.query_sessions(QueryAST())}
    assert "sess-a" in ids
