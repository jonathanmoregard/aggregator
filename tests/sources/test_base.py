import pytest

from aggregator.sources.base import (
    IngestResult,
    ObservationRow,
    QueryAST,
    Record,
    SessionRow,
    Source,
    stable_id_for,
)


def test_record_defaults():
    r = Record(stable_id="s:1", source="s", subject="t", body="b")
    assert r.tags == []
    assert r.extra == {}
    assert r.created_at is None
    assert r.updated_at is None


def test_ingest_result_defaults():
    ir = IngestResult(added=0, updated=0, skipped=0)
    assert ir.errors == []


def test_query_ast_defaults():
    ast = QueryAST()
    assert ast.source is None
    assert ast.tags == []
    assert ast.from_date is None
    assert ast.to_date is None
    assert ast.text is None
    assert ast.extra == {}
    # v2 keys default to None so unset queries don't accidentally route to the
    # sessions path.
    assert ast.session_id is None
    assert ast.top_session_id is None
    assert ast.agent_id is None
    assert ast.obs_type is None
    assert ast.active_from is None
    assert ast.active_to is None


def test_source_protocol_declares_required_methods():
    """v2 Source protocol keeps ingest + record_shape as core.

    ``iter_records`` (Record-shaped) and ``iter_entities`` (v2 entity-shaped)
    are per-source additions; the CLI's dispatch checks for one or the other.
    """
    assert hasattr(Source, "ingest")
    assert hasattr(Source, "record_shape")


def test_source_protocol_structural_check_accepts_conforming_class():
    """A class implementing name/ingest/record_shape satisfies the shape."""

    class Fake:
        name = "fake"

        def ingest(self, since):
            return IngestResult(added=0, updated=0, skipped=0)

        def record_shape(self):
            return {"stable_id": "str"}

    f = Fake()
    assert f.name == "fake"
    assert isinstance(f.ingest(None), IngestResult)
    assert f.record_shape() == {"stable_id": "str"}


def test_session_row_and_observation_row_dataclass_shapes():
    """v2 entities carry the shape the store writes into sessions/observations."""
    from datetime import UTC, datetime

    s = SessionRow(
        session_id="sess-x",
        root_session_id="sess-x",
        parent_session_id=None,
        kind="session",
        agent_id=None,
        agent_type=None,
        spawned_by_tool_use_id=None,
        cwd="/x",
        git_branch="main",
        first_ts=datetime(2026, 7, 25, tzinfo=UTC),
        last_ts=datetime(2026, 7, 25, tzinfo=UTC),
        jsonl_path="/tmp/x.jsonl",
    )
    o = ObservationRow(
        obs_id="o1",
        session_id="sess-x",
        root_session_id="sess-x",
        parent_obs_id=None,
        type="user",
        ts=datetime(2026, 7, 25, tzinfo=UTC),
        model=None,
        input_tokens=None,
        output_tokens=None,
        tool_name=None,
        tool_use_id=None,
        body="hi",
    )
    assert s.kind in ("session", "subagent")
    assert o.type == "user"


def test_stable_id_for_formats_source_and_id():
    assert stable_id_for("sessions", "abc-123") == "sessions:abc-123"
    assert stable_id_for("github", "owner/repo:42") == "github:owner/repo:42"


def test_stable_id_for_rejects_empty_source():
    with pytest.raises(ValueError, match="invalid source"):
        stable_id_for("", "abc")


def test_stable_id_for_rejects_colon_in_source():
    """Source name must not contain ':' so parsing '<source>:<rest>' stays unambiguous."""
    with pytest.raises(ValueError, match="invalid source"):
        stable_id_for("bad:source", "abc")


def test_stable_id_for_rejects_empty_id():
    with pytest.raises(ValueError, match="non-empty"):
        stable_id_for("sessions", "")
