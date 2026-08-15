"""Tests for the v2 sessions source (Schema B, Langfuse-derived).

Ontology: JSONL files → SessionRow + ObservationRow entities.

* Top-level file → ``kind='session'``, no parent.
* ``<sessionId>/subagents/agent-<agentId>.jsonl`` → ``kind='subagent'``,
  composite key ``<sessionId>:<agentId>``, ``parent_session_id`` from dir name,
  ``spawned_by_tool_use_id`` recovered from parent's Task tool_use window.
* Resume prefix-copy: file starts with parent's sessionId lines, then switches
  to the file's own dominant sessionId; only the dominant portion is ingested.

Fixtures under ``tests/fixtures/sessions-v2/``.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from aggregator.sources.base import ObservationRow, SessionRow
from aggregator.sources.sessions import SessionsSource


@pytest.fixture
def fixtures_dir(repo_root):
    """Back-date v2 fixtures past the 5-min live window so the parser sees them."""
    d = Path(repo_root) / "tests" / "fixtures" / "sessions-v2"
    old = time.time() - 24 * 60 * 60
    for p in d.rglob("*.jsonl"):
        os.utime(p, (old, old))
    return d


def _split_entities(src):
    sessions: list[SessionRow] = []
    observations: list[ObservationRow] = []
    errors: list[str] = []
    for e in src.iter_entities(errors=errors):
        if isinstance(e, SessionRow):
            sessions.append(e)
        elif isinstance(e, ObservationRow):
            observations.append(e)
    return sessions, observations, errors


def test_top_level_session_row(fixtures_dir):
    src = SessionsSource(projects_root=str(fixtures_dir))
    sessions, _obs, errs = _split_entities(src)
    assert errs == []
    top = next(s for s in sessions if s.session_id == "sess-top-abc")
    assert top.kind == "session"
    assert top.root_session_id == "sess-top-abc"
    assert top.parent_session_id is None
    assert top.agent_id is None
    assert top.cwd == "/home/u/proj-alpha"
    assert top.git_branch == "main"
    # first_ts / last_ts derived from observations, NOT file mtime.
    assert top.first_ts.isoformat().startswith("2026-07-25T10:00:01")
    assert top.last_ts.isoformat().startswith("2026-07-25T10:04:00")


def test_emitted_session_rows_default_origin_claude_code(fixtures_dir):
    """v3: sessions.py's emission is untouched — every SessionRow it yields
    must land with origin='claude-code' via the dataclass default."""
    src = SessionsSource(projects_root=str(fixtures_dir))
    sessions, _obs, _errs = _split_entities(src)
    assert sessions, "fixtures must yield at least one session row"
    assert all(s.origin == "claude-code" for s in sessions), (
        f"non-claude-code origins: {[(s.session_id, s.origin) for s in sessions if s.origin != 'claude-code']}"
    )


def test_subagent_session_row_with_composite_key(fixtures_dir):
    src = SessionsSource(projects_root=str(fixtures_dir))
    sessions, _obs, _errs = _split_entities(src)
    sub = next(s for s in sessions if s.kind == "subagent")
    assert sub.session_id == "sess-top-abc:agent001"
    assert sub.root_session_id == "sess-top-abc"
    assert sub.parent_session_id == "sess-top-abc"
    assert sub.agent_id == "agent001"
    assert sub.agent_type == "analyzer"


def test_subagent_spawn_id_recovered_from_agent_toolresult_structured(fixtures_dir):
    """B1 fix: parent Agent tool_result carries the child agentId in a
    structured ``toolUseResult.agentId`` field. Parser joins agentId → the
    parent's Agent tool_use_id (exact match, no time-window fuzz).

    Fixture: tu-task-1 (Agent tool) → tool_result on line 6 with
    ``toolUseResult.agentId == "agent001"`` → subagent agent001 recovers.
    """
    src = SessionsSource(projects_root=str(fixtures_dir))
    sessions, _obs, _errs = _split_entities(src)
    sub = next(s for s in sessions if s.kind == "subagent")
    assert sub.spawned_by_tool_use_id == "tu-task-1"


def test_subagent_spawn_id_recovered_from_agentid_text_fallback(tmp_path):
    """B1 fix, fallback path: some parent tool_results only echo the child
    agentId in the LLM-facing text (``agentId: <id>``) without a top-level
    ``toolUseResult`` field. Parser must still recover via regex.
    """
    proj = tmp_path / "proj" / "sess-parent-xyz"
    proj.mkdir(parents=True)
    subdir = proj / "subagents"
    subdir.mkdir()
    # Parent JSONL — Agent tool_use + tool_result WITHOUT toolUseResult;
    # only text-echoes the agentId.
    parent_path = tmp_path / "proj" / "sess-parent-xyz.jsonl"
    parent_path.write_text(
        json.dumps(
            {
                "type": "assistant",
                "sessionId": "sess-parent-xyz",
                "uuid": "u-p-1",
                "cwd": "/x",
                "timestamp": "2026-07-27T10:00:00.000Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "tu-agent-42", "name": "Agent",
                         "input": {"description": "d"}},
                    ],
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "user",
                "sessionId": "sess-parent-xyz",
                "uuid": "u-p-2",
                "cwd": "/x",
                "timestamp": "2026-07-27T10:00:05.000Z",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu-agent-42",
                            "content": [
                                {"type": "text",
                                 "text": "Async agent launched successfully.\n"
                                          "agentId: xff42abcdef1234 (internal ID)"},
                            ],
                        }
                    ],
                },
            }
        )
        + "\n"
    )
    # Subagent file for xff42abcdef1234
    sub_path = subdir / "agent-xff42abcdef1234.jsonl"
    sub_path.write_text(
        json.dumps(
            {
                "type": "user",
                "sessionId": "xff42abcdef1234",
                "uuid": "u-s-1",
                "cwd": "/x",
                "timestamp": "2026-07-27T10:00:10.000Z",
                "agentId": "xff42abcdef1234",
                "isSidechain": True,
                "message": {"role": "user", "content": "start"},
            }
        )
        + "\n"
    )
    old = time.time() - 24 * 60 * 60
    for f in (parent_path, sub_path):
        os.utime(f, (old, old))
    src = SessionsSource(projects_root=str(tmp_path))
    sessions, _obs, _errs = _split_entities(src)
    sub = next(s for s in sessions if s.kind == "subagent")
    assert sub.agent_id == "xff42abcdef1234"
    assert sub.spawned_by_tool_use_id == "tu-agent-42"


def test_resume_prefix_copy_is_filtered(fixtures_dir):
    """Resume file starts with lines under ``sess-original-parent`` (prefix
    copy from parent) then switches to ``sess-resumed-xyz``. Only the dominant
    (filename-matching) sessionId's lines are ingested under the resumed
    session; the prefix-copy portion is dropped for this file's session."""
    src = SessionsSource(projects_root=str(fixtures_dir))
    sessions, observations, _errs = _split_entities(src)
    resumed = next(s for s in sessions if s.session_id == "sess-resumed-xyz")
    # First ts of the resumed session must be the "resume" line (09:00:00),
    # NOT the earlier prefix-copy timestamps (08:00:xx).
    assert resumed.first_ts.isoformat().startswith("2026-07-26T09:00:00")
    # No observations under sess-original-parent should exist (that
    # sessionId is not the dominant one for this file, and no file names
    # it).
    orig_obs = [o for o in observations if o.session_id == "sess-original-parent"]
    assert orig_obs == []
    # The three own lines of the resumed session should be present.
    resumed_obs = [o for o in observations if o.session_id == "sess-resumed-xyz"]
    assert len(resumed_obs) == 3


def test_observation_body_extracts_first_text_block(fixtures_dir):
    src = SessionsSource(projects_root=str(fixtures_dir))
    _s, observations, _errs = _split_entities(src)
    # Assistant obs u-top-2 has a text block "I'll refactor foo.py..." plus
    # a tool_use. Body captures the text; tool_name captures the tool.
    o = next(o for o in observations if o.obs_id == "u-top-2")
    assert "refactor foo.py" in o.body
    assert o.tool_name == "Edit"
    assert o.tool_use_id == "tu-edit-1"
    # Line type promoted from 'assistant' to 'tool_use' when a tool_use block
    # is present, so the DSL type: filter can distinguish.
    assert o.type == "tool_use"


def test_observation_tool_result_type_and_body(fixtures_dir):
    src = SessionsSource(projects_root=str(fixtures_dir))
    _s, observations, _errs = _split_entities(src)
    # user line u-top-3 is a tool_result; type promoted to 'tool_result' when
    # the content contains no text-block-only body.
    o = next(o for o in observations if o.obs_id == "u-top-3")
    assert o.type == "tool_result"
    assert o.tool_use_id == "tu-edit-1"


def test_tool_use_only_message_serialises_name_and_input_into_body(tmp_path):
    """B3 regression: tool_use blocks with NO preceding text block used to
    produce empty body → FTS `type:tool_use foo` never hit anything in real
    data. Parser must fold tool_name + input into body so FTS indexes it.
    """
    p = tmp_path / "sess-tool.jsonl"
    p.write_text(
        json.dumps(
            {
                "type": "assistant",
                "sessionId": "sess-tool",
                "uuid": "u-tool-1",
                "cwd": "/x",
                "timestamp": "2026-07-27T10:00:00Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu-1",
                            "name": "suggest_doc_edit",
                            "input": {"docId": "abc", "edits": [{"note": "fix typo"}]},
                        }
                    ],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            }
        )
        + "\n"
    )
    old = time.time() - 24 * 60 * 60
    os.utime(p, (old, old))
    src = SessionsSource(projects_root=str(tmp_path))
    _s, observations, _errs = _split_entities(src)
    o = next(o for o in observations if o.obs_id == "u-tool-1")
    assert o.type == "tool_use"
    assert o.tool_name == "suggest_doc_edit"
    # Body must contain the tool NAME plus its input payload so FTS finds it.
    assert "suggest_doc_edit" in o.body
    assert "abc" in o.body
    assert "fix typo" in o.body


def test_observation_denormalises_root_session_id(fixtures_dir):
    """Langfuse trick: root_session_id on every observation lets
    "everything under X" be an indexed equality."""
    src = SessionsSource(projects_root=str(fixtures_dir))
    _s, observations, _errs = _split_entities(src)
    sub_obs = [o for o in observations if o.session_id == "sess-top-abc:agent001"]
    assert sub_obs, "expected subagent observations"
    for o in sub_obs:
        assert o.root_session_id == "sess-top-abc"


def test_skips_files_modified_within_5min(tmp_path):
    live = tmp_path / "live.jsonl"
    live.write_text(
        json.dumps(
            {
                "type": "user",
                "sessionId": "sess-live",
                "uuid": "u-live-1",
                "cwd": "/x",
                "timestamp": "2026-07-27T10:00:00Z",
                "message": {"role": "user", "content": "hi"},
            }
        )
        + "\n"
    )
    now = time.time()
    os.utime(live, (now, now))
    src = SessionsSource(projects_root=str(tmp_path))
    sessions, observations, _errs = _split_entities(src)
    assert sessions == []
    assert observations == []


def test_corrupt_line_skipped_not_aborted(tmp_path):
    """Bad JSON line must not abort the ingest; error recorded."""
    p = tmp_path / "corrupt.jsonl"
    p.write_text(
        '{"type":"user","sessionId":"sess-c","uuid":"c1","timestamp":"2026-07-26T10:00:01Z",'
        '"cwd":"/x","message":{"role":"user","content":"ok"}}\n'
        "garbage that is not json\n"
        '{"type":"assistant","sessionId":"sess-c","uuid":"c2","timestamp":"2026-07-26T10:00:02Z",'
        '"cwd":"/x","message":{"role":"assistant","content":"hi"}}\n'
    )
    old = time.time() - 24 * 60 * 60
    os.utime(p, (old, old))
    src = SessionsSource(projects_root=str(tmp_path))
    sessions, observations, errors = _split_entities(src)
    assert any("corrupt" in e for e in errors)
    # The good lines still parse into one session + two observations.
    assert len(sessions) == 1
    assert len(observations) == 2


def test_a_corrupt_line_is_reported_exactly_once(tmp_path):
    """Both passes read every top-level file, so a line-level fault used to be
    appended to the same sink twice. The CLI prints ``errors[:5]``, so a file
    with three bad lines could crowd every other fault in the run out of the
    only view an operator gets."""
    p = tmp_path / "sess-dup.jsonl"
    p.write_text(
        '{"type":"user","sessionId":"sess-dup","uuid":"d1","timestamp":"2026-07-26T10:00:01Z",'
        '"cwd":"/x","message":{"role":"user","content":"ok"}}\n'
        "garbage that is not json\n"
    )
    os.utime(p, (time.time() - 24 * 60 * 60,) * 2)
    _s, _o, errors = _split_entities(SessionsSource(projects_root=str(tmp_path)))
    assert len([e for e in errors if "corrupt" in e]) == 1


def test_a_badly_corrupt_file_does_not_produce_one_error_per_line(tmp_path):
    """Round-3 MEDIUM. One entry per file, capped examples, EXACT count.

    Sessions is the largest source in the index (~359k observations). A bad
    line appended one error each, so one corrupt file ballooned the run report,
    the desktop notification payload and the memory the list occupies — and the
    CLI prints only ``errors[:5]``, so that one file also crowded every other
    fault in the run out of the only view an operator gets.

    Capping must not become the fault hiding, which is why the total is exact
    and the examples are what get capped. Same shape as github's dropped rows.
    """
    p = tmp_path / "sess-many.jsonl"
    p.write_text("not json at all\n" * 500)
    os.utime(p, (time.time() - 24 * 60 * 60,) * 2)

    _s, _o, errors = _split_entities(SessionsSource(projects_root=str(tmp_path)))

    assert len(errors) == 1
    assert "DROPPED 500 corrupt line(s)" in errors[0], "the exact count must survive"
    assert "line 1, 2, 3, 4, 5, ..." in errors[0]
    assert "6" not in errors[0].split("(line", 1)[1], "the identifiers are capped at five"


def test_the_two_line_faults_are_reported_separately(tmp_path):
    """Corrupt JSON and valid-JSON-wrong-shape are different diagnoses with
    different causes, so they aggregate into one entry EACH rather than into a
    single "some lines were bad"."""
    p = tmp_path / "sess-mixed.jsonl"
    p.write_text("{not json\n[1, 2, 3]\n{also not json\n")
    os.utime(p, (time.time() - 24 * 60 * 60,) * 2)

    _s, _o, errors = _split_entities(SessionsSource(projects_root=str(tmp_path)))

    assert len(errors) == 2
    assert any("DROPPED 2 corrupt line(s)" in e for e in errors)
    assert any("DROPPED 1 line(s) that are valid JSON" in e for e in errors)


def test_an_unstattable_file_reaches_the_errors_sink(tmp_path):
    """Round-5 HIGH 3. A whole JSONL dropped by a failed ``stat`` was invisible.

    ``_iter_jsonl_files`` consults mtime for the live-file window and answered
    any OSError with a bare ``continue`` — no error, no log. Sessions is the
    largest source in the index (~5,678 sessions / ~359k observations), so a
    permission change on one project directory, or a dangling symlink left by a
    moved project, silently shrinks the index on a run that reports success.

    A dangling symlink is used because it raises the same OSError as the
    permission case without leaving an unreadable directory behind for pytest
    to trip over during cleanup.
    """
    good = tmp_path / "sess-good.jsonl"
    good.write_text(
        '{"type":"user","sessionId":"sess-good","uuid":"g1","timestamp":"2026-07-26T10:00:01Z",'
        '"cwd":"/x","message":{"role":"user","content":"ok"}}\n'
    )
    os.utime(good, (time.time() - 24 * 60 * 60,) * 2)
    (tmp_path / "sess-gone.jsonl").symlink_to(tmp_path / "no-such-target.jsonl")

    sessions, _obs, errors = _split_entities(SessionsSource(projects_root=str(tmp_path)))

    # Per-item, not fatal: the readable file still lands. That is by design.
    assert [s.session_id for s in sessions] == ["sess-good"]
    assert len(errors) == 1
    assert "sess-gone.jsonl" in errors[0]


def test_a_non_dict_jsonl_line_reaches_the_errors_sink(tmp_path):
    """Round-5 HIGH 3. A JSONL line that parses but is not an object was
    skipped with NO LOG AT ALL — the quietest drop in the largest source.

    It is exactly what a truncated-then-appended file or a writer bug leaves
    behind, and it is indistinguishable from a session that simply had fewer
    observations. Per-item, so the surrounding good lines still land.
    """
    p = tmp_path / "sess-shapes.jsonl"
    p.write_text(
        '{"type":"user","sessionId":"sess-shapes","uuid":"s1",'
        '"timestamp":"2026-07-26T10:00:01Z","cwd":"/x",'
        '"message":{"role":"user","content":"ok"}}\n'
        '"a bare string, valid JSON, not an object"\n'
        "[1, 2, 3]\n"
    )
    os.utime(p, (time.time() - 24 * 60 * 60,) * 2)

    sessions, observations, errors = _split_entities(
        SessionsSource(projects_root=str(tmp_path))
    )

    assert len(sessions) == 1
    assert len(observations) == 1
    # One entry per file per fault CLASS, with the exact count in it — see
    # ``test_a_badly_corrupt_file_does_not_produce_one_error_per_line``.
    assert len(errors) == 1
    assert "sess-shapes.jsonl" in errors[0]
    assert "DROPPED 2" in errors[0]
    assert "2 (str)" in errors[0] and "3 (list)" in errors[0]


def test_attachment_and_progress_types_preserved_not_bucketed_as_other(tmp_path):
    """M2 fix: real Claude Code JSONLs emit ``type='attachment'`` (hook
    results, file uploads) and ``type='progress'`` (in-flight markers). These
    used to fold into ``other`` (74k+ rows on real cache, 99.8% of ``other``).
    Parser must preserve them verbatim so ``type:attachment`` queries hit.
    """
    p = tmp_path / "sess-mixed.jsonl"
    p.write_text(
        json.dumps({
            "type": "attachment",
            "sessionId": "sess-mixed",
            "uuid": "u-att-1",
            "timestamp": "2026-07-28T10:00:01.000Z",
            "cwd": "/x",
            "attachment": {"type": "hook_success", "hookName": "UserPromptSubmit"},
        })
        + "\n"
        + json.dumps({
            "type": "progress",
            "sessionId": "sess-mixed",
            "uuid": "u-prog-1",
            "timestamp": "2026-07-28T10:00:02.000Z",
            "cwd": "/x",
        })
        + "\n"
        + json.dumps({
            "type": "queue-operation",
            "sessionId": "sess-mixed",
            "uuid": "u-queue-1",
            "timestamp": "2026-07-28T10:00:03.000Z",
            "cwd": "/x",
            "operation": "enqueue",
        })
        + "\n"
        + json.dumps({
            "type": "user",
            "sessionId": "sess-mixed",
            "uuid": "u-anchor-1",
            "timestamp": "2026-07-28T10:00:04.000Z",
            "cwd": "/x",
            "message": {"role": "user", "content": "hi"},
        })
        + "\n"
    )
    old = time.time() - 24 * 60 * 60
    os.utime(p, (old, old))
    src = SessionsSource(projects_root=str(tmp_path))
    _s, observations, _errs = _split_entities(src)
    by_id = {o.obs_id: o for o in observations}
    assert by_id["u-att-1"].type == "attachment"
    assert by_id["u-prog-1"].type == "progress"
    assert by_id["u-queue-1"].type == "queue-operation"
    # ``other`` reserved for truly unknown future types.
    assert all(o.type != "other" for o in observations)


def test_record_shape_documents_v2_fields():
    src = SessionsSource(projects_root="/tmp")
    shape = src.record_shape()
    for key in (
        "session_id",
        "root_session_id",
        "parent_session_id",
        "kind",
        "agent_id",
        "spawned_by_tool_use_id",
        "first_ts",
        "last_ts",
        "obs_type",
    ):
        assert key in shape, f"missing key {key!r} in record_shape"


def test_since_filter_drops_sessions_ending_before_cutoff(fixtures_dir):
    from datetime import UTC, datetime

    src = SessionsSource(projects_root=str(fixtures_dir))
    # Everything ended before 2026-08-01 in our fixtures.
    late_cutoff = datetime(2026, 8, 1, tzinfo=UTC)
    sessions = [
        e for e in src.iter_entities(since=late_cutoff)
        if isinstance(e, SessionRow)
    ]
    assert sessions == []


def test_ingest_returns_counts_via_result():
    """``ingest`` retained for protocol compat — counts sessions+obs, no persist."""
    src = SessionsSource(projects_root="/tmp/nonexistent")
    result = src.ingest(since=None)
    assert result.added == 0
    assert result.errors == []
