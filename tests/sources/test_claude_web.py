"""Tests for the claude-web source (Claude.ai data-export parser, Chunk 3).

Ontology: ``conversations.json`` (bare file or inside the vendor export zip)
→ ``SessionRow`` (one per conversation) + ``ObservationRow`` (one per
chat_message, plus one per tool_use/tool_result content block).

Key behaviours under test:

* Stable-ID prefixing: ``claude-web:<uuid>`` for sessions and observations.
* Nil-UUID parent sentinel (``00000000-0000-4000-8000-…``) → ``parent_obs_id
  = None``.
* Branching is real — regenerated siblings share ``parent_message_uuid`` and
  ALL are kept.
* Tool blocks become separate observations parented on the message obs.
* Shape sniff: ChatGPT-shaped ``conversations.json`` in the same drops dir is
  silently skipped (a parallel source owns those).

Committed fixture: ``tests/fixtures/claude-web/conversations.json``. The zip
fixture is built in-test via :mod:`zipfile` — no committed binaries.
"""
from __future__ import annotations

import json
import os
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from aggregator.sources.base import ObservationRow, SessionRow
from aggregator.sources.claude_web import ClaudeWebSource

CONV1 = "aaaa1111-2222-4333-8444-555566667777"
CONV2 = "bbbb1111-2222-4333-8444-555566667777"
MSG = "11111111-aaaa-4bbb-8ccc-00000000000{n}"


def _m(n: int) -> str:
    return MSG.format(n=n)


@pytest.fixture
def fixtures_dir(repo_root):
    return Path(repo_root) / "tests" / "fixtures" / "claude-web"


def _split_entities(src, since=None):
    sessions: list[SessionRow] = []
    observations: list[ObservationRow] = []
    errors: list[str] = []
    for e in src.iter_entities(since=since, errors=errors):
        if isinstance(e, SessionRow):
            sessions.append(e)
        elif isinstance(e, ObservationRow):
            observations.append(e)
    return sessions, observations, errors


def test_session_row_shape(fixtures_dir):
    src = ClaudeWebSource(drops_dir=str(fixtures_dir))
    sessions, _obs, errs = _split_entities(src)
    assert errs == []
    assert len(sessions) == 2
    s = next(s for s in sessions if s.session_id == f"claude-web:{CONV1}")
    assert s.root_session_id == f"claude-web:{CONV1}"
    assert s.parent_session_id is None
    assert s.kind == "session"
    assert s.origin == "claude-web"
    assert s.agent_id is None
    assert s.agent_type is None
    assert s.spawned_by_tool_use_id is None
    assert s.cwd is None
    assert s.git_branch is None
    # ISO-8601 Z timestamps, microsecond AND millisecond precision both parse.
    assert s.first_ts == datetime(2026, 7, 1, 10, 0, 0, 123456, tzinfo=UTC)
    assert s.last_ts == datetime(2026, 7, 1, 10, 6, 0, 999000, tzinfo=UTC)
    assert s.jsonl_path == str(fixtures_dir / "conversations.json")


def test_session_row_yielded_before_its_observations(fixtures_dir):
    src = ClaudeWebSource(drops_dir=str(fixtures_dir))
    seen_sessions: set[str] = set()
    for e in src.iter_entities():
        if isinstance(e, SessionRow):
            seen_sessions.add(e.session_id)
        elif isinstance(e, ObservationRow):
            assert e.session_id in seen_sessions, (
                f"observation {e.obs_id} yielded before its SessionRow"
            )


def test_parent_chain_and_nil_sentinel(fixtures_dir):
    src = ClaudeWebSource(drops_dir=str(fixtures_dir))
    _s, observations, _errs = _split_entities(src)
    by_id = {o.obs_id: o for o in observations}
    # Root message: nil-UUID sentinel parent → None.
    root = by_id[f"claude-web:{_m(1)}"]
    assert root.parent_obs_id is None
    assert root.type == "user"
    # Child points at the prefixed parent obs id.
    child = by_id[f"claude-web:{_m(2)}"]
    assert child.parent_obs_id == f"claude-web:{_m(1)}"
    assert child.type == "assistant"
    # Nil sentinel matches on PREFIX (conv2 uses a non-zero tail).
    conv2_root = by_id["claude-web:22222222-aaaa-4bbb-8ccc-000000000001"]
    assert conv2_root.parent_obs_id is None


def test_sibling_regenerations_both_present(fixtures_dir):
    src = ClaudeWebSource(drops_dir=str(fixtures_dir))
    _s, observations, _errs = _split_entities(src)
    siblings = [
        o for o in observations
        if o.parent_obs_id == f"claude-web:{_m(1)}"
    ]
    assert {o.obs_id for o in siblings} == {
        f"claude-web:{_m(2)}",
        f"claude-web:{_m(3)}",
    }


def test_thinking_folded_into_body(fixtures_dir):
    src = ClaudeWebSource(drops_dir=str(fixtures_dir))
    _s, observations, _errs = _split_entities(src)
    o = next(o for o in observations if o.obs_id == f"claude-web:{_m(2)}")
    # thinking + text blocks concatenated with a blank line, in content order.
    assert o.body == (
        "The user greets me. Plan a helpful reply.\n\nHello! How can I help?"
    )


def test_tool_use_block_becomes_separate_observation(fixtures_dir):
    src = ClaudeWebSource(drops_dir=str(fixtures_dir))
    _s, observations, _errs = _split_entities(src)
    o = next(
        o for o in observations if o.obs_id == f"claude-web:{_m(5)}:b0"
    )
    assert o.type == "tool_use"
    assert o.tool_name == "web_search"
    assert o.tool_use_id == "toolu_01ABC"
    assert o.parent_obs_id == f"claude-web:{_m(5)}"
    # ts from the block's start_timestamp, not the message created_at.
    assert o.ts == datetime(2026, 7, 1, 10, 5, 30, tzinfo=UTC)
    # body = json.dumps of the block input.
    assert json.loads(o.body) == {"query": "aggregator patterns"}


def test_tool_result_block_becomes_separate_observation(fixtures_dir):
    src = ClaudeWebSource(drops_dir=str(fixtures_dir))
    _s, observations, _errs = _split_entities(src)
    o = next(
        o for o in observations if o.obs_id == f"claude-web:{_m(6)}:b0"
    )
    assert o.type == "tool_result"
    assert o.tool_name is None
    assert o.tool_use_id == "toolu_01ABC"
    assert o.parent_obs_id == f"claude-web:{_m(6)}"
    assert "Top result: langfuse traces." in o.body
    # The carrying message keeps its own text-block body.
    msg = next(o for o in observations if o.obs_id == f"claude-web:{_m(6)}")
    assert msg.body == "Based on the search, Langfuse-style traces fit."


def test_text_fallback_when_content_missing(fixtures_dir):
    """Message 4 has no content[] at all — body comes from the top-level
    ``text`` field."""
    src = ClaudeWebSource(drops_dir=str(fixtures_dir))
    _s, observations, _errs = _split_entities(src)
    o = next(o for o in observations if o.obs_id == f"claude-web:{_m(4)}")
    assert o.body == "Search the web for aggregator patterns"
    assert o.type == "user"


def test_pure_tool_use_message_keeps_anchor_obs(fixtures_dir):
    """Message 5 is a pure tool_use (empty text, no text blocks). The message
    obs must still be emitted (empty body) so the block obs + the next
    message's parent chain resolve."""
    src = ClaudeWebSource(drops_dir=str(fixtures_dir))
    _s, observations, _errs = _split_entities(src)
    o = next(o for o in observations if o.obs_id == f"claude-web:{_m(5)}")
    assert o.body == ""
    assert o.type == "assistant"
    assert o.parent_obs_id == f"claude-web:{_m(4)}"


def test_observations_denormalise_root_session_id(fixtures_dir):
    src = ClaudeWebSource(drops_dir=str(fixtures_dir))
    _s, observations, _errs = _split_entities(src)
    conv1_obs = [o for o in observations if o.session_id == f"claude-web:{CONV1}"]
    # 6 messages + 1 tool_use block + 1 tool_result block.
    assert len(conv1_obs) == 8
    for o in conv1_obs:
        assert o.root_session_id == f"claude-web:{CONV1}"
        assert o.model is None
        assert o.input_tokens is None
        assert o.output_tokens is None


def test_shape_sniff_skips_chatgpt_file(tmp_path, fixtures_dir):
    """A ChatGPT-shaped conversations.json in the drops dir (first element
    has ``mapping``, not ``chat_messages``) is silently skipped — the
    parallel chatgpt source owns it."""
    drops = tmp_path / "drops"
    drops.mkdir()
    (drops / "conversations.json").write_text(
        json.dumps(
            [
                {
                    "id": "cgpt-conv-1",
                    "title": "a chatgpt chat",
                    "create_time": 1750000000.0,
                    "update_time": 1750000100.0,
                    "mapping": {
                        "node-1": {"id": "node-1", "message": None, "children": []}
                    },
                }
            ]
        )
    )
    src = ClaudeWebSource(drops_dir=str(drops))
    sessions, observations, errs = _split_entities(src)
    assert sessions == []
    assert observations == []
    assert errs == []


def test_zip_discovery_reads_member_without_extraction(tmp_path, fixtures_dir):
    drops = tmp_path / "drops"
    drops.mkdir()
    zip_path = drops / "data-2026-07-02-claude-export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(
            "conversations.json",
            (fixtures_dir / "conversations.json").read_text(encoding="utf-8"),
        )
        # Claude export zips also carry users.json / projects.json — ignored.
        zf.writestr("users.json", json.dumps([{"uuid": "u1"}]))
        zf.writestr("projects.json", json.dumps([]))
    src = ClaudeWebSource(drops_dir=str(drops))
    sessions, observations, errs = _split_entities(src)
    assert errs == []
    assert {s.session_id for s in sessions} == {
        f"claude-web:{CONV1}",
        f"claude-web:{CONV2}",
    }
    s = next(s for s in sessions if s.session_id == f"claude-web:{CONV1}")
    assert s.jsonl_path == f"{zip_path}!conversations.json"
    assert any(o.obs_id == f"claude-web:{_m(5)}:b0" for o in observations)


def test_since_filter_on_updated_at(fixtures_dir):
    src = ClaudeWebSource(drops_dir=str(fixtures_dir))
    cutoff = datetime(2026, 6, 15, tzinfo=UTC)
    sessions, observations, _errs = _split_entities(src, since=cutoff)
    # conv2 (updated 2026-06-01) filtered out; conv1 (2026-07-01) kept.
    assert {s.session_id for s in sessions} == {f"claude-web:{CONV1}"}
    assert all(o.session_id == f"claude-web:{CONV1}" for o in observations)


def test_malformed_conversation_error_sink_continues(tmp_path):
    drops = tmp_path / "drops"
    drops.mkdir()
    good = {
        "uuid": "cccc1111-2222-4333-8444-555566667777",
        "created_at": "2026-07-10T08:00:00.000Z",
        "updated_at": "2026-07-10T08:01:00.000Z",
        "chat_messages": [
            {
                "uuid": "33333333-aaaa-4bbb-8ccc-000000000001",
                "text": "hi",
                "sender": "human",
                "created_at": "2026-07-10T08:00:01.000Z",
                "parent_message_uuid": "00000000-0000-4000-8000-000000000000",
            }
        ],
    }
    bad = {
        # chat_messages present (so shape sniff would pass were it first) but
        # no uuid → cannot mint a stable session id → error sink.
        "created_at": "2026-07-10T09:00:00.000Z",
        "updated_at": "2026-07-10T09:01:00.000Z",
        "chat_messages": [],
    }
    (drops / "conversations.json").write_text(json.dumps([good, bad]))
    src = ClaudeWebSource(drops_dir=str(drops))
    sessions, observations, errs = _split_entities(src)
    assert len(errs) == 1
    assert "uuid" in errs[0]
    # The good conversation still lands.
    assert [s.session_id for s in sessions] == [
        "claude-web:cccc1111-2222-4333-8444-555566667777"
    ]
    assert len(observations) == 1


def test_drops_dir_env_override(fixtures_dir, monkeypatch):
    monkeypatch.setenv("AGGREGATOR_DROPS_DIR", str(fixtures_dir))
    src = ClaudeWebSource()
    sessions, _obs, _errs = _split_entities(src)
    assert len(sessions) == 2


def test_missing_drops_dir_yields_nothing():
    src = ClaudeWebSource(drops_dir="/nonexistent/drops-dir")
    sessions, observations, errs = _split_entities(src)
    assert sessions == []
    assert observations == []
    assert errs == []


def test_ingest_counts_via_result(fixtures_dir):
    src = ClaudeWebSource(drops_dir=str(fixtures_dir))
    result = src.ingest(since=None)
    # 2 sessions + (8 conv1 obs + 2 conv2 obs) = 12.
    assert result.added == 12
    assert result.errors == []


def test_record_shape_documents_fields():
    src = ClaudeWebSource(drops_dir="/tmp")
    shape = src.record_shape()
    for key in ("session_id", "origin", "first_ts", "last_ts", "obs_type"):
        assert key in shape, f"missing key {key!r} in record_shape"


# -- duplicate exports (newest file wins) -----------------------------------


def test_duplicate_conversation_across_export_files_newest_file_wins(tmp_path):
    """The same conversation uuid in TWO export files must emit ONE
    SessionRow, from the NEWEST file (file mtime). Without the per-run claim,
    both copies emit and the stored row flip-flops on jsonl_path every ingest
    tick — the substack churn's sibling, one export file away from live."""
    drops = tmp_path / "drops"
    drops.mkdir()
    dup = "dddd1111-2222-4333-8444-555566667777"
    conv_old = {
        "uuid": dup,
        "created_at": "2026-07-01T10:00:00.000Z",
        "updated_at": "2026-07-01T10:06:00.000Z",
        "chat_messages": [],
    }
    conv_new = dict(conv_old, updated_at="2026-07-02T09:00:00.000Z")
    # The OLDER file sorts first by name, so name-order emission would keep
    # the stale copy — the mtime has to be what decides.
    older = drops / "conversations-a.json"
    newer = drops / "conversations-b.json"
    older.write_text(json.dumps([conv_old]))
    newer.write_text(json.dumps([conv_new]))
    old_ts = datetime(2026, 7, 1, tzinfo=UTC).timestamp()
    new_ts = datetime(2026, 8, 1, tzinfo=UTC).timestamp()
    os.utime(older, (old_ts, old_ts))
    os.utime(newer, (new_ts, new_ts))

    src = ClaudeWebSource(drops_dir=str(drops))
    sessions, _obs, errs = _split_entities(src)
    assert errs == []
    assert [s.session_id for s in sessions] == [f"claude-web:{dup}"]
    assert sessions[0].jsonl_path == str(newer)
    assert sessions[0].last_ts == datetime(2026, 7, 2, 9, 0, tzinfo=UTC)
