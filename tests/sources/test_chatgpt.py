"""Tests for the ChatGPT export source (Chunk 2, chat-exports plan 2026-08-02).

Ontology: ``conversations.json`` (array of conversation objects, each with a
``mapping`` node DAG) → ``SessionRow`` (origin='chatgpt') + ``ObservationRow``.

Key mapping rules under test:

* session_id = ``chatgpt:<conversation_id or id>`` (prefix for collision
  safety against Claude Code session UUIDs).
* first_ts/last_ts from ``create_time``/``update_time`` (unix epoch float,
  UTC); missing update_time falls back to create_time.
* mapping nodes with ``message: null`` (synthetic root) are skipped and
  their children re-parented to None.
* ALL branches kept (regenerated siblings) — no current_node filtering.
* ``author.role`` preserved RAW (user/assistant/system/tool) — never
  bucketed to 'other'.
* Empty-body non-tool nodes (regeneration stubs with empty parts) skipped;
  empty tool nodes KEPT.
* Input discovery: drops dir accepts bare ``conversations.json`` /
  ``conversations-*.json`` shards AND ``*.zip`` wrapping either.

Fixture: ``tests/fixtures/chatgpt/conversations.json``. Zip fixtures are
built in-test (no committed binaries).
"""
from __future__ import annotations

import json
import os
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from aggregator.sources.base import ObservationRow, SessionRow
from aggregator.sources.chatgpt import ChatGPTSource

# Epoch anchors used by the fixture (verified by hand):
# 1750000000 → 2025-06-15T15:06:40Z; 1750003600 → +1h; 1750010000 → +2h46m40s.
CONV1_CREATE = datetime(2025, 6, 15, 15, 6, 40, tzinfo=UTC)
CONV1_UPDATE = datetime(2025, 6, 15, 16, 6, 40, tzinfo=UTC)
CONV2_CREATE = datetime.fromtimestamp(1750010000.0, tz=UTC)


@pytest.fixture
def fixtures_dir(repo_root):
    return Path(repo_root) / "tests" / "fixtures" / "chatgpt"


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


# -- session row shape -----------------------------------------------------


def test_session_row_shape_prefix_origin_ts(fixtures_dir):
    src = ChatGPTSource(drops_dir=str(fixtures_dir))
    sessions, _obs, errs = _split_entities(src)
    assert errs == []
    s = next(s for s in sessions if s.session_id == "chatgpt:conv-uuid-1")
    assert s.origin == "chatgpt"
    assert s.kind == "session"
    assert s.root_session_id == "chatgpt:conv-uuid-1"
    assert s.parent_session_id is None
    assert s.agent_id is None
    assert s.agent_type is None
    assert s.spawned_by_tool_use_id is None
    assert s.cwd is None
    assert s.git_branch is None
    # Epoch float seconds → aware UTC datetimes.
    assert s.first_ts == CONV1_CREATE
    assert s.last_ts == CONV1_UPDATE
    assert s.jsonl_path.endswith("conversations.json")


def test_session_id_prefers_conversation_id_over_id(fixtures_dir):
    """conv-1 carries both conversation_id and a decoy ``id`` field."""
    src = ChatGPTSource(drops_dir=str(fixtures_dir))
    sessions, _obs, _errs = _split_entities(src)
    ids = {s.session_id for s in sessions}
    assert "chatgpt:conv-uuid-1" in ids
    assert "chatgpt:legacy-id-ignored" not in ids


def test_session_id_falls_back_to_id_and_create_time(fixtures_dir):
    """conv-2 has only ``id``, no update_time → last_ts falls back to
    create_time."""
    src = ChatGPTSource(drops_dir=str(fixtures_dir))
    sessions, _obs, _errs = _split_entities(src)
    s = next(s for s in sessions if s.session_id == "chatgpt:conv-uuid-2")
    assert s.first_ts == CONV2_CREATE
    assert s.last_ts == CONV2_CREATE


def test_missing_mapping_emits_session_row_only(fixtures_dir):
    src = ChatGPTSource(drops_dir=str(fixtures_dir))
    _sessions, observations, errs = _split_entities(src)
    assert errs == []
    assert [o for o in observations if o.session_id == "chatgpt:conv-uuid-2"] == []


# -- observation DAG -------------------------------------------------------


def test_synthetic_root_skipped_and_children_reparented(fixtures_dir):
    src = ChatGPTSource(drops_dir=str(fixtures_dir))
    _s, observations, _e = _split_entities(src)
    assert all(o.obs_id != "chatgpt:root-1" for o in observations)
    u1 = next(o for o in observations if o.obs_id == "chatgpt:node-u1")
    assert u1.parent_obs_id is None


def test_parent_chain_preserved_with_prefix(fixtures_dir):
    src = ChatGPTSource(drops_dir=str(fixtures_dir))
    _s, observations, _e = _split_entities(src)
    a1 = next(o for o in observations if o.obs_id == "chatgpt:node-a1")
    assert a1.parent_obs_id == "chatgpt:node-u1"
    t1 = next(o for o in observations if o.obs_id == "chatgpt:node-t1")
    assert t1.parent_obs_id == "chatgpt:node-a2"


def test_regenerated_sibling_branches_both_present(fixtures_dir):
    """No current_node filtering — both children of node-u1 are emitted."""
    src = ChatGPTSource(drops_dir=str(fixtures_dir))
    _s, observations, _e = _split_entities(src)
    siblings = [o for o in observations if o.parent_obs_id == "chatgpt:node-u1"]
    assert {o.obs_id for o in siblings} == {"chatgpt:node-a1", "chatgpt:node-a2"}


def test_root_session_id_denormalised_on_observations(fixtures_dir):
    src = ChatGPTSource(drops_dir=str(fixtures_dir))
    _s, observations, _e = _split_entities(src)
    conv1_obs = [o for o in observations if o.session_id == "chatgpt:conv-uuid-1"]
    assert conv1_obs
    assert all(o.root_session_id == "chatgpt:conv-uuid-1" for o in conv1_obs)


# -- types / bodies / tools ------------------------------------------------


def test_author_role_preserved_raw(fixtures_dir):
    src = ChatGPTSource(drops_dir=str(fixtures_dir))
    _s, observations, _e = _split_entities(src)
    by_id = {o.obs_id: o for o in observations}
    assert by_id["chatgpt:node-u1"].type == "user"
    assert by_id["chatgpt:node-a1"].type == "assistant"
    assert by_id["chatgpt:node-t1"].type == "tool"
    assert all(o.type != "other" for o in observations)


def test_tool_node_gets_tool_name_from_author_name(fixtures_dir):
    src = ChatGPTSource(drops_dir=str(fixtures_dir))
    _s, observations, _e = _split_entities(src)
    t1 = next(o for o in observations if o.obs_id == "chatgpt:node-t1")
    assert t1.tool_name == "python"
    assert t1.body == "24"
    # Non-tool nodes never get a tool_name.
    a1 = next(o for o in observations if o.obs_id == "chatgpt:node-a1")
    assert a1.tool_name is None


def test_model_slug_extracted(fixtures_dir):
    src = ChatGPTSource(drops_dir=str(fixtures_dir))
    _s, observations, _e = _split_entities(src)
    a1 = next(o for o in observations if o.obs_id == "chatgpt:node-a1")
    assert a1.model == "gpt-4o"
    u1 = next(o for o in observations if o.obs_id == "chatgpt:node-u1")
    assert u1.model is None


def test_multimodal_parts_flattened(fixtures_dir):
    """Dict parts without ``text`` become ``[non-text part: <content_type>]``;
    string parts kept. Null message create_time falls back to conversation
    create_time."""
    src = ChatGPTSource(drops_dir=str(fixtures_dir))
    _s, observations, _e = _split_entities(src)
    m1 = next(o for o in observations if o.obs_id == "chatgpt:node-m1")
    assert "[non-text part: image_asset_pointer]" in m1.body
    assert "Look at this image" in m1.body
    assert m1.ts == CONV1_CREATE


def test_empty_parts_node_skipped_thoughts_node_not_crashing(fixtures_dir):
    """node-e1 (empty parts, assistant) and node-th1 (thoughts content_type:
    no parts, no text field → empty body, non-tool) must be silently skipped
    — never crash, never emit empty non-tool bodies."""
    src = ChatGPTSource(drops_dir=str(fixtures_dir))
    _s, observations, errs = _split_entities(src)
    assert errs == []
    ids = {o.obs_id for o in observations}
    assert "chatgpt:node-e1" not in ids
    assert "chatgpt:node-th1" not in ids


# -- input discovery -------------------------------------------------------


def test_zip_drop_read_without_extraction(fixtures_dir, tmp_path):
    drops = tmp_path / "drops"
    drops.mkdir()
    zp = drops / "chatgpt-export-2026.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.write(fixtures_dir / "conversations.json", arcname="conversations.json")
        zf.writestr("chat.html", "<html></html>")  # must be ignored
        zf.writestr("user.json", "{}")  # must be ignored
    src = ChatGPTSource(drops_dir=str(drops))
    sessions, observations, errs = _split_entities(src)
    assert errs == []
    assert {s.session_id for s in sessions} == {
        "chatgpt:conv-uuid-1",
        "chatgpt:conv-uuid-2",
    }
    s1 = next(s for s in sessions if s.session_id == "chatgpt:conv-uuid-1")
    assert s1.jsonl_path == f"{zp}!conversations.json"
    assert any(o.obs_id == "chatgpt:node-u1" for o in observations)


def test_sharded_conversations_glob(fixtures_dir, tmp_path):
    """2026 exports may shard into conversations-*.json — accepted bare in
    the drops dir."""
    shard = tmp_path / "conversations-001.json"
    shard.write_text((fixtures_dir / "conversations.json").read_text())
    src = ChatGPTSource(drops_dir=str(tmp_path))
    sessions, _obs, errs = _split_entities(src)
    assert errs == []
    assert any(s.session_id == "chatgpt:conv-uuid-1" for s in sessions)


def test_env_var_overrides_drops_dir(fixtures_dir, monkeypatch):
    monkeypatch.setenv("AGGREGATOR_DROPS_DIR", str(fixtures_dir))
    src = ChatGPTSource()
    sessions, _obs, _errs = _split_entities(src)
    assert any(s.session_id == "chatgpt:conv-uuid-1" for s in sessions)


def test_missing_drops_dir_yields_nothing():
    src = ChatGPTSource(drops_dir="/nonexistent/drops")
    sessions, observations, errs = _split_entities(src)
    assert sessions == []
    assert observations == []
    assert errs == []


# -- since filter ----------------------------------------------------------


def test_since_filters_conversations_by_update_time(fixtures_dir):
    """Cutoff between conv-1's update_time and conv-2's create_time keeps
    only conv-2."""
    src = ChatGPTSource(drops_dir=str(fixtures_dir))
    cutoff = datetime.fromtimestamp(1750005000.0, tz=UTC)
    sessions, observations, _errs = _split_entities(src, since=cutoff)
    assert [s.session_id for s in sessions] == ["chatgpt:conv-uuid-2"]
    assert [o for o in observations if o.session_id == "chatgpt:conv-uuid-1"] == []


# -- robustness ------------------------------------------------------------


def test_malformed_conversation_recorded_and_skipped(tmp_path):
    good = {
        "conversation_id": "conv-good",
        "create_time": 1750000000.0,
        "update_time": 1750000100.0,
        "mapping": {},
    }
    (tmp_path / "conversations.json").write_text(
        json.dumps([good, "not an object", {"title": "no id at all", "create_time": 1}])
    )
    src = ChatGPTSource(drops_dir=str(tmp_path))
    sessions, _obs, errors = _split_entities(src)
    assert [s.session_id for s in sessions] == ["chatgpt:conv-good"]
    assert len(errors) == 2


def test_unparseable_file_recorded_not_crashing(tmp_path):
    (tmp_path / "conversations.json").write_text("{this is not json")
    src = ChatGPTSource(drops_dir=str(tmp_path))
    sessions, observations, errors = _split_entities(src)
    assert sessions == []
    assert observations == []
    assert errors


# -- protocol --------------------------------------------------------------


def test_ingest_returns_counts(fixtures_dir):
    src = ChatGPTSource(drops_dir=str(fixtures_dir))
    result = src.ingest(since=None)
    # 2 sessions + 6 observations from conv-1 (u1, a1, a2, t1, m1; e1/th1
    # skipped) — count re-derived here to pin the emission set.
    sessions, observations, _errs = _split_entities(src)
    assert result.added == len(sessions) + len(observations)
    assert len(sessions) == 2
    assert len(observations) == 5
    assert result.errors == []


def test_record_shape_documents_fields():
    src = ChatGPTSource(drops_dir="/tmp")
    shape = src.record_shape()
    for key in ("session_id", "root_session_id", "kind", "origin", "first_ts", "last_ts", "obs_type"):
        assert key in shape, f"missing key {key!r} in record_shape"


# -- duplicate exports (newest file wins) -----------------------------------


def test_duplicate_conversation_across_export_files_newest_file_wins(tmp_path):
    """The same conversation_id in TWO export files must emit ONE SessionRow,
    from the NEWEST file (file mtime). Same union-across-files mechanism as
    substack and claude-web: without the per-run claim, both copies emit and
    the stored row flip-flops on jsonl_path every ingest tick."""
    conv_old = {
        "conversation_id": "dup-conv",
        "create_time": 1750000000.0,
        "update_time": 1750003600.0,
        "mapping": {},
    }
    conv_new = dict(conv_old, update_time=1750010000.0)
    # The OLDER file sorts first by name, so name-order emission would keep
    # the stale copy — the mtime has to be what decides.
    older = tmp_path / "conversations-a.json"
    newer = tmp_path / "conversations-b.json"
    older.write_text(json.dumps([conv_old]))
    newer.write_text(json.dumps([conv_new]))
    old_ts = datetime(2026, 7, 1, tzinfo=UTC).timestamp()
    new_ts = datetime(2026, 8, 1, tzinfo=UTC).timestamp()
    os.utime(older, (old_ts, old_ts))
    os.utime(newer, (new_ts, new_ts))

    src = ChatGPTSource(drops_dir=str(tmp_path))
    sessions, _obs, errors = _split_entities(src)
    assert errors == []
    assert [s.session_id for s in sessions] == ["chatgpt:dup-conv"]
    assert sessions[0].jsonl_path == str(newer)
    assert sessions[0].last_ts == datetime.fromtimestamp(1750010000.0, tz=UTC)
