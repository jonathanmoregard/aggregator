"""Tests for the shared chat-export discovery helper (Chunk 4).

``discover_export_files(kind)`` scans BOTH the drops dir
(``~/.local/share/aggregator/drops``, override ``AGGREGATOR_DROPS_DIR``) AND
``~/Downloads`` (override ``AGGREGATOR_DOWNLOADS_DIR`` — owner drops vendor
export zips straight into Downloads, no manual move).

Classification is by CONTENT, not filename:

* ``*.zip`` → member list via zipfile (central directory only); a member
  named ``conversations.json`` / ``conversations-*.json`` marks a chat
  export; the member's first array element disambiguates the vendor
  (``mapping`` key → chatgpt, ``chat_messages`` → claude-web).
* bare ``conversations.json`` / ``conversations-*.json`` files: same sniff.
* zips without a matching member (unrelated Downloads zips) are silently
  skipped; corrupt zips are skipped without crashing.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from aggregator.sources.exportdrops import ExportFile, discover_export_files

CHATGPT_CONVS = [
    {
        "conversation_id": "conv-1",
        "title": "a chatgpt chat",
        "create_time": 1750000000.0,
        "update_time": 1750003600.0,
        "mapping": {},
    }
]

CLAUDE_CONVS = [
    {
        "uuid": "aaaa1111-2222-4333-8444-555566667777",
        "created_at": "2026-07-01T10:00:00.000Z",
        "updated_at": "2026-07-01T10:06:00.000Z",
        "chat_messages": [],
    }
]


def _write_zip(path: Path, members: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return path


@pytest.fixture
def drops(tmp_path, monkeypatch) -> Path:
    d = tmp_path / "drops"
    d.mkdir()
    monkeypatch.setenv("AGGREGATOR_DROPS_DIR", str(d))
    return d


@pytest.fixture
def downloads(tmp_path, monkeypatch) -> Path:
    d = tmp_path / "downloads"
    d.mkdir()
    monkeypatch.setenv("AGGREGATOR_DOWNLOADS_DIR", str(d))
    return d


# -- zip classification -----------------------------------------------------


def test_chatgpt_zip_in_downloads_found_and_classified(drops, downloads):
    zp = _write_zip(
        downloads / "chatgpt-export.zip",
        {"conversations.json": json.dumps(CHATGPT_CONVS), "chat.html": "<html>"},
    )
    found = discover_export_files("chatgpt")
    assert [(f.path, f.member) for f in found] == [(zp, "conversations.json")]
    # The same zip is NOT claude-web's.
    assert discover_export_files("claude-web") == []


def test_claude_zip_in_downloads_classified(drops, downloads):
    zp = _write_zip(
        downloads / "data-2026-08-01-claude.zip",
        {
            "conversations.json": json.dumps(CLAUDE_CONVS),
            "users.json": "[]",
            "projects.json": "[]",
        },
    )
    found = discover_export_files("claude-web")
    assert [(f.path, f.member) for f in found] == [(zp, "conversations.json")]
    assert discover_export_files("chatgpt") == []


def test_unrelated_zip_silently_skipped(drops, downloads):
    _write_zip(downloads / "random-stuff.zip", {"foo.txt": "hello"})
    errors: list[str] = []
    assert discover_export_files("chatgpt", errors=errors) == []
    assert discover_export_files("claude-web", errors=errors) == []
    assert errors == []


def test_corrupt_zip_skipped_without_crash(drops, downloads):
    (downloads / "broken.zip").write_bytes(b"PK\x03\x04 not really a zip")
    assert discover_export_files("chatgpt") == []
    assert discover_export_files("claude-web") == []


def test_sharded_conversations_member_in_zip_matched(drops, downloads):
    zp = _write_zip(
        downloads / "chatgpt-2026-shard.zip",
        {"conversations-001.json": json.dumps(CHATGPT_CONVS)},
    )
    found = discover_export_files("chatgpt")
    assert [(f.path, f.member) for f in found] == [(zp, "conversations-001.json")]


# -- bare json classification -----------------------------------------------


def test_bare_json_classified_by_shape(drops, downloads):
    p = downloads / "conversations.json"
    p.write_text(json.dumps(CHATGPT_CONVS))
    found = discover_export_files("chatgpt")
    assert [(f.path, f.member) for f in found] == [(p, None)]
    assert discover_export_files("claude-web") == []


def test_bare_sharded_json_matched(drops, downloads):
    p = drops / "conversations-002.json"
    p.write_text(json.dumps(CLAUDE_CONVS))
    found = discover_export_files("claude-web")
    assert [(f.path, f.member) for f in found] == [(p, None)]


def test_non_conversations_filename_not_scanned(drops, downloads):
    """Candidate selection is by conversations*.json naming; a chat-shaped
    payload under another name is not picked up."""
    (downloads / "notes.json").write_text(json.dumps(CHATGPT_CONVS))
    assert discover_export_files("chatgpt") == []


def test_invalid_json_bare_file_goes_to_errors_sink(drops, downloads):
    (drops / "conversations.json").write_text("{this is not json")
    errors: list[str] = []
    assert discover_export_files("chatgpt", errors=errors) == []
    assert len(errors) == 1
    assert "conversations.json" in errors[0]


# -- dir merging + overrides ------------------------------------------------


def test_both_dirs_merged(drops, downloads):
    zp = _write_zip(
        drops / "claude-export.zip", {"conversations.json": json.dumps(CLAUDE_CONVS)}
    )
    bare = downloads / "conversations.json"
    bare.write_text(json.dumps(CLAUDE_CONVS))
    found = discover_export_files("claude-web")
    assert {(f.path, f.member) for f in found} == {
        (zp, "conversations.json"),
        (bare, None),
    }


def test_env_overrides_point_discovery_at_both_dirs(drops, downloads):
    """No explicit dirs: AGGREGATOR_DROPS_DIR + AGGREGATOR_DOWNLOADS_DIR
    (set by the fixtures) drive the scan."""
    (drops / "conversations.json").write_text(json.dumps(CHATGPT_CONVS))
    found = discover_export_files("chatgpt")
    assert [f.path for f in found] == [drops / "conversations.json"]


def test_explicit_dirs_param_overrides_env(drops, downloads, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "conversations.json").write_text(json.dumps(CHATGPT_CONVS))
    (drops / "conversations.json").write_text(json.dumps(CHATGPT_CONVS))
    found = discover_export_files("chatgpt", dirs=[other])
    assert [f.path for f in found] == [other / "conversations.json"]


def test_duplicate_dirs_deduped(drops, downloads, tmp_path):
    p = drops / "conversations.json"
    p.write_text(json.dumps(CHATGPT_CONVS))
    found = discover_export_files("chatgpt", dirs=[drops, drops])
    assert [f.path for f in found] == [p]


def test_missing_dirs_yield_nothing(tmp_path):
    found = discover_export_files(
        "chatgpt", dirs=[tmp_path / "nope-a", tmp_path / "nope-b"]
    )
    assert found == []


# -- ExportFile surface ------------------------------------------------------


def test_export_file_label_and_read_bytes_for_zip_member(drops, downloads):
    zp = _write_zip(
        downloads / "chatgpt-export.zip",
        {"conversations.json": json.dumps(CHATGPT_CONVS)},
    )
    (f,) = discover_export_files("chatgpt")
    assert f.label == f"{zp}!conversations.json"
    assert json.loads(f.read_bytes()) == CHATGPT_CONVS


def test_export_file_label_and_read_bytes_for_bare_file(drops, downloads):
    p = drops / "conversations.json"
    p.write_text(json.dumps(CLAUDE_CONVS))
    (f,) = discover_export_files("claude-web")
    assert f.label == str(p)
    assert json.loads(f.read_bytes()) == CLAUDE_CONVS
    assert isinstance(f, ExportFile)


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        discover_export_files("nope-not-a-kind")


# -- substack (Chunk 5) -----------------------------------------------------


def test_substack_zip_in_downloads_classified(drops, downloads):
    zp = _write_zip(
        downloads / "substack-export.zip",
        {
            "posts.csv": "post_id,title\n",
            "email_list.honest.csv": "email\n",
            "posts/100.hello.html": "<h1>Hello</h1>",
            "posts/100.delivers.csv": "email,sent_at\n",
            "posts/100.opens.csv": "email,opened_at\n",
        },
    )
    found = discover_export_files("substack")
    assert [(f.path, f.member) for f in found] == [(zp, None)]
    # A substack zip is neither chatgpt nor claude-web.
    assert discover_export_files("chatgpt") == []
    assert discover_export_files("claude-web") == []


def test_substack_bare_files_not_discovered(drops, downloads):
    """Bare posts/ directory or bare HTML is chat-drops semantic and out
    of scope for v1 (substack ships zip only)."""
    posts = downloads / "posts"
    posts.mkdir()
    (posts / "100.hello.html").write_text("<h1>Hi</h1>")
    assert discover_export_files("substack") == []


def test_chat_zip_not_misclassified_as_substack(drops, downloads):
    _write_zip(
        downloads / "chatgpt-only.zip",
        {"conversations.json": json.dumps(CHATGPT_CONVS)},
    )
    assert discover_export_files("substack") == []
