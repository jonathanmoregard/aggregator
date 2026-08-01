import os
import time
from pathlib import Path

import pytest

from aggregator.sources.sessions import SessionsSource


@pytest.fixture
def fixtures_dir(repo_root):
    """Return the on-disk fixtures dir, back-dated past the 5-min live window.

    Fixtures freshly checked out from git carry current mtime and would be
    skipped by the live-session heuristic. We back-date them here so the
    parse/ingest tests see them as normal historical files.
    """
    d = Path(repo_root) / "tests" / "fixtures" / "sessions"
    old = time.time() - 24 * 60 * 60
    for p in d.glob("*.jsonl"):
        os.utime(p, (old, old))
    return d


def test_parse_simple_session(fixtures_dir):
    src = SessionsSource(projects_root=str(fixtures_dir))
    records = list(src._iter_records())
    # simple.jsonl -> 1 session, corrupt.jsonl -> 1 (partial, unclosed) session
    simple = next(r for r in records if r.extra["session_id"] == "sess-simple-001")
    assert simple.stable_id == "sessions:sess-simple-001"
    assert "refactor foo.py" in simple.body
    assert simple.extra["model"] == "claude-opus-4-7"
    assert simple.extra["cost_usd"] == 0.42
    assert simple.extra["project"] == "proj-alpha"
    assert any(tc["name"] == "Edit" for tc in simple.extra["top_tool_calls"])


def test_skips_files_modified_within_5min(tmp_path):
    live = tmp_path / "live.jsonl"
    live.write_text(
        '{"type": "session_start", "session_id": "sess-live", "cwd": "/x", '
        '"started_at": "2026-07-27T10:00:00Z", "model": "m"}\n'
    )
    now = time.time()
    os.utime(live, (now, now))
    src = SessionsSource(projects_root=str(tmp_path))
    records = list(src._iter_records())
    assert records == []


def test_corrupt_line_skipped_not_aborted(fixtures_dir):
    src = SessionsSource(projects_root=str(fixtures_dir))
    result = src.ingest(since=None)
    # simple.jsonl + corrupt.jsonl both counted; corrupt line skipped internally
    assert result.added >= 2
    # errors list may include the corrupt-line path
    assert any("corrupt" in e for e in result.errors)


def test_record_shape_documents_extra_fields():
    src = SessionsSource(projects_root="/tmp")
    shape = src.record_shape()
    assert "session_id" in shape
    assert "project" in shape
    assert "model" in shape
