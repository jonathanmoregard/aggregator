import json
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
    # simple.jsonl -> 2 sessions (sess-simple-001, sess-simple-002);
    # corrupt.jsonl -> 1 (sess-corrupt) with garbage lines skipped.
    simple = next(
        r for r in records if r.extra["session_id"] == "sess-simple-001"
    )
    assert simple.stable_id == "sessions:sess-simple-001"
    assert "refactor foo.py" in simple.body
    assert simple.extra["model"] == "claude-opus-4-7"
    # Real JSONLs carry usage, not cost; parser leaves cost_usd None.
    assert simple.extra["cost_usd"] is None
    assert simple.extra["project"] == "proj-alpha"
    assert any(tc["name"] == "Edit" for tc in simple.extra["top_tool_calls"])
    assert any(tc["name"] == "Bash" for tc in simple.extra["top_tool_calls"])
    # started/ended come from min/max timestamps across the session's lines.
    assert simple.created_at is not None
    assert simple.updated_at is not None
    assert simple.updated_at >= simple.created_at

    # Second session in the same file must yield its own record.
    second = next(
        r for r in records if r.extra["session_id"] == "sess-simple-002"
    )
    assert second.extra["project"] == "proj-beta"
    assert second.extra["model"] == "claude-sonnet-4-6"


def test_skips_files_modified_within_5min(tmp_path):
    live = tmp_path / "live.jsonl"
    live.write_text(
        json.dumps(
            {
                "type": "user",
                "sessionId": "sess-live",
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
    records = list(src._iter_records())
    assert records == []


def test_corrupt_line_skipped_not_aborted(fixtures_dir):
    src = SessionsSource(projects_root=str(fixtures_dir))
    result = src.ingest(since=None)
    # simple.jsonl -> 2 sessions, corrupt.jsonl -> 1 session (garbage
    # lines skipped internally, valid content still parsed).
    assert result.added >= 2
    # errors list must include at least one corrupt-line entry.
    assert any("corrupt" in e for e in result.errors)


def test_record_shape_documents_extra_fields():
    src = SessionsSource(projects_root="/tmp")
    shape = src.record_shape()
    assert "session_id" in shape
    assert "project" in shape
    assert "model" in shape


def test_iter_records_since_handles_naive_ended_without_typeerror(tmp_path):
    """Round-3 LOW: sessions log lines with a ``Z`` suffix parse into
    UTC-aware datetimes, but a fixture that omits the ``Z`` (or any tz)
    yields a naive ``ended``. Comparing naive < aware raises TypeError in
    Python 3, so an old/malformed log used to abort the whole ingest.

    Fix: normalise both sides to UTC-aware for the comparison (or skip the
    record when ``ended`` is naive — either is fine, but no TypeError).
    """
    from datetime import UTC, datetime

    jsonl = tmp_path / "naive.jsonl"
    # Note the missing 'Z' on the timestamp -> naive datetime after parse.
    jsonl.write_text(
        '{"type": "user", "sessionId": "sess-naive", "cwd": "/x", '
        '"timestamp": "2026-07-20T10:00:00", '
        '"message": {"role": "user", "content": "hi"}}\n'
    )
    old = time.time() - 24 * 60 * 60
    os.utime(jsonl, (old, old))

    src = SessionsSource(projects_root=str(tmp_path))
    aware_since = datetime(2026, 7, 15, tzinfo=UTC)
    # Pre-fix: TypeError raised here. Post-fix: iteration completes
    # (either yielding the record or skipping it, both are acceptable).
    records = list(src.iter_records(since=aware_since))
    # No assertion on count — the important thing is "did not TypeError".
    assert isinstance(records, list)
