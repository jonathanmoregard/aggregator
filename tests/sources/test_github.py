"""Tests for aggregator.sources.github (M1b).

Covers:
- Scope parsing + write-scope detection (unit).
- Ingest refuses/allows write-capable tokens per env override (spec §Security).
- Subprocess-mocked default scope fetcher for both read-only and writeable headers.
- Record shape for PR and issue.
- record_shape() advertises DSL-facing filter fields.
- gh-api error path: ingest records the error and does not crash.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from aggregator.sources.github import (
    GitHubSource,
    WriteCapableTokenError,
    _default_scope_fetcher,
    _has_write_scope,
    _parse_scopes,
)

FIX = Path(__file__).parent.parent / "fixtures" / "github"


# -- pure helpers -----------------------------------------------------------


def test_parse_scopes_extracts_from_header_line():
    line = (FIX / "scopes_readonly.txt").read_text().strip()
    scopes = _parse_scopes(line)
    assert "public_repo" in scopes
    assert "repo:status" in scopes
    assert "repo" not in scopes  # readonly variant only has repo:status


def test_parse_scopes_handles_bare_csv_without_header_prefix():
    # `_parse_scopes` accepts a raw CSV as well (defensive).
    scopes = _parse_scopes("public_repo, read:org")
    assert scopes == ["public_repo", "read:org"]


def test_write_scope_detected():
    write = _parse_scopes((FIX / "scopes_writeable.txt").read_text().strip())
    read = _parse_scopes((FIX / "scopes_readonly.txt").read_text().strip())
    assert _has_write_scope(write) is True
    assert _has_write_scope(read) is False


# -- default subprocess-backed scope fetcher --------------------------------


def _fake_completed_process(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["gh", "api", "-i", "/rate_limit"],
        returncode=returncode,
        stdout=stdout,
        stderr="",
    )


def test_default_scope_fetcher_parses_readonly_header(monkeypatch):
    header_line = (FIX / "scopes_readonly.txt").read_text().strip()
    fake_response = (
        f"HTTP/2 200\n"
        f"Content-Type: application/json\n"
        f"{header_line}\n"
        f"X-RateLimit-Remaining: 4999\n\n"
        f'{{"rate": {{}}}}'
    )
    with patch(
        "aggregator.sources.github.subprocess.run",
        return_value=_fake_completed_process(fake_response),
    ):
        scopes = _default_scope_fetcher()
    assert "public_repo" in scopes
    assert _has_write_scope(scopes) is False


def test_default_scope_fetcher_parses_writeable_header(monkeypatch):
    header_line = (FIX / "scopes_writeable.txt").read_text().strip()
    fake_response = f"HTTP/2 200\n{header_line}\n\n{{}}"
    with patch(
        "aggregator.sources.github.subprocess.run",
        return_value=_fake_completed_process(fake_response),
    ):
        scopes = _default_scope_fetcher()
    assert "repo" in scopes
    assert _has_write_scope(scopes) is True


def test_default_scope_fetcher_returns_empty_on_gh_missing():
    with patch(
        "aggregator.sources.github.subprocess.run",
        side_effect=FileNotFoundError("gh not installed"),
    ):
        scopes = _default_scope_fetcher()
    assert scopes == []


def test_default_scope_fetcher_returns_empty_on_gh_failure():
    with patch(
        "aggregator.sources.github.subprocess.run",
        side_effect=subprocess.CalledProcessError(returncode=1, cmd=["gh"]),
    ):
        scopes = _default_scope_fetcher()
    assert scopes == []


# -- ingest scope enforcement ----------------------------------------------


def test_ingest_refuses_write_scope_without_override(monkeypatch):
    monkeypatch.delenv("AGGREGATOR_ALLOW_WRITE_TOKEN", raising=False)
    src = GitHubSource(_scope_fetcher=lambda: ["repo", "admin:repo_hook"])
    with pytest.raises(WriteCapableTokenError):
        src.ingest(since=None)


def test_ingest_allows_write_scope_with_override(monkeypatch):
    monkeypatch.setenv("AGGREGATOR_ALLOW_WRITE_TOKEN", "1")
    src = GitHubSource(
        _scope_fetcher=lambda: ["repo"],
        _api_fetcher=lambda path: [],
    )
    # should not raise
    result = src.ingest(since=None)
    assert result.errors == []


def test_ingest_allows_readonly_scope_without_override(monkeypatch):
    monkeypatch.delenv("AGGREGATOR_ALLOW_WRITE_TOKEN", raising=False)
    src = GitHubSource(
        _scope_fetcher=lambda: ["public_repo", "repo:status", "read:org"],
        _api_fetcher=lambda path: [],
    )
    result = src.ingest(since=None)
    assert result.errors == []


def test_ingest_error_message_is_actionable(monkeypatch):
    monkeypatch.delenv("AGGREGATOR_ALLOW_WRITE_TOKEN", raising=False)
    src = GitHubSource(_scope_fetcher=lambda: ["repo"])
    with pytest.raises(WriteCapableTokenError) as excinfo:
        src.ingest(since=None)
    msg = str(excinfo.value)
    assert "AGGREGATOR_ALLOW_WRITE_TOKEN" in msg
    assert "repo" in msg


# -- ingest happy path counts records --------------------------------------


def test_ingest_counts_records_from_all_endpoints(monkeypatch):
    monkeypatch.delenv("AGGREGATOR_ALLOW_WRITE_TOKEN", raising=False)
    pr = json.loads((FIX / "pr_open_passing.json").read_text())
    issue = json.loads((FIX / "issue_assigned.json").read_text())

    def fake_fetch(path: str) -> list[dict]:
        if "is:pr" in path:
            return [pr]
        if "is:issue" in path:
            return [issue]
        return []

    src = GitHubSource(
        _scope_fetcher=lambda: ["public_repo"],
        _api_fetcher=fake_fetch,
    )
    result = src.ingest(since=None)
    # 2 pr queries (authored + review-requested) + 2 issue queries (authored + assigned) = 4
    assert result.added == 4
    assert result.errors == []


def test_ingest_records_errors_without_crashing(monkeypatch):
    monkeypatch.delenv("AGGREGATOR_ALLOW_WRITE_TOKEN", raising=False)

    def boom(path: str) -> list[dict]:
        raise RuntimeError("gh api transport failure")

    src = GitHubSource(
        _scope_fetcher=lambda: ["public_repo"],
        _api_fetcher=boom,
    )
    result = src.ingest(since=None)
    assert result.errors
    assert "transport failure" in result.errors[0]


# -- record shape ----------------------------------------------------------


def test_pr_to_record_shape():
    src = GitHubSource(_scope_fetcher=lambda: ["public_repo"])
    pr = json.loads((FIX / "pr_open_passing.json").read_text())
    r = src._pr_to_record(pr)
    assert r.stable_id == "github:acme/api:42"
    assert r.source == "github"
    assert "rate limiter" in r.subject.lower()
    assert r.extra["state"] == "open"
    assert r.extra["mergeable"] is True
    assert r.extra["checks"] == "pass"
    assert "acme/api" in r.tags


def test_pr_closed_failing_record_shape():
    src = GitHubSource(_scope_fetcher=lambda: ["public_repo"])
    pr = json.loads((FIX / "pr_closed_failing.json").read_text())
    r = src._pr_to_record(pr)
    assert r.stable_id == "github:acme/api:41"
    assert r.extra["state"] == "closed"
    assert r.extra["mergeable"] is False
    assert r.extra["checks"] == "fail"
    assert r.extra["author"] == "jonathan-more"
    assert r.extra["url"].endswith("/pull/41")


def test_issue_to_record_shape():
    src = GitHubSource(_scope_fetcher=lambda: ["public_repo"])
    issue = json.loads((FIX / "issue_assigned.json").read_text())
    r = src._issue_to_record(issue, kind="assigned")
    assert r.stable_id == "github:acme/api:7"
    assert r.extra["state"] == "open"
    assert "assigned" in r.tags
    assert "jonathan-more" in r.extra["assignees"]


def test_record_body_excerpt_is_bounded():
    src = GitHubSource(_scope_fetcher=lambda: ["public_repo"])
    big = "x" * 2000
    pr = {
        "base": {"repo": {"full_name": "acme/api"}},
        "number": 1,
        "title": "big",
        "state": "open",
        "body": big,
    }
    r = src._pr_to_record(pr)
    assert len(r.extra["body_excerpt"]) == 500


def test_pr_missing_repo_falls_back_gracefully():
    """gh api occasionally returns malformed rows; must not crash."""
    src = GitHubSource(_scope_fetcher=lambda: ["public_repo"])
    r = src._pr_to_record({"number": 5, "title": "x", "state": "open"})
    assert r.stable_id == "github:unknown/unknown:5"


def test_record_shape_documents_filters():
    src = GitHubSource(_scope_fetcher=lambda: ["public_repo"])
    shape = src.record_shape()
    assert "state" in shape
    assert "mergeable" in shape
    assert "author" in shape
    assert "checks" in shape
