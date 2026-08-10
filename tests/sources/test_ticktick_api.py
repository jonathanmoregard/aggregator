from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from urllib.error import HTTPError, URLError

import pytest

from aggregator.sources import ticktick_api, ticktick_csv

TOKEN = "sup3r-s3cret-token"


class _FakeResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """No test in this file may reach the real TickTick API.

    The token is a write-scoped credential and the endpoint is the user's live
    task list; a test that escaped the mock could mutate real data. ``_open`` is
    this module's single network seam, so patching it blocks only this module —
    patching ``urllib.request.urlopen`` would have blocked the whole process for
    the duration of every test in the file. Every test that wants a response
    patches ``_open`` again, which wins over this.
    """

    def _forbidden(*args, **kwargs):
        raise AssertionError("test attempted a real network call")

    monkeypatch.setattr(ticktick_api, "_open", _forbidden)


# --- the write guard ------------------------------------------------------


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE", "get", ""])
def test_request_refuses_non_get(method):
    with pytest.raises(ticktick_api.WriteAttemptError):
        ticktick_api._request(method, "https://api.ticktick.com/open/v1/task", token=TOKEN)


def test_request_refuses_non_get_before_any_network_call():
    """The guard is structural: the refusal happens before the network seam.

    The autouse ``_no_network`` fixture turns any real call into an
    AssertionError, so a WriteAttemptError here proves nothing was sent.
    """
    with pytest.raises(ticktick_api.WriteAttemptError):
        ticktick_api._request("POST", "https://api.ticktick.com/open/v1/task", token=TOKEN)


def test_request_refuses_plaintext_url():
    """A bearer token must never go out over http:// (or file://, or gopher://)."""
    with pytest.raises(ValueError, match="https"):
        ticktick_api._request("GET", "http://api.ticktick.com/open/v1/project", token=TOKEN)


def test_get_sends_bearer_and_get_method(monkeypatch):
    seen = {}

    def fake_open(req, timeout=None):
        seen["method"] = req.get_method()
        seen["auth"] = req.get_header("Authorization")
        seen["url"] = req.full_url
        return _FakeResponse({"ok": True})

    monkeypatch.setattr(ticktick_api, "_open", fake_open)
    assert ticktick_api._request("GET", "https://x/y", token="tok") == {"ok": True}
    assert seen["method"] == "GET"
    assert seen["auth"] == "Bearer tok"
    assert seen["url"] == "https://x/y"


# --- the token must not survive a redirect (review M1) --------------------


def _captured_request(monkeypatch):
    box = {}

    def fake_open(req, timeout=None):
        box["req"] = req
        return _FakeResponse({"ok": True})

    monkeypatch.setattr(ticktick_api, "_open", fake_open)
    ticktick_api._request("GET", "https://api.ticktick.com/open/v1/project", token=TOKEN)
    return box["req"]


def test_authorization_is_an_unredirected_header(monkeypatch):
    """urllib copies req.headers onto a redirect; unredirected_hdrs it never does."""
    req = _captured_request(monkeypatch)
    assert "Authorization" in req.unredirected_hdrs
    assert not any(k.lower() == "authorization" for k in req.headers)


def test_a_redirect_request_does_not_carry_the_bearer_token(monkeypatch):
    """The measured defect: a 302 used to hand this write-scoped token to the target."""
    req = _captured_request(monkeypatch)
    redirected = ticktick_api._HttpsOnlyRedirectHandler().redirect_request(
        req, None, 302, "Found", {}, "https://elsewhere.example/target"
    )
    assert not any(k.lower() == "authorization" for k in redirected.headers)
    assert TOKEN not in json.dumps(dict(redirected.headers))


def test_a_redirect_to_plaintext_is_refused():
    """https -> http is a scheme downgrade urllib permits by default. We do not."""
    handler = ticktick_api._HttpsOnlyRedirectHandler()
    req = ticktick_api.request.Request("https://api.ticktick.com/open/v1/project")
    with pytest.raises(URLError, match="non-https"):
        handler.redirect_request(req, None, 302, "Found", {}, "http://127.0.0.1:8080/target")


def test_the_module_opener_replaces_the_default_redirect_handler():
    """A guarantee worth asserting structurally, not just on the happy path."""
    handlers = ticktick_api._OPENER.handlers
    assert any(isinstance(h, ticktick_api._HttpsOnlyRedirectHandler) for h in handlers)
    assert all(
        isinstance(h, ticktick_api._HttpsOnlyRedirectHandler)
        for h in handlers
        if isinstance(h, ticktick_api.request.HTTPRedirectHandler)
    )


# --- project walk ---------------------------------------------------------


def test_fetch_open_tasks_walks_projects(monkeypatch):
    calls = []

    def fake_request(method, url, token, timeout=30):
        calls.append(url)
        if url.endswith("/project"):
            return [{"id": "p1", "name": "Work"}, {"id": "p2", "name": "Home"}]
        if url.endswith("/project/p1/data"):
            return {"tasks": [{"id": "t1", "title": "Ship it", "projectId": "p1"}]}
        return {"tasks": [{"id": "t2", "title": "Dishes", "projectId": "p2"}]}

    monkeypatch.setattr(ticktick_api, "_request", fake_request)
    tasks = ticktick_api.fetch_open_tasks("tok")
    assert {t["id"] for t in tasks} == {"t1", "t2"}
    assert {t["_projectName"] for t in tasks} == {"Work", "Home"}
    assert len(calls) == 3


def test_fetch_open_tasks_url_encodes_the_project_id(monkeypatch):
    """The id comes from TickTick, but interpolating it raw is a habit worth not having."""
    calls = []

    def fake_request(method, url, token, timeout=30):
        calls.append(url)
        if url.endswith("/project"):
            return [{"id": "a b/c?d", "name": "Odd"}]
        return {"tasks": []}

    monkeypatch.setattr(ticktick_api, "_request", fake_request)
    ticktick_api.fetch_open_tasks("tok")
    assert calls[1] == f"{ticktick_api.BASE_URL}/project/a%20b%2Fc%3Fd/data"


def test_fetch_open_tasks_one_bad_project_does_not_abort(monkeypatch):
    def fake_request(method, url, token, timeout=30):
        if url.endswith("/project"):
            return [{"id": "p1", "name": "Work"}, {"id": "p2", "name": "Home"}]
        if url.endswith("/project/p1/data"):
            raise HTTPError(url, 500, "boom", {}, None)
        return {"tasks": [{"id": "t2", "title": "Dishes", "projectId": "p2"}]}

    monkeypatch.setattr(ticktick_api, "_request", fake_request)
    errors: list[str] = []
    tasks = ticktick_api.fetch_open_tasks("tok", errors=errors)
    assert {t["id"] for t in tasks} == {"t2"}
    assert len(errors) == 1


def test_fetch_open_tasks_propagates_project_list_failure(monkeypatch):
    """A dead API must not read as "you have no tasks" (fail loudly, 2026-08-08)."""

    def fake_request(method, url, token, timeout=30):
        raise HTTPError(url, 401, "unauthorized", {}, None)

    monkeypatch.setattr(ticktick_api, "_request", fake_request)
    errors: list[str] = []
    with pytest.raises(HTTPError):
        ticktick_api.fetch_open_tasks("tok", errors=errors)


def test_fetch_open_tasks_rejects_non_list_project_payload(monkeypatch):
    monkeypatch.setattr(
        ticktick_api, "_request", lambda method, url, token, timeout=30: {"error": "nope"}
    )
    with pytest.raises(ValueError, match="/project"):
        ticktick_api.fetch_open_tasks("tok")


def test_fetch_open_tasks_records_malformed_project_data(monkeypatch):
    """A project whose payload is not an object is skipped, not an AttributeError."""

    def fake_request(method, url, token, timeout=30):
        if url.endswith("/project"):
            return [{"id": "p1", "name": "Work"}, {"id": "p2", "name": "Home"}]
        if url.endswith("/project/p1/data"):
            return ["not", "an", "object"]
        return {"tasks": [{"id": "t2", "title": "Dishes", "projectId": "p2"}]}

    monkeypatch.setattr(ticktick_api, "_request", fake_request)
    errors: list[str] = []
    tasks = ticktick_api.fetch_open_tasks("tok", errors=errors)
    assert {t["id"] for t in tasks} == {"t2"}
    assert len(errors) == 1
    assert "p1" in errors[0]


def test_fetch_open_tasks_skips_projects_without_id(monkeypatch):
    def fake_request(method, url, token, timeout=30):
        if url.endswith("/project"):
            return [{"name": "Nameless"}, {"id": "p2", "name": "Home"}]
        return {"tasks": [{"id": "t2", "title": "Dishes", "projectId": "p2"}]}

    monkeypatch.setattr(ticktick_api, "_request", fake_request)
    assert [t["id"] for t in ticktick_api.fetch_open_tasks("tok")] == ["t2"]


def test_fetch_open_tasks_never_leaks_the_token(monkeypatch, caplog):
    def fake_request(method, url, token, timeout=30):
        if url.endswith("/project"):
            return [{"id": "p1", "name": "Work"}]
        raise HTTPError(url, 500, "boom", {}, None)

    monkeypatch.setattr(ticktick_api, "_request", fake_request)
    errors: list[str] = []
    with caplog.at_level(logging.DEBUG, logger="aggregator.sources.ticktick_api"):
        ticktick_api.fetch_open_tasks(TOKEN, errors=errors)
    assert TOKEN not in caplog.text
    assert TOKEN not in " ".join(errors)


# --- record mapping -------------------------------------------------------


def test_task_to_record_shape():
    task = {
        "id": "t1",
        "title": "Ship it",
        "content": "details here",
        "priority": 5,
        "dueDate": "2026-08-09T12:00:00+0000",
        "tags": ["work"],
        "_projectName": "Work",
    }
    rec = ticktick_api.task_to_record(task)
    assert rec.stable_id == "ticktick:t1"
    assert rec.source == "ticktick"
    assert rec.subject == "Ship it"
    assert rec.body == "details here"
    assert set(rec.tags) >= {"work", "Work", "open"}
    assert rec.extra["provenance"] == "api"
    assert rec.extra["status"] == "0"
    assert rec.extra["priority"] == "high"
    assert "completed_time_approx" not in rec.extra


def test_task_to_record_inferred_completion_uses_status_2():
    """TickTick has no status 1: `0 Normal / -1 Abandoned / 2 Completed`."""
    when = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)
    rec = ticktick_api.task_to_record(
        {"id": "t1", "title": "Ship it", "_projectName": "Work"},
        completed_at=when,
        provenance="api-inferred-complete",
    )
    assert rec.extra["status"] == "2"
    assert rec.extra["status"] != "1"
    assert "completed" in rec.tags
    assert "open" not in rec.tags
    assert "archived" not in rec.tags
    assert rec.extra["provenance"] == "api-inferred-complete"
    assert rec.extra["completed_time_approx"] is True
    assert rec.updated_at == when


def test_task_to_record_matches_csv_extra_vocabulary():
    """Task 8 merges API and CSV records by stable_id; the keys must line up."""
    rec = ticktick_api.task_to_record({"id": "t1", "title": "Ship it"})
    assert {
        "provenance",
        "status",
        "priority",
        "due_date",
        "start_date",
        "repeat",
        "parent_id",
    } <= set(rec.extra)


def test_task_to_record_untitled_falls_back_to_id():
    rec = ticktick_api.task_to_record({"id": "t9", "title": "   "})
    assert rec.subject == "t9"


def test_task_to_record_missing_fields_are_empty_not_none():
    rec = ticktick_api.task_to_record({"id": "t1"})
    assert rec.body == ""
    assert rec.extra["due_date"] == ""
    assert rec.extra["parent_id"] == ""
    # Priority is the exception: absent means TickTick's own default level, 0.
    assert rec.extra["priority"] == "none"
    assert rec.created_at is None
    assert rec.updated_at is None


def test_task_to_record_priority_uses_the_shared_names():
    """Names, not digits — and the same table the CSV leg maps through."""
    assert ticktick_api.priority_name is ticktick_csv.priority_name
    for level, name in ticktick_csv.PRIORITY_NAMES.items():
        rec = ticktick_api.task_to_record({"id": "t1", "title": "x", "priority": level})
        assert rec.extra["priority"] == name


def test_task_to_record_normalises_timestamps_to_utc():
    """store.py compares timestamps as ISO text, so a foreign offset must not survive."""
    rec = ticktick_api.task_to_record(
        {"id": "t1", "title": "x", "createdTime": "2026-08-01T10:00:00+0200"}
    )
    assert rec.created_at == datetime(2026, 8, 1, 8, 0, tzinfo=UTC)
    assert rec.created_at.isoformat().endswith("+00:00")


@pytest.mark.parametrize(
    "raw",
    [
        "2026-08-09T12:00:00+0000",
        "2026-08-09T12:00:00.000+0000",
        "2026-08-09T12:00:00Z",
        "2026-08-09T12:00:00.000Z",
    ],
)
def test_task_to_record_parses_api_timestamp_formats(raw):
    rec = ticktick_api.task_to_record({"id": "t1", "title": "x", "createdTime": raw})
    assert rec.created_at == datetime(2026, 8, 9, 12, 0, tzinfo=UTC)


def test_task_to_record_tolerates_unparseable_timestamp():
    rec = ticktick_api.task_to_record({"id": "t1", "title": "x", "createdTime": "not a date"})
    assert rec.created_at is None


# --- token resolution -----------------------------------------------------


def test_resolve_token_absent_is_a_skip_not_an_error():
    """No token configured at all: the API leg is skipped, CSV still runs."""
    assert ticktick_api.resolve_token(None, None) is None
    assert ticktick_api.resolve_token("   ", None) is None


def test_resolve_token_prefers_explicit_token(tmp_path):
    token_file = tmp_path / "tok"
    token_file.write_text("from-file\n", encoding="utf-8")
    assert ticktick_api.resolve_token("explicit", str(token_file)) == "explicit"


def test_resolve_token_reads_file_and_strips(tmp_path):
    token_file = tmp_path / "tok"
    token_file.write_text("  from-file\n", encoding="utf-8")
    assert ticktick_api.resolve_token(None, str(token_file)) == "from-file"


def test_resolve_token_empty_file_is_a_skip(tmp_path):
    """The agenix secret exists but is declared-and-empty until the user fills it."""
    token_file = tmp_path / "tok"
    token_file.write_text("\n", encoding="utf-8")
    assert ticktick_api.resolve_token(None, str(token_file)) is None


def test_resolve_token_unreadable_file_raises(tmp_path):
    """A configured-but-broken secret is a failure, not a silent skip."""
    with pytest.raises(OSError):
        ticktick_api.resolve_token(None, str(tmp_path / "missing"))
