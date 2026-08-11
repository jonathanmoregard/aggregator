from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from aggregator.sources import ticktick_api, ticktick_csv

TOKEN = "sup3r-s3cret-token"
API_LOG = "aggregator.sources.ticktick_api"


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


@pytest.fixture(autouse=True)
def _no_real_credentials(monkeypatch, tmp_path):
    """No test in this file may read the developer's own TickTick token.

    ``resolve_token`` falls back to the shared ``~/.config/todo/env`` store,
    which on a developer machine holds a live write-scoped token. Without this,
    "no token configured" tests would silently pass *because* a real token was
    found, and the suite's result would depend on whose machine it ran on.
    Tests that want an env file point ``DEFAULT_ENV_FILE`` at their own.
    """
    monkeypatch.setattr(
        ticktick_api, "DEFAULT_ENV_FILE", str(tmp_path / "no-such-env")
    )
    monkeypatch.delenv("TICKTICK_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("TICKTICK_TOKEN_EXPIRES_AT", raising=False)


def full_task(**over):
    """A plausible Open API Task object: every documented field, populated.

    The payloads this file used to carry were 2-6 key stubs that never included
    ``status``, ``completedTime``, ``desc`` or ``items`` — which is exactly why
    a green suite could not see that the module read none of them. Reach for
    this fixture by default; use a bare stub only when the point of the test is
    an absent field.
    """
    task = {
        "id": "6247ee29c0f5d21f4c1f6f88",
        "projectId": "6226ff9877acee87727f6bca",
        "title": "Ship it",
        "isAllDay": False,
        "completedTime": "2026-08-09T04:00:00.000+0000",
        "content": "details here",
        "desc": "the longer note",
        "dueDate": "2026-08-09T03:00:00.000+0000",
        "items": [
            {"id": "i1", "status": 0, "title": "first step"},
            {"id": "i2", "status": 1, "title": "second step"},
        ],
        "priority": 5,
        "reminders": ["TRIGGER:P0DT9H0M0S"],
        "repeatFlag": "RRULE:FREQ=DAILY;INTERVAL=1",
        "sortOrder": -2199023255552,
        "startDate": "2026-08-09T03:00:00.000+0000",
        "status": 2,
        "timeZone": "America/Los_Angeles",
        "tags": ["errands"],
        "_projectName": "Errands",
    }
    task.update(over)
    return task


def _open_task(**over):
    """The same object as TickTick would send it for a still-open task.

    ``createdTime``/``modifiedTime`` are spelled in explicitly here rather than
    in :func:`full_task`, because they are *not* in TickTick's documented Task
    object and this module's use of those names is unverified (see
    ``TIMESTAMP_FIELDS``). Keeping them out of the baseline fixture is what
    stops these tests from quietly asserting that a guess is true.
    """
    dated = {
        "status": 0,
        "completedTime": None,
        "createdTime": "2026-08-01T09:00:00.000+0000",
        "modifiedTime": "2026-08-08T10:00:00.000+0000",
    }
    return full_task(**{**dated, **over})


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
            return {"tasks": [_open_task(id="t1", projectId="p1")]}
        return {"tasks": [_open_task(id="t2", title="Dishes", projectId="p2")]}

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
        return {"tasks": [_open_task(id="t2", title="Dishes", projectId="p2")]}

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
        return {"tasks": [_open_task(id="t2", title="Dishes", projectId="p2")]}

    monkeypatch.setattr(ticktick_api, "_request", fake_request)
    errors: list[str] = []
    tasks = ticktick_api.fetch_open_tasks("tok", errors=errors)
    assert {t["id"] for t in tasks} == {"t2"}
    assert len(errors) == 1
    assert "p1" in errors[0]


@pytest.mark.parametrize(
    ("bad_tasks", "expect_error"),
    [
        ({"t1": {"id": "t1"}}, True),  # tasks as an object
        (["t1", "t2"], True),  # list of strings
        ("t1", True),  # a bare string
        (None, False),  # explicitly null
        (12, True),  # a number
    ],
)
def test_fetch_open_tasks_survives_a_malformed_tasks_value(monkeypatch, bad_tasks, expect_error):
    """One project with a surprising ``tasks`` shape must not kill the other nine.

    Every one of these used to raise TypeError from ``task["_projectName"] = …``,
    outside the try/except, aborting the whole walk.
    """
    payload = {"tasks": bad_tasks} if bad_tasks is not None else {"tasks": None}

    def fake_request(method, url, token, timeout=30):
        if url.endswith("/project"):
            return [{"id": "p1", "name": "Work"}, {"id": "p2", "name": "Home"}]
        if url.endswith("/project/p1/data"):
            return payload
        return {"tasks": [_open_task(id="t2", title="Dishes", projectId="p2")]}

    monkeypatch.setattr(ticktick_api, "_request", fake_request)
    errors: list[str] = []
    tasks = ticktick_api.fetch_open_tasks("tok", errors=errors)
    assert {t["id"] for t in tasks} == {"t2"}
    assert bool(errors) is expect_error


def test_fetch_open_tasks_tasks_key_missing_entirely(monkeypatch):
    def fake_request(method, url, token, timeout=30):
        if url.endswith("/project"):
            return [{"id": "p1", "name": "Work"}]
        return {"project": {"id": "p1"}}

    monkeypatch.setattr(ticktick_api, "_request", fake_request)
    errors: list[str] = []
    assert ticktick_api.fetch_open_tasks("tok", errors=errors) == []
    assert errors == []


def test_fetch_open_tasks_skips_projects_without_id(monkeypatch):
    def fake_request(method, url, token, timeout=30):
        if url.endswith("/project"):
            return [{"name": "Nameless"}, {"id": "p2", "name": "Home"}]
        return {"tasks": [_open_task(id="t2", title="Dishes", projectId="p2")]}

    monkeypatch.setattr(ticktick_api, "_request", fake_request)
    assert [t["id"] for t in ticktick_api.fetch_open_tasks("tok")] == ["t2"]


def test_fetch_open_tasks_never_leaks_the_token(monkeypatch, caplog):
    def fake_request(method, url, token, timeout=30):
        if url.endswith("/project"):
            return [{"id": "p1", "name": "Work"}]
        raise HTTPError(url, 500, "boom", {}, None)

    monkeypatch.setattr(ticktick_api, "_request", fake_request)
    errors: list[str] = []
    with caplog.at_level(logging.DEBUG, logger=API_LOG):
        ticktick_api.fetch_open_tasks(TOKEN, errors=errors)
    assert TOKEN not in caplog.text
    assert TOKEN not in " ".join(errors)


# --- coverage limits are reported, not left to be inferred ----------------


def test_fetch_open_tasks_notes_the_missing_inbox(monkeypatch, caplog):
    """59 of 1302 tasks (5 of 238 open) live in an Inbox the listing never returns.

    Deliberately **no** ``caplog.at_level``. Nothing under ``aggregator/`` calls
    basicConfig/dictConfig/addHandler, so an operator running the CLI has an
    unconfigured root logger, and ``logging.lastResort`` — level WARNING — is the
    only thing that prints. This note was emitted at INFO and therefore produced
    literally zero output; forcing the capture level to INFO made the test assert
    the implementation back to itself instead of what an operator can see.
    """

    def fake_request(method, url, token, timeout=30):
        if url.endswith("/project"):
            return [{"id": "p1", "name": "Work"}]
        return {"tasks": [_open_task(id="t1", projectId="p1")]}

    monkeypatch.setattr(ticktick_api, "_request", fake_request)
    ticktick_api.fetch_open_tasks("tok")
    notes = [r for r in caplog.records if "Inbox" in r.getMessage()]
    assert notes, "the Inbox coverage note is below the default level, so nobody sees it"
    assert notes[0].levelno >= logging.lastResort.level


def test_no_inbox_note_when_the_listing_does_contain_one(monkeypatch, caplog):
    """Self-checking: if TickTick ever lists the Inbox, the caveat goes quiet."""

    def fake_request(method, url, token, timeout=30):
        if url.endswith("/project"):
            return [{"id": "p1", "name": "Inbox"}]
        return {"tasks": [_open_task(id="t1", projectId="p1")]}

    monkeypatch.setattr(ticktick_api, "_request", fake_request)
    with caplog.at_level(logging.INFO, logger=API_LOG):
        ticktick_api.fetch_open_tasks("tok")
    assert "out of scope" not in caplog.text


def test_a_batch_with_no_parseable_timestamp_is_loud(monkeypatch, caplog):
    """The whole point of C2: 100% null timestamps must not look like success.

    ``createdTime``/``modifiedTime`` are unverified against TickTick's published
    Task object. If they are the wrong names, the first run with a real token
    says so — naming every field it looked under — instead of quietly indexing
    every API record with no date at all.
    """

    def fake_request(method, url, token, timeout=30):
        if url.endswith("/project"):
            return [{"id": "p1", "name": "Work"}]
        # A task object with no timestamp under any name this module knows.
        return {"tasks": [{"id": "t1", "title": "Ship it", "status": 0}]}

    monkeypatch.setattr(ticktick_api, "_request", fake_request)
    errors: list[str] = []
    with caplog.at_level(logging.WARNING, logger=API_LOG):
        tasks = ticktick_api.fetch_open_tasks("tok", errors=errors)
    assert len(tasks) == 1
    assert len(errors) == 1
    for field in ticktick_api.TIMESTAMP_FIELDS:
        assert field in errors[0]
        assert field in caplog.text


def test_a_batch_with_timestamps_is_quiet(monkeypatch, caplog):
    def fake_request(method, url, token, timeout=30):
        if url.endswith("/project"):
            return [{"id": "p1", "name": "Work"}]
        return {
            "tasks": [
                {"id": "t1", "title": "no dates here"},
                _open_task(id="t2", modifiedTime="2026-08-09T04:00:00.000+0000"),
            ]
        }

    monkeypatch.setattr(ticktick_api, "_request", fake_request)
    errors: list[str] = []
    with caplog.at_level(logging.WARNING, logger=API_LOG):
        ticktick_api.fetch_open_tasks("tok", errors=errors)
    assert errors == []
    assert "TIMESTAMP_FIELDS" not in caplog.text


def test_an_empty_batch_is_not_a_timestamp_failure(monkeypatch):
    """No tasks at all is a legitimate state; only a *populated* dateless batch is not."""

    def fake_request(method, url, token, timeout=30):
        if url.endswith("/project"):
            return [{"id": "p1", "name": "Work"}]
        return {"tasks": []}

    monkeypatch.setattr(ticktick_api, "_request", fake_request)
    errors: list[str] = []
    assert ticktick_api.fetch_open_tasks("tok", errors=errors) == []
    assert errors == []


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


def test_task_to_record_maps_a_full_payload():
    """One assertion set over a realistic object, so a dropped field is visible."""
    rec = ticktick_api.task_to_record(full_task())
    assert rec.stable_id == "ticktick:6247ee29c0f5d21f4c1f6f88"
    assert rec.subject == "Ship it"
    assert "details here" in rec.body
    assert "the longer note" in rec.body
    assert "first step" in rec.body
    assert "second step" in rec.body
    assert set(rec.tags) == {"errands", "Errands", "completed"}
    assert rec.extra["status"] == "2"
    assert rec.extra["priority"] == "high"
    assert rec.extra["due_date"] == "2026-08-09T03:00:00+0000"
    assert rec.extra["start_date"] == "2026-08-09T03:00:00+0000"
    assert rec.extra["repeat"] == "RRULE:FREQ=DAILY;INTERVAL=1"
    assert rec.extra["project_id"] == "6226ff9877acee87727f6bca"
    assert rec.updated_at == datetime(2026, 8, 9, 4, 0, tzinfo=UTC)


# --- C1: the payload's own status wins over any inference -----------------


def test_task_to_record_trusts_the_payload_status_over_the_open_assumption():
    """Measured defect: a payload saying `"status": 2` was recorded as open.

    Task 8 lets the fresher API observation beat the CSV row, so this resurrected
    a completed task in the index and discarded the backup's evidence for it.
    """
    rec = ticktick_api.task_to_record(full_task())
    assert rec.extra["status"] == "2"
    assert "completed" in rec.tags
    assert "open" not in rec.tags


def test_task_to_record_warns_when_the_api_serves_a_non_open_task(caplog):
    """It contradicts coverage limit 1, so it is reported rather than absorbed."""
    with caplog.at_level(logging.WARNING, logger=API_LOG):
        ticktick_api.task_to_record(full_task())
    assert "coverage limit" in caplog.text


def test_task_to_record_payload_status_minus_one_is_abandoned():
    rec = ticktick_api.task_to_record(full_task(status=-1))
    assert rec.extra["status"] == "-1"
    assert "abandoned" in rec.tags
    assert "open" not in rec.tags


def test_task_to_record_absent_status_is_open():
    rec = ticktick_api.task_to_record({"id": "t1", "title": "x"})
    assert rec.extra["status"] == ticktick_csv.STATUS_OPEN
    assert "open" in rec.tags


def test_task_to_record_unknown_status_is_verbatim_and_warns(caplog):
    """TickTick has no status 1. A new vendor code must be visible, not coerced."""
    with caplog.at_level(logging.WARNING, logger=API_LOG):
        rec = ticktick_api.task_to_record(full_task(status=1))
    assert rec.extra["status"] == "1"
    assert "open" in rec.tags
    assert "'1'" in caplog.text


def test_task_to_record_uses_the_payloads_completed_time():
    """A real completion timestamp, not an approximation — so no approx flag."""
    rec = ticktick_api.task_to_record(full_task())
    assert rec.updated_at == datetime(2026, 8, 9, 4, 0, tzinfo=UTC)
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
    assert rec.extra["completed_time_approx"] == "true"
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


def test_task_to_record_extra_values_are_all_strings():
    """extra is json.dumps'd: an int lands a JSON number where the CSV leg lands text."""
    rec = ticktick_api.task_to_record(
        full_task(dueDate=1785000000000, startDate=None, repeatFlag=7, parentId=42, projectId=9)
    )
    for key, value in rec.extra.items():
        assert isinstance(value, str), f"extra[{key!r}] is {type(value).__name__}"
    assert json.loads(json.dumps(rec.extra))["due_date"] == "1785000000000"


def test_extra_values_are_all_strings_on_an_inferred_completion():
    """No exemptions. ``completed_time_approx`` was a bool, so json.dumps wrote the
    literal ``true`` and a DSL filter would have had to match that instead of text."""
    rec = ticktick_api.task_to_record(
        {"id": "t1", "title": "x"},
        completed_at=datetime(2026, 8, 9, 12, 0, tzinfo=UTC),
        provenance="api-inferred-complete",
    )
    for key, value in rec.extra.items():
        assert isinstance(value, str), f"extra[{key!r}] is {type(value).__name__}"
    assert json.loads(json.dumps(rec.extra))["completed_time_approx"] == "true"


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


# --- M4: the id is validated where the record is minted -------------------


def test_task_to_record_rejects_a_null_id():
    """``str(None)`` minted the plausible literal "ticktick:None", one shared id
    for every null-id task in the batch."""
    with pytest.raises(ValueError, match="usable id"):
        ticktick_api.task_to_record({"id": None, "title": "x"})


@pytest.mark.parametrize("bad", [{}, {"id": ""}, {"id": "   "}])
def test_task_to_record_rejects_an_unusable_id(bad):
    with pytest.raises(ValueError, match="usable id"):
        ticktick_api.task_to_record(bad)


def test_task_to_record_keeps_a_zero_id():
    """A falsy-but-real id must survive: ``if not task.get("id")`` would drop it."""
    rec = ticktick_api.task_to_record({"id": 0, "title": "x"})
    assert rec.stable_id == "ticktick:0"


# --- M5 / m2: body and text coercion --------------------------------------


def test_task_to_record_body_carries_desc_and_checklist_items():
    """The CSV leg's ``Content`` includes checklist text; dropping it here made
    text that *was* searchable unsearchable as soon as an API record won the merge."""
    rec = ticktick_api.task_to_record(full_task())
    assert rec.body.splitlines() == [
        "details here",
        "the longer note",
        "- first step",
        "- second step",
    ]


def test_task_to_record_body_tolerates_a_malformed_items_value(caplog):
    with caplog.at_level(logging.WARNING, logger=API_LOG):
        rec = ticktick_api.task_to_record(full_task(items="not a list"))
    assert rec.body == "details here\nthe longer note"
    assert "items" in caplog.text


def test_task_to_record_coerces_non_string_title_and_content():
    """A payload is untrusted input: ``.strip()`` on an int raised AttributeError,
    and a dict ``content`` reached ``scrub()`` in the store as a dict."""
    rec = ticktick_api.task_to_record({"id": "t1", "title": 123, "content": {"a": 1}})
    assert rec.subject == "123"
    assert isinstance(rec.body, str)
    assert "a" in rec.body


# --- M3: tags -------------------------------------------------------------


def test_task_to_record_splits_a_string_tags_value():
    """A bare string was iterated per character: "work" became w,o,r,k."""
    rec = ticktick_api.task_to_record({"id": "t1", "title": "x", "tags": "work, home"})
    assert set(rec.tags) == {"work", "home", "open"}


def test_task_to_record_ignores_a_malformed_tags_value(caplog):
    with caplog.at_level(logging.WARNING, logger=API_LOG):
        rec = ticktick_api.task_to_record({"id": "t1", "title": "x", "tags": {"a": 1}})
    assert rec.tags == ["open"]
    assert "tags" in caplog.text


def test_task_to_record_coerces_non_string_tag_entries():
    rec = ticktick_api.task_to_record({"id": "t1", "title": "x", "tags": ["work", 7, "", None]})
    assert set(rec.tags) == {"work", "7", "open"}


# --- M5: dates are spelled the way the CSV leg spells them ----------------


def test_task_to_record_strips_the_millisecond_fraction_from_dates():
    """The CSV export writes `+0000` with no fraction, on all 1129 dated rows."""
    rec = ticktick_api.task_to_record(full_task())
    assert rec.extra["due_date"] == "2026-08-09T03:00:00+0000"
    assert rec.extra["start_date"] == "2026-08-09T03:00:00+0000"


def test_task_to_record_date_text_is_normalised_to_utc():
    rec = ticktick_api.task_to_record(full_task(dueDate="2026-08-09T05:00:00.000+0200"))
    assert rec.extra["due_date"] == "2026-08-09T03:00:00+0000"


def test_task_to_record_unparseable_date_is_kept_verbatim_and_warns(caplog):
    with caplog.at_level(logging.WARNING, logger=API_LOG):
        rec = ticktick_api.task_to_record(full_task(dueDate="whenever"))
    assert rec.extra["due_date"] == "whenever"
    assert "whenever" in caplog.text


def test_date_normalisation_is_the_shared_one():
    """Same function object, so the two legs cannot drift apart (task 8 merge)."""
    assert ticktick_api.normalize_date_text is ticktick_csv.normalize_date_text
    assert ticktick_api.status_tag is ticktick_csv.status_tag


# --- m4: an API-sourced warning is attributed to the API module -----------


@pytest.mark.parametrize(
    "task",
    [
        {"id": "t1", "title": "x", "priority": 2},
        {"id": "t1", "title": "x", "dueDate": "whenever"},
        {"id": "t1", "title": "x", "status": 7},
    ],
)
def test_warnings_about_api_values_name_the_api_module(caplog, task):
    """An operator debugging one of these must not be sent to the backup file."""
    with caplog.at_level(logging.WARNING):
        ticktick_api.task_to_record(task)
    assert [r.name for r in caplog.records] == [API_LOG]


# --- C2: timestamps ------------------------------------------------------


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


def test_task_to_record_tolerates_unparseable_timestamp(caplog):
    with caplog.at_level(logging.WARNING, logger=API_LOG):
        rec = ticktick_api.task_to_record({"id": "t1", "title": "x", "createdTime": "not a date"})
    assert rec.created_at is None
    assert "createdTime" in caplog.text


def test_a_non_string_timestamp_warns_and_names_the_field(caplog):
    """Measured: an epoch int produced a null date and zero diagnostic output."""
    with caplog.at_level(logging.WARNING, logger=API_LOG):
        rec = ticktick_api.task_to_record({"id": "t1", "title": "x", "createdTime": 1785000000000})
    assert rec.created_at is None
    assert "createdTime" in caplog.text
    assert "int" in caplog.text


def test_an_absent_timestamp_is_quiet(caplog):
    """An open task legitimately has no completedTime; that is not a fault."""
    with caplog.at_level(logging.WARNING, logger=API_LOG):
        ticktick_api.task_to_record({"id": "t1", "title": "x"})
    assert caplog.text == ""


def test_updated_at_prefers_completion_then_modification_then_creation():
    created = "2026-08-01T00:00:00+0000"
    modified = "2026-08-05T00:00:00+0000"
    completed = "2026-08-09T00:00:00+0000"
    base = {"id": "t1", "title": "x", "createdTime": created}
    assert ticktick_api.task_to_record(base).updated_at == datetime(2026, 8, 1, tzinfo=UTC)
    with_mod = ticktick_api.task_to_record({**base, "modifiedTime": modified})
    assert with_mod.updated_at == datetime(2026, 8, 5, tzinfo=UTC)
    with_done = ticktick_api.task_to_record(
        {**base, "modifiedTime": modified, "completedTime": completed, "status": 2}
    )
    assert with_done.updated_at == datetime(2026, 8, 9, tzinfo=UTC)


def test_timestamp_fields_is_derived_not_a_second_copy():
    """The tripwire and the record mapping must look under the same names."""
    derived = (*ticktick_api.UPDATED_FIELDS, ticktick_api.CREATED_FIELD)
    assert derived == ticktick_api.TIMESTAMP_FIELDS


def test_correcting_the_field_names_moves_task_to_record_too(monkeypatch):
    """The tripwire tells an operator to correct these constants. Measured before
    the fix: doing exactly that silenced the warning and left every record
    dateless, because ``task_to_record`` spelled the old names a second time —
    which trades the loud failure back for the silent one."""
    monkeypatch.setattr(ticktick_api, "CREATED_FIELD", "bornAt")
    monkeypatch.setattr(ticktick_api, "UPDATED_FIELDS", ("touchedAt",))
    task = {
        "id": "t1",
        "title": "x",
        "bornAt": "2026-08-01T00:00:00+0000",
        "touchedAt": "2026-08-05T00:00:00+0000",
    }
    rec = ticktick_api.task_to_record(task)
    assert rec.created_at == datetime(2026, 8, 1, tzinfo=UTC)
    assert rec.updated_at == datetime(2026, 8, 5, tzinfo=UTC)


# --- token resolution -----------------------------------------------------


def test_resolve_token_absent_is_a_skip_not_an_error():
    """No token configured at all: the API leg is skipped, CSV still runs."""
    assert ticktick_api.resolve_token(None, None) is None
    assert ticktick_api.resolve_token("   ", None) is None


def test_resolve_token_prefers_explicit_token(tmp_path):
    token_file = tmp_path / "tok"
    token_file.write_text("from-file\n", encoding="utf-8")
    assert ticktick_api.resolve_token("explicit", str(token_file)) == "explicit"


def test_resolve_token_blank_explicit_token_falls_through_to_the_file(tmp_path):
    """A whitespace-only token used to shadow a perfectly good token file."""
    token_file = tmp_path / "tok"
    token_file.write_text("from-file\n", encoding="utf-8")
    assert ticktick_api.resolve_token("   ", str(token_file)) == "from-file"


def test_resolve_token_reads_file_and_strips(tmp_path):
    token_file = tmp_path / "tok"
    token_file.write_text("  from-file\n", encoding="utf-8")
    assert ticktick_api.resolve_token(None, str(token_file)) == "from-file"


def test_resolve_token_empty_file_is_a_skip(tmp_path):
    """The agenix secret exists but is declared-and-empty until the user fills it."""
    token_file = tmp_path / "tok"
    token_file.write_text("\n", encoding="utf-8")
    assert ticktick_api.resolve_token(None, str(token_file)) is None


# --- the shared ~/.config/todo/env credential store -------------------------
#
# The token is not this project's to own: ~/.claude/todo/backends/ticktick.py
# already holds a TickTick OAuth client that writes a refreshed access token
# back to ~/.config/todo/env, and router-agent's add_task MCP reads it from
# there. A second copy in an agenix secret would go stale the moment that
# backend refreshed, and the two would disagree with no way to tell which was
# live. So this leg reads the same file instead of a private one.


def _env_file(tmp_path, **pairs):
    """Write a ``KEY=VALUE`` env file in the shared store's format."""
    path = tmp_path / "env"
    path.write_text(
        "".join(f"{k}={v}\n" for k, v in pairs.items()), encoding="utf-8"
    )
    return str(path)


def test_resolve_token_reads_the_shared_todo_env_file(tmp_path, monkeypatch):
    """With nothing else configured, the shared store is the token's home."""
    monkeypatch.setattr(
        ticktick_api,
        "DEFAULT_ENV_FILE",
        _env_file(tmp_path, TICKTICK_ACCESS_TOKEN="from-shared-store"),
    )
    assert ticktick_api.resolve_token(None, None) == "from-shared-store"


def test_resolve_token_strips_quotes_the_shared_store_may_carry(tmp_path, monkeypatch):
    """The todo backend's own reader strips quotes, so this one must agree.

    A token read as ``"abc"`` instead of ``abc`` produces a 401 that looks like
    an expired credential, sending the operator to re-authorize a token that was
    never broken.
    """
    monkeypatch.setattr(
        ticktick_api,
        "DEFAULT_ENV_FILE",
        _env_file(tmp_path, TICKTICK_ACCESS_TOKEN='"quoted-token"  '),
    )
    assert ticktick_api.resolve_token(None, None) == "quoted-token"


def test_resolve_token_ignores_comments_and_other_keys(tmp_path, monkeypatch):
    path = tmp_path / "env"
    path.write_text(
        "# a comment\n"
        "\n"
        "TICKTICK_CLIENT_SECRET=not-the-token\n"
        "TICKTICK_ACCESS_TOKEN=the-token\n"
        "malformed-line-without-equals\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ticktick_api, "DEFAULT_ENV_FILE", str(path))
    assert ticktick_api.resolve_token(None, None) == "the-token"


def test_resolve_token_missing_shared_store_is_a_skip_not_an_error(tmp_path, monkeypatch):
    """An absent shared store means "API leg not set up", which is supported."""
    monkeypatch.setattr(
        ticktick_api, "DEFAULT_ENV_FILE", str(tmp_path / "nope" / "env")
    )
    assert ticktick_api.resolve_token(None, None) is None


def test_resolve_token_empty_agenix_file_falls_through_to_the_shared_store(
    tmp_path, monkeypatch
):
    """The merged agenix placeholder must not shadow the real credential.

    ``ticktick-api-token.age`` ships as a single newline, so before this the
    empty read short-circuited to None and the API leg was skipped even with a
    live token sitting in the shared store.
    """
    monkeypatch.setattr(
        ticktick_api,
        "DEFAULT_ENV_FILE",
        _env_file(tmp_path, TICKTICK_ACCESS_TOKEN="from-shared-store"),
    )
    placeholder = tmp_path / "tok"
    placeholder.write_text("\n", encoding="utf-8")
    assert ticktick_api.resolve_token(None, str(placeholder)) == "from-shared-store"


def test_resolve_token_prefers_the_environment_over_the_shared_store(
    tmp_path, monkeypatch
):
    """A systemd unit passing the token in the environment wins over the file."""
    monkeypatch.setattr(
        ticktick_api,
        "DEFAULT_ENV_FILE",
        _env_file(tmp_path, TICKTICK_ACCESS_TOKEN="from-shared-store"),
    )
    monkeypatch.setenv("TICKTICK_ACCESS_TOKEN", "from-environment")
    assert ticktick_api.resolve_token(None, None) == "from-environment"


def test_resolve_token_expired_shared_token_raises_and_names_the_fix(
    tmp_path, monkeypatch
):
    """An expired token must fail loudly, not 401 halfway through the walk.

    This leg runs unattended from a timer and cannot do the browser
    re-authorization the todo backend does, so the only useful thing it can do
    is stop and say which command the human must run.
    """
    monkeypatch.setattr(
        ticktick_api,
        "DEFAULT_ENV_FILE",
        _env_file(
            tmp_path,
            TICKTICK_ACCESS_TOKEN="stale",
            TICKTICK_TOKEN_EXPIRES_AT="1000000000",  # 2001
        ),
    )
    with pytest.raises(ticktick_api.TokenUnavailableError) as excinfo:
        ticktick_api.resolve_token(None, None)
    assert "todo-add --login" in str(excinfo.value)


def test_resolve_token_unparseable_expiry_does_not_block_a_usable_token(
    tmp_path, monkeypatch
):
    """Expiry is an optimization; a garbled one must not cost a working run."""
    monkeypatch.setattr(
        ticktick_api,
        "DEFAULT_ENV_FILE",
        _env_file(
            tmp_path, TICKTICK_ACCESS_TOKEN="usable", TICKTICK_TOKEN_EXPIRES_AT="soon"
        ),
    )
    assert ticktick_api.resolve_token(None, None) == "usable"


def test_resolve_token_explicit_token_skips_the_expiry_check(tmp_path, monkeypatch):
    """An explicitly passed token has no expiry metadata to check against."""
    monkeypatch.setattr(
        ticktick_api,
        "DEFAULT_ENV_FILE",
        _env_file(
            tmp_path,
            TICKTICK_ACCESS_TOKEN="stale",
            TICKTICK_TOKEN_EXPIRES_AT="1000000000",
        ),
    )
    assert ticktick_api.resolve_token("explicit", None) == "explicit"


def test_resolve_token_unreadable_file_raises(tmp_path):
    """A configured-but-broken secret is a failure, not a silent skip."""
    with pytest.raises(OSError):
        ticktick_api.resolve_token(None, str(tmp_path / "missing"))


def test_resolve_token_unreadable_file_raises_a_distinguishable_error(tmp_path):
    """"the secret is broken" and "TickTick is down" both arrive as OSError.

    A caller wrapping the whole API leg in ``except OSError`` cannot tell them
    apart; a dedicated subclass can, without breaking the OSError contract.
    """
    with pytest.raises(ticktick_api.TokenUnavailableError) as excinfo:
        ticktick_api.resolve_token(None, str(tmp_path / "missing"))
    assert issubclass(ticktick_api.TokenUnavailableError, OSError)
    assert not isinstance(excinfo.value, HTTPError)


# --- the open-task state file ---------------------------------------------
#
# WHY IT EXISTS: the Open API serves open tasks only and has no endpoint that
# reports a completion. A task in yesterday's batch that is missing from
# today's has either been completed or deleted, and the API will never say
# which. Written-down prior state is the only thing to diff against; without
# it the task simply stops appearing and an "I finished that last week" search
# comes back empty with nothing anywhere explaining why.


def test_state_roundtrip(tmp_path):
    path = tmp_path / "open_tasks.json"
    now = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
    ticktick_api.save_state(path, [{"id": "t1", "title": "A"}], now)
    state = ticktick_api.load_state(path)
    assert state["t1"]["task"]["title"] == "A"
    assert state["t1"]["last_seen"] == now.isoformat()


def test_load_missing_state_returns_empty(tmp_path):
    """The first ever poll has no baseline; that is normal, not a failure."""
    assert ticktick_api.load_state(tmp_path / "nope.json") == {}


@pytest.mark.parametrize(
    "write",
    [
        pytest.param(lambda p: p.write_text("{not json", encoding="utf-8"), id="truncated"),
        pytest.param(lambda p: p.write_text("[]", encoding="utf-8"), id="a-list-not-a-map"),
        pytest.param(lambda p: p.write_text('"scalar"', encoding="utf-8"), id="a-bare-scalar"),
        pytest.param(lambda p: p.write_bytes(b"\xff\xfe not utf-8"), id="undecodable"),
        pytest.param(lambda p: p.mkdir(), id="not-even-a-file"),
    ],
)
def test_load_corrupt_state_returns_empty_and_says_so(tmp_path, caplog, write):
    """Unusable state costs one poll's inference and self-heals — but loudly.

    It is a cache, not a database, so failing the whole ingest over it would be
    worse than the gap. Silently returning ``{}`` would be worse still: that is
    indistinguishable from a first poll, so a state file corrupted once would
    keep losing every completion and never explain why. No ``caplog.at_level``
    on purpose — nothing under ``aggregator/`` configures logging, so
    ``logging.lastResort`` (WARNING) is all an operator ever sees.
    """
    path = tmp_path / "open_tasks.json"
    write(path)
    assert ticktick_api.load_state(path) == {}
    notes = [r for r in caplog.records if "open_tasks.json" in r.getMessage()]
    assert notes, "an unreadable state file was discarded without telling anyone"
    assert notes[0].levelno >= logging.lastResort.level


def test_save_state_creates_its_directory(tmp_path):
    """The default path is three levels below $XDG_STATE_HOME, none of which exist.

    On a fresh machine the very first save is the one that has to create them,
    and a save that raises there would mean inference never starts working.
    """
    path = tmp_path / "state" / "aggregator" / "ticktick" / "open_tasks.json"
    ticktick_api.save_state(path, [{"id": "t1"}], datetime(2026, 8, 8, tzinfo=UTC))
    assert ticktick_api.load_state(path).keys() == {"t1"}


def test_a_failed_save_leaves_the_previous_state_intact(tmp_path, monkeypatch):
    """The write goes to a temp file and is renamed into place, never in situ.

    A save interrupted partway through an in-place write leaves truncated JSON,
    which costs the *next* poll its inference too — and by the poll after that
    the baseline is fresh again, so every completion that happened in between is
    gone for good. The rename is atomic, so the file is always either the old
    poll or the new one.
    """
    path = tmp_path / "open_tasks.json"
    first = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)
    ticktick_api.save_state(path, [{"id": "t1", "title": "A"}], first)

    real_write = Path.write_text

    def fail_on_the_scratch_file(self, *args, **kwargs):
        if self != path:
            raise OSError("no space left on device")
        return real_write(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_on_the_scratch_file)
    with pytest.raises(OSError):
        ticktick_api.save_state(path, [{"id": "t2", "title": "B"}], first)
    assert ticktick_api.load_state(path)["t1"]["task"]["title"] == "A"


def test_save_state_keys_by_the_modules_one_id_rule(tmp_path, caplog):
    """Keys are minted by ``_task_id``, the same rule ``task_to_record`` uses.

    Three things a hand-rolled ``if task.get("id")`` filter gets wrong, all of
    them tested here: the legitimate id ``0`` is falsy and would be dropped
    (the module already has a test refusing to lose it), a payload with no
    ``id`` key at all raises KeyError from ``str(task["id"])`` and takes the
    whole baseline down with it, and a padded id stored unstripped would never
    match the stripped id the next poll mints — inventing a completion for a
    task that never left.
    """
    path = tmp_path / "open_tasks.json"
    ticktick_api.save_state(
        path,
        [
            {"id": 0, "title": "falsy but real"},
            {"id": "  t2  ", "title": "padded"},
            {"id": None, "title": "null id"},
            {"title": "no id key at all"},
        ],
        datetime(2026, 8, 8, tzinfo=UTC),
    )
    assert ticktick_api.load_state(path).keys() == {"0", "t2"}
    dropped = [r for r in caplog.records if "2" in r.getMessage() and "state" in r.getMessage()]
    assert dropped, "tasks were dropped from the baseline without telling anyone"
    assert dropped[0].levelno >= logging.lastResort.level


def test_disappeared_task_becomes_inferred_completion(tmp_path):
    """Open last poll, absent this poll: the only completion signal there is.

    The record is built entirely from the payload the state file kept — the API
    will never serve this task again — which is what the stored ``task`` blob is
    for. ``completed_time_approx`` is the string ``"true"``, not the bool: every
    value in ``extra`` is text (``test_extra_values_are_all_strings_on_an_
    inferred_completion``), because ``extra`` is json.dumps'd and a bool would
    land the JSON literal ``true`` where a DSL filter expects a word.
    """
    path = tmp_path / "open_tasks.json"
    first = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)
    second = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
    ticktick_api.save_state(
        path,
        [{"id": "t1", "title": "Gone", "_projectName": "Work"}, {"id": "t2", "title": "Stays"}],
        first,
    )
    prev = ticktick_api.load_state(path)
    records = ticktick_api.infer_completions(prev, current_ids={"t2"}, now=second)
    assert [r.stable_id for r in records] == ["ticktick:t1"]
    rec = records[0]
    assert rec.subject == "Gone"
    assert "Work" in rec.tags
    assert rec.extra["provenance"] == "api-inferred-complete"
    assert rec.extra["completed_time_approx"] == "true"
    assert rec.extra["status"] == ticktick_csv.STATUS_COMPLETED
    assert rec.updated_at == second
    assert "completed" in rec.tags


def test_no_inference_on_first_ever_poll():
    """An empty baseline can complete nothing — including after a corrupt read.

    ``load_state`` answers a missing or unreadable file with ``{}``, so this is
    also the guarantee that a broken state file costs one poll's inference and
    nothing else. Green from the moment ``infer_completions`` existed: it is a
    regression guard on the direction of the diff, not a driver.
    """
    records = ticktick_api.infer_completions({}, current_ids={"t1"}, now=datetime.now(UTC))
    assert records == []


def test_a_poll_that_returned_nothing_at_all_is_reported(tmp_path, caplog):
    """Every open task vanishing at once is an outage shape, not 238 completions.

    ``fetch_open_tasks`` sinks a failed project into ``errors`` and carries on,
    so a run where the project listing succeeded and every project then 500'd
    comes back as an empty task list — indistinguishable, here, from the user
    finishing everything. The records are still emitted (the next healthy poll
    serves those tasks again with a fresher observation, and the merge reverts
    them, so suppressing would cost a real last completion permanently) but the
    operator is told, because a batch of approximate completions appearing in
    the index with no explanation is the silent-failure shape this repo forbids.
    """
    previous = {
        "t1": {"task": {"id": "t1", "title": "A"}, "last_seen": "2026-08-07T12:00:00+00:00"},
        "t2": {"task": {"id": "t2", "title": "B"}, "last_seen": "2026-08-07T12:00:00+00:00"},
    }
    records = ticktick_api.infer_completions(
        previous, current_ids=set(), now=datetime(2026, 8, 8, tzinfo=UTC)
    )
    assert [r.stable_id for r in records] == ["ticktick:t1", "ticktick:t2"]
    notes = [r for r in caplog.records if "outage" in r.getMessage()]
    assert notes, "a wholesale disappearance was recorded as completions in silence"
    assert notes[0].levelno >= logging.lastResort.level


def test_a_garbled_state_entry_does_not_cost_the_other_inferences(caplog):
    """One unusable entry must not take the whole poll's inference with it.

    The file parses as JSON, so ``load_state`` hands it over intact; it is the
    entries that are wrong. ``entry.get`` on a string raises AttributeError and
    an entry whose stored payload has no id raises ValueError out of
    ``task_to_record`` — either one, uncaught, loses every other completion in
    the batch. Same failure shape as the malformed ``tasks`` payload that used
    to abort the whole project walk.
    """
    previous = {
        "t1": "not an entry object at all",
        "t2": {"task": {"title": "no id in the stored payload"}},
        "t3": {"task": {"id": "t3", "title": "Gone"}, "last_seen": "2026-08-07T12:00:00+00:00"},
    }
    records = ticktick_api.infer_completions(
        previous, current_ids={"t4"}, now=datetime(2026, 8, 8, tzinfo=UTC)
    )
    assert [r.stable_id for r in records] == ["ticktick:t3"]
    notes = [r for r in caplog.records if "unusable" in r.getMessage()]
    assert notes, "state entries were skipped without telling anyone"
    assert notes[0].levelno >= logging.lastResort.level


def test_default_state_path_respects_xdg(monkeypatch, tmp_path):
    """State, not data or cache: it is regenerable but losing it loses history."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    assert (
        ticktick_api.default_state_path()
        == tmp_path / "aggregator" / "ticktick" / "open_tasks.json"
    )


@pytest.mark.parametrize("unset", ["absent", "empty"])
def test_default_state_path_falls_back_to_the_xdg_default(monkeypatch, unset):
    """Most desktops never export XDG_STATE_HOME, and systemd units export less.

    An empty value is the same case: reading it literally yields a *relative*
    path, so the baseline would land wherever the timer happened to be started
    from and the next run — started elsewhere — would find no state and infer
    nothing, forever.
    """
    if unset == "absent":
        monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    else:
        monkeypatch.setenv("XDG_STATE_HOME", "")
    path = ticktick_api.default_state_path()
    assert path.is_absolute()
    assert path == Path.home() / ".local" / "state" / "aggregator" / "ticktick" / "open_tasks.json"


# --- the state store as a port --------------------------------------------


class _InMemoryState:
    """A state store that is not a file, written against the port only.

    Its existence is the assertion: ``reconcile_open_tasks`` is typed against
    ``OpenTaskState``, so a caller can hand it any object with ``load``/``save``
    — this fake, or a future store — without the inference logic knowing.
    """

    def __init__(self, entries: dict | None = None):
        self.entries = entries or {}
        self.calls: list[str] = []

    def load(self) -> dict[str, dict]:
        self.calls.append("load")
        return self.entries

    def save(self, tasks, now) -> None:
        self.calls.append("save")
        self.entries = {
            str(t["id"]): {"task": t, "last_seen": now.isoformat()} for t in tasks
        }


def test_reconcile_infers_against_the_previous_poll_then_records_this_one():
    """Load, diff, save — in that order, which is the part worth pinning down.

    Save-before-load overwrites the baseline with the current poll, so nothing
    ever looks disappeared and inference is dead. It fails green: no error, no
    warning, just an index that never gains another completion. Sequencing it
    here means task 8 wires up one call instead of three it could order wrongly.
    """
    now = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
    state = _InMemoryState(
        {"t1": {"task": {"id": "t1", "title": "Gone"}, "last_seen": "2026-08-07T12:00:00+00:00"}}
    )
    records = ticktick_api.reconcile_open_tasks(state, [{"id": "t2", "title": "Stays"}], now)
    assert [r.stable_id for r in records] == ["ticktick:t1"]
    assert state.calls == ["load", "save"]
    assert state.entries.keys() == {"t2"}


def test_the_json_file_is_one_adapter_of_the_port(tmp_path):
    """The shipped store: a JSON file, satisfying the port structurally.

    Both it and the in-memory fake are accepted by the same ``isinstance``
    check, which is what makes the protocol a seam rather than decoration.
    """
    state = ticktick_api.JsonFileState(tmp_path / "open_tasks.json")
    assert isinstance(state, ticktick_api.OpenTaskState)
    assert isinstance(_InMemoryState(), ticktick_api.OpenTaskState)

    first = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)
    second = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
    assert ticktick_api.reconcile_open_tasks(state, [{"id": "t1"}, {"id": "t2"}], first) == []
    (rec,) = ticktick_api.reconcile_open_tasks(state, [{"id": "t2"}], second)
    assert rec.stable_id == "ticktick:t1"
    assert rec.extra["provenance"] == "api-inferred-complete"
    assert state.load().keys() == {"t2"}


def test_reconcile_matches_ids_the_way_the_baseline_keys_them(tmp_path):
    """One id rule on both sides of the diff, or every poll invents completions.

    The baseline keys through ``_task_id`` (stripped, ``0`` kept). A caller
    hand-rolling ``str(task["id"])`` for the current set spells a padded id
    differently from its own stored key, so an unchanged task reads as
    disappeared *every single poll* — and a task with no ``id`` key raises
    KeyError and takes the whole poll down, where the baseline merely skips it.
    """
    state = ticktick_api.JsonFileState(tmp_path / "open_tasks.json")
    first = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)
    second = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
    tasks = [{"id": "  t1  ", "title": "Padded"}, {"id": 0, "title": "Zero"}, {"title": "No id"}]
    assert ticktick_api.reconcile_open_tasks(state, tasks, first) == []
    assert ticktick_api.reconcile_open_tasks(state, tasks, second) == []
