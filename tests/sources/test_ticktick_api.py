from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
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
    """59 of 1302 tasks (5 of 238 open) live in an Inbox the listing never returns."""

    def fake_request(method, url, token, timeout=30):
        if url.endswith("/project"):
            return [{"id": "p1", "name": "Work"}]
        return {"tasks": [_open_task(id="t1", projectId="p1")]}

    monkeypatch.setattr(ticktick_api, "_request", fake_request)
    with caplog.at_level(logging.INFO, logger=API_LOG):
        ticktick_api.fetch_open_tasks("tok")
    assert "Inbox" in caplog.text


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


def test_task_to_record_extra_values_are_all_strings():
    """extra is json.dumps'd: an int lands a JSON number where the CSV leg lands text."""
    rec = ticktick_api.task_to_record(
        full_task(dueDate=1785000000000, startDate=None, repeatFlag=7, parentId=42, projectId=9)
    )
    for key, value in rec.extra.items():
        if key == "completed_time_approx":
            continue
        assert isinstance(value, str), f"extra[{key!r}] is {type(value).__name__}"
    assert json.loads(json.dumps(rec.extra))["due_date"] == "1785000000000"


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
