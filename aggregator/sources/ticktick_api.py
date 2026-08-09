"""TickTick Open API client — read-only by construction.

This is the repo's first direct HTTP call; the GitHub source shells out to
`gh` instead. Rather than introduce `requests` or `httpx` for a handful of
GETs, it uses stdlib urllib.

SECURITY: TickTick issues no read-only token — every token carries write
scope. The compensating control is that ``_request`` refuses any method
other than GET *before* it builds a request object, so no code path in this
repo can mutate the user's tasks. It also refuses non-https URLs, so the
bearer token cannot go out in cleartext. Both are tested in
tests/sources/test_ticktick_api.py. The token is never logged.

COVERAGE LIMIT: the Open API filters completed tasks out of every read
endpoint. This module sees open tasks only; completions are inferred here
(by disappearance, task 7) and corrected later from the CSV backup.

Status vocabulary is TickTick's own, shared verbatim with
``ticktick_csv.py`` so the two legs merge cleanly: ``0`` normal, ``2``
completed, ``-1`` abandoned. There is no status ``1``.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from urllib import request
from urllib.error import URLError

from aggregator.sources.base import Record, stable_id_for

log = logging.getLogger(__name__)

BASE_URL = "https://api.ticktick.com/open/v1"
SOURCE_NAME = "ticktick"
DEFAULT_TIMEOUT = 30

# TickTick priority values: 0 none, 1 low, 3 medium, 5 high.
PRIORITY_NAMES = {0: "none", 1: "low", 3: "medium", 5: "high"}

# Status codes, quoted from the backup export's own preamble. An inferred
# completion uses "2" — the same code the CSV leg writes — so task 8 can merge
# API and CSV records without translating between two vocabularies.
STATUS_OPEN = "0"
STATUS_COMPLETED = "2"


class WriteAttemptError(RuntimeError):
    """Raised when any non-GET request is attempted. See module docstring."""


def _request(method: str, url: str, token: str, timeout: int = DEFAULT_TIMEOUT) -> object:
    """Perform one GET and return the decoded JSON body.

    Refuses anything but GET, and anything but https. Both checks run before
    a Request object exists, so there is no window in which a write could be
    issued. Network, HTTP and decode failures propagate: an ingest that
    swallowed them would report "no tasks" for an outage.
    """
    if method != "GET":
        raise WriteAttemptError(
            f"refusing {method} {url}: the ticktick source is read-only by construction"
        )
    if not url.startswith("https://"):
        raise ValueError(f"refusing to send a bearer token to a non-https url: {url}")
    req = request.Request(url, method="GET")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/json")
    with request.urlopen(req, timeout=timeout) as response:  # noqa: S310 - https enforced above
        return json.loads(response.read().decode("utf-8"))


def fetch_open_tasks(token: str, errors: list[str] | None = None) -> list[dict]:
    """Return every currently-open task across all projects.

    A project that fails to fetch is recorded and skipped: one 500 must not
    cost us the other nine projects. The project *listing* is different — if
    that fails we know nothing, so it propagates rather than returning an
    empty list that looks like "you have no tasks".
    """
    projects = _request("GET", f"{BASE_URL}/project", token)
    if not isinstance(projects, list):
        raise ValueError(f"unexpected /project payload: {type(projects).__name__}")

    tasks: list[dict] = []
    for project in projects:
        project_id = project.get("id") if isinstance(project, dict) else None
        if not project_id:
            continue
        try:
            data = _request("GET", f"{BASE_URL}/project/{project_id}/data", token)
            if not isinstance(data, dict):
                raise ValueError(f"unexpected payload: {type(data).__name__}")
        except (URLError, OSError, ValueError) as e:
            # Never interpolate the token into a message that gets logged or
            # surfaced to the CLI.
            log.warning("ticktick project %s fetch failed: %s", project_id, e)
            if errors is not None:
                errors.append(f"ticktick project {project_id}: {e}")
            continue
        for task in data.get("tasks") or []:
            task["_projectName"] = project.get("name", "")
            tasks.append(task)
    return tasks


def task_to_record(
    task: dict,
    *,
    completed_at: datetime | None = None,
    provenance: str = "api",
) -> Record:
    """Map one API task payload to a Record.

    ``completed_at`` is set only for inferred completions, and always alongside
    ``provenance="api-inferred-complete"``.
    """
    inferred = completed_at is not None
    tags = [str(t) for t in (task.get("tags") or [])]
    project_name = task.get("_projectName") or ""
    if project_name:
        tags.append(project_name)
    tags.append("completed" if inferred else "open")

    priority = task.get("priority", 0)
    extra: dict[str, object] = {
        "provenance": provenance,
        "status": STATUS_COMPLETED if inferred else STATUS_OPEN,
        "priority": PRIORITY_NAMES.get(priority, str(priority or "")),
        "due_date": task.get("dueDate") or "",
        "start_date": task.get("startDate") or "",
        "repeat": task.get("repeatFlag") or "",
        "parent_id": task.get("parentId") or "",
        "project_id": task.get("projectId") or "",
    }
    if inferred:
        # Never let an approximate timestamp pass for a real one.
        extra["completed_time_approx"] = True

    return Record(
        stable_id=stable_id_for(SOURCE_NAME, str(task["id"])),
        source=SOURCE_NAME,
        subject=(task.get("title") or "").strip() or str(task["id"]),
        body=task.get("content") or "",
        tags=tags,
        created_at=_parse_api_dt(task.get("createdTime")),
        updated_at=completed_at or _parse_api_dt(task.get("modifiedTime")),
        extra=extra,
    )


def _parse_api_dt(value: object) -> datetime | None:
    """Parse a TickTick API timestamp, normalising to UTC.

    ``store.py`` compares and orders created_at/updated_at as ISO *text*, so a
    surviving foreign offset would sort wrongly — same rule the CSV leg follows.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    for parse in (
        datetime.fromisoformat,
        lambda v: datetime.strptime(v, "%Y-%m-%dT%H:%M:%S%z"),
        lambda v: datetime.strptime(v, "%Y-%m-%dT%H:%M:%S.%f%z"),
    ):
        try:
            parsed = parse(value.strip())
        except ValueError:
            continue
        return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    log.warning("unparseable ticktick api timestamp: %r", value)
    return None


def resolve_token(token: str | None, token_file: str | None) -> str | None:
    """Return the bearer token, or None when the API leg should be skipped.

    An absent or empty token is a supported state, not an error: the source
    falls back to CSV-only. That includes a present-but-empty secret file,
    which is exactly how an agenix secret looks before the user has populated
    it. An unreadable token FILE is different — the user asked for the API leg
    and the secret is broken — so that raises.
    """
    if token:
        return token.strip() or None
    if token_file:
        from pathlib import Path

        content = Path(token_file).read_text(encoding="utf-8").strip()
        return content or None
    return None
