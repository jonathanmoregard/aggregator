"""TickTick Open API client — read-only by construction.

This is the repo's first direct HTTP call; the GitHub source shells out to
`gh` instead. Rather than introduce `requests` or `httpx` for a handful of
GETs, it uses stdlib urllib.

SECURITY: TickTick issues no read-only token — every token carries write
scope. Three compensating controls, all tested in
tests/sources/test_ticktick_api.py:

* ``_request`` refuses any method other than GET *before* it builds a request
  object, so no code path in this repo can mutate the user's tasks.
* It refuses a non-https URL, so the token is never sent in cleartext.
* The token goes on as an *unredirected* header and the module's opener
  refuses a non-https redirect target. urllib's default opener copies
  ``req.headers`` onto a redirected request and permits an https->http
  downgrade, so ``add_header`` alone would let a 30x walk this write-scoped
  bearer token onto a plaintext host — measured, not theoretical.

The token is never logged.

COVERAGE LIMITS — two of them, both structural:

1. The Open API filters completed tasks out of every read endpoint. This
   module sees open tasks only; completions are inferred here (by
   disappearance, task 7) and corrected later from the CSV backup. The
   payload's own ``status`` is still read and trusted over that assumption, so
   if the endpoint ever does serve a completed task the record says so and a
   warning fires, instead of the task being silently resurrected as open.
2. ``GET /open/v1/project`` does not list the Inbox, so Inbox tasks are never
   fetched and can never "disappear" for task 7's completion inference either.
   Measured blast radius on the user's real export: 59 of 1302 tasks, 5 of the
   238 currently open. ``fetch_open_tasks`` *warns* when the listing comes back
   without an Inbox — at WARNING specifically, because nothing under
   ``aggregator/`` configures logging, so ``logging.lastResort`` (level WARNING)
   is the only thing that prints and anything quieter reaches nobody. A first
   real-token run therefore reports a task count that can be compared against
   the backup without the gap looking like a bug.

VOCABULARY: the status codes (``0`` normal, ``2`` completed, ``-1``
abandoned — there is no ``1``), the status->tag mapping, the priority names and
the canonical date spelling are all *imported* from ``ticktick_csv.py`` rather
than restated here. They live there because the backup export documents them in
its own preamble, and a second hand-kept copy is exactly how the two legs would
drift into writing different words for the same value once task 8 merges them
by stable_id.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from urllib import request
from urllib.error import URLError
from urllib.parse import quote

from aggregator.sources.base import Record, stable_id_for

# One vocabulary, defined once, in the module whose file format documents it.
# An inferred completion therefore writes the very code the CSV leg writes, a
# priority becomes the very word the CSV leg writes, and a due date is spelled
# the way the CSV leg spells it, so task 8 can merge API and CSV records without
# translating between two dialects.
from aggregator.sources.ticktick_csv import (
    STATUS_COMPLETED,
    STATUS_OPEN,
    STATUS_TAGS,
    normalize_date_text,
    priority_name,
    status_tag,
)

log = logging.getLogger(__name__)

BASE_URL = "https://api.ticktick.com/open/v1"
SOURCE_NAME = "ticktick"
DEFAULT_TIMEOUT = 30

# The only place this module spells a timestamp field name.
#
# UNVERIFIED. TickTick's *documented* Task object is `id projectId title
# isAllDay completedTime content desc dueDate items priority reminders
# repeatFlag sortOrder startDate status timeZone` — ``createdTime`` and
# ``modifiedTime`` may well not be in it at all, and that could not be confirmed
# (every attempt to read the published docs was blocked, and there is no other
# web access). So the names are not guessed at harder: ``_warn_no_timestamps``
# makes the truth self-reporting on first contact with a real token instead.
#
# ``task_to_record`` reads these two constants and nothing else, which is the
# point: the tripwire's message tells an operator to correct the names here, and
# a second hand-kept copy inside ``task_to_record`` meant following that advice
# silenced the warning while leaving every record dateless — trading the loud
# failure back for the silent one.
CREATED_FIELD = "createdTime"
UPDATED_FIELDS = ("completedTime", "modifiedTime")  # ``updated_at`` preference order

# Derived, not a third copy. Every field a date may come from, in preference
# order; used by the tripwire and by the tests.
TIMESTAMP_FIELDS = (*UPDATED_FIELDS, CREATED_FIELD)


class WriteAttemptError(RuntimeError):
    """Raised when any non-GET request is attempted. See module docstring."""


class TokenUnavailableError(OSError):
    """The API leg was configured but its secret could not be read.

    Subclasses OSError so the documented contract — an unreadable token *file*
    raises — still holds for existing callers, but is its own type so task 8 can
    tell "the secret is broken" apart from "TickTick is down". Both otherwise
    arrive as a bare OSError and an ``except OSError`` around the whole API leg
    would conflate the two.
    """


class _HttpsOnlyRedirectHandler(request.HTTPRedirectHandler):
    """Refuse a redirect that would leave https. See the SECURITY note."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: N802 - stdlib API
        if not newurl.lower().startswith("https://"):
            raise URLError(f"refusing a {code} redirect to a non-https url: {newurl}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


# Module-private opener rather than install_opener(): this must not change the
# redirect behaviour of every other urllib user in the process.
_OPENER = request.build_opener(_HttpsOnlyRedirectHandler())


def _open(req: request.Request, timeout: int):
    """The module's single network seam. Tests patch this, not urllib's globals."""
    return _OPENER.open(req, timeout=timeout)


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
    # add_unredirected_header, not add_header: urllib copies req.headers onto a
    # redirected request but never the unredirected ones, so a 30x cannot carry
    # this write-scoped bearer token to another host.
    req.add_unredirected_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/json")
    with _open(req, timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _note(errors: list[str] | None, message: str) -> None:
    """Log a fault and, when the caller is collecting them, surface it too."""
    log.warning("%s", message)
    if errors is not None:
        errors.append(message)


def _project_tasks(
    data: dict, project_name: str, errors: list[str] | None, label: str
) -> list[dict]:
    """Return the well-formed task objects in one ``/project/{id}/data`` payload.

    Every shape assumption is checked here rather than in the caller's loop: a
    ``tasks`` value that is a string, or a list of strings, used to raise an
    uncaught TypeError from ``task["_projectName"] = ...`` and kill the whole
    walk — one surprising project costing us all the others, which is exactly
    what the per-project error sink exists to prevent.
    """
    raw = data.get("tasks")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"unexpected tasks payload: {type(raw).__name__}")
    tasks = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        entry["_projectName"] = project_name
        tasks.append(entry)
    if len(tasks) != len(raw):
        _note(errors, f"{label}: skipped {len(raw) - len(tasks)} non-object task entr(ies)")
    return tasks


def fetch_open_tasks(token: str, errors: list[str] | None = None) -> list[dict]:
    """Return every currently-open task across all projects.

    A project that fails to fetch is recorded and skipped: one 500 must not
    cost us the other nine projects. The project *listing* is different — if
    that fails we know nothing, so it propagates rather than returning an
    empty list that looks like "you have no tasks".

    Two coverage facts are reported rather than left to be inferred from a
    surprising count: the Inbox is not in the listing (see the module
    docstring), and a batch that yields no parseable timestamp at all is an
    error, not a quiet success.
    """
    projects = _request("GET", f"{BASE_URL}/project", token)
    if not isinstance(projects, list):
        raise ValueError(f"unexpected /project payload: {type(projects).__name__}")

    tasks: list[dict] = []
    names: list[str] = []
    for project in projects:
        project_id = project.get("id") if isinstance(project, dict) else None
        if not project_id:
            continue
        project_name = project.get("name") or ""
        names.append(str(project_name))
        label = f"ticktick project {project_id}"
        try:
            url = f"{BASE_URL}/project/{quote(str(project_id), safe='')}/data"
            data = _request("GET", url, token)
            if not isinstance(data, dict):
                raise ValueError(f"unexpected payload: {type(data).__name__}")
            tasks.extend(_project_tasks(data, str(project_name), errors, label))
        except (URLError, OSError, ValueError) as e:
            # Never interpolate the token into a message that gets logged or
            # surfaced to the CLI.
            _note(errors, f"{label}: {e}")
            continue

    _note_inbox_gap(names, len(tasks))
    _warn_no_timestamps(tasks, errors)
    return tasks


def _note_inbox_gap(project_names: list[str], task_count: int) -> None:
    """Say out loud that the Inbox is missing from the listing, when it is.

    Self-checking rather than asserted: if TickTick ever does list the Inbox,
    this goes quiet on its own instead of documenting a limit that no longer
    exists. Deliberately a log line and not an ``errors`` entry — a permanent,
    known coverage limit that appended an error on every healthy poll would fire
    a CRITICAL desktop notification every time the timer runs, and an alert that
    always fires is an alert nobody reads.

    WARNING, not INFO, and that is load-bearing. Nothing under ``aggregator/``
    calls ``basicConfig``/``dictConfig``/``addHandler``, so the root logger has
    no handlers and ``logging.lastResort`` — level WARNING — is what prints. At
    INFO this line produced literally zero output, which is the same failure
    shape as not logging it at all.
    """
    if any(name.strip().lower() == "inbox" for name in project_names):
        return
    log.warning(
        "ticktick api: walked %d project(s) for %d task(s); no Inbox in the listing, so "
        "Inbox tasks are out of scope for both this poll and task 7's completion inference "
        "(59 of 1302 tasks, 5 of 238 open, in the reference backup)",
        len(project_names),
        task_count,
    )


def _warn_no_timestamps(tasks: list[dict], errors: list[str] | None) -> None:
    """Fail loudly when a non-empty batch carries no parseable timestamp at all.

    An API leg that produces 100% null timestamps is indistinguishable from one
    working perfectly — the index just quietly loses every API record from its
    date filters, and under task 8's merge a null ``created_at`` overwrites the
    real one the CSV leg parsed. That is the "empty result looks like success"
    shape the fail-loudly constraint forbids.

    The field names are unverified against TickTick's published Task object
    (see the comment on ``CREATED_FIELD``). This is how that gets settled: the
    first run against a real token either parses timestamps or names, in the
    errors list, exactly which fields it looked under and found nothing.
    """
    if not tasks:
        return
    dated = sum(
        1
        for task in tasks
        if any(_parse_api_dt(task.get(f), warn=False) is not None for f in TIMESTAMP_FIELDS)
    )
    if dated:
        return
    _note(
        errors,
        f"ticktick api: {len(tasks)} task(s) returned and not one carried a parseable "
        f"timestamp in any of {', '.join(TIMESTAMP_FIELDS)} — every API record would have a "
        "null created_at/updated_at, which is invisible in the index. These field names are "
        "unverified against TickTick's published Task object: dump a raw payload and correct "
        "CREATED_FIELD / UPDATED_FIELDS in aggregator/sources/ticktick_api.py — those two are "
        "the only copy, so correcting them fixes the records as well as this warning",
    )


def task_to_record(
    task: dict,
    *,
    completed_at: datetime | None = None,
    provenance: str = "api",
) -> Record:
    """Map one API task payload to a Record.

    ``completed_at`` is set only for inferred completions, and always alongside
    ``provenance="api-inferred-complete"``. Absent that — i.e. on every task the
    API actually served — the status tag comes from the payload's own ``status``
    whenever it has one, never from the mere presence of a completed timestamp
    (the rule the CSV leg follows), because an inference that overrode a vendor
    field is the defect this repo already shipped once. An inferred completion
    is the one case that does not consult the payload; see ``_payload_status``
    for why that is safe.

    KNOWN MERGE DIVERGENCE: the CSV leg also tags with ``Folder Name``. The Open
    API's project object carries a ``groupId`` but no group name and there is no
    documented endpoint to resolve one, so an API record cannot reproduce that
    tag. Task 8 should union tags across the two legs rather than let the API
    record's tag list replace the CSV record's.
    """
    task_id = _task_id(task)  # first, so an unusable payload fails before it warns
    inferred = completed_at is not None
    status = _payload_status(task, inferred=inferred)
    tags = _task_tags(task.get("tags"))
    project_name = _text(task.get("_projectName")).strip()
    if project_name:
        tags.append(project_name)
    tags.append(status_tag(status, logger=log))

    # Every value here is a str, with no exceptions — the annotation says so, so
    # the next exception has to argue with a type-checker. extra is serialised
    # with json.dumps, and a non-string would land a JSON number or literal where
    # the CSV leg lands text, so an exact-match DSL filter would miss it.
    extra: dict[str, str] = {
        "provenance": provenance,
        "status": status,
        "priority": priority_name(task.get("priority"), logger=log),
        "due_date": normalize_date_text(task.get("dueDate"), logger=log),
        "start_date": normalize_date_text(task.get("startDate"), logger=log),
        "repeat": _text(task.get("repeatFlag")),
        "parent_id": _text(task.get("parentId")),
        "project_id": _text(task.get("projectId")),
    }
    if inferred:
        # Never let an approximate timestamp pass for a real one. Spelled
        # "true", not True: a bool serialises to the JSON literal `true`, so a
        # filter would have to match that rather than the text every other value
        # in extra uses.
        extra["completed_time_approx"] = "true"

    created = _parse_api_dt(task.get(CREATED_FIELD), field=CREATED_FIELD)
    return Record(
        stable_id=stable_id_for(SOURCE_NAME, task_id),
        source=SOURCE_NAME,
        subject=_text(task.get("title")).strip() or task_id,
        body=_task_body(task),
        tags=tags,
        created_at=created,
        updated_at=completed_at or _first_dt(task, UPDATED_FIELDS) or created,
        extra=extra,
    )


def _first_dt(task: dict, fields: tuple[str, ...]) -> datetime | None:
    """The first parseable timestamp among ``fields``, in order, or None.

    Driven off ``UPDATED_FIELDS`` rather than a hand-written ``a or b`` chain so
    that the field names live in exactly one place — see ``CREATED_FIELD``.
    """
    for field in fields:
        parsed = _parse_api_dt(task.get(field), field=field)
        if parsed is not None:
            return parsed
    return None


def _task_id(task: dict) -> str:
    """Return the payload's task id, or raise.

    ``str(task["id"])`` turned a null id into the plausible-looking literal
    ``"None"``, which sails past ``stable_id_for``'s empty-id guard and collapses
    every null-id task in a batch onto one record. Validated here, in the public
    entry point, rather than in a caller's ``if not task.get("id")`` — which
    would also silently drop the legitimate id ``0``.
    """
    raw = task.get("id")
    task_id = "" if raw is None else str(raw).strip()
    if not task_id:
        raise ValueError(f"ticktick task payload has no usable id: {raw!r}")
    return task_id


def _text(value: object) -> str:
    """Coerce a payload value to text. Never returns None, never raises.

    TickTick's own client will not send an int title, but a payload is untrusted
    input: ``(task.get("title") or "").strip()`` raised AttributeError on one,
    and a dict ``content`` reached ``scrub()`` in the store as a dict.
    """
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _task_tags(value: object) -> list[str]:
    """Normalise the payload's ``tags`` into a list of non-empty strings.

    A bare string was iterated per character — ``"work"`` became
    ``['w','o','r','k']`` — which is a live confusion, not a hypothetical: the
    CSV leg's ``Tags`` column *is* a comma-joined string, so it is exactly the
    shape someone would hand this function.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [t.strip() for t in value.split(",") if t.strip()]
    if isinstance(value, (list, tuple, set)):
        return [_text(t).strip() for t in value if _text(t).strip()]
    log.warning("unexpected ticktick tags payload of type %s; ignoring", type(value).__name__)
    return []


def _task_body(task: dict) -> str:
    """Everything free-text the CSV leg's ``Content`` would carry.

    The API splits what the export flattens: the note is in ``desc`` and a
    checklist is in ``items[]``. Dropping them made a task's checklist text
    unsearchable the moment task 8's merge let a fresher API observation replace
    the CSV record it had been searchable from.

    The export's exact spelling of a checklist is unmeasured — all 1302 rows of
    the reference backup have ``Is Check list = N`` — so this aims at
    searchability, not byte-equality with a CSV body.
    """
    parts = [_text(task.get("content")), _text(task.get("desc"))]
    items = task.get("items")
    if isinstance(items, list):
        for item in items:
            title = _text(item.get("title") if isinstance(item, dict) else item).strip()
            if title:
                parts.append(f"- {title}")
    elif items is not None:
        log.warning("unexpected ticktick items payload of type %s; ignoring", type(items).__name__)
    return "\n".join(p for p in parts if p.strip())


def _payload_status(task: dict, *, inferred: bool) -> str:
    """The status code to record: the payload's own, unless we inferred one.

    ``inferred`` short-circuits: an inferred completion returns
    ``STATUS_COMPLETED`` without reading the payload at all. That is not the
    override this function exists to prevent — task 7 only infers a completion
    for a task that *was* in the open batch and has since disappeared, so the
    payload in hand is the stale open one and its ``status`` is ``0`` by
    construction. Every task the API actually served takes the branch below and
    is recorded as the payload says.

    Reading the payload was the whole point. The previous version derived status
    purely from whether the *caller* passed ``completed_at``, so a payload that
    said ``"status": 2`` was recorded as open — and task 8 lets the fresher API
    observation beat the CSV row, so a task the backup correctly recorded as
    completed would be resurrected as open with the CSV evidence gone.

    A non-open status also contradicts coverage limit 1, so it warns: either the
    endpoint's behaviour changed or the limit was never true, and both are
    things the operator needs told.
    """
    if inferred:
        return STATUS_COMPLETED
    raw = task.get("status")
    text = "" if raw is None else str(raw).strip()
    if not text:
        return STATUS_OPEN
    if text != STATUS_OPEN and text in STATUS_TAGS:
        log.warning(
            "ticktick api returned a task with status %r (%s); the module's coverage limit "
            "assumes the Open API only serves open tasks. Trusting the payload.",
            text,
            STATUS_TAGS[text],
        )
    return text  # an unlisted code is warned about by status_tag and kept verbatim


def _parse_api_dt(value: object, *, field: str = "timestamp", warn: bool = True) -> datetime | None:
    """Parse a TickTick API timestamp, normalising to UTC.

    ``store.py`` compares and orders created_at/updated_at as ISO *text*, so a
    surviving foreign offset would sort wrongly — same rule the CSV leg follows.

    An absent value is normal (an open task has no ``completedTime``) and is
    quiet. A *present* value that is not a string — an epoch int, say — used to
    return None with no diagnostic at all; it now warns, because a timestamp
    silently becoming null is the failure this module cannot afford to hide.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        if warn:
            log.warning(
                "ticktick api %s is a %s, not a timestamp string: %r; recording no date",
                field,
                type(value).__name__,
                value,
            )
        return None
    text = value.strip()
    if not text:
        return None
    for parse in (
        datetime.fromisoformat,
        lambda v: datetime.strptime(v, "%Y-%m-%dT%H:%M:%S%z"),
        lambda v: datetime.strptime(v, "%Y-%m-%dT%H:%M:%S.%f%z"),
    ):
        try:
            parsed = parse(text)
        except ValueError:
            continue
        return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    if warn:
        log.warning("unparseable ticktick api %s: %r", field, value)
    return None


def resolve_token(token: str | None, token_file: str | None) -> str | None:
    """Return the bearer token, or None when the API leg should be skipped.

    An absent or empty token is a supported state, not an error: the source
    falls back to CSV-only. That includes a present-but-empty secret file,
    which is exactly how an agenix secret looks before the user has populated
    it. An unreadable token FILE is different — the user asked for the API leg
    and the secret is broken — so that raises :class:`TokenUnavailableError`.

    A whitespace-only explicit token falls through to the file rather than
    shadowing it: treating it as "configured" silently skipped the API leg while
    a perfectly good token file sat right there.
    """
    explicit = (token or "").strip()
    if explicit:
        return explicit
    if token_file:
        from pathlib import Path

        try:
            content = Path(token_file).read_text(encoding="utf-8").strip()
        except OSError as e:
            raise TokenUnavailableError(
                f"ticktick token file is unreadable: {token_file} ({e})"
            ) from e
        return content or None
    return None
