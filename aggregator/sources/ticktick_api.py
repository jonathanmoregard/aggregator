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
   module sees open tasks only; completions are inferred here — by diffing each
   poll against the previous one, which is what the open-task state file at the
   bottom of this module exists to remember — and corrected later from the CSV
   backup. The
   payload's own ``status`` is still read and trusted over that assumption, so
   if the endpoint ever does serve a completed task the record says so and a
   warning fires, instead of the task being silently resurrected as open.
2. ``GET /open/v1/project`` does not list the Inbox, so Inbox tasks are never
   fetched and can never "disappear" for task 7's completion inference either.
   Measured blast radius on the user's real export: 59 of 1302 tasks, 5 of the
   238 currently open. ``poll_open_tasks`` *warns* when the listing comes back
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
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable
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


class StateUnreadableError(RuntimeError):
    """The open-task baseline exists on disk and could not be read.

    Its own type, and NOT an ``OSError``, so no ``except OSError`` anywhere in
    the ingest path can quietly re-absorb it into "the filesystem hiccuped".
    The one thing this must never become again is an empty baseline: see
    :func:`load_state`.
    """


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


@dataclass(frozen=True)
class OpenTaskPoll:
    """One poll of the Open API, and whether it saw ALL of the open tasks.

    The two travel together on purpose. ``complete`` is not decoration: the
    completion inference at the bottom of this module reads "task was in the
    baseline, is absent now" as "task was finished", so a poll that merely
    FAILED to look at one project is indistinguishable from one that watched
    every task in it get ticked off. Handing the caller a bare ``list[dict]``
    made that distinction something a caller had to remember to ask about, and
    the source did not — a single 500 on one project marked every open task in
    it completed, permanently, because the Open API only ever serves open tasks
    and can therefore never contradict the inference. Only a manual CSV export
    could.

    ``complete is False`` means "some of what is open MAY not have been
    observed", from any cause: an empty project listing, a project whose fetch
    failed, a project the listing gave with no usable id, a project whose
    payload carried no ``tasks`` key at all, or task entries dropped for being
    the wrong shape. The bar is deliberately "may": the flag guards an
    irreversible act, so anything this module cannot positively tell apart from
    a missed task counts as one.

    It does NOT cover the Inbox, which ``GET /open/v1/project`` never lists at
    all — that gap is permanent, known and reported separately by
    ``_note_inbox_gap``; folding it in here would mean ``complete`` was False on
    every healthy poll and inference would be dead forever. Nor does a ``tasks``
    key that is present and EMPTY, which is a project answering "nothing open
    here" and is a complete view of it.
    """

    tasks: list[dict]
    complete: bool


# Payload keys that would mean "there is more where this came from".
#
# UNVERIFIED, and a tripwire rather than a guess acted upon. TickTick's
# ``/project/{id}/data`` is documented as a single ``{project, tasks, columns}``
# object with no cursor, but that could not be confirmed — every attempt to read
# the published docs was blocked and there is no other web access — and this is
# the same situation as ``CREATED_FIELD``, handled the same way: make the truth
# self-reporting on first contact with a real token instead of guessing harder.
#
# Deliberately a SHORT list of unambiguous "there is more" names. ``total``,
# ``page``, ``limit`` and ``offset`` are omitted on purpose: benign metadata
# under any of those names would clear ``complete`` on every healthy poll and
# kill completion inference outright, which is the failure mode the Inbox note
# is careful to avoid.
_PAGING_KEYS = frozenset({"nextpagetoken", "nextcursor", "hasmore", "nextpage"})


def _paging_signals(data: dict) -> list[str]:
    """The truthy pagination-cursor keys in a project payload, if any.

    Truthy only: a ``"hasMore": false`` is the endpoint explicitly saying this
    IS the whole list, which is the opposite of a fault.
    """
    return sorted(
        str(key)
        for key, value in data.items()
        if str(key).lower().replace("_", "") in _PAGING_KEYS and value
    )


def _project_tasks(
    data: dict, project_name: str, errors: list[str] | None, label: str
) -> tuple[list[dict], bool]:
    """The well-formed task objects in one ``/project/{id}/data`` payload.

    Returns them alongside whether the payload was fully understood. Every
    shape assumption is checked here rather than in the caller's loop: a
    ``tasks`` value that is a string, or a list of strings, used to raise an
    uncaught TypeError from ``task["_projectName"] = ...`` and kill the whole
    walk — one surprising project costing us all the others, which is exactly
    what the per-project error sink exists to prevent.

    A dropped entry makes the poll INCOMPLETE, not merely noisy: the dropped
    task is open, it is now missing from the poll, and that is the exact input
    the completion inference misreads.

    So does an ABSENT ``tasks`` key, which is the distinction this function
    turns on. See below.
    """
    raw = data.get("tasks")
    if raw is None:
        # NOT "this project has nothing open". A project with nothing in it
        # answers ``"tasks": []`` — a key that is PRESENT and empty, which falls
        # through to the loop below and is a complete, understood view of that
        # project. An ABSENT key is the payload declining to say anything about
        # this project's tasks at all, and the causes are schema drift, a
        # partial 200, pagination this client does not implement, and a project
        # the token's scope will not fully serve. Nothing here can tell those
        # apart from one another, and — the point — nothing here can tell any of
        # them from a genuinely empty project either, because the one signal
        # that would (an empty list) is exactly what is missing. Undecidable, so
        # the answer has to be the safe one.
        #
        # Reported complete, this armed the precise input the completion
        # inference misreads: every open task in the project is absent from the
        # poll, gets recorded as finished, and the baseline advances past it.
        # The Open API serves open tasks only and can therefore never contradict
        # that; only a manual CSV export could.
        _note(
            errors,
            f"{label}: the payload carried no 'tasks' key at all, so this "
            f"project's open tasks were NOT observed. Treating the poll as "
            f"incomplete rather than as a project with nothing open, which "
            f"would infer a completion for every open task in it. An empty "
            f"project answers with an empty 'tasks' LIST; if TickTick ever "
            f"starts omitting the key for empty projects instead, this fires "
            f"on every healthy poll and the check in _project_tasks is what "
            f"has to change",
        )
        return [], False
    if not isinstance(raw, list):
        raise ValueError(f"unexpected tasks payload: {type(raw).__name__}")
    tasks = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        entry["_projectName"] = project_name
        tasks.append(entry)
    complete = True
    if len(tasks) != len(raw):
        _note(errors, f"{label}: skipped {len(raw) - len(tasks)} non-object task entr(ies)")
        complete = False
    cursors = _paging_signals(data)
    if cursors:
        # A PAGED payload is the one way this project's task list can be short
        # without anything above noticing: the key is present, the value is a
        # list, every entry is a well-formed object, and it is still only the
        # first page. That is HIGH 1's exact damage — unseen open tasks inferred
        # completed — arriving through a channel the absent-key check cannot
        # see, because a truncated list is indistinguishable from a short one.
        _note(
            errors,
            f"{label}: the payload carries {', '.join(cursors)}, which means "
            f"this client is reading only the FIRST PAGE of the project's "
            f"tasks and the rest were not observed. Treating the poll as "
            f"incomplete; completion inference stays off until "
            f"_project_tasks learns to follow the cursor",
        )
        complete = False
    return tasks, complete


def poll_open_tasks(token: str, errors: list[str] | None = None) -> OpenTaskPoll:
    """Every currently-open task across all projects, plus a coverage verdict.

    A project that fails to fetch is recorded and skipped: one 500 must not
    cost us the other nine projects. The project *listing* is different — if
    that fails we know nothing, so it propagates rather than returning an
    empty list that looks like "you have no tasks".

    Anything that costs the walk a task it should have seen also clears
    :attr:`OpenTaskPoll.complete`, so the caller can keep the partial records
    (they are genuine, fresh observations of open tasks) while refusing to run
    the completion inference over them. See :class:`OpenTaskPoll`.

    Two coverage facts are reported rather than left to be inferred from a
    surprising count: the Inbox is not in the listing (see the module
    docstring), and a batch that yields no parseable timestamp at all is an
    error, not a quiet success. Neither clears ``complete`` — the Inbox gap is
    permanent and known, and a dateless batch is a field-naming fault, not a
    missing task.
    """
    projects = _request("GET", f"{BASE_URL}/project", token)
    if not isinstance(projects, list):
        raise ValueError(f"unexpected /project payload: {type(projects).__name__}")

    tasks: list[dict] = []
    names: list[str] = []
    complete = True
    if not projects:
        # An empty listing is the purest form of the same defect: nothing was
        # observed, so with ``complete`` True every task in the baseline reads
        # as finished at once, the baseline advances past all of them, and the
        # run exits 0. And it is not distinguishable from a token whose scope
        # stopped covering the user's projects, which answers 200 with [] rather
        # than 401.
        #
        # The cost of choosing incomplete: an account whose only tasks live in
        # the Inbox — which ``GET /open/v1/project`` never lists (see the module
        # docstring) — now reports this on every poll. It has nothing to infer
        # either way, because a poll that walks no projects never puts anything
        # in the baseline to diff against, so the choice costs that account
        # noise and costs everyone else a permanent, uncontradictable data loss.
        _note(
            errors,
            "ticktick api: /project returned an empty listing, so this poll "
            "observed no projects and therefore no open tasks. Not treated as "
            "a complete view: it is indistinguishable from a token whose scope "
            "no longer covers them, and calling it complete would record every "
            "task in the open-task baseline as completed in one go",
        )
        complete = False
    for project in projects:
        project_id = project.get("id") if isinstance(project, dict) else None
        if not project_id:
            # Its tasks are open and are about to be missing from this poll,
            # which is the completion inference's input. Skipped, but never as
            # a silent success: recorded, and the poll is no longer a complete
            # view of what is open.
            _note(errors, f"ticktick project listing entry has no usable id: {project!r}")
            complete = False
            continue
        project_name = project.get("name") or ""
        names.append(str(project_name))
        label = f"ticktick project {project_id}"
        try:
            url = f"{BASE_URL}/project/{quote(str(project_id), safe='')}/data"
            data = _request("GET", url, token)
            if not isinstance(data, dict):
                raise ValueError(f"unexpected payload: {type(data).__name__}")
            found, whole = _project_tasks(data, str(project_name), errors, label)
            tasks.extend(found)
            complete = complete and whole
        except (URLError, OSError, ValueError) as e:
            # Never interpolate the token into a message that gets logged or
            # surfaced to the CLI.
            _note(errors, f"{label}: {e}")
            complete = False
            continue

    _note_inbox_gap(names, len(tasks))
    _warn_no_timestamps(tasks, errors)
    return OpenTaskPoll(tasks=tasks, complete=complete)


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
    errors: list[str] | None = None,
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

    ``errors`` is the run's fault sink, forwarded to ``status_tag`` so a status
    code neither leg recognises makes the run exit non-zero rather than being
    quietly filed as open.
    """
    task_id = _task_id(task)  # first, so an unusable payload fails before it warns
    inferred = completed_at is not None
    status = _payload_status(task, inferred=inferred)
    tags = _task_tags(task.get("tags"))
    project_name = _text(task.get("_projectName")).strip()
    if project_name:
        tags.append(project_name)
    tags.append(status_tag(status, logger=log, errors=errors))

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


def _read_env_file(path: str) -> dict[str, str]:
    """Parse a ``KEY=VALUE`` env file, or return {} when there isn't one.

    Deliberately mirrors ``~/.claude/todo/_envfile.py`` and the todo backend's
    own reader — same comment handling, same quote stripping. A reader that
    disagreed by one quote character would hand out a token that 401s, which
    reads as an expired credential and sends the operator to re-authorize
    something that was never broken.
    """
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError:
        return {}
    values: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


# The credential store shared with ``~/.claude/todo`` and router-agent's
# add_task MCP. That backend owns the OAuth dance and writes each refreshed
# access token back here, so this is where the live token is by definition;
# a private copy would go stale the moment it refreshed.
DEFAULT_ENV_FILE = os.path.expanduser("~/.config/todo/env")

TOKEN_ENV_VAR = "TICKTICK_ACCESS_TOKEN"
EXPIRY_ENV_VAR = "TICKTICK_TOKEN_EXPIRES_AT"

# What a human must run to fix an expired token. This leg runs unattended from
# a timer, so it cannot do the browser re-authorization itself.
RELOGIN_COMMAND = "~/.claude/todo-add --login"


def _reject_if_expired(expires_at: str | None, source: str) -> None:
    """Raise when the accompanying expiry says the token is already dead.

    Only advisory metadata: an absent or unparseable expiry is not an error,
    because a token that still works must not be thrown away over a garbled
    sidecar value. A *definitely* expired one is worth stopping for — otherwise
    the failure surfaces as a 401 partway through the project walk, after the
    run has already reported progress.
    """
    if not expires_at:
        return
    try:
        deadline = int(expires_at)
    except ValueError:
        return
    if deadline and deadline <= time.time():
        raise TokenUnavailableError(
            f"ticktick access token from {source} expired at "
            f"{datetime.fromtimestamp(deadline, UTC).isoformat()}; "
            f"re-authorize with `{RELOGIN_COMMAND}`"
        )


def _reject_if_blank(value: str, source: str) -> None:
    """Raise when a token source exists but holds nothing.

    "The key is not there" and "the key is there and empty" are opposite
    diagnoses with opposite fixes, and collapsing them into ``None`` made a
    broken OAuth refresh look exactly like a machine that never configured the
    API leg. Only the second one is a fault, and only a human can clear it.
    """
    if value.strip():
        return
    raise TokenUnavailableError(
        f"ticktick access token from {source} is set but empty — the API leg "
        f"is configured and its credential is gone (this is what a failed "
        f"OAuth refresh leaves behind). Re-authorize with "
        f"`{RELOGIN_COMMAND}`, or unset it if the API leg is not wanted."
    )


def resolve_token(
    token: str | None, token_file: str | None, env_file: str | None = None
) -> str | None:
    """Return the bearer token, or None when the API leg should be skipped.

    Sources, in order: an explicit token, a token *file* (an agenix secret),
    the process environment, then the shared ``~/.config/todo/env`` store that
    the todo backend keeps current.

    An absent or empty token is a supported state, not an error: the source
    falls back to CSV-only. An unreadable token FILE is different — the user
    asked for the API leg and named a secret that is broken — so that raises
    :class:`TokenUnavailableError`. An *empty* token file is neither: that is
    exactly how the merged ``ticktick-api-token.age`` placeholder looks, and
    short-circuiting on it would let the placeholder shadow a live credential
    sitting in the shared store.

    A whitespace-only explicit token falls through to the file rather than
    shadowing it: treating it as "configured" silently skipped the API leg while
    a perfectly good token file sat right there.

    A ``TICKTICK_ACCESS_TOKEN`` that is PRESENT BUT EMPTY — in the process
    environment or in the shared store — is a broken credential, not an absent
    one, and raises. That is precisely what a failed OAuth refresh leaves
    behind: the todo backend rewrites ``~/.config/todo/env`` on every refresh,
    so a refresh that produced nothing writes the key with no value. Returning
    None for it made a dead API leg identical to a machine that never
    configured one — reported, if at all, at a log level nothing prints — so
    the leg silently disabled itself and the run still exited 0.

    The token FILE keeps the opposite rule and that asymmetry is deliberate:
    an empty file is exactly how the merged ``ticktick-api-token.age``
    placeholder looks, so treating it as broken would let the placeholder
    shadow a live credential in the shared store.
    """
    explicit = (token or "").strip()
    if explicit:
        return explicit
    if token_file:
        try:
            content = Path(token_file).read_text(encoding="utf-8").strip()
        except OSError as e:
            raise TokenUnavailableError(
                f"ticktick token file is unreadable: {token_file} ({e})"
            ) from e
        if content:
            return content
    raw_env = os.environ.get(TOKEN_ENV_VAR)
    if raw_env is not None:
        _reject_if_blank(raw_env, f"${TOKEN_ENV_VAR}")
        _reject_if_expired(os.environ.get(EXPIRY_ENV_VAR), f"${TOKEN_ENV_VAR}")
        return raw_env.strip()
    path = env_file or DEFAULT_ENV_FILE
    values = _read_env_file(path)
    if TOKEN_ENV_VAR not in values:
        return None
    shared = values[TOKEN_ENV_VAR]
    _reject_if_blank(shared, f"{TOKEN_ENV_VAR} in {path}")
    _reject_if_expired(values.get(EXPIRY_ENV_VAR), path)
    return shared.strip()


# --- the open-task state file ----------------------------------------------
#
# WHY THIS EXISTS AT ALL. The Open API serves open tasks and nothing else, and
# no endpoint of it reports a completion. So a task that was in the previous
# poll and is missing from this one has either been completed or deleted, and
# the API will never say which. Prior state written to disk is the only thing
# there is to diff against: without it a finished task simply stops appearing,
# the index loses it, and an "I did that last week" search comes back empty with
# nothing anywhere explaining the gap. With it, the disappearance becomes an
# explicit, flagged-as-approximate completion record that the CSV backup later
# corrects with the real Completed Time.


@runtime_checkable
class OpenTaskState(Protocol):
    """The port: wherever the previous poll's open-task set is kept.

    A Protocol rather than a base class because structural typing is Python's
    spelling of a port — the JSON file below satisfies it without importing or
    inheriting anything, and so does a test double. Callers depend on this, not
    on the file, so "where the baseline lives" stays one decision made in one
    place.
    """

    def load(self) -> dict[str, dict]:
        """The previous poll's tasks keyed by id, or {} when there is none.

        ``{}`` means "there is no baseline" and nothing else. An implementation
        that HAS a baseline it cannot read must raise
        :class:`StateUnreadableError` rather than answer ``{}`` — the caller
        turns ``{}`` into "advance the baseline from scratch", which destroys
        whatever the unreadable one still held.
        """
        ...

    def save(self, tasks: Iterable[dict], now: datetime) -> None:
        """Replace the baseline with the tasks this poll returned."""
        ...


def default_state_path() -> Path:
    """Where the baseline lives when the caller names no path.

    ``$XDG_STATE_HOME``, defaulting to the spec's ``~/.local/state`` — state,
    not data and not cache: it is regenerable, but regenerating it costs a
    poll's completions, and unlike a cache nothing else can reconstruct them.

    An unset *or empty* variable takes the default. Reading an empty one
    literally yields a relative path, so the baseline would be written wherever
    the timer happened to start and the next run, started elsewhere, would find
    no state and infer nothing — forever, silently.
    """
    base = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
    return Path(base) / "aggregator" / "ticktick" / "open_tasks.json"


def save_state(path: Path, tasks: Iterable[dict], now: datetime) -> None:
    """Persist the current open-task set as the next poll's baseline, atomically.

    Written to a scratch file in the same directory and renamed into place, so
    an interrupted write cannot leave truncated JSON behind: the file is always
    either the previous poll or this one. That matters more here than the usual
    tidiness argument — a corrupt baseline now stops inference and the baseline
    advance dead until a human clears it (``load_state`` raises rather than
    reading it as empty), so leaving truncated JSON behind costs every poll
    until somebody notices, not just the next one.

    Keys come from ``_task_id`` — the module's single id rule, the one
    ``task_to_record`` mints stable_ids with — so a baseline key is always
    exactly the id the next poll will compare against. A task whose id is
    unusable is skipped rather than allowed to abort the save, and is counted
    out loud: a baseline that quietly lost entries would invent a completion
    for every one of them on the very next poll.

    Written 0600, not at the default umask. This file holds whole task
    payloads — titles and notes, i.e. the user's medical appointments, legal
    matters and everything else they put in a task manager. No token, so it is
    not a credential, but a world-readable copy of somebody's to-do list is not
    something a cache file should create on their behalf. The mode is set on
    the scratch fd BEFORE any bytes are written, so there is no window at 0644,
    and applied explicitly rather than relying on O_CREAT's mode argument,
    which does nothing when a scratch file from an earlier run already exists.
    """
    payload: dict[str, dict] = {}
    skipped = 0
    for task in tasks:
        try:
            payload[_task_id(task)] = {"task": task, "last_seen": now.isoformat()}
        except (ValueError, AttributeError):
            skipped += 1
    if skipped:
        log.warning("ticktick state: skipped %d task(s) with no usable id", skipped)
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_name(path.name + ".tmp")
    fd = os.open(scratch, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), 0o600)
        handle.write(json.dumps(payload))
    scratch.replace(path)


def _unreadable(path: Path, reason: str) -> StateUnreadableError:
    """The one message an operator gets for a broken baseline. Says what to do.

    Naming the recovery is the point: this is a permanent, every-run error
    until a human acts, so an operator who cannot tell what action clears it
    has been handed an alarm they can only learn to ignore.
    """
    return StateUnreadableError(
        f"ticktick open-task baseline at {path} exists but cannot be read "
        f"({reason}). REFUSING to treat it as an empty baseline: this poll's "
        f"tasks would be written straight over it and every completion still "
        f"pending in it would be unrecoverable, because the Open API serves "
        f"open tasks only and can never report one again. Completion inference "
        f"and the baseline advance are both skipped until this is resolved. "
        f"Inspect the file; deleting it accepts the loss of whatever "
        f"completions it still held and lets the next poll start a fresh "
        f"baseline"
    )


def load_state(path: Path) -> dict[str, dict]:
    """Return the previous poll's open tasks keyed by task id, {} for no file.

    THE ABSENT FILE AND THE BROKEN FILE ARE OPPOSITE ANSWERS, and collapsing
    them is the defect this signature exists to prevent. A path that resolves to
    NOTHING is the first-ever poll: legitimate, silent, ``{}``. Anything that is
    there and could not be turned into a baseline — a half-written file, a wrong
    shape, undecodable bytes, a read this process is not permitted — RAISES.

    The line is drawn at "is anything there to lose", not at the exception's
    name. ``NotADirectoryError`` joins ``FileNotFoundError`` on the quiet side
    because an ancestor component being a plain file means no baseline exists at
    that path or ever did, so starting fresh destroys nothing (and the save that
    follows fails on its own, loudly). ``IsADirectoryError`` does not: something
    is occupying the baseline path and this cannot say what.

    It used to log a warning and return ``{}`` for both, which reads as "there
    is no baseline". The caller's very next act is to write this poll as the new
    baseline, so the unreadable file was replaced from an empty state and every
    completion still pending in it was destroyed — on a run that exited 0 and
    notified nobody. Losing the file costs one poll's inference and is
    recoverable; overwriting it is not, and the Open API cannot re-serve a
    completed task to make it good.

    A log line was never going to be enough here. Nothing under ``aggregator/``
    configures logging, and even at ``logging.lastResort``'s WARNING an
    unattended timer run has nobody reading stderr. Raising is what puts this in
    the run's ``errors``, which is what makes it exit 3 and notify.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, NotADirectoryError):
        return {}
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        raise _unreadable(path, f"{type(e).__name__}: {e}") from e
    if not isinstance(data, dict):
        raise _unreadable(path, f"it holds a {type(data).__name__}, not an id->task map")
    return data


def infer_completions(
    previous: dict[str, dict],
    current_ids: set[str],
    now: datetime,
    errors: list[str] | None = None,
) -> list[Record]:
    """Records for the tasks that were open last poll and are absent now.

    The Open API cannot report a completion, so disappearance is the only
    signal available — and it is ambiguous: a task that was deleted looks
    exactly like one that was finished. Recorded as completed anyway, because
    the alternative is losing it from the index entirely, and never as a plain
    fact: ``provenance`` says the completion was inferred and
    ``completed_time_approx`` says the timestamp is the poll's, not TickTick's.
    A later CSV backup overwrites both with the real Completed Time.

    Sorted by id so a run's output is deterministic.

    ``errors`` is the run's fault sink and an entry here means DATA WAS LOST.
    An entry this cannot turn into a record is dropped — and then the write
    barrier commits the advanced baseline, which is the act that makes the
    disappearance unrepeatable, so that task's completion is gone for good.
    A log line alone left that on an exit-0 run where every count looked
    healthy; routed here it makes the run exit 3 and reach the notifier.
    """
    if previous and not current_ids:
        # A poll whose projects all failed no longer reaches here at all —
        # ``poll_open_tasks`` marks it incomplete and the planner refuses to
        # infer over it. What is left is a COMPLETE poll that legitimately
        # returned nothing, i.e. the user really did finish everything, plus
        # the residue that no coverage flag can catch: every project healthy
        # and genuinely empty. That is a real state and the inference runs —
        # but not in silence, because it is also what a TickTick-side outage
        # that answers 200-with-no-tasks would look like.
        #
        # ``_note``, not ``log.warning``: the caller is about to commit the
        # advanced baseline, which is what makes every one of these
        # disappearances unrepeatable. A log line left the whole open-task list
        # being written off as completed on a run that exited 0 and notified
        # nobody — the same gap as the two cases above, in the one place where
        # the blast radius is every task at once.
        _note(
            errors,
            f"ticktick api: the poll returned no open tasks at all while "
            f"{len(previous)} were open last time, so all {len(previous)} are being "
            f"recorded as inferred completions. If TickTick or the network was down "
            f"this is an outage, not {len(previous)} completions; the next successful "
            f"poll re-observes them as open and corrects the index",
        )
    records = []
    unusable: list[str] = []
    for task_id, entry in sorted(previous.items()):
        if task_id in current_ids:
            continue
        try:
            task = entry.get("task") or {}
            records.append(
                task_to_record(task, completed_at=now, provenance="api-inferred-complete")
            )
        except (AttributeError, ValueError):
            # A garbled entry — not an object, or a stored payload with no
            # usable id. Skipped rather than allowed to abort the loop: one bad
            # entry must not cost every other completion in the batch, the same
            # rule the project walk follows for a malformed ``tasks`` payload.
            unusable.append(task_id)
    if unusable:
        _note(
            errors,
            f"ticktick state: {len(unusable)} baseline entr(ies) could not be "
            f"turned into completions and are being DROPPED "
            f"({', '.join(unusable[:5])}"
            f"{', ...' if len(unusable) > 5 else ''}). The advanced baseline no "
            f"longer holds them, and the Open API serves open tasks only, so "
            f"these completions cannot be re-derived. Check the open-task "
            f"baseline for a truncated or hand-edited entry",
        )
    return records


def _open_task_ids(tasks: Iterable[dict]) -> set[str]:
    """The ids of this poll's tasks, keyed exactly as the baseline keys them.

    Both sides of the diff must spell an id the same way or an unchanged task
    reads as disappeared on every poll: ``_task_id`` strips, so a stored key of
    ``t1`` would never match a hand-rolled ``str(task["id"])`` of ``"  t1  "``.
    Ids the baseline could not store are left out here too — ``save_state``
    already counted and reported them.
    """
    ids = set()
    for task in tasks:
        try:
            ids.add(_task_id(task))
        except (ValueError, AttributeError):
            continue
    return ids


@dataclass(frozen=True)
class JsonFileState:
    """The shipped adapter: the baseline as one JSON file on disk.

    Deliberately thin — the reading and writing rules are the two module
    functions above, which is also what the plan's own brief has task 8 call.
    This is that behaviour wearing the port's shape, not a second
    implementation of it.
    """

    path: Path

    def load(self) -> dict[str, dict]:
        return load_state(self.path)

    def save(self, tasks: Iterable[dict], now: datetime) -> None:
        save_state(self.path, tasks, now)


INCOMPLETE_POLL_NOTE = (
    "ticktick api: this poll did NOT see every open task (a project failed, "
    "was unidentifiable, or served entries this client could not read), so "
    "completion inference is being SKIPPED and the open-task baseline is NOT "
    "being advanced. Inference reads 'was open last poll, absent now' as "
    "'completed', and the Open API only ever serves open tasks, so running it "
    "over a partial view would mark every unseen open task completed with no "
    "way for a later poll to disagree. The next complete poll reconciles "
    "normally; nothing is lost by waiting"
)


def plan_open_task_reconcile(
    state: OpenTaskState,
    poll: OpenTaskPoll,
    now: datetime,
    errors: list[str] | None = None,
) -> tuple[list[Record], Callable[[], None]]:
    """Diff this poll against the baseline; hand back the save as a callable.

    TAKES AN :class:`OpenTaskPoll`, not a bare task list, and that is
    load-bearing rather than typing taste. Inference is only sound over a
    COMPLETE view of what is open: it reads absence as completion, so a poll
    that failed to fetch one project reports every open task in that project
    as finished — and the Open API serves open tasks only, so no later poll can
    ever contradict it. Only a manual CSV export could. Taking the tasks
    without the verdict made "is this a full view?" a question the caller had
    to think to ask, and the source did not.

    An INCOMPLETE poll therefore infers nothing and hands back a commit that
    does nothing: the baseline is neither advanced nor armed, so the next
    complete poll diffs against the same, still-true baseline and reconciles
    everything that really did get completed in the meantime. The skip is
    recorded in ``errors`` — a run that quietly declined to infer would be the
    same silent-success shape in the other direction.

    TWO-PHASE, and that is the point. The diff has to happen now — it needs
    the poll — but advancing the baseline is a destructive act: it is what
    makes a disappearance unrepeatable. The Open API serves OPEN tasks only,
    so a completion is reported exactly once, as a gap between two polls; a
    baseline advanced before the inferred records were WRITTEN turns any
    later store or sink failure into permanent loss, with a re-run unable to
    recover it.

    So the caller gets the records and a ``commit`` it may only invoke once
    those records have landed. Skipping ONE commit costs one poll's worth of
    inference and nothing else: the next poll diffs against the same baseline
    and infers the same completions again. Never committing at all is a
    different and worse thing — the baseline freezes, a task created after the
    freeze never enters it, and its later disappearance is therefore invisible
    to every future poll. Do not read "safe to skip" as "optional".

    The load/diff/save ORDER remains the trap it always was — saving before
    loading overwrites the baseline with the current poll and inference is
    silently dead — which is why this stays one function and not three calls
    at the call site.

    ``errors`` is forwarded to :func:`infer_completions`, where an entry means
    a completion is being dropped and the commit is about to make that
    permanent. Pass the run's sink; a bare log line reaches nobody on a timer.
    """
    tasks = list(poll.tasks)
    if not poll.complete:
        _note(errors, INCOMPLETE_POLL_NOTE)
        return [], lambda: None

    try:
        previous = state.load()
    except StateUnreadableError as e:
        # Caught HERE rather than left to propagate, and that placement is the
        # whole fix. Propagating would take the CSV leg's 1302 backup rows out
        # with it (``_api_candidates`` calls this planner outside its try), and
        # swallowing it would put us back where we started. So: recorded in the
        # run's sink, which is what makes the run exit 3 and notify, and handed
        # back with a commit that DOES NOTHING — the baseline is not advanced,
        # so the file survives for a human to look at. That is the asymmetry
        # this turns on: losing the file costs one poll's inference and is
        # recoverable, overwriting it is permanent.
        _note(errors, str(e))
        return [], lambda: None

    records = infer_completions(previous, _open_task_ids(tasks), now, errors)

    def commit() -> None:
        state.save(tasks, now)

    return records, commit


def reconcile_open_tasks(
    state: OpenTaskState,
    poll: OpenTaskPoll,
    now: datetime,
    errors: list[str] | None = None,
) -> list[Record]:
    """Diff this poll against the baseline, then make this poll the baseline.

    The single-phase form: commits immediately. Correct only for a caller that
    has nothing to write, or that treats the diff as advisory — see
    :func:`plan_open_task_reconcile`, which is what the source uses.

    Typed against :class:`OpenTaskState`, so nothing here knows or cares that
    the baseline is a JSON file.
    """
    records, commit = plan_open_task_reconcile(state, poll, now, errors)
    commit()
    return records
