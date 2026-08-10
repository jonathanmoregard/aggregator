"""TickTick backup-CSV parsing.

The CSV export is the ONLY source of completed-task history: TickTick's
official Open API filters completed tasks out of every read endpoint, which
is why no TickTick MCP server can back a history index on its own. See the
design doc for the evidence.

Backups are detected by structure, not filename — a short metadata preamble,
then a header row carrying TickTick's column names — so an arbitrary CSV
sitting in ~/Downloads is ignored rather than half-parsed into garbage records.

The header is *discovered*, never counted to. TickTick's third preamble field
is a single quoted value spanning four physical lines, so "six metadata lines"
is only three csv rows, and the count varies with the export version. Scanning
for the required columns is version-proof; counting rows silently yields zero
records, which is indistinguishable from "no backup present".

This module also owns TickTick's shared vocabulary — the status codes, the
priority names, the status->tag mapping and the canonical date spelling —
because the export documents them in its own preamble, so here they are
grounded in evidence rather than in a guess. ``ticktick_api.py`` imports them;
the dependency only ever points that way (the CSV parser never needs the API
client), so the two legs cannot drift into writing different words for the same
value when task 8 merges them by stable_id.

The shared helpers take an optional ``logger`` so a warning about an API-sourced
value is attributed to ``aggregator.sources.ticktick_api`` rather than sending
an operator looking through the backup file for a value that never came from it.
"""
from __future__ import annotations

import csv
import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from aggregator.sources.base import Record, stable_id_for

log = logging.getLogger(__name__)

SOURCE_NAME = "ticktick"

# Where the header sits in a TickTick v7.2 export, counted in csv rows (not
# physical lines: preamble row 2 is a quoted field spanning four lines).
# Informational — fixtures use it; the parser discovers the header instead.
HEADER_LINE_INDEX = 3

# How far into a file we look for the header before giving up. Bounded so an
# unrelated multi-megabyte CSV is not read end to end just to reject it.
MAX_PREAMBLE_ROWS = 20

# A file must have all of these on its header line to be treated as a backup.
REQUIRED_COLUMNS = frozenset({"Title", "taskId", "Status", "Created Time"})

# TickTick's own vocabulary, quoted verbatim from the export's preamble:
#   Status:
#   0 Normal
#   -1 Abandoned
#   2 Completed
# There is no status 1. Anything unlisted is tagged 'open' *and warned about*
# (session constraint 2026-08-08, "fail loudly") so a new vendor status code
# cannot silently repeat the inverted-mapping bug this replaces.
#
# ticktick_api.py imports STATUS_OPEN/STATUS_COMPLETED so an API-inferred
# completion writes the same code the CSV leg writes.
STATUS_OPEN = "0"
STATUS_COMPLETED = "2"
STATUS_ABANDONED = "-1"
STATUS_TAGS = {
    STATUS_OPEN: "open",
    STATUS_COMPLETED: "completed",
    STATUS_ABANDONED: "abandoned",
}

# TickTick priority levels: 0 none, 1 low, 3 medium, 5 high. There is no 2 or 4.
PRIORITY_NAMES = {0: "none", 1: "low", 3: "medium", 5: "high"}

# How every date *string* is spelled in the index, on both legs. Measured on the
# real export: all 1129 non-empty ``Due Date`` values are `+0000` with no
# milliseconds, so this is the export's own spelling, not an invention. The API
# writes the same instant with a `.000` fraction; task 8 merges the two legs by
# stable_id, so without one canonical form the same task's due_date would flip
# spelling depending on which leg wrote last and an exact-match DSL filter would
# miss half of them.
DATE_TEXT_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


def _parse_dt(value: str | None, *, logger: logging.Logger | None = None) -> datetime | None:
    """Parse a TickTick timestamp, tolerating both ISO offsets and ``+0000``."""
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    for parse in (
        datetime.fromisoformat,
        lambda v: datetime.strptime(v, "%Y-%m-%dT%H:%M:%S%z"),
        lambda v: datetime.strptime(v, "%Y-%m-%d %H:%M:%S"),
    ):
        try:
            parsed = parse(text)
        except ValueError:
            continue
        # Normalise to UTC: store.py compares and orders created_at/updated_at
        # as ISO *text*, so a surviving foreign offset would sort wrongly.
        return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    (logger or log).warning("unparseable ticktick timestamp: %r", value)
    return None


def normalize_date_text(value: object, *, logger: logging.Logger | None = None) -> str:
    """Canonicalise a TickTick date to :data:`DATE_TEXT_FORMAT`. Shared by both legs.

    ``extra["due_date"]`` / ``extra["start_date"]`` are indexed as text, so the
    two legs must spell the same instant identically — see DATE_TEXT_FORMAT.
    The CSV export already writes this exact form, so this is a no-op on all
    1129 dated rows of the real export; it is the API's `.000` fraction and any
    future non-UTC offset that it actually converts.

    Always returns a ``str``: a non-string payload value (TickTick could send an
    epoch int) would otherwise land a JSON *number* in ``extra`` where the CSV
    leg lands a string, and a DSL filter on the key would then miss it. An
    unparseable value is kept verbatim and warned about rather than dropped.
    """
    if value is None:
        return ""
    text = (value if isinstance(value, str) else str(value)).strip()
    if not text:
        return ""
    parsed = _parse_dt(text, logger=logger)
    if parsed is None:
        return text  # _parse_dt already warned
    return parsed.strftime(DATE_TEXT_FORMAT)


def _find_header(reader: Iterator[list[str]]) -> list[str] | None:
    """Advance ``reader`` past the preamble, returning the header row.

    Leaves the reader positioned on the first data row. Returns None (reader
    position unspecified) if no header turns up within MAX_PREAMBLE_ROWS.
    """
    for index, row in enumerate(reader):
        if index >= MAX_PREAMBLE_ROWS:
            return None
        if REQUIRED_COLUMNS.issubset(row):
            return row
    return None


def _header_fields(path: Path) -> list[str] | None:
    """Return the backup's header fields, or None if this is not a backup."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            return _find_header(csv.reader(fh))
    except (OSError, UnicodeDecodeError, csv.Error) as e:
        log.debug("not a readable csv: %s (%s)", path, e)
    return None


def is_ticktick_backup(path: Path) -> bool:
    """True when a header carrying every REQUIRED_COLUMN is found in the preamble."""
    return _header_fields(path) is not None


def parse_backup(path: Path) -> list[dict[str, str]]:
    """Return the backup's task rows, or [] if the file has no TickTick header.

    Rows are keyed by the discovered header, so extra vendor columns (v7.2 added
    ``Kind`` and ``projectKind``) ride along instead of shifting every field.
    Rows without a taskId are dropped. Unlike :func:`is_ticktick_backup` this
    does not swallow read errors — gate on detection first.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        header = _find_header(reader)
        if header is None:
            return []
        rows = [dict(zip(header, row, strict=False)) for row in reader if row]
    kept = [r for r in rows if r.get("taskId")]
    # A dropped row is never silent: an ingest that quietly loses tasks is
    # indistinguishable from a backup that had none (session constraint
    # 2026-08-08, "fail loudly").
    if len(kept) != len(rows):
        log.warning(
            "dropped %d ticktick row(s) without a taskId from %s",
            len(rows) - len(kept),
            path,
        )
    return kept


def status_tag(status: str, *, logger: logging.Logger | None = None) -> str:
    """Map a raw ``Status`` value to its tag, warning on anything unrecognised.

    Shared by both legs: the API leg reads the payload's own ``status`` and must
    tag it with the same word the CSV leg would, or task 8's merge flips a task
    between ``completed`` and ``open`` depending on which leg wrote last.
    """
    tag = STATUS_TAGS.get(status)
    if tag is None:
        (logger or log).warning("unexpected ticktick Status %r; tagging as 'open'", status)
        return "open"
    return tag


def priority_name(value: object, *, logger: logging.Logger | None = None) -> str:
    """Map a TickTick priority to its name. Shared by both legs.

    The CSV export writes ``"5"`` and the Open API writes ``5`` for the same
    priority, but task 8 merges records from both legs by stable_id, so both
    must land the same word in ``extra["priority"]`` — otherwise a task's
    priority flips between ``"5"`` and ``"high"`` depending on which leg wrote
    last, and a search for one misses records written by the other. Names win:
    they are what a human types into the index.

    Absent or blank means TickTick's own default level, 0 (none). An unlisted
    level — there is no 2 or 4 — is indexed verbatim and warned about rather
    than coerced to "none", so a new vendor level is visible instead of being
    silently downgraded (session constraint 2026-08-08, "fail loudly").
    """
    text = "0" if value is None else (str(value).strip() or "0")
    try:
        return PRIORITY_NAMES[int(text)]
    except (ValueError, KeyError):
        (logger or log).warning("unexpected ticktick priority %r; indexing it verbatim", value)
        return text


def row_to_record(row: dict[str, str], source_file: str) -> Record:
    """Map one backup row to a Record.

    The status tag comes strictly from ``Status``, never from the presence of a
    ``Completed Time`` — the real export has open rows carrying one.
    """
    status = (row.get("Status") or "0").strip()
    created = _parse_dt(row.get("Created Time"))
    completed = _parse_dt(row.get("Completed Time"))
    tags = [t.strip() for t in (row.get("Tags") or "").split(",") if t.strip()]
    for key in ("List Name", "Folder Name"):
        value = (row.get(key) or "").strip()
        if value:
            tags.append(value)
    tags.append(status_tag(status))

    return Record(
        stable_id=stable_id_for(SOURCE_NAME, row["taskId"]),
        source=SOURCE_NAME,
        subject=(row.get("Title") or "").strip() or row["taskId"],
        body=row.get("Content") or "",
        tags=tags,
        created_at=created,
        updated_at=completed or created,
        extra={
            "provenance": "csv",
            "status": status,
            "priority": priority_name(row.get("Priority")),
            # Canonical spelling, identical on both legs — see DATE_TEXT_FORMAT.
            "due_date": normalize_date_text(row.get("Due Date")),
            "start_date": normalize_date_text(row.get("Start Date")),
            "repeat": (row.get("Repeat") or "").strip(),
            "parent_id": (row.get("parentId") or "").strip(),
            "source_file": source_file,
        },
    )
