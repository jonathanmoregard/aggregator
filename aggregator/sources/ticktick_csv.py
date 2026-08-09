"""TickTick backup-CSV parsing.

The CSV export is the ONLY source of completed-task history: TickTick's
official Open API filters completed tasks out of every read endpoint, which
is why no TickTick MCP server can back a history index on its own. See the
design doc for the evidence.

Backups are detected by structure, not filename — six metadata lines, then
the header on line 7 — so an arbitrary CSV sitting in ~/Downloads is ignored
rather than half-parsed into garbage records.
"""
from __future__ import annotations

import csv
import logging
from datetime import UTC, datetime
from pathlib import Path

from aggregator.sources.base import Record, stable_id_for

log = logging.getLogger(__name__)

SOURCE_NAME = "ticktick"

# TickTick writes six metadata lines before the real header.
HEADER_LINE_INDEX = 6

# A file must have all of these on its header line to be treated as a backup.
REQUIRED_COLUMNS = frozenset({"Title", "taskId", "Status", "Created Time"})

STATUS_TAGS = {"0": "open", "1": "completed", "2": "archived"}


def _parse_dt(value: str | None) -> datetime | None:
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
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    log.warning("unparseable ticktick timestamp: %r", value)
    return None


def _header_fields(path: Path) -> list[str] | None:
    """Return the backup's header fields, or None if this is not a backup."""
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            for index, row in enumerate(reader):
                if index < HEADER_LINE_INDEX:
                    continue
                return row
    except (OSError, UnicodeDecodeError, csv.Error) as e:
        log.debug("not a readable csv: %s (%s)", path, e)
    return None


def is_ticktick_backup(path: Path) -> bool:
    fields = _header_fields(path)
    return fields is not None and REQUIRED_COLUMNS.issubset(set(fields))


def parse_backup(path: Path) -> list[dict[str, str]]:
    """Return the backup's task rows. Rows without a taskId are dropped."""
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        for _ in range(HEADER_LINE_INDEX):
            next(reader, None)
        header = next(reader, None)
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


def row_to_record(row: dict[str, str], source_file: str) -> Record:
    """Map one backup row to a Record."""
    status = (row.get("Status") or "0").strip()
    created = _parse_dt(row.get("Created Time"))
    completed = _parse_dt(row.get("Completed Time"))
    tags = [t.strip() for t in (row.get("Tags") or "").split(",") if t.strip()]
    for key in ("List Name", "Folder Name"):
        value = (row.get(key) or "").strip()
        if value:
            tags.append(value)
    tags.append(STATUS_TAGS.get(status, "open"))

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
            "priority": (row.get("Priority") or "").strip(),
            "due_date": (row.get("Due Date") or "").strip(),
            "start_date": (row.get("Start Date") or "").strip(),
            "repeat": (row.get("Repeat") or "").strip(),
            "parent_id": (row.get("parentId") or "").strip(),
            "source_file": source_file,
        },
    )
