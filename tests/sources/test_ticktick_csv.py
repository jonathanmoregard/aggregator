from __future__ import annotations

import logging
from datetime import UTC, datetime

from aggregator.sources.ticktick_csv import (
    HEADER_LINE_INDEX,
    MAX_PREAMBLE_ROWS,
    is_ticktick_backup,
    parse_backup,
    row_to_record,
)

# The real v7.2 export's 25 columns, verbatim — `Kind` after `Title` and
# `projectKind` last are the two the original spec did not know about.
HEADER = (
    "Folder Name,List Name,Title,Kind,Tags,Content,Is Check list,Start Date,Due Date,"
    "Reminder,Repeat,Priority,Status,Created Time,Completed Time,Order,Timezone,"
    "Is All Day,Is Floating,Column Name,Column Order,View Mode,taskId,parentId,"
    "projectKind"
)

# TickTick's real preamble: three csv rows over six physical lines, because the
# third is one quoted field with embedded newlines. Counting lines instead of
# rows is what broke detection against the real export.
REAL_PREAMBLE = (
    '"Date: 2026-08-09+0000"\n'
    '"Version: 7.2"\n'
    '"Status: \n'
    "0 Normal\n"
    "-1 Abandoned \n"
    '2 Completed"'
)

# The shape the original (wrong) spec assumed: six single-line preamble rows.
FLAT_PREAMBLE = "\n".join(f'"Date: line {i}"' for i in range(6))


def _backup(tmp_path, rows, name="TickTick.csv", preamble=REAL_PREAMBLE, bom=True):
    """Write a backup fixture. Defaults mirror the real export: BOM + v7.2 preamble."""
    p = tmp_path / name
    p.write_text(
        "\n".join([preamble, HEADER, *rows]) + "\n",
        encoding="utf-8-sig" if bom else "utf-8",
    )
    return p


def _row(**over):
    values = {
        "folder": "Personal",
        "list": "Inbox",
        "title": "Buy milk",
        "kind": "TEXT",
        "tags": "errand",
        "content": "from the good shop",
        "start": "",
        "due": "2026-08-02T09:00:00+0000",
        "repeat": "",
        "priority": "3",
        "status": "0",
        "created": "2026-08-01T08:00:00+0000",
        "completed": "",
        "task_id": "abc123",
        "parent_id": "",
    }
    values.update(over)
    return (
        f'{values["folder"]},{values["list"]},{values["title"]},{values["kind"]},'
        f'{values["tags"]},"{values["content"]}",false,{values["start"]},'
        f'{values["due"]},,{values["repeat"]},{values["priority"]},{values["status"]},'
        f'{values["created"]},{values["completed"]},0,UTC,false,false,,,,'
        f'{values["task_id"]},{values["parent_id"]},TASK'
    )


def test_detects_ticktick_backup(tmp_path):
    assert is_ticktick_backup(_backup(tmp_path, [_row()]))


def test_detects_backup_behind_flat_preamble(tmp_path):
    """Header discovery must not care how many preamble rows a version writes."""
    assert is_ticktick_backup(_backup(tmp_path, [_row()], preamble=FLAT_PREAMBLE))


def test_header_line_index_matches_real_preamble():
    """The v7.2 header is csv row 3, even though six physical lines precede it."""
    assert len(REAL_PREAMBLE.splitlines()) == 6
    assert HEADER_LINE_INDEX == 3


def test_rejects_unrelated_csv(tmp_path):
    p = tmp_path / "bank.csv"
    p.write_text("date,amount\n2026-01-01,42\n", encoding="utf-8")
    assert not is_ticktick_backup(p)


def test_header_scan_is_bounded(tmp_path):
    """A header buried past the scan bound is not a backup — no end-to-end read."""
    filler = "\n".join(f'"noise {i}"' for i in range(MAX_PREAMBLE_ROWS + 5))
    assert not is_ticktick_backup(_backup(tmp_path, [_row()], preamble=filler))


def test_rejects_binary_file_without_raising(tmp_path):
    p = tmp_path / "weird.csv"
    p.write_bytes(b"\x00\x01\x02\xff")
    assert not is_ticktick_backup(p)


def test_parses_rows(tmp_path):
    rows = parse_backup(_backup(tmp_path, [_row(), _row(task_id="def456", title="Call bank")]))
    assert [r["taskId"] for r in rows] == ["abc123", "def456"]
    assert rows[0]["Title"] == "Buy milk"


def test_bom_does_not_corrupt_first_column(tmp_path):
    """utf-8-sig: the BOM must not end up glued to a header or field name."""
    rows = parse_backup(_backup(tmp_path, [_row()], preamble=FLAT_PREAMBLE))
    assert set(rows[0]) == set(HEADER.split(","))
    assert rows[0]["Folder Name"] == "Personal"


def test_parses_wide_v7_columns(tmp_path):
    """`Kind` and `projectKind` ride along instead of shifting every other field."""
    rows = parse_backup(_backup(tmp_path, [_row()]))
    assert rows[0]["Kind"] == "TEXT"
    assert rows[0]["projectKind"] == "TASK"
    assert rows[0]["Status"] == "0"


def test_parses_multiline_quoted_content(tmp_path):
    row = _row(content="line one\nline two")
    rows = parse_backup(_backup(tmp_path, [row]))
    assert rows[0]["Content"] == "line one\nline two"


def test_row_to_record_open_task(tmp_path):
    row = parse_backup(_backup(tmp_path, [_row()]))[0]
    rec = row_to_record(row, source_file="TickTick.csv")
    assert rec.stable_id == "ticktick:abc123"
    assert rec.source == "ticktick"
    assert rec.subject == "Buy milk"
    assert rec.body == "from the good shop"
    assert set(rec.tags) == {"errand", "Inbox", "Personal", "open"}
    assert rec.created_at == datetime(2026, 8, 1, 8, 0, tzinfo=UTC)
    assert rec.updated_at == rec.created_at
    assert rec.extra["provenance"] == "csv"
    assert rec.extra["status"] == "0"
    assert rec.extra["source_file"] == "TickTick.csv"


def test_row_to_record_completed_task_uses_completed_time(tmp_path):
    row = parse_backup(
        _backup(tmp_path, [_row(status="2", completed="2026-08-03T17:30:00+0000")])
    )[0]
    rec = row_to_record(row, source_file="TickTick.csv")
    assert "completed" in rec.tags
    assert rec.updated_at == datetime(2026, 8, 3, 17, 30, tzinfo=UTC)
    assert rec.extra.get("completed_time_approx") is None


def test_status_2_is_completed_not_archived(tmp_path):
    """TickTick's preamble says `2 Completed`. `archived` was never its word."""
    row = parse_backup(_backup(tmp_path, [_row(status="2")]))[0]
    tags = row_to_record(row, source_file="x.csv").tags
    assert "completed" in tags
    assert "archived" not in tags


def test_status_minus_one_is_abandoned(tmp_path):
    """`-1 Abandoned` must not be reported as outstanding work."""
    row = parse_backup(_backup(tmp_path, [_row(status="-1")]))[0]
    tags = row_to_record(row, source_file="x.csv").tags
    assert "abandoned" in tags
    assert "open" not in tags


def test_unknown_status_defaults_open_and_warns(tmp_path, caplog):
    """A status TickTick has never emitted (e.g. 1) is loud, not silently coerced."""
    row = parse_backup(_backup(tmp_path, [_row(status="1")]))[0]
    with caplog.at_level(logging.WARNING, logger="aggregator.sources.ticktick_csv"):
        rec = row_to_record(row, source_file="x.csv")
    assert "open" in rec.tags
    assert "'1'" in caplog.text


def test_status_wins_over_completed_time(tmp_path):
    """Open rows carrying a Completed Time exist in the real export."""
    row = parse_backup(
        _backup(tmp_path, [_row(status="0", completed="2026-08-03T17:30:00+0000")])
    )[0]
    rec = row_to_record(row, source_file="x.csv")
    assert "open" in rec.tags
    assert "completed" not in rec.tags
    assert rec.updated_at == datetime(2026, 8, 3, 17, 30, tzinfo=UTC)


def test_timestamps_normalised_to_utc(tmp_path):
    """store.py compares ISO text, so a foreign offset must not survive."""
    row = parse_backup(_backup(tmp_path, [_row(created="2026-08-01T10:00:00+0200")]))[0]
    rec = row_to_record(row, source_file="x.csv")
    assert rec.created_at == datetime(2026, 8, 1, 8, 0, tzinfo=UTC)
    assert rec.created_at.isoformat().endswith("+00:00")


def test_row_without_task_id_is_dropped(tmp_path):
    rows = parse_backup(_backup(tmp_path, [_row(task_id="")]))
    assert rows == []


def test_unrelated_csv_parses_to_nothing(tmp_path):
    p = tmp_path / "bank.csv"
    p.write_text(
        "date,amount\n" + "\n".join(f"2026-01-0{i % 9 + 1},{i}" for i in range(30)),
        encoding="utf-8",
    )
    assert parse_backup(p) == []
