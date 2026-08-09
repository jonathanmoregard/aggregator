from __future__ import annotations

from datetime import UTC, datetime

from aggregator.sources.ticktick_csv import (
    HEADER_LINE_INDEX,
    is_ticktick_backup,
    parse_backup,
    row_to_record,
)

HEADER = (
    "Folder Name,List Name,Title,Tags,Content,Is Check list,Start Date,Due Date,"
    "Reminder,Repeat,Priority,Status,Created Time,Completed Time,Order,Timezone,"
    "Is All Day,Is Floating,Column Name,Column Order,View Mode,taskId,parentId"
)


def _backup(tmp_path, rows, name="TickTick.csv"):
    preamble = "\n".join(f'"Date: line {i}"' for i in range(HEADER_LINE_INDEX))
    p = tmp_path / name
    p.write_text("\n".join([preamble, HEADER, *rows]) + "\n", encoding="utf-8")
    return p


def _row(**over):
    values = {
        "folder": "Personal",
        "list": "Inbox",
        "title": "Buy milk",
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
        f'{values["folder"]},{values["list"]},{values["title"]},{values["tags"]},'
        f'"{values["content"]}",false,{values["start"]},{values["due"]},,'
        f'{values["repeat"]},{values["priority"]},{values["status"]},'
        f'{values["created"]},{values["completed"]},0,UTC,false,false,,,,'
        f'{values["task_id"]},{values["parent_id"]}'
    )


def test_detects_ticktick_backup(tmp_path):
    assert is_ticktick_backup(_backup(tmp_path, [_row()]))


def test_rejects_unrelated_csv(tmp_path):
    p = tmp_path / "bank.csv"
    p.write_text("date,amount\n2026-01-01,42\n", encoding="utf-8")
    assert not is_ticktick_backup(p)


def test_rejects_binary_file_without_raising(tmp_path):
    p = tmp_path / "weird.csv"
    p.write_bytes(b"\x00\x01\x02\xff")
    assert not is_ticktick_backup(p)


def test_parses_rows(tmp_path):
    rows = parse_backup(_backup(tmp_path, [_row(), _row(task_id="def456", title="Call bank")]))
    assert [r["taskId"] for r in rows] == ["abc123", "def456"]
    assert rows[0]["Title"] == "Buy milk"


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
        _backup(tmp_path, [_row(status="1", completed="2026-08-03T17:30:00+0000")])
    )[0]
    rec = row_to_record(row, source_file="TickTick.csv")
    assert "completed" in rec.tags
    assert rec.updated_at == datetime(2026, 8, 3, 17, 30, tzinfo=UTC)
    assert rec.extra.get("completed_time_approx") is None


def test_row_to_record_archived_status(tmp_path):
    row = parse_backup(_backup(tmp_path, [_row(status="2")]))[0]
    assert "archived" in row_to_record(row, source_file="x.csv").tags


def test_row_without_task_id_is_dropped(tmp_path):
    rows = parse_backup(_backup(tmp_path, [_row(task_id="")]))
    assert rows == []
