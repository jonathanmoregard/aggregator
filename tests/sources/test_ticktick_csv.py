from __future__ import annotations

import logging
from datetime import UTC, datetime

import pytest

from aggregator.sources import ticktick_api
from aggregator.sources.ticktick_csv import (
    HEADER_LINE_INDEX,
    MAX_PREAMBLE_ROWS,
    STATUS_TAGS,
    UNKNOWN_STATUS_TAG,
    PerRowFaults,
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


def test_a_row_without_a_taskid_is_dropped_into_the_errors_sink(tmp_path):
    """Same shape as the github source's dropped rows, and worse consequences.

    The backup is the ONLY place completed-task history exists and nothing
    regenerates it, so a row lost here is lost for good. A ``log.warning`` alone
    left that on a run that exited 0 and notified nobody.
    """
    errors: list[str] = []
    rows = parse_backup(
        _backup(tmp_path, [_row(), _row(task_id="", title="Orphan")]), errors
    )
    assert [r["taskId"] for r in rows] == ["abc123"]
    assert len(errors) == 1
    assert "1 ticktick row(s) with no taskId" in errors[0]


def test_a_clean_backup_adds_no_errors(tmp_path):
    errors: list[str] = []
    parse_backup(_backup(tmp_path, [_row()]), errors)
    assert errors == []


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
    assert rec.extra["priority"] == "medium"
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


def test_unknown_status_is_never_guessed_at_as_open(tmp_path, caplog):
    """A status TickTick has never emitted (e.g. 1) is loud, not silently coerced.

    "There is no status 1" was measured against one export in 2026 — an
    observation about the vendor's current behaviour, not a guarantee. Coercing
    drift to ``open`` misclassified the task on an exit-0 run: a search for open
    work would return something TickTick considers finished, with nothing
    anywhere saying the index had guessed.
    """
    row = parse_backup(_backup(tmp_path, [_row(status="1")]))[0]
    errors: list[str] = []
    with caplog.at_level(logging.WARNING, logger="aggregator.sources.ticktick_csv"):
        rec = row_to_record(row, source_file="x.csv", errors=errors)
    assert "open" not in rec.tags
    assert UNKNOWN_STATUS_TAG in rec.tags
    # Emitted, not dropped: nothing regenerates the backup, so losing the row
    # would be permanent and strictly worse than an honestly-labelled unknown.
    assert rec.stable_id == "ticktick:abc123"
    # The raw code rides along verbatim, so the drifted rows stay findable.
    assert rec.extra["status"] == "1"
    assert "'1'" in caplog.text
    assert len(errors) == 1
    assert "'1'" in errors[0]


def test_a_recognised_status_adds_no_error(tmp_path):
    """Only DRIFT is loud. Every real row must stay silent, or the alert that
    fires on every healthy run is an alert nobody reads."""
    errors: list[str] = []
    for status in STATUS_TAGS:
        row = parse_backup(_backup(tmp_path, [_row(status=status)]))[0]
        row_to_record(row, source_file="x.csv", errors=errors)
    assert errors == []


def test_per_row_faults_collapses_repeats_and_keeps_the_exact_count():
    faults = PerRowFaults()
    for _ in range(1302):
        faults.append("same fault")
    errors: list[str] = []

    faults.flush("TickTick.csv", errors)

    assert len(errors) == 1
    assert "1302 rows share this fault" in errors[0]
    assert errors[0].startswith("TickTick.csv: ")


def test_per_row_faults_names_a_lone_fault_without_a_count_preamble():
    """A single occurrence must read exactly as it did before this sink existed —
    the message is carefully written and a "1 rows share this" preamble would
    only get in its way."""
    faults = PerRowFaults()
    faults.append("one odd row")
    errors: list[str] = []

    faults.flush("TickTick.csv", errors)

    assert errors == ["TickTick.csv: one odd row"]


def test_per_row_faults_is_idempotent_and_drops_nothing_twice():
    faults = PerRowFaults()
    faults.append("x")
    errors: list[str] = []
    faults.flush("f", errors)
    faults.flush("f", errors)
    assert errors == ["f: x"]


def test_per_row_faults_without_a_sink_reports_nowhere_and_does_not_raise():
    """``errors is None`` means the source was driven without a sink at all —
    a degradation ``sync_bridge.unwired_sink_note`` reports one level up."""
    faults = PerRowFaults()
    faults.append("x")
    faults.flush("f", None)  # must not raise


def test_status_wins_over_completed_time(tmp_path):
    """Open rows carrying a Completed Time exist in the real export."""
    row = parse_backup(
        _backup(tmp_path, [_row(status="0", completed="2026-08-03T17:30:00+0000")])
    )[0]
    rec = row_to_record(row, source_file="x.csv")
    assert "open" in rec.tags
    assert "completed" not in rec.tags
    assert rec.updated_at == datetime(2026, 8, 3, 17, 30, tzinfo=UTC)


def test_api_status_codes_are_csv_status_codes():
    """One vocabulary: every code the API leg writes must be one the CSV leg tags."""
    assert STATUS_TAGS[ticktick_api.STATUS_OPEN] == "open"
    assert STATUS_TAGS[ticktick_api.STATUS_COMPLETED] == "completed"


@pytest.mark.parametrize(
    ("raw", "name"), [("0", "none"), ("1", "low"), ("3", "medium"), ("5", "high")]
)
def test_priority_digit_becomes_a_name(tmp_path, raw, name):
    """The index holds the word a human would search for, not the export's digit."""
    row = parse_backup(_backup(tmp_path, [_row(priority=raw)]))[0]
    assert row_to_record(row, source_file="x.csv").extra["priority"] == name


def test_blank_priority_is_ticktick_default_none(tmp_path):
    """An absent level is TickTick's own default, 0 — the same default the API leg uses."""
    row = parse_backup(_backup(tmp_path, [_row(priority="")]))[0]
    assert row_to_record(row, source_file="x.csv").extra["priority"] == "none"


def test_unknown_priority_is_kept_verbatim_and_warns(tmp_path, caplog):
    """TickTick has no priority 2 or 4; an unlisted level must not read as 'none'."""
    row = parse_backup(_backup(tmp_path, [_row(priority="2")]))[0]
    with caplog.at_level(logging.WARNING, logger="aggregator.sources.ticktick_csv"):
        rec = row_to_record(row, source_file="x.csv")
    assert rec.extra["priority"] == "2"
    assert "'2'" in caplog.text


@pytest.mark.parametrize(
    ("level", "name"), [(0, "none"), (1, "low"), (3, "medium"), (5, "high")]
)
def test_both_legs_write_the_same_priority_word(tmp_path, level, name):
    """The bug this replaces: the CSV leg wrote "5" where the API leg wrote "high".

    Task 8 merges the two legs by stable_id — asserted equal here — so a
    divergent vocabulary would flip a task's priority depending on which leg
    happened to write last, and a search for one would miss the other.
    """
    row = parse_backup(_backup(tmp_path, [_row(priority=str(level))]))[0]
    csv_rec = row_to_record(row, source_file="x.csv")
    api_rec = ticktick_api.task_to_record(
        {"id": "abc123", "title": "Buy milk", "priority": level}
    )
    assert csv_rec.stable_id == api_rec.stable_id
    assert csv_rec.extra["priority"] == api_rec.extra["priority"] == name


def test_both_legs_keep_an_unknown_priority_the_same_way(tmp_path):
    """Divergence must not sneak back in through the fallback path either."""
    row = parse_backup(_backup(tmp_path, [_row(priority="4")]))[0]
    csv_rec = row_to_record(row, source_file="x.csv")
    api_rec = ticktick_api.task_to_record({"id": "abc123", "title": "Buy milk", "priority": 4})
    assert csv_rec.extra["priority"] == api_rec.extra["priority"] == "4"


@pytest.mark.parametrize(
    ("code", "tag"), [("0", "open"), ("2", "completed"), ("-1", "abandoned")]
)
def test_both_legs_write_the_same_status_word(tmp_path, code, tag):
    """The API leg used to derive status from an inference and ignore the payload.

    Task 8 lets the fresher API observation beat the CSV row, so a completed task
    the backup had right came back as `open` with the CSV evidence discarded.
    Same task id, same status code, same tag — asserted across both legs.
    """
    row = parse_backup(_backup(tmp_path, [_row(status=code)]))[0]
    csv_rec = row_to_record(row, source_file="x.csv")
    api_rec = ticktick_api.task_to_record(
        {"id": "abc123", "title": "Buy milk", "status": int(code)}
    )
    assert csv_rec.stable_id == api_rec.stable_id
    assert csv_rec.extra["status"] == api_rec.extra["status"] == code
    assert tag in csv_rec.tags
    assert tag in api_rec.tags


def test_both_legs_spell_the_same_due_date_identically(tmp_path):
    """The API writes `.000`, the export does not — same instant, two spellings.

    ``extra`` is indexed as text, so an exact-match DSL filter on due_date would
    have matched only whichever leg happened to write the record last.
    """
    row = parse_backup(_backup(tmp_path, [_row(due="2026-08-09T03:00:00+0000")]))[0]
    csv_rec = row_to_record(row, source_file="x.csv")
    api_rec = ticktick_api.task_to_record(
        {"id": "abc123", "title": "Buy milk", "dueDate": "2026-08-09T03:00:00.000+0000"}
    )
    assert csv_rec.extra["due_date"] == api_rec.extra["due_date"] == "2026-08-09T03:00:00+0000"


def test_csv_due_dates_keep_the_exports_own_spelling(tmp_path):
    """Normalisation must be a no-op on the real export's format, not a rewrite.

    Measured: all 1129 dated rows are `+0000` with no fraction. If canonicalising
    changed them, every existing indexed due_date would silently stop matching.
    """
    raw = "2026-08-02T09:00:00+0000"
    row = parse_backup(_backup(tmp_path, [_row(due=raw, start=raw)]))[0]
    rec = row_to_record(row, source_file="x.csv")
    assert rec.extra["due_date"] == raw
    assert rec.extra["start_date"] == raw


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
