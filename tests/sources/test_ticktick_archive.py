"""Round-2 MEDIUM-5 and MEDIUM-6: the backup archive is data, not cache.

The archive is described by ``sources/ticktick.py`` as the ONLY surviving copy
of a manual export once ~/Downloads is cleared, and nothing can regenerate it:
the Open API serves open tasks only, so completed-task history exists nowhere
else.

M5 — ``_archive`` ran BEFORE ``parse_backup`` and clobbered a same-named
archive unconditionally, so a truncated or unparseable download replaced the
good copy with a worse one before anything had even looked at it.

M6 — ``shutil.copy2`` copies the mode too. A file downloaded into ~/Downloads
is typically 0644, so the archived full task history landed world-readable —
while ``open_tasks.json``, the same class of data, is deliberately written
0600.
"""
from __future__ import annotations

import os
import stat
from datetime import UTC, datetime

import pytest

from aggregator.sources import ticktick_api
from aggregator.sources.ticktick import TickTickSource
from aggregator.sources.ticktick_csv import parse_backup
from tests.sources.test_ticktick_csv import HEADER, _row


@pytest.fixture(autouse=True)
def _no_network_or_credentials(monkeypatch, tmp_path):
    def _forbidden(*args, **kwargs):
        raise AssertionError("test attempted a real network call")

    monkeypatch.setattr(ticktick_api, "_open", _forbidden)
    monkeypatch.setattr(
        ticktick_api, "DEFAULT_ENV_FILE", str(tmp_path / "no-such-env")
    )
    for var in (
        "TICKTICK_ACCESS_TOKEN",
        "AGGREGATOR_TICKTICK_TOKEN",
        "AGGREGATOR_TICKTICK_TOKEN_FILE",
        "AGGREGATOR_TICKTICK_DIR",
    ):
        monkeypatch.delenv(var, raising=False)


def _backup(path, rows, *, mode: int | None = None):
    preamble = "\n".join(f'"Date: line {i}"' for i in range(6))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([preamble, HEADER, *rows]) + "\n", encoding="utf-8")
    if mode is not None:
        path.chmod(mode)
    return path


def _source(tmp_path, **kw):
    return TickTickSource(
        backup_dir=tmp_path / "downloads",
        archive_dir=tmp_path / "archive",
        state_file=tmp_path / "state.json",
        **kw,
    )


def _rows(n: int) -> list[str]:
    return [_row(task_id=f"t{i}", title=f"task {i}") for i in range(n)]


# -- M5: an inferior download must not replace the archive ----------------


def test_a_truncated_download_does_not_clobber_the_archive(tmp_path):
    """THE finding. Both files are called TickTick.csv — the export's own
    name — so the copy lands straight on top of the only surviving history."""
    archived = _backup(tmp_path / "archive" / "TickTick.csv", _rows(5))
    _backup(tmp_path / "downloads" / "TickTick.csv", _rows(1))

    list(_source(tmp_path).iter_records(None))

    assert len(parse_backup(archived)) == 5, "the deep history was overwritten"


def test_an_unparseable_download_does_not_clobber_the_archive(tmp_path):
    """Detection and parse are two separate reads and ~/Downloads is live: a
    browser can truncate a file between them. The archive copy used to be
    taken before the parse was even attempted."""
    archived = _backup(tmp_path / "archive" / "TickTick.csv", _rows(5))
    # Big enough that the corruption is past the header sniff's read buffer:
    # the file IS detected as a backup and only fails on the full parse, which
    # is the exact window a live ~/Downloads opens. Bigger than the archive
    # too, so nothing but the parse-first ordering can save the good copy.
    bad = _backup(tmp_path / "downloads" / "TickTick.csv", _rows(400))
    bad.write_bytes(bad.read_bytes()[:-2000] + b"\xff\xfe" * 8)

    errors: list[str] = []
    list(_source(tmp_path).iter_records(None, errors=errors))

    assert errors, "an unparseable backup must be reported"
    assert len(parse_backup(archived)) == 5


def test_a_fuller_export_still_replaces_the_archive(tmp_path):
    """The other half: archiving has to keep working, or the archive freezes
    at whatever export was seen first and every later one is lost."""
    archived = _backup(tmp_path / "archive" / "TickTick.csv", _rows(2))
    _backup(tmp_path / "downloads" / "TickTick.csv", _rows(9))

    list(_source(tmp_path).iter_records(None))

    assert len(parse_backup(archived)) == 9


def test_a_first_time_export_is_archived(tmp_path):
    """Nothing to protect, so nothing is refused."""
    _backup(tmp_path / "downloads" / "TickTick.csv", _rows(3))

    list(_source(tmp_path).iter_records(None))

    assert len(parse_backup(tmp_path / "archive" / "TickTick.csv")) == 3


# -- M6: the archived copy holds the same data class as open_tasks.json ---


def test_the_archived_copy_is_not_world_readable(tmp_path):
    """A downloaded file is typically 0644 and ``copy2`` preserves the mode.
    This file is the user's whole task history — titles and notes, i.e. their
    medical appointments and legal matters — and ``open_tasks.json`` is
    deliberately written 0600 for exactly that reason. Same data class,
    opposite treatment."""
    _backup(tmp_path / "downloads" / "TickTick.csv", _rows(2), mode=0o644)

    list(_source(tmp_path).iter_records(None))

    mode = (tmp_path / "archive" / "TickTick.csv").stat().st_mode
    assert stat.S_IMODE(mode) == 0o600


def test_the_archive_directory_is_not_world_readable(tmp_path):
    """The directory listing alone leaks the export dates."""
    _backup(tmp_path / "downloads" / "TickTick.csv", _rows(2), mode=0o644)

    list(_source(tmp_path).iter_records(None))

    mode = (tmp_path / "archive").stat().st_mode
    assert stat.S_IMODE(mode) == 0o700


def test_the_archived_copy_keeps_the_exports_own_mtime(tmp_path):
    """Tightening the mode must not cost the staleness signal.
    ``newest_backup_mtime`` is what the run-all path turns into "your export is
    31 days stale"; a copy stamped 'now' would make a two-year-old export read
    as fresh forever."""
    exported = _backup(tmp_path / "downloads" / "TickTick.csv", _rows(2))
    old = datetime(2026, 6, 1, tzinfo=UTC).timestamp()
    os.utime(exported, (old, old))

    src = _source(tmp_path)
    list(src.iter_records(None))
    exported.unlink()  # the user clears ~/Downloads

    archived = tmp_path / "archive" / "TickTick.csv"
    assert int(archived.stat().st_mtime) == int(old)
    assert _source(tmp_path).newest_backup_mtime() == datetime(
        2026, 6, 1, tzinfo=UTC
    )


def test_an_already_tightened_archive_is_left_alone(tmp_path):
    """Re-running must not widen anything back."""
    _backup(tmp_path / "downloads" / "TickTick.csv", _rows(2), mode=0o644)
    src = _source(tmp_path)
    list(src.iter_records(None))
    list(_source(tmp_path).iter_records(None))

    archived = tmp_path / "archive" / "TickTick.csv"
    assert stat.S_IMODE(archived.stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "archive").stat().st_mode) == 0o700


def test_refusing_to_archive_is_not_silent(tmp_path, caplog):
    """An archive that stopped updating looks exactly like one that is up to
    date, which is the shape this repo keeps ruling out."""
    _backup(tmp_path / "archive" / "TickTick.csv", _rows(5))
    _backup(tmp_path / "downloads" / "TickTick.csv", _rows(1))

    with caplog.at_level("WARNING"):
        list(_source(tmp_path).iter_records(None))

    assert any(
        "TickTick.csv" in r.getMessage() and "not archiving" in r.getMessage()
        for r in caplog.records
    )
