"""The three write-scratch-then-rename paths must be DURABLE, not just atomic.

Round-11 MEDIUM 3. ``ticktick_api._write_state``, ``TickTickSource.
_copy_private`` and ``IngestMarkers.save`` all renamed a scratch file over a
target with NO fsync of the file and no fsync of the directory. An atomic
rename guarantees a reader never sees a half-written file; it does NOT
guarantee the bytes survive power loss, and two of the three docstrings claimed
it did. On ext4's default ``data=ordered`` a crash inside the commit window can
leave the rename applied and the data not — a target that exists and is zero
length. For ``open_tasks.json`` that destroys every pending completion the Open
API can never re-serve; for the TickTick archive it destroys the only surviving
copy of an unregenerable export.

WHAT THESE TESTS CAN AND CANNOT PROVE. They assert the SEQUENCE of syscalls —
fsync the file, then rename, then fsync the directory — against a recorder that
still performs the real calls. They cannot prove the physics: verifying that the
bytes actually survive requires cutting power to real hardware mid-write, which
this suite has no way to do. A drive that lies about its write cache defeats the
fix and nothing here would notice. See ``aggregator/core/durable.py``.

No network seam is touched by any of this, and the autouse guard below makes
that structural rather than incidental.
"""
from __future__ import annotations

import errno
import os
import stat
from datetime import UTC, datetime
from pathlib import Path

import pytest

from aggregator.core.durable import flush_to_disk, fsync_dir, replace_durably
from aggregator.imports.ingest_state import STALE_INPUTS, IngestMarkers
from aggregator.sources import ticktick_api
from aggregator.sources.ticktick import TickTickSource

NOW = datetime(2026, 8, 15, 3, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """No test in this file may reach the real TickTick API.

    Same guard, same reason, as ``tests/sources/test_ticktick_api.py``: ``_open``
    is that module's single network seam and the token behind it is
    write-scoped. Nothing here should ever call it — this fixture is what makes
    that a fact rather than a belief.
    """

    def _forbidden(*args, **kwargs):
        raise AssertionError("test attempted a real network call")

    monkeypatch.setattr(ticktick_api, "_open", _forbidden)


@pytest.fixture
def syscalls(monkeypatch):
    """Record the ORDER of fsync/rename while still performing them for real.

    The fd is classified by ``fstat``, not by bookkeeping in the fake, so a fix
    that fsyncs the same file twice cannot pass by looking like file-then-dir.
    """
    events: list[str] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def fake_fsync(fd: int) -> None:
        kind = "dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file"
        events.append(f"fsync:{kind}")
        real_fsync(fd)

    def fake_replace(src, dst) -> None:
        events.append("rename")
        real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", fake_fsync)
    monkeypatch.setattr(os, "replace", fake_replace)
    return events


DURABLE = ["fsync:file", "rename", "fsync:dir"]


# -- the helper itself ------------------------------------------------------


def test_replace_durably_renames_then_commits_the_directory(tmp_path, syscalls):
    scratch = tmp_path / "t.json.tmp"
    scratch.write_text("payload", encoding="utf-8")

    replace_durably(scratch, tmp_path / "t.json")

    assert syscalls == ["rename", "fsync:dir"]
    assert (tmp_path / "t.json").read_text(encoding="utf-8") == "payload"
    assert not scratch.exists()


def test_flush_to_disk_flushes_before_it_fsyncs(tmp_path, monkeypatch):
    """``fsync`` alone is the mistake that looks like it worked: Python's own
    buffer has not reached the kernel yet, so there is nothing to sync.

    Asserted INSIDE the ``with``, because closing the handle flushes again and
    would hide a helper that only ever relied on ``close``.
    """
    order: list[str] = []
    real_fsync = os.fsync

    def traced_fsync(fd: int) -> None:
        order.append("fsync")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", traced_fsync)
    with (tmp_path / "f.txt").open("w", encoding="utf-8") as handle:
        real_flush = handle.flush

        def traced_flush() -> None:
            order.append("flush")
            real_flush()

        handle.flush = traced_flush  # type: ignore[method-assign]
        handle.write("x" * 4096)
        flush_to_disk(handle)
        assert order == ["flush", "fsync"]


def test_fsync_dir_tolerates_only_the_unsupported_errno(tmp_path, monkeypatch):
    """A filesystem answering EINVAL means "nothing here to do"; anything else
    is a real I/O failure, and a durability helper that swallows those is worse
    than no helper at all."""

    def raise_errno(code):
        def _raise(fd):
            raise OSError(code, os.strerror(code))

        return _raise

    monkeypatch.setattr(os, "fsync", raise_errno(errno.EINVAL))
    fsync_dir(tmp_path)  # tolerated

    monkeypatch.setattr(os, "fsync", raise_errno(errno.EIO))
    with pytest.raises(OSError) as excinfo:
        fsync_dir(tmp_path)
    assert excinfo.value.errno == errno.EIO


# -- the three write paths --------------------------------------------------


def test_ticktick_baseline_save_is_durable(tmp_path, syscalls):
    """The one that costs the most: a zero-length ``open_tasks.json`` loses
    every pending completion, and the Open API serves open tasks only."""
    path = tmp_path / "ticktick" / "open_tasks.json"

    ticktick_api.save_state(path, [{"id": "t1", "title": "x"}], NOW)

    assert syscalls == DURABLE
    assert ticktick_api.load_state(path)["t1"]["task"]["title"] == "x"


def test_ticktick_compare_and_swap_save_is_durable(tmp_path, syscalls):
    """``replace_state`` is the path the shipped adapter actually uses."""
    path = tmp_path / "ticktick" / "open_tasks.json"

    ticktick_api.replace_state(path, [{"id": "t1"}], NOW, expect=None)

    assert syscalls == DURABLE


def test_ticktick_backup_archive_copy_is_durable(tmp_path, syscalls):
    """The archive is the ONLY surviving copy of an export nothing regenerates."""
    source = tmp_path / "TickTick.csv"
    source.write_text("Title,taskId,Status,Created Time\nx,1,0,\n", encoding="utf-8")
    archive = tmp_path / "archive"
    archive.mkdir()

    TickTickSource._copy_private(source, archive / "TickTick.csv")

    assert syscalls == DURABLE
    assert (archive / "TickTick.csv").read_text(encoding="utf-8") == source.read_text(
        encoding="utf-8"
    )


def test_ingest_markers_save_is_durable(tmp_path, syscalls):
    """The mildest of the three — reading fails safe — but the same recipe, and
    a copy that quietly drops a step is how the recipe rots."""
    markers = IngestMarkers(tmp_path / "ingest" / "markers.json")

    markers.save(STALE_INPUTS, {"substack": {"reported_for": "2026-08-15"}})

    assert syscalls == DURABLE
    assert markers.load(STALE_INPUTS) == {"substack": {"reported_for": "2026-08-15"}}


def test_the_scratch_file_is_fsynced_not_the_target(tmp_path, monkeypatch):
    """Order matters, and this pins WHICH file gets step 1.

    Syncing after the rename would be a different (and weaker) recipe: the
    window between rename and fsync is exactly the one being closed.
    """
    synced: list[Path] = []
    real_fsync = os.fsync

    def fake_fsync(fd: int) -> None:
        synced.append(Path(os.readlink(f"/proc/self/fd/{fd}")))
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fake_fsync)
    path = tmp_path / "ticktick" / "open_tasks.json"

    ticktick_api.save_state(path, [{"id": "t1"}], NOW)

    assert synced == [path.with_name(path.name + ".tmp"), path.parent]
