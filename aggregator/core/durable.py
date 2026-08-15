"""Making a write-scratch-then-rename actually survive a power cut.

WHAT AN ATOMIC RENAME DOES AND DOES NOT BUY. Three files in this repo are
written by the same recipe: open a scratch file beside the target, write it,
rename it over the target. The rename is atomic, and that guarantee is real —
no reader ever observes a half-written file, because the directory entry flips
from one complete inode to another in a single step.

It says NOTHING about the bytes surviving. ``write()`` puts them in the page
cache and ``rename()`` only changes a directory entry; both are free to sit in
RAM for the filesystem's commit interval (ext4's default is 5 seconds). A crash
or a power cut inside that window can leave the rename applied and the data
not, which on ext4's default ``data=ordered`` means a target that exists and is
ZERO LENGTH. For ``ticktick/open_tasks.json`` that is not a lost cache entry:
the Open API serves open tasks only, so every completion still pending in the
baseline becomes unrecoverable, and the module's own docstrings claimed this
could not happen. For the TickTick backup archive it is the only surviving copy
of an export nothing regenerates.

So the recipe needs two more steps, in this order and no other:

1. ``fsync`` the SCRATCH FILE before closing it — the data is on the medium.
2. ``rename`` — the target now names an inode whose contents are durable.
3. ``fsync`` the CONTAINING DIRECTORY — the rename itself is durable.

Step 3 is the one that gets forgotten, and without it the first two are not
enough: the file's data is safe but the directory entry pointing at it may not
have been committed, so the old name can come back.

WHAT IS STILL NOT GUARANTEED, stated plainly because the point of this module
is to stop overclaiming. ``fsync`` returning 0 means the kernel handed the data
to the device and the device acknowledged it. A drive that lies about its write
cache, a virtualised disk with an unsafe cache mode, or a filesystem bug can
still lose it, and none of that is observable from here. The tests assert the
call sequence, not the physics — a real power-cut test needs hardware this
suite does not have.
"""
from __future__ import annotations

import errno
import logging
import os
from pathlib import Path
from typing import IO

log = logging.getLogger(__name__)

# ``fsync`` on a directory file descriptor is not universally implemented. On
# every filesystem this repo runs on it works; on some (notably a few network
# and FUSE filesystems) it answers EINVAL, which means "there is nothing here
# for me to do", not "the write failed". Those are tolerated; anything else is
# a real I/O failure and is raised, because a durability helper that swallows
# errors is worse than no helper at all.
_DIR_FSYNC_UNSUPPORTED = frozenset(
    e for e in (errno.EINVAL, getattr(errno, "ENOTSUP", None), errno.EOPNOTSUPP) if e
)


def flush_to_disk(handle: IO[object]) -> None:
    """Force a still-open file's bytes onto the medium.

    Both halves are needed: ``flush()`` moves Python's buffer into the kernel,
    ``fsync()`` moves the kernel's page cache onto the device. Calling only the
    first is the mistake that looks like it worked.
    """
    handle.flush()
    os.fsync(handle.fileno())


def fsync_dir(directory: Path) -> None:
    """Commit a directory entry — i.e. make a rename into it durable.

    Opened read-only, which is the only way a directory may be opened; the fd
    is closed in a ``finally`` so a failing fsync cannot leak it.
    """
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError as e:
        if e.errno not in _DIR_FSYNC_UNSUPPORTED:
            raise
        log.debug("directory fsync unsupported on %s (%s); rename not hardened", directory, e)
    finally:
        os.close(fd)


def replace_durably(scratch: Path, target: Path) -> None:
    """Rename ``scratch`` over ``target`` and commit the directory entry.

    Step 2 and step 3 of the recipe in the module docstring. The caller is
    responsible for step 1 (:func:`flush_to_disk` before the handle closes),
    because only the caller still holds the handle.
    """
    os.replace(scratch, target)
    fsync_dir(target.parent)


__all__ = ["flush_to_disk", "fsync_dir", "replace_durably"]
