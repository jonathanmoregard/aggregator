"""Dropbox source: prose and documents from the locally-synced ``~/Dropbox``.

Records-shaped, one Record per file. Dropbox syncs to local disk, so this is
a filesystem source — no API, no tokens, no export ritual.

The tree is ~25k files and 4 GB, the overwhelming majority of which is
checked-in source code, media, and node_modules from project backups. Only
prose and documents are indexed; see SUPPORTED_EXTS.

``AGGREGATOR_DROPBOX_EXCLUDE`` exists from day one rather than being deferred
to "when someone needs it": this index is exposed to Claude over MCP and the
tree contains contracts, health records, and coaching material. Users need a
way to keep a folder out without patching code.
"""
from __future__ import annotations

import fnmatch
import logging
import os
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

from aggregator.core.textextract import SUPPORTED_EXTS

log = logging.getLogger(__name__)

DEFAULT_ROOT = "~/Dropbox"

# Pruned unconditionally — never worth indexing, and node_modules alone is
# ~10k files of the tree.
SKIP_DIR_NAMES = {"node_modules", ".git", ".dropbox.cache"}


def _is_skipped_dir(name: str) -> bool:
    """Directories pruned during the walk: known junk plus any dot-directory."""
    return name in SKIP_DIR_NAMES or name.startswith(".")


def _matches_exclude(relpath: str, patterns: tuple[str, ...]) -> bool:
    """True when relpath is covered by a user exclude pattern.

    A pattern matches either the path itself or anything beneath it, so
    ``Private`` excludes ``Private/deep/nested/secret.md`` without the user
    having to write ``Private/**``.
    """
    for pat in patterns:
        stem = pat.rstrip("/")
        if fnmatch.fnmatch(relpath, stem) or fnmatch.fnmatch(relpath, f"{stem}/*"):
            return True
        if PurePosixPath(relpath).match(stem):
            return True
    return False


class DropboxSource:
    """Source implementation for local Dropbox prose and documents."""

    name = "dropbox"

    def __init__(
        self,
        root: str | os.PathLike[str] | None = None,
        exclude: str | None = None,
    ):
        self.root = Path(
            root
            or os.environ.get("AGGREGATOR_DROPBOX_ROOT")
            or os.path.expanduser(DEFAULT_ROOT)
        )
        raw = exclude if exclude is not None else os.environ.get("AGGREGATOR_DROPBOX_EXCLUDE", "")
        self.exclude: tuple[str, ...] = tuple(p for p in raw.split(":") if p)

    def _iter_candidate_paths(self) -> Iterator[Path]:
        """Yield indexable file paths, pruning junk directories during the walk.

        Pruning happens by mutating ``dirnames`` in place so os.walk never
        descends into node_modules at all — on this tree that is the
        difference between statting 25k files and statting 12k.
        """
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if not _is_skipped_dir(d))
            here = Path(dirpath)
            rel_dir = here.relative_to(self.root)
            if str(rel_dir) != "." and _matches_exclude(rel_dir.as_posix(), self.exclude):
                dirnames[:] = []
                continue
            for filename in sorted(filenames):
                path = here / filename
                if path.suffix.lower() not in SUPPORTED_EXTS:
                    continue
                rel = path.relative_to(self.root).as_posix()
                if _matches_exclude(rel, self.exclude):
                    continue
                yield path
