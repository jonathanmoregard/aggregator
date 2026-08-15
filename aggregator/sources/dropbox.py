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
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from aggregator.core.textextract import (
    PDF_EXTS,
    SUPPORTED_EXTS,
    ExtractionError,
    extract_text,
    first_markdown_heading,
)
from aggregator.sources.base import IngestResult, Record, stable_id_for

log = logging.getLogger(__name__)

DEFAULT_ROOT = "~/Dropbox"

# Pruned unconditionally — never worth indexing, and node_modules alone is
# ~10k files of the tree.
SKIP_DIR_NAMES = {"node_modules", ".git", ".dropbox.cache"}

# A 2 MB text file is not prose. The only files over this limit in the tree
# today are two copies of a 5.4 MB Swedish wordlist.
MAX_TEXT_BYTES = 2 * 1024 * 1024
MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_BODY_CHARS = 200_000

# Below this, a PDF is an image scan with no text layer. Skipped and counted,
# not errored — we made a deliberate no-OCR choice, so this is an expected
# outcome and must not page anyone.
MIN_PDF_TEXT_CHARS = 50


class DropboxRootUnavailableError(OSError):
    """The configured Dropbox ROOT could not be listed at all.

    A HARD failure, deliberately, and the only one this source raises. Every
    other fault here is per-item: one corrupt PDF, one unreadable file, one
    permission-denied subtree — the tree is still there and partial ingest
    beats total loss. A root that cannot be listed is a different KIND of fact:
    the source is not configured as this machine believes it is. Nothing was
    scanned, so "0 records" carries no information at all, and there is no
    input-freshness seam here (Dropbox's own client owns the tree) to catch it
    on the second pass.

    Raising rather than appending to ``errors`` is what makes the difference
    visible where it matters. On the per-source CLI path the errors sink is
    dropped when the iterator raises, but the run prints the failure and exits
    1 instead of the 3 it gives a partial run; on the run-all path the runner's
    isolation boundary records ``DropboxRootUnavailableError: ...`` in
    ``report.errors``, so the record exists either way. It also lands BEFORE
    any store write, which is what keeps ``--rebuild`` on an unmounted laptop
    from reaching the DELETE at all.
    """


def _root_unavailable(root: Path, exc: OSError) -> DropboxRootUnavailableError:
    """The one message an operator gets for a root that will not list.

    Names the likely cause and the fix, because on an unattended timer this is
    the failure that repeats every 30 minutes until a human acts.
    """
    return DropboxRootUnavailableError(
        f"the Dropbox root {root} could not be listed ({type(exc).__name__}: "
        f"{exc}). REFUSING to report this as an empty scan: nothing was walked, "
        f"so 0 records here means 'the source is not configured as believed', "
        f"not 'nothing changed'. Usually the Dropbox client is not running or "
        f"the tree is not mounted yet; check that, or point "
        f"AGGREGATOR_DROPBOX_ROOT at the right directory"
    )


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

    def rebuild_input(self) -> str:
        """``sources.base.SupportsRebuild`` — why ``--rebuild`` is allowed here.

        The Dropbox client keeps this tree current continuously; the source
        walks it live. A row the re-scan does not reproduce is a file that is
        genuinely gone from the tree, which is what the DELETE should mean.
        """
        return "a local Dropbox tree the Dropbox client keeps synced continuously"

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

    def _iter_candidate_paths(self, errors: list[str] | None = None) -> Iterator[Path]:
        """Yield indexable file paths, pruning junk directories during the walk.

        Pruning happens by mutating ``dirnames`` in place so os.walk never
        descends into node_modules at all — on this tree that is the
        difference between statting 25k files and statting 12k.

        ``onerror`` IS PASSED, and its absence was a silent-loss bug. os.walk
        defaults to ``onerror=None``, which swallows every scandir failure: a
        root that is not mounted and a subtree this process may not read both
        produced zero records, ``errors=0`` and exit 0, which is exactly what a
        healthy run with nothing new looks like. The two failures are routed
        differently on purpose — see :class:`DropboxRootUnavailableError` for
        why the root is the one hard failure this source has.
        """

        def _on_walk_error(exc: OSError) -> None:
            failed = getattr(exc, "filename", None)
            if failed is None or Path(failed) == self.root:
                raise _root_unavailable(self.root, exc)
            message = (
                f"{failed}: directory listing failed, so this subtree is NOT "
                f"indexed and every file under it is missing from the index: "
                f"{type(exc).__name__}: {exc}"
            )
            log.warning("%s", message)
            if errors is not None:
                errors.append(message)

        for dirpath, dirnames, filenames in os.walk(self.root, onerror=_on_walk_error):
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

    def record_shape(self) -> dict[str, str]:
        """DSL-facing field surface (M2 help generator)."""
        return {
            "subject": "str (first markdown heading, or filename stem)",
            "body": "str (extracted file text)",
            "relpath": "str (path relative to the Dropbox root)",
            "ext": "str (file extension, lowercased, with dot)",
            "size_bytes": "int (file size on disk)",
            "truncated": "bool (present only when the body was cut)",
        }

    def _size_limit(self, path: Path) -> int:
        return MAX_PDF_BYTES if path.suffix.lower() in PDF_EXTS else MAX_TEXT_BYTES

    def _to_record(self, path: Path, mtime: datetime, text: str, size: int) -> Record:
        rel = path.relative_to(self.root).as_posix()
        ext = path.suffix.lower()
        truncated = len(text) > MAX_BODY_CHARS
        body = text[:MAX_BODY_CHARS] if truncated else text
        subject = (first_markdown_heading(text) if ext in {".md", ".markdown"} else None) or path.stem
        top = PurePosixPath(rel).parts[0] if len(PurePosixPath(rel).parts) > 1 else None
        extra: dict[str, object] = {
            "relpath": rel,
            "ext": ext,
            "size_bytes": size,
        }
        if truncated:
            extra["truncated"] = True
        return Record(
            stable_id=stable_id_for(self.name, rel),
            source=self.name,
            subject=subject,
            body=body,
            # created_at deliberately unset: filesystem birth time is not
            # preserved across the Dropbox sync boundary, so it would be a
            # confident-looking lie.
            created_at=None,
            updated_at=mtime,
            tags=[t for t in (top, ext.lstrip(".")) if t],
            extra=extra,
        )

    def iter_records(
        self,
        since: datetime | None,
        errors: list[str] | None = None,
    ) -> Iterator[Record]:
        """Yield one Record per indexable Dropbox file.

        ``since``: files with mtime <= since are skipped before extraction, so
        an incremental run costs a stat per file rather than a parse.

        Error policy: a file that cannot be parsed appends to ``errors`` and is
        skipped — one corrupt PDF never aborts an ingest of 1600 files. A
        directory that cannot be listed does the same for its whole subtree. An
        image-only PDF is NOT an error (see MIN_PDF_TEXT_CHARS). The single
        exception is the root itself, which raises — see
        :class:`DropboxRootUnavailableError`.
        """
        since_utc: datetime | None = None
        if since is not None:
            since_utc = since if since.tzinfo is not None else since.replace(tzinfo=UTC)

        for path in self._iter_candidate_paths(errors):
            try:
                stat = path.stat()
            except OSError as e:
                log.warning("stat failed for %s: %s", path, e)
                if errors is not None:
                    errors.append(f"{path}: stat failed: {e}")
                continue

            mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
            if since_utc is not None and mtime <= since_utc:
                continue
            if stat.st_size > self._size_limit(path):
                log.info("skipping oversized file %s (%d bytes)", path, stat.st_size)
                continue

            try:
                text = extract_text(path)
            except ExtractionError as e:
                log.warning("extraction failed for %s: %s", path, e)
                if errors is not None:
                    errors.append(f"{path}: {e}")
                continue
            except OSError as e:
                log.warning("read failed for %s: %s", path, e)
                if errors is not None:
                    errors.append(f"{path}: read failed: {e}")
                continue

            if path.suffix.lower() in PDF_EXTS and len(text.strip()) < MIN_PDF_TEXT_CHARS:
                log.info("skipping image-only pdf %s", path)
                continue
            if not text.strip():
                continue

            yield self._to_record(path, mtime, text, stat.st_size)

    def ingest(self, since: datetime | None) -> IngestResult:
        """Count-only path for protocol compat; persistence is the CLI's job."""
        errors: list[str] = []
        added = sum(1 for _ in self.iter_records(since, errors=errors))
        return IngestResult(added=added, updated=0, skipped=0, errors=errors)
