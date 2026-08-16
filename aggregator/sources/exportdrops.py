"""Shared discovery for chat-export drops (Chunk 4, chat-exports plan).

Both chat-export sources (chatgpt, claude-web) ingest the same vendor
artefacts — a data-export zip or a pre-extracted ``conversations.json`` —
and both vendors name the payload file identically. This module centralises
WHERE to look and WHICH vendor owns each file so the sources only keep
their per-file parsing.

Scan locations (owner change, 2026-08-02): BOTH the drops dir AND
``~/Downloads`` — the owner downloads vendor export zips straight into
Downloads and should not have to move them.

* drops dir: ``~/.local/share/aggregator/drops`` (override
  ``AGGREGATOR_DROPS_DIR``).
* downloads dir: ``~/Downloads`` (override ``AGGREGATOR_DOWNLOADS_DIR``).

Classification is by CONTENT, not filename:

* ``*.zip`` — member list via :mod:`zipfile` (reads the central directory
  only — cheap even for large archives). A member named
  ``conversations.json`` / ``conversations-*.json`` marks a chat export;
  that member's first array element disambiguates the vendor
  (``mapping`` key → chatgpt, ``chat_messages`` → claude-web). Zips with
  no matching member (unrelated Downloads zips) are silently skipped;
  corrupt zips are skipped with a log warning (Downloads legitimately
  holds partial/broken archives — not an ingest error).
* bare ``conversations.json`` / ``conversations-*.json`` — same shape
  sniff on the file content. Invalid JSON in a conversations-named file
  goes to the ``errors`` sink (corruption of an intentional drop should
  not vanish silently).

The sniff parses the candidate payload once; the owning source parses it
again at ingest time. Two parses of one or two files per manual ingest is
an accepted cost for keeping the ExportFile surface a plain file
reference (correctness + simple API over micro-optimisation).
"""
from __future__ import annotations

import fnmatch
import json
import logging
import os
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_DROPS_DIR = "~/.local/share/aggregator/drops"
DEFAULT_DOWNLOADS_DIR = "~/Downloads"

EXPORT_KINDS = ("chatgpt", "claude-web", "substack")

# Both the flat 2024-era single file and the 2026 sharded layout.
_CONV_PATTERNS = ("conversations.json", "conversations-*.json")


def is_conversations_name(name: str) -> bool:
    """Match a filename (or zip member path) against the accepted shapes.

    Zip members may sit under a subdirectory; only the basename matters.
    """
    base = name.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatch(base, pat) for pat in _CONV_PATTERNS)


def drops_dir() -> Path:
    """Resolve the drops dir (env override honoured at call time)."""
    return Path(
        os.environ.get("AGGREGATOR_DROPS_DIR") or DEFAULT_DROPS_DIR
    ).expanduser()


def downloads_dir() -> Path:
    """Resolve the Downloads scan dir (env override honoured at call time)."""
    return Path(
        os.environ.get("AGGREGATOR_DOWNLOADS_DIR") or DEFAULT_DOWNLOADS_DIR
    ).expanduser()


def default_export_dirs() -> list[Path]:
    """Both scan roots, drops first (drops are deliberate; Downloads is
    convenience)."""
    return [drops_dir(), downloads_dir()]


@dataclass(frozen=True)
class ExportFile:
    """One discovered chat-export payload: a bare JSON file, or a member
    inside a vendor zip (``member`` set)."""

    path: Path
    member: str | None = None

    @property
    def label(self) -> str:
        """Human/log identifier; doubles as the stored ``jsonl_path``."""
        return f"{self.path}!{self.member}" if self.member else str(self.path)

    def read_bytes(self) -> bytes:
        """Raw payload bytes. Zip members are read without extraction."""
        if self.member is None:
            return self.path.read_bytes()
        with zipfile.ZipFile(self.path) as zf:
            return zf.read(self.member)


def classify_export(data: Any) -> str | None:
    """Sniff a parsed conversations payload: which vendor owns it?

    First array element decides: ``mapping`` → chatgpt, ``chat_messages``
    → claude-web. Anything else (empty list, non-list, unknown object
    shape) → None.
    """
    if isinstance(data, list) and data and isinstance(data[0], dict):
        if "mapping" in data[0]:
            return "chatgpt"
        if "chat_messages" in data[0]:
            return "claude-web"
    return None


def discover_export_files(
    kind: str,
    dirs: list[Path] | None = None,
    errors: list[str] | None = None,
) -> list[ExportFile]:
    """Return every export payload of ``kind`` across the scan dirs.

    ``dirs`` overrides the default drops+Downloads pair (sources pass their
    constructor-resolved drops dir plus the live Downloads dir). Duplicate
    dirs are scanned once. Missing dirs are skipped. Order: per-dir sorted
    filenames, zip members sorted within each zip — deterministic emission.
    """
    if kind not in EXPORT_KINDS:
        raise ValueError(f"unknown export kind {kind!r}; expected {EXPORT_KINDS}")
    sink = errors if errors is not None else []
    scan_dirs = [Path(d) for d in (dirs if dirs is not None else default_export_dirs())]
    found: list[ExportFile] = []
    seen_dirs: set[str] = set()
    for d in scan_dirs:
        d = d.expanduser()
        key = os.path.realpath(d)
        if key in seen_dirs:
            continue
        seen_dirs.add(key)
        if not d.is_dir():
            continue
        for path in sorted(d.iterdir()):
            if not path.is_file():
                continue
            if is_conversations_name(path.name):
                _classify_bare(path, kind, found, sink)
            elif path.suffix.lower() == ".zip":
                _classify_zip(path, kind, found, sink)
    return found


def newest_export_mtime(
    kind: str, dirs: list[Path] | None = None
) -> datetime | None:
    """When the newest export archive of ``kind`` was last written, or None.

    These archives are MANUAL: a human asks the vendor for an export, waits,
    downloads the zip and drops it here. Nothing on this machine refreshes
    them, so a timer would re-import the same months-old file forever and
    report success every run — the "empty result looks like success" shape the
    fail-loudly constraint exists to stop. This is the number that lets a run
    say "substack input is 31 days stale" instead.

    The FILE's mtime, not any member's: a vendor zip's internal timestamps are
    whenever the vendor built the archive, while the question here is when the
    human last acquired one.

    ``None`` means no export was found at all — unknown, NOT fresh. Discovery
    errors are swallowed on purpose: this is a probe, and the ingest run
    itself reports them through the ``errors`` sink. Reporting them twice
    would double-count one broken file in the run report.
    """
    newest: float | None = None
    for f in discover_export_files(kind, dirs=dirs, errors=[]):
        try:
            mtime = f.path.stat().st_mtime
        except OSError:
            continue
        if newest is None or mtime > newest:
            newest = mtime
    return datetime.fromtimestamp(newest, tz=UTC) if newest is not None else None


def _classify_bare(
    path: Path, kind: str, found: list[ExportFile], sink: list[str]
) -> None:
    try:
        raw = path.read_bytes()
    except OSError as e:
        sink.append(f"{path}: read failed: {e}")
        return
    data = _load_json(str(path), raw, sink)
    if classify_export(data) == kind:
        found.append(ExportFile(path=path))


def _classify_zip(
    path: Path, kind: str, found: list[ExportFile], sink: list[str]
) -> None:
    """Classify a zip against the requested ``kind``.

    Chat exports (chatgpt, claude-web) match on the presence of a
    ``conversations.json`` / ``conversations-*.json`` member whose first
    array element carries the vendor-discriminating key. Substack exports
    match on ANY ``posts/*.html`` member (glob semantics; the top-level
    ``posts.csv`` does not match — ``fnmatch`` on ``posts/*.html`` is
    segment-literal). Chat sniff runs first so any theoretical
    conversations-shaped payload sitting alongside posts/*.html would win
    the chat kinds; substack classification is only checked when the
    caller asked for kind='substack', so the two paths do not compete.
    """
    try:
        with zipfile.ZipFile(path) as zf:
            names = sorted(zf.namelist())
            if kind in ("chatgpt", "claude-web"):
                members = [m for m in names if is_conversations_name(m)]
                for member in members:
                    try:
                        raw = zf.read(member)
                    except (OSError, zipfile.BadZipFile, KeyError) as e:
                        sink.append(f"{path}!{member}: read failed: {e}")
                        continue
                    data = _load_json(f"{path}!{member}", raw, sink)
                    if classify_export(data) == kind:
                        found.append(ExportFile(path=path, member=member))
                return
            if kind == "substack":
                # Defensive: if this zip is ALSO a chat export, let the
                # chat kinds own it — a well-formed substack export
                # never carries a conversations.json member.
                if any(is_conversations_name(m) for m in names):
                    return
                if any(fnmatch.fnmatch(m, "posts/*.html") for m in names):
                    found.append(ExportFile(path=path))
                return
    except (OSError, zipfile.BadZipFile) as e:
        # Downloads legitimately holds unrelated broken/partial archives —
        # skip without polluting the ingest error report.
        log.warning("skipping unreadable zip %s: %s", path, e)


def _load_json(label: str, raw: bytes, sink: list[str]) -> Any:
    """Parse candidate bytes; invalid JSON is an ingest-visible error
    (a conversations-named file that doesn't parse is a corrupt drop,
    not an unrelated file)."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        sink.append(f"{label}: invalid JSON: {e}")
        return None
