"""Research-reports source: ``~/Repos/research-agent/reports/*.md`` → records.

Records-shaped (one Record per report file — units-of-work ontology, like
GitHub PRs/issues). Local-file source: no export ritual, the timer can hit
it directly.

Security constraint: only reports that PASSED the injection scan live at the
top level of the reports dir. ``reports/_quarantine/`` holds scanner-REJECTED
reports plus audit JSONLs and must NEVER be read. Two layers enforce that:

1. Structural: the non-recursive ``glob("*.md")`` cannot descend into
   ``_quarantine/`` (or any subdir).
2. Belt and braces: ``_is_quarantined`` skips any path whose parts include
   ``_quarantine`` — covering even a misconfigured root that points *at*
   the quarantine dir.

Content note: passed reports legitimately contain
``<untrusted_external_content>`` wrapper tags in their markdown. They are
ingested verbatim — the store scrubs secrets on write and the MCP layer
re-wraps on return. Do not strip the tags here.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from aggregator.sources.base import IngestResult, Record, stable_id_for

log = logging.getLogger(__name__)

QUARANTINE_DIRNAME = "_quarantine"

# Only scan this many leading lines for the first ``# `` heading; a report
# without an early H1 falls back to its filename stem (hex report id).
HEADING_SCAN_LINES = 50

DEFAULT_REPORTS_DIR = "~/Repos/research-agent/reports"


def _is_quarantined(path: Path) -> bool:
    """True when any path segment is the quarantine dir name.

    Belt-and-braces guard behind the structural top-level-only glob: even if
    the configured root itself is (or is inside) ``_quarantine/``, nothing
    from it is ingested.
    """
    return QUARANTINE_DIRNAME in path.parts


def _first_heading(text: str) -> str | None:
    """Return the first ``# `` heading within the leading scan window,
    stripped of leading ``#`` marks and surrounding whitespace."""
    for line in text.splitlines()[:HEADING_SCAN_LINES]:
        if line.startswith("# "):
            return line.lstrip("#").strip()
    return None


class ResearchReportsSource:
    """Source implementation for injection-scan-passed research reports."""

    name = "research"

    def rebuild_input(self) -> str:
        """``sources.base.SupportsRebuild`` — why ``--rebuild`` is allowed here.

        The reports directory is written by the research agent on this machine
        and every stored row comes from a file still in it, so a re-scan
        reproduces the whole population the DELETE can reach.
        """
        return (
            "the research reports directory on this machine, which the "
            "research agent writes and keeps"
        )

    def __init__(self, reports_dir: str | os.PathLike[str] | None = None):
        self.reports_dir = Path(
            reports_dir
            or os.environ.get("AGGREGATOR_RESEARCH_REPORTS_DIR")
            or os.path.expanduser(DEFAULT_REPORTS_DIR)
        )

    def record_shape(self) -> dict[str, str]:
        """DSL-facing field surface (M2 help generator)."""
        return {
            "subject": "str (first markdown heading, or report id)",
            "body": "str (full report markdown)",
            "path": "str (source file path)",
        }

    def _report_to_record(self, path: Path, mtime: datetime, text: str) -> Record:
        subject = _first_heading(text) or path.stem
        return Record(
            stable_id=stable_id_for(self.name, path.stem),
            source=self.name,
            subject=subject,
            body=text,
            tags=["research"],
            created_at=mtime,
            updated_at=mtime,
            extra={"path": str(path)},
        )

    def iter_records(
        self,
        since: datetime | None,
        errors: list[str] | None = None,
    ) -> Iterator[Record]:
        """Yield one Record per top-level ``*.md`` report.

        ``since``: skip files whose mtime <= since (naive ``since`` treated
        as UTC, matching the CLI's ``--since`` stamping). Unreadable files
        are appended to the ``errors`` sink and skipped — partial ingest
        beats total loss (same policy as the github source). Decode never
        hard-fails: reports are UTF-8, read with ``errors='replace'`` so
        stray bytes degrade to replacement chars instead of crashing.
        """
        since_utc: datetime | None = None
        if since is not None:
            since_utc = since if since.tzinfo is not None else since.replace(tzinfo=UTC)

        for path in sorted(self.reports_dir.glob("*.md")):
            if _is_quarantined(path):
                log.warning("skipping quarantined path: %s", path)
                continue
            try:
                mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
            except OSError as e:
                log.warning("stat failed for %s: %s", path, e)
                if errors is not None:
                    errors.append(f"{path}: stat failed: {e}")
                continue
            if since_utc is not None and mtime <= since_utc:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                log.warning("read failed for %s: %s", path, e)
                if errors is not None:
                    errors.append(f"{path}: read failed: {e}")
                continue
            yield self._report_to_record(path, mtime, text)

    def ingest(self, since: datetime | None) -> IngestResult:
        """Count-only path for protocol compat; persistence is the CLI's job."""
        errors: list[str] = []
        added = sum(1 for _ in self.iter_records(since, errors=errors))
        return IngestResult(added=added, updated=0, skipped=0, errors=errors)
