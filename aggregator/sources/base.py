"""Abstract source protocol. Every ingester (sessions, github, future) implements Source.

Security note (spec §Security): sources MUST use read-only credentials only. The
github source enforces this via `gh auth` scope inspection (see spec constraint 1).
Every source's `ingest()` scrubs pre-store (Presidio + gitleaks) via core/scrub.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


@dataclass
class Record:
    """Uniform record shape across all sources.

    stable_id: mint once on first cache, never mutate (Gooen stable-ID discipline).
    Format: "<source>:<source-specific-id>", e.g. "sessions:abc123-uuid" or
    "github:owner/repo:42". Enforced by `stable_id_for()` below and by the store's
    upsert path (M2: store must reject a change to an existing stable_id).
    """

    stable_id: str
    source: str
    subject: str  # short label for triage (session summary line, PR title, etc.)
    body: str  # full text for FTS indexing (already scrubbed by ingest pipeline)
    tags: list[str] = field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # source-specific metadata


@dataclass
class IngestResult:
    added: int
    updated: int
    skipped: int
    errors: list[str] = field(default_factory=list)


@dataclass
class QueryAST:
    """Parsed DSL query. Populated by core/dsl.py in M2."""

    source: str | None = None
    tags: list[str] = field(default_factory=list)
    from_date: datetime | None = None
    to_date: datetime | None = None
    text: str | None = None  # freeform FTS terms
    extra: dict[str, str] = field(default_factory=dict)  # per-source keys


@runtime_checkable
class Source(Protocol):
    """Every ingester conforms to this shape. runtime_checkable so the DSL help
    generator (M2) can duck-type registered sources via isinstance()."""

    name: str

    def ingest(self, since: datetime | None) -> IngestResult: ...

    def search(self, ast: QueryAST) -> list[Record]: ...

    def record_shape(self) -> dict[str, str]:
        """Return {field_name: type_description}. Used by DSL help generator."""
        ...


def stable_id_for(source: str, source_specific_id: str) -> str:
    """Mint a stable local ID. Enforces the "<source>:<id>" convention centrally
    so no source module hand-rolls the format inconsistently. See spec §Components
    and non-negotiable #5.
    """
    if not source or ":" in source:
        raise ValueError(f"invalid source name: {source!r} (must be non-empty, no colons)")
    if not source_specific_id:
        raise ValueError("source_specific_id must be non-empty")
    return f"{source}:{source_specific_id}"
