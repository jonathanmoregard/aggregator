"""Wrap returned content in <ExternalContent> delimiters. Called by MCP AND CLI.

Security (spec §Security): every record leaving the aggregator boundary is wrapped
so that model consumers treat the body as untrusted data, not as instructions. The
MCP tool docstring (M3) reinforces this at the tool level; this helper enforces it
mechanically at the return-shape level.
"""
from __future__ import annotations

from aggregator.sources.base import Record


def wrap_record(record: Record) -> str:
    """Return record body wrapped so the model treats it as untrusted data."""
    return (
        f'<ExternalContent source="{record.stable_id}">\n'
        f"{record.body}\n"
        f"</ExternalContent>"
    )


def wrap_records(records: list[Record]) -> str:
    return "\n\n".join(wrap_record(r) for r in records)
