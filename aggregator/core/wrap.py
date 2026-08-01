"""Wrap returned content in <ExternalContent> delimiters. Called by MCP AND CLI.

Security (spec §Security): every record leaving the aggregator boundary is wrapped
so that model consumers treat the body as untrusted data, not as instructions. The
MCP tool docstring (M3) reinforces this at the tool level; this helper enforces it
mechanically at the return-shape level.

Delimiter-injection defence (advisor round-1 HIGH-1):
* ``stable_id`` is html-escaped so an attacker-controlled id cannot break out
  of the ``source="…"`` attribute (e.g. ``id="x"><script>…</script>``).
* Bodies containing a literal ``</ExternalContent>`` (any casing) are
  neutralised in-place — the closing bracket is backslash-escaped so the
  string is preserved (aggregator must not silently drop records) but the
  wrapper cannot be closed early by adversarial payloads.
"""
from __future__ import annotations

import html
import re

from aggregator.sources.base import Record

# Case-insensitive match for the wrapper's closing tag. Compiled once because
# ``wrap_record`` is called on every record leaving the boundary (query + CLI).
_CLOSING_TAG_RE = re.compile(r"</ExternalContent>", re.IGNORECASE)


def _neutralise_closing_delimiter(body: str) -> str:
    """Rewrite any literal ``</ExternalContent>`` (any case) to a form that
    won't close the wrapper. Backslash-escape the closing bracket:
    ``</ExternalContent\\>``. The visible content is preserved so callers who
    read past the boundary still see the payload — we only break the
    delimiter's structural role.
    """
    return _CLOSING_TAG_RE.sub(lambda m: m.group(0)[:-1] + r"\>", body)


def wrap_record(record: Record) -> str:
    """Return record body wrapped so the model treats it as untrusted data.

    Security-hardened per advisor round-1 HIGH-1:
    * ``stable_id`` is html-escaped (with ``quote=True``) inside the
      ``source="…"`` attribute.
    * Any occurrence of ``</ExternalContent>`` (case-insensitive) in the
      body is neutralised so it cannot prematurely close the wrapper.
    """
    escaped_id = html.escape(record.stable_id, quote=True)
    safe_body = _neutralise_closing_delimiter(record.body)
    return (
        f'<ExternalContent source="{escaped_id}">\n'
        f"{safe_body}\n"
        f"</ExternalContent>"
    )


def wrap_records(records: list[Record]) -> str:
    return "\n\n".join(wrap_record(r) for r in records)
