"""Flat filter DSL: ``source:X tag:a,b from:D to:D key:val ... freeform text``.

Design notes (spec §DSL, plan M2):

* Split each whitespace-delimited token on the **first** colon only. Source-
  specific stable IDs like ``github:owner/repo:42`` contain their own colons;
  splitting on all colons would corrupt any downstream matching against them.
* Known top-level keys (``source``, ``tag``, ``from``, ``to``) are hoisted onto
  ``QueryAST`` directly. Every other ``key:val`` token lands in
  ``ast.extra[key] = val`` verbatim, so per-source ``Source.search()``
  implementations can interpret their own vocabulary (``state:``, ``author:``,
  ``check:``, ``mergeable:``, ``project:``, ``model:``, ...) without the DSL
  parser having to enumerate the world.
* Tokens without a colon join the freeform ``text`` (FTS5 query).
* ``format_help`` is called by M3's ``aggregator_capabilities`` tool with the
  actual cached inventory (sources, top tags per source, date range) so the
  model always sees valid options — Chughtai's dynamic-help pattern.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import UTC, datetime

from aggregator.sources.base import QueryAST


class DSLError(ValueError):
    """Raised on malformed DSL input (bad date, unparseable token shape)."""


# Reserved top-level keys. Everything else falls through to ``ast.extra`` so
# per-source ``Source.search()`` can interpret its own vocabulary.
#
# v2 (Schema B) session-ontology keys:
#   session:X   — everything under a session root (matches root_session_id;
#                 includes any subagents spawned by that session).
#   top:X       — only the top-level stream (matches session_id; excludes
#                 subagents).
#   agent:Y     — only rows belonging to the named subagent.
#   type:T      — filter observations by type
#                 (user|assistant|tool_use|tool_result|system|other).
#   active:A..B — sessions whose activity window overlaps [A, B]
#                 (first_ts <= B AND last_ts >= A). A or B may be omitted:
#                 ``active:..2026-07-01`` or ``active:2026-07-01..``.
KNOWN_KEYS = {
    "source", "tag", "from", "to",
    "session", "top", "agent", "type", "active",
}

_TOKEN_RE = re.compile(r"\S+")


def parse(query: str) -> QueryAST:
    """Parse a DSL string into a ``QueryAST``.

    Raises ``DSLError`` on structurally bad input (currently: unparseable
    ``from:``/``to:`` dates). Unknown keys are *not* an error — they pass
    through to ``ast.extra`` for the source to decide about.
    """
    ast = QueryAST()
    if not query or not query.strip():
        return ast
    text_bits: list[str] = []
    for tok in _TOKEN_RE.findall(query):
        if ":" not in tok:
            text_bits.append(tok)
            continue
        key, _, val = tok.partition(":")  # partition = split on first colon
        key = key.lower()
        if key == "source":
            ast.source = val
        elif key == "tag":
            ast.tags.extend([t for t in val.split(",") if t])
        elif key == "from":
            ast.from_date = _parse_date(val, "from")
        elif key == "to":
            ast.to_date = _parse_date(val, "to")
        elif key == "session":
            ast.session_id = val
        elif key == "top":
            ast.top_session_id = val
        elif key == "agent":
            ast.agent_id = val
        elif key == "type":
            ast.obs_type = val
        elif key == "active":
            lo, hi = _parse_active_range(val)
            ast.active_from = lo
            ast.active_to = hi
        else:
            ast.extra[key] = val
    if text_bits:
        ast.text = " ".join(text_bits)
    return ast


def _parse_date(val: str, label: str) -> datetime:
    try:
        dt = datetime.fromisoformat(val)
    except ValueError as e:
        raise DSLError(
            f"bad {label}: date must be YYYY-MM-DD or ISO8601 (got {val!r})"
        ) from e
    # Bare dates parse tz-naive; force UTC so store comparisons are unambiguous.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _parse_active_range(val: str) -> tuple[datetime | None, datetime | None]:
    """Parse ``active:LO..HI`` (either side optional).

    Examples: ``active:2026-07-30..2026-08-01``, ``active:..2026-07-30``,
    ``active:2026-07-30..``. Empty string returns (None, None) — degenerate
    but not an error.
    """
    if ".." not in val:
        raise DSLError(
            f"bad active: expected LO..HI or ..HI or LO.. (got {val!r})"
        )
    lo_str, _, hi_str = val.partition("..")
    lo = _parse_date(lo_str, "active-from") if lo_str else None
    hi = _parse_date(hi_str, "active-to") if hi_str else None
    return lo, hi


def format_help(
    sources: Iterable[str],
    tags_by_source: dict[str, list[str]],
    date_range: tuple[str, str] | None = None,
) -> str:
    """Build a help block from actual cached inventory (Chughtai pattern).

    M3's ``aggregator_capabilities`` tool populates ``sources`` / ``tags_by_source``
    from the live store so the model always sees valid options.
    """
    lines = [
        "Aggregator DSL:",
        "  source:X tag:a,b from:YYYY-MM-DD to:YYYY-MM-DD [freeform FTS text]",
        "",
        "Sources currently cached:",
    ]
    for s in sources:
        lines.append(f"  - {s}")
    lines.append("")
    lines.append("Tags by source (top 20):")
    for s, ts in tags_by_source.items():
        lines.append(f"  {s}: {', '.join(ts[:20])}")
    if date_range:
        lines.append("")
        lines.append(f"Cached date range: {date_range[0]} .. {date_range[1]}")
    lines.append("")
    lines.append("Per-source keys (see aggregator_capabilities for the full list):")
    lines.append(
        "  github: state:open|closed check:pass|fail|pending "
        "mergeable:conflict author:@me"
    )
    lines.append("  sessions: project:<name> model:<name>")
    lines.append("")
    lines.append("Session-ontology keys (v2, Schema B):")
    lines.append("  session:<id>   everything under a session root (incl. subagents)")
    lines.append("  top:<id>       only the top-level stream (no subagents)")
    lines.append("  agent:<id>     only rows from that subagent")
    lines.append(
        "  type:<T>       observations of type user|assistant|tool_use|"
        "tool_result|system|other"
    )
    lines.append(
        "  active:LO..HI  sessions whose activity window overlaps [LO, HI] "
        "(dates ISO8601)"
    )
    return "\n".join(lines)
