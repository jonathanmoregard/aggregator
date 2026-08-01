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
KNOWN_KEYS = {"source", "tag", "from", "to"}

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
    return "\n".join(lines)
