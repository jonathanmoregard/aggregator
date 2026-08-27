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
from datetime import UTC, datetime, timedelta

from aggregator.core.provenance import MACHINE, PROVENANCE_VALUES
from aggregator.sources.base import SCOPE_VALUES, QueryAST


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
#
# v6 authorship key:
#   by:P        — filter observations by WHO COMPOSED them
#                 (human|agent|hook|command|system, plus the shorthand
#                 ``machine`` for any of the four non-human members).
#                 ``type:`` is the channel a line arrived on; this is the
#                 author. Absent means NO FILTER, exactly like ``type:``.
#
# v6 conjunction-scope key:
#   scope:U     — the UNIT every free-text term must be satisfied in
#                 (observation|session). Absent = ``observation``: one turn has
#                 to carry all of them. ``scope:session`` lets the terms be
#                 spread across different turns of one session. It is NOT a row
#                 filter — it changes what the text matches — so it is applied
#                 in ``Store._text_hit_scope``, not in ``Store._obs_where``.
KNOWN_KEYS = {
    "source", "tag", "from", "to",
    "session", "top", "agent", "type", "active", "by", "scope",
}

#: Everything ``by:`` accepts. A CLOSED SET, which is why an unknown value is a
#: parse error rather than an empty page — see ``_parse_provenance``.
BY_VALUES: tuple[str, ...] = (*PROVENANCE_VALUES, MACHINE)

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
        elif key == "by":
            ast.provenance = _parse_provenance(val)
        elif key == "scope":
            ast.scope = _parse_scope(val)
        elif key == "active":
            lo, hi = _parse_active_range(val)
            ast.active_from = lo
            ast.active_to = hi
        else:
            ast.extra[key] = val
    if text_bits:
        ast.text = " ".join(text_bits)
    return ast


def _parse_provenance(val: str) -> str:
    """Validate ``by:``. An unknown value is an ERROR, not an empty page.

    ``type:`` is deliberately open — its value set is additive, and the store
    comment says so — so a typo there returns nothing and that is tolerable.
    ``by:`` has exactly six accepted values, so the same silence would be
    indistinguishable from "there is nothing in your history", which is the
    failure this whole change exists to remove. Naming the accepted set in the
    error means one round trip instead of a guess.
    """
    value = val.strip().lower()
    if value not in BY_VALUES:
        raise DSLError(
            f"bad by: {val!r} is not an authorship class. Use one of "
            f"{', '.join(BY_VALUES)} — ``machine`` is the shorthand for any of "
            f"{', '.join(v for v in PROVENANCE_VALUES if v != 'human')}."
        )
    return value


def _parse_scope(val: str) -> str:
    """Validate ``scope:``. Closed set, so an unknown value is an ERROR.

    Same argument as ``by:``: two accepted values means a typo would otherwise
    come back as a silently DIFFERENT question rather than as an empty page —
    ``scope:sessions`` landing in ``ast.extra`` and quietly leaving the default
    in place is precisely the kind of silence this mission exists to remove.
    """
    value = val.strip().lower()
    if value not in SCOPE_VALUES:
        raise DSLError(
            f"bad scope: {val!r} is not a conjunction scope. Use one of "
            f"{', '.join(SCOPE_VALUES)} — 'observation' (the default) requires "
            f"every term in ONE turn, 'session' lets them be spread across "
            f"different turns of one session."
        )
    return value


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


_BARE_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _parse_active_range(val: str) -> tuple[datetime | None, datetime | None]:
    """Parse ``active:LO..HI`` (either side optional).

    Examples: ``active:2026-07-30..2026-08-01``, ``active:..2026-07-30``,
    ``active:2026-07-30..``. Empty string returns (None, None) — degenerate
    but not an error.

    Codex Phase 2 MEDIUM: a bare-date HI (``YYYY-MM-DD``) is treated as
    the exclusive start of the NEXT day so the range is inclusive of
    everything on day HI (documented semantics). A full ISO datetime is
    honoured as-is. Callers rely on this at the store layer where the
    comparison is ``first_ts < active_to``.
    """
    if ".." not in val:
        raise DSLError(
            f"bad active: expected LO..HI or ..HI or LO.. (got {val!r})"
        )
    lo_str, _, hi_str = val.partition("..")
    lo = _parse_date(lo_str, "active-from") if lo_str else None
    hi: datetime | None = None
    if hi_str:
        hi = _parse_date(hi_str, "active-to")
        if _BARE_DATE_RE.match(hi_str):
            # End-of-day inclusive: HI covers everything on that date without
            # crossing midnight (store predicate is ``first_ts <= active_to``).
            hi = hi + timedelta(days=1) - timedelta(microseconds=1)
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
        "                 NOTE: type: is a TRANSPORT ROLE, not an authorship "
        "claim. type:user means the line arrived on the user channel; measured, "
        "59% of those were composed by a machine (hook prompts, headless "
        "briefs, subagent briefs, slash-command output, client notices). Use "
        "by: for authorship."
    )
    lines.append(
        f"  by:<P>         WHO COMPOSED the observation: "
        f"{'|'.join(BY_VALUES)}. 'machine' is any of "
        f"{'/'.join(v for v in PROVENANCE_VALUES if v != 'human')}; 'human' is "
        f"a residual (nothing claimed a machine wrote it), never a positive "
        f"identification. Absent = no filter. Rows not yet classified are "
        f"excluded by EVERY value of by: — run "
        f"`aggregator provenance --backfill`."
    )
    lines.append(
        "  active:LO..HI  sessions whose activity window overlaps [LO, HI] "
        "(dates ISO8601)"
    )
    lines.append(
        f"  scope:<U>      the UNIT every free-text term must be found in: "
        f"{'|'.join(SCOPE_VALUES)}. DEFAULT is 'observation' — all terms in "
        f"ONE turn. A double-quoted run is one term, so \"PR link\" matches "
        f"those two words next to each other and PR link matches them "
        f"anywhere in the turn. scope:session widens it: the terms may sit in "
        f"different turns of the same session, hours apart, which answers "
        f"'which session covered both' rather than 'which moment said it'."
    )
    return "\n".join(lines)
