"""Sessions source: walk Claude Code JSONL project logs into Records.

Skips files modified within the last 5 minutes (live session heuristic).
Corrupt JSONL lines are skipped and logged, never abort the ingest.

Real Claude Code JSONL shape (empirically verified 2026-08-01):
    Per-line JSON with a ``type`` field. There is NO synthetic
    ``session_start`` / ``session_end`` bracketing — sessions are
    identified by the ``sessionId`` field present on every message-type
    line. Types observed include ``queue-operation`` (control, no cwd),
    ``user``, ``assistant``, ``system``, ``rate_limit_event``,
    ``tool_use``, ``tool_result``, ``permission-mode``,
    ``file-history-snapshot``, ``attachment``, ``last-prompt``.

    Only ``user`` / ``assistant`` lines carry conversational content in
    ``message.content`` (list of blocks: ``text``, ``tool_use``,
    ``tool_result``, ``thinking``, ...). ``message.model`` appears on
    assistant lines. Timestamps are ISO-8601 with ``Z``.

    Grouping is per ``sessionId`` INSIDE each file (rare but possible
    to see mixed sessionIds in one file — we handle it).
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Reserved seam: LLM wrapper for future ingest enrichment (see spec §Error
# handling). v1 makes no LLM calls, so the import previously wired here was
# dropped (advisor round-1 MEDIUM: dead `run_sync` import). When enrichment
# lands, wire the runner at the call site, not as an unused module-level
# import.
from aggregator.sources.base import IngestResult, QueryAST, Record

log = logging.getLogger(__name__)

LIVE_WINDOW_SECONDS = 5 * 60

# Line ``type`` values that carry no conversational content — skipped for
# text extraction but still consulted for ``sessionId`` grouping when the
# field is present (rare; control lines usually omit it or share the file's
# sole session).
_CONTROL_TYPES = frozenset(
    {
        "queue-operation",
        "permission-mode",
        "file-history-snapshot",
        "attachment",
        "last-prompt",
        "rate_limit_event",
        "system",
    }
)


def _project_from_cwd(cwd: str) -> str:
    """Encoded cwd -> short project label (last path component)."""
    if not cwd:
        return "unknown"
    return Path(cwd).name or "unknown"


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _extract_text(content: Any) -> str:
    """Flatten a message.content payload into plain text.

    Real content is either a string (older sessions) or a list of blocks
    like ``{"type": "text", "text": "..."}``. Non-text blocks (tool_use,
    tool_result, thinking) are skipped for the conversational body — the
    tool call counter picks up ``tool_use`` names separately.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    # Unknown shape — degrade to str() so callers still see something.
    return str(content)


def _iter_tool_use_names(content: Any) -> Iterator[str]:
    """Yield tool_use ``name`` values from an assistant message.content list."""
    if not isinstance(content, list):
        return
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            name = block.get("name")
            if isinstance(name, str) and name:
                yield name


@dataclass
class _SessionAggregate:
    session_id: str
    project: str = "unknown"
    started: datetime | None = None
    ended: datetime | None = None
    model: str = ""
    cost_usd: float | None = None
    user_turns: list[str] = field(default_factory=list)
    assistant_turns: list[str] = field(default_factory=list)
    tool_call_counter: Counter = field(default_factory=Counter)
    first_user_prompt: str = ""


class SessionsSource:
    name = "sessions"

    def __init__(self, projects_root: str | None = None):
        self.projects_root = Path(
            projects_root or os.path.expanduser("~/.claude/projects")
        )

    def record_shape(self) -> dict[str, str]:
        return {
            "session_id": "str",
            "project": "str (last path segment of cwd)",
            "started": "datetime UTC",
            "ended": "datetime UTC",
            "model": "str",
            "cost_usd": "float | None (real logs carry usage, not cost)",
            "first_user_prompt": "str",
            "top_tool_calls": "list[{name, count}]",
            "tail_summary": "str (last user + assistant turn)",
        }

    def _iter_jsonl_files(self) -> Iterator[Path]:
        if not self.projects_root.exists():
            return
        now = time.time()
        for path in self.projects_root.rglob("*.jsonl"):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if now - mtime < LIVE_WINDOW_SECONDS:
                continue
            yield path

    def _parse_file(
        self, path: Path, errors: list[str]
    ) -> Iterator[_SessionAggregate]:
        """Group lines by ``sessionId`` within the file, yield one
        aggregate per group.

        Real Claude Code JSONLs have no explicit start/end record — every
        message-type line carries ``sessionId``. Multiple sessionIds per
        file are possible (rare), so we bucket rather than assume one
        session per file. Missing-sessionId lines (control chatter) are
        threaded onto the most recently seen sessionId or discarded when
        no session has appeared yet.
        """
        aggregates: dict[str, _SessionAggregate] = {}
        # Preserve first-seen order for deterministic yield.
        order: list[str] = []
        last_session_id: str | None = None

        def _get(sid: str) -> _SessionAggregate:
            agg = aggregates.get(sid)
            if agg is None:
                agg = _SessionAggregate(session_id=sid)
                aggregates[sid] = agg
                order.append(sid)
            return agg

        with path.open(encoding="utf-8", errors="replace") as f:
            for lineno, raw in enumerate(f, 1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    errors.append(f"{path}:{lineno} corrupt line")
                    continue
                if not isinstance(obj, dict):
                    continue

                sid = obj.get("sessionId")
                if not isinstance(sid, str) or not sid:
                    # Some pure-control lines omit sessionId; fall back to
                    # the last one we saw so their timestamp still updates
                    # the aggregate window if useful. If there is none
                    # yet, skip — nothing to attach to.
                    sid = last_session_id
                    if sid is None:
                        continue
                last_session_id = sid

                agg = _get(sid)

                # Timestamp window (every message-type line has one).
                ts = _parse_iso(obj.get("timestamp"))
                if ts is not None:
                    if agg.started is None or ts < agg.started:
                        agg.started = ts
                    if agg.ended is None or ts > agg.ended:
                        agg.ended = ts

                # cwd -> project label (first non-null wins).
                if agg.project == "unknown":
                    cwd = obj.get("cwd")
                    if isinstance(cwd, str) and cwd:
                        agg.project = _project_from_cwd(cwd)

                t = obj.get("type")
                if t in _CONTROL_TYPES:
                    continue

                message = obj.get("message")
                if not isinstance(message, dict):
                    continue
                content = message.get("content")

                if t == "user":
                    text = _extract_text(content)
                    if text:
                        if not agg.first_user_prompt:
                            agg.first_user_prompt = text[:280]
                        agg.user_turns.append(text)
                elif t == "assistant":
                    text = _extract_text(content)
                    if text:
                        agg.assistant_turns.append(text)
                    model = message.get("model")
                    if isinstance(model, str) and model:
                        # Last-seen assistant model wins.
                        agg.model = model
                    for name in _iter_tool_use_names(content):
                        agg.tool_call_counter[name] += 1

        for sid in order:
            agg = aggregates[sid]
            # Fall back to a stable project label when cwd never appeared
            # (all control lines) — preserves _project_from_cwd("")'s
            # "unknown" default without a second helper call.
            yield agg

    def _iter_records(self) -> Iterator[Record]:
        """Legacy alias for ``iter_records(since=None)``.

        Kept for the existing record-level tests (M1a) that predate the
        ``iter_records`` protocol method. Both entrypoints share the same
        underlying pass so the fixtures/behaviour don't diverge.
        """
        yield from self.iter_records(since=None)

    def iter_records(
        self,
        since: datetime | None,
        errors: list[str] | None = None,
    ) -> Iterator[Record]:
        """Yield records parsed from every JSONL project file.

        Applies the same ``since`` bound as the old ``ingest``: skip records
        whose ``ended`` timestamp precedes ``since``. The caller is
        responsible for persisting the yielded records (see
        ``cli._cmd_ingest``).

        When ``errors`` is provided, parse errors are appended so callers
        that need structured error surfacing (``ingest`` returns them in
        ``IngestResult.errors``) can share the same iteration path — no
        double-walk of the JSONL tree required (round-2 MEDIUM).
        """
        sink = errors if errors is not None else []
        for path in self._iter_jsonl_files():
            for agg in self._parse_file(path, sink):
                if since and agg.ended:
                    # Round-3 LOW: JSONL logs with a Z suffix parse to
                    # UTC-aware; logs missing tz yield naive datetimes.
                    # Comparing naive < aware raises TypeError in Py3 —
                    # a malformed log used to abort the whole ingest.
                    # Normalise both sides to UTC-aware for the compare.
                    ended = (
                        agg.ended
                        if agg.ended.tzinfo is not None
                        else agg.ended.replace(tzinfo=UTC)
                    )
                    since_utc = (
                        since
                        if since.tzinfo is not None
                        else since.replace(tzinfo=UTC)
                    )
                    if ended < since_utc:
                        continue
                yield self._to_record(agg, source_path=path)

    def _to_record(self, agg: _SessionAggregate, source_path: Path) -> Record:
        tail = ""
        if agg.user_turns:
            tail += f"USER: {agg.user_turns[-1]}\n"
        if agg.assistant_turns:
            tail += f"ASSISTANT: {agg.assistant_turns[-1]}\n"
        body = "\n".join(
            [f"USER: {u}" for u in agg.user_turns]
            + [f"ASSISTANT: {a}" for a in agg.assistant_turns]
        )
        top_tools = [
            {"name": n, "count": c}
            for n, c in agg.tool_call_counter.most_common(10)
        ]
        tags = [agg.project]
        if agg.model:
            tags.append(agg.model)
        return Record(
            stable_id=f"sessions:{agg.session_id}",
            source="sessions",
            subject=agg.first_user_prompt or f"session {agg.session_id}",
            body=body,
            tags=tags,
            created_at=agg.started,
            updated_at=agg.ended or agg.started,
            extra={
                "session_id": agg.session_id,
                "project": agg.project,
                "model": agg.model,
                "cost_usd": agg.cost_usd,
                "first_user_prompt": agg.first_user_prompt,
                "top_tool_calls": top_tools,
                "tail_summary": tail,
                "source_path": str(source_path),
            },
        )

    def ingest(self, since: datetime | None) -> IngestResult:
        """Count-only path retained for protocol compat + integration tests.

        Round-2 MEDIUM: pre-fix, this re-walked the JSONL tree once purely
        to collect parse errors (because ``iter_records`` had no way to
        surface them), then walked again via ``iter_records`` to count.
        Post-fix, ``iter_records`` accepts a shared ``errors`` sink, so
        one pass produces both the count and the parse-error list.

        Persistence is the CLI's job — this method only counts.
        """
        errors: list[str] = []
        added = sum(1 for _ in self.iter_records(since, errors=errors))
        return IngestResult(added=added, updated=0, skipped=0, errors=errors)

    def search(self, ast: QueryAST) -> list[Record]:
        """Sessions.search() is a thin passthrough; M2's Store handles FTS.
        In M1a this iterates + filters in Python for the record-level tests."""
        out: list[Record] = []
        for r in self._iter_records():
            if ast.from_date and r.created_at and r.created_at < ast.from_date:
                continue
            if ast.to_date and r.created_at and r.created_at > ast.to_date:
                continue
            if ast.text and ast.text.lower() not in r.body.lower():
                continue
            out.append(r)
        return out
