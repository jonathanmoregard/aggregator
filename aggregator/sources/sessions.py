"""Sessions source: walk Claude Code JSONL project logs into Records.

Skips files modified within the last 5 minutes (live session heuristic).
Corrupt JSONL lines are skipped and logged, never abort the ingest.
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

# Reserved seam: LLM wrapper for future ingest enrichment (see spec §Error
# handling). v1 makes no LLM calls, so the import previously wired here was
# dropped (advisor round-1 MEDIUM: dead `run_sync` import). When enrichment
# lands, wire the runner at the call site, not as an unused module-level
# import.
from aggregator.sources.base import IngestResult, QueryAST, Record

log = logging.getLogger(__name__)

LIVE_WINDOW_SECONDS = 5 * 60


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


@dataclass
class _SessionAggregate:
    session_id: str
    project: str = "unknown"
    started: datetime | None = None
    ended: datetime | None = None
    model: str = ""
    cost_usd: float = 0.0
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
            "cost_usd": "float",
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
        current: _SessionAggregate | None = None
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
                t = obj.get("type")
                if t == "session_start":
                    current = _SessionAggregate(
                        session_id=obj.get("session_id", f"unknown-{path.name}"),
                        project=_project_from_cwd(obj.get("cwd", "")),
                        started=_parse_iso(obj.get("started_at")),
                        model=obj.get("model", ""),
                    )
                elif t == "user" and current is not None:
                    text = obj.get("text", "")
                    if not current.first_user_prompt:
                        current.first_user_prompt = text[:280]
                    current.user_turns.append(text)
                elif t == "assistant" and current is not None:
                    current.assistant_turns.append(obj.get("text", ""))
                    for tc in obj.get("tool_calls", []) or []:
                        name = tc.get("name", "unknown")
                        count = tc.get("count", 1)
                        current.tool_call_counter[name] += count
                elif t == "session_end" and current is not None:
                    current.ended = _parse_iso(obj.get("ended_at"))
                    current.cost_usd = float(obj.get("cost_usd", 0.0))
                    yield current
                    current = None
            # yield unclosed session at EOF too (crashed session)
            if current is not None:
                yield current

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
