"""Sessions source (v2, Schema B): emit ``SessionRow`` + ``ObservationRow``.

Walks ``~/.claude/projects/**/*.jsonl``. Each JSONL line becomes one
``ObservationRow``; each unique (file, sessionId) pair becomes one
``SessionRow``. Subagent files (``<sessionId>/subagents/agent-<agentId>.jsonl``)
become ``kind='subagent'`` sessions with synthesized composite key
``<parentSessionId>:<agentId>`` and ``parent_session_id`` derived from the
parent dir name.

Design decisions (documented for the migration):

* **Row-per-JSONL-line granularity** (spec §Work step 2). One observation per
  line — mirrors the file's own event granularity. Multi-block ``message.content``
  is collapsed: first text block into ``body``, first tool_use/tool_result block
  into ``tool_name`` / ``tool_use_id``. Multi-block messages are rare in
  practice; the collapse keeps queries simple. If richer per-block indexing is
  ever needed, split into multiple observations with synthesized child uuids.
* **Subagent detection**: by path first (``/subagents/agent-*.jsonl``) — the
  authoritative signal in current Claude Code layouts (2.1.x per research §5).
  Fallback: if every line has ``isSidechain: true``, treat as subagent even
  outside the expected path (legacy root-level layout).
* **Resume prefix-copy**: JSONL files under ``--resume`` start with a prefix
  copy of the parent session under the OLD sessionId, then switch to the NEW
  sessionId. We detect the switch and only ingest lines under the file's
  DOMINANT sessionId (the one the file was named after), matching research
  §5's guidance ("only ingest the NEW-sessionId portion under the new
  sessionId"). Non-matching-sessionId lines are dropped for that file — they
  belong to the parent file's stream.
* **Spawned_by_tool_use_id recovery**: best-effort. On first pass we collect
  Task ``tool_use`` entries per parent session with their ts + tool_use_id.
  On second pass, for each subagent, we look up Task calls in the parent whose
  ts is <= the subagent's first observation ts (closest predecessor within a
  60s window). Store ``None`` when ambiguous — concurrent subagents produce
  overlapping Task windows.
* **Live-file skip** (5-min window) preserved. File mtime is a container
  signal, per research §5 (never the source of truth for observation ts).
* **Timestamps** parsed from record ``timestamp``; session ``first_ts`` /
  ``last_ts`` = min/max across the session's own observations.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aggregator.sources.base import (
    IngestResult,
    ObservationRow,
    SessionEntity,
    SessionRow,
)

log = logging.getLogger(__name__)

LIVE_WINDOW_SECONDS = 5 * 60
SPAWN_LOOKBACK_SECONDS = 60  # Task tool_use → subagent-start ts window


# Line ``type`` values we still record (as observations of type ``system`` or
# ``other``) but that carry no message.content. We keep them because the model
# ``system`` events sometimes carry noteworthy state (permission-mode,
# rate_limit_event). Nothing here is dropped; the ``type`` column preserves
# provenance.
_KNOWN_TYPES = frozenset(
    {"user", "assistant", "tool_use", "tool_result", "system"}
)


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


_MAX_TOOL_INPUT_CHARS = 4096  # per obs row, keeps FTS index bounded


def _tool_use_body(tool_name: Any, tool_input: Any) -> str:
    """Serialise a tool_use block into a searchable body string.

    Format: ``"<tool_name> <compact-json-input>"`` so FTS finds both the
    tool name and any argument text (paths, queries, prompts). Input is
    truncated to ``_MAX_TOOL_INPUT_CHARS`` chars to keep the FTS index
    from blowing up on giant tool payloads (Write, Bash with big scripts).
    """
    name = tool_name if isinstance(tool_name, str) else ""
    if tool_input is None:
        return name
    try:
        payload = json.dumps(tool_input, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        payload = str(tool_input)
    if len(payload) > _MAX_TOOL_INPUT_CHARS:
        payload = payload[:_MAX_TOOL_INPUT_CHARS] + " …[truncated]"
    return f"{name} {payload}".strip() if name else payload


def _has_tool_result_block(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict) and block.get("type") == "tool_result"
        for block in content
    )


def _extract_text_and_tools(content: Any) -> tuple[str, str | None, str | None]:
    """From a ``message.content`` payload return ``(body_text, tool_name, tool_use_id)``.

    Per the design decision (docstring §Row-per-JSONL-line granularity):

    * ``body_text`` = first ``text`` block's ``text`` (or the whole thing when
      ``content`` is a bare string). Multi-block bodies collapse to the first
      text; documented tradeoff.
    * ``tool_name`` = first ``tool_use.name`` block's name, if any.
    * ``tool_use_id`` = first ``tool_use.id`` OR ``tool_result.tool_use_id``.

    Both tool fields may be non-None simultaneously — assistant messages can
    interleave text + tool_use in one message.content list.
    """
    if content is None:
        return "", None, None
    if isinstance(content, str):
        return content, None, None
    if not isinstance(content, list):
        return str(content), None, None

    body_text = ""
    tool_name: str | None = None
    tool_use_id: str | None = None
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text" and not body_text:
            t = block.get("text")
            if isinstance(t, str):
                body_text = t
        elif btype == "tool_use" and tool_name is None:
            n = block.get("name")
            if isinstance(n, str):
                tool_name = n
            tid = block.get("id")
            if isinstance(tid, str) and tool_use_id is None:
                tool_use_id = tid
            # B3 fix: fold tool name + input into body so FTS can find them
            # (previously tool_use rows had empty body → `type:tool_use foo`
            # queries never hit). Compact JSON keeps the text tokenisable
            # without wrapping quotes bloating the index.
            if not body_text:
                body_text = _tool_use_body(n, block.get("input"))
        elif btype == "tool_result" and tool_use_id is None:
            tid = block.get("tool_use_id")
            if isinstance(tid, str):
                tool_use_id = tid
            # Record tool_result content as body text so FTS catches it.
            if not body_text:
                rc = block.get("content")
                if isinstance(rc, str):
                    body_text = rc
                elif isinstance(rc, list):
                    for sub in rc:
                        if isinstance(sub, dict) and sub.get("type") == "text":
                            t = sub.get("text")
                            if isinstance(t, str):
                                body_text = t
                                break
        elif btype == "thinking" and not body_text:
            t = block.get("text") or block.get("thinking")
            if isinstance(t, str):
                body_text = t
    return body_text, tool_name, tool_use_id


@dataclass
class _ParsedLine:
    """One JSONL line, parsed enough to bucket into sessions."""

    session_id: str
    uuid: str
    parent_uuid: str | None
    ts: datetime | None
    line_type: str
    is_sidechain: bool
    agent_id: str | None
    agent_type: str | None
    cwd: str | None
    git_branch: str | None
    body: str
    tool_name: str | None
    tool_use_id: str | None
    model: str | None
    input_tokens: int | None
    output_tokens: int | None


class SessionsSource:
    """v2 sessions source. Yields ``SessionRow`` + ``ObservationRow``.

    Ingest lifecycle (called from CLI):
    1. First pass — walk every JSONL, collect Task ``tool_use`` ts+id per
       top-level session (for spawn-id recovery).
    2. Second pass — walk every JSONL, emit sessions + observations. Subagents
       receive ``spawned_by_tool_use_id`` from the pass-1 index when a unique
       predecessor exists.
    """

    name = "sessions"

    def __init__(self, projects_root: str | None = None):
        self.projects_root = Path(
            projects_root or os.path.expanduser("~/.claude/projects")
        )

    def record_shape(self) -> dict[str, str]:
        return {
            "session_id": "str",
            "root_session_id": "str",
            "parent_session_id": "str | None",
            "kind": "'session' | 'subagent'",
            "agent_id": "str | None",
            "agent_type": "str | None",
            "spawned_by_tool_use_id": "str | None (recovered from Task tool_use window)",
            "cwd": "str | None",
            "git_branch": "str | None",
            "first_ts": "datetime (min message ts)",
            "last_ts": "datetime (max message ts)",
            "obs_type": "'user'|'assistant'|'tool_use'|'tool_result'|'system'|'other'",
        }

    # -- filesystem walk --------------------------------------------------

    def _iter_jsonl_files(self) -> Iterator[Path]:
        """Yield every non-live JSONL under the projects root.

        Live-file skip (5-min window) matches v1 semantics — an actively
        appended file could split observations mid-record. File mtime is a
        container signal only (research §5); we still consult it here to
        avoid reading a partial line.
        """
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

    @staticmethod
    def _is_subagent_path(path: Path) -> bool:
        """Path-based subagent detection (research §5, current 2.1.x layout).

        ``.../<sessionId>/subagents/agent-<agentId>.jsonl`` → subagent.
        Legacy ``.../agent-*.jsonl`` at project root also matched.
        """
        parts = path.parts
        if "subagents" in parts:
            return True
        # Legacy layout — ``agent-<hex>.jsonl`` at project root, no subagents/ dir
        # in path. Only claim it as a subagent if the filename actually matches.
        return path.name.startswith("agent-") and path.suffix == ".jsonl"

    @staticmethod
    def _parent_session_from_path(path: Path) -> str | None:
        """For a subagent file, return the parent's sessionId from the dir name.

        Current layout: ``<project>/<sessionId>/subagents/agent-<agentId>.jsonl``
        → parent = ``<sessionId>``. Legacy layout puts ``agent-*.jsonl`` at the
        project root with no session-scoped parent dir; return None so the
        caller records ``parent_session_id=NULL``.
        """
        parts = path.parts
        if "subagents" in parts:
            idx = parts.index("subagents")
            if idx > 0:
                return parts[idx - 1]
        return None

    @staticmethod
    def _agent_id_from_path(path: Path) -> str | None:
        """Extract ``<agentId>`` from ``agent-<agentId>.jsonl``."""
        stem = path.stem
        if stem.startswith("agent-"):
            return stem[len("agent-"):]
        return None

    # -- JSONL line parsing ----------------------------------------------

    @staticmethod
    def _parse_line(obj: dict) -> _ParsedLine | None:
        sid = obj.get("sessionId")
        uid = obj.get("uuid")
        if not isinstance(sid, str) or not sid:
            return None
        if not isinstance(uid, str) or not uid:
            # Control lines (queue-operation, rate_limit_event ...) often omit
            # uuid; without one we have no primary key, so drop.
            return None
        parent_uuid = obj.get("parentUuid")
        if not isinstance(parent_uuid, str):
            parent_uuid = None
        ts = _parse_iso(obj.get("timestamp"))
        line_type = obj.get("type") or "other"
        if not isinstance(line_type, str):
            line_type = "other"
        if line_type not in _KNOWN_TYPES:
            line_type = "other"
        is_sidechain = bool(obj.get("isSidechain"))
        agent_id = obj.get("agentId") if isinstance(obj.get("agentId"), str) else None
        agent_type = obj.get("agentType") if isinstance(obj.get("agentType"), str) else None
        cwd = obj.get("cwd") if isinstance(obj.get("cwd"), str) else None
        git_branch = obj.get("gitBranch") if isinstance(obj.get("gitBranch"), str) else None

        body_text = ""
        tool_name: str | None = None
        tool_use_id: str | None = None
        model: str | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None

        message = obj.get("message")
        if isinstance(message, dict):
            body_text, tool_name, tool_use_id = _extract_text_and_tools(
                message.get("content")
            )
            m = message.get("model")
            if isinstance(m, str):
                model = m
            usage = message.get("usage")
            if isinstance(usage, dict):
                it = usage.get("input_tokens")
                ot = usage.get("output_tokens")
                if isinstance(it, int):
                    input_tokens = it
                if isinstance(ot, int):
                    output_tokens = ot

        # Refine line_type from message content when the top-level ``type`` is
        # generic. E.g. an assistant line with a tool_use block gets promoted
        # to ``tool_use`` observation type so the DSL type: filter finds it.
        # A user line whose message.content is a tool_result block set gets
        # promoted to 'tool_result' (detection: message.content is a list
        # containing at least one tool_result block).
        if line_type == "assistant" and tool_name:
            line_type = "tool_use"
        elif line_type == "user" and _has_tool_result_block(
            message.get("content") if isinstance(message, dict) else None
        ):
            line_type = "tool_result"

        return _ParsedLine(
            session_id=sid,
            uuid=uid,
            parent_uuid=parent_uuid,
            ts=ts,
            line_type=line_type,
            is_sidechain=is_sidechain,
            agent_id=agent_id,
            agent_type=agent_type,
            cwd=cwd,
            git_branch=git_branch,
            body=body_text,
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _iter_parsed(
        self, path: Path, errors: list[str]
    ) -> Iterator[_ParsedLine]:
        try:
            fh = path.open(encoding="utf-8", errors="replace")
        except OSError as e:
            errors.append(f"{path}: open failed: {e}")
            return
        try:
            for lineno, raw in enumerate(fh, 1):
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
                parsed = self._parse_line(obj)
                if parsed is not None:
                    yield parsed
        finally:
            fh.close()

    @staticmethod
    def _dominant_session_id(parsed_lines: list[_ParsedLine], path: Path) -> str | None:
        """For a top-level file, pick the sessionId that "owns" the file.

        Resume produces a prefix copy under the parent's sessionId, then the
        real (new) sessionId. The filename is the NEW sessionId. Match by
        filename stem when possible; fall back to the sessionId with the most
        lines.
        """
        stem = path.stem
        counts: dict[str, int] = {}
        for p in parsed_lines:
            counts[p.session_id] = counts.get(p.session_id, 0) + 1
        if not counts:
            return None
        if stem in counts:
            return stem
        # Fallback: max-count sessionId (deterministic tiebreak by string).
        return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]

    # -- Pass 1: collect Task tool_use ts per top-level session ----------

    def _collect_task_calls(
        self, errors: list[str]
    ) -> dict[str, list[tuple[datetime, str]]]:
        """Return ``{parent_session_id: [(ts, tool_use_id), ...]}`` for Task
        tool_use entries in every top-level JSONL. Used in pass 2 to recover
        ``spawned_by_tool_use_id`` for subagents.
        """
        index: dict[str, list[tuple[datetime, str]]] = {}
        for path in self._iter_jsonl_files():
            if self._is_subagent_path(path):
                continue
            parsed = list(self._iter_parsed(path, errors))
            dominant = self._dominant_session_id(parsed, path)
            if dominant is None:
                continue
            for p in parsed:
                if p.session_id != dominant:
                    continue
                if p.tool_name == "Task" and p.tool_use_id and p.ts:
                    index.setdefault(dominant, []).append((p.ts, p.tool_use_id))
        # Sort each list by ts ascending — used for closest-predecessor lookup.
        for lst in index.values():
            lst.sort(key=lambda t: t[0])
        return index

    @staticmethod
    def _recover_spawn_tool_use_id(
        task_index: dict[str, list[tuple[datetime, str]]],
        parent_session_id: str | None,
        subagent_first_ts: datetime | None,
    ) -> str | None:
        """Best-effort: closest Task ``tool_use_id`` in the parent whose ts
        precedes the subagent's first observation by ≤ ``SPAWN_LOOKBACK_SECONDS``.

        Returns ``None`` when:
        * parent session unknown (legacy layout),
        * no Task tool_use in that parent,
        * two or more Task calls fall in the window (ambiguous → concurrent
          subagents, we refuse to guess),
        * subagent has no observations (nothing to anchor to).
        """
        if not parent_session_id or subagent_first_ts is None:
            return None
        calls = task_index.get(parent_session_id, [])
        if not calls:
            return None
        window_start = subagent_first_ts.timestamp() - SPAWN_LOOKBACK_SECONDS
        window_end = subagent_first_ts.timestamp()
        candidates = [
            (ts, tid) for ts, tid in calls
            if window_start <= ts.timestamp() <= window_end
        ]
        if len(candidates) == 1:
            return candidates[0][1]
        return None

    # -- Pass 2: yield SessionRow + ObservationRow ------------------------

    def iter_entities(
        self,
        since: datetime | None = None,
        errors: list[str] | None = None,
    ) -> Iterator[SessionEntity]:
        """Main ingest entrypoint. Yields SessionRow before its ObservationRows.

        ``since`` filters observations by ts (advisory — a session with any
        observation past ``since`` is emitted in full).
        """
        sink = errors if errors is not None else []
        task_index = self._collect_task_calls(sink)

        for path in self._iter_jsonl_files():
            yield from self._iter_file_entities(path, task_index, since, sink)

    def _iter_file_entities(
        self,
        path: Path,
        task_index: dict[str, list[tuple[datetime, str]]],
        since: datetime | None,
        errors: list[str],
    ) -> Iterator[SessionEntity]:
        parsed = list(self._iter_parsed(path, errors))
        if not parsed:
            return

        is_subagent_file = self._is_subagent_path(path)
        if is_subagent_file:
            # All lines in a subagent file belong to that stream. Group by
            # sessionId still (defensive; typically only one).
            for entity in self._emit_subagent(path, parsed, task_index, since):
                yield entity
        else:
            dominant = self._dominant_session_id(parsed, path)
            if dominant is None:
                return
            # Only ingest lines matching the dominant sessionId — drops the
            # resume prefix-copy portion (which belongs to the parent file).
            own_lines = [p for p in parsed if p.session_id == dominant]
            for entity in self._emit_top_level(path, dominant, own_lines, since):
                yield entity

    def _emit_top_level(
        self,
        path: Path,
        session_id: str,
        lines: list[_ParsedLine],
        since: datetime | None,
    ) -> Iterator[SessionEntity]:
        if not lines:
            return
        tss = [p.ts for p in lines if p.ts is not None]
        if not tss:
            return
        first_ts, last_ts = min(tss), max(tss)
        since_utc = _normalise_utc(since) if since else None
        if since_utc and last_ts < since_utc:
            return
        cwd = next((p.cwd for p in lines if p.cwd), None)
        git_branch = next((p.git_branch for p in lines if p.git_branch), None)
        yield SessionRow(
            session_id=session_id,
            root_session_id=session_id,
            parent_session_id=None,
            kind="session",
            agent_id=None,
            agent_type=None,
            spawned_by_tool_use_id=None,
            cwd=cwd,
            git_branch=git_branch,
            first_ts=first_ts,
            last_ts=last_ts,
            jsonl_path=str(path),
        )
        for p in lines:
            if p.ts is None:
                continue
            yield ObservationRow(
                obs_id=p.uuid,
                session_id=session_id,
                root_session_id=session_id,
                parent_obs_id=p.parent_uuid,
                type=p.line_type,
                ts=p.ts,
                model=p.model,
                input_tokens=p.input_tokens,
                output_tokens=p.output_tokens,
                tool_name=p.tool_name,
                tool_use_id=p.tool_use_id,
                body=p.body,
            )

    def _emit_subagent(
        self,
        path: Path,
        lines: list[_ParsedLine],
        task_index: dict[str, list[tuple[datetime, str]]],
        since: datetime | None,
    ) -> Iterator[SessionEntity]:
        parent_sid = self._parent_session_from_path(path)
        # Agent id: prefer the file name (authoritative); fall back to first
        # line's ``agentId`` field.
        agent_id = self._agent_id_from_path(path)
        if agent_id is None:
            agent_id = next((p.agent_id for p in lines if p.agent_id), None)
        if agent_id is None:
            # Nothing to name the composite key with. Give up on this file —
            # would collide with any other unknown-id subagent.
            return
        # Composite key: parent_sid may be None in legacy layout; use "orphan"
        # sentinel so keys still collide-avoid across orphan subagents.
        composite_key = f"{parent_sid or 'orphan'}:{agent_id}"
        root_session_id = parent_sid or composite_key
        # A subagent file may have multiple sessionIds in it (e.g. its own
        # unique id + prefix-copied context). Only take the lines whose
        # sessionId matches the file's dominant id.
        dominant = self._dominant_session_id(lines, path)
        own_lines = [p for p in lines if p.session_id == dominant]

        tss = [p.ts for p in own_lines if p.ts is not None]
        if not tss:
            return
        first_ts, last_ts = min(tss), max(tss)
        since_utc = _normalise_utc(since) if since else None
        if since_utc and last_ts < since_utc:
            return
        agent_type = next((p.agent_type for p in own_lines if p.agent_type), None)
        spawned = self._recover_spawn_tool_use_id(task_index, parent_sid, first_ts)
        cwd = next((p.cwd for p in own_lines if p.cwd), None)
        git_branch = next((p.git_branch for p in own_lines if p.git_branch), None)
        yield SessionRow(
            session_id=composite_key,
            root_session_id=root_session_id,
            parent_session_id=parent_sid,
            kind="subagent",
            agent_id=agent_id,
            agent_type=agent_type,
            spawned_by_tool_use_id=spawned,
            cwd=cwd,
            git_branch=git_branch,
            first_ts=first_ts,
            last_ts=last_ts,
            jsonl_path=str(path),
        )
        for p in own_lines:
            if p.ts is None:
                continue
            yield ObservationRow(
                obs_id=p.uuid,
                session_id=composite_key,
                root_session_id=root_session_id,
                parent_obs_id=p.parent_uuid,
                type=p.line_type,
                ts=p.ts,
                model=p.model,
                input_tokens=p.input_tokens,
                output_tokens=p.output_tokens,
                tool_name=p.tool_name,
                tool_use_id=p.tool_use_id,
                body=p.body,
            )

    # -- Protocol methods -------------------------------------------------

    def ingest(self, since: datetime | None) -> IngestResult:
        """Count-only path retained for protocol compat + integration tests.

        Persistence is the CLI's job — this method only counts sessions +
        observations emitted, and surfaces per-file parse errors.
        """
        errors: list[str] = []
        sessions = 0
        observations = 0
        for e in self.iter_entities(since, errors=errors):
            if isinstance(e, SessionRow):
                sessions += 1
            elif isinstance(e, ObservationRow):
                observations += 1
        return IngestResult(
            added=sessions + observations,
            updated=0,
            skipped=0,
            errors=errors,
        )


def _normalise_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
