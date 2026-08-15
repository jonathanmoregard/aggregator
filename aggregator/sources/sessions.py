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
* **Spawned_by_tool_use_id recovery** (B1 fix — real-data-verified). Claude
  Code 2026 layout emits async subagent launches via the ``Agent`` tool
  (previously assumed ``Task``, which produced 0/1170 recovery). The
  ``tool_result`` for that ``Agent`` call carries the child ``agentId`` in TWO
  places:

  1. As a top-level ``toolUseResult.agentId`` sibling to ``message`` — the
     structured, authoritative source.
  2. Embedded in the tool_result text as ``"agentId: <id> (internal ID …)"``
     — the LLM-facing echo string. Same value.

  Pass 1 walks every top-level JSONL, collects ``{parent_session_id:
  {child_agent_id: tool_use_id}}`` from either channel (structured first,
  text-regex fallback). Pass 2 looks up each subagent's ``agent_id`` in its
  parent's map — an EXACT-MATCH JOIN, no fuzzy time windows. Real-cache
  measurement: 1084/1207 = 89.8% recovery (up from 0.0%). The remaining ~10%
  is truly irrecoverable — parent JSONL not on disk (deleted / not yet
  ingested / cache leftover from moved projects). ``None`` is preserved for
  those.
* **Live-file skip** (5-min window) preserved. File mtime is a container
  signal, per research §5 (never the source of truth for observation ts).
* **Timestamps** parsed from record ``timestamp``; session ``first_ts`` /
  ``last_ts`` = min/max across the session's own observations.
"""
from __future__ import annotations

import json
import logging
import os
import re
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

# Regex for the LLM-facing echo of a spawned agentId inside a tool_result
# text block: e.g. "agentId: a54062c3d0038bcfa (internal ID - do not mention…)".
# Character class permits hex ids AND human-tagged ones (e.g.
# ``aside_question-24844801cae43a91`` — seen in real cache). Length range
# 10–50 keeps stray matches out.
_AGENT_ID_TEXT_RE = re.compile(r"agentId:\s*([A-Za-z0-9_\-]{10,50})")


# M2 fix (real-data-verified): raw JSONL ``type`` values seen in ~/.claude
# beyond the message-carrying five. ``attachment`` alone accounted for 99.8%
# of the previous ``other`` bucket (74253/74404 obs on live cache); adding it
# + ``progress`` here collapses ``other`` to <0.1%. Additional known types
# from research §5 (``queue-operation``, ``last-prompt``, ``hook``,
# ``file-history-snapshot``, ``rate_limit_event``, ``permission-mode``) are
# pre-listed so future emissions land in their own bucket instead of
# ``other``. No CHECK constraint on ``observations.type`` — expanding the
# enum is purely additive; no migration needed.
_KNOWN_TYPES = frozenset(
    {
        "user",
        "assistant",
        "tool_use",
        "tool_result",
        "system",
        # v2 M2 additions — preserve raw provenance so type: filters find them
        "attachment",             # UserPromptSubmit hook results, file uploads (99.8% of legacy `other`)
        "progress",               # in-flight tool progress markers
        "queue-operation",        # cross-session prompt queue events
        "last-prompt",            # tail-of-history bookmarks
        "hook",                   # generic hook-fired events (defensive)
        "file-history-snapshot",  # session-snapshot markers (research §5)
        "rate_limit_event",       # api rate-limit signals
        "permission-mode",        # permission-mode changes
    }
)


# How many line identifiers a per-file drop report names before it says
# ", ...". Deliberately the same five github's ``_note_dropped`` uses, so the
# two read as one convention rather than two arbitrary limits.
_MAX_NAMED_DROPS = 5


def _note_dropped_lines(
    path: Path, what: str, dropped: list[str], errors: list[str]
) -> None:
    """Report the lines one file lost, as ONE entry with an exact count.

    The cap is on the identifiers, never on the count. A 359k-line corrupt file
    must not put 359k strings in the run report, the notification payload and
    memory — but "some lines were bad" is not a fault report, so the exact
    total leads and the examples follow. Capping that hid the magnitude would
    trade one failure mode for another.

    Nothing is emitted for a clean file: an entry per healthy file would make
    the errors list — which is what decides the run's exit code — meaningless.
    """
    if not dropped:
        return
    errors.append(
        f"{path}: DROPPED {len(dropped)} {what} — they are NOT in the index "
        f"(line {', '.join(dropped[:_MAX_NAMED_DROPS])}"
        f"{', ...' if len(dropped) > _MAX_NAMED_DROPS else ''})"
    )


# The two ways a timestamp costs data, spelled out where the reports are made
# rather than inline, because they are different diagnoses. A line the parser
# cannot date is dropped from the session it belongs to; a FILE in which not one
# line can be dated loses its session row and every observation under it,
# because ``first_ts``/``last_ts`` have nothing to be derived from.
#
# This is the shape a vendor format change takes. Every other line-level fault
# here is a corrupt or wrongly-shaped line, i.e. damage to one line; a
# timestamp-spelling change is uniform, hits every line at once, and would have
# emptied ~5,700 sessions and ~359k observations at exit 0. Measured against the
# real ~/.claude/projects tree at the time of writing: 0 of 186,285 parsed lines
# lack a parseable timestamp, so neither of these reports fires on healthy input
# and neither can crowd another source's line out of the notification budget.
_UNDATED_LINES = "line(s) whose timestamp is missing or unparseable"
_UNDATED_FILE = (
    "line(s) whose timestamp is missing or unparseable — EVERY line in the file, "
    "so its session row and every observation under it are dropped whole, which "
    "is the shape a vendor timestamp-format change takes"
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
    """One JSONL line, parsed enough to bucket into sessions.

    ``lineno`` is carried purely so a line dropped LATER — after ``_iter_parsed``
    has handed it over — can still be named in a fault report the same way the
    corrupt-line and wrong-shape reports name theirs. Without it a drop report
    could only cite uuids, which do not tell an operator where in the file to
    look.
    """

    lineno: int
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

    def rebuild_input(self) -> str:
        """``sources.base.SupportsRebuild`` — why ``--rebuild`` is allowed here.

        ``~/.claude/projects`` is written by Claude Code on this machine and
        the scan reads it whole. The DELETE is additionally SCOPED to the
        ``claude-code`` origin (``cli.SESSIONS_REBUILD_ORIGINS``), so the
        vendor-export sessions sharing these tables are out of its reach.
        """
        return (
            "~/.claude/projects, written by Claude Code on this machine; the "
            "rebuild is scoped to the claude-code origin it can regenerate"
        )

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
            "spawned_by_tool_use_id": "str | None (Agent tool_use_id from parent JSONL, exact-matched via agentId)",
            "cwd": "str | None",
            "git_branch": "str | None",
            "first_ts": "datetime (min message ts)",
            "last_ts": "datetime (max message ts)",
            "obs_type": (
                "'user'|'assistant'|'tool_use'|'tool_result'|'system'|"
                "'attachment'|'progress'|'queue-operation'|'last-prompt'|"
                "'hook'|'file-history-snapshot'|'rate_limit_event'|"
                "'permission-mode'|'other'"
            ),
        }

    # -- filesystem walk --------------------------------------------------

    def _iter_jsonl_files(self, errors: list[str] | None = None) -> Iterator[Path]:
        """Yield every non-live JSONL under the projects root.

        Live-file skip (5-min window) matches v1 semantics — an actively
        appended file could split observations mid-record. File mtime is a
        container signal only (research §5); we still consult it here to
        avoid reading a partial line.

        A ``stat`` that FAILS is a different thing from a file that is merely
        live, and conflating them is what this ``errors`` parameter exists to
        stop. The mtime read answered any OSError with a bare ``continue``: the
        whole file — every session and every observation in it — vanished from
        the walk with no error and no log. This is the largest source in the
        index, so a permission change on one project directory, or a dangling
        symlink left behind by a moved project, silently shrank it on a run that
        reported success and exited 0. Recorded and skipped, never just skipped:
        per-file faults are not fatal here by design, but they are not free
        either.
        """
        if not self.projects_root.exists():
            return
        now = time.time()
        for path in self.projects_root.rglob("*.jsonl"):
            try:
                mtime = path.stat().st_mtime
            except OSError as e:
                if errors is not None:
                    errors.append(f"{path}: stat failed, file skipped entirely: {e}")
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
    def _parse_line(obj: dict, lineno: int) -> _ParsedLine | None:
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
        # M2 fix: preserve the raw JSONL ``type`` verbatim when we recognise
        # it. Previously any type outside the message-carrying five collapsed
        # to ``other`` — hiding 74k+ attachment/progress/etc. rows behind a
        # single opaque bucket. ``_KNOWN_TYPES`` is now an inclusive
        # allowlist; unknown future types still fall through to ``other`` so
        # a rename downstream doesn't blow up.
        raw_type = obj.get("type") or "other"
        if not isinstance(raw_type, str):
            raw_type = "other"
        line_type = raw_type if raw_type in _KNOWN_TYPES else "other"
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
            lineno=lineno,
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
        # AGGREGATED PER FILE, not per line. A dropped line used to append one
        # error EACH, and this is the largest source in the index (~359k
        # observations): one corrupt file balloons the run report, the desktop
        # notification payload and the memory the errors list occupies, and the
        # CLI only prints ``errors[:5]`` anyway, so a single bad file could
        # crowd every other fault in the run out of the only view an operator
        # gets. Same shape as github's dropped rows — a capped list of
        # identifiers plus an EXACT total, because capping must not become the
        # fault hiding: the count is the number that says how bad it is.
        corrupt: list[int] = []
        wrong_shape: list[str] = []
        try:
            for lineno, raw in enumerate(fh, 1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    corrupt.append(lineno)
                    continue
                if not isinstance(obj, dict):
                    # Valid JSON, wrong shape — a bare string, a list, a
                    # number. It used to be dropped with no error and no log at
                    # all, which is the quietest failure in the whole source: an
                    # observation simply ceases to exist and the session it
                    # belonged to looks like it just had fewer lines. Exactly
                    # what a truncated-then-reappended file leaves behind.
                    wrong_shape.append(f"{lineno} ({type(obj).__name__})")
                    continue
                parsed = self._parse_line(obj, lineno)
                if parsed is not None:
                    yield parsed
        finally:
            fh.close()
            # In the ``finally`` so an abandoned generator still reports what
            # it dropped before the consumer walked away.
            _note_dropped_lines(path, "corrupt line(s)", [str(n) for n in corrupt], errors)
            _note_dropped_lines(
                path,
                "line(s) that are valid JSON but not a JSON object",
                wrong_shape,
                errors,
            )

    @staticmethod
    def _dominant_session_id(parsed_lines: list[_ParsedLine], path: Path) -> str | None:
        """For a top-level file, pick the sessionId that "owns" the file.

        Resume produces a prefix copy under the parent's sessionId, then the
        real (new) sessionId. The filename is the NEW sessionId. Match by
        filename stem when possible; fall back to the sessionId with the most
        lines.

        Caveat — resume-of-resume orphans spawns (round-1 MEDIUM): the caller
        drops every non-dominant line (see ``_iter_file_entities``). If a
        resume-of-resume happens, the prefix-copied lines from the middle
        session's ``Agent`` tool_use are dropped along with the rest, so any
        subagent spawned from that middle session ends up with no discoverable
        parent tool_use_id in the spawn index — the parent file is on disk,
        walked, but the specific spawn line was filtered out here. This is a
        second orphan-spawn root cause distinct from "parent JSONL not
        ingested". Optional future improvement: index Agent tool_use lines
        across ALL sessionIds present in a file, not just the dominant one.
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

    # -- Pass 1: agentId → Agent-tool_use_id index (B1 fix) --------------

    def _collect_agent_spawn_index(
        self, errors: list[str], paths: list[Path] | None = None
    ) -> dict[str, dict[str, str]]:
        """Return ``{parent_session_id: {child_agent_id: tool_use_id}}``.

        ``paths`` is the file list pass 2 is going to walk. Passing it in is
        what lets the two passes agree: they used to run ``_iter_jsonl_files``
        independently, so a file crossing the 5-minute live boundary between
        them was seen by one pass and not the other, and every file was stat'd
        twice. It also decides where faults get reported — see the ``errors``
        note below. Left None (``scripts/backfill_spawn_ids.py`` calls this
        standalone) it walks for itself.

        ERRORS: this pass reports nothing about the files it reads,
        deliberately. Every file it opens is also opened and read in full by
        pass 2, which reports each unopenable file and each bad line exactly
        once; appending here too put the same fault in the sink twice, and the
        CLI only ever prints ``errors[:5]``, so one file with a few bad lines
        could crowd every other fault in the run out of the only view an
        operator gets. The sink is still taken, because the standalone call
        above has to report the stat failures nobody else will see.

        Walks every top-level JSONL and joins each parent-side
        ``Agent``-tool ``tool_result`` back to the child ``agentId`` it
        launched. Two data channels are consulted (real-cache verified):

        1. **Structured** (preferred): the top-level ``toolUseResult`` object
           on the tool-result JSONL line carries ``agentId`` as a sibling of
           ``prompt``/``agentType``/``content``. The corresponding
           ``message.content[].tool_result.tool_use_id`` in the same line
           gives us the parent's Agent tool_use_id.
        2. **Text regex** (fallback): the ``tool_result.content`` text block
           echoes ``"agentId: <id> (internal ID …)"``. Same value, LLM-facing
           form. Regex used when the structured field is missing.

        This is an EXACT-MATCH join — no time windows, no ambiguity.
        """
        index: dict[str, dict[str, str]] = {}
        if paths is None:
            paths = list(self._iter_jsonl_files(errors))
        for path in paths:
            if self._is_subagent_path(path):
                continue
            dominant_id = path.stem  # filename == top-level sessionId
            try:
                fh = path.open(encoding="utf-8", errors="replace")
            except OSError:
                continue  # pass 2 reports it; see the docstring
            try:
                # No line numbers: this pass reports nothing, so it has
                # nothing to number. Pass 2 carries the line-level diagnostics.
                for raw in fh:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # pass 2 reports it; see the docstring
                    if not isinstance(obj, dict):
                        continue  # pass 2 reports it; see the docstring
                    if obj.get("sessionId") != dominant_id:
                        # Resume prefix-copy: skip lines that don't belong
                        # to this file's dominant session.
                        continue
                    self._index_agent_launch(obj, dominant_id, index)
            finally:
                fh.close()
        return index

    @staticmethod
    def _index_agent_launch(
        obj: dict[str, Any],
        parent_sid: str,
        index: dict[str, dict[str, str]],
    ) -> None:
        """Extract (agent_id, tool_use_id) from an Agent-tool_result JSONL
        line and record it under ``index[parent_sid]``. No-op when the line
        isn't a tool_result carrying an agentId."""
        message = obj.get("message") or {}
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            return
        # Find the tool_result block's tool_use_id — the parent's Agent id.
        tool_use_id: str | None = None
        text_body = ""
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_result":
                continue
            tid = block.get("tool_use_id")
            if isinstance(tid, str) and tool_use_id is None:
                tool_use_id = tid
            bc = block.get("content")
            if isinstance(bc, str):
                text_body += bc
            elif isinstance(bc, list):
                for sub in bc:
                    if isinstance(sub, dict) and sub.get("type") == "text":
                        t = sub.get("text")
                        if isinstance(t, str):
                            text_body += t
        if tool_use_id is None:
            return
        # Strategy 1 (preferred): structured toolUseResult.agentId
        tur = obj.get("toolUseResult")
        if isinstance(tur, dict):
            aid = tur.get("agentId")
            if isinstance(aid, str) and aid:
                index.setdefault(parent_sid, {})[aid] = tool_use_id
                return
        # Strategy 2 (fallback): text regex on the tool_result body
        m = _AGENT_ID_TEXT_RE.search(text_body)
        if m:
            index.setdefault(parent_sid, {}).setdefault(m.group(1), tool_use_id)

    @staticmethod
    def _recover_spawn_tool_use_id(
        spawn_index: dict[str, dict[str, str]],
        parent_session_id: str | None,
        agent_id: str | None,
    ) -> str | None:
        """Exact-match lookup: parent's Agent tool_use_id for ``agent_id``.

        Returns ``None`` when:
        * parent session unknown (legacy / orphan layout),
        * parent JSONL not on disk / not walked (no index entry),
        * the parent's index has no record of spawning this agent_id.
          Several distinct causes:
            - live-window skip when ingest ran while the parent was open;
            - the launching line's sessionId was filtered out as
              non-dominant (resume-of-resume — see caveat on
              ``_dominant_session_id``);
            - the launch used a mechanism we don't yet parse.
        """
        if not parent_session_id or not agent_id:
            return None
        by_parent = spawn_index.get(parent_session_id)
        if not by_parent:
            return None
        return by_parent.get(agent_id)

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
        # ONE walk, shared by both passes. Walking twice stat'd every file
        # twice and let the passes disagree about which files exist: a file
        # crossing the 5-minute live boundary between them was indexed by one
        # and not the other. It also decided nothing about where a stat failure
        # gets reported, so threading the sink into both would have put every
        # such failure in the run's errors twice.
        paths = list(self._iter_jsonl_files(sink))
        spawn_index = self._collect_agent_spawn_index(sink, paths)

        for path in paths:
            yield from self._iter_file_entities(path, spawn_index, since, sink)

    def _iter_file_entities(
        self,
        path: Path,
        spawn_index: dict[str, dict[str, str]],
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
            for entity in self._emit_subagent(path, parsed, spawn_index, since, errors):
                yield entity
        else:
            dominant = self._dominant_session_id(parsed, path)
            if dominant is None:
                return
            # Only ingest lines matching the dominant sessionId — drops the
            # resume prefix-copy portion (which belongs to the parent file).
            own_lines = [p for p in parsed if p.session_id == dominant]
            for entity in self._emit_top_level(path, dominant, own_lines, since, errors):
                yield entity

    @staticmethod
    def _dated(
        path: Path, lines: list[_ParsedLine], errors: list[str]
    ) -> list[datetime] | None:
        """The timestamps ``lines`` carry, or None when NOT ONE of them has any.

        Both answers are reported, and neither used to be. A line with no
        parseable timestamp was dropped by a bare ``if p.ts is None: continue``,
        and a file in which every line is like that returned before emitting
        anything at all — no errors entry, no log line, exit 0.

        Sessions is the largest source in the index. The earlier dropped-line
        reporting covers corrupt JSON and wrong-shaped JSON only, which are
        per-line accidents; a timestamp-spelling change by the vendor is uniform
        and would have emptied ~5,700 sessions and ~359k observations while the
        30-minute timer went on reporting success.

        Aggregated per file — capped example line numbers, EXACT count — for the
        same reason the corrupt-line report is: a format drift hits every line
        at once, and one error per line would bury every other fault in the run
        in the CLI's ``errors[:5]`` view and in the notification payload.
        """
        undated = [str(p.lineno) for p in lines if p.ts is None]
        dated = [p.ts for p in lines if p.ts is not None]
        if not dated:
            _note_dropped_lines(path, _UNDATED_FILE, undated, errors)
            return None
        _note_dropped_lines(path, _UNDATED_LINES, undated, errors)
        return dated

    def _emit_top_level(
        self,
        path: Path,
        session_id: str,
        lines: list[_ParsedLine],
        since: datetime | None,
        errors: list[str],
    ) -> Iterator[SessionEntity]:
        if not lines:
            return
        tss = self._dated(path, lines, errors)
        if tss is None:
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
        spawn_index: dict[str, dict[str, str]],
        since: datetime | None,
        errors: list[str],
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

        tss = self._dated(path, own_lines, errors)
        if tss is None:
            return
        first_ts, last_ts = min(tss), max(tss)
        since_utc = _normalise_utc(since) if since else None
        if since_utc and last_ts < since_utc:
            return
        agent_type = next((p.agent_type for p in own_lines if p.agent_type), None)
        spawned = self._recover_spawn_tool_use_id(spawn_index, parent_sid, agent_id)
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
