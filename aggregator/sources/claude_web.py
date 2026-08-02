"""Claude.ai (claude-web) export source: ``conversations.json`` → Schema B.

Parses Claude.ai data-export drops from ``~/.local/share/aggregator/drops/``
(override via ``AGGREGATOR_DROPS_DIR``). Two accepted shapes per drop:

* ``*.zip`` — the vendor export zip; ``conversations.json`` is read straight
  from the archive via :mod:`zipfile` (no extraction). The zip's other members
  (``users.json``, ``projects.json``) are ignored (plan §Non-goals).
* bare ``conversations.json`` — a pre-extracted array file.

Design decisions (mirroring the plan's Claude.ai mapping section):

* **Shape sniff, not filename ownership.** A parallel chatgpt source scans the
  SAME drops dir and both vendors name their file ``conversations.json``. The
  first array element disambiguates: Claude conversations carry
  ``chat_messages``; ChatGPT ones carry ``mapping``. Files that don't match
  OUR shape are skipped silently — the other source owns them. Unreadable
  JSON, however, goes to the errors sink (corruption should not vanish).
* **Stable-ID prefixing** (plan §Collision safety): vendor UUIDs share a
  namespace with Claude Code session UUIDs, so every stored id is prefixed —
  ``claude-web:<uuid>`` for sessions AND observations (parent pointers
  prefixed consistently so the chain resolves in-store).
* **Nil-UUID parent sentinel**: Claude marks conversation roots with a
  ``parent_message_uuid`` starting ``00000000-0000-4000-8000-`` (tail
  varies) → ``parent_obs_id=None``. Prefix match, not full-string equality.
* **Branching is real.** Regenerations appear as multiple messages sharing
  one ``parent_message_uuid`` and ``chat_messages`` is a flat list, NOT a
  linear transcript. All siblings are kept (completeness over canonicality —
  consistent with the sessions source keeping the whole DAG).
* **Tool blocks become separate observations** so ``type:tool_use`` DSL
  queries span origins: content block ``i`` of type tool_use/tool_result →
  ``obs_id = claude-web:<msg-uuid>:b<i>`` parented on the message obs. A
  message that is PURE tool_use (empty text) still emits its (empty-body)
  message obs as the anchor; messages with neither body nor tool blocks are
  skipped.
* **Per-conversation atomicity**: each conversation is materialised into a
  full entity list before anything is yielded, so a malformed conversation
  lands in the errors sink without leaking a partial session.
"""
from __future__ import annotations

import json
import os
import zipfile
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aggregator.sources.base import (
    IngestResult,
    ObservationRow,
    SessionEntity,
    SessionRow,
)

DEFAULT_DROPS_DIR = "~/.local/share/aggregator/drops"

_ID_PREFIX = "claude-web:"
# Root-marker sentinel. Claude exports use a nil-style v4 UUID whose tail
# varies across exports — match on the stable prefix.
_NIL_PARENT_PREFIX = "00000000-0000-4000-8000-"

_SENDER_MAP = {"human": "user", "assistant": "assistant"}


def _parse_iso(s: Any) -> datetime | None:
    """Lenient ISO-8601 parse: ``Z`` suffix, milli or microsecond precision
    (``datetime.fromisoformat`` handles both on py3.11+ once ``Z`` is
    normalised). Naive values are pinned to UTC."""
    if not isinstance(s, str) or not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _flatten_tool_result_content(content: Any) -> str:
    """Flatten a tool_result block's ``content`` list into body text."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for c in content:
        if isinstance(c, dict):
            t = c.get("text", "")
            parts.append(t if isinstance(t, str) else "")
    return "\n".join(parts)


class ClaudeWebSource:
    """Claude.ai export source. Yields ``SessionRow`` + ``ObservationRow``.

    ``iter_entities`` signature matches ``SessionsSource`` so the CLI's
    entity-shaped ingest dispatch handles both identically.
    """

    name = "claude-web"

    def __init__(self, drops_dir: str | None = None):
        root = (
            drops_dir
            or os.environ.get("AGGREGATOR_DROPS_DIR")
            or DEFAULT_DROPS_DIR
        )
        self.drops_dir = Path(os.path.expanduser(root))

    def record_shape(self) -> dict[str, str]:
        return {
            "session_id": "str ('claude-web:<conversation-uuid>')",
            "root_session_id": "str (= session_id; exports have no subagents)",
            "parent_session_id": "None",
            "kind": "'session'",
            "origin": "'claude-web'",
            "first_ts": "datetime (conversation created_at)",
            "last_ts": "datetime (conversation updated_at)",
            "obs_type": (
                "'user' | 'assistant' | 'tool_use' | 'tool_result' | "
                "<raw sender> (unknown senders preserved verbatim)"
            ),
        }

    # -- drop discovery ----------------------------------------------------

    def _iter_conversation_files(
        self, errors: list[str]
    ) -> Iterator[tuple[str, list[Any]]]:
        """Yield ``(source_path, conversations)`` for every Claude-shaped
        drop. ``source_path`` is the bare file path, or ``<zip>!<member>``
        for zip members."""
        root = self.drops_dir
        if not root.exists():
            return
        for zpath in sorted(root.rglob("*.zip")):
            matches: list[tuple[str, list[Any]]] = []
            try:
                with zipfile.ZipFile(zpath) as zf:
                    for member in zf.namelist():
                        if member.rsplit("/", 1)[-1] != "conversations.json":
                            continue  # users.json / projects.json / etc.
                        src_path = f"{zpath}!{member}"
                        try:
                            with zf.open(member) as fh:
                                data = json.load(fh)
                        except json.JSONDecodeError as e:
                            errors.append(f"{src_path}: invalid JSON: {e}")
                            continue
                        if self._is_claude_shape(data):
                            matches.append((src_path, data))
            except (zipfile.BadZipFile, OSError) as e:
                errors.append(f"{zpath}: unreadable zip: {e}")
                continue
            yield from matches
        for jpath in sorted(root.rglob("conversations.json")):
            try:
                with jpath.open(encoding="utf-8") as fh:
                    data = json.load(fh)
            except json.JSONDecodeError as e:
                errors.append(f"{jpath}: invalid JSON: {e}")
                continue
            except OSError as e:
                errors.append(f"{jpath}: open failed: {e}")
                continue
            if self._is_claude_shape(data):
                yield str(jpath), data

    @staticmethod
    def _is_claude_shape(data: Any) -> bool:
        """Sniff the first array element: ours carry ``chat_messages``;
        ChatGPT's carry ``mapping``. Anything else is not ours either."""
        return (
            isinstance(data, list)
            and len(data) > 0
            and isinstance(data[0], dict)
            and "chat_messages" in data[0]
        )

    # -- entity emission ---------------------------------------------------

    def iter_entities(
        self,
        since: datetime | None = None,
        errors: list[str] | None = None,
    ) -> Iterator[SessionEntity]:
        """Main ingest entrypoint. Yields SessionRow before its ObservationRows.

        ``since`` filters conversations by ``updated_at`` (a conversation
        updated past the cutoff is emitted in full).
        """
        sink = errors if errors is not None else []
        for src_path, conversations in self._iter_conversation_files(sink):
            for index, conv in enumerate(conversations):
                try:
                    entities = self._conversation_entities(conv, src_path, since)
                except (KeyError, TypeError, ValueError, AttributeError) as e:
                    cid = conv.get("uuid") if isinstance(conv, dict) else None
                    sink.append(f"{src_path}[{index}] uuid={cid!r}: {e}")
                    continue
                yield from entities

    def _conversation_entities(
        self,
        conv: Any,
        src_path: str,
        since: datetime | None,
    ) -> list[SessionEntity]:
        """Materialise one conversation. Raises ``ValueError`` on malformed
        input (caught by ``iter_entities`` → errors sink)."""
        if not isinstance(conv, dict):
            raise ValueError("conversation is not an object")
        cuuid = conv.get("uuid")
        if not isinstance(cuuid, str) or not cuuid:
            raise ValueError("conversation missing uuid")
        messages = conv.get("chat_messages")
        if not isinstance(messages, list):
            raise ValueError("chat_messages is not a list")

        msg_ts = [
            t
            for t in (
                _parse_iso(m.get("created_at"))
                for m in messages
                if isinstance(m, dict)
            )
            if t is not None
        ]
        first_ts = _parse_iso(conv.get("created_at")) or (min(msg_ts) if msg_ts else None)
        last_ts = _parse_iso(conv.get("updated_at")) or (max(msg_ts) if msg_ts else None)
        last_ts = last_ts or first_ts
        if first_ts is None or last_ts is None:
            raise ValueError("no parseable timestamps")

        if since is not None:
            since_utc = since if since.tzinfo is not None else since.replace(tzinfo=UTC)
            if last_ts < since_utc:
                return []

        session_id = _ID_PREFIX + cuuid
        entities: list[SessionEntity] = [
            SessionRow(
                session_id=session_id,
                root_session_id=session_id,
                parent_session_id=None,
                kind="session",
                agent_id=None,
                agent_type=None,
                spawned_by_tool_use_id=None,
                cwd=None,
                git_branch=None,
                first_ts=first_ts,
                last_ts=last_ts,
                jsonl_path=src_path,
                origin="claude-web",
            )
        ]
        for m in messages:
            if not isinstance(m, dict):
                continue  # lenient — a stray non-object entry isn't fatal
            entities.extend(self._message_entities(m, session_id, first_ts))
        return entities

    def _message_entities(
        self,
        m: dict[str, Any],
        session_id: str,
        session_first_ts: datetime,
    ) -> list[SessionEntity]:
        muuid = m.get("uuid")
        if not isinstance(muuid, str) or not muuid:
            return []  # no primary key — cannot store
        obs_id = _ID_PREFIX + muuid

        parent_raw = m.get("parent_message_uuid")
        parent_obs_id: str | None = None
        if (
            isinstance(parent_raw, str)
            and parent_raw
            and not parent_raw.startswith(_NIL_PARENT_PREFIX)
        ):
            parent_obs_id = _ID_PREFIX + parent_raw

        sender = m.get("sender")
        if isinstance(sender, str) and sender:
            obs_type = _SENDER_MAP.get(sender, sender)  # unknown → raw
        else:
            obs_type = "other"

        ts = _parse_iso(m.get("created_at")) or session_first_ts

        body_parts: list[str] = []
        tool_rows: list[ObservationRow] = []
        content = m.get("content")
        if isinstance(content, list):
            for i, block in enumerate(content):
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    t = block.get("text")
                    if isinstance(t, str) and t:
                        body_parts.append(t)
                elif btype == "thinking":
                    t = block.get("thinking")
                    if not (isinstance(t, str) and t):
                        t = block.get("text")
                    if isinstance(t, str) and t:
                        body_parts.append(t)
                elif btype in ("tool_use", "tool_result"):
                    tool_rows.append(
                        self._tool_block_obs(
                            block, btype, i, muuid, obs_id, session_id, ts
                        )
                    )
                # unknown block types ignored, never fatal

        body = "\n\n".join(body_parts)
        if not body:
            t = m.get("text")  # top-level fallback when no text/thinking blocks
            if isinstance(t, str):
                body = t

        entities: list[SessionEntity] = []
        # Skip empty-body message obs ONLY when there are no tool blocks —
        # a pure-tool_use message must keep its anchor obs so the block obs
        # and any child message's parent pointer resolve.
        if body or tool_rows:
            entities.append(
                ObservationRow(
                    obs_id=obs_id,
                    session_id=session_id,
                    root_session_id=session_id,
                    parent_obs_id=parent_obs_id,
                    type=obs_type,
                    ts=ts,
                    model=None,
                    input_tokens=None,
                    output_tokens=None,
                    tool_name=None,
                    tool_use_id=None,
                    body=body,
                )
            )
        entities.extend(tool_rows)
        return entities

    @staticmethod
    def _tool_block_obs(
        block: dict[str, Any],
        btype: str,
        index: int,
        msg_uuid: str,
        parent_obs_id: str,
        session_id: str,
        msg_ts: datetime,
    ) -> ObservationRow:
        ts = _parse_iso(block.get("start_timestamp")) or msg_ts
        tool_name: str | None = None
        tool_use_id: str | None = None
        if btype == "tool_use":
            n = block.get("name")
            if isinstance(n, str):
                tool_name = n
            tid = block.get("id")
            if isinstance(tid, str):
                tool_use_id = tid
            tool_input = block.get("input")
            body = (
                ""
                if tool_input is None
                else json.dumps(tool_input, ensure_ascii=False, default=str)
            )
        else:  # tool_result
            tid = block.get("tool_use_id")
            if isinstance(tid, str):
                tool_use_id = tid
            body = _flatten_tool_result_content(block.get("content"))
        return ObservationRow(
            obs_id=f"{_ID_PREFIX}{msg_uuid}:b{index}",
            session_id=session_id,
            root_session_id=session_id,
            parent_obs_id=parent_obs_id,
            type=btype,
            ts=ts,
            model=None,
            input_tokens=None,
            output_tokens=None,
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            body=body,
        )

    # -- Protocol methods --------------------------------------------------

    def ingest(self, since: datetime | None) -> IngestResult:
        """Count-only path retained for protocol compat (persistence is the
        CLI's job — mirrors ``SessionsSource.ingest``)."""
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
