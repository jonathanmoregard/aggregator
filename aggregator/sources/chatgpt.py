"""ChatGPT export source (Chunk 2, chat-exports plan 2026-08-02).

Ingests ChatGPT data-export drops from ``~/.local/share/aggregator/drops/``
(override via ``AGGREGATOR_DROPS_DIR`` or the constructor) AND
``~/Downloads`` (override ``AGGREGATOR_DOWNLOADS_DIR``) into the
sessions/observations ontology:

* accepted inputs: vendor ``*.zip`` containing ``conversations.json`` or
  sharded ``conversations-*.json``, OR those JSON files dropped bare in
  either dir. Discovery + vendor classification live in
  :mod:`aggregator.sources.exportdrops` (content sniff: ``mapping`` in the
  first array element marks a ChatGPT export; claude-web files are the
  parallel source's). Zips are read via :mod:`zipfile` without extraction;
  ``jsonl_path`` for zip members is ``<zip path>!<member name>``.
* conversation → ``SessionRow``: session_id = ``chatgpt:<conversation_id or
  id>`` (prefix for collision safety against Claude Code session UUIDs —
  plan §Collision safety), origin='chatgpt', kind='session'. first_ts /
  last_ts from ``create_time``/``update_time`` (unix epoch float seconds,
  UTC); missing update_time falls back to create_time; missing both falls
  back to min/max node message ts (defensive — real exports always carry
  create_time).
* mapping node (``message != null``) → ``ObservationRow``: obs_id =
  ``chatgpt:<node id>``; parent_obs_id = ``chatgpt:<parent>`` unless the
  parent node is absent or not emitted (synthetic null-message root,
  empty-body stub) → None. ``type`` = ``author.role`` preserved RAW
  (user/assistant/system/tool) — repo lesson from sessions.py M2: never
  bucket raw provenance into 'other' when the vendor gave us a real value.
* ALL branches kept (regenerated siblings share a parent) — no
  ``current_node`` filtering (completeness over canonicality, consistent
  with the sessions source).
* body = flattened ``content.parts``: string parts kept; dict parts →
  ``part["text"]`` or ``[non-text part: <content_type>]``. Content types
  drift constantly (12+ known: text, multimodal_text, code,
  execution_output, thoughts, reasoning_recap, user_editable_context,
  tether_quote, system_error, ...) — unknown types must never crash: when
  ``parts`` is missing/non-list we fall back to ``str(content["text"])``
  when present, else empty. Empty-body non-tool nodes are skipped
  (regeneration stubs); empty tool nodes are KEPT (tool calls may carry
  structure elsewhere).
* robustness: orphaned mapping nodes fine (parent lookup is set-membership);
  conversation without ``mapping`` emits the session row only; malformed
  conversations/files are recorded in the errors sink and skipped, never
  aborting the ingest.
"""
from __future__ import annotations

import json
import logging
import os
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aggregator.core.provenance import classify
from aggregator.sources.base import (
    IngestResult,
    ObservationRow,
    SessionEntity,
    SessionRow,
)
from aggregator.sources.exportdrops import (
    DEFAULT_DROPS_DIR,
    discover_export_files,
    downloads_dir,
)

log = logging.getLogger(__name__)


def _epoch_to_dt(value: Any) -> datetime | None:
    """Unix epoch float/int seconds → aware UTC datetime; anything else → None.

    ``bool`` is excluded explicitly (it's an ``int`` subclass — ``true`` in
    drifted JSON must not become 1970-01-01T00:00:01Z).
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        return datetime.fromtimestamp(value, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


def _flatten_body(content: Any) -> str:
    """Flatten a ChatGPT ``message.content`` payload into searchable text.

    * ``parts`` list: string parts kept verbatim (empties dropped); dict
      parts (multimodal image pointers, audio transcripts...) contribute
      their ``text`` field when present, else a ``[non-text part: <type>]``
      placeholder so FTS still records that something was there.
    * no/non-list ``parts`` (thoughts, reasoning_recap, tether_quote,
      unknown future types): fall back to ``content["text"]`` when present
      (tether_quote carries one), else empty string. Never raises.
    """
    if not isinstance(content, dict):
        return ""
    parts = content.get("parts")
    if isinstance(parts, list):
        pieces: list[str] = []
        for part in parts:
            if isinstance(part, str):
                if part:
                    pieces.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and text:
                    pieces.append(text)
                else:
                    ptype = part.get("content_type")
                    label = ptype if isinstance(ptype, str) and ptype else "unknown"
                    pieces.append(f"[non-text part: {label}]")
        return "\n".join(pieces)
    text = content.get("text")
    if text is None:
        return ""
    return text if isinstance(text, str) else str(text)


@dataclass
class _ParsedNode:
    """One mapping node with a non-null message, parsed for emission."""

    node_id: str
    parent_id: str | None
    role: str
    ts: datetime | None  # message create_time; None → session first_ts
    body: str
    model: str | None
    tool_name: str | None


class ChatGPTSource:
    """ChatGPT export drops → ``SessionRow`` + ``ObservationRow``.

    Mirrors ``SessionsSource``'s protocol surface: ``iter_entities(since,
    errors)`` yields each SessionRow before its ObservationRows; ``ingest``
    is the count-only protocol-compat path (persistence is the CLI's job).
    """

    name = "chatgpt"

    def __init__(self, drops_dir: str | None = None):
        raw = drops_dir or os.environ.get("AGGREGATOR_DROPS_DIR") or DEFAULT_DROPS_DIR
        self.drops_dir = Path(raw).expanduser()

    def manual_export_input(self) -> str:
        """``sources.base.ReadsManualExport`` — what ``--rebuild`` is refused on."""
        return (
            "a ChatGPT data-export zip, requested from Settings and emailed as "
            "a link that expires in 24h; nothing on this machine refreshes it, "
            "so a conversation whose export is gone exists only in this store"
        )

    def record_shape(self) -> dict[str, str]:
        return {
            "session_id": "str ('chatgpt:<conversation_id>')",
            "root_session_id": "str (= session_id; no subagents in exports)",
            "kind": "'session' (always)",
            "origin": "'chatgpt' (always)",
            "first_ts": "datetime (conversation create_time, UTC)",
            "last_ts": "datetime (conversation update_time, UTC)",
            "obs_type": "'user'|'assistant'|'system'|'tool' (author.role, raw)",
            "model": "str | None (message.metadata.model_slug)",
            "tool_name": "str | None (author.name when role='tool')",
        }

    # -- input discovery ---------------------------------------------------

    def _iter_conversation_payloads(
        self, errors: list[str]
    ) -> Iterator[tuple[str, bytes]]:
        """Yield ``(source_label, raw_json_bytes)`` for every ChatGPT-owned
        input across drops dir + Downloads.

        Discovery + vendor classification are the shared helper's job
        (``exportdrops.discover_export_files``); claude-web-shaped files
        never surface here. Unreadable files go to the errors sink; the
        walk continues.
        """
        files = discover_export_files(
            "chatgpt", dirs=[self.drops_dir, downloads_dir()], errors=errors
        )
        for f in files:
            try:
                yield f.label, f.read_bytes()
            except (OSError, zipfile.BadZipFile, KeyError) as e:
                errors.append(f"{f.label}: read failed: {e}")

    # -- conversation parsing ---------------------------------------------

    @staticmethod
    def _parse_mapping(mapping: dict[str, Any]) -> list[_ParsedNode]:
        """Parse every emittable node out of a conversation ``mapping``.

        Skips: non-dict nodes, null-message nodes (synthetic root), and
        empty-body non-tool nodes (regeneration stubs). Tool nodes are kept
        even with empty bodies.
        """
        nodes: list[_ParsedNode] = []
        for key, node in mapping.items():
            if not isinstance(node, dict):
                continue
            message = node.get("message")
            if not isinstance(message, dict):
                continue  # synthetic root (message: null) or malformed
            author = message.get("author")
            if not isinstance(author, dict):
                author = {}
            role = author.get("role")
            if not isinstance(role, str) or not role:
                role = "other"
            body = _flatten_body(message.get("content"))
            if not body and role != "tool":
                continue  # empty-parts regeneration stub
            tool_name: str | None = None
            if role == "tool":
                name = author.get("name")
                if isinstance(name, str) and name:
                    tool_name = name
            metadata = message.get("metadata")
            model: str | None = None
            if isinstance(metadata, dict):
                slug = metadata.get("model_slug")
                if isinstance(slug, str) and slug:
                    model = slug
            parent = node.get("parent")
            nodes.append(
                _ParsedNode(
                    node_id=key,
                    parent_id=parent if isinstance(parent, str) else None,
                    role=role,
                    ts=_epoch_to_dt(message.get("create_time")),
                    body=body,
                    model=model,
                    tool_name=tool_name,
                )
            )
        return nodes

    def _emit_conversation(
        self,
        conv: Any,
        source_label: str,
        since_utc: datetime | None,
        errors: list[str],
    ) -> Iterator[SessionEntity]:
        if not isinstance(conv, dict):
            errors.append(f"{source_label}: conversation element is not an object")
            return
        raw_id = conv.get("conversation_id") or conv.get("id")
        if not isinstance(raw_id, str) or not raw_id:
            errors.append(
                f"{source_label}: conversation missing conversation_id/id "
                f"(title={conv.get('title')!r})"
            )
            return
        session_id = f"chatgpt:{raw_id}"

        mapping = conv.get("mapping")
        if mapping is not None and not isinstance(mapping, dict):
            errors.append(f"{source_label}: {raw_id}: mapping is not an object")
            mapping = None
        nodes = self._parse_mapping(mapping) if isinstance(mapping, dict) else []

        create = _epoch_to_dt(conv.get("create_time"))
        update = _epoch_to_dt(conv.get("update_time"))
        node_tss = [n.ts for n in nodes if n.ts is not None]
        first_ts = create or update or (min(node_tss) if node_tss else None)
        last_ts = update or create or (max(node_tss) if node_tss else None)
        if first_ts is None or last_ts is None:
            errors.append(f"{source_label}: {raw_id}: no usable timestamps")
            return
        if since_utc and last_ts < since_utc:
            return

        yield SessionRow(
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
            jsonl_path=source_label,
            origin="chatgpt",
        )

        # Parent pointers only survive to a node that is actually emitted —
        # children of the skipped synthetic root (and of empty-body stubs)
        # re-parent to None instead of dangling.
        emitted_ids = {n.node_id for n in nodes}
        for n in nodes:
            parent_obs_id = (
                f"chatgpt:{n.parent_id}"
                if n.parent_id is not None and n.parent_id in emitted_ids
                else None
            )
            yield ObservationRow(
                obs_id=f"chatgpt:{n.node_id}",
                session_id=session_id,
                root_session_id=session_id,
                parent_obs_id=parent_obs_id,
                type=n.role,
                ts=n.ts if n.ts is not None else first_ts,
                model=n.model,
                input_tokens=None,
                output_tokens=None,
                tool_name=n.tool_name,
                tool_use_id=None,
                body=n.body,
                # Same ontology and same absence of structural fields as the
                # claude-web export: type and body are the whole evidence.
                provenance=classify(n.role, n.body),
            )

    # -- protocol surface --------------------------------------------------

    def iter_entities(
        self,
        since: datetime | None = None,
        errors: list[str] | None = None,
    ) -> Iterator[SessionEntity]:
        """Main ingest entrypoint. Yields each SessionRow before its
        ObservationRows. ``since`` filters conversations by update_time
        (last_ts) — mirrors the sessions source's advisory semantics.

        A CONVERSATION ID IN TWO EXPORT FILES EMITS ONCE, from the newest
        file: discovery hands files over newest-first and ``claimed`` makes
        the first sighting the only one — same union-across-files mechanism
        and same fix as claude-web and substack (the SessionRows differ on
        ``jsonl_path``, so duplicates would flip-flop the stored row every
        tick). Claimed before the ``since`` filter, so the newest file owns
        the id for the whole run even when the window already excludes its
        copy. A conversation without a usable id skips the claim and falls
        through to ``_emit_conversation``'s existing error handling.
        """
        sink = errors if errors is not None else []
        since_utc = _normalise_utc(since) if since else None
        claimed: set[str] = set()
        for source_label, raw in self._iter_conversation_payloads(sink):
            try:
                conversations = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                sink.append(f"{source_label}: invalid JSON: {e}")
                continue
            if not isinstance(conversations, list):
                sink.append(f"{source_label}: top level is not a conversation array")
                continue
            # Second guard behind the discovery helper's classification:
            # a claude-web-shaped array is the parallel source's, not ours.
            if (
                conversations
                and isinstance(conversations[0], dict)
                and "chat_messages" in conversations[0]
            ):
                continue
            for conv in conversations:
                if isinstance(conv, dict):
                    raw_id = conv.get("conversation_id") or conv.get("id")
                    if isinstance(raw_id, str) and raw_id:
                        if raw_id in claimed:
                            continue
                        claimed.add(raw_id)
                yield from self._emit_conversation(conv, source_label, since_utc, sink)

    def ingest(self, since: datetime | None) -> IngestResult:
        """Count-only path retained for protocol compat (persistence is the
        CLI's job — same split as ``SessionsSource.ingest``)."""
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
