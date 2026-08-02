# Chat-export sources: chatgpt + claude-web (2026-08-02)

Research: `~/Repos/research-agent/reports/f2ac113e37d147508e55d25f93d9150f.md`
(2026 export formats, parser gotchas, prior-art repos).

## Goal

Ingest ChatGPT and Claude.ai data-export ZIPs into the existing
sessions/observations ontology (Schema B). Export-drop pattern: user
drops the vendor ZIP (or extracted conversations.json) into
`~/.local/share/aggregator/drops/`, runs `aggregator ingest chatgpt` /
`aggregator ingest claude-web` (or the timer picks it up later).

## Schema change (v2 → v3)

- `sessions.origin TEXT NOT NULL DEFAULT 'claude-code'` — values:
  `claude-code` | `chatgpt` | `claude-web`.
- Migration: `ALTER TABLE sessions ADD COLUMN origin ...` guarded by
  user_version check. NO rebuild needed (default backfills existing).
- `SessionRow` gains `origin: str = "claude-code"` field.
- DSL: `source:chatgpt` / `source:claude-web` route to sessions
  ontology filtered on origin (extend `_sessions_where` / `_obs_where`
  source handling; obs filter via session subselect on origin).
- `source:sessions`/`source:subagents` keep meaning kind split within
  origin='claude-code' (backward compat) — chatgpt/claude-web rows are
  all kind='session' (no subagents in either export).
- capabilities: counts + freshness per origin.

## Mapping

### ChatGPT (conversations.json — array; 2026 exports may shard
`conversations-*.json`; accept both + zip)

- conversation → SessionRow: session_id=`conversation_id` (fall back
  `id`), origin='chatgpt', kind='session', first_ts/last_ts from
  create_time/update_time (epoch float, UTC), cwd=None,
  git_branch=None, jsonl_path=<source file path>.
- mapping node (message != null) → ObservationRow:
  obs_id=node id, parent_obs_id=node parent (skip synthetic null-message
  root; re-parent its children to None), type=author.role preserved raw
  (user/assistant/system/tool), ts=create_time (nullable → session
  first_ts), body = flattened content.parts (strings kept; dict parts →
  `.text` or `[non-text part]`), model from metadata.model_slug.
- Branches kept: ALL nodes emitted (siblings from regenerations
  included) — parent pointers preserve the DAG. No current_node
  filtering (completeness over canonicality; consistent with sessions
  source which keeps everything).
- Gotchas handled: null message on root, empty parts filter,
  polymorphic parts, unknown content_type fallback (raw type preserved
  in obs body prefix if not text).

### Claude.ai (conversations.json — array)

- conversation → SessionRow: session_id=`uuid`, origin='claude-web',
  kind='session', first_ts/last_ts from created_at/updated_at
  (ISO-8601 Z).
- chat_message → ObservationRow: obs_id=`uuid`,
  parent_obs_id=`parent_message_uuid` (nil-UUID sentinel
  `00000000-0000-4000-8000-000000000000` → None), type=sender
  (human→'user', assistant→'assistant'), ts=created_at, body from
  content[] blocks (text + thinking; `text` top-level field as
  fallback when no text blocks).
- tool_use / tool_result blocks → SEPARATE ObservationRows:
  obs_id=`{msg_uuid}:b{index}`, parent_obs_id=msg_uuid,
  type='tool_use'/'tool_result', tool_name from block name,
  tool_use_id from block id / tool_use_id. Aligns chat exports with
  the Claude Code ontology so `type:tool_use` queries work.
- Branching: real (regenerations = siblings sharing
  parent_message_uuid). Keep all.

## Collision safety

Session IDs are vendor UUIDs; same namespace as Claude Code session
UUIDs. Prefix stable IDs: `chatgpt:<id>` / `claude-web:<uuid>` as the
stored session_id. (Claude Code sessions stay unprefixed — existing
data + spawn logic untouched.)

## Files

- `aggregator/sources/chatgpt.py` — new
- `aggregator/sources/claude_web.py` — new
- `aggregator/core/store.py` — v3 migration, origin filters
- `aggregator/core/dsl.py` — no change (source:X already generic)
- `aggregator/mcp.py` — routing: chatgpt/claude-web are session-shaped
  sources; capabilities counts
- `aggregator/cli.py` — ingest subcommand dispatch
- `tests/sources/test_chatgpt.py`, `tests/sources/test_claude_web.py`,
  fixtures under `tests/fixtures/{chatgpt,claude-web}/`
- `tests/core/test_store.py` — origin migration + filter tests

## Non-goals

- No automation of the vendor export request (ToS risk).
- No artifact/attachment binary handling (exports don't ship them).
- No canonical-path filtering (keep whole DAG).
- projects.json / users.json ignored v1.

## Test fixtures

Hand-written minimal JSON per research shapes: chatgpt 1 conversation
(root null-message node + user + assistant + regenerated sibling +
tool node + multimodal part), claude-web 1 conversation (human +
assistant with text/thinking/tool_use/tool_result blocks + regenerated
sibling + nil-UUID root sentinel). Zip fixture for the zip path.
