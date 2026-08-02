"""One-shot: backfill ``sessions.spawned_by_tool_use_id`` for existing subagent
rows without full re-ingest.

Rationale: B1 fix landed the exact-match agentId → Agent-tool_use_id join. The
live cache has 1207 subagents with ``spawned_by_tool_use_id IS NULL`` from
the old (broken) Task-window recovery. Rebuilding the whole cache to re-emit
these rows takes minutes; instead this script rebuilds only the
parent-side index and UPDATEs the existing subagent rows in place.

Usage:
    uv run --directory ~/Repos/aggregator python scripts/backfill_spawn_ids.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aggregator.core.store import Store  # noqa: E402
from aggregator.sources.sessions import SessionsSource  # noqa: E402


def main() -> int:
    store = Store()
    store.migrate()
    src = SessionsSource()
    errors: list[str] = []

    print("Building parent → agentId → tool_use_id index …")
    spawn_index = src._collect_agent_spawn_index(errors)
    print(
        f"  parents indexed: {len(spawn_index)}, "
        f"agents mapped: {sum(len(v) for v in spawn_index.values())}"
    )
    if errors:
        print(f"  parse errors: {len(errors)} (first 3): {errors[:3]}")

    conn = sqlite3.connect(store.db_path)
    conn.row_factory = sqlite3.Row
    subs = conn.execute(
        "SELECT session_id, agent_id, parent_session_id "
        "FROM sessions WHERE kind='subagent'"
    ).fetchall()
    print(f"Subagents in cache: {len(subs)}")

    updated = 0
    already = 0
    skipped = 0
    for row in subs:
        sid = row["session_id"]
        agent_id = row["agent_id"]
        parent = row["parent_session_id"]
        if not agent_id or not parent:
            skipped += 1
            continue
        recovered = src._recover_spawn_tool_use_id(spawn_index, parent, agent_id)
        if recovered is None:
            skipped += 1
            continue
        cur = conn.execute(
            "UPDATE sessions SET spawned_by_tool_use_id=? "
            "WHERE session_id=? AND (spawned_by_tool_use_id IS NULL "
            "OR spawned_by_tool_use_id != ?)",
            (recovered, sid, recovered),
        )
        if cur.rowcount:
            updated += 1
        else:
            already += 1
    conn.commit()

    print(f"Updated: {updated}, already-correct: {already}, unrecoverable: {skipped}")
    total_recoverable = updated + already
    if subs:
        print(
            f"Recovery rate: {total_recoverable}/{len(subs)} = "
            f"{100 * total_recoverable / len(subs):.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
