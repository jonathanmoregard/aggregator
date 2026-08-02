"""One-shot: re-classify existing ``observations.type='other'`` rows via the
M2 expanded ``_KNOWN_TYPES`` allowlist — no full re-ingest.

Rationale: M2 fix landed an expanded type enum (attachment, progress,
queue-operation, etc.). The 74404 rows currently sitting in ``other`` were
written by the old parser; rebuilding the whole cache to reclassify them
takes minutes. Instead this script re-reads each obs's JSONL line by
(jsonl_path, obs_id), extracts the raw ``type`` field, and UPDATEs the
observation row in place.

Usage:
    uv run --directory ~/Repos/aggregator python scripts/reclassify_other_obs.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aggregator.core.store import Store  # noqa: E402
from aggregator.sources.sessions import _KNOWN_TYPES  # noqa: E402


def main() -> int:
    store = Store()
    store.migrate()
    conn = sqlite3.connect(store.db_path)
    conn.row_factory = sqlite3.Row

    # Group all 'other' obs by their file so we scan each JSONL once.
    print("Loading obs where type='other' …")
    rows = conn.execute(
        "SELECT o.obs_id, s.jsonl_path "
        "FROM observations o "
        "JOIN sessions s ON o.session_id = s.session_id "
        "WHERE o.type = 'other'"
    ).fetchall()
    print(f"  {len(rows)} rows to consider")

    per_file: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        per_file[r["jsonl_path"]].add(r["obs_id"])

    updates: list[tuple[str, str]] = []  # (new_type, obs_id)
    missing_files = 0
    unresolved = 0
    for path, obs_ids in per_file.items():
        p = Path(path)
        if not p.exists():
            missing_files += 1
            continue
        try:
            fh = p.open(encoding="utf-8", errors="replace")
        except OSError:
            missing_files += 1
            continue
        with fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                uid = obj.get("uuid")
                if not isinstance(uid, str) or uid not in obs_ids:
                    continue
                raw_t = obj.get("type")
                if isinstance(raw_t, str) and raw_t in _KNOWN_TYPES and raw_t != "other":
                    updates.append((raw_t, uid))
                else:
                    unresolved += 1

    print(
        f"Files scanned: {len(per_file)}, "
        f"missing on disk: {missing_files}, "
        f"resolved: {len(updates)}, unresolved: {unresolved}"
    )

    # Report which raw types will land where
    type_tally = Counter(new_t for new_t, _ in updates)
    for t, n in type_tally.most_common():
        print(f"  → {t}: {n}")

    if not updates:
        print("Nothing to update.")
        return 0

    print("Applying UPDATEs …")
    conn.executemany(
        "UPDATE observations SET type=? WHERE obs_id=?",
        updates,
    )
    conn.commit()
    print("Done.")

    # Post-check
    other = conn.execute("SELECT COUNT(*) FROM observations WHERE type='other'").fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    print(f"Post: type='other' = {other}/{total} = {100*other/total:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
