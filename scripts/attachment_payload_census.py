"""Census the `attachment` payloads that ingest currently discards.

WHY THIS EXISTS. 22.4% of `observations` are `type='attachment'` with an empty
``body``, which reads like a coverage gap and is not one: they are structural
nodes in the ``parent_obs_id`` chain, and ~86% of what ingest dropped is harness
plumbing repeated near-verbatim in every session. Embedding that bulk would
poison the vector arm with near-duplicates rather than close a gap. See
``docs/rag-next-steps.md``.

The numbers in that document came from here, so they can be re-derived rather
than believed. Reads the SOURCE jsonl, not the cache — the cache already threw
the payload away, mapping nothing from the top-level ``attachment`` key to
``body``.

READ-ONLY, and deliberately so: it opens ``cache.db`` with ``mode=ro`` purely to
enumerate ``jsonl_path``, and opens every transcript for reading. It writes
nothing anywhere.

    python scripts/attachment_payload_census.py            # 400-file sample
    python scripts/attachment_payload_census.py --all      # every file
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sqlite3
import sys

# The subtypes that carry content a human would search for, as opposed to
# context the harness injected. Kept here rather than in prose because the
# ingest filter this motivates has to name exactly this set, and the two
# drifting apart is the failure worth preventing.
CONTENT_BEARING = frozenset(
    {
        "file",
        "edited_text_file",
        "nested_memory",
        "queued_command",
        "invoked_skills",
        # 3 rows in the whole corpus, but a mean 6.3 KB of referenced plan —
        # in on payload, not on frequency. The other four subtypes the census
        # surfaced (auto_mode, plan_mode, plan_mode_exit, directory) are mode
        # markers averaging 5-195 bytes and are listed as plumbing below.
        "plan_file_reference",
    }
)

DEFAULT_SAMPLE = 400
# Fixed so two runs of the sampling path are comparable. --all ignores it.
SEED = 11


def _default_cache() -> str:
    return os.path.join(
        os.environ.get(
            "XDG_DATA_HOME", os.path.expanduser("~/.local/share")
        ),
        "aggregator",
        "cache.db",
    )


def _text_len(payload: object) -> int:
    """Roughly, how much readable text this payload carries.

    Sums string values and serialises nested structure, skipping ``type``
    itself. Approximate on purpose — it exists to separate a 20 KB skill
    catalogue from a 2-byte todo marker, not to predict a token count.
    """
    if not isinstance(payload, dict):
        return len(str(payload))
    total = 0
    for key, value in payload.items():
        if key == "type":
            continue
        if isinstance(value, str):
            total += len(value)
        elif isinstance(value, (list, dict)):
            total += len(json.dumps(value))
    return total


def _transcript_paths(cache: str) -> list[str]:
    if not os.path.exists(cache):
        sys.exit(f"no cache at {cache} — nothing to enumerate")
    conn = sqlite3.connect(f"file:{cache}?mode=ro", uri=True)
    try:
        return [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT jsonl_path FROM sessions "
                "WHERE origin='claude-code' AND jsonl_path IS NOT NULL"
            )
        ]
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=_default_cache())
    ap.add_argument(
        "--all",
        action="store_true",
        help="scan every transcript instead of a seeded sample (slow)",
    )
    ap.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    args = ap.parse_args()

    paths = _transcript_paths(args.cache)
    if args.all:
        chosen = paths
    else:
        random.seed(SEED)
        chosen = random.sample(paths, min(args.sample, len(paths)))
    print(f"transcripts known: {len(paths)}; scanning {len(chosen)}")

    rows: collections.Counter[str] = collections.Counter()
    raw_bytes: collections.Counter[str] = collections.Counter()
    text_len: collections.Counter[str] = collections.Counter()
    read = 0
    unreadable = 0

    for path in chosen:
        try:
            with open(path, encoding="utf-8") as handle:
                read += 1
                for line in handle:
                    try:
                        entry = json.loads(line)
                    except ValueError:
                        continue
                    if entry.get("type") != "attachment":
                        continue
                    payload = entry.get("attachment")
                    kind = (
                        payload.get("type", "<no-type>")
                        if isinstance(payload, dict)
                        else "<not-a-dict>"
                    )
                    rows[kind] += 1
                    raw_bytes[kind] += len(line)
                    text_len[kind] += _text_len(payload)
        except OSError:
            # A transcript the cache remembers and the filesystem no longer
            # has. Counted rather than ignored: a large number here means the
            # census is describing a corpus that has partly evaporated. A
            # failure part-way through a file also lands here, so `read` and
            # `unreadable` can both count the same path — deliberate, since
            # the alternative is reporting a partial scan as a whole one.
            unreadable += 1
            continue

    total = sum(rows.values())
    print(f"transcripts read: {read} (unreadable: {unreadable})")
    print(f"attachment rows seen: {total}")
    if not total:
        print("no attachment rows in this sample")
        return 0

    print()
    header = f"{'attachment.type':<34}{'rows':>8}{'% rows':>8}{'MB raw':>9}{'mean text':>11}"
    print(header)
    print("-" * len(header))
    for kind, count in rows.most_common():
        mean_text = text_len[kind] / count
        marker = "  <- content" if kind in CONTENT_BEARING else ""
        print(
            f"{kind:<34}{count:>8}{100 * count / total:>7.1f}%"
            f"{raw_bytes[kind] / 1e6:>9.1f}{mean_text:>11.0f}{marker}"
        )

    carried = sum(rows[k] for k in CONTENT_BEARING)
    print()
    print(
        f"content-bearing subtypes: {carried} rows "
        f"({100 * carried / total:.1f}% of attachments) — the slice worth "
        f"writing to `body`"
    )
    print(
        f"harness plumbing: {total - carried} rows "
        f"({100 * (total - carried) / total:.1f}%) — near-duplicate across "
        f"sessions; embedding it would poison the vector arm"
    )
    unknown = set(rows) - CONTENT_BEARING
    novel = {k for k in unknown if rows[k] and k not in _KNOWN_PLUMBING}
    if novel:
        print()
        print(
            "subtypes not in either list — classify before writing the ingest "
            f"filter: {', '.join(sorted(novel))}"
        )
    return 0


# Everything observed in the 2026-08-24 census that is harness plumbing. Listed
# so a subtype nobody has seen before is REPORTED rather than silently counted
# as plumbing — the same reason the FTS5 MATCH scanner refuses to bucket a site
# it does not recognise.
_KNOWN_PLUMBING = frozenset(
    {
        "hook_success",
        "hook_additional_context",
        "task_reminder",
        "skill_listing",
        "deferred_tools_delta",
        "mcp_instructions_delta",
        "agent_listing_delta",
        "async_hook_response",
        "total_tokens_reminder",
        "todo_reminder",
        "command_permissions",
        "budget_usd",
        "max_turns_reached",
        "date_change",
        "dynamic_skill",
        "hook_non_blocking_error",
        "read_truncation_notice",
        "hook_permission_decision",
        "hook_cancelled",
        "hook_system_message",
        "task_status",
        "hook_blocking_error",
        "ultrathink_effort",
        "compact_file_reference",
        # Surfaced by the unknown-subtype guard on the first full scan, which
        # is what it is for. All four are mode/state markers: 5-64 bytes of
        # payload each, except `directory` at 195 bytes and a single row.
        "auto_mode",
        "plan_mode",
        "plan_mode_exit",
        "directory",
    }
)


if __name__ == "__main__":
    raise SystemExit(main())
