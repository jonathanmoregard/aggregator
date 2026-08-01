"""aggregator CLI. Doubles as the Raycast target.

Subcommands:
  query "DSL"     - run a DSL query, print wrapped records or JSON
  ingest SOURCE   - trigger one source's ingest cycle (does the actual work;
                    the MCP ``aggregator_ingest`` tool from M3 only prints
                    instructions pointing here — this is the human-approve gate)
  status          - print capabilities (sources, freshness, cache path)

Wraps the tested ``aggregator_query`` / ``aggregator_capabilities`` helpers
from ``aggregator.mcp`` (M3) so the CLI never re-implements query semantics.
Ingest talks to source objects directly (they are not exposed via MCP by
design — spec §Security constraint 1).

Injection seams:
* ``_store`` — swap the SQLite backing for tests (default: XDG cache).
* ``_sources`` — swap the source registry for tests (default: real
  ``SessionsSource`` + ``GitHubSource``).

Argparse (not click) per plan; click is not a project dependency and the
subcommand surface is small enough that argparse is cleaner.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from typing import Any

from aggregator.core.store import Store
from aggregator.mcp import (
    aggregator_capabilities as _mcp_capabilities,
)
from aggregator.mcp import (
    aggregator_query as _mcp_query,
)
from aggregator.sources.github import GitHubSource
from aggregator.sources.sessions import SessionsSource


def _default_sources() -> dict[str, Any]:
    """Real source registry. Kept in a function so tests can bypass with
    ``_sources={...}`` without importing the real sources' heavy deps
    (subprocess to ``gh``, ``~/.claude/projects`` filesystem scan)."""
    return {"sessions": SessionsSource(), "github": GitHubSource()}


def _cmd_query(args: argparse.Namespace, store: Store) -> int:
    result = _mcp_query(
        dsl=args.dsl,
        fields=args.fields,
        page_size=args.page_size,
        _store=store,
    )
    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("ok") else 1
    if not result.get("ok"):
        print(f"error: {result.get('reason')}", file=sys.stderr)
        print(f"remediation: {result.get('remediation')}", file=sys.stderr)
        return 1
    for rec in result["records"]:
        print(f"# {rec['source']} :: {rec['subject']}  ({rec['stable_id']})")
        print(rec["content"])
        print()
    if "notice" in result:
        print(f"# notice: {result['notice']}")
    print(f"# total: {result['total']}")
    if "next_page_token" in result:
        print(f"# next_page_token: {result['next_page_token']}")
    return 0


def _cmd_status(args: argparse.Namespace, store: Store) -> int:
    caps = _mcp_capabilities(_store=store)
    if args.json:
        print(json.dumps(caps, indent=2, default=str))
        return 0
    print(f"cache_path: {caps['cache_path']}")
    print(f"schema_version: {caps['schema_version']}")
    print(f"tool_tier: {caps['tool_tier']}")
    print("sources:")
    for s in caps["sources"]:
        fresh = caps["freshness"].get(s, "n/a")
        print(f"  {s}: last_updated={fresh}")
    return 0


def _cmd_ingest(
    args: argparse.Namespace, store: Store, sources: dict[str, Any]
) -> int:
    src = sources.get(args.source)
    if src is None:
        print(f"unknown source: {args.source}", file=sys.stderr)
        print(f"known sources: {sorted(sources)}", file=sys.stderr)
        return 2
    since: datetime | None = None
    if args.since:
        try:
            since = datetime.fromisoformat(args.since)
        except ValueError:
            print(f"bad --since: {args.since}", file=sys.stderr)
            return 2
        if since.tzinfo is None:
            since = since.replace(tzinfo=UTC)
    # Persistence is the CLI's job (advisor round-1 BLOCKER fix). Sources
    # yield ``Record``s via ``iter_records``; we collect and upsert. Old-style
    # sources that only implement ``ingest()`` fall through to the legacy
    # count-only path (kept for the existing dispatch smoke test).
    if hasattr(src, "iter_records"):
        try:
            records = list(src.iter_records(since))
        except Exception as e:  # noqa: BLE001 -- surface as CLI error, don't crash
            print(f"ingest {args.source} failed: {e}", file=sys.stderr)
            return 1
        # Round-2 MEDIUM: when --rebuild is set, run the DELETE + upsert
        # atomically so a fault during upsert can't leave the store empty
        # for the source. Without --rebuild the plain upsert path is fine
        # (idempotent per stable_id; no destructive prior step).
        if args.rebuild:
            store.rebuild_and_upsert(args.source, records)
        else:
            store.upsert(records)
        added = len(records)
        errors: list[str] = []
    else:
        result = src.ingest(since=since)
        added = result.added
        errors = list(result.errors)
    print(
        f"ingest {args.source}: added={added} updated=0 skipped=0 "
        f"errors={len(errors)}"
    )
    if errors:
        for e in errors[:5]:
            print(f"  error: {e}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aggregator")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("query", help="run a DSL query")
    q.add_argument(
        "dsl", help='DSL string, e.g. "source:sessions from:2026-07-25"'
    )
    q.add_argument("--fields", choices=["summary", "full"], default="summary")
    q.add_argument("--page-size", type=int, default=50)
    q.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )

    st = sub.add_parser("status", help="print capabilities / freshness")
    st.add_argument("--json", action="store_true")

    ing = sub.add_parser("ingest", help="run one source's ingest cycle")
    ing.add_argument("source", help="source name, e.g. sessions or github")
    ing.add_argument("--since", help="ISO date to bound the ingest window")
    ing.add_argument(
        "--rebuild",
        action="store_true",
        help="drop this source's rows and re-scan raw",
    )

    return p


def main(
    argv: list[str] | None = None,
    _store: Store | None = None,
    _sources: dict[str, Any] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    store = _store or Store()
    store.migrate()
    sources = _sources if _sources is not None else _default_sources()
    if args.cmd == "query":
        return _cmd_query(args, store)
    if args.cmd == "status":
        return _cmd_status(args, store)
    if args.cmd == "ingest":
        # Round-2 MEDIUM: the atomic DELETE + upsert lives inside
        # ``_cmd_ingest`` via ``store.rebuild_and_upsert`` when
        # ``args.rebuild`` is set. Do NOT call ``store.rebuild`` here —
        # doing so would commit the DELETE before the transaction and
        # reintroduce the non-atomic gap this fix closes.
        return _cmd_ingest(args, store, sources)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
