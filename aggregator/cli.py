"""aggregator CLI. Doubles as the Raycast target.

Subcommands:
  query "DSL"     - run a DSL query, print wrapped records or JSON
  ingest SOURCE   - trigger one source's ingest cycle (does the actual work;
                    the MCP ``aggregator_ingest`` tool only prints
                    instructions pointing here — this is the human-approve gate)
  status          - print capabilities (sources, freshness, cache path)

v2 (Schema B): sessions ingest routes through the entity path
(``iter_entities`` → ``store.upsert_entities`` / ``rebuild_and_upsert_entities``).
GitHub keeps the Record path unchanged.

Chunk 4 wiring: chat-export sources (``chatgpt``, ``claude-web``) ride the
entity path (session-shaped; discovery scans the drops dir AND ~/Downloads);
``research`` and ``sota-watch`` ride the Record path (records-shaped, like
github).

Injection seams:
* ``_store`` — swap the SQLite backing for tests (default: XDG cache).
* ``_sources`` — swap the source registry for tests.

Argparse (not click) per plan.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from typing import Any

from aggregator.core.store import EmptyRebuildRefusedError, Store
from aggregator.mcp import (
    aggregator_capabilities as _mcp_capabilities,
)
from aggregator.mcp import (
    aggregator_query as _mcp_query,
)
from aggregator.sources.base import ObservationRow, SessionRow
from aggregator.sources.chatgpt import ChatGPTSource
from aggregator.sources.claude_web import ClaudeWebSource
from aggregator.sources.github import GitHubSource
from aggregator.sources.research_reports import ResearchReportsSource
from aggregator.sources.sessions import SessionsSource
from aggregator.sources.sota_watch import SotaWatchSource

# Round-1 HIGH-2: partial-parse silent-wipe threshold.
# When --rebuild would drop >20% of rows for a source that already holds
# >100 rows, refuse unless --force. --force still prompts on stdin unless
# --yes is passed (scripted use). Small stores (<=100 existing) bypass the
# ratio check — nothing to protect at that scale.
_RATIO_GUARD_MIN_EXISTING = 100
_RATIO_GUARD_KEEP_FRACTION = 0.8  # refuse if new < 0.8 * existing


def _ratio_guard_would_trip(new_count: int, existing_count: int) -> bool:
    """True when the shrink from ``existing`` to ``new`` exceeds the guard.

    Guard: existing > 100 AND new < 0.8 * existing. Callers still need to
    handle the empty-with-errors case separately (that's a different fault
    mode — see the wipe-on-transient-failure guard around it)."""
    return (
        existing_count > _RATIO_GUARD_MIN_EXISTING
        and new_count < _RATIO_GUARD_KEEP_FRACTION * existing_count
    )


def _confirm_force_on_stdin(prompt: str) -> bool:
    """Ask stdin for a 'y' to confirm. Any other answer returns False.

    Kept out of the ingest paths so tests can monkeypatch sys.stdin without
    reaching into either. EOF / empty answer also returns False.
    """
    print(prompt, file=sys.stderr)
    try:
        answer = sys.stdin.readline().strip().lower()
    except (EOFError, OSError):
        return False
    return answer == "y"


def _default_sources() -> dict[str, Any]:
    """Real source registry. Kept in a function so tests can bypass with
    ``_sources={...}`` without importing the real sources' heavy deps
    (subprocess to ``gh``, ``~/.claude/projects`` filesystem scan).

    All constructors are side-effect-free (env/dir resolution only); no
    filesystem or network work happens until the chosen source's ingest
    runs."""
    return {
        "sessions": SessionsSource(),
        "github": GitHubSource(),
        "chatgpt": ChatGPTSource(),
        "claude-web": ClaudeWebSource(),
        "research": ResearchReportsSource(),
        "sota-watch": SotaWatchSource(),
    }


def _cmd_query(args: argparse.Namespace, store: Store) -> int:
    result = _mcp_query(
        dsl=args.dsl,
        fields=args.fields,
        page_size=args.page_size,
        drilldown=args.drilldown,
        _store=store,
    )
    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("ok") else 1
    if not result.get("ok"):
        print(f"error: {result.get('reason')}", file=sys.stderr)
        print(f"remediation: {result.get('remediation')}", file=sys.stderr)
        return 1
    mode = result.get("mode", "records")
    for rec in result["records"]:
        if mode == "sessions":
            print(
                f"# {rec['source']} :: {rec['subject']}  "
                f"({rec['stable_id']}, matches={rec.get('matching_observations', 0)})"
            )
        elif mode == "observations":
            print(
                f"# obs {rec['type']} @{rec['ts']}  "
                f"({rec['obs_id']}, session={rec['session_id']})"
            )
        else:
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


def _cmd_ingest_entities(
    args: argparse.Namespace,
    store: Store,
    src: Any,
    since: datetime | None,
) -> int:
    """v2 ingest path: source yields SessionRow + ObservationRow entities.

    ``--rebuild`` swaps sessions + observations atomically via
    ``store.rebuild_and_upsert_entities``. Applies the same round-3 HIGH
    silent-wipe guard as the Record path: refuse if the iterator yielded
    zero session rows while errors surfaced.
    """
    errors: list[str] = []
    try:
        try:
            entities = list(src.iter_entities(since, errors=errors))
        except TypeError:
            entities = list(src.iter_entities(since))
    except Exception as e:  # noqa: BLE001
        print(f"ingest {args.source} failed: {e}", file=sys.stderr)
        return 1

    session_count = sum(1 for e in entities if isinstance(e, SessionRow))
    obs_count = sum(1 for e in entities if isinstance(e, ObservationRow))

    if args.rebuild:
        # Chunk 4 guard: `rebuild_and_upsert_entities` atomically replaces
        # the ENTIRE sessions + observations tables — it has no per-origin
        # granularity. Running it for a chat-export source (chatgpt,
        # claude-web) would silently wipe every claude-code session. Only
        # the sessions source (which scans all claude-code JSONLs) may
        # drive the whole-table rebuild; chat exports re-ingest via the
        # idempotent upsert instead.
        if args.source != "sessions":
            print(
                f"ERROR: --rebuild is not supported for source "
                f"{args.source!r}: the entity rebuild path replaces the "
                f"entire sessions/observations tables and would wipe other "
                f"origins' rows. Re-run without --rebuild (ingest is an "
                f"idempotent upsert per session/observation id).",
                file=sys.stderr,
            )
            return 2
        existing = store.count_by_source(args.source)
        if session_count == 0 and (errors or existing > 0):
            print(
                f"ERROR: refusing to rebuild {args.source}: iterator yielded "
                f"0 sessions but {len(errors)} errors occurred "
                f"(store has {existing} existing sessions); store left intact",
                file=sys.stderr,
            )
            for e in errors[:5]:
                print(f"  error: {e}", file=sys.stderr)
            return 1
        # Round-1 HIGH-2: partial-parse ratio guard.
        if _ratio_guard_would_trip(session_count, existing):
            if not getattr(args, "force", False):
                print(
                    f"ERROR: refusing to rebuild {args.source}: got "
                    f"{session_count} new but store has {existing} "
                    f"(would drop >20%); use --force to override",
                    file=sys.stderr,
                )
                return 1
            if not getattr(args, "yes", False):
                confirmed = _confirm_force_on_stdin(
                    f"--force will drop {existing - session_count} rows "
                    f"from {args.source} ({existing} -> {session_count}). "
                    f"Continue? [y/N]"
                )
                if not confirmed:
                    print(
                        f"aborted: {args.source} rebuild not confirmed; "
                        f"store left intact",
                        file=sys.stderr,
                    )
                    return 1
        min_sessions = 1 if existing > 0 else 0
        try:
            store.rebuild_and_upsert_entities(entities, min_sessions=min_sessions)
        except EmptyRebuildRefusedError as e:
            print(f"ERROR: {e}; store left intact", file=sys.stderr)
            return 1
    else:
        store.upsert_entities(entities)

    print(
        f"ingest {args.source}: sessions={session_count} "
        f"observations={obs_count} errors={len(errors)}"
    )
    if errors:
        for e in errors[:5]:
            print(f"  error: {e}", file=sys.stderr)
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
    # v2: entity-shaped sources (sessions) route through iter_entities +
    # upsert_entities. Record-shaped (github) fall through to iter_records +
    # upsert. Sources exposing only ingest() (old test stubs) use the
    # count-only path.
    if hasattr(src, "iter_entities"):
        return _cmd_ingest_entities(args, store, src, since)
    if hasattr(src, "iter_records"):
        errors: list[str] = []
        try:
            # Round-3 HIGH/MEDIUM#3: plumb the errors sink into iter_records
            # so we can (a) refuse a wipe when every endpoint degrades to []
            # with errors present and (b) surface warnings post-ingest.
            # Older iter_records signatures without an ``errors`` kwarg
            # (e.g. the CLI persistence stub) are handled via TypeError
            # fallback so we don't break narrow test doubles.
            try:
                records = list(src.iter_records(since, errors=errors))
            except TypeError:
                records = list(src.iter_records(since))
        except Exception as e:  # noqa: BLE001 -- surface as CLI error, don't crash
            print(f"ingest {args.source} failed: {e}", file=sys.stderr)
            return 1
        # Round-2 MEDIUM: when --rebuild is set, run the DELETE + upsert
        # atomically so a fault during upsert can't leave the store empty
        # for the source. Without --rebuild the plain upsert path is fine
        # (idempotent per stable_id; no destructive prior step).
        if args.rebuild:
            # Round-3 HIGH: refuse a rebuild that would silently wipe rows.
            # Two overlapping conditions trip the refusal:
            #   (a) records is empty AND at least one endpoint reported an
            #       error — classic transient-failure wipe pattern;
            #   (b) records is empty AND the store currently holds rows for
            #       this source — historical data is at stake, so demand
            #       positive evidence (a nonzero record set) before allowing
            #       the DELETE to commit.
            existing = store.count_by_source(args.source)
            if not records and (errors or existing > 0):
                print(
                    f"ERROR: refusing to rebuild {args.source}: iterator "
                    f"yielded 0 records but {len(errors)} errors occurred "
                    f"(store has {existing} existing rows); store left intact",
                    file=sys.stderr,
                )
                for e in errors[:5]:
                    print(f"  error: {e}", file=sys.stderr)
                return 1
            # Round-1 HIGH-2: partial-parse ratio guard (records path).
            if _ratio_guard_would_trip(len(records), existing):
                if not getattr(args, "force", False):
                    print(
                        f"ERROR: refusing to rebuild {args.source}: got "
                        f"{len(records)} new but store has {existing} "
                        f"(would drop >20%); use --force to override",
                        file=sys.stderr,
                    )
                    return 1
                if not getattr(args, "yes", False):
                    confirmed = _confirm_force_on_stdin(
                        f"--force will drop {existing - len(records)} rows "
                        f"from {args.source} ({existing} -> {len(records)}). "
                        f"Continue? [y/N]"
                    )
                    if not confirmed:
                        print(
                            f"aborted: {args.source} rebuild not confirmed; "
                            f"store left intact",
                            file=sys.stderr,
                        )
                        return 1
            # Belt-and-braces: pass the store guard too, in case a future
            # caller reaches this path without the CLI's early return.
            min_records = 1 if existing > 0 else 0
            try:
                store.rebuild_and_upsert(
                    args.source, records, min_records=min_records
                )
            except EmptyRebuildRefusedError as e:
                print(f"ERROR: {e}; store left intact", file=sys.stderr)
                return 1
        else:
            store.upsert(records)
        added = len(records)
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


def _cmd_github_token_status(
    args: argparse.Namespace,
    store: Store,
    sources: dict[str, Any],
) -> int:
    """Report which token the github ingest path would use + scopes.

    Read-only, idempotent, no store writes. Consumes the same source
    registry as ``ingest`` so tests can inject a stub without touching
    the real gh CLI.
    """
    _ = store  # signature symmetry; deliberately unused
    src = sources.get("github")
    if src is None:
        print(
            "unknown source: github (registry has: "
            f"{sorted(sources)})",
            file=sys.stderr,
        )
        return 2
    if not hasattr(src, "token_status"):
        print(
            "registered github source does not support token_status()",
            file=sys.stderr,
        )
        return 2
    status = src.token_status()
    if args.json:
        # Dataclasses aren't natively JSON-serialisable but their __dict__
        # is a flat dict of primitives here, so this is safe.
        print(json.dumps(status.__dict__, indent=2, default=str))
        return 0
    # Human-readable summary. Kept dense — one line per field, then the
    # recommendation on its own row for quick eyeballing.
    print(f"token source: {status.source}")
    if status.scopes:
        print(f"scopes:       {', '.join(status.scopes)}")
    else:
        print("scopes:       (none / unverified)")
    print(f"write_capable: {status.write_capable}")
    print(
        f"override:     "
        f"{'AGGREGATOR_ALLOW_WRITE_TOKEN=1' if status.override_active else 'unset'}"
    )
    if status.scope_error:
        print(f"scope_error:  {status.scope_error}")
    print(f"recommendation: {status.recommendation}")
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
    q.add_argument(
        "--drilldown",
        action="store_true",
        help=(
            "for session-shaped queries, return observation rows for the "
            "matching sessions (default: session-level hit list only)"
        ),
    )

    st = sub.add_parser("status", help="print capabilities / freshness")
    st.add_argument("--json", action="store_true")

    ing = sub.add_parser("ingest", help="run one source's ingest cycle")
    ing.add_argument(
        "source",
        help=(
            "source name: sessions | github | chatgpt | claude-web | "
            "research | sota-watch"
        ),
    )
    ing.add_argument("--since", help="ISO date to bound the ingest window")
    ing.add_argument(
        "--rebuild",
        action="store_true",
        help="drop this source's rows and re-scan raw",
    )
    ing.add_argument(
        "--force",
        action="store_true",
        help=(
            "bypass the >20%% row-drop guard on --rebuild (still requires "
            "'y' confirmation on stdin unless --yes is also passed)"
        ),
    )
    ing.add_argument(
        "--yes",
        action="store_true",
        help="assume 'y' for any --force confirmation (scripted use)",
    )

    tks = sub.add_parser(
        "github-token-status",
        help=(
            "report which token the github ingest would use + its scopes "
            "+ a read-only-token recommendation"
        ),
    )
    tks.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON",
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
    if args.cmd == "github-token-status":
        return _cmd_github_token_status(args, store, sources)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
