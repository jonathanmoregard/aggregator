"""aggregator CLI. Doubles as the Raycast target.

Subcommands:
  query "DSL"     - run a DSL query, print wrapped records or JSON
  ingest SOURCE   - trigger one source's ingest cycle (does the actual work;
                    the MCP ``aggregator_ingest`` tool only prints
                    instructions pointing here — this is the human-approve gate)
  ingest --all    - run EVERY source through the unified import runner
                    (``imports/runner.py``), one pass, one report. This is the
                    command a single systemd timer runs: per-source failures
                    are isolated, counts are real, and inputs that only a
                    human refreshes get a staleness warning. Exit 3 when any
                    source ended with errors.
  status          - print capabilities (sources, freshness, cache path)

v2 (Schema B): sessions ingest routes through the entity path
(``iter_entities`` → ``store.upsert_entities`` / ``rebuild_and_upsert_entities``).
GitHub keeps the Record path unchanged.

Chunk 4 wiring: chat-export sources (``chatgpt``, ``claude-web``) ride the
entity path (session-shaped; discovery scans the drops dir AND ~/Downloads);
``research``, ``sota-watch``, and ``substack`` ride the Record path
(records-shaped, like github).

Injection seams:
* ``_store`` — swap the SQLite backing for tests (default: XDG cache).
* ``_sources`` — swap the source registry for tests.
* ``_adapters`` — swap the import-port registry driving ``ingest --all``.
* ``_notify`` — the run-report notifier. Fires on EVERY ``--all`` run, not
  only failing ones: a run that imported nothing because the export archive
  is a month old exits 0 and is otherwise indistinguishable from a healthy
  no-op. Defaults to doing nothing.

Argparse (not click) per plan.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from aggregator.core.store import EmptyRebuildRefusedError, Store
from aggregator.imports.port import ImportAdapter
from aggregator.imports.registry import default_adapters
from aggregator.imports.runner import NotifyHook, RunReport, run_imports
from aggregator.imports.store_sink import StoreSink, count_writes
from aggregator.imports.sync_bridge import accepts_errors_kwarg
from aggregator.mcp import (
    aggregator_capabilities as _mcp_capabilities,
)
from aggregator.mcp import (
    aggregator_query as _mcp_query,
)
from aggregator.sources.base import ObservationRow, SessionRow
from aggregator.sources.chatgpt import ChatGPTSource
from aggregator.sources.claude_web import ClaudeWebSource
from aggregator.sources.dropbox import DropboxSource
from aggregator.sources.github import GitHubSource
from aggregator.sources.research_reports import ResearchReportsSource
from aggregator.sources.sessions import SessionsSource
from aggregator.sources.sota_watch import SotaWatchSource
from aggregator.sources.substack import SubstackSource
from aggregator.sources.ticktick import TickTickSource

# Round-1 HIGH-2: partial-parse silent-wipe threshold.
# When --rebuild would drop >20% of rows for a source that already holds
# >100 rows, refuse unless --force. --force still prompts on stdin unless
# --yes is passed (scripted use). Small stores (<=100 existing) bypass the
# ratio check — nothing to protect at that scale.
_RATIO_GUARD_MIN_EXISTING = 100
_RATIO_GUARD_KEEP_FRACTION = 0.8  # refuse if new < 0.8 * existing

# Exit codes. 0 clean; 1 hard failure; 2 usage error (unknown source, bad
# --since, unknown subcommand) — both pre-existing and deliberately not
# renumbered. 3 is the new one: the run completed but ended with a non-empty
# errors list. Distinct from 2 so the systemd wrapper can tell a typo'd
# source name from a run that dropped three PDFs; distinct from 0 because
# `tasks/session-constraints.md` requires a failed ingest to be loud, and a
# timer that reads 0 as success lets the index rot unnoticed.
EXIT_COMPLETED_WITH_ERRORS = 3

# How old a manually-refreshed input may get before `ingest --all` nags.
# Overridable with `--stale-after-days`.
#
# 14 days, chosen against the ritual it is nagging about: the export-archive
# sources (chatgpt, claude-web, substack, and ticktick's CSV leg) are realistic
# to refresh about monthly, so two weeks is half a cycle — late enough that a
# freshly-run ritual is never nagged, early enough that the operator is told
# before the gap is a month wide. Tighter and the warning fires on healthy
# runs and gets tuned out, which is worse than not warning at all.
DEFAULT_STALE_AFTER_DAYS = 14

# Which ``sessions.origin`` populations `ingest sessions --rebuild` is allowed
# to DELETE. Exactly the one the sessions source can regenerate: it scans
# ~/.claude/projects and every row it emits carries the ``SessionRow.origin``
# default, ``'claude-code'``.
#
# The sessions and observations tables also hold ``chatgpt`` and
# ``claude-web`` rows, and those are NOT regenerable — their only source is a
# vendor export archive a human downloads by hand, so once the drop is gone
# the rows in this database are the last copy. Before this scope existed the
# rebuild's DELETE was unqualified and took them too, and the >20% shrink
# guard could not catch it: it compared the incoming claude-code count against
# the WHOLE table, so a store of 840 claude-code + 160 claude-web rows read as
# a 16% shrink, sailed through the guard, exited 0, and destroyed all 160.
SESSIONS_REBUILD_ORIGINS = ("claude-code",)

# Record-shaped sources whose ``--rebuild`` is refused, and why.
#
# --rebuild adds exactly one thing over a plain ingest: it DELETEs the rows a
# re-scan did not produce. For a source whose stored rows are strictly a
# superset of what any future scan can produce, that DELETE can only ever
# destroy data — there is no state it can usefully correct.
REBUILD_UNSUPPORTED_SOURCES: dict[str, str] = {
    "ticktick": (
        "its stored rows include api-inferred-complete tasks that nothing can "
        "regenerate. The Open API serves OPEN tasks only and reports a "
        "completion exactly once, as a disappearance between two polls, so a "
        "completed task is never in a poll again; and the CSV backups that "
        "would confirm it are a manual export whose only surviving copy is "
        "the local archive. A rebuild that shrinks the source by under 20% "
        "clears the shrink guard silently and takes those rows with it."
    ),
}


def _ratio_guard_would_trip(new_count: int, existing_count: int) -> bool:
    """True when the shrink from ``existing`` to ``new`` exceeds the guard.

    Guard: existing > 100 AND new < 0.8 * existing. Callers still need to
    handle the empty-with-errors case separately (that's a different fault
    mode — see the wipe-on-transient-failure guard around it)."""
    return (
        existing_count > _RATIO_GUARD_MIN_EXISTING
        and new_count < _RATIO_GUARD_KEEP_FRACTION * existing_count
    )


def _parse_since(raw: str | None) -> datetime | None:
    """Parse ``--since`` into an aware UTC datetime. Raises ``ValueError``.

    Naive input is stamped UTC because every timestamp downstream is aware and
    comparing the two raises. Shared by both ingest paths so the single-source
    and run-all windows cannot mean different things.
    """
    if not raw:
        return None
    since = datetime.fromisoformat(raw)
    return since if since.tzinfo is not None else since.replace(tzinfo=UTC)


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


class _UnbuildableSource:
    """Stands in for a source whose constructor RAISED.

    The mirror of ``imports/registry._UnbuildableAdapter``, for the
    single-source path. Same reasoning: construction here is documented as
    side-effect-free (env and path resolution only), which is exactly why an
    environment-dependent raise is the plausible failure — and exactly why it
    must cost only its own source.

    It carries no ``iter_entities`` / ``iter_records`` / ``ingest``, so it can
    never be mistaken for a working source and quietly ingest nothing; the
    ingest path checks for it by type and reports the original error.
    """

    def __init__(self, name: str, error: BaseException) -> None:
        self.name = name
        self.error = error

    def __str__(self) -> str:
        return (
            f"{self.name}: source could not be constructed: "
            f"{type(self.error).__name__}: {self.error}"
        )


def _build_source(name: str, factory: Callable[[], Any]) -> Any:
    try:
        return factory()
    except Exception as e:  # noqa: BLE001 -- isolation boundary, see the class
        return _UnbuildableSource(name, e)


def _default_sources() -> dict[str, Any]:
    """Real source registry. Kept in a function so tests can bypass with
    ``_sources={...}`` without importing the real sources' heavy deps
    (subprocess to ``gh``, ``~/.claude/projects`` filesystem scan).

    All constructors are side-effect-free (env/dir resolution only); no
    filesystem or network work happens until the chosen source's ingest
    runs. A constructor that raises anyway yields an ``_UnbuildableSource``
    under the same name instead of propagating — one broken source costs its
    own ``ingest <name>`` and nothing else.

    Called LAZILY, from the commands that need it (see ``main``). Built
    eagerly before dispatch, as it was, a single raising constructor took down
    ``query`` and ``status``, which consult no source at all, and ``ingest
    --all``, which drives the adapter registry instead — and it did so
    UPSTREAM of ``_UnbuildableAdapter``, so the run-all isolation never got to
    contain anything."""
    factories: list[tuple[str, Callable[[], Any]]] = [
        ("sessions", SessionsSource),
        ("github", GitHubSource),
        ("chatgpt", ChatGPTSource),
        ("claude-web", ClaudeWebSource),
        ("research", ResearchReportsSource),
        ("sota-watch", SotaWatchSource),
        ("substack", SubstackSource),
        ("dropbox", DropboxSource),
        ("ticktick", TickTickSource),
    ]
    return {name: _build_source(name, factory) for name, factory in factories}


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


def _iterate(
    iter_fn: Callable[..., Any],
    since: datetime | None,
    errors: list[str],
) -> Any:
    """Call a source's iterator, passing ``errors`` only if it takes one.

    The signature is PROBED, never discovered by calling and catching. Both
    ingest paths used to do this::

        try:
            records = list(src.iter_records(since, errors=errors))
        except TypeError:
            records = list(src.iter_records(since))

    Argument binding raises TypeError at call time, so that handled the
    old-signature case — and equally handled a genuine TypeError raised from
    arbitrary depth inside the iteration, whereupon it silently ran the source
    AGAIN. Iterating a source is not a read-only act: the TickTick poll's
    ``reconcile_open_tasks`` advances the on-disk open-task baseline while it
    runs, and the API only ever serves OPEN tasks, so the second pass sees no
    disappearances and that poll's inferred completions are gone for good —
    on a run that then exits 0 because the retry succeeded.

    Same helper the run-all path uses (``sync_bridge.accepts_errors_kwarg``),
    so the two surfaces cannot drift on which sources get an errors sink.
    """
    if accepts_errors_kwarg(iter_fn):
        return iter_fn(since, errors=errors)
    return iter_fn(since)


def _cmd_ingest_entities(
    args: argparse.Namespace,
    store: Store,
    src: Any,
    since: datetime | None,
) -> int:
    """v2 ingest path: source yields SessionRow + ObservationRow entities.

    ``--rebuild`` swaps sessions + observations atomically via
    ``store.rebuild_and_upsert_entities``, SCOPED to the origins the source
    can regenerate (see ``SESSIONS_REBUILD_ORIGINS``). Applies the same
    round-3 HIGH silent-wipe guard as the Record path: refuse if the iterator
    yielded zero session rows while errors surfaced.
    """
    errors: list[str] = []
    try:
        entities = list(_iterate(src.iter_entities, since, errors))
    except Exception as e:  # noqa: BLE001
        print(f"ingest {args.source} failed: {e}", file=sys.stderr)
        return 1

    session_count = sum(1 for e in entities if isinstance(e, SessionRow))
    obs_count = sum(1 for e in entities if isinstance(e, ObservationRow))

    if args.rebuild:
        # Whether --rebuild is supported at all was settled by
        # ``_rebuild_refusal`` BEFORE the iterator ran — see there for why the
        # order matters.
        #
        # Counted over the SAME origins the DELETE will reach. Counted over
        # the whole table instead, the chat-export rows pad the denominator
        # and hide a shrink the operator would otherwise be asked about.
        existing = store.count_sessions_by_origin(SESSIONS_REBUILD_ORIGINS)
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
            store.rebuild_and_upsert_entities(
                entities,
                min_sessions=min_sessions,
                origins=SESSIONS_REBUILD_ORIGINS,
            )
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
        return EXIT_COMPLETED_WITH_ERRORS
    return 0


def _rebuild_refusal(name: str, src: Any) -> str | None:
    """Why ``--rebuild`` is refused for this source, or None if it is allowed.

    CALLED BEFORE THE SOURCE IS ITERATED, and that ordering is the point.
    ``_cmd_ingest`` used to list the iterator first and only then decide
    whether the rebuild was permitted or whether a guard refused it — but
    iterating a source is not a read-only act. The TickTick poll's
    ``reconcile_open_tasks`` diffs against the previous open-task baseline and
    then makes THIS poll the new baseline, on disk, during iteration. A run
    that consumed that baseline and then exited without writing anything took
    every completion it had just inferred with it: the API only ever serves
    OPEN tasks, so a task that disappeared between two polls is reported
    exactly once, and there is no second chance to notice. Deciding first
    means a refused run never touches it.

    Refusals are exit code 2 (usage error): the flag cannot mean what it says
    for this source, which is a different thing from a guard refusing a
    particular run's numbers.
    """
    if hasattr(src, "iter_entities") and name != "sessions":
        # `rebuild_and_upsert_entities` replaces sessions + observations for
        # the origins it is scoped to. A chat-export source re-ingests via the
        # idempotent per-id upsert instead; nothing about its rows needs a
        # DELETE first.
        return (
            f"ERROR: --rebuild is not supported for source {name!r}: the "
            f"entity rebuild path replaces the sessions/observations rows "
            f"wholesale and this source's export archive is the only copy of "
            f"them. Re-run without --rebuild (ingest is an idempotent upsert "
            f"per session/observation id)."
        )
    reason = REBUILD_UNSUPPORTED_SOURCES.get(name)
    if reason is not None:
        return (
            f"ERROR: --rebuild is not supported for source {name!r}: "
            f"{reason} Re-run without --rebuild (ingest is an idempotent "
            f"upsert per stable_id, so a re-scan already overwrites every row "
            f"it can produce)."
        )
    return None


def _cmd_ingest(
    args: argparse.Namespace, store: Store, sources: dict[str, Any]
) -> int:
    src = sources.get(args.source)
    if src is None:
        print(f"unknown source: {args.source}", file=sys.stderr)
        print(f"known sources: {sorted(sources)}", file=sys.stderr)
        return 2
    if isinstance(src, _UnbuildableSource):
        # 1 (hard failure), not 2 (usage error): the operator typed a real
        # source name, and this machine's environment is why it cannot run.
        print(f"ingest {src}", file=sys.stderr)
        return 1
    try:
        since = _parse_since(args.since)
    except ValueError:
        print(f"bad --since: {args.since}", file=sys.stderr)
        return 2
    if args.rebuild:
        refusal = _rebuild_refusal(args.source, src)
        if refusal is not None:
            print(refusal, file=sys.stderr)
            return 2
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
            records = list(_iterate(src.iter_records, since, errors))
        except Exception as e:  # noqa: BLE001 -- surface as CLI error, don't crash
            print(f"ingest {args.source} failed: {e}", file=sys.stderr)
            return 1
        # Counts are probed BEFORE any write. Every write path here is an
        # upsert (and --rebuild deletes first), so once the write has landed
        # there is no way to tell an insert from an overwrite — which is how
        # this summary came to print ``added=len(records) updated=0`` on
        # every run, identical whether the run imported 313 new PRs or
        # re-wrote the same 313 rows. Same helper the runner's sink uses, so
        # the two surfaces cannot drift apart on what "added" means.
        # On --rebuild the question is still "was this id already known?",
        # answered against the pre-run store, not the emptied one.
        counts = count_writes(store, "records", [r.stable_id for r in records])
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
        added, updated, skipped = counts.added, counts.updated, counts.skipped
    else:
        result = src.ingest(since=since)
        added = result.added
        updated = result.updated
        skipped = result.skipped
        errors = list(result.errors)
    print(
        f"ingest {args.source}: added={added} updated={updated} "
        f"skipped={skipped} errors={len(errors)}"
    )
    if errors:
        for e in errors[:5]:
            print(f"  error: {e}", file=sys.stderr)
        # 3, not 0: a run that completed but dropped files is not a success,
        # and a partially-successful run is not a successful run — some
        # records landing does not make the missing ones acceptable. A timer
        # reporting success while the index rots is indistinguishable from
        # one with nothing to do. 3 rather than 2 because 2 is this file's
        # usage-error code and the systemd wrapper must tell them apart:
        # "you typed a bad source name" and "the run dropped three PDFs"
        # need different notification text and different human responses.
        return EXIT_COMPLETED_WITH_ERRORS
    return 0


def _ingest_usage_error(args: argparse.Namespace) -> str | None:
    """Reject ingest invocations that cannot mean what they say. None = fine.

    Every case here would otherwise SUCCEED while doing something other than
    what was typed, which is the failure shape this repo keeps ruling out —
    a run that looks like it worked is worse than one that stops.
    """
    if args.all_sources and args.source:
        return (
            f"ingest: --all takes no source name (got {args.source!r}); "
            f"run either `aggregator ingest {args.source}` or "
            f"`aggregator ingest --all`"
        )
    if not args.all_sources and not args.source:
        return (
            "ingest: name a source, or pass --all to run every source "
            "through the unified runner"
        )
    if args.all_sources and args.rebuild:
        # --rebuild drops a source's rows before re-scanning and is guarded
        # per-source (ratio guard, empty-result guard, stdin confirmation).
        # Across nine sources at once those guards would each need their own
        # answer, and the entity rebuild replaces the WHOLE sessions +
        # observations tables — it would wipe the chat-export origins. The
        # runner path is upsert-only, which is idempotent per id anyway.
        return (
            "ingest: --rebuild is not supported with --all; rebuild one "
            "source at a time (`aggregator ingest <name> --rebuild`)"
        )
    if args.rebuild and args.since:
        # Round-2 HIGH-3. The two flags contradict each other on the one thing
        # --rebuild adds over a plain ingest: the DELETE of every row the scan
        # did not reproduce. --since narrows what the scan CAN reproduce, so
        # the combination deletes the history outside the window — data the
        # run never even looked at.
        #
        # Neither guard catches it. The ratio guard is bypassed outright below
        # 100 existing rows (measured: 5 rows in, window covers 1, store ends
        # at 1, exit 0), and above it a window covering >=80% of the store
        # keeps the shrink under the threshold. Nothing prints a ``deleted=``
        # count, so the summary of that run reads
        # ``added=0 updated=1 skipped=0 errors=0``.
        return (
            f"ingest: --rebuild cannot be combined with --since "
            f"({args.since!r}): --rebuild DELETEs every row the scan did not "
            f"reproduce, and --since narrows the scan, so rows outside the "
            f"window would be deleted without ever being read. Drop --since "
            f"for a full re-scan, or drop --rebuild (a plain ingest is an "
            f"idempotent upsert and deletes nothing)."
        )
    # Flags that parse fine and then do nothing. Same rule as the cases above:
    # an invocation that succeeds while ignoring what was typed is worse than
    # one that stops, and these three are the ones where the operator's mental
    # model is furthest from what happened — somebody passing --force believes
    # they have authorised a destructive run.
    if args.stale_after_days is not None and not args.all_sources:
        return (
            "ingest: --stale-after-days only applies to --all (the input "
            "staleness check runs over the whole registry); drop it or run "
            "`aggregator ingest --all --stale-after-days "
            f"{args.stale_after_days}`"
        )
    unused = [
        flag
        for flag, present in (("--force", args.force), ("--yes", args.yes))
        if present
    ]
    if unused and not args.rebuild:
        return (
            f"ingest: {' and '.join(unused)} only applies to --rebuild (it "
            f"overrides the >20% row-drop guard, and nothing is dropped "
            f"without --rebuild); drop it or add --rebuild"
        )
    return None


def _silent_notification(report: RunReport) -> None:
    """The CLI's default notifier: do nothing.

    The runner deliberately refuses to shell out to ``notify-send`` itself,
    naming the CLI / systemd layer as where a real notifier belongs. This is
    that seam standing empty: the desktop wiring is unit configuration, not
    Python, and an interactive `aggregator ingest --all` should not pop a
    desktop toast. Injected through ``main(_notify=...)``.
    """


def _cmd_ingest_all(
    args: argparse.Namespace,
    store: Store,
    adapters: Sequence[ImportAdapter],
    notify: NotifyHook = _silent_notification,
) -> int:
    """Drive every adapter through the one runner and report what happened.

    This is the command a single systemd timer runs. Everything it prints has
    to be legible in a journal entry after the fact, because that is where the
    operator will read it.

    Per-source failure isolation is the runner's (``_run_one`` contains each
    adapter's exception and keeps the others going); this function's job is to
    turn the resulting report into a summary and an exit code.

    ``since`` is already baked into each adapter at construction — the port is
    a single-verb interface, so acquisition knobs live on the instance.

    ``notify`` is the seam the desktop / systemd layer plugs a real notifier
    into. It fires on every run, including clean ones, because a run that
    imported nothing from a month-old export is clean and is still the thing
    an operator needs told — see ``runner.run_imports``. Defaults to the
    runner's no-op so library and test callers stay silent.
    """
    report = asyncio.run(
        run_imports(
            adapters,
            StoreSink(store),
            notify=notify,
            stale_after_days=(
                args.stale_after_days
                if args.stale_after_days is not None
                else DEFAULT_STALE_AFTER_DAYS
            ),
        )
    )
    _print_run_report(report)
    for warning in report.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    if report.errors:
        for e in report.errors[:10]:
            print(f"  error: {e}", file=sys.stderr)
        # Same 3 as the single-source path, for the same reason: a run that
        # completed but dropped files is not a success, and a timer that reads
        # 0 as success lets the index rot unnoticed.
        return EXIT_COMPLETED_WITH_ERRORS
    return 0


def _print_run_report(report: RunReport) -> None:
    """One line per source, then the run total.

    Counts come from the sink, which probes the store BEFORE writing. The
    single-source path used to print ``added=len(records) updated=0`` on every
    run, so importing 313 new PRs and re-writing the same 313 rows produced
    identical output; that is not repeated here.

    ``warnings`` rides in the total because it is the only field that
    distinguishes a healthy no-op from a run that imported nothing off a
    month-old export. Both print ``added=0 ... errors=0``; only one of them
    also prints ``warnings=1``, and that is what makes the difference legible
    in a journal entry read after the fact.
    """
    print("ingest --all:")
    for name, a in report.adapters.items():
        print(
            f"  {name}: added={a.added} updated={a.updated} "
            f"skipped={a.skipped} errors={len(a.errors)}"
        )
    print(
        f"  total: added={report.added} updated={report.updated} "
        f"skipped={report.skipped} errors={len(report.errors)} "
        f"warnings={len(report.warnings)}"
    )


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
    if isinstance(src, _UnbuildableSource):
        print(f"github-token-status: {src}", file=sys.stderr)
        return 1
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

    ing = sub.add_parser(
        "ingest", help="run one source's ingest cycle, or --all of them"
    )
    ing.add_argument(
        "source",
        nargs="?",
        help=(
            "source name: sessions | github | chatgpt | claude-web | "
            "research | sota-watch | substack | dropbox | ticktick"
        ),
    )
    ing.add_argument(
        "--all",
        dest="all_sources",
        action="store_true",
        help=(
            "run every source through the unified runner instead of one "
            "named source (one timer, one report; per-source failures are "
            "isolated)"
        ),
    )
    ing.add_argument("--since", help="ISO date to bound the ingest window")
    ing.add_argument(
        "--stale-after-days",
        type=int,
        # No argparse default: `None` is how "the operator did not type this"
        # is told apart from "they typed the default", which is what lets the
        # usage check reject it on a run where it cannot do anything.
        # ``_cmd_ingest_all`` applies DEFAULT_STALE_AFTER_DAYS.
        default=None,
        help=(
            "with --all: warn when a manually-refreshed input (chat exports, "
            "the TickTick CSV) is older than this many days "
            f"(default: {DEFAULT_STALE_AFTER_DAYS})"
        ),
    )
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
    _adapters: Sequence[ImportAdapter] | None = None,
    _notify: NotifyHook | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    store = _store or Store()
    store.migrate()

    def sources() -> dict[str, Any]:
        """Build the source registry ON THE COMMAND THAT NEEDS IT.

        Not before dispatch. ``query`` and ``status`` consult no source, and
        ``ingest --all`` drives the ADAPTER registry (which has its own
        construction isolation); building nine real sources for them meant a
        single raising constructor aborted those commands with a bare
        traceback, upstream of every guard downstream of it.
        """
        return _sources if _sources is not None else _default_sources()

    if args.cmd == "query":
        return _cmd_query(args, store)
    if args.cmd == "status":
        return _cmd_status(args, store)
    if args.cmd == "ingest":
        usage_error = _ingest_usage_error(args)
        if usage_error is not None:
            print(usage_error, file=sys.stderr)
            return 2
        if args.all_sources:
            try:
                since = _parse_since(args.since)
            except ValueError:
                print(f"bad --since: {args.since}", file=sys.stderr)
                return 2
            adapters = (
                _adapters
                if _adapters is not None
                else default_adapters(since=since)
            )
            return _cmd_ingest_all(
                args, store, adapters, _notify or _silent_notification
            )
        # Round-2 MEDIUM: the atomic DELETE + upsert lives inside
        # ``_cmd_ingest`` via ``store.rebuild_and_upsert`` when
        # ``args.rebuild`` is set. Do NOT call ``store.rebuild`` here —
        # doing so would commit the DELETE before the transaction and
        # reintroduce the non-atomic gap this fix closes.
        return _cmd_ingest(args, store, sources())
    if args.cmd == "github-token-status":
        return _cmd_github_token_status(args, store, sources())
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
