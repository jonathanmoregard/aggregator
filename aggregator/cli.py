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
  no-op. Defaults to doing nothing INTERACTIVELY; ``--all --notify`` or
  ``$AGGREGATOR_NOTIFY_COMMAND`` installs the real desktop notifier, which is
  what an unattended timer run needs (stderr on an exit-0 run reaches nobody).

Argparse (not click) per plan.
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import logging
import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aggregator.core.chunk import chunk_body
from aggregator.core.embed import MODEL_DOWNLOAD_ENV, Embedder, downloads_allowed
from aggregator.core.provenance import classify
from aggregator.core.store import (
    EMBED_BACKLOG_ORDER,
    EmptyRebuildRefusedError,
    Store,
)

# ONE DIRECTION ONLY: the CLI depends on the eval package, never the reverse.
# ``aggregator.evals`` deliberately imports nothing from here, so the harness
# stays runnable from a test, a REPL or a future service without dragging nine
# source constructors and an argparse tree along with it. Pinned in
# ``tests/test_cli_retrieval_regression.py``.
from aggregator.evals.harness import retrieval_regression_command
from aggregator.evals.search import SEARCH_MODES
from aggregator.imports.ingest_state import (
    POISON_MAX_ATTEMPTS,
    SOURCE_CURSORS,
    PoisonLedger,
    Watermarks,
    default_marker_path,
    stale_input_markers,
)
from aggregator.imports.port import Delivery, ImportAdapter
from aggregator.imports.registry import default_adapters
from aggregator.imports.runner import (
    NotifyHook,
    RunReport,
    commit_fault_receipts,
    commit_report_barriers,
    commit_staleness_receipts,
    graceful_shutdown,
    run_imports,
)
from aggregator.imports.store_sink import StoreSink, count_writes
from aggregator.imports.sync_bridge import accepts_errors_kwarg, unwired_sink_note

# ``_get_reranker`` is the MCP server's LAZY SINGLETON, borrowed rather than
# re-implemented. The CLI needs the cross-encoder built BEFORE the query runs,
# so that a load failure is a refusal instead of a silently unranked page —
# and the query itself asks for the same object a moment later. Constructing a
# second one here would cost ~2 GB RSS and a second model load to answer one
# question.
from aggregator.mcp import (
    _RERANK_WINDOW,
)
from aggregator.mcp import (
    _get_reranker as _mcp_get_reranker,
)
from aggregator.mcp import (
    aggregator_capabilities as _mcp_capabilities,
)
from aggregator.mcp import (
    aggregator_query as _mcp_query,
)
from aggregator.sources import ticktick_api
from aggregator.sources.base import (
    ObservationRow,
    ReadsManualExport,
    SessionRow,
    SupportsRebuild,
)
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

# How many of a run's errors the single-source path prints. Named because it is
# no longer only a display choice: what is elided was not reported, so a run that
# truncated cannot claim its report reached anybody — see ``_stderr_delivery``.
ERROR_PRINT_LIMIT = 5

# How many error lines fit in a desktop toast before it stops being readable.
# Same rule as above and the same round-7 consequence: whatever this elides was
# not delivered, so nothing it gates goes quiet. ``_notification_text`` spends
# the budget on the lines that gate a receipt first — see there.
NOTIFY_ERROR_LIMIT = 5

# How many of a run's errors `ingest --all` prints to stderr. Larger than the
# toast because a terminal scrolls and a journal is read after the fact; a
# person watching that terminal is a delivery channel for exactly these lines.
RUN_ERROR_PRINT_LIMIT = 10

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

# How a run reaches a human when nobody is watching the terminal.
#
# ``--notify`` installs the real notifier; ``$AGGREGATOR_NOTIFY_COMMAND``
# installs it too AND replaces the program, so a systemd unit can wire this up
# through its ``Environment=`` without anyone editing an argv. The value is
# shlex-split, so it may carry arguments ("dunstify --replace 42").
NOTIFY_COMMAND_ENV_VAR = "AGGREGATOR_NOTIFY_COMMAND"
DEFAULT_NOTIFY_COMMAND = "notify-send"
# A hung notification daemon must not wedge the timer's unit.
NOTIFY_TIMEOUT_SECONDS = 10

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

# Per-source ``--rebuild`` refusals whose reason NO PROPERTY CAPTURES.
#
# Not the general rule. The general rule is ``sources.base.ReadsManualExport``
# — a source declares that its input is an archive a human downloads, and
# ``_rebuild_refusal`` derives the refusal from that. This dict was the whole
# mechanism until round 2, which is how ``substack`` (the same Settings →
# Exports zip as chatgpt and claude-web) ended up with --rebuild allowed.
#
# What stays here is the reason that is specific to one source's ontology
# rather than to how its input is acquired.
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


#: The ONE command that is allowed to fetch model weights, spelled exactly as
#: a human would type it. Every message that needs somebody to go and get
#: weights names this, so "what do I run?" is never left as an exercise.
SEED_MODELS_COMMAND = f"{MODEL_DOWNLOAD_ENV}=1 aggregator embed --seed-models"


def _reranker_load_failure(error: BaseException) -> str:
    """What to print when ``--rerank`` cannot have the model it asked for."""
    return (
        f"ERROR: --rerank could not load the cross-encoder "
        f"({type(error).__name__}: {error}).\n"
        f"No results were printed: without the reranker this command returns "
        f"the fused/recency order, which is exactly what --rerank was asked "
        f"to replace, and a page that silently is not ranked is worse than "
        f"no page.\n"
        f"If the weights are simply not on this machine yet, fetch them once "
        f"with `{SEED_MODELS_COMMAND}` (needs network); otherwise re-run "
        f"without --rerank."
    )


#: ``--fields`` was not typed. argparse cannot otherwise tell "the operator
#: left this off" from "the operator typed the default value on purpose", and
#: the two must diverge under ``--rerank``: one gets the bodies it needs, the
#: other gets refused.
_FIELDS_UNSET = None


def _rerank_needs_full_fields() -> str:
    """What to print when ``--rerank`` and ``--fields summary`` are both typed."""
    return (
        "ERROR: --rerank and --fields summary ask for incompatible things.\n"
        "--rerank ranks document BODIES with a cross-encoder, and "
        "--fields summary does not return any — the ranking would be computed "
        "over empty documents, at several minutes per query, for an ordering "
        "over nothing.\n"
        "Re-run with --fields full, or omit --fields entirely (--rerank "
        "supplies --fields full for you), or drop --rerank to keep the "
        "default recency ordering over summaries.\n"
        "Nothing was spent: no model was loaded and no query was run."
    )


def _cmd_query(args: argparse.Namespace, store: Store) -> int:
    # WHAT ``--rerank`` ON ITS OWN MEANS. The cross-encoder scores an item's
    # ``content``, and summary items have none — so under the ``--fields``
    # default, ``aggregator query "..." --rerank`` refused itself. That is the
    # batch surface's most obvious invocation, and the one the MCP tool
    # description sends callers here for. So the flag now supplies the mode it
    # needs.
    #
    # NOT the auto-upgrade the MCP layer deliberately refuses. There, ``fields``
    # is a function parameter with a real default and no unset state, so
    # promoting it would change the payload shape behind a caller that had said
    # nothing — invisible. Here the promotion is attached to a flag the operator
    # typed, is documented on that flag in ``--help``, and applies ONLY when
    # ``--fields`` was left off. Typed explicitly, ``--fields summary`` still
    # reaches the refusal below.
    fields = args.fields
    if fields is _FIELDS_UNSET:
        fields = "full" if args.rerank else "summary"
    elif args.rerank and fields != "full":
        # BEFORE THE MODEL LOAD, because this outcome is knowable from argv.
        # The MCP layer refuses the same pair, but only after the CLI has paid
        # ~2 GB RSS and a cross-encoder load to reach a query that was already
        # doomed — spending the thing the refusal exists to save.
        print(_rerank_needs_full_fields(), file=sys.stderr)
        return 1
    if args.rerank:
        # BEFORE THE QUERY, and loudly. ``_maybe_rerank`` catches a rerank
        # failure and returns the page in its fused order — right for the MCP
        # tool, where a lost ordering must not cost the caller their answer,
        # and wrong here, where the ordering IS what was asked for.
        #
        # That degrade is no longer silent: it now sets ``rerank_applied:
        # False`` and leads the response with a ``notice`` naming the exception
        # and pointing at ``aggregator embed --seed-models``, which this
        # command prints. What the up-front load still buys is WHEN and WITH
        # WHAT EXIT CODE. Measured on this path with a reranker that loads and
        # then raises while scoring: exit 0, the whole page printed, and the
        # notice on the line after the last row — an operator who waited four
        # and a half minutes
        # reads the report only after scrolling past the results it disclaims,
        # and a script or timer sees a success. Loading first converts that
        # post-work degradation into a pre-work refusal that names its fix and
        # exits 1, and the object lands in the singleton the query then reuses.
        #
        # It narrows the window rather than closing it — a cross-encoder can
        # still fail after loading — so the reporting below is the backstop,
        # not dead weight.
        try:
            _mcp_get_reranker()
        except Exception as e:  # noqa: BLE001 - reported, not handled
            print(_reranker_load_failure(e), file=sys.stderr)
            return 1
        # "SCORES EVERY HIT" WAS FALSE, and round 3's M4. ``_maybe_rerank``
        # reorders at most ``_RERANK_WINDOW`` items; --page-size defaults to
        # 50. So the flag's most ordinary invocation returned a 40%-ranked page
        # while the one line the operator reads during the multi-minute wait
        # told them the whole thing had been scored.
        #
        # AND THE WAIT IT NAMED WAS AN ORDER OF MAGNITUDE SHORT. "47 s median"
        # was measured while the cross-encoder scored session cards against
        # their own subjects — near-empty documents — and while the pages
        # holding a genuinely long document were being OOM-killed instead of
        # timed. Both are fixed (see ``mcp._RERANK_WINDOW`` for the table), and
        # the re-measured figure on real bodies is 273 s median / 304 s p95.
        # The unit is in the sentence on purpose: three digits of seconds reads
        # as small, and "four and a half minutes" is what an operator actually
        # decides against.
        print(
            f"note: --rerank scores the first {_RERANK_WINDOW} hits of the "
            f"page with a cross-encoder and reorders those; any hit after them "
            f"keeps the default recency order. Measured at 273 s median and "
            f"304 s at p95 per query on this CPU — about four and a half "
            f"minutes — plus a one-off model load. This is a batch facility. "
            f"Pass --page-size {_RERANK_WINDOW} or less for a page that is "
            f"ranked all the way down.",
            file=sys.stderr,
        )
    result = _mcp_query(
        dsl=args.dsl,
        fields=fields,
        page_size=args.page_size,
        # THREADED, because this command PRINTS one. Without it the token at
        # the bottom of every long result set addressed a page the CLI then
        # refused as an unrecognised argument, so page 2 was unreachable from
        # here and the only evidence of that was an argparse usage error.
        page_token=args.page_token,
        drilldown=args.drilldown,
        rerank=args.rerank,
        _store=store,
        # THE ZERO-RESULT LOG IS WRITTEN FROM HERE, AND ONLY FROM HERE.
        # ``aggregator_query`` defaults it off so the MCP surface — annotated
        # readOnlyHint=True, and advertised to the client as writing nothing —
        # keeps that promise. This command is a human at a terminal or at
        # Raycast, in a process that already writes; a question a person asked
        # and got nothing for is exactly what the golden set wants.
        _log_misses=True,
    )
    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("ok") else 1
    if not result.get("ok"):
        print(f"error: {result.get('reason')}", file=sys.stderr)
        print(f"remediation: {result.get('remediation')}", file=sys.stderr)
        return 1
    mode = result.get("mode", "records")
    # WHERE THE RANKED PREFIX ENDS. Below this line the rows were never scored
    # — they are in the ordinary recency order — and they are otherwise
    # indistinguishable from the ranked ones, so a reader scrolling a 50-row
    # page reads 30 rows of recency ordering as relevance ordering. Printed
    # only when there is genuinely a seam: the rerank applied AND the page is
    # longer than the window.
    ranked_upto = (
        _RERANK_WINDOW
        if result.get("rerank_applied") and len(result["records"]) > _RERANK_WINDOW
        else None
    )
    for i, rec in enumerate(result["records"]):
        if ranked_upto is not None and i == ranked_upto:
            print(
                f"# ---- end of the {ranked_upto} hits ranked by relevance; "
                f"the {len(result['records']) - ranked_upto} below are in the "
                f"default recency order and were NOT scored ----"
            )
            print()
        if mode == "sessions":
            print(
                f"# {rec['source']} :: {rec['subject']}  "
                f"({rec['stable_id']}, matches={rec.get('matching_observations', 0)})"
            )
        elif mode == "observations":
            # ``by=`` sits next to ``type`` because the two are constantly
            # confused and only one of them is about authorship: ``type=user``
            # says the line came in on the user channel, ``by=hook`` says a
            # program wrote it. ``unclassified`` rather than a blank, so an
            # un-backfilled cache reads as "nobody has looked" instead of as a
            # missing field.
            print(
                f"# obs {rec['type']} by={rec.get('provenance') or 'unclassified'} "
                f"@{rec['ts']}  ({rec['obs_id']}, session={rec['session_id']})"
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
    caps = dict(_mcp_capabilities(_store=store))
    # WHERE A SUPPRESSED ALARM STAYS VISIBLE. The TickTick poll reports a
    # vanished project once and then stops (a permanently-red signal is one an
    # operator learns to ignore, which costs the next real failure its
    # audience), and this is the other half of that bargain: the tasks it is
    # still holding, and cannot ever resolve, are listed here on demand. Quiet
    # is only acceptable because it is not the same as forgotten.
    #
    # Read here rather than added to ``aggregator_capabilities``: that surface
    # is the MCP tool's read-only view of the CACHE, and this is on-disk source
    # state that no MCP client asked for.
    uncovered = ticktick_api.uncovered_projects(ticktick_api.default_state_path())
    # THE SAME BARGAIN, one source of noise over. A hand-downloaded export that
    # has gone stale is warned about once per episode and then goes quiet — a
    # toast every 30 minutes until somebody visits a vendor's export page is how
    # an operator learns to dismiss toasts unread — so the sources currently
    # being held quiet, and the export date each was held quiet about, are
    # listed here on demand.
    stale_inputs = stale_input_markers()
    # WHERE "IS THIS RUN INCREMENTAL?" IS ANSWERED WITHOUT READING CODE. The
    # high-water marks and the records that could not be written are both
    # things a run may legitimately go quiet about — a mark that stopped moving
    # and a record set aside three weeks ago produce no output at all on a
    # healthy tick — so the on-demand view is the other half of that bargain.
    # Quiet is only acceptable while it is not the same as forgotten.
    ingest_state = store.all_ingest_state()
    held_records = store.poison_summary()
    # THE SAME BARGAIN AGAIN, for the noisiest case of all. Input that will
    # never parse is reported loudly the first time its identity is seen and
    # never again — which on 2026-08-16 was the difference between four
    # CRITICAL toasts in six hours and one. The price of that quiet is this
    # listing: every fault, the file it is in, how many records it costs the
    # index, and the date a human was first told. Nothing here is a count
    # summary — a quarantine you cannot name is one you cannot fix.
    known_faults = store.fault_summary()
    # WHICH SOURCES THE VECTOR ARM CAN ACTUALLY ANSWER FOR. The backfill is a
    # measured 25-30 days on this hardware and runs in the user's chosen order
    # (dropbox, substack, claude-web/chatgpt, sessions, subagents, then the
    # rest), so for most of its life this cache is PARTIALLY embedded — and a
    # source nobody has reached yet returns exactly what a finished source with
    # nothing on the topic returns: no vector hits. This is where those two are
    # told apart, per source, by name.
    #
    # Read here rather than added to ``aggregator_capabilities``: that surface
    # is on the MCP connect path, and a full table scan — 0.27 s warm, 4.3 s
    # cold on the live corpus — belongs on a command a human typed.
    embedding_progress = store.embed_progress_by_source()
    if args.json:
        caps["ticktick_uncovered_projects"] = uncovered
        caps["stale_input_markers"] = stale_inputs
        caps["ingest_state"] = ingest_state
        caps["held_records"] = held_records
        caps["known_faults"] = known_faults
        caps["embedding_progress"] = embedding_progress
        print(json.dumps(caps, indent=2, default=str))
        return 0
    print(f"cache_path: {caps['cache_path']}")
    print(f"schema_version: {caps['schema_version']}")
    print(f"tool_tier: {caps['tool_tier']}")
    print("sources:")
    for s in caps["sources"]:
        fresh = caps["freshness"].get(s, "n/a")
        print(f"  {s}: last_updated={fresh}")
    print("embedding progress by source (highest priority first):")
    for row in embedding_progress:
        print(
            f"  {row['source']}: {row['state']} — {row['embedded']}/{row['total']} "
            f"embedded, {row['pending']} pending, {row['skipped']} nothing to "
            f"embed, {row['errors']} held ({row['kind']})"
        )
    print("ingest windows (per-source high-water marks):")
    for name in sorted(SOURCE_CURSORS):
        cursor = SOURCE_CURSORS[name]
        state = ingest_state.get(name, {})
        mark = state.get("cursor_value")
        if not cursor.is_incremental:
            # SAID OUT LOUD, EVERY TIME IT IS ASKED. A source that full-scans
            # while reporting like an incremental one is the failure this whole
            # change exists to remove, so its line names the kind rather than
            # showing an empty mark that reads as "nothing new yet".
            print(f"  {name}: FULL SCAN every run — {cursor.note}")
            continue
        print(
            f"  {name}: {cursor.kind} on {cursor.field}, "
            f"mark={mark or 'none yet (next run is a full scan)'}, "
            f"last_run={state.get('last_run_at') or 'never'}, "
            f"failures={state.get('consecutive_failures') or 0}"
        )
        if state.get("last_error"):
            print(f"    last error: {state['last_error']}")
    if held_records:
        print("records set aside after write failures (retained, never deleted):")
        for entry in held_records:
            when = "terminal, never retried" if entry["terminal"] else "will retry"
            print(
                f"  {entry['source']}: {entry['count']} x "
                f"{entry['error_type']} ({when})"
            )
    if known_faults:
        total = sum(int(f["count"]) for f in known_faults)
        print(
            f"permanently-bad input (reported once, then held quiet): "
            f"{len(known_faults)} fault(s), {total} record(s) NOT in the index"
        )
        for fault in known_faults:
            print(
                f"  {fault['source']}: {fault['count']} record(s) — "
                f"{fault['reason']} in {fault['scope']}"
            )
            print(
                f"    line(s) {fault['detail']}; first reported "
                f"{fault['first_seen_at']}, last seen {fault['last_seen_at']}"
            )
    if uncovered:
        print(
            "ticktick uncovered projects (reported once; tasks retained, never "
            "inferred completed):"
        )
        for project_id, info in uncovered.items():
            task_ids = info["task_ids"]
            count = len(task_ids) if isinstance(task_ids, list) else 0
            print(
                f"  {project_id or '<no projectId in the baseline entry>'}: "
                f"{count} task(s), first reported {info['first_reported'] or 'unknown'}"
            )
        print(f"  baseline: {ticktick_api.default_state_path()}")
    if stale_inputs:
        print(
            "stale inputs (reported once; warning stays quiet until a fresh "
            "export lands or the threshold is lowered):"
        )
        for name, mark in sorted(stale_inputs.items()):
            newest = mark.get("input_newest_at") if isinstance(mark, dict) else None
            threshold = (
                mark.get("stale_after_days") if isinstance(mark, dict) else None
            )
            first = mark.get("first_reported") if isinstance(mark, dict) else None
            print(
                f"  {name}: input {newest or 'MISSING ENTIRELY'}"
                f"{f', threshold {threshold} days' if threshold is not None else ''}"
                f", first reported {first or 'unknown'}"
            )
        print(f"  markers: {default_marker_path()}")
    return 0


def _cmd_retrieval_regression(args: argparse.Namespace, store: Store) -> int:
    """Freeze a retrieval baseline, or re-run it and report drift.

    CRITERION A'S SURFACE. The harness landed before any retrieval change on
    this branch, on purpose — it is what makes "retrieval got better" a
    falsifiable claim rather than an assertion — and until now it had no
    caller. An entry point reachable only from a Python REPL is one nobody runs
    before a change and nobody runs after it, and the freeze/run ORDER is where
    all of its value lives: freeze first, change, then run. A baseline frozen
    afterwards has baselined the bug.

    A THIN TRANSLATION AND NOTHING ELSE. Every decision — which exit code means
    what, when a drift number is allowed to fail a run, how the report reads —
    belongs to the harness and is documented there. Duplicating any of it here
    would give the two surfaces room to disagree.

    ``db_path`` IS THREADED FROM THE STORE THIS PROCESS ALREADY OPENED, not
    left to the harness's default. ``--cache`` and ``AGGREGATOR_DB`` (and a
    test's ``_store=``) all move the file the rest of the CLI is talking to,
    and an eval that silently measured a different cache would be worse than no
    eval — the whole package exists to stop exactly that class of claim. The
    harness opens its own READ-ONLY handle on it; nothing here writes.
    """
    return retrieval_regression_command(
        args.action,
        mode=args.mode,
        drift_threshold=args.drift_threshold,
        db_path=store.db_path,
    )


def _commit_after_write(src: Any, errors: list[str]) -> None:
    """Let a source advance state that its records had to land first.

    The single-source half of ``imports/port.SupportsWriteBarrier``; the
    runner does the same after its final flush. Reached ONLY once the write
    above returned — every failing path returns before here, which is the
    whole guarantee. TickTick's open-task baseline is what needs it: advancing
    it is what makes a completion unrepeatable, and it used to happen during
    iteration, before a single row was written.

    A failure is recorded, not raised. The records are already in the store,
    so the ingest itself succeeded; but a baseline that never advances loses
    every completion from here on, so it becomes an ``errors`` entry and the
    run exits 3.
    """
    commit = getattr(src, "commit_after_write", None)
    if commit is None:
        return
    try:
        commit()
    except Exception as e:  # noqa: BLE001 -- reported, never fatal to the write
        errors.append(f"{type(e).__name__}: {e}")


def _print_errors(errors: Sequence[str], limit: int) -> list[str]:
    """Put this run's errors on stderr and return EXACTLY what went there.

    The return value is what a delivery declaration is derived from, so no
    caller can print one thing and claim another: shrink the print, or the
    limit, or reorder it, and the claim shrinks with it — because the claim is
    read out of the printed text rather than asserted next to it. See
    :func:`_stderr_delivery` and ``imports/port.Delivery``.
    """
    shown = list(errors[:limit])
    for e in shown:
        print(f"  error: {e}", file=sys.stderr)
    return shown


def _print_warnings(warnings: Sequence[str]) -> list[str]:
    """Put this run's staleness warnings on stderr and return what went there.

    Same contract as :func:`_print_errors`, and now for the same reason rather
    than for symmetry: a staleness warning goes quiet once a human has been
    told, so what this printed is what a watched terminal may claim to have
    delivered. Unlimited, unlike the errors — there are at most four sources
    that can go stale, and truncating the list would silently cost the elided
    one its only channel on an interactive run.

    The ``WARNING: `` prefix is display only and is not part of the line, the
    same way ``  error: `` is not: the return value is the report's own text,
    which is what ``Delivery`` matches whole lines against.
    """
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    return list(warnings)


def _stderr_delivery(printed: Sequence[str], reported: Sequence[str]) -> Delivery:
    """Which of ``reported`` a person actually SAW on stderr. Possibly none.

    ONLY AT AN INTERACTIVE TERMINAL. ``aggregator ingest <source>`` installs no
    notifier at all (``--notify`` is refused without ``--all``), so the print is
    the entire channel — and a channel nobody is standing in front of is not a
    channel. Under the systemd timer that runs this repo's ingests, stderr is the
    journal: it is written, it is retained, and no human reads it unprompted.
    Treating that as delivery is the round-4/5/6 defect one surface over, and it
    is worse here than in the runner, because this path has no fallback notifier
    to be misconfigured — it is silent by design.

    ``isatty`` is the question actually being asked ("is somebody there?") and
    the one the shell answers correctly for every redirection: ``2>log``, a pipe,
    a systemd unit and a cron job all say no; a person at a prompt says yes.

    TRUNCATION IS THE SECOND WAY THE PRINT MISSES, and it is no longer a
    separate rule. The caller shows the first :data:`ERROR_PRINT_LIMIT` errors
    and hands that list in as ``printed``; a line that was pushed off the end is
    not in the text, so it is not in the result, so nothing it gates goes quiet.
    Round 6 spelled this as ``if len(errors) > ERROR_PRINT_LIMIT``, which was a
    check the next surface (the desktop toast, round 7) did not have.

    Never raises: a closed or replaced stream answers "nothing delivered", which
    is the direction that keeps reporting rather than the one that goes quiet.
    """
    try:
        watched = sys.stderr.isatty()
    except (AttributeError, ValueError):  # pragma: no cover - closed/odd stream
        return Delivery()
    if not watched:
        return Delivery()
    return Delivery.accepted("\n".join(printed), reported)


def _commit_after_report(
    src: Any, delivery: Delivery, reported: Sequence[str]
) -> str | None:
    """Let a source record that its report reached a human. Or why not.

    The single-source half of ``imports/port.SupportsReportBarrier``; the runner
    does the same, per adapter, in ``runner.commit_report_barriers``. Reached
    only once the summary and every error line are on stderr — and only if
    ``delivery`` covers EVERY line this run reported. One source runs on this
    path, so its report is the whole report, and a run that could only show some
    of it cannot tell which of its own sentences it is safe to stop saying.

    ``delivery`` is a PARAMETER rather than something computed in here, so the
    call site has to name the channel it is claiming. There is exactly one such
    channel on this path today; a future one (a mailer, a webhook) is then a new
    value passed in, not an assumption already baked into this function.

    ASKS THE SOURCE BY ATTRIBUTE, where the runner asks the adapter for an
    explicit ``gates_report`` (``port.is_report_gating``). Not an inconsistency:
    the runner's problem was that ``SyncSourceAdapter`` hands every source a
    forwarding ``commit_after_report``, so presence there meant nothing. On a
    SOURCE the method is hand-written — no source in this repo inherits from
    anything — and this path drives exactly one source, so there is no second
    adapter whose lines could be confused with its own or starve it out of a
    payload. Presence is a declaration here; it was an artefact there.

    Returns the fault instead of appending to ``errors``, because ``errors`` has
    already been printed by the time this runs; the caller prints what comes
    back and takes exit 3 for it. Loud, but the loss is small: all a failed
    receipt costs is one more report of a disappearance already reported.
    """
    if not delivery.covers(reported):
        return None
    commit = getattr(src, "commit_after_report", None)
    if commit is None:
        return None
    try:
        commit()
    except Exception as e:  # noqa: BLE001 -- reported, never fatal to the write
        return f"{type(e).__name__}: {e}"
    return None


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
    so the two surfaces cannot drift on which sources get an errors sink — and
    the same note when the probe says no, so they cannot drift on whether that
    degradation is reported either.

    THE FALLBACK IS NOT FREE, and it used to be silent. Driving a source
    without ``errors`` means every per-item failure it takes internally has
    nowhere to land, so the run prints ``errors=0`` and exits 0 for a source
    that may have skipped half its input. The source still runs — that is the
    whole point of the fallback — but the run says the count cannot be
    trusted for it.
    """
    if accepts_errors_kwarg(iter_fn):
        return iter_fn(since, errors=errors)
    errors.append(unwired_sink_note(getattr(iter_fn, "__qualname__", str(iter_fn))))
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
    _commit_after_write(src, errors)

    print(
        f"ingest {args.source}: sessions={session_count} "
        f"observations={obs_count} errors={len(errors)}"
    )
    shown = _print_errors(errors, ERROR_PRINT_LIMIT)
    # Same order and same reason as the records path: stderr is the channel, so
    # the receipt is only earned once the lines above are on it AND somebody was
    # there to read them — all of them, not the first five.
    late = _commit_after_report(src, _stderr_delivery(shown, errors), errors)
    if late is not None:
        print(f"  error: {late}", file=sys.stderr)
    if errors or late is not None:
        return EXIT_COMPLETED_WITH_ERRORS
    return 0


def _rebuild_refusal(name: str, src: Any) -> str | None:
    """Why ``--rebuild`` is refused for this source, or None if it is allowed.

    REFUSAL IS THE DEFAULT. A source is allowed the destructive path only if it
    declares ``sources.base.SupportsRebuild``; everything below that is a more
    specific, better-worded refusal for a case we can name. Round 3: the checks
    used to be the whole rule, so they only ever caught evidence AGAINST a
    rebuild and a source that declared nothing at all — the normal state of a
    source whose author never read this function — got the DELETE. Measured on
    a fresh record-shaped source: 150 stored, 140 re-scanned, 10 last-copy rows
    deleted, ``added=0 updated=140 skipped=0 errors=0``, exit 0. Forgetting a
    declaration now costs a refusal an operator can read, which is recoverable;
    the old default was not.

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
    reason = REBUILD_UNSUPPORTED_SOURCES.get(name)
    if reason is not None:
        return (
            f"ERROR: --rebuild is not supported for source {name!r}: "
            f"{reason} Re-run without --rebuild (ingest is an idempotent "
            f"upsert per stable_id, so a re-scan already overwrites every row "
            f"it can produce)."
        )
    if isinstance(src, ReadsManualExport):
        # THE RULE, and it is a property, not a list. Round 2: substack reads
        # a Settings → Exports zip exactly like chatgpt and claude-web, and its
        # --rebuild was allowed anyway — the refusal was decided by the
        # hand-kept dict above plus the accident that the other two are
        # entity-shaped. An old or partial archive then deleted last-copy rows
        # under the ratio guard's slack, at exit 0 (measured: 150 stored, 140
        # re-scanned, 10 gone, `added=0 updated=140 skipped=0 errors=0`).
        return (
            f"ERROR: --rebuild is not supported for source {name!r}: its "
            f"input is {src.manual_export_input()}. --rebuild DELETEs every "
            f"row the re-scan did not reproduce, so an older or partial "
            f"archive silently destroys the rest. Re-run without --rebuild "
            f"(ingest is an idempotent upsert, so a re-scan already overwrites "
            f"every row it can produce)."
        )
    if hasattr(src, "iter_entities") and name != "sessions":
        # A second, shape-derived reason that outlives the property: the entity
        # rebuild path replaces the sessions + observations rows wholesale for
        # the origins it is scoped to, and only the sessions source can
        # regenerate its origin. Kept for a future entity-shaped source whose
        # input is NOT a manual export.
        return (
            f"ERROR: --rebuild is not supported for source {name!r}: the "
            f"entity rebuild path replaces the sessions/observations rows "
            f"wholesale and only the sessions source can regenerate the origin "
            f"it is scoped to. Re-run without --rebuild (ingest is an "
            f"idempotent upsert per session/observation id)."
        )
    if not isinstance(src, SupportsRebuild):
        # THE DEFAULT, and it is deliberately the last word rather than the
        # first: the checks above produce a better message for a case we can
        # name, and this catches everything else — including the case that
        # matters most, a source nobody has thought about yet.
        return (
            f"ERROR: --rebuild is not supported for source {name!r}: it does "
            f"not declare that a re-scan reproduces everything the DELETE "
            f"would remove. --rebuild DELETEs every row the re-scan did not "
            f"produce, so a source whose stored rows outlive its current input "
            f"silently destroys the difference. Re-run without --rebuild "
            f"(ingest is an idempotent upsert, so a re-scan already overwrites "
            f"every row it can produce). If this source really can regenerate "
            f"its whole population, give it a rebuild_input() saying what keeps "
            f"its input current (sources.base.SupportsRebuild)."
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
        _commit_after_write(src, errors)
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
    shown = _print_errors(errors, ERROR_PRINT_LIMIT)
    # AFTER the errors are on stderr, which is this path's entire delivery
    # channel — there is no notify hook here — and only when that stderr is a
    # terminal somebody is watching, and only for the lines it actually showed.
    # See ``_stderr_delivery``.
    late = _commit_after_report(src, _stderr_delivery(shown, errors), errors)
    if late is not None:
        print(f"  error: {late}", file=sys.stderr)
    if errors or late is not None:
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
    if args.notify and not args.all_sources:
        return (
            "ingest: --notify only applies to --all (the run report the "
            "notifier describes is the runner's); drop it or run "
            "`aggregator ingest --all --notify`"
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
    """The CLI's default notifier: do nothing. Never counts as delivery.

    An interactive ``aggregator ingest --all`` prints its report to a terminal
    somebody is already looking at, so it should not also pop a desktop toast.
    An UNATTENDED run is the opposite case — see ``_desktop_notification`` and
    ``--notify``.

    ``-> None`` for the same reason as ``runner._no_notification``: a hook that
    does nothing must not be able to declare that a human was told, and the way
    to guarantee that is to leave it nothing to return. See ``port.Delivery``.

    THAT DOES NOT MAKE THE INTERACTIVE RUN UNDELIVERABLE, which was the round-7
    MEDIUM. This hook is silent because the CHANNEL IS ELSEWHERE: ``ingest
    --all`` prints the report itself, after ``run_imports`` has returned, and
    declares what that print showed a watched terminal. See ``_cmd_ingest_all``.
    """


def _notification_text(report: RunReport) -> tuple[str, str, str] | None:
    """``(urgency, summary, body)`` for this run, or None to stay quiet.

    The runner fires ``notify`` on every run and leaves it to the hook to
    decide what is worth telling a human. Two things are:

    * errors — CRITICAL, per the 2026-08-08 fail-loudly constraint.
    * warnings with no errors — a hand-refreshed export has gone stale or is
      missing. Nothing failed, so not critical, but this is precisely the run
      that is invisible everywhere else: exit 0, every count 0, identical to a
      healthy no-op.

    A wholly clean run says nothing. A toast on every timer tick is how an
    operator learns to dismiss them without reading, which would cost the two
    cases above the only channel they have.

    THE BUDGET IS SPENT ON THE GATING LINES FIRST. A toast has to stay readable,
    so only :data:`NOTIFY_ERROR_LIMIT` error lines go in it — and round 7 was
    five failures from other sources pushing TickTick's uncovered-project line,
    the one line whose delivery decides whether that gap re-alarms every 30
    minutes, off the end. ``Delivery`` now makes eliding it merely loud instead
    of silent (the receipt cannot be stamped for a line that is not in this
    text), and ordering by what gates a receipt is what keeps "merely loud" from
    meaning "loud forever": an adapter without a report barrier repeats its
    errors next run regardless, so its line is the cheaper one to drop.
    """
    if report.errors:
        gating = set(report.gating_errors)
        ordered = [
            *(e for e in report.errors if e in gating),
            *(e for e in report.errors if e not in gating),
        ]
        return (
            "critical",
            f"aggregator ingest: {len(report.errors)} error(s)",
            "\n".join([*ordered[:NOTIFY_ERROR_LIMIT], *report.warnings[:3]]),
        )
    if report.warnings:
        return (
            "normal",
            f"aggregator ingest: {len(report.warnings)} warning(s)",
            "\n".join(report.warnings[:5]),
        )
    return None


def _desktop_notification(report: RunReport) -> Delivery:
    """Tell a human, via ``notify-send`` (or ``$AGGREGATOR_NOTIFY_COMMAND``).

    DECLARES WHAT IT SENT, in exactly one place: after the notification program
    has exited zero, out of the very text that was handed to it. Everywhere else
    — nothing worth saying, an unresolvable program (which raises), a non-zero
    exit (which raises) — nothing was delivered, so an adapter holding a report
    barrier keeps reporting. See ``port.Delivery``.

    ROUND 7 WAS THE GAP BETWEEN "SENT" AND "SENT WHAT". This returned a run-wide
    ``Delivery.DELIVERED`` while the body was ``report.errors[:5]``, so five
    failures from other sources bought permanent silence for a line that never
    left the process. The declaration is now read out of ``body``, which means
    the truncation above cannot outrun it: they are the same string.

    THE CLI IS WHERE THIS BELONGS. ``imports/runner.py`` refuses to shell out
    and names this layer as the one that injects a real notifier — but until
    round 2 nothing could: ``_notify`` is a Python-only seam, the console entry
    point is ``aggregator.cli:main``, and no argv or env reached it. Every real
    invocation therefore got ``_silent_notification``, so the round-1 "notify
    fires on every run" fix moved the silence instead of removing it: on a
    timer the warnings were stderr text on an exit-0 run, and nothing reads
    that.

    Still an injected callable — ``run_imports`` takes it as a parameter and
    the default stays a no-op, so library and test callers shell out to
    nothing. Only ``main`` installs this one, and only when asked.

    A failure to notify propagates: the runner records it in ``run_errors``,
    which turns into exit 3. A notifier that cannot notify is a fault in its
    own right, and on an otherwise-clean run it is the only thing that says so.
    """
    # BEFORE the "nothing worth saying" return, deliberately. A misconfigured
    # notifier that is only checked when there is something to send stays
    # latent until the first FAILING run — and that run's notify failure then
    # reaches only the journal, which is the exact channel the notifier exists
    # to replace. Checked on every run, the config fault surfaces on the next
    # clean one, while the operator still has a working channel to hear it on.
    argv = _notify_argv()
    text = _notification_text(report)
    if text is None:
        # Nothing was sent, so nothing was delivered. Saying otherwise would be
        # the round-6 defect in miniature — "no error occurred" read as "a human
        # heard it". Costs nothing in practice: a report barrier's receipt only
        # ever exists alongside the error line that earned it, and an error is
        # exactly what ``_notification_text`` refuses to stay quiet about.
        return Delivery()
    urgency, summary, body = text
    # No shell=True: the value is operator configuration, split with shlex and
    # exec'd directly. Timeout because a hung notification daemon must not
    # wedge the timer's unit forever.
    #
    # ``--`` because summary and body are POSITIONAL and their content is not
    # ours: a body line beginning with ``-`` is an option to notify-send's and
    # dunstify's GOption parsers, and the notification is then lost to
    # "option -x not recognized" rather than delivered. Safe today only by the
    # accident that every line happens to be prefixed "<adapter name>: ", and
    # adapter names are not validated against a leading dash.
    subprocess.run(
        [*argv, "-u", urgency, "-a", "aggregator", "--", summary, body],
        check=True,
        timeout=NOTIFY_TIMEOUT_SECONDS,
    )
    # ``check=True``, so reaching this line means the notification daemon
    # accepted it — and ``body`` is verbatim what it accepted, so the lines this
    # run may now go quiet about are read back out of it rather than asserted.
    #
    # ``report.reported``, not ``report.errors``: a staleness warning is now
    # suppressed once a human has been told, by the same machinery, and a line
    # that is not offered here can never be in the delivered set — so its marker
    # could never be earned and it would toast every 30 minutes forever.
    return Delivery.accepted(body, report.reported)


def _notify_argv() -> list[str]:
    """The notify program and its arguments, or raise saying what is wrong.

    Two config faults, both of which used to be quiet in the wrong direction.

    A SET-BUT-BLANK value. ``AGGREGATOR_NOTIFY_COMMAND=`` is the shape a
    systemd unit produces from ``Environment=AGGREGATOR_NOTIFY_COMMAND=``, and
    it was falsy at both gates: it did not install the notifier and it fell
    back to the default program. So an operator who had written the line
    believed notifications were on and they were off, while a whitespace-only
    value — the same intent, one keystroke different — raised loudly. Backwards:
    a variable that is present is a statement of intent, and a blank one is a
    broken statement, which is the loud case.

    AN UNRESOLVABLE PROGRAM. ``notify-sned`` is not detectable at any point
    except by trying, and the only run that used to try was a failing one.
    Checked here, on every run.

    Raising is the reporting channel: ``run_imports`` records a notify-hook
    failure in ``run_errors``, so the run exits 3 and the summary says which
    variable to fix.
    """
    raw = os.environ.get(NOTIFY_COMMAND_ENV_VAR)
    argv = shlex.split(DEFAULT_NOTIFY_COMMAND if raw is None else raw)
    if not argv:
        raise ValueError(
            f"${NOTIFY_COMMAND_ENV_VAR} is set but blank ({raw!r}), so no "
            f"notifier could be installed; unset it to get the default "
            f"({DEFAULT_NOTIFY_COMMAND}) or name a program"
        )
    if shutil.which(argv[0]) is None:
        raise ValueError(
            f"notify command {argv[0]!r} is not executable or not on PATH "
            f"(from ${NOTIFY_COMMAND_ENV_VAR}"
            f"{' — unset, so this is the default' if raw is None else ''}); "
            f"no notification can be delivered until it is fixed"
        )
    return argv


def _resolve_notify(
    args: argparse.Namespace, injected: NotifyHook | None
) -> NotifyHook:
    """Which notifier this invocation gets.

    Injection wins, so a library or test caller is never surprised by an env
    var set on the developer's machine. Otherwise ``--notify`` or a set
    ``$AGGREGATOR_NOTIFY_COMMAND`` installs the real one — the env var alone is
    enough so a unit file can wire this up without changing anyone's argv.

    PRESENCE, not truthiness. ``AGGREGATOR_NOTIFY_COMMAND=`` — what
    ``Environment=AGGREGATOR_NOTIFY_COMMAND=`` in a unit file produces — is a
    statement that the operator wants notifications, spelled wrong. Read as
    falsy it installed nothing and said nothing, which is the one outcome an
    operator who wrote that line cannot detect. Installed, ``_notify_argv``
    refuses it out loud and the run exits 3.
    """
    if injected is not None:
        return injected
    if getattr(args, "notify", False) or NOTIFY_COMMAND_ENV_VAR in os.environ:
        return _desktop_notification
    return _silent_notification


def _configure_ingest_logging() -> None:
    """Send the runner's progress lines somewhere a human can find them.

    NOTHING UNDER ``aggregator/`` CONFIGURED LOGGING, which is why the
    2026-08-15 run was silent: ``logging.lastResort`` prints WARNING and above,
    so every INFO-level progress line this pipeline might have emitted would
    have gone nowhere anyway. pypdf's WARNING-level chatter got through, was
    the only thing that did, and its absence was then read as a hang.

    stderr, so the report on stdout stays parseable and the journal gets both.
    ``basicConfig`` no-ops when the root logger already has handlers, so a
    library embedder's or pytest's configuration wins — which is the correct
    precedence for a CLI that is also an importable package.
    """
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _cmd_ingest_all(
    args: argparse.Namespace,
    store: Store,
    adapters: Sequence[ImportAdapter],
    notify: NotifyHook = _silent_notification,
    watermarks: Watermarks | None = None,
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

    THE PRINT BELOW IS ALSO A CHANNEL, and this function is the only place that
    can say so: the runner's hook has already fired by the time these lines
    reach stderr. Interactively that stderr is a terminal with a person in front
    of it, which is exactly as much of an audience as a toast; under the timer
    it is the journal, which is none. Both answers come out of the same
    ``_stderr_delivery`` the single-source path uses.
    """
    _configure_ingest_logging()
    # ONE LEDGER FOR THE WHOLE COMMAND, for the same reason there is one
    # ``Watermarks``: the runner reconciles this run's permanent faults against
    # it, and the receipt commit below writes into it after the print. Two
    # instances could not disagree today, but a run that decided a fault was
    # known against one and recorded it into another is a bug nobody would find.
    ledger = PoisonLedger(store) if watermarks is not None else None

    async def _drive() -> RunReport:
        # THE SIGNAL HANDLER LIVES HERE, not in the runner: the process owns
        # its signals and library code merely reads the flag. Installed inside
        # the coroutine because ``add_signal_handler`` needs the running loop.
        with graceful_shutdown() as stop:
            return await run_imports(
                adapters,
                StoreSink(store),
                notify=notify,
                stale_after_days=(
                    args.stale_after_days
                    if args.stale_after_days is not None
                    else DEFAULT_STALE_AFTER_DAYS
                ),
                watermarks=watermarks,
                poison=ledger,
                stop=stop,
            )

    report = asyncio.run(_drive())
    _print_run_report(report)
    shown_warnings = _print_warnings(report.warnings)
    shown = _print_errors(report.errors, RUN_ERROR_PRINT_LIMIT)
    # THE TERMINAL IS A CHANNEL AND THE RUNNER CANNOT SEE IT. Round-7 MEDIUM: an
    # interactive ``ingest --all`` resolves to ``_silent_notification`` and does
    # its reporting HERE, after ``run_imports`` has returned — so a person
    # watching these lines scroll past was told, the runner never heard about
    # it, and the same TickTick gap re-reported on every single run, forever.
    # Declared here because this is where the print happens; derived from
    # ``shown`` because that is what the print actually put on screen.
    #
    # NOT A WEAKENING OF THE UNATTENDED CASE: ``_stderr_delivery`` answers
    # "nothing" unless stderr is a tty, and under the timer it is the journal.
    # A second call is safe — the barriers cleared whatever the notify hook
    # already earned — and it can only ever ADD lines a human saw.
    late = len(report.run_errors)
    watched = _stderr_delivery([*shown, *shown_warnings], report.reported)
    commit_report_barriers(adapters, report, watched)
    # The staleness markers get the same second chance, and need it for the same
    # reason: an interactive ``ingest --all`` installs no notifier at all, so the
    # WARNING lines above are the only channel there is, and without this a
    # person who read them off their own terminal would be told again on every
    # single run. Under the timer ``_stderr_delivery`` answers "nothing" — the
    # journal is not an audience — so the unattended case is untouched.
    commit_staleness_receipts(report, watched)
    # And the poison ledger, for the third time on the same declaration. A
    # person who read a never-seen-before corrupt-line report off their own
    # terminal has been told; without this the interactive run — which installs
    # no notifier at all — could never record anything, so every manual
    # ``ingest --all`` would report the same permanent faults forever.
    commit_fault_receipts(report, watched, ledger)
    # A barrier that raised lands in ``run_errors`` after the block above has
    # printed, so it would otherwise change the exit code with nothing on stderr
    # to explain it. Same treatment as the single-source path's ``late``.
    _print_errors(report.run_errors[late:], RUN_ERROR_PRINT_LIMIT)
    if report.errors:
        # Same 3 as the single-source path, for the same reason: a run that
        # completed but dropped files is not a success, and a timer that reads
        # 0 as success lets the index rot unnoticed.
        return EXIT_COMPLETED_WITH_ERRORS
    # KNOWN POISON EXITS 0, AND THAT IS A DECISION, not an oversight. See
    # ``EXIT_COMPLETED_WITH_ERRORS`` for the argument in full: a third exit code
    # would still be non-zero, the unit treats every non-zero as a failure and
    # notifies, so introducing one reproduces the bug this fixes until somebody
    # hand-edits a systemd unit in another repository. The visibility that a
    # non-zero code would have bought is bought instead by the ``poison=`` count
    # in the summary above, the per-fault notes under each source, and
    # ``aggregator status`` — none of which can be lost to a stale unit file.
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
    print(f"ingest --all: run={report.run_id}")
    for name, a in report.adapters.items():
        # ``unchanged`` and ``window`` are what tell an incremental run from
        # the doom loop in a journal read after the fact. Both printed a run
        # that "updated" ~372k rows; only one of them did any work, and only
        # one of them was given a window.
        print(
            f"  {name}: added={a.added} updated={a.updated} "
            f"unchanged={a.unchanged} skipped={a.skipped} "
            f"errors={len(a.errors)} poison={len(a.known_faults)} "
            f"window={a.window or 'n/a'}"
            f"{' INTERRUPTED' if a.interrupted else ''}"
            f"{' RESTING' if a.skipped_for_backoff else ''}"
        )
        for note in a.notes:
            print(f"    note: {note}")
    # ``poison`` is on the total line for the same reason ``unchanged`` is: it
    # is the field that distinguishes a genuinely clean run from one that exits
    # 0 while several records have been missing from the index for months.
    # Zero-visibility is the failure mode; a number nobody reads is not.
    print(
        f"  total: added={report.added} updated={report.updated} "
        f"unchanged={report.unchanged} skipped={report.skipped} "
        f"errors={len(report.errors)} warnings={len(report.warnings)} "
        f"poison={len(report.known_faults)} "
        f"({report.quarantined_records} record(s) held out of the index)"
        f"{' (INTERRUPTED — stopped at a chunk boundary; the next run resumes)' if report.interrupted else ''}"
    )


def _approve_vector_reindex(store: Store, *, assume_yes: bool) -> bool:
    """Show what ``--reindex`` would destroy, then ask. Round 3's H1, CLI half.

    THE ASYMMETRY THIS REMOVES. ``ingest --rebuild`` drops rows it can re-fetch
    in minutes, and it still prints a count and demands a ``y`` on stdin.
    Deleting the vector index costs 25-30 days of continuous CPU to put back,
    and its opt-in was a shell variable that every command honoured in silence.
    The cheaper operation had the louder gate.

    So: the same gate, on the more expensive operation. The count comes off
    disk rather than out of an estimate, because "some vectors" is not a
    number anyone can weigh a month of CPU against.

    A REINDEX WITH NOTHING TO DELETE IS NOT A QUESTION. ``migrate()`` already
    adopts an empty or absent index for free — nothing computed exists, so
    nothing is at stake — and prompting there would train the operator to
    answer ``y`` without reading, which is exactly how the real prompt stops
    working.

    ``--yes`` is for scripted use and mirrors ``ingest --yes``. It skips the
    question, never the report: the counts are printed either way, so a run
    in a journal still says what it destroyed.
    """
    vectors, rows = store.vector_reindex_preview()
    if not vectors:
        print(
            "embed --reindex: no computed vectors on disk, so there is "
            "nothing to delete and nothing to confirm. Continuing."
        )
        return True
    print(
        f"embed --reindex will DELETE {vectors} vector(s) from this cache and "
        f"return {rows} row(s) to the embed backlog.\n"
        f"They cannot be converted, only recomputed: the last full backfill "
        f"of this corpus was measured at 25-30 days of continuous CPU.\n"
        f"Do this only if the embedding model genuinely changed. If "
        f"AGGREGATOR_EMBED_BACKEND is merely exported in this shell, "
        f"`unset AGGREGATOR_EMBED_BACKEND` is the whole fix and costs nothing.",
        file=sys.stderr,
    )
    if assume_yes:
        print("embed --reindex: --yes given; proceeding without asking.")
        return True
    if _confirm_force_on_stdin("Type 'y' to delete them and re-embed: "):
        return True
    print(
        "aborted: vector reindex not confirmed. NOTHING WAS DELETED — the "
        "vectors on disk are intact and the backlog is untouched. The vector "
        "arm stays switched off for mismatched provenance; FTS5 keyword "
        "search is unaffected.",
        file=sys.stderr,
    )
    return False


def _cmd_embed(args: argparse.Namespace, _store: Store | None = None) -> int:
    """Background embed worker — fills the v5 vector index.

    ``--catchup`` drains the whole backlog and IS WHAT THE TIMER RUNS
    (``embed --catchup --source both``, see ``nix/aggregator.nix``);
    ``--once`` does a single batch and exits, and no deployed unit uses it —
    it is a hand-run probe. This comment used to say the opposite, which was
    wrong in the direction that costs the most: a batch is bounded at
    ``_MAX_BATCH_CHUNKS`` chunks, about fifteen minutes of encoder time, so one
    batch per 30-minute tick would run a backfill measured in weeks at half
    speed at best.

    Both take an OS-level ``flock`` on ``<cache>.embed.lock``, so a slow
    catchup and a timer tick cannot both consume the same batch. A tick that
    finds the lock held is a healthy no-op and exits 0.

    REFUSES TO RUN WITHOUT A WRITABLE VECTOR INDEX, loudly and non-zero, and
    the arm can be unusable for two unrelated reasons. With sqlite-vec missing
    the store's vec writers no-op; embedding the batch anyway would advance
    ``embedding_state`` past rows that have no vector, and nothing ever looks
    at a row twice. That is the watermark-ahead-of-data failure the ingest
    rules forbid. With the extension loaded but the index under an S1
    provenance refusal, the writes would be this build's vectors landing in a
    table another model filled. Both are checked HERE, before a row is
    selected, so the run is refused and the backlog is left exactly where it
    was — see the second check for what discovering it late used to cost.

    THIS IS ALSO THE ONLY COMMAND THAT MAY DELETE THE VECTOR INDEX, under
    ``--reindex``, and that is round 3's H1. The consent used to be
    ``AGGREGATOR_VECTOR_REINDEX=1`` read inside ``Store.migrate()`` — which
    every subcommand calls — so it was ambient, sticky, and honoured by reads.
    It belongs here because the reindex is embed-side maintenance: this is the
    command that owns the index, and the only one that can put back what it
    destroys.
    """
    store = _store or Store()
    # BEFORE ``migrate()``, because migrate is what would do the deleting.
    if args.reindex and not _approve_vector_reindex(store, assume_yes=args.yes):
        return 1

    if not store.vector_available:
        print(
            "ERROR: aggregator embed cannot run — the sqlite-vec extension "
            "did not load, so no vector can be written. Refusing rather than "
            "advancing embedding_state past rows with no vector. FTS5 search "
            "is unaffected. Reinstall the `sqlite-vec` wheel for this "
            "interpreter and re-run `aggregator embed --catchup`.",
            file=sys.stderr,
        )
        return 1

    # THE OTHER WAY THE ARM IS OFF, and round 3's H2. ``vector_available``
    # answers "did the extension load?" — which under an S1 provenance refusal
    # is YES, while the index it loaded is one this build may not write to.
    # Without this check the worker got all the way to
    # ``commit_embed_batch``, where ``_require_vector`` raises, having already
    # embedded a full batch: an uncaught traceback, the unit marked failed, a
    # CRITICAL toast naming missing weights and a missing wheel (neither of
    # which applied), the run's ledger report discarded with the exception,
    # and 500 rows of CPU spent on work that was never written. On a
    # 30-minute timer, every tick, indefinitely.
    #
    # The store's own refusal text is reprinted verbatim rather than
    # summarised: it already names what disagreed, that nothing was deleted,
    # and the two opposite remedies — and only the operator knows which one
    # they meant.
    #
    # ASKED TWICE, AND BOTH ARE CHEAP. This first call uses the no-argument
    # form — "what would ``Embedder()`` load here?" — which is the answer the
    # read path uses and costs one indexed ``meta`` lookup. It is asked BEFORE
    # the weights load so the common mismatch (a stray
    # ``AGGREGATOR_EMBED_BACKEND``) is refused without paying a model load.
    refusal = store.vector_quarantine
    if refusal is not None:
        print(
            f"ERROR: aggregator embed cannot run — {refusal}\n"
            f"No row was embedded and the backlog is untouched.",
            file=sys.stderr,
        )
        return 1

    # THE EMBEDDER IS BUILT BEFORE ``migrate()``, and that is round 4's
    # triple-converged finding. ``migrate(embedder=)`` landed in round 3 so the
    # provenance stamp could name the model that actually fills the index —
    # and nothing ever passed it. ``grep -rn 'migrate(embedder' aggregator/``
    # returned nothing, so the only write path in the codebase went on stamping
    # from ``AGGREGATOR_EMBED_BACKEND`` and ``vector_provenance(embedder)`` was
    # dead code: a fix with no production caller, which is a passing test and
    # not a fix.
    #
    # The order costs one model load ahead of the second quarantine check
    # below, and that is the right trade now that the version string carries
    # the QUANTIZATION and the CHUNKER version. Neither is knowable from the
    # environment, so a stamp written before the embedder exists cannot be
    # right about them — while the load itself is seconds against a backfill
    # measured in weeks, and the cheap refusal above has already caught the
    # common case.
    embedder = Embedder()
    store.migrate(allow_vector_reindex=args.reindex, embedder=embedder)

    # THE SECOND ASKING, now that the index has been reconciled against the
    # embedder that will write it. The first call answered for the process; a
    # mismatch that only the embedder's own identity reveals surfaces here, and
    # it must still refuse before a row is selected.
    refusal = store.vector_quarantine
    if refusal is not None:
        print(
            f"ERROR: aggregator embed cannot run — {refusal}\n"
            f"No row was embedded and the backlog is untouched.",
            file=sys.stderr,
        )
        return 1

    if _would_start_a_second_index_by_accident(store):
        return 1

    lock_path = Path(str(store.db_path) + ".embed.lock")
    lock_path.touch(exist_ok=True)
    lock_fd = os.open(str(lock_path), os.O_RDWR)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another embed worker is running; exiting")
            return 0

        plan = _embed_plan(args.source)
        outcome = _EmbedOutcome()
        ledger = PoisonLedger(store)
        # BEFORE any row is selected: a claim left on disk means the previous
        # worker did not survive that row, and it must be set aside now or the
        # very next select hands it back and this process dies the same way.
        _blame_crashed_row(store, ledger, outcome)
        try:
            # WHAT MAKES THE CLAIM A CRASH DETECTOR RATHER THAN A REBOOT
            # DETECTOR. The claim is written before the attempt and can only
            # be cleared by code that gets to run, so under SIGTERM's default
            # disposition — no handler, no unwinding — a `systemctl stop`, a
            # reboot or a deploy left one behind and the NEXT run read it as a
            # kill: a good row held in the quarantine ledger, marked 'error',
            # printed to stderr, a non-zero exit and a CRITICAL toast. On a
            # 25-30 day backfill that is a false alarm every reboot, and the
            # row it names stays out of the index.
            #
            # Same shape ingest already uses (``graceful_shutdown``): the
            # handler only sets a flag, the row in flight finishes and
            # releases its claim, and the loop stops at a boundary it chose.
            # A signal that CANNOT be handled — SIGKILL, an OOM kill, a
            # segfault — still leaves the claim, so the crash-blame path is
            # untouched. Which signal the process could handle is exactly the
            # crash/shutdown distinction, so it is the one being read.
            with graceful_shutdown() as stop:
                # ONCE PER ONTOLOGY, ABOVE THE WALK. See ``_requeue_due_rows``:
                # the walk visits an ontology up to four times, and a requeue
                # inside it would make "no row is retried by the run that held
                # it" depend on the backoff outlasting a pass.
                for kind in dict.fromkeys(k for k, _ in plan):
                    _requeue_due_rows(store, ledger, kind)
                for kind, source in plan:
                    worked = _embed_backlog(
                        store, embedder, kind, args, ledger, outcome, stop,
                        source=source,
                    )
                    if outcome.interrupted:
                        break
                    # ``--once`` IS ONE BATCH, NOT ONE PER GROUP — and it has to
                    # skip PAST the groups that are already drained, or a
                    # finished dropbox starves everything ranked behind it and
                    # the priority order becomes a deadlock.
                    if args.once and worked:
                        break
            _flip_completed_pointer(store, args, outcome)
            _report_embed_progress(store)
        except (EmbedderUnhealthyError, EmbedStoreUnavailableError) as e:
            # ANNOUNCE FIRST, THEN ABORT. Rows attributed earlier in this run
            # passed their own health probe and have ledger entries already
            # written; swallowing their lines here would leave each one
            # known-but-never-reported, i.e. quiet on this run and quiet on
            # every run after it. Exactly the silence the ledger's bargain
            # forbids.
            _report_embed_outcome(outcome)
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
    finally:
        os.close(lock_fd)
    return _report_embed_outcome(outcome)


def _would_start_a_second_index_by_accident(store: Store) -> bool:
    """Refuse a whole new backfill that only a shell variable asked for.

    ROUND 3'S H1, TRANSPOSED. Keying vectors on ``(chunk_id, model)`` means a
    stray ``AGGREGATOR_EMBED_BACKEND`` can no longer DELETE the index — the old
    vectors keep their own key and nothing is dropped. What it can still do is
    commit this machine to building a second one, and on this hardware that is
    a multi-week backfill (``docs/embedding-throughput.md``) for a model nobody
    chose. "Costs weeks of CPU on the strength of a leftover export" is the
    failure being guarded, and it does not care which direction the weeks go.

    THE DISCRIMINATOR IS THE VARIABLE, not the change. A model change that came
    from SOURCE — a new pin in ``aggregator/core/embed.py``, deployed as a new
    store path — is deliberate by construction: someone edited a constant,
    reviewed it and rebuilt. There is no ambiguity to resolve and no prompt
    worth showing. A change that exists only because a variable is exported in
    this shell is exactly the ambiguous case, and it is the one this refuses.

    So there is no new flag: ``unset AGGREGATOR_EMBED_BACKEND`` is the fix when
    it was a mistake, and editing the pin is the fix when it was not.
    """
    if not os.environ.get("AGGREGATOR_EMBED_BACKEND", "").strip():
        return False
    other = store.other_indexed_model()
    if other is None:
        return False
    print(
        f"ERROR: aggregator embed cannot run — refusing to use the vector "
        f"index this shell is pointing at. This cache holds vectors for "
        f"{other!r}, this process is configured for "
        f"{store.embedding_model!r}, and the difference comes from "
        f"AGGREGATOR_EMBED_BACKEND being exported here. Starting the backfill "
        f"would commit this machine to building a SECOND index from scratch — "
        f"weeks of CPU on this hardware, see docs/embedding-throughput.md.\n"
        f"NOTHING WAS DELETED and the backlog is untouched: vectors are keyed "
        f"(chunk_id, model), so the existing index is intact and still serves "
        f"any process configured for it.\n"
        f"If this was a leftover export, `unset AGGREGATOR_EMBED_BACKEND` is "
        f"the whole fix. If the model change is intended, make it in source — "
        f"the pin in aggregator/core/embed.py — so the deployed artifact and "
        f"the index agree; a shell variable cannot say that.",
        file=sys.stderr,
    )
    return True


def _flip_completed_pointer(
    store: Store, args: argparse.Namespace, outcome: _EmbedOutcome
) -> None:
    """Publish the index once the backlog is genuinely drained.

    RULE 4 OF THE REFERENCE DESIGN, the caller's half. ``completed_at`` is what
    keeps a half-built index for a NEW model from being served while the
    previous one is still good — a partially filled embedding space answers
    with plausible scores drawn from whichever rows happened to be embedded
    first, and nothing in a result says so.

    ONLY FROM A RUN THAT COULD SEE THE WHOLE BACKLOG. ``--once`` does a single
    batch by design, and an interrupted or stalled ``--catchup`` stopped
    somewhere it chose rather than at the end. The store refuses over a live
    backlog anyway — this is the cheaper guard in front of that one, and it
    keeps a ``--once`` probe from paying for a scan it cannot act on.

    Best-effort and quiet on refusal: rows arriving from ingest between the
    last batch and here are an ordinary race, and the next tick flips it.
    """
    if args.once or outcome.interrupted or outcome.stalled:
        return
    if args.source != "both":
        # Only one ontology was drained, so "the index is complete" is not a
        # claim this run is entitled to make about the other.
        return
    try:
        model = store.mark_embedding_version_complete()
    except RuntimeError:
        return
    if store.embedding_version_state(model)["completed_at"]:
        print(f"embed: index complete for {model}")


#: Rows per committed chunk of the provenance backfill.
#:
#: SIZED AGAINST WHAT A KILL COSTS, not against throughput. Each chunk is one
#: ``executemany`` of single-column UPDATEs plus one COMMIT, and under the
#: narrowed ``observations_au`` trigger the whole 549,952-row table takes a
#: measured ~37 s — so the chunk size barely moves the wall clock and entirely
#: decides the blast radius of a SIGTERM. 2,000 is well under a second of work.
_PROVENANCE_CHUNK = 2000


def _cmd_provenance(
    args: argparse.Namespace, store: Store, sources: dict[str, Any]
) -> int:
    """Classify who composed each observation. Resumable, chunked, pure UPDATE.

    WHY THIS IS A SUBCOMMAND AND NOT PART OF ``migrate()``. ``migrate()`` runs
    on EVERY subcommand, read-only queries included (``store.py``'s dispatch in
    ``main``). A classification pass over half a million rows behind an
    ``aggregator query`` is precisely the hours-long unattended surprise the
    incremental-ingest rules exist to abolish. It is also not something a read
    should ever pay for.

    TWO ROUTES, IN THIS ORDER, AND THE ORDER IS THE ACCURACY.

    1. **JSONL-derived.** A read-only walk of the sessions archive, producing
       ``uuid -> provenance`` from the raw lines. Measured 59% machine share of
       ``type='user'``, because the vendor's structural fields (``isSidechain``,
       ``isMeta``, ``promptSource``, ``origin``) only exist there.
    2. **DB-only.** Type, body and the owning session's ``kind`` — everything
       route 1 could not reach: the 3,621 ``claude-web`` rows whose export is
       not on disk, live files the walk skips, and any session whose archive has
       been deleted. Measured 100% precision, 58% recall against route 1, which
       is why it runs SECOND and only over what is left.

    Both call the same ``classify``, so ingest and backfill cannot disagree.

    IT IS A PURE UPDATE OF ONE COLUMN. No re-ingest, no re-scrub, no ``body``
    write, no ``src_hash`` write, no ``embedding_state`` reset. That is not an
    optimisation: a re-ingest of this corpus is ~11 hours of Presidio at the
    measured 827 rows/min, and resetting ``embedding_state`` would discard the
    observation vector arm — weeks of CPU — the moment that arm is warm.

    RESUMABILITY IS ``provenance IS NULL`` AND NOTHING ELSE. The watermark lives
    in the rows it describes, so there is no sidecar to fall out of step with
    them and a kill at any moment costs at most one chunk. A file whose rows are
    all classified is skipped without being opened.

    CPU: single-threaded by construction. It parses JSON and runs SQLite; it
    loads no model and holds no thread pool, so there is no pool to pin and it
    cannot take more than one core. It is also hand-run — no timer starts it.
    """
    src = sources.get("sessions")
    errors: list[str] = []
    if args.reclassify:
        # THE DOCUMENTED PATH AFTER A CLASSIFIER REVISION, and it is cheap
        # exactly because provenance is not in ``_src_hash``: resetting the
        # column tells nothing else in the corpus that anything changed.
        c = store._c()
        c.execute("UPDATE observations SET provenance = NULL")
        c.commit()
        print("provenance: reset every row to unclassified")

    classified = 0
    with graceful_shutdown() as stop:
        from_archive, stopped = _provenance_from_archive(
            store, src, args.chunk_size, stop, errors
        )
        classified += from_archive
        if not stopped:
            from_db, stopped = _provenance_from_db(
                store, args.chunk_size, stop, errors
            )
            classified += from_db

    remaining = store.count_unclassified_observations()
    print(
        f"provenance: classified={classified} still_unclassified={remaining}"
        f"{' (INTERRUPTED — stopped at a chunk boundary; the next run resumes)' if stopped else ''}"
    )
    if errors:
        for line in errors[:ERROR_PRINT_LIMIT]:
            print(f"ERROR: {line}", file=sys.stderr)
        if len(errors) > ERROR_PRINT_LIMIT:
            print(
                f"ERROR: ... and {len(errors) - ERROR_PRINT_LIMIT} more",
                file=sys.stderr,
            )
        # NON-ZERO, because a pass that skipped part of the corpus and a pass
        # that classified all of it must not look the same from a shell.
        return EXIT_COMPLETED_WITH_ERRORS
    return 0


def _owed_archive_paths(store: Store) -> set[str]:
    """JSONL paths that still own at least one unclassified observation.

    THE WORK LIST, AND IT IS WHY THIS IS HISTORY-AWARE WITHOUT A LEDGER. One
    indexed query answers "which files still owe rows"; every other file in the
    archive is skipped without being opened, so a re-run over a classified
    corpus reads nothing. The result is ~11k paths — a work list, not the
    corpus — and it is recomputed from the database on every run, so a kill
    cannot leave it stale.
    """
    return {
        row[0]
        for row in store._c().execute(
            "SELECT DISTINCT s.jsonl_path FROM sessions s "
            "JOIN observations o ON o.session_id = s.session_id "
            "WHERE o.provenance IS NULL"
        )
        if row[0]
    }


def _provenance_from_archive(
    store: Store,
    src: Any,
    chunk_size: int,
    stop: Callable[[], bool],
    errors: list[str],
) -> tuple[int, bool]:
    """Route 1: classify from the raw JSONL, one file at a time.

    Streams. The per-file generator is the sessions source's own parser, so the
    lines this reads are byte-for-byte the lines ingest reads — a second parser
    here would be a second thing to keep in step, and a divergence between them
    would be invisible because both sides would look classified.

    A file that cannot be read is recorded and skipped: one unreadable archive
    must not cost the other 11,763 their classification, and the run still exits
    non-zero so the gap is not mistaken for completeness.

    THE STOP IS CHECKED AT EVERY CHUNK, NOT AT EVERY FILE. Files in this archive
    are not uniform — the largest hold tens of thousands of lines — so a stop
    honoured only between files is a stop a SIGTERM can wait a long time for.
    The chunk boundary is already the durable point; it is therefore also the
    right place to stop.
    """
    if src is None or not hasattr(src, "_iter_parsed"):
        return 0, False
    owed = _owed_archive_paths(store)
    if not owed:
        return 0, False
    pending: list[tuple[str, str]] = []
    written = 0
    for path in src._iter_jsonl_files(errors):
        if stop():
            return written + _flush_provenance(store, pending), True
        if str(path) not in owed:
            continue
        try:
            for parsed in src._iter_parsed(path, errors):
                pending.append((parsed.provenance, parsed.uuid))
                if len(pending) >= chunk_size:
                    written += _flush_provenance(store, pending)
                    if stop():
                        return written, True
        except OSError as e:
            # Caught here as well as inside the parser because the walk itself
            # can fail on a file the stat succeeded for — a permission change
            # or a vanished symlink between the two calls.
            errors.append(f"{path}: provenance scan failed: {e}")
    written += _flush_provenance(store, pending)
    _drop_declared_faults(src, errors)
    return written, False


def _drop_declared_faults(src: Any, errors: list[str]) -> None:
    """Take the parser's PERMANENT faults back out of this run's errors.

    A line the JSON parser rejects will never parse, no run will ever classify
    it, and there is no row in the database for it either — ingest could not
    store it. Counting it as a backfill failure would make
    ``aggregator provenance --backfill`` exit non-zero for ever over damage
    that has nothing to do with provenance: a permanently-red alarm, which is
    the alarm an operator learns to dismiss unread. Exactly the distinction
    ``PermanentFault`` exists to draw, so this reuses the source's own
    declaration rather than re-deciding it here.

    Everything the source did NOT declare — a failed ``stat``, a file it could
    not open, a vendor format change — stays loud, on every run, until a human
    looks at it.
    """
    if not hasattr(src, "drain_faults"):
        return
    declared = {fault.line for fault in src.drain_faults()}
    if declared:
        errors[:] = [e for e in errors if e not in declared]


def _provenance_from_db(
    store: Store,
    chunk_size: int,
    stop: Callable[[], bool],
    errors: list[str],
) -> tuple[int, bool]:
    """Route 2: classify what is left from type, body and the session's kind.

    The chunk is ``WHERE provenance IS NULL LIMIT n`` — the same shape
    ``select_unembedded`` uses, and resumable for the same reason: the query IS
    the ledger, so running it twice equals running it once.

    A CHUNK THAT SELECTS ROWS AND WRITES NONE STOPS THE RUN. Without that the
    next iteration asks the identical question and gets the identical rows, for
    ever — a poison row silently retried in a loop, which the ingest rules
    forbid twice over. It cannot happen while ``classify`` is total, so if it
    ever does, the classifier is broken and saying so beats spinning.
    """
    written = 0
    while True:
        if stop():
            return written, True
        rows = list(
            store._c().execute(
                "SELECT o.obs_id AS obs_id, o.type AS type, o.body AS body, "
                "       s.kind AS kind "
                "FROM observations o "
                "LEFT JOIN sessions s ON s.session_id = o.session_id "
                "WHERE o.provenance IS NULL LIMIT ?",
                (chunk_size,),
            )
        )
        if not rows:
            return written, False
        pending = []
        for row in rows:
            value = classify(row["type"], row["body"], session_kind=row["kind"])
            if value:
                pending.append((value, row["obs_id"]))
        if not pending:
            errors.append(
                f"provenance backfill stalled: {len(rows)} unclassified row(s) "
                f"came back and the classifier placed none of them, so the next "
                f"chunk would select the same rows again. Refusing to loop. "
                f"First id: {rows[0]['obs_id']!r}."
            )
            return written, False
        written += _flush_provenance(store, pending)


def _flush_provenance(store: Store, pending: list[tuple[str, str]]) -> int:
    """Write one chunk and COMMIT it. Empties ``pending`` in place.

    ``AND provenance IS NULL`` makes the write idempotent and monotone: a row
    that already carries a value — because ingest classified it, or because an
    earlier chunk did — is never rewritten, so a re-run costs no writes at all
    and two overlapping runs cannot fight.
    """
    if not pending:
        return 0
    c = store._c()
    before = c.total_changes
    c.executemany(
        "UPDATE observations SET provenance = ? "
        "WHERE obs_id = ? AND provenance IS NULL",
        pending,
    )
    # COUNTED FROM THE DATABASE, not from ``len(pending)``. The archive walk
    # sees lines that never became rows — non-dominant resume-prefix copies,
    # lines dropped for want of a timestamp — and billing those as classified
    # would report a corpus finished while rows are still NULL.
    written = c.total_changes - before
    c.commit()
    pending.clear()
    return written


def _cmd_seed_models() -> int:
    """Fetch/verify the model weights. NO DATABASE, NO ROWS, NO LOCK.

    WHY THIS IS ITS OWN COMMAND. The seed unit ran ``embed --once --source
    observations``, which builds only the ``Embedder``. Nothing anywhere else
    builds the ``Reranker`` outside a live ``rerank=True`` call, and every
    model load is offline unless a human opts in — so the reranker's weights
    were never fetched by ANY path, and ``rerank`` was guaranteed to fail on
    this machine forever no matter how often the seed unit ran.

    The second half is what it was doing instead: opening the index,
    migrating it, taking the embed lock, claiming a row of an untrusted
    corpus and advancing a watermark, all as a side effect of "make sure the
    model is downloaded". This command is dispatched before ``main`` builds a
    ``Store`` at all, so warming a model cache cannot write to the index.

    BOTH ARE ATTEMPTED EVEN IF THE FIRST FAILS. The operator is doing this to
    find out what is missing; reporting one model per invocation would make
    them run it twice to learn something one run already knew.
    """
    from aggregator.core.rerank import Reranker

    allowed = downloads_allowed()
    failures: list[tuple[str, BaseException]] = []
    for label, build in (("embedder", Embedder), ("reranker", Reranker)):
        try:
            build()
        except Exception as e:  # noqa: BLE001 - reported per model, not handled
            failures.append((label, e))
        else:
            print(f"embed --seed-models: {label} ready")
    if not failures:
        return 0
    for label, error in failures:
        print(
            f"ERROR: the {label} weights could not be loaded "
            f"({type(error).__name__}: {error})",
            file=sys.stderr,
        )
    if allowed:
        print(
            f"Downloads were permitted ({MODEL_DOWNLOAD_ENV} is set), so this "
            f"is not the offline guard — the hub, the network or the disk is "
            f"the problem. Re-run `{SEED_MODELS_COMMAND}` once it is fixed.",
            file=sys.stderr,
        )
    else:
        # NAMED IN FULL, because the whole failure mode this command exists
        # for is weights nothing was ever going to fetch. "Enable downloads"
        # would leave the operator to work out how.
        print(
            f"Model loads are offline unless a human asks for them, and "
            f"{MODEL_DOWNLOAD_ENV} is not set, so nothing was fetched. Run "
            f"this once, with network access:\n"
            f"    {SEED_MODELS_COMMAND}",
            file=sys.stderr,
        )
    return 1


def _report_embed_outcome(outcome: _EmbedOutcome) -> int:
    """Say what was set aside, and decide whether this run failed.

    THE KNOWN-POISON BARGAIN, one layer below where PR #5 struck it. A row the
    worker has never failed on before is NEWS: its id and its error go to
    stderr and the run exits non-zero, because "fail loudly" means an operator
    learns of each new problem exactly once. A row already in the ledger is
    NOT news: it becomes a count on stdout and the run exits 0, because the
    same permanent problem shouting twice an hour is how an operator learns to
    ignore the notifier, and that costs the next real failure its audience.

    Quiet is only defensible while it is not the same as forgotten, so the
    quiet line still carries a number and still names where the detail lives.
    """
    if outcome.interrupted:
        # Said out loud on every interrupted run, because "stopped early" and
        # "drained the backlog" both otherwise print nothing at all, and the
        # difference is whether the next tick has work left.
        print(
            "embed: INTERRUPTED — a stop was requested, the row in flight "
            "finished and its vectors were committed, and the rest of the "
            "backlog is untouched. The next run resumes from here."
        )
    if outcome.new_failures:
        for line in outcome.new_failures:
            print(line, file=sys.stderr)
        print(
            f"embed: {len(outcome.new_failures)} row(s) set aside this run and "
            f"recorded in the quarantine ledger. Each is retried with backoff "
            f"and given up on after {POISON_MAX_ATTEMPTS} attempts; the rest of "
            f"the backlog was embedded. `aggregator status` lists them. This is "
            f"reported once per row — later runs stay quiet.",
            file=sys.stderr,
        )
        return 1
    if outcome.known_failures:
        print(
            f"embed: {outcome.known_failures} known-bad row(s) failed again "
            f"(already reported once; see `aggregator status`)"
        )
    if outcome.released:
        print(f"embed: {outcome.released} previously-bad row(s) embedded cleanly")
    if outcome.superseded:
        print(
            f"embed: {outcome.superseded} row(s) were edited by ingest while "
            f"being embedded; the stale vector was discarded and they stay in "
            f"the backlog for the next run"
        )
    if outcome.stalled:
        # SAID OUT LOUD, because "drained the backlog" and "gave up on it"
        # otherwise both print nothing and exit 0, and the difference is
        # whether the index is still filling. Not a failure: every row is
        # intact at NULL, and a writer that is outrunning the worker is a
        # condition the next tick may well not find.
        print(
            f"embed: STOPPED WITHOUT PROGRESS — {_MAX_STALLED_BATCHES} "
            f"consecutive batches moved no row out of the backlog, so this run "
            f"ended rather than re-reading the same rows indefinitely. The "
            f"usual cause is ingest rewriting those bodies faster than they "
            f"can be embedded. Nothing was lost: every row is still queued and "
            f"the next run re-reads it from its current body. If `aggregator "
            f"status` shows the pending count flat across several runs, the "
            f"writer is not backing off and the two timers need separating."
        )
    return 0


def _positive_int(raw: str) -> int:
    """An ``argparse`` type that refuses the two values SQLite reinterprets.

    Both failures are silent, which is why this is a parser-level refusal
    rather than a runtime check. ``--batch-size 0`` becomes ``LIMIT 0``, so
    ``select_unembedded`` returns nothing, ``_embed_backlog`` concludes the
    backlog is drained, and ``--catchup`` exits 0 having embedded nothing —
    under a timer, an index that never fills while every run looks
    successful. ``--batch-size -5`` becomes ``LIMIT -1``, which SQLite reads
    as NO limit, collapsing the chunked and checkpointed worker into one
    unbounded batch over 483k rows.

    The Nix option already enforces a positive int; the CLI did not, so a
    hand-run command had no such protection.
    """
    try:
        value = int(raw)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"{raw!r} is not an integer") from e
    if value < 1:
        raise argparse.ArgumentTypeError(
            f"must be at least 1, got {value}. 0 makes --catchup a silent "
            f"no-op (LIMIT 0 returns no rows, so the backlog reads as "
            f"drained); a negative value reaches SQLite as LIMIT -1, i.e. one "
            f"unbounded batch over the whole corpus."
        )
    return value


@dataclass
class _EmbedOutcome:
    """What one ``aggregator embed`` invocation did to the poison ledger."""

    #: Ready-to-print stderr lines, one per row failing for the FIRST time.
    new_failures: list[str] = field(default_factory=list)
    #: Rows that failed again having already been reported. Counted, not named.
    known_failures: int = 0
    #: Rows that used to fail and embedded cleanly this time.
    released: int = 0
    #: Rows whose body was edited while the worker was embedding it. Their old
    #: vector was discarded and they stay in the backlog for the next run —
    #: self-healing, so this is a count rather than a failure.
    superseded: int = 0
    #: A stop was requested (SIGTERM/SIGINT) and the run ended at a boundary
    #: with backlog still to do. Reported, but NOT a failure: the same event
    #: ``ingest`` prints as INTERRUPTED. A run that exited non-zero for being
    #: asked to stop is the false alarm this flag exists to replace.
    interrupted: bool = False
    #: ``--catchup`` ended because consecutive batches moved no row out of the
    #: backlog, not because the backlog was empty. Reported, not failed: every
    #: row is intact at NULL and the next tick re-reads it from its current
    #: body. See ``_MAX_STALLED_BATCHES``.
    stalled: bool = False


class EmbedderUnhealthyError(RuntimeError):
    """The embedder could not embed a known-good probe. Not a bad row.

    Raised out of the batch loop so the whole run stops without attributing
    anything to a record. See :func:`_embedder_is_healthy`.
    """


#: A body the model must be able to embed if it is working at all: short,
#: plain ASCII, nothing a chunker or tokenizer has any reason to choke on.
_EMBED_HEALTH_PROBE = "aggregator embed health probe"

#: How many consecutive batches may move NO row out of the backlog before
#: ``--catchup`` gives up and lets the next tick try.
#:
#: Three, not one: a single zero-write batch is an ordinary race — ingest
#: rewrote a body while the worker held it — and the very next pass re-reads
#: the new fingerprint and succeeds, so stopping on the first would abandon a
#: healthy backlog over one unlucky interleaving. Three consecutive passes that
#: all move nothing is no longer a race; it is a writer keeping ahead of the
#: worker, and more attempts inside this run will not change that.
#:
#: Not large, either. The cost of stopping is one timer tick (30 minutes) and
#: nothing else — every row is still at NULL with its vectors intact — while
#: the cost of not stopping is a worker spinning at full CPU on a laptop.
_MAX_STALLED_BATCHES = 3

#: The measured cost of one full chunk, in wall-clock seconds. 4000 characters
#: ≈ 804 tokens at the ~40 tokens/second of ``docs/embedding-throughput.md``.
#: Named here so the cap below and the ``--help`` sentence describing it are the
#: same arithmetic rather than two numbers that can drift apart.
_SECONDS_PER_CHUNK = 20

#: How much ENCODER TIME one checkpointed batch may buy, counted in chunks.
#:
#: THE BATCH IS BOUNDED BY CHUNKS BECAUSE THE BILL IS PER CHUNK. ``--batch-size``
#: bounds rows, and rows differ in chunk count by two orders of magnitude, so
#: the interval between durable checkpoints was whatever the backlog happened to
#: hold rather than anything anyone chose. Measured read-only against the live
#: cache on 2026-08-25, at the chunker's own ``chunk-4000-400`` geometry:
#: dropbox is 1536 records ≈ 3838 chunks, and the FIRST 500-row batch of it is
#: ≈1149 chunks — ~6.4 hours before a single vector is written. Confirmed live
#: rather than inferred: twenty minutes into a run, ``chunk_embeddings`` was
#: still empty. The 2026-08-16 directive asks for "bounded batches with
#: committed checkpoints, so a kill at any moment loses at most one chunk"; 6.4
#: unprotected hours is not that, and it fell out of a unit mismatch.
#:
#: FORTY-FIVE, WHICH IS ABOUT FIFTEEN MINUTES at the rate above. That is half
#: the embed timer's 30-minute period, so an ungraceful death costs at most half
#: a tick — the largest loss that still reads as "the next tick catches up"
#: rather than as work thrown away. There is no throughput argument for a bigger
#: number: one SELECT plus one commit is a few hundred milliseconds against 900
#: seconds of encoding, i.e. well under 0.1% overhead, and
#: ``docs/embedding-throughput.md`` measured enlarging the model's own batch at
#: under 10% and negative past 8, so nothing is being amortised here either.
#:
#: A CONSTANT AND NOT A FLAG, like ``_MAX_STALLED_BATCHES`` beside it. It is
#: derived from a measured rate rather than from a preference: raising it
#: re-creates the defect it exists to close, and lowering it buys nothing the
#: overhead arithmetic above has not already given away.
_MAX_BATCH_CHUNKS = 45

#: How many chunks may go to the model in ONE ``embed_documents`` call.
#:
#: THIS IS WHAT MAKES A STOP REACHABLE. The stop flag is read between calls and
#: nowhere else, so the largest call is the longest a SIGTERM can go unanswered.
#: A whole row used to be one call: measured read-only against the live cache at
#: ``chunk-4000-400``, 1,348 rows exceed 300 s that way and the largest is 257
#: chunks — about 86 minutes — against a ``TimeoutStopSec`` of 5 minutes. So
#: systemd escalated to SIGKILL with the claim held, and the next run's
#: ``_blame_crashed_row`` booked a good row as poison; three of those and the
#: row is terminal, permanently absent from the vector arm.
#:
#: FOUR, which is ~80 seconds at ``_SECONDS_PER_CHUNK``. That leaves the rest of
#: the 300-second window for the flush and commit that follow, with room to
#: spare on a machine under load — the number is chosen against the window it
#: has to fit inside, not for its own sake.
#:
#: AND IT COSTS ESSENTIALLY NOTHING, which is why it is not a larger number
#: hedged with a comment. ``docs/embedding-throughput.md`` measured enlarging
#: the model's own batch at under 10% and NEGATIVE past 8, so four is already
#: near the flat part of that curve; the throughput given up is a rounding
#: error against a backfill measured in weeks, and it buys the difference
#: between a stop that works and one that loses a row.
#:
#: NOT A FLAG, for the same reason as the two constants above: raising it
#: re-opens the gap it exists to close.
_MAX_CHUNKS_PER_ENCODE = 4


def _embedder_is_healthy(embedder: Embedder) -> bool:
    """THE TRANSIENT/PERMANENT DISCRIMINATOR, asked at the moment of failure.

    "Was that the row's fault or the machine's?" cannot be answered from the
    exception: a model that has not finished loading, an allocator that ran out
    of memory, and a body that deterministically kills the tokenizer all arrive
    as ``RuntimeError`` from ``sentence-transformers``. Sniffing types would be
    a guess, and a wrong guess in either direction is expensive — condemn a
    transient and the row is lost from the index, excuse a permanent and the
    backlog never drains.

    So ask the question directly, by re-running the same call on input that is
    known to be fine. If the probe embeds, the embedder works and the failure
    discriminated between two bodies, which makes it a property of the ROW. If
    the probe fails too, the failure discriminates between nothing, so nothing
    may be blamed on a record: the caller aborts, marks no row, holds no row,
    and leaves the backlog exactly where it was.

    That covers each named transient exactly. A cold model fails the probe. An
    OOM fails the probe. An I/O blip on the model files fails the probe. A body
    that kills the tokenizer does not.

    It is not asked to cover everything, and it does not have to: the second
    axis is reproducibility. A row that passes this test is HELD with backoff,
    not condemned — a large row that OOMed while the small probe fit is
    attempted again later and released the moment it succeeds. Only failing
    ``POISON_MAX_ATTEMPTS`` times across separate runs, minimum fifteen minutes
    apart, makes a row terminal, and by then "deterministic" is evidence rather
    than inference.

    Costs one short embed per FAILING row and nothing at all on a healthy run.
    Deliberately not memoised across a batch: an embedder that dies partway
    through must be caught then, not excused by a probe that passed earlier.
    """
    try:
        embedder.embed_documents([_EMBED_HEALTH_PROBE])
    except Exception:
        # Including the probe's own unexpected failures. Every unknown here
        # resolves to "do not attribute", which is the direction that costs a
        # re-read rather than a permanently dropped row.
        return False
    return True


class EmbedStoreUnavailableError(RuntimeError):
    """The CACHE failed mid-batch. Not a bad row, and not the embedder either.

    Its own type so the run reports an environment fault in the vocabulary of
    an environment fault. The sibling of :class:`EmbedderUnhealthyError`, for
    the half of the try block round 2's probe reasoning never covered.
    """


def _embed_store_unavailable(
    kind: str, error: BaseException
) -> EmbedStoreUnavailableError:
    """The message an operator gets when the cache, not the corpus, failed.

    It has to say the row was NOT blamed, because the whole failure mode is a
    good row being condemned for a transient lock and quietly leaving the
    index.

    IT MUST NO LONGER BLAME THE INGEST TIMER, and that correction is the point
    of this revision. It used to read "the usual cause is another writer
    holding the database — the ingest timer runs every 30 minutes and a long
    run overlaps the embed tick", which was true when it was written and is
    now the one thing this cannot be: aggregator's writers queue for
    ``Store._CacheWriteLock`` before touching SQLite, so an overlap makes embed
    WAIT, not fail. Pointing a human at the ingest timer would now send them to
    read a schedule that is working as designed.

    What is left when a lock still surfaces here is a writer outside that
    scheme, and the two real ones are worth naming because they need opposite
    responses: a half-deployed upgrade, where one unit is running a build that
    predates the lock and therefore does not queue, and a human holding a
    transaction at an interactive ``sqlite3`` prompt.
    """
    return EmbedStoreUnavailableError(
        f"aggregator embed stopped: the cache could not be written while "
        f"embedding {kind} "
        f"({type(error).__name__}: {error}). That is the STORE failing, not "
        f"bad data and not the embedder, so NO row was blamed for it, nothing "
        f"was added to the poison ledger, and every row still unembedded "
        f"stays in the backlog exactly where it was. The next run resumes from "
        f"the watermark. This should be RARE: aggregator's own writers queue "
        f"for an OS lock on <cache>.write.lock and wait for each other, so a "
        f"normal ingest/embed overlap can no longer cause this. Look instead "
        f"for a writer outside that scheme — a half-finished deploy leaving one "
        f"unit on a build older than the lock, or an interactive `sqlite3` "
        f"session holding a transaction open."
    )


class EmbedWorkerKilledError(RuntimeError):
    """A previous worker process died on a row without raising anything.

    Its own type so the ledger entry, and therefore ``aggregator status``,
    names the failure mode rather than showing a generic RuntimeError next to
    genuinely different faults.
    """


def _blame_crashed_row(
    store: Store, ledger: PoisonLedger, outcome: _EmbedOutcome
) -> None:
    """Set aside the row a previous process died on, if there is one.

    THE WEDGE THIS EXISTS TO BREAK. Chunk N's isolation depends on catching an
    exception: the row is held, marked ``'error'``, and the backlog drains
    past it. A row that OOM-kills the worker, or segfaults inside a native
    tokenizer, raises nothing at all — no handler runs, no ledger entry is
    written, ``embedding_state`` stays NULL, and ``select_unembedded``'s
    ``ORDER BY ts DESC`` hands the identical row to the next tick. Twice an
    hour, forever, and the only external symptom is ``vector_index`` counts
    that stop moving, which is not something a human watches for a month.

    A ROUTINE SHUTDOWN NEVER GETS HERE. SIGTERM and SIGINT are handled in
    ``_cmd_embed``: the row in flight finishes, its claim is released, and the
    run stops at a boundary. So a claim that survives is one no handler could
    run for — a SIGKILL, an OOM kill, a segfault — which is what makes the
    claim evidence of a crash rather than evidence of a reboot. Before that
    handler existed, every `systemctl stop` condemned whichever good row was
    in flight, twice an hour on a month-long backfill.

    Routed through the SAME ledger as every other per-row failure, so the
    residual case — a SIGTERM the worker could not act on before
    ``TimeoutStopSec`` escalated it to SIGKILL — costs one delayed row rather
    than a condemned one: the row is held with backoff, comes back when it is
    due, and only becomes terminal after ``POISON_MAX_ATTEMPTS`` sightings at
    least fifteen minutes apart. A row that reliably kills the worker stops
    being attempted.
    """
    claim = store.pending_embed_claim()
    if claim is None:
        return
    kind, row_id = claim
    source = _embed_ledger_source(kind)
    error = EmbedWorkerKilledError(
        f"a previous `aggregator embed` process died while embedding {kind} "
        f"row {row_id!r} — no exception was raised, so this is a kill "
        f"(OOM, segfault in a native extension, or an external SIGKILL) "
        f"rather than bad data the worker could catch. The row is set aside "
        f"so the backlog can drain past it."
    )
    previous = ledger.entries(source).get(row_id)
    held = ledger.hold(source, row_id, error, previous=previous)
    # ``expected=None`` said out loud: this makes no claim about the row's
    # content, only that a previous process died on it.
    store.mark_embedded(kind, [row_id], state="error", expected=None)
    store.release_embed_claim()
    if previous is None:
        outcome.new_failures.append(
            f"embed: {kind} row {row_id!r} killed a previous worker process "
            f"and was set aside — {held.error_detail}"
        )
    else:
        outcome.known_failures += 1


def _embed_ledger_source(kind: str) -> str:
    """What the embed worker calls itself in the quarantine ledger.

    Namespaced per ontology so an ``obs_id`` and a ``stable_id`` that happen to
    collide cannot inherit each other's attempt count, and so the line
    ``aggregator status`` prints says which worker set the row aside rather
    than colliding with an ingest source of the same name.
    """
    return f"embed:{kind}"


def _embed_plan(source_arg: str) -> list[tuple[str, str]]:
    """The ``(kind, source)`` groups this run drains, IN PRIORITY ORDER.

    THE USER'S 2026-08-21 DIRECTIVE, executed. ``dropbox -> blog -> llm ->
    claude code``, then the sources they did not rank. The worker used to loop
    over ontologies — all observations, then all records — and no ordering
    inside ``select_unembedded`` could have produced the user's sequence,
    because it cuts across the ontologies: two records sources come before
    every observation and four more come after. So the loop is over the plan
    and the ontology is a property of each step.

    That matters at the scale this runs at. The full backfill is a measured
    25-30 days of continuous CPU; under the old loop dropbox — the source the
    user put FIRST — queued behind 505k observations, which is weeks.

    ``--source observations|records`` still narrows to one ontology, and
    narrowing does not flatten: the surviving groups keep their relative order.
    """
    if source_arg == "both":
        return list(EMBED_BACKLOG_ORDER)
    return [(k, s) for k, s in EMBED_BACKLOG_ORDER if k == source_arg]


def _report_embed_progress(store: Store, out=None) -> None:
    """Print how far each source has got. The answer to the question asked.

    "Which sources are fully embedded" is what a user wants to know about a
    multi-week backfill, and a global percentage cannot answer it: one "62%"
    is compatible with dropbox untouched and with dropbox finished, which are
    opposite answers to "can I search my notes yet".

    EVERY GROUP IS LISTED, INCLUDING THE EMPTY ONES. A source holding no rows
    and a source fully embedded return the same zero vector hits for every
    query, so ``empty`` is printed as its own word rather than rolled into
    ``complete`` or omitted. Same reason the states are the ones
    ``Store.vector_index_state`` already uses.

    Two grouped queries per ontology — measured against a snapshot of the live
    cache at 0.27 s warm and 4.3 s on a cold page cache — so a 30-minute timer
    can afford it once per run, at the end, where it costs nothing the embed
    pass has not already paid. Best-effort: a progress display must never be
    the thing that fails a run which embedded rows successfully.
    """
    out = out or sys.stdout
    try:
        rows = store.embed_progress_by_source()
    except Exception as e:  # noqa: BLE001 - reporting must not fail the run
        print(f"embedding progress unavailable: {e}", file=sys.stderr)
        return
    print("embedding progress by source (highest priority first):", file=out)
    for row in rows:
        print(
            f"  {row['source']}: {row['state']} — {row['embedded']}/{row['total']} "
            f"embedded, {row['pending']} pending, {row['skipped']} nothing to "
            f"embed, {row['errors']} held ({row['kind']})",
            file=out,
        )


def _requeue_due_rows(store: Store, ledger: PoisonLedger, kind: str) -> None:
    """Put back the rows whose backoff has expired. ONCE PER RUN, PER ONTOLOGY.

    A row that failed left the backlog under ``embedding_state = 'error'``, so
    ``select_unembedded`` cannot see it and no amount of draining would ever
    try it again. Putting the due ones back is what makes the hold a RETRY
    rather than a deletion.

    IT LIVES HERE, ABOVE THE PRIORITY WALK, and that placement is the whole
    reason it is its own function. It used to sit at the top of
    ``_embed_backlog``, which was called once per ontology; the walk calls that
    up to eight times per run, so leaving it there would have re-asked the
    ledger for due rows in the middle of the run that set some of them aside.
    ``PoisonLedger.due`` filters on ``next_attempt_at``, so nothing would
    actually have been requeued early — but "a row cannot be requeued into the
    same run that just held it" would have gone from a structural guarantee to
    a coincidence of the backoff being longer than one pass.
    """
    due = ledger.due(_embed_ledger_source(kind))
    if due:
        store.requeue_embedding(kind, sorted(due))


def _embed_row_text(kind: str, row: sqlite3.Row) -> tuple[str, str]:
    """The id, and the EXACT text the model will be handed, for one backlog row.

    ONE DEFINITION, TWO CALLERS, which is the whole reason it is a function.
    ``_embed_batch`` encodes this string; ``_row_chunk_cost`` sizes the batch
    from it. If the two ever disagreed the cap would be enforced against text
    nobody embeds. Records are the trap: they go to the model as
    ``subject + "\\n\\n" + body``, so a records row costs more chunks than its
    body alone, and a cost function reading ``body`` would under-count every
    record in the corpus.
    """
    if kind == "observations":
        return row["obs_id"], (row["body"] or "")
    return row["stable_id"], f"{row['subject']}\n\n{row['body']}"


def _row_chunk_cost(kind: str, row: sqlite3.Row) -> int:
    """What this row will cost the encoder, in chunks. COUNTED, not estimated.

    By actually chunking it, because an estimate that can under-count does not
    bound anything. ``chunk_body`` has two branches — greedy paragraph packing
    when every paragraph fits the window, hard windowing otherwise — and a
    length-based estimate describes only the second. On prose whose paragraphs
    run just over half a window, packing yields one chunk per paragraph while
    the arithmetic predicts roughly half that: a ~2x under-count, and a cap that
    can be exceeded twofold is not a cap. The price of counting exactly is
    string slicing, against ~20 seconds per chunk of encoding. It does not
    register.

    A ROW THAT CANNOT BE COUNTED COSTS 1, rather than raising. Every per-row
    failure in this file is caught inside ``_embed_batch`` precisely so that one
    bad row costs one row, and moving ``chunk_body`` earlier must not move a
    raise outside that isolation — a chunker that threw here would abort the
    whole run on a traceback, which is the shape of the 2026-08-15 doom loop.
    Costing it 1 is honest as well as safe: a row this cannot chunk is a row the
    worker cannot embed either, so it will spend no encoder time. It joins the
    batch, fails there, and is attributed there, exactly as before.
    """
    try:
        return len(chunk_body(_embed_row_text(kind, row)[1]))
    except Exception:  # noqa: BLE001 - attribution belongs to ``_embed_batch``
        return 1


def _pack_batch(kind: str, rows: list, chunk_cap: int) -> list:
    """The longest PREFIX of ``rows`` whose chunks fit under ``chunk_cap``.

    The cap is counted in CHUNKS, which the name says because a ROW count —
    ``args.batch_size`` — is in scope at the only call site and would type-check
    here without complaint. Swapping the two would silently change what a
    checkpoint interval is made of, and nothing downstream would notice.

    A PREFIX, NOT A SELECTION. ``select_unembedded`` hands back an ordered
    backlog — ``ts DESC``, within one group of ``EMBED_BACKLOG_ORDER`` — and
    stepping over a row that does not fit, to reach smaller ones behind it,
    would let a large row be passed over for as long as small ones keep
    arriving. Stopping at the first row that does not fit leaves it at the head
    of the next SELECT, so the queue always moves forward and nothing can be
    perpetually deferred.

    THE FIRST ROW IS ALWAYS TAKEN, whatever it costs, and that exemption is
    load-bearing in two directions at once.

    A row is indivisible here. Committing part of its chunks would write
    ``chunk_embeddings`` rows under its ``owner_id``, and ``select_unembedded``
    is a LEFT JOIN against that table — the row would read as embedded and
    nothing would ever come back for the rest of it. So the cap cannot be
    honoured by splitting; it can only be honoured by refusing, and refusing is
    the worse bug.

    And an empty batch writes nothing, which ``_embed_backlog`` reads as a
    stall. Three of those end the run, and ``ts DESC`` puts the same oversized
    row at the head of the next SELECT — so a packer that refused it would turn
    the cap into a row filter that strands that row and everything queued behind
    it, forever, while every count reports the index simply not filling. The cap
    bounds how many rows are GROUPED. It is not a statement about which rows are
    eligible.

    Rows with nothing to embed cost 0 and are therefore never rationed by this.
    Deliberate: about a third of the corpus has an empty body
    (``docs/embedding-throughput.md``), those rows are marked ``'skip'`` and
    spend no encoder time at all, and bounding them is what ``--batch-size`` is
    still for.
    """
    batch: list = []
    total = 0
    for row in rows:
        cost = _row_chunk_cost(kind, row)
        if batch and total + cost > chunk_cap:
            break
        batch.append(row)
        total += cost
    return batch


def _embed_backlog(
    store: Store,
    embedder: Embedder,
    kind: str,
    args: argparse.Namespace,
    ledger: PoisonLedger,
    outcome: _EmbedOutcome,
    stop: Callable[[], bool] | None = None,
    source: str | None = None,
) -> bool:
    """Drain one backlog group in bounded batches. Returns WHETHER IT WORKED.

    Chunked and checkpointed for the same reason ingest is: each batch commits
    its vectors and its watermark before the next one starts, so a kill at any
    moment costs at most one batch and the next run resumes where this stopped.

    AND THE BATCH IS BOUNDED BY CHUNKS, which is what makes "at most one batch"
    a number rather than a shrug. ``--batch-size`` bounds the SELECT, but rows
    differ in chunk count by two orders of magnitude and the encoder is billed
    per chunk, so a row bound alone left the checkpoint interval to whatever the
    backlog happened to hold — measured at ~6.4 hours on the first batch of
    dropbox. ``_pack_batch`` trims the SELECT's result to a prefix that fits
    ``_MAX_BATCH_CHUNKS``.

    ``source`` NAMES ONE GROUP OF ``EMBED_BACKLOG_ORDER``. Resumability inside a
    group needs nothing extra: the backlog is a LEFT JOIN, so a run killed
    halfway through dropbox asks the same question next time and gets what it
    had not reached. Mid-source resume and between-source resume are the same
    mechanism.

    THE RETURN VALUE IS FOR ``--once``. That flag means one batch, and the walk
    visits up to eight groups — so the caller has to be able to tell "this
    group did a batch" from "this group was already empty". Without it, --once
    either does eight batches or stops dead at the first finished source and
    never reaches the ones behind it.
    """
    stalls = 0
    worked = False
    while True:
        if stop is not None and stop():
            outcome.interrupted = True
            return worked
        rows = store.select_unembedded(kind, limit=args.batch_size, source=source)
        if not rows:
            return worked
        # BOUNDED TWICE, because the two bounds answer different questions.
        # ``--batch-size`` bounds ROWS, which is what keeps a batch of rows that
        # cost the encoder nothing — empty bodies, about a third of the corpus —
        # from being unbounded. ``_MAX_BATCH_CHUNKS`` bounds CHUNKS, which is
        # what the interval between durable checkpoints is actually made of. The
        # SELECT can only express the first: chunk count is not a column and
        # cannot be one, since it depends on the chunker's geometry. So the
        # second is applied here, to the SELECT's own result.
        batch = _pack_batch(kind, rows, chunk_cap=_MAX_BATCH_CHUNKS)
        moved = _embed_batch(store, embedder, kind, batch, ledger, outcome, stop)
        worked = True
        if args.once:
            return worked
        # THE TERMINATION ARGUMENT. Every other exit from this loop is "the
        # backlog is empty"; this one is "the backlog is not emptying".
        #
        # Until round 2's S4 the first was enough, because a batch always
        # emptied itself — each row left as 'ok', 'skip' or 'error', so the
        # next SELECT could not return it. S4 made the writes a
        # compare-and-swap, and a batch whose CAS all fails writes NOTHING:
        # every row stays at NULL and the identical batch is selected again.
        #
        # In practice each pass re-reads a fresh ``src_hash``, so one failed
        # CAS self-corrects on the next pass and a real spin needs rewrites
        # landing faster than the worker embeds. Rare — and not the point. The
        # 2026-08-16 shape requires each chunk to commit a checkpoint, which
        # means every pass must either advance the watermark or end the run.
        # A loop that cannot say why it stops is one that can fail to.
        #
        # RE-DERIVED FOR THE CHUNK BOUND, not assumed to carry over, because
        # that argument turns on "the identical batch is selected again" and
        # ``_pack_batch`` changed what the batch IS. It still holds: packing is
        # a pure function of the rows the SELECT returned and a constant cap,
        # and a zero-write pass leaves every one of those rows exactly where it
        # found them — so the next SELECT returns the same rows and the same
        # prefix comes out. It also cannot degenerate into a stall of its own
        # making: the packer never returns an empty batch for a non-empty
        # SELECT, so every pass still attempts at least one row and a stall
        # remains evidence about the CAS rather than about the packing.
        #
        # THREE IS STILL THE RIGHT NUMBER, and for a slightly sharper reason
        # than before. Batches are smaller now, so a single concurrent rewrite
        # can zero a whole one where it used to be diluted across 500 rows —
        # which makes a stall MORE informative, not less. ``ts DESC`` means
        # three consecutive zero-write passes are three failures on the same
        # head row, i.e. a writer holding that row, which is exactly the
        # condition this bound exists to stop spinning on.
        if moved:
            stalls = 0
            continue
        stalls += 1
        if stalls >= _MAX_STALLED_BATCHES:
            outcome.stalled = True
            return worked


def _chunk_sha(text: str) -> str:
    """The content address of one chunk: sha256 of the exact bytes encoded.

    OF THE CHUNK, NOT OF THE ROW. ``src_hash`` already fingerprints the row and
    answers a different question ("did the body move under the worker?"). This
    one answers "have these exact bytes been through the model already", and it
    has to describe what the encoder saw or a reuse would hand back a vector
    for different text.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _embed_batch(
    store: Store,
    embedder: Embedder,
    kind: str,
    rows: list,
    ledger: PoisonLedger,
    outcome: _EmbedOutcome,
    stop: Callable[[], bool] | None = None,
) -> int:
    """Embed one batch, then advance the watermark — IN THAT ORDER.

    RETURNS HOW MANY ROWS LEFT THE BACKLOG, which is what makes ``--catchup``'s
    loop terminating rather than merely usually-terminating. Since round 2's S4
    the writes are a compare-and-swap, so a batch can legitimately write
    nothing at all and leave every row exactly where it found it — and the next
    SELECT then returns that same batch. See ``_MAX_STALLED_BATCHES``.

    Vectors are written before ``embedding_state`` moves. The reverse order is
    the one that loses rows: a crash between the two would leave rows marked
    embedded that nothing will ever come back for.

    A row with no chunks (empty or whitespace-only body — roughly half the
    corpus sits in that bucket) is marked ``'skip'`` rather than ``'ok'``. It
    has to leave the backlog or the worker re-reads it every run forever, but
    "nothing to embed" and "embedded" are different facts and the table keeps
    them apart.

    ONE BAD ROW COSTS ONE ROW. Every per-row failure is caught here, and the
    reason it must be is the 30-minute timer: ``select_unembedded`` orders
    ``ts DESC``, so a row that aborted the run was the first row of the next
    run too, forever. That is the 2026-08-15 ingest doom loop with a different
    query at the bottom of it. A caught failure is held in the quarantine
    ledger, marked ``'error'`` so the backlog can drain past it, and — if this
    is the first time — named on stderr.

    THE LEDGER IS WRITTEN BEFORE THE MARK, and that order is the safe one. A
    kill in between leaves a ledger entry for a row still sitting at NULL: it
    is attempted again next run and its attempt count runs one high, which
    costs one embed. The reverse leaves a row marked ``'error'`` with nothing
    holding it, which no ``due`` set would ever requeue — a row silently gone
    from the index with no record of why, i.e. the failure this whole file is
    about.

    ``except Exception``, never ``BaseException``: a ``KeyboardInterrupt``, a
    ``SystemExit`` or a test's deliberate ``BaseException`` sentinel means stop,
    not "this row is poison".

    THE STOP CHECK IS PER ROW, not per batch, and the granularity is the
    point. A batch is up to ``_MAX_BATCH_CHUNKS`` chunks of encoding — about a
    quarter of an hour — and a ``TimeoutStopSec`` is far shorter than that, so
    "finish the batch in flight" would still be SIGKILLed partway — leaving
    the claim that makes the next run condemn a good row, which is the whole
    thing being fixed. One row is the unit of work here and the unit the claim
    covers, so it is the boundary. Whatever embedded before the stop is
    flushed and committed below; the rest stays in the backlog.

    AND IT IS READ INSIDE THE ROW TOO, which is what makes the row boundary an
    honest promise rather than an aspiration. A row used to be a single
    ``embed_documents`` call over all of its chunks, so the flag was not read
    again until that call returned: measured read-only against the live cache,
    1348 rows exceed 300 s that way and the largest is 257 chunks — about 86
    minutes — against a 5-minute ``TimeoutStopSec``. A stop landing inside one
    of those was SIGKILLed with the claim held, and ``_blame_crashed_row`` then
    attributed it to a good row. The encode is sliced at
    ``_MAX_CHUNKS_PER_ENCODE`` now, so the longest a stop can go unanswered is
    ~80 seconds regardless of how long the row is.

    A ROW INTERRUPTED PARTWAY IS NOT MARKED AND DOES NOT COMMIT. The row stays
    the unit that commits even though it is no longer the unit that encodes:
    chunks written under a partially-embedded row's ``owner_id`` would make
    ``select_unembedded``'s LEFT JOIN read it as done, and the rest of it would
    never be asked for again. It stays at NULL and the next run re-embeds it
    whole — one row of repeated work, against a row silently missing from the
    index.

    A ROW THAT MOVED WHILE THE WORKER HELD IT IS NOT MARKED. Ingest sets
    ``embedding_state`` back to NULL and drops a row's vectors whenever its
    body changes — that is the only thing that ever re-embeds an edited row.
    Interleave the two and the worker undoes it: it embeds the OLD body,
    ingest invalidates the row, and the worker then upserts the old vector
    back and marks the row ``'ok'``. ``select_unembedded`` only looks at NULL,
    so nothing comes back for it until the body is edited AGAIN, and until
    then the vector arm answers with text the row no longer contains while the
    keyword arm (which has a trigger) answers with the text it does.

    The fingerprint the batch is judged against therefore comes from
    ``select_unembedded``'s OWN row set, alongside the body, and the comparison
    happens inside the writes themselves — see ``Store.commit_embed_batch``.
    Re-reading it here instead left a window: an edit landing between the
    SELECT and the re-read makes both snapshots show the new hash, so the row
    reads as untouched while the text in hand is stale. Rows that moved are
    counted ``superseded`` rather than failed, because they stay at NULL and
    the next run embeds them from their current body — it self-heals, and a
    ledger entry would only make ``aggregator status`` overstate the damage.
    """
    source = _embed_ledger_source(kind)
    entries = ledger.entries(source)
    id_key = "obs_id" if kind == "observations" else "stable_id"
    # The body and its fingerprint out of one statement, so "unchanged since
    # the worker took it" is a claim the SELECT itself can support.
    expected = {r[id_key]: r["src_hash"] for r in rows}
    ok_ids: list[str] = []
    skip_ids: list[str] = []
    error_ids: list[str] = []
    all_vecs: list[tuple[str, str, Any]] = []
    #: ``chunk_id -> content_sha256``, handed to the store so a later run can
    #: prove the same bytes were already embedded. See ``reusable_chunk_vectors``.
    chunk_hashes: dict[str, str] = {}
    #: ``content_sha256 -> vector`` for text this batch already has an answer
    #: for, whether from the index on disk or from an earlier row of this same
    #: batch. THE ONLY LEVER THAT MOVES THE BACKFILL. Measured on this hardware
    #: the encoder runs at ~40 tokens/second, so one 4000-character chunk is
    #: ~19 seconds and a lookup that avoids one is worth ~19 seconds; batching,
    #: by contrast, was measured at under 10% and negative past batch 8
    #: (``docs/embedding-throughput.md``). Chat transcripts repeat themselves
    #: constantly — quoted text, re-pasted tool output, the same file read
    #: twice — and every repeat is a chunk that does not have to be computed.
    known: dict[str, Any] = {}
    unhealthy: EmbedderUnhealthyError | None = None
    store_fault: sqlite3.Error | None = None
    for row in rows:
        if stop is not None and stop():
            outcome.interrupted = True
            break
        # THROUGH THE SAME HELPER THE PACKER USED. The batch was sized from
        # this exact string; reading the body a second way here would let the
        # cap describe text that never reaches the model.
        row_id, body = _embed_row_text(kind, row)
        try:
            chunks = chunk_body(body)
            if not chunks:
                skip_ids.append(row_id)
                continue
            hashes = [_chunk_sha(text) for text in chunks]
            # ASK THE INDEX BEFORE ASKING THE MODEL. One indexed SELECT per
            # row, against ~19 seconds per chunk it can save. Scoped to this
            # model by the store — a hash match under a different model is the
            # same text in a different embedding space, and reusing it would be
            # exactly the silent mixing the (chunk_id, model) key exists to
            # prevent.
            unknown = [h for h in hashes if h not in known]
            if unknown:
                known.update(store.reusable_chunk_vectors(unknown))
            missing = [i for i, h in enumerate(hashes) if h not in known]
            if missing:
                # WRITTEN AND COMMITTED BEFORE THE ATTEMPT. Everything else in
                # this handler needs an exception; a row that OOM-kills or
                # segfaults the process raises nothing, so the claim on disk is
                # the only trace that survives. See ``Store.claim_embed_row``.
                #
                # Only when there is something to attempt: a row whose every
                # chunk was reused never reaches the model, so there is no
                # crash to attribute and claiming it would be a lie about what
                # this process was doing.
                store.claim_embed_row(kind, row_id)
                # SLICED, SO THE STOP FLAG IS REACHABLE INSIDE THE ROW. This
                # used to be one ``embed_documents`` call over every chunk of
                # the row, and the flag was not read again until it returned —
                # which made "the stop check is per row" a promise the longest
                # rows could not keep. See ``_MAX_CHUNKS_PER_ENCODE``.
                interrupted_mid_row = False
                for at in range(0, len(missing), _MAX_CHUNKS_PER_ENCODE):
                    if stop is not None and stop():
                        interrupted_mid_row = True
                        break
                    part = missing[at : at + _MAX_CHUNKS_PER_ENCODE]
                    fresh = embedder.embed_documents([chunks[i] for i in part])
                    for i, vec in zip(part, fresh, strict=False):
                        known[hashes[i]] = vec
                store.release_embed_claim()
                if interrupted_mid_row:
                    # THE ROW IS PUT DOWN, NOT PUT AWAY. It is not appended to
                    # ``ok_ids`` and nothing of it reaches ``all_vecs``, so
                    # ``commit_embed_batch`` never sees a partial row: chunks
                    # written under this ``owner_id`` would make
                    # ``select_unembedded``'s LEFT JOIN read it as embedded and
                    # the missing chunks would be unreachable forever. It stays
                    # at NULL and the next run re-embeds it from scratch.
                    #
                    # The slices that DID encode stay in ``known``, which is
                    # not a partial commit: that dict is keyed by content hash,
                    # so it only ever writes a chunk on behalf of a row that
                    # completes. If another row in this batch holds the same
                    # text, the work is still saved.
                    outcome.interrupted = True
                    break
            vecs = [known[h] for h in hashes]
        except sqlite3.Error as e:
            # THE STORE FAILED, NOT THE ROW — round 3's M3. Two of the three
            # calls above are writes to the cache, and the ingest timer can
            # lock it at any moment. ``_embedder_is_healthy`` cannot see that:
            # it probes the EMBEDDER, which is working perfectly, so the
            # handler below concluded the fault discriminated between bodies
            # and blamed whichever row was in flight. Reproduced: a single
            # `database is locked` put all three rows of the batch in the
            # poison ledger, each with an attempt count, on their way to
            # becoming terminal after POISON_MAX_ATTEMPTS — permanently out of
            # the vector arm because two writers overlapped once.
            #
            # A lock discriminates between nothing, exactly like a cold model
            # or an OOM, so it gets the same answer round 2 gave those: abort,
            # blame nobody, leave the backlog untouched.
            #
            # AND DISOWN THE ROW ON THE WAY OUT — 2026-08-27, observed in
            # production four times in three hours. "Blame nobody" was true of
            # this run and false of the next one: the claim written before the
            # encode was still on disk, so the following run's
            # `_blame_crashed_row` read it as a kill and booked a row that had
            # embedded perfectly well into the poison ledger. The intent was
            # defeated one function call away.
            #
            # This is reachable now because the claim lives beside the
            # database rather than inside it (`Store.embed_claim_path`), so
            # releasing it does not need the lock that just failed.
            # `release_embed_claim` never raises, by construction, so this
            # cannot become one more way to leave a claim behind.
            store.release_embed_claim()
            store_fault = e
            break
        except Exception as e:
            # NO GUARD AROUND THE RELEASE ANY MORE, and its absence is the
            # point. This used to catch `sqlite3.Error` here because the
            # release was a database write and could fail for the very reason
            # the row had — so the run aborted to avoid leaving a claim it
            # could not clear. The claim is a file beside the database now
            # (`Store.embed_claim_path`) and `release_embed_claim` swallows
            # its own errors, so there is no failure left to route: the row is
            # disowned before anything else is decided about it.
            store.release_embed_claim()
            if not _embedder_is_healthy(embedder):
                unhealthy = EmbedderUnhealthyError(
                    f"aggregator embed stopped after {kind} row {row_id!r} failed "
                    f"({type(e).__name__}: {e}) and the embedder then could not "
                    f"embed a known-good probe string either. That is an "
                    f"environment fault — a model that has not loaded, an OOM, "
                    f"an I/O blip — and not bad data, so that row was NOT "
                    f"blamed for it and every row still unembedded stays in "
                    f"the backlog. Fix the model and re-run "
                    f"`aggregator embed --catchup`."
                )
                break
            held = ledger.hold(source, row_id, e, previous=entries.get(row_id))
            error_ids.append(row_id)
            if row_id in entries:
                outcome.known_failures += 1
            else:
                outcome.new_failures.append(
                    f"embed: {kind} row {row_id!r} could not be embedded and was "
                    f"set aside — {held.error_detail}"
                )
            continue
        for i, vec in enumerate(vecs):
            chunk_id = row_id if len(chunks) == 1 else f"{row_id}:{i}"
            all_vecs.append((row_id, chunk_id, vec))
            chunk_hashes[chunk_id] = hashes[i]
        ok_ids.append(row_id)
        if row_id in entries:
            # It used to fail and does not any more. A fault that no longer
            # reproduces describes no gap, and leaving the row behind would
            # have ``aggregator status`` overstating the damage forever.
            ledger.release(source, row_id)
            outcome.released += 1
    # The vectors and the watermark, in one guarded transaction. Anything the
    # guard rejected moved underneath the worker and is reported as such.
    #
    # ATTEMPTED EVEN AFTER A STORE FAULT, because the rows that embedded before
    # the lock are real work and discarding them means re-doing it next tick —
    # which the 2026-08-16 rules call a bug even when the answer comes out
    # right. If the cache is still locked this raises too, and it becomes the
    # fault that is reported.
    try:
        written_ok, written_skip = store.commit_embed_batch(
            kind,
            vectors=all_vecs,
            ok_ids=ok_ids,
            skip_ids=skip_ids,
            error_ids=error_ids,
            expected=expected,
            hashes=chunk_hashes,
        )
    except sqlite3.Error as e:
        raise _embed_store_unavailable(kind, store_fault or e) from e
    outcome.superseded += (len(ok_ids) - len(written_ok)) + (
        len(skip_ids) - len(written_skip)
    )
    # ``error_ids`` count: they are marked unguarded, so they always leave the
    # backlog, and a batch that only set rows aside has still made progress.
    moved = len(written_ok) + len(written_skip) + len(error_ids)
    if unhealthy is not None:
        # The rows that DID embed before the model died are flushed above and
        # keep their vectors: a run that threw away completed work would be
        # re-doing it on the next tick, which the ingest rules call a bug even
        # when the answer comes out right.
        raise unhealthy
    if store_fault is not None:
        raise _embed_store_unavailable(kind, store_fault)
    return moved


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
    q.add_argument(
        "--fields",
        choices=["summary", "full"],
        # Sentinel, not "summary": ``_cmd_query`` has to tell an omitted
        # --fields from one typed with the default value, because --rerank
        # supplies the former and refuses the latter.
        default=_FIELDS_UNSET,
        help=(
            "how much of each hit to print: 'summary' (subject and metadata "
            "only — the default) or 'full' (document bodies too). Left off, "
            "--rerank raises this to 'full'"
        ),
    )
    q.add_argument("--page-size", type=int, default=50)
    q.add_argument(
        "--page-token",
        default=None,
        help=(
            "continue from a previous page: pass back the value this command "
            "printed as '# next_page_token:' (or the JSON field of the same "
            "name)"
        ),
    )
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
    q.add_argument(
        "--rerank",
        action="store_true",
        help=(
            f"re-order the head of the page by cross-encoder relevance instead "
            f"of recency. THE HEAD ONLY: the first {_RERANK_WINDOW} hits are "
            f"scored and reordered, and anything after them stays in recency "
            f"order — so at the default --page-size 50 most of the page is NOT "
            f"ranked, and the output marks where the ranked part ends. Pass "
            f"--page-size {_RERANK_WINDOW} or less to have the whole page "
            f"ranked. SLOW — 273 s median and 304 s at p95 per query on this "
            f"CPU, about four and a half minutes, against 0.65 s without, "
            f"plus a one-off model load; this command is the "
            f"batch surface that cost is documented for. Refuses out loud if "
            "the weights cannot be loaded rather than returning an unranked "
            "page. IMPLIES --fields full when --fields is not given, because "
            "the cross-encoder ranks document bodies and summary mode returns "
            "none — so this also CHANGES THE OUTPUT: full bodies are printed "
            "under each hit, not subject lines alone. Passing --fields summary "
            "explicitly is refused rather than silently overridden"
        ),
    )

    st = sub.add_parser("status", help="print capabilities / freshness")
    st.add_argument("--json", action="store_true")

    rr = sub.add_parser(
        "retrieval-regression",
        help=(
            "freeze a retrieval baseline, or re-run it and report drift "
            "against the frozen one"
        ),
        description=(
            "THE ORDER IS THE TOOL. Freeze BEFORE changing retrieval, run "
            "AFTER, and the diff between the two is the evidence. Freeze "
            "afterwards and you have baselined the bug. Exit 0 clean, 1 a "
            "regression that needs no labels (a negative query stopped "
            "abstaining, or mean drift above an explicit --drift-threshold), "
            "2 the harness could not run."
        ),
    )
    rr.add_argument(
        "action",
        nargs="?",
        default="run",
        choices=("freeze", "run"),
        help=(
            "'run' (default) re-runs the golden set and reports drift; "
            "'freeze' records today's top-10 ids per query as the baseline "
            "that later runs are measured against, OVERWRITING any existing "
            "one for this --mode"
        ),
    )
    rr.add_argument(
        "--mode",
        default="lexical",
        choices=SEARCH_MODES,
        help=(
            "what to measure. 'lexical' and 'hybrid' read the Store and CANNOT "
            "SEE aggregator.mcp — no fusion membership rule, no vector floor, "
            "no confidence signal — so a clean run in either says nothing "
            "about the server. 'mcp' drives the same entry point the agent "
            "calls and covers all of it. Baselines are per-mode and result ids "
            "differ between them, so runs are never compared across modes. "
            "'hybrid' and 'mcp' both REFUSE on a cache with no vector index "
            "rather than quietly measuring the keyword arm and filing it under "
            "the mode you asked for. Every report prints what its mode could "
            "not reach"
        ),
    )
    rr.add_argument(
        "--drift-threshold",
        type=float,
        default=None,
        help=(
            "fail (exit 1) when mean drift exceeds this. Off by default and "
            "deliberately so: drift is DIRECTIONLESS — fixing retrieval scores "
            "exactly the same drift as breaking it — so a default threshold "
            "would block every intentional improvement. Pass one when you are "
            "asserting that a change should NOT move the ranking"
        ),
    )

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
        "--notify",
        action="store_true",
        help=(
            "with --all: send a desktop notification when the run ends with "
            "errors (CRITICAL) or staleness warnings (normal). For unattended "
            f"runs; ${NOTIFY_COMMAND_ENV_VAR} overrides the program "
            f"(default: {DEFAULT_NOTIFY_COMMAND}) and enables this on its own"
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

    p_embed = sub.add_parser(
        "embed",
        help="background embed worker (fills the v5 vector index)",
    )
    mode = p_embed.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--catchup",
        action="store_true",
        help=(
            "embed every unembedded row, then exit, in bounded per-batch "
            "committed chunks — this is what the systemd timer runs (with "
            "--source both), and what to run by hand against a stalled index. "
            "Sources are drained in priority order — "
            + " then ".join(s for _, s in EMBED_BACKLOG_ORDER[:6])
            + ", then everything unranked — and each is finished before the "
            "next begins, so `aggregator status` says which are searchable "
            "today rather than one percentage for the whole corpus"
        ),
    )
    mode.add_argument(
        "--once",
        action="store_true",
        help=(
            "embed a single batch and exit. A HAND-RUN PROBE — no deployed "
            "unit runs this: the timer runs --catchup and the seed unit runs "
            "--seed-models. Useful to watch one batch go through; useless for "
            "filling the index, since a batch is bounded at about fifteen "
            "minutes of encoder time, so one per 30-minute tick runs a "
            "backfill measured in weeks at half speed at best. Use --catchup "
            "for that"
        ),
    )
    mode.add_argument(
        "--seed-models",
        dest="seed_models",
        action="store_true",
        help=(
            "load the embedder AND the cross-encoder reranker, then exit — no "
            "database is opened and no row is read or written. This is the "
            "only path that fetches the reranker's weights, and it is what "
            f"the human-triggered seed unit runs. Set {MODEL_DOWNLOAD_ENV}=1 "
            "to permit the download; without it this reports what is missing "
            "and exits non-zero"
        ),
    )
    p_embed.add_argument(
        "--source",
        choices=["observations", "records", "both"],
        # ``both``, matching the deployed unit and every remediation string in
        # this codebase — all of which tell the operator to run a bare
        # `aggregator embed --catchup`. A default of ``observations`` made that
        # exact command leave records keyword-only, so `vector_index` kept
        # reporting records as never started while the person who had just
        # "fixed" it believed otherwise.
        default="both",
        help=(
            "which ontology to embed (default: both, matching the timer unit). "
            "This narrows the priority walk; it does not flatten it — the "
            "surviving sources keep their relative order"
        ),
    )
    p_embed.add_argument(
        "--batch-size",
        type=_positive_int,
        default=500,
        dest="batch_size",
        help=(
            f"rows per checkpointed batch (default: 500, must be >= 1). NO "
            f"LONGER THE ONLY BOUND, and usually not the one that binds: a "
            f"batch also stops at {_MAX_BATCH_CHUNKS} chunks, which is about "
            f"{_MAX_BATCH_CHUNKS * _SECONDS_PER_CHUNK // 60} minutes of "
            f"encoder time at the measured ~{_SECONDS_PER_CHUNK} s per "
            f"4000-character chunk (docs/embedding-throughput.md), and that "
            f"chunk cap is the CEILING on the interval between durable "
            f"checkpoints. This flag still means what it always did — rows "
            f"per batch — and a row count low enough to bind first still "
            f"shortens that interval; what it can no longer do is lengthen it "
            f"past the ceiling. What it bounds on its own is rows the encoder "
            f"never sees: an empty body costs no chunks, and about a third of "
            f"the corpus is empty bodies"
        ),
    )
    p_embed.add_argument(
        "--reindex",
        action="store_true",
        help=(
            "DELETE every vector in the cache and re-embed from scratch. Only "
            "does anything when the index on disk was written by a different "
            "model or dimension than this build produces — otherwise it is a "
            "no-op. Prints how many vectors it would destroy and asks for a "
            "'y' on stdin first (--yes skips the question, not the report). "
            "The last full backfill of this corpus took 25-30 days of CPU, so "
            "check `unset AGGREGATOR_EMBED_BACKEND` is not the real fix before "
            "using this. Nothing outside this flag can authorise the deletion"
        ),
    )
    p_embed.add_argument(
        "--yes",
        action="store_true",
        help="assume 'y' for the --reindex confirmation (scripted use)",
    )

    p_prov = sub.add_parser(
        "provenance",
        help="classify who composed each observation (fills the v6 column)",
    )
    prov_mode = p_prov.add_mutually_exclusive_group(required=True)
    prov_mode.add_argument(
        "--backfill",
        action="store_true",
        help=(
            "classify every observation whose provenance is still null, then "
            "exit, in bounded committed chunks. RESUMABLE: `provenance IS NULL` "
            "is the watermark and it lives in the rows it describes, so a kill "
            "costs at most one chunk and a re-run over a classified corpus "
            "reads nothing. It is a pure UPDATE of one column — no re-ingest, "
            "no re-scrub, no embedding_state reset. Single-threaded: it loads "
            "no model and holds no thread pool, so it cannot take more than "
            "one core"
        ),
    )
    prov_mode.add_argument(
        "--reclassify",
        action="store_true",
        help=(
            "reset every observation to unclassified and run the backfill "
            "again. This is what to run after CHANGING the classifier. It is "
            "cheap and safe precisely because provenance is not part of "
            "`_src_hash`: nothing else in the corpus notices, no body is "
            "re-scrubbed and no vector is discarded"
        ),
    )
    p_prov.add_argument(
        "--chunk-size",
        type=_positive_int,
        default=_PROVENANCE_CHUNK,
        dest="chunk_size",
        help=(
            f"rows per committed chunk (default: {_PROVENANCE_CHUNK}, must be "
            f">= 1). This bounds what a SIGTERM costs, not the throughput: the "
            f"whole table is a measured ~37 s under the narrowed FTS trigger"
        ),
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
    # BEFORE ANY STORE IS BUILT. Seeding a model cache must not open, create
    # or migrate the index — see ``_cmd_seed_models``. Every other subcommand
    # needs the store, so it stays eagerly built for them.
    if args.cmd == "embed" and args.seed_models:
        return _cmd_seed_models()
    store = _store or Store()
    # ``embed`` MIGRATES ITSELF, and must be the one to do it. It is the only
    # command that may authorise a vector reindex (``--reindex``), and that
    # authority is an argument to ``migrate()``. Migrating eagerly here would
    # run the provenance check first, without the argument — logging a refusal
    # that names two fixes, immediately before the run that applies one of
    # them. Every other subcommand still gets its schema up front.
    if args.cmd != "embed":
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
    if args.cmd == "retrieval-regression":
        return _cmd_retrieval_regression(args, store)
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
            # ONE ``Watermarks`` FOR THE WHOLE RUN, shared by the registry
            # (which turns each mark into the adapter's ``since``) and the
            # runner (which describes the window in the report and advances the
            # mark afterwards). Two instances would be two reads of the same
            # table and could not disagree today — but they could tomorrow, and
            # a report describing a different window from the one the adapters
            # were handed is worse than no report.
            watermarks = Watermarks(store, override=since)
            adapters = (
                _adapters
                if _adapters is not None
                else default_adapters(watermarks=watermarks)
            )
            return _cmd_ingest_all(
                args,
                store,
                adapters,
                _resolve_notify(args, _notify),
                watermarks=watermarks,
            )
        # Round-2 MEDIUM: the atomic DELETE + upsert lives inside
        # ``_cmd_ingest`` via ``store.rebuild_and_upsert`` when
        # ``args.rebuild`` is set. Do NOT call ``store.rebuild`` here —
        # doing so would commit the DELETE before the transaction and
        # reintroduce the non-atomic gap this fix closes.
        return _cmd_ingest(args, store, sources())
    if args.cmd == "embed":
        return _cmd_embed(args, _store=store)
    if args.cmd == "provenance":
        return _cmd_provenance(args, store, sources())
    if args.cmd == "github-token-status":
        return _cmd_github_token_status(args, store, sources())
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
