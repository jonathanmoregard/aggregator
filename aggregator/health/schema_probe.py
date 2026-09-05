"""Detect the schema skew between the cache, the MCP reader, and the writer.

WHAT WENT WRONG, AND WHY NOTHING SAW IT

Three components share one SQLite cache and each of them knows only its own
half of the contract:

  * the READER — ``aggregator-mcp``, the binary ``~/.claude.json``'s
    ``mcpServers.aggregator.command`` names, which on this host is a Nix
    wrapper chain ending in an ``-env`` derivation's site-packages — opens the
    cache ``mode=ro`` and refuses every call when
    ``PRAGMA user_version < SCHEMA_VERSION`` (``mcp.py``,
    ``_ensure_cache_ready``). It can never migrate: read-only by construction.
  * the WRITER — the ``aggregator`` on ``$PATH`` and the code
    ``aggregator-ingest.timer`` execs — is a Nix build from a rev pinned in
    nixos-config's ``flake.lock``. It runs ``migrate()``, which ENDS by
    stamping ``PRAGMA user_version = SCHEMA_VERSION`` — its own constant.
  * the CACHE carries whatever the writer last stamped.

On 2026-08-27 ``SCHEMA_VERSION`` went 5 -> 6 in the tree (commit ``3c9a29d``).
The reader picked it up immediately, because it runs from the tree. The pin
did not move, so the writer stayed at 5, re-stamped ``user_version = 5`` every
thirty minutes, and **exited 0 every single time**. ``OnFailure=`` fires on a
failed unit; there was no failed unit. The in-process notifier fires on a run
with something to say; the run had nothing to say. Recall was 100% dead for
three days and the only outward sign was an agent that quietly went back to
grepping transcripts.

The gap is structural, not an oversight. No component holds more than two of
the three numbers, and every pair looks healthy from inside:

  * the writer sees cache 5 and itself 5 — agreement. It cannot know 6 exists;
    the code that would notice was compiled from the same stale rev.
  * the reader sees cache 5 and itself 6 — disagreement, and it does say so,
    on every call. But a tool result only exists if a tool is called, and a
    tool that returns ``{"ok": false}`` is indistinguishable from a tool
    nobody used. That channel was already firing and already reaching nobody.

So the detector has to be a fourth thing that holds all three at once, and
that is all this module is.

TWO RULES THIS FILE EXISTS TO OBEY

**1. Never probe by running the thing under test.** ``cli.py``'s ``main()``
calls ``store.migrate()`` for every subcommand except ``embed``, and
``migrate()`` writes ``user_version``. So ``aggregator status`` — the obvious
probe, and the command the MCP's own remediation string still recommends — is
a WRITE that re-stamps the cache at the prober's version. Probing with it from
the schema-5 build performs the exact damage it is being asked to report on,
and destroys the evidence in the same breath. Everything here is read-only:
one SQLite connection opened ``mode=ro``, and two source files read as text.
The writer's version comes out of its packaged ``store.py``, never out of
running it — which also means a writer too broken to execute is still
measurable, and a wedged one cannot hang the probe. There are no subprocesses
in this module at all.

**2. Unknown means warn, never "fine".** Silence is this check's entire
budget: a detector that speaks on every run is one nobody reads by the time it
matters. That budget is only spent on a verdict it actually verified. An
absent cache, a corrupt one, a missing checkout, an unresolvable writer — none
of those are "no news". They are "could not tell", they are announced under
the DOWN headline, and the reasoning is the operator's own: a reader that does
not recognise the value must treat the thing as possibly broken and warn, never
as "unparseable, therefore fine".

WHICH READER, AND WHY IT IS NOT THIS CHECKOUT

The reader under test is whatever ``~/.claude.json``'s ``mcpServers.aggregator``
starts. Nothing else is evidence of what Claude Code executes. Until 2026-09-05
this file read only the ``--directory`` argument of that entry, and when there
was none — the entry now names a Nix wrapper directly, with empty ``args`` — it
fell back to its OWN checkout. On a host whose checkout sat at schema 7 while
the deployed reader, the writer and the cache were all at 6, that fallback
announced RECALL IS DEAD to every new session. A false alarm is not a cheap
error here: this check's entire value is that it stays quiet unless something is
wrong, and one confident lie spends the credibility the next real alarm needs.

So ``command`` is resolved the way the writer's binary always was: follow the
``exec`` line of each wrapper hop and take the first
``<prefix>/lib/python3*/site-packages`` that carries
``aggregator/core/store.py``. One walker, ``_wrapper_chain_prefixes``, serves
both. The own-checkout fallback survives only for the case it was written for —
no ``mcpServers.aggregator`` entry at all, and this file sitting in a tree with
a ``pyproject.toml``. An entry that exists and cannot be resolved is UNKNOWN,
which is rule 2 applied to the question "whose version am I even reading".

STATES, AND WHY FOUR RATHER THAN A BOOLEAN

``FINE``     cache == reader's requirement AND writer == it. Silent.
``DEAD``     cache != requirement. Recall is refusing RIGHT NOW.
``WILL_ROT`` the writer does not already stamp what the cache must end up at —
             it differs from the reader's requirement, or it is below the
             cache. Recall may work this minute, but the writer re-stamps the
             cache at its own version on the next tick, so a hand-run migration
             reverts within thirty minutes. This is the state a two-quantity
             check cannot see, and it is the one that explains why the incident
             kept coming back. Measured against ALL THREE for the same reason
             ``_forward_target`` is: at cache 7, reader 6, writer 6 a
             writer-versus-reader test is silent while that writer is queued to
             undo the DEAD finding's own remedy. The writer-below-cache half
             needs NO reader at all — it is a two-quantity fact — so it is
             still reported when the reader is UNKNOWN.
``UNKNOWN``  some quantity could not be read.

``!=``, NOT ``<``, AND THAT IS THE READER'S OWN RULE. The gate in ``mcp.py``
is ``version != SCHEMA_VERSION`` — ``_ensure_cache_ready``, which splits into
``_stale_cache_response`` and ``_ahead_cache_response``. This file compared
only ``<`` for its first life, so a cache stamped ABOVE the reader made every
``aggregator_search_memory`` call return ``ok:false`` while the probe printed
"healthy" and exited 0. A probe STRICTER than the gate trains its operator to
ignore it; a probe LOOSER than the gate is the incident it was written to
detect, running with the detector's own blessing.

``DEAD`` and ``WILL_ROT`` co-occur — that was the live incident — and both
have to survive into the report, because they have different remedies and
fixing only the first leaves a machine that breaks itself again on the next
tick.

THE REMEDY MOVES THE LAGGING SIDE UP, WHICHEVER SIDE THAT IS

Never down. Two components disagreeing on a version is repaired by bringing
the older one forward, and offering "or make the newer side accept the old
schema" as the other arm of a choice is not a neutral presentation of options —
the schema-6 reader wants columns a schema-5 cache does not have, so accepting
5 means reading a cache that cannot answer, which is the failure wearing a
different hat.

Which side lags is not fixed, and getting it wrong is worse than saying
nothing. When the cache is BEHIND, the writer lags and a newer writer is
deployed. When the cache is AHEAD, the cache is the current side and the
READER lags — so those messages name the reader, and name no writer command at
all, because running an older writer against a newer cache re-stamps
``user_version`` downward and turns a cache one component cannot read into a
cache that is wrong for all of them.

CONSUMERS

Two of them, sharing this one implementation rather than each growing their own
copy of the predicate — a detector that disagrees with itself about whether the
machine is healthy is worse than either half alone:

  * a systemd **user** timer, which reaches the operator through ``notify-send``
    on a machine with no agent session open;
  * a Claude Code **SessionStart** hook, which reaches the actual victim — a
    session that would otherwise believe recall works.

Either can invoke this file two ways, and the packaged one is preferred where
it exists: ``aggregator-schema-probe``, the console script declared in
pyproject.toml, so that it CAN be installed next to ``aggregator-mcp`` — built
from the same rev as the reader it measures — once the packaging enumerates it.
Declaring the script is only half of that; until nixos-config's
``overlays/aggregator.nix`` lists the name among the programs it wraps, the
profile does not carry it and the fallback is what runs: a bare script under
plain ``python3``, ``python3 .../schema_probe.py --json``, which is also the
normal shape in a dev checkout with nothing deployed.

Both routes must stay cheap, so this file is STDLIB ONLY and must stay that way,
and ``aggregator/__init__.py`` and ``aggregator/health/__init__.py`` must stay
EMPTY — the console script walks through both on its way here. Pulling the
package in would drag in torch and sentence-transformers, and a SessionStart
hook that blows its budget has its output DISCARDED — which for a health check
is the same as never noticing.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

# --- states -----------------------------------------------------------------

FINE = "fine"
DEAD = "dead"
WILL_ROT = "will-rot"
UNKNOWN = "unknown"

# --- severities, which pick the headline the consumers print ----------------
#
# Kept separate from the states because two different states share one
# headline: "recall is refusing now" and "recall cannot be verified" call for
# the same urgency even though they are different diagnoses. Collapsing states
# into severities would lose the diagnosis; collapsing severities into states
# would make the notifier re-derive the headline and drift.

DOWN = "down"
REDUCED = "reduced"
SILENT = "silent"

# --- exit codes: the machine-readable verdict -------------------------------
#
# Distinct per state rather than a plain zero/non-zero, so a systemd unit or a
# shell caller can branch on WHICH failure without parsing JSON. Spaced by ten
# to leave room, and deliberately not 1 or 2: a probe that crashes exits 1 from
# the interpreter, and conflating "the machine is sick" with "the probe is
# broken" is how a health check gets muted.

EXIT_FINE = 0
EXIT_WILL_ROT = 10
EXIT_DEAD = 20
EXIT_UNKNOWN = 30

_EXIT_CODES = {
    FINE: EXIT_FINE,
    WILL_ROT: EXIT_WILL_ROT,
    DEAD: EXIT_DEAD,
    UNKNOWN: EXIT_UNKNOWN,
}

# Worst-first. ``DEAD`` outranks ``UNKNOWN`` because it is strictly more
# actionable — it names the two numbers and the fix — while ``UNKNOWN``
# outranks ``WILL_ROT`` because a probe that could not measure must not be
# reported as the milder, works-today state.
_SEVERITY_ORDER = [DEAD, UNKNOWN, WILL_ROT, FINE]

_SEVERITY_OF = {
    DEAD: DOWN,
    UNKNOWN: DOWN,
    WILL_ROT: REDUCED,
    FINE: SILENT,
}

# --- environment overrides --------------------------------------------------
#
# Every input is overridable, because the tests must be able to forge all
# three quantities independently and because a probe that can only ever look
# at the live machine cannot be tested at all — which for a health check is
# the failure mode, not an inconvenience.

CACHE_DB_ENV = "AGGREGATOR_CACHE_DB"
READER_DIR_ENV = "AGGREGATOR_READER_DIR"
WRITER_BIN_ENV = "AGGREGATOR_WRITER_BIN"

# Nothing is read past this from any source file. These are a SQLite header
# and two Python modules; a multi-megabyte file at one of those paths is a
# fault in itself, not something to spend a session-start budget scanning.
_SOURCE_SCAN_LIMIT = 2 * 1024 * 1024

# ``~/.claude.json`` gets its own, larger bound, because it is not a source
# file and legitimately gets big: Claude Code stores per-project ``history``
# arrays in the same document, so a machine with a few long-lived projects
# carries a config two orders of magnitude past any store.py.
#
# A BOUND ON A DOCUMENT THAT MUST BE PARSED WHOLE IS A DECISION, NOT A CLAMP.
# Truncating a source file is harmless — the constant either appeared in the
# prefix or it did not. Truncating JSON yields a buffer cut mid-token, and
# ``json.loads`` rejects it in the same breath and with the same exception it
# uses for genuine corruption. So the size is checked BEFORE the read and a
# file past the cap is never parsed at all: the two causes are only
# distinguishable here, and conflating them hands the operator "repair
# ~/.claude.json" about a file that is already valid — an unfollowable chore,
# announced every session.
_CLAUDE_CONFIG_LIMIT = 16 * 1024 * 1024

# ``SCHEMA_VERSION = 6`` at column zero. Matched as TEXT rather than imported,
# because importing either side's ``store.py`` costs the whole dependency
# tree. Anchored to the line start so a mention inside a comment or a string
# cannot be picked up ahead of the real assignment.
_SCHEMA_CONST = re.compile(r"^SCHEMA_VERSION\s*=\s*(\d+)", re.MULTILINE)

# The last line of a Nix wrapper: ``exec "/nix/store/...-env/bin/aggregator" "$@"``.
# The writer on this host is always reached through at least one such hop, and
# only the far end carries site-packages.
_WRAPPER_EXEC = re.compile(r"^\s*exec\s+(?:-a\s+\S+\s+)?[\"']?([^\"'\s]+)", re.MULTILINE)

# How many wrapper hops to follow before giving up. Wrapper chains are one or
# two deep in practice; the bound is here so a symlink or exec cycle reports
# UNKNOWN instead of spinning.
_MAX_WRAPPER_HOPS = 8


class _ConfigFault:
    """Why ``~/.claude.json`` yielded no entry, when the answer is not "none".

    Returned INSTEAD of ``None`` so the own-checkout fallback is skipped: the
    distinction the fallback turns on is that absent means nobody configured a
    reader, while a fault means somebody did and this probe cannot tell what.

    A CLASS, AND NOT TWO EMPTY DICTS. Both markers used to be plain ``{}``, so
    ``==`` said the two faults were the same value — and also that either was
    the same value as the ordinary ``{}`` returned for an entry that exists and
    is empty. Three different facts behind one comparison, kept apart only by
    every call site happening to use ``is``. That is a property of the readers
    rather than of the values, and it is load-bearing in what an operator is
    told: one fault says go and repair a broken file, the other says the file
    is FINE and merely too big to read. Sending someone to repair valid JSON is
    the whole thing this pair exists to prevent, so it must not hinge on which
    operator the next reader reaches for.
    """

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"<claude.json {self.name}>"


#: Present, and could not be understood — bad JSON, not an object, or an
#: ``mcpServers`` this cannot read. The operator's file to repair.
_CONFIG_UNREADABLE = _ConfigFault("unreadable")

#: Present, valid as far as anyone knows, and larger than
#: ``_CLAUDE_CONFIG_LIMIT``. Nothing to repair: this probe declines to read it
#: on a session-start budget, and ``AGGREGATOR_READER_DIR`` gets past it.
_CONFIG_TOO_LARGE = _ConfigFault("too-large")


@dataclass(frozen=True)
class Finding:
    """One thing worth saying, with the fix attached.

    ``remedy`` is not optional decoration. "Your schema versions disagree"
    tells an operator nothing to do at 03:00, and an announcement with no
    action is one that gets acknowledged and forgotten — which is how this
    incident survived three days of a tool returning ``ok: false`` on every
    call.
    """

    state: str
    detail: str
    remedy: str

    def text(self) -> str:
        return f"{self.detail} {self.remedy}"


@dataclass
class Verdict:
    """Everything the probe measured, and everything it concluded.

    The raw numbers ride along with the conclusion on purpose. A consumer that
    only got a state would have to re-derive "5 against 6" to say anything
    useful, and the notifier and the hook would then each own a copy of the
    formatting.
    """

    state: str
    severity: str
    states: list[str]
    findings: list[Finding] = field(default_factory=list)
    cache_version: int | None = None
    cache_meta_version: int | None = None
    reader_version: int | None = None
    writer_version: int | None = None
    cache_db: str | None = None
    reader_dir: str | None = None
    writer_bin: str | None = None

    def exit_code(self) -> int:
        return _EXIT_CODES[self.state]

    def explain(self) -> str:
        """One human-readable paragraph. The only rendering, used by both
        consumers so their wording cannot drift apart."""
        if not self.findings:
            return (
                f"aggregator recall is healthy: cache schema {self.cache_version}, "
                f"MCP reader requires {self.reader_version}, writer builds "
                f"{self.writer_version}."
            )
        return " ".join(f.text() for f in self.findings)

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "severity": self.severity,
            "states": list(self.states),
            "exit_code": self.exit_code(),
            "cache_version": self.cache_version,
            "cache_meta_version": self.cache_meta_version,
            "reader_version": self.reader_version,
            "writer_version": self.writer_version,
            "cache_db": self.cache_db,
            "reader_dir": self.reader_dir,
            "writer_bin": self.writer_bin,
            "summary": self.explain(),
            "findings": [
                {"state": f.state, "detail": f.detail, "remedy": f.remedy}
                for f in self.findings
            ],
        }


# --- resolving the three inputs ---------------------------------------------


def resolve_cache_db(env: dict[str, str] | None = None) -> Path:
    """Where the cache lives, resolved WITHOUT calling into the package.

    ``store._default_db_path()`` computes the same path but also creates the
    parent directories on the way — a side effect this module will not have.
    Duplicating four lines is the cheaper of the two evils; the shape
    (``$XDG_DATA_HOME/aggregator/cache.db``) has been stable since v1 and any
    drift shows up as an absent cache, which warns rather than passing.
    """
    env = os.environ if env is None else env
    override = env.get(CACHE_DB_ENV)
    if override:
        return Path(override).expanduser()
    root = env.get("XDG_DATA_HOME") or os.path.join(
        env.get("HOME") or str(Path.home()), ".local", "share"
    )
    return Path(root) / "aggregator" / "cache.db"


def _which(name: str, env: dict[str, str]) -> Path | None:
    """``shutil.which`` for one name, over the ``PATH`` in ``env``.

    Spelled out rather than imported to keep this a single self-contained file,
    and taking the environment as an argument rather than reading the process's:
    the systemd unit, a Claude Code session and a test all have different
    ``PATH``s, and a resolver that quietly consulted the wrong one would report
    on a binary nobody runs. Shared by the writer's own lookup and by the
    reader's bare-``command`` lookup, for the same reason there is one wrapper
    walker and not two.
    """
    for d in (env.get("PATH") or "").split(os.pathsep):
        if not d:
            continue
        candidate = Path(d) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _claude_config_path(env: dict[str, str]) -> Path:
    """The config that decides which reader is under test."""
    return Path(env.get("HOME") or str(Path.home())) / ".claude.json"


def _claude_mcp_entry(env: dict[str, str]) -> dict | _ConfigFault | None:
    """``~/.claude.json``'s TOP-LEVEL ``mcpServers.aggregator``, or ``None``.

    Top-level only, deliberately. The same file carries per-project
    ``projects["<dir>"].mcpServers`` blocks, and this host has a stale one from
    an unrelated repo pointing at a command that resolves to nothing. Those
    apply only to sessions started in that directory; reading them here would
    let a dead entry from someone else's project decide what this probe
    measures.

    Three outcomes, and the split between the last two is the whole point:

    * a dict — the entry, ready to resolve.
    * ``None`` — "there is no entry to be had", and no reason to doubt that:
      the file does not exist, or it parses and simply configures no aggregator
      server. Only this outcome lets the caller fall back to its own checkout,
      because a HOME with no ``.claude.json`` is every CI run and every fresh
      clone and must not be an alarm.
    * ``_CONFIG_UNREADABLE`` — the file IS there and could not be understood.
      Not the same fact at all. An operator configured something and this probe
      cannot see what, so guessing "the checkout" would announce a measurement
      of a tree that may have nothing to do with the reader — which is exactly
      the 2026-09-05 false alarm, re-entering through a second door. Fail
      loudly: the caller turns this into UNKNOWN.

    An entry that EXISTS but is not an object also comes back as ``{}``:
    something is configured, so the checkout is not the answer.

    A FOURTH outcome, ``_CONFIG_TOO_LARGE``, splits off the one shape that used
    to masquerade as the third. The file is read under a bound
    (``_CLAUDE_CONFIG_LIMIT``) and a bounded read of a VALID document that is
    bigger than the bound comes back cut mid-token, which ``json.loads``
    rejects exactly the way it rejects corruption. So the size decides first
    and an oversized file is never parsed: reporting a 17 MB valid config as
    unparseable told the operator to go and repair a file with nothing wrong
    with it, every session, forever.
    """
    path = _claude_config_path(env)
    try:
        size = os.stat(path).st_size
    except (FileNotFoundError, NotADirectoryError):
        # Genuinely absent. The one silent case.
        return None
    except OSError:
        return _CONFIG_UNREADABLE
    if size > _CLAUDE_CONFIG_LIMIT:
        return _CONFIG_TOO_LARGE

    try:
        with open(path, "rb") as fh:
            # One byte past the cap, so a file that GREW between the stat and
            # this read is caught by length rather than parsed truncated. The
            # stat is what makes the common case cheap; this is what makes it
            # correct.
            raw = fh.read(_CLAUDE_CONFIG_LIMIT + 1)
    except (FileNotFoundError, NotADirectoryError):
        # Genuinely absent. The one silent case.
        return None
    except OSError:
        # Present and this process cannot read it — a permission or IO fault,
        # which is a fault to report rather than a licence to guess.
        return _CONFIG_UNREADABLE
    if len(raw) > _CLAUDE_CONFIG_LIMIT:
        return _CONFIG_TOO_LARGE

    try:
        doc = json.loads(raw.decode("utf-8", "replace"))
    except ValueError:
        return _CONFIG_UNREADABLE
    if not isinstance(doc, dict):
        return _CONFIG_UNREADABLE

    servers = doc.get("mcpServers")
    if servers is None:
        return None
    if not isinstance(servers, dict):
        # An entry could be hiding in a shape this cannot read.
        return _CONFIG_UNREADABLE
    if "aggregator" not in servers:
        return None
    entry = servers["aggregator"]
    return entry if isinstance(entry, dict) else {}


def _reader_dir_from_command(command: object, env: dict[str, str]) -> Path | None:
    """The install directory behind ``mcpServers.aggregator.command``.

    That command is what Claude Code execs, so its chain is the only statement
    of which reader is under test that cannot be stale. A bare name (no
    separator) is looked up on the PATH this probe was handed, exactly as a
    shell would; anything with a separator, or a ``~``, is a path. The chain
    from there to a packaged ``store.py`` is the writer's chain in every
    respect, so it goes through the same walker.

    ``None`` when the name resolves nowhere, or resolves to something carrying
    no aggregator package — the caller turns that into UNKNOWN, which is the
    honest verdict and the loud one.
    """
    return resolve_package_dir(_command_binary(command, env))


def _command_binary(command: object, env: dict[str, str]) -> Path | None:
    """The FILE a ``command`` string names, resolved the way a shell would.

    Split out from ``_reader_dir_from_command`` because the writer needs the
    binary and not the package: its own lookup starts from the reader's
    ``bin/`` directory, which only exists as a fact about the file, not about
    the site-packages the chain ends in.
    """
    if not isinstance(command, str) or not command:
        return None
    return (
        Path(command).expanduser()
        if os.sep in command or command.startswith("~")
        else _which(command, env)
    )


def _directory_arg(args: object) -> Path | None:
    """``--directory <dir>`` or ``--directory=<dir>``, or ``None``.

    One reading of the dev shape, asked by two callers now: the reader wants
    the tree, and the writer wants to know that the ``command`` is ``uv``
    rather than a reader — so a second copy would be a second thing to drift.

    BOTH SPELLINGS. ``uv run`` accepts them interchangeably and this config is
    hand-edited, so which one an operator typed cannot decide whether the check
    works. Parsing only the split form sent the joined one down the ``command``
    branch, where the command is ``uv``: the wrapper walk finds uv's own
    install, no ``aggregator/core/store.py`` is there, and a session pointed at
    a perfectly good checkout was told its recall health could not be verified
    — on the strength of a space.

    An empty value is not a directory and is passed over in BOTH spellings,
    the same way a trailing bare ``--directory`` with nothing after it is. The
    joined form guarded that from the start; the split form did not, and
    ``Path("")`` is ``Path(".")`` — so ``["--directory", ""]`` answered with
    the PROBE'S OWN WORKING DIRECTORY and every downstream message named it as
    the reader. That is the 2026-09-05 false alarm again with a worse tree
    substituted: the checkout fallback at least names a tree that holds an
    aggregator, while a cwd is wherever systemd or a session hook was started.
    Passing over means resolution CONTINUES — at the ``command``, then at the
    fallbacks — never that the entry is abandoned.
    """
    if not isinstance(args, list):
        return None
    for i, a in enumerate(args):
        if a == "--directory" and i + 1 < len(args):
            value = str(args[i + 1])
            if value:
                return Path(value).expanduser()
        if isinstance(a, str) and a.startswith("--directory="):
            value = a.partition("=")[2]
            if value:
                return Path(value).expanduser()
    return None


def _writer_beside_the_reader(env: dict[str, str]) -> Path | None:
    """The ``aggregator`` sitting in the same ``bin/`` as the MCP reader.

    THE WRITER UNDER TEST IS THE DEPLOYED ONE, and a PATH search does not
    reliably name it. ``aggregator-ingest.timer`` execs ``pkgs.aggregator``;
    the profile's ``aggregator-mcp`` is a program of that same package, so on
    this host the two are neighbours in one ``bin/``. Inside a checkout, a
    ``uv run``, or a devShell, the first ``aggregator`` on PATH is instead the
    checkout's own venv CLI — a binary no timer runs. Measuring it is wrong
    twice over: a deployed writer that really has fallen behind never fires
    WILL_ROT, and a checkout that happens to be ahead invents a skew nobody
    has.

    Only when the reader was resolved from a ``command``. A ``--directory``
    entry means the command is ``uv``, whose neighbours are uv's install, and
    an ``AGGREGATOR_READER_DIR`` override means the config does not describe
    the reader at all — in both cases a sibling would be a binary picked by
    coincidence of directory layout.

    ``None`` defers to the PATH search rather than concluding anything: a
    profile that ships the MCP server without the CLI beside it has no sibling
    to find, and turning that into UNKNOWN would break a working install.
    """
    if env.get(READER_DIR_ENV):
        return None
    entry = _claude_mcp_entry(env)
    if not isinstance(entry, dict):
        return None
    if _directory_arg(entry.get("args")) is not None:
        return None
    binary = _command_binary(entry.get("command"), env)
    if binary is None:
        return None
    sibling = binary.parent / "aggregator"
    return sibling if sibling.is_file() and os.access(sibling, os.X_OK) else None


def resolve_reader_dir(env: dict[str, str] | None = None) -> Path | None:
    """Which install the MCP reader actually runs from, in priority order.

    Asked of ``~/.claude.json`` rather than assumed, because that file is what
    Claude Code executes, so it is the only source that cannot be out of date
    with respect to the reader under test. A hard-coded ``~/Repos/aggregator``
    would keep reporting on a checkout the reader had stopped using, and would
    do it silently.

    The order, and the reason for each step:

    1. ``AGGREGATOR_READER_DIR`` — the escape hatch every input here has.
    2. ``args`` carrying ``--directory <dir>`` or ``--directory=<dir>`` — the
       dev shape, ``uv run --directory <checkout> aggregator-mcp``. A session
       pointed at a checkout IS running that checkout, and resolving ``uv``
       through the wrapper walk would find uv's own install.
    3. ``command`` — the deployed shape, ``{"command": "<nix wrapper>",
       "args": []}``. Resolved through the wrapper chain to the site-packages
       that carries ``aggregator/core/store.py``: the directory the number is
       actually read from, which is what the verdict then reports.
    4. No ``mcpServers.aggregator`` entry AT ALL, in a config that was readable
       (or absent entirely) — a checkout with no MCP wiring. Only here does this
       file fall back to its own tree, and only when that tree has a
       ``pyproject.toml``: installed into site-packages this module also sits
       beside a ``core/store.py``, and reading THAT as the reader's requirement
       would compare the writer against itself and report every skew as healthy.
    5. Otherwise ``None`` — an entry exists and could not be resolved, or the
       config itself could not be parsed, or it was past
       ``_CLAUDE_CONFIG_LIMIT`` and deliberately not parsed at all. UNKNOWN,
       never a substitute measurement of something else. A config that is
       PRESENT and unusable gets no fallback: somebody configured a reader, so
       this tree is not it.

    Step 3 is the whole point of this function's second life. Until 2026-09-05
    only step 2 existed, and an entry with empty ``args`` fell straight through
    to step 4: on a host whose checkout was at schema 7 while the deployed
    reader, writer and cache were all at 6, every session was told RECALL IS
    DEAD about a machine where recall was fine.
    """
    env = os.environ if env is None else env
    override = env.get(READER_DIR_ENV)
    if override:
        return Path(override).expanduser()

    entry = _claude_mcp_entry(env)
    if entry is None:
        own = Path(__file__).resolve().parent.parent.parent
        return own if (own / "pyproject.toml").is_file() else None
    if isinstance(entry, _ConfigFault):
        # Step 5, said out loud. While both faults were spelled ``{}`` this
        # branch did not need to exist: an empty dict answers ``.get`` with
        # ``None`` twice and falls out of the bottom returning ``None`` by
        # accident. Accidentally right is not a contract, and the accident
        # died the moment the two faults became distinguishable values.
        return None

    directory = _directory_arg(entry.get("args"))
    if directory is not None:
        return directory

    return _reader_dir_from_command(entry.get("command"), env)


def resolve_writer_bin(env: dict[str, str] | None = None) -> Path | None:
    """The ``aggregator`` THE INGEST TIMER would actually run, in priority order.

    1. ``AGGREGATOR_WRITER_BIN`` — the escape hatch every input here has.
    2. The ``aggregator`` beside the resolved reader command. On this host
       ``aggregator-ingest.timer`` execs ``pkgs.aggregator`` and the profile's
       ``aggregator-mcp`` is a program of that same package, so the deployed
       writer is the reader's neighbour in one ``bin/``. See
       ``_writer_beside_the_reader`` for when this step declines to answer.
    3. ``_which("aggregator", env)`` — ``shutil.which`` semantics over the
       ``PATH`` passed in rather than the process's, sharing the reader's
       lookup.

    STEP 2 IS NOT A SHORTCUT, IT IS THE CORRECTION. Step 3 alone answers with
    whatever is first on the probe's PATH, and in a checkout, a ``uv run`` or
    a devShell that is the checkout's own venv CLI — a binary no timer runs
    and no cache is ever stamped by. Reporting it as "the writer" hides a
    deployed writer that has genuinely fallen behind and invents a skew when
    the checkout is merely ahead. Step 3 survives underneath because a profile
    can ship the MCP server without the CLI beside it, and the verdict keeps
    reporting ``writer_bin`` either way, so which one was read is never a
    guess the operator has to make.
    """
    env = os.environ if env is None else env
    override = env.get(WRITER_BIN_ENV)
    if override:
        return Path(override).expanduser()
    return _writer_beside_the_reader(env) or _which("aggregator", env)


# --- reading the three quantities -------------------------------------------


def read_cache_versions(cache_db: Path) -> tuple[int | None, int | None, str | None]:
    """``(user_version, meta.schema_version, error)`` — read-only, always.

    ``mode=ro`` is the same URI the MCP reader opens the cache with, so this
    sees exactly what the reader sees, including failing in the same way when
    the file is missing or malformed. That correspondence is the point: a
    probe that could read a cache the reader cannot would report healthy on a
    machine where recall is refusing.

    Both stamps are returned. ``migrate()`` writes them together and the gate
    reads only the pragma, so the pragma decides the verdict — but they are
    two independent readings of one fact, and a cache where they disagree was
    written by something that is not ``migrate()``. Saying so costs one query.

    ``sqlite3.DatabaseError`` and not ``OperationalError``: a corrupt file
    ("database disk image is malformed") raises the parent class, and the
    narrower clause would miss the single most important thing it looks like
    it catches. This is the same correction ``mcp.py`` already carries.
    """
    try:
        path = cache_db.resolve()
    except OSError as exc:  # pragma: no cover - resolve() on a broken mount
        return None, None, f"{type(exc).__name__}: {exc}"

    if not path.exists():
        return None, None, "no such file"

    con = None
    try:
        con = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
        row = con.execute("PRAGMA user_version").fetchone()
        user_version = int(row[0]) if row else None
    except (sqlite3.DatabaseError, ValueError, TypeError) as exc:
        if con is not None:
            con.close()
        return None, None, f"{type(exc).__name__}: {exc}"

    meta_version: int | None = None
    try:
        meta_row = con.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        if meta_row is not None:
            meta_version = int(meta_row[0])
    except (sqlite3.DatabaseError, ValueError, TypeError):
        # No ``meta`` table, or a value that is not a number. Corroboration is
        # a bonus; its absence is not itself a fault, and a freshly rebuilt
        # cache legitimately has none yet.
        meta_version = None
    finally:
        con.close()

    return user_version, meta_version, None


def _read_schema_const(store_py: Path) -> int | None:
    try:
        with open(store_py, "rb") as fh:
            text = fh.read(_SOURCE_SCAN_LIMIT).decode("utf-8", "replace")
    except OSError:
        return None
    m = _SCHEMA_CONST.search(text)
    return int(m.group(1)) if m else None


def _wrapper_chain_prefixes(binary: Path) -> list[Path]:
    """Every ``<prefix>`` of a ``<prefix>/bin/<name>`` seen along a wrapper chain.

    ONE walker, for the reader and the writer both. They are reached through
    chains of the identical shape on this host — a profile entry symlinked into
    a derivation's ``bin/``, a shell wrapper there whose last line ``exec``s a
    console script inside an ``-env`` derivation, and only that env carrying
    ``lib/python3.11/site-packages`` — and two copies of this walk would be two
    chances to drift. The half that drifted would be the half nobody was
    looking at.

    Every prefix along the way is recorded, in order, because which hop owns
    site-packages is not knowable in advance: a plain venv answers on the first,
    the Nix chain on the second. The walk stops at a file that is not a script
    (a real ELF binary is the end of the road), at a script with no ``exec``
    line, at a cycle, and at ``_MAX_WRAPPER_HOPS`` — so a symlink loop reports
    UNKNOWN instead of spinning.
    """
    prefixes: list[Path] = []
    seen: set[str] = set()
    current: Path | None = Path(binary)

    for _ in range(_MAX_WRAPPER_HOPS):
        if current is None:
            break
        try:
            real = current.resolve()
        except OSError:
            break
        key = str(real)
        if key in seen or not real.is_file():
            break
        seen.add(key)

        # ``<prefix>/bin/<name>`` -> ``<prefix>``.
        if real.parent.name == "bin":
            prefixes.append(real.parent.parent)

        try:
            with open(real, "rb") as fh:
                head = fh.read(_SOURCE_SCAN_LIMIT)
        except OSError:
            break
        if not head.startswith(b"#!"):
            # A real ELF binary: the chain ends here.
            break
        m = _WRAPPER_EXEC.search(head.decode("utf-8", "replace"))
        current = Path(m.group(1)) if m else None

    return prefixes


def resolve_package_dir(binary: Path | None) -> Path | None:
    """The directory an installed binary's chain puts the aggregator package in.

    A directory rather than a version, because both callers need different
    things from it: the writer wants the number, and the reader wants the
    number AND a path to report. It is the first prefix the constant can
    actually be READ from, never merely the first that exists — so a verdict
    naming this directory names a file the number came out of. That is worth
    reading a sub-2 MB source file twice.
    """
    if binary is None:
        return None
    for prefix in _wrapper_chain_prefixes(binary):
        for lib in sorted(prefix.glob("lib/python3*/site-packages")):
            if _read_schema_const(lib / "aggregator" / "core" / "store.py") is not None:
                return lib
    return None


def read_reader_version(reader_dir: Path | None) -> int | None:
    """The version the MCP reader will refuse anything below.

    Read as text out of whichever directory holds the reader's package — a
    checkout in dev, an env derivation's site-packages once deployed. The same
    reading serves the writer, whose directory is found the same way.
    Emphatically not an import: ``aggregator.core.store`` pulls
    sentence-transformers and torch, which is seconds of model-loading
    machinery, and both consumers here run on budgets measured in single-digit
    seconds.
    """
    if reader_dir is None:
        return None
    return _read_schema_const(Path(reader_dir) / "aggregator" / "core" / "store.py")


def read_writer_version(writer_bin: Path | None) -> int | None:
    """The version the packaged writer will stamp the cache with.

    Obtained by READING the writer's packaged source, never by executing it.
    Three reasons, all load-bearing: running it would migrate the cache (rule 1
    at the top of this file); a wedged install would hang the probe, which on a
    hook budget means the output is discarded and nothing is reported; and the
    binary being broken is itself one of the conditions the probe must survive
    in order to speak.

    Nothing below this line is writer-specific any more. The path from a binary
    to its packaged ``store.py`` is a Nix wrapper chain, the reader is reached
    through one of exactly the same shape, and both go through
    ``resolve_package_dir`` — see ``_wrapper_chain_prefixes`` for why there is
    one walker and not two.
    """
    return read_reader_version(resolve_package_dir(writer_bin))


# --- the predicate ----------------------------------------------------------


def _forward_target(*versions: int | None) -> int | None:
    """The one version every remedy names: the highest anything is at.

    ONE NUMBER FOR ALL OF THEM, because the findings are handed to an operator
    together and get followed together. Computed pairwise against the reader
    alone, a three-way skew forked into advice that undid itself: at cache 7,
    reader 6, writer 5 the DEAD finding called the cache the current side and
    said leave it alone, while the WILL_ROT finding beside it asked for a
    writer "at least 6" — and a schema-6 writer re-stamps that schema-7 cache
    DOWN on the next tick. The mirror, cache 5 / reader 6 / writer 7, told the
    operator to bring a writer already at 7 "up to at least 6", which is inert
    read charitably and an instruction to install a downgrade read literally.

    The maximum is the only choice that cannot ask anything to move backwards:
    it is at or above every quantity measured, so each component either already
    satisfies it or has to come up. ``None`` when nothing was measured, and
    unreadable quantities simply do not vote — a missing writer version must
    not drag the target below a cache that was read.

    Note this is deliberately the max over ALL THREE and not over the reader
    and cache alone. The writer is the component that stamps, so a writer above
    both is precisely the case where a reader-and-cache target names a version
    something would have to be downgraded to.
    """
    known = [v for v in versions if v is not None]
    return max(known) if known else None


def probe(
    *,
    cache_db: Path | None = None,
    reader_dir: Path | None = None,
    writer_bin: Path | None = None,
    env: dict[str, str] | None = None,
) -> Verdict:
    """Read all three quantities and decide. The only place the rule lives."""
    env = os.environ if env is None else env
    cache_db = Path(cache_db) if cache_db is not None else resolve_cache_db(env)
    if reader_dir is None:
        reader_dir = resolve_reader_dir(env)
    if writer_bin is None:
        writer_bin = resolve_writer_bin(env)

    cache_version, meta_version, cache_error = read_cache_versions(cache_db)
    reader_version = read_reader_version(reader_dir)
    writer_version = read_writer_version(writer_bin)

    findings: list[Finding] = []
    # One number for every remedy below. See ``_forward_target``: computed
    # pairwise against the reader alone, a three-way skew produced advice
    # that undid itself.
    target = _forward_target(cache_version, reader_version, writer_version)

    # --- could-not-measure first. Each of these makes some later comparison
    # unanswerable, and an unanswerable comparison must never be quietly
    # skipped into silence.

    if reader_version is None:
        # THREE causes land here with three different fixes, so they get three
        # remedies. A config this probe could not parse is the operator's file
        # to repair; a config too big to read under this probe's budget is a
        # file with nothing wrong with it and needs the override instead; a
        # config that resolved to an install with no packaged source is the
        # install's problem. Only asked when the resolution already failed, so
        # the healthy path pays nothing for it.
        config = _claude_config_path(env)
        entry = _claude_mcp_entry(env) if reader_dir is None else None
        if entry is _CONFIG_TOO_LARGE:
            try:
                size = os.stat(config).st_size
            except OSError:  # pragma: no cover - it was there a moment ago
                size = -1
            findings.append(
                Finding(
                    UNKNOWN,
                    "aggregator recall health CANNOT BE VERIFIED: "
                    f"{config} is {size} bytes, past the {_CLAUDE_CONFIG_LIMIT}-byte "
                    "cap this probe reads it under, so which MCP reader Claude "
                    "Code starts is unknown. The file was NOT parsed and is NOT "
                    "being called corrupt: a bounded read of a valid document "
                    "comes back cut mid-token, and there is nothing wrong with "
                    "the JSON. Claude Code keeps per-project `history` arrays in "
                    "this same document, which is what grows it. This probe runs "
                    "on a session-start budget and will not scan an unbounded "
                    "file to find one server entry.",
                    f"FIX: set {READER_DIR_ENV} to the directory the MCP reader's "
                    "package lives in — lib/python3*/site-packages for a deployed "
                    "build, the checkout root for a dev one — which skips this "
                    f"file entirely. Or bring {config} back under "
                    f"{_CLAUDE_CONFIG_LIMIT} bytes by pruning its per-project "
                    "`history` entries.",
                )
            )
        elif entry is _CONFIG_UNREADABLE:
            findings.append(
                Finding(
                    UNKNOWN,
                    "aggregator recall health CANNOT BE VERIFIED: "
                    f"{config} exists but could not be parsed, so which MCP "
                    "reader Claude Code starts is unknown. This probe will NOT "
                    "guess by reading its own checkout — that guess is what "
                    "announced a false RECALL IS DEAD on 2026-09-05, on a host "
                    "where the real entry sat unparsed in this very file.",
                    f"FIX: repair {config} — it must be valid JSON whose "
                    "top-level mcpServers.aggregator names the reader, via a "
                    "`command` or a `--directory` argument. "
                    "AGGREGATOR_READER_DIR overrides the file entirely.",
                )
            )
        else:
            findings.append(
                Finding(
                    UNKNOWN,
                    "aggregator recall health CANNOT BE VERIFIED: the MCP reader's "
                    "required schema version could not be read from "
                    f"{reader_dir or '(no reader install located)'} — expected "
                    "`SCHEMA_VERSION = <n>` in aggregator/core/store.py. Without it "
                    "there is no number to compare the cache and the writer against, "
                    "so nothing here can be called healthy.",
                    "FIX: check ~/.claude.json's top-level mcpServers.aggregator — "
                    "its `command` must resolve to an install carrying "
                    "lib/python3*/site-packages/aggregator/core/store.py, or its "
                    "args must name a real aggregator tree with `--directory`. "
                    "AGGREGATOR_READER_DIR overrides both.",
                )
            )

    if cache_version is None:
        findings.append(
            Finding(
                UNKNOWN,
                f"aggregator recall is DOWN or unverifiable: the cache at {cache_db} "
                f"could not be read ({cache_error or 'unknown error'}). The MCP "
                "opens this file read-only and cannot create or repair one, so an "
                "absent or malformed cache means recall is refusing right now — "
                "this is not 'nothing has broken yet'.",
                "FIX: run `aggregator ingest --all` from a writer whose "
                "SCHEMA_VERSION is at least the reader's, which will create and "
                "stamp a fresh cache.",
            )
        )

    if writer_version is None:
        findings.append(
            Finding(
                UNKNOWN,
                "the aggregator WRITER's schema version could not be determined "
                f"(looked at {writer_bin or 'no `aggregator` on PATH'}). The writer "
                # Says "agrees with" rather than "matches" deliberately.
                # tests/test_fts5_match_site_enumeration.py collects EVERY string
                # literal containing the word "match" in any case — the
                # case-insensitivity is load-bearing there, because the injection
                # shape it exists to catch is an interpolated lowercase
                # ``{table} match '{q}'`` — and requires each one to be classified
                # as an FTS5 site, a vector site, or declared prose. This module
                # contains no SQL beyond a read-only PRAGMA, so the honest fix is
                # to not use the word rather than to grow a set that file freezes
                # on purpose.
                "is the component that stamps the cache, so without its version a "
                "cache that agrees with the reader today still cannot be trusted "
                "to agree with it after the next ingest tick.",
                "FIX: check that `aggregator` resolves on PATH and that its "
                "install carries lib/python3*/site-packages/aggregator/core/store.py.",
            )
        )

    # --- the two real skews.

    if cache_version is not None and reader_version is not None:
        if cache_version < reader_version:
            findings.append(
                Finding(
                    DEAD,
                    "AGGREGATOR RECALL IS DEAD: the cache is stamped at schema "
                    f"{cache_version} and the MCP reader refuses anything below "
                    f"{reader_version}, so every aggregator_search_memory call is "
                    "returning ok:false and an agent that relies on recall is "
                    "silently falling back to grepping transcripts.",
                    "FIX (forward only): bring the WRITER up to at least "
                    f"{target} — bump nixos-config's `aggregator-src` input "
                    "past the schema bump and rebuild — then let one ingest tick "
                    "re-stamp the cache. Do NOT run `aggregator status` to "
                    "investigate: every subcommand but `embed` calls migrate(), "
                    "which re-stamps the cache at the OLD version and destroys the "
                    "evidence.",
                )
            )
        elif cache_version > reader_version:
            # The mirror, and the half this probe used to call healthy. See
            # ``_ahead_cache_response`` in mcp.py: the gate is ``!=``, not
            # ``<``, so a cache ABOVE the reader is refused just as hard —
            # every call comes back ok:false while a probe comparing only
            # ``<`` prints "healthy" and exits 0. The remedy is the opposite
            # one, and naming the writer here would be destructive rather than
            # merely useless: an older writer re-stamps user_version DOWN.
            findings.append(
                Finding(
                    DEAD,
                    "AGGREGATOR RECALL IS DEAD: the cache is stamped at schema "
                    f"{cache_version} and the MCP reader understands exactly "
                    f"{reader_version}, so every aggregator_search_memory call is "
                    "returning ok:false. The READER is the lagging side here — "
                    "the cache was written by a build newer than the one Claude "
                    "Code is launching.",
                    "FIX: bring the READER up. The `aggregator-mcp` that "
                    "~/.claude.json's mcpServers.aggregator starts must be at "
                    f"least schema {target}: update or redeploy that build "
                    "and then RESTART the MCP server — a server process is held "
                    "for the life of the client that spawned it, so new code on "
                    "disk changes nothing until the process is replaced. Leave "
                    "the CACHE alone: it is the current side, and no command run "
                    "against the data can make an older reader understand a newer "
                    "schema.",
                )
            )
        elif meta_version is not None and meta_version != cache_version:
            # Only worth raising once the pragma itself is not already the
            # headline: a DEAD cache has a bigger problem than an inconsistent
            # second stamp, and two messages about one file compete.
            findings.append(
                Finding(
                    UNKNOWN,
                    f"the cache at {cache_db} carries two disagreeing schema stamps "
                    f"— PRAGMA user_version = {cache_version} but the meta table's "
                    f"schema_version row says {meta_version}. migrate() writes both "
                    "together, so something that is not migrate() has written this "
                    "file and its true schema cannot be vouched for. The MCP gate "
                    "reads the pragma only, so recall may still work.",
                    "FIX: re-run a full ingest from a current writer so migrate() "
                    "rewrites both stamps together.",
                )
            )

    if writer_version is not None:
        # BEHIND IS MEASURED AGAINST THE HIGHEST SIDE, NOT AGAINST THE READER.
        # The writer is the component that STAMPS, so any cache above it is a
        # cache it pulls down on the next tick. Comparing writer to reader
        # alone left one world entirely unspoken: cache 7, reader 6, writer 6.
        # The DEAD finding there correctly calls the cache the current side and
        # says bring the READER up to 7, never down-stamp the cache — and the
        # schema-6 writer beside it does precisely that down-stamp thirty
        # minutes later, exits 0 doing it, and no finding had named it. The
        # operator follows the remedy, redeploys the reader, and watches the
        # cache revert with nothing to explain why.
        #
        # AND THE READER IS NOT REQUIRED FOR THAT HALF. This block used to be
        # gated on a known reader, so an unresolvable or oversized
        # ~/.claude.json made a MEASURED writer-versus-cache skew unspeakable:
        # at cache 7, reader UNKNOWN, writer 6 the operator was sent to repair
        # a config file while the schema-6 writer down-stamped the cache to 6
        # on the next tick, exiting 0. Which reader Claude Code starts is
        # genuinely unknown there; that the writer will pull a schema-7 cache
        # down to 6 is not, and rule 2 (never call an unmeasured thing fine)
        # does not license staying silent about the thing that WAS measured.
        # The reader-relative branches still require a reader; the
        # cache-relative one does not.
        writer_target = _forward_target(reader_version, cache_version)
        if reader_version is not None and writer_version < reader_version:
            findings.append(
                Finding(
                    WILL_ROT,
                    "the aggregator WRITER IS BEHIND THE READER: the packaged writer "
                    f"builds schema {writer_version} while the MCP reader requires "
                    f"{reader_version}. migrate() ends by stamping PRAGMA user_version "
                    "with the writer's own constant, so the writer re-stamps the cache "
                    f"DOWN to {writer_version} on every ingest tick and exits 0 doing "
                    "it. Recall cannot stay healthy while this holds, and a hand-run "
                    "migration will revert within one timer period.",
                    "FIX (forward only): bump nixos-config's `aggregator-src` flake "
                    f"input to a rev whose SCHEMA_VERSION is at least {target} "
                    "and rebuild. Lowering the reader is not the alternative — a "
                    "newer reader wants columns an older cache does not have.",
                )
            )
        elif writer_target is not None and writer_version < writer_target:
            # Reached when the CACHE is above the writer and the reader is not
            # the lagging side: either the writer agrees with the reader or is
            # past it — a cache-ahead DEAD finding is then sitting beside this
            # one, and this is the half that says why fixing the reader alone
            # does not hold — or the reader could not be measured at all, in
            # which case the UNKNOWN finding is what sits beside it. Both
            # neighbours describe a machine whose cache is about to be pulled
            # down, and neither of them names the writer that does it.
            findings.append(
                Finding(
                    WILL_ROT,
                    "the aggregator WRITER WILL DOWN-STAMP THE CACHE: the packaged "
                    f"writer builds schema {writer_version} while the cache is "
                    f"stamped at {cache_version}. migrate() ends by stamping PRAGMA "
                    "user_version with the writer's own constant, so the next ingest "
                    f"tick re-stamps the cache DOWN to {writer_version} and exits 0 "
                    "doing it. Bringing the reader up on its own therefore does not "
                    "hold: the cache the reader was raised to meet is gone within "
                    "one timer period, and nothing fires to say so.",
                    "FIX (forward only): bump nixos-config's `aggregator-src` flake "
                    f"input to a rev whose SCHEMA_VERSION is at least {target} and "
                    "rebuild, together with the reader. Do NOT let an older writer "
                    "keep running against the newer cache to make the numbers meet "
                    "— down-stamping is the incident this check exists to detect.",
                )
            )
        elif reader_version is not None and writer_version > reader_version:
            # The same countdown pointing the other way, and it only became a
            # fault when the gate became ``!=``. While the reader refused
            # merely ``<``, a writer past the reader was the sanctioned repair
            # and flagging it would have argued against this file's own remedy.
            # Now the next tick stamps the cache ABOVE the reader, and the
            # reader refuses that too — so this is DEAD on a timer, which is
            # WILL_ROT by definition.
            findings.append(
                Finding(
                    WILL_ROT,
                    "the aggregator WRITER IS AHEAD OF THE READER: the packaged "
                    f"writer builds schema {writer_version} while the MCP reader "
                    f"understands exactly {reader_version}. migrate() ends by "
                    "stamping PRAGMA user_version with the writer's own constant, "
                    f"so the next ingest tick stamps the cache at {writer_version} "
                    "— which the reader refuses just as hard as one that is too "
                    "old. Recall may answer this minute and will be returning "
                    "ok:false within one timer period.",
                    "FIX: bring the READER up to at least "
                    f"{target} — the `aggregator-mcp` that ~/.claude.json's "
                    "mcpServers.aggregator starts, redeployed and then RESTARTED, "
                    "since a running server keeps the code it was launched with. "
                    "Do NOT pin the writer back down to make the numbers meet: "
                    "that re-stamps caches downward and is the incident this check "
                    "exists to detect.",
                )
            )

    states = sorted({f.state for f in findings}) or [FINE]
    primary = next(s for s in _SEVERITY_ORDER if s in states or s == FINE)
    return Verdict(
        state=primary,
        severity=_SEVERITY_OF[primary],
        states=states,
        findings=findings,
        cache_version=cache_version,
        cache_meta_version=meta_version,
        reader_version=reader_version,
        writer_version=writer_version,
        cache_db=str(cache_db),
        reader_dir=str(reader_dir) if reader_dir is not None else None,
        writer_bin=str(writer_bin) if writer_bin is not None else None,
    )


# --- entry point ------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aggregator-schema-probe",
        description=(
            "Compare the aggregator cache's schema stamp against the MCP "
            "reader's requirement and the packaged writer's version. Read-only: "
            "never migrates, never runs the aggregator CLI."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the full verdict as JSON (default; --text for one line)",
    )
    parser.add_argument(
        "--text", action="store_true", help="emit one human-readable line instead"
    )
    args = parser.parse_args(argv)

    verdict = probe()
    if args.text:
        print(verdict.explain())
    else:
        json.dump(verdict.to_dict(), sys.stdout, indent=2)
        sys.stdout.write("\n")
    return verdict.exit_code()


if __name__ == "__main__":
    sys.exit(main())
