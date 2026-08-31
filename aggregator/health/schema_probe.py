"""Detect the schema skew between the cache, the MCP reader, and the writer.

WHAT WENT WRONG, AND WHY NOTHING SAW IT

Three components share one SQLite cache and each of them knows only its own
half of the contract:

  * the READER — ``aggregator-mcp``, run by Claude Code out of the live
    working tree — opens the cache ``mode=ro`` and refuses every call when
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

STATES, AND WHY FOUR RATHER THAN A BOOLEAN

``FINE``     cache >= reader's requirement AND writer >= it. Silent.
``DEAD``     cache < requirement. Recall is refusing RIGHT NOW.
``WILL_ROT`` writer < requirement. Recall may work this minute, but the writer
             re-stamps the cache down to its own version on the next tick, so
             a hand-run migration reverts within thirty minutes. This is the
             state a two-quantity check cannot see, and it is the one that
             explains why the incident kept coming back.
``UNKNOWN``  some quantity could not be read.

``DEAD`` and ``WILL_ROT`` co-occur — that was the live incident — and both
have to survive into the report, because they have different remedies and
fixing only the first leaves a machine that breaks itself again on the next
tick.

THE REMEDY IS ALWAYS FORWARD

Every message here says: bring the WRITER up. Never lower the reader. Two
components disagreeing on a version is repaired by moving the lagging side up,
and offering "or make the reader accept the old schema" as the other arm of a
choice is not a neutral presentation of options — the schema-6 reader wants
columns a schema-5 cache does not have, so accepting 5 means reading a cache
that cannot answer, which is the failure wearing a different hat.

CONSUMERS

Two, and they share this one implementation rather than each growing their
own copy of the predicate — a detector that disagrees with itself about
whether the machine is healthy is worse than either half alone:

  * a systemd **user** timer, which reaches the operator through ``notify-send``
    on a machine with no agent session open;
  * a Claude Code **SessionStart** hook, which reaches the actual victim — a
    session that would otherwise believe recall works.

Both invoke this file as a bare script under plain ``python3``:
``python3 .../schema_probe.py --json``. It is therefore STDLIB ONLY and must
stay that way. Importing the aggregator package here would drag in torch and
sentence-transformers, and a SessionStart hook that blows its budget has its
output DISCARDED — which for a health check is the same as never noticing.
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


def resolve_reader_dir(env: dict[str, str] | None = None) -> Path | None:
    """Which checkout the MCP reader actually runs from.

    Asked of ``~/.claude.json`` rather than assumed, because that file is what
    Claude Code executes — ``{"command": "uv", "args": ["run", "--directory",
    "<dir>", "aggregator-mcp"]}`` — so it is the only source that cannot be
    out of date with respect to the reader under test. A hard-coded
    ``~/Repos/aggregator`` would keep reporting on a checkout the reader had
    stopped using, and would do it silently.

    Falls back to this file's own checkout, but only when that checkout has a
    ``pyproject.toml``: installed into site-packages this module sits beside a
    ``core/store.py`` too, and reading THAT as "the reader's requirement"
    would compare the writer against itself and report every skew as healthy.
    The discriminator is cheap and the failure it prevents is total.
    """
    env = os.environ if env is None else env
    override = env.get(READER_DIR_ENV)
    if override:
        return Path(override).expanduser()

    home = env.get("HOME") or str(Path.home())
    try:
        with open(os.path.join(home, ".claude.json"), "rb") as fh:
            doc = json.loads(fh.read(_SOURCE_SCAN_LIMIT * 8).decode("utf-8", "replace"))
        args = (((doc.get("mcpServers") or {}).get("aggregator") or {}).get("args")) or []
        for i, a in enumerate(args):
            if a == "--directory" and i + 1 < len(args):
                return Path(str(args[i + 1])).expanduser()
    except (OSError, ValueError, AttributeError, TypeError):
        # Missing, unreadable or a shape this does not know. Fall through to
        # the checkout fallback; if that fails too the caller reports UNKNOWN,
        # which is the correct answer and not an error to raise here.
        pass

    own = Path(__file__).resolve().parent.parent.parent
    if (own / "pyproject.toml").is_file():
        return own
    return None


def resolve_writer_bin(env: dict[str, str] | None = None) -> Path | None:
    """The ``aggregator`` a human — or the ingest timer — would actually run.

    ``shutil.which`` semantics, spelled out rather than imported so this stays
    a single self-contained file, and so the ``PATH`` it searches is the one
    passed in rather than the process's.
    """
    env = os.environ if env is None else env
    override = env.get(WRITER_BIN_ENV)
    if override:
        return Path(override).expanduser()
    for d in (env.get("PATH") or "").split(os.pathsep):
        if not d:
            continue
        candidate = Path(d) / "aggregator"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


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


def read_reader_version(reader_dir: Path | None) -> int | None:
    """The version the MCP reader will refuse anything below.

    Read as text out of the checkout's ``store.py``. Emphatically not an
    import: ``aggregator.core.store`` pulls sentence-transformers and torch,
    which is seconds of model-loading machinery, and both consumers here run
    on budgets measured in single-digit seconds.
    """
    if reader_dir is None:
        return None
    return _read_schema_const(Path(reader_dir) / "aggregator" / "core" / "store.py")


def read_writer_version(writer_bin: Path | None) -> int | None:
    """The version the packaged writer will stamp the cache with.

    Obtained by READING the writer's packaged source, never by executing it.
    Three reasons, all load-bearing: running it would migrate the cache (rule
    1 at the top of this file); a wedged install would hang the probe, which
    on a hook budget means the output is discarded and nothing is reported;
    and the binary being broken is itself one of the conditions the probe must
    survive in order to speak.

    The path from binary to source is a Nix wrapper chain. On this host
    ``/etc/profiles/.../bin/aggregator`` is a shell script whose last line
    execs a second ``bin/aggregator`` inside an ``-env`` derivation, and only
    that far end carries ``lib/python3.11/site-packages``. Every prefix along
    the way is tried, in order, so a plain venv (no wrapper at all) and a
    two-hop Nix chain both resolve without a special case.
    """
    if writer_bin is None:
        return None

    prefixes: list[Path] = []
    seen: set[str] = set()
    current: Path | None = Path(writer_bin)

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

        # ``<prefix>/bin/aggregator`` -> ``<prefix>``. Recorded for every hop
        # because which one owns site-packages is not knowable in advance.
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

    for prefix in prefixes:
        for lib in sorted(prefix.glob("lib/python3*/site-packages")):
            version = _read_schema_const(lib / "aggregator" / "core" / "store.py")
            if version is not None:
                return version
    return None


# --- the predicate ----------------------------------------------------------


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

    # --- could-not-measure first. Each of these makes some later comparison
    # unanswerable, and an unanswerable comparison must never be quietly
    # skipped into silence.

    if reader_version is None:
        findings.append(
            Finding(
                UNKNOWN,
                "aggregator recall health CANNOT BE VERIFIED: the MCP reader's "
                "required schema version could not be read from "
                f"{reader_dir or '(no checkout located)'} — expected "
                "`SCHEMA_VERSION = <n>` in aggregator/core/store.py. Without it "
                "there is no number to compare the cache and the writer against, "
                "so nothing here can be called healthy.",
                "FIX: confirm the checkout named by ~/.claude.json's "
                "mcpServers.aggregator `--directory` argument exists and is a "
                "real aggregator tree, or set AGGREGATOR_READER_DIR.",
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
                    f"{reader_version} — bump nixos-config's `aggregator-src` input "
                    "past the schema bump and rebuild — then let one ingest tick "
                    "re-stamp the cache. Do NOT run `aggregator status` to "
                    "investigate: every subcommand but `embed` calls migrate(), "
                    "which re-stamps the cache at the OLD version and destroys the "
                    "evidence.",
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

    if writer_version is not None and reader_version is not None and writer_version < reader_version:
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
                f"input to a rev whose SCHEMA_VERSION is at least {reader_version} "
                "and rebuild. Lowering the reader is not the alternative — the "
                "schema-6 reader needs columns a schema-5 cache does not have.",
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
