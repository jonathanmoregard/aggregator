"""Scrubber: applied pre-store AND pre-return (spec §Security constraint 3).

Two layers:

1. **Secrets**, in two passes, both mandatory:

   a. **Regex floor** (``SECRET_PATTERNS``) — the token formats most likely
      to appear in this user's history (OpenAI, Anthropic incl. OAuth/admin,
      GitHub classic + fine-grained PATs, Slack, Tavily, Cloudflare v2, Google
      OAuth client secret, JWT, AWS access key, PEM headers). Pure Python,
      always on.
   b. **gitleaks** (``gitleaks stdin`` with the vendored
      ``secret-rules.toml``: every default rule — ~220 providers — plus the
      custom formats gitleaks lacks). Each finding's secret string is
      replaced by ``[REDACTED:<rule-id>]``. FAIL CLOSED: if the binary is
      missing or errors, ``scrub``/``scrub_many`` raise
      :class:`ScrubUnavailableError` and the caller must neither store nor
      return the text. ``AGGREGATOR_GITLEAKS`` names the binary explicitly
      (the Nix wrapper sets it to a store path); otherwise ``PATH`` is used.

   A gitleaks call costs ~100 ms of process start before it reads a byte, so
   callers with many texts use :func:`scrub_many`, which scans them in ONE
   call and gives the same per-text result as ``scrub`` on each.

2. **PII detection**:
   - If `presidio_analyzer` + `presidio_anonymizer` are importable, use them
     for entity-based PII detection (names, orgs, locations, credit cards, ...).
   - Regardless of Presidio availability, always run the explicit regex
     patterns for SSN / email / phone — Presidio's stock recognizers do NOT
     cover these formats reliably out of the box (verified: Presidio 2.2.x
     with en_core_web_lg detects EMAIL_ADDRESS/URL but misses SSN and
     +1-555-… phones). Redundancy here is the point.

PRESIDIO IS INITIALISED LAZILY, AND THAT IS A STARTUP BUDGET, NOT A STYLE
CHOICE. This module used to build ``AnalyzerEngine()`` and ``AnonymizerEngine()``
at import. Presidio's predefined recognizers import ``transformers`` (hence
``torch``), and the spaCy-model probe imports ``spacy`` (hence ``thinc``, hence
``torch`` again): 49.0 s of the 57.6 s that importing ``aggregator.mcp`` cost on
the user's laptop, measured with ``PYTHONPROFILEIMPORTTIME``. Claude Code gives
an MCP server 30 s to answer ``initialize``, so the recall server stopped
connecting at all. ``ensure_presidio_ready`` now does that work on first use —
once, under a lock, blocking — and ``start_background_init`` lets a process that
must answer something else first pay for it in parallel. ``scrub`` is unchanged
in behaviour: the first call initialises, every later call is one atomic read.

`scrub()` is intentionally idempotent — the pre-store + pre-return arrangement
means the same text may be scrubbed twice; the second pass MUST NOT mangle the
first pass's ``[REDACTED:*]`` tokens. All patterns are chosen so redactions
don't themselves match any other pattern.

Returns a structured ``ScrubResult`` with counts by finding-type. Counts are
metadata only — the redacted content of individual findings is never surfaced
(spec §Error handling: log the type, not the value).
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from aggregator.core.gitleaks_driver import GitleaksError

#: The scrubber's fail-closed error: the scan did not complete, so the text
#: must be neither stored nor returned. An alias, not a subclass, so it is the
#: same class after ``importlib.reload`` of this module (tests reload it).
ScrubUnavailableError = GitleaksError

#: Bound at import, not looked up per call: a test that stubs the global
#: ``subprocess.run``/``shutil.which`` seam for ANOTHER tool (the ``claude``
#: CLI in test_cli_tag.py) must not silently stub the secret scanner too.
_subprocess_run = subprocess.run
_which = shutil.which

log = logging.getLogger(__name__)


# --- Presidio detection (optional dependency; regex fallback covers CI/dev env)

def _spacy_model_present() -> bool:
    """True when the spaCy model Presidio is configured for is already installed.

    Presidio builds its NLP engine eagerly and, if the model is missing, calls
    ``spacy.cli.download`` — which shells out to pip and, on failure, calls
    ``sys.exit(1)``. Two problems with letting that happen at import time:

    * ``sys.exit`` raises ``SystemExit``, a ``BaseException``. The ``except
      Exception`` below does NOT catch it, so a missing model took the whole
      process down at import instead of taking the regex fallback this module
      documents. That is how CI died: pytest aborted with ``INTERNALERROR>
      SystemExit: 1`` while merely collecting tests.
    * Even when it succeeds it is an unannounced network install triggered by
      an ``import``.

    So: look before leaping, and never let the download run. Any failure to
    answer the question is treated as "not present" — the regex path is the
    safe direction to fail in.
    """
    try:
        import spacy.util
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        configured = {
            m["model_name"]
            for m in NlpEngineProvider().nlp_configuration.get("models", [])
        }
        installed = set(spacy.util.get_installed_models())
        return bool(configured) and configured <= installed
    except Exception:  # noqa: BLE001 -- unanswerable -> assume absent
        return False


#: Engines, and whether they are usable. Written exactly once, by
#: ``ensure_presidio_ready`` under ``_INIT_LOCK``; read everywhere else only
#: after ``_INIT_DONE`` is set, which is what publishes those writes.
_analyzer: object | None = None
_anonymizer: object | None = None
_PRESIDIO_OK = False

#: Held for the whole of one initialisation attempt. Serialises concurrent
#: first callers so the engines are built exactly once.
_INIT_LOCK = threading.Lock()

#: Set once initialisation has finished, success or failure. The lock-free
#: fast path: ``scrub`` runs per stored row, hundreds of thousands of times per
#: ingest, and must not take a mutex to learn a decision that was made once.
_INIT_DONE = threading.Event()

#: The warm-up thread, if one was started. Guarded by its OWN lock — reusing
#: ``_INIT_LOCK`` here would make ``start_background_init`` block behind the
#: initialisation it exists to avoid waiting for.
_WARM_LOCK = threading.Lock()
_warm_thread: threading.Thread | None = None


def _build_presidio_engines() -> tuple[object, object]:
    """Import Presidio and construct the two engines. SLOW — SECONDS TO A MINUTE.

    Every heavy import in this module lives here and nowhere else. Presidio's
    predefined recognizers import ``transformers`` (hence ``torch``), and
    ``_spacy_model_present`` imports ``spacy`` (hence ``thinc``, hence
    ``torch`` again): measured at 49 s of the 57 s it used to take to import
    ``aggregator.mcp``, which is why none of it may happen at module scope.

    Raises rather than returning a sentinel — the caller owns the fallback.
    """
    if not _spacy_model_present():
        raise RuntimeError(
            "spaCy model for Presidio is not installed; install it with "
            "`python -m spacy download en_core_web_lg`"
        )
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine

    return AnalyzerEngine(), AnonymizerEngine()


def ensure_presidio_ready() -> bool:
    """Initialise Presidio once, blocking until the answer is known.

    Idempotent and thread-safe. The first caller builds the engines; every
    other caller — including one that arrives mid-build — waits on the lock and
    then returns the same answer. Failure is not an error: it degrades to the
    regex-only path this module documents, and logs the reason exactly once,
    because initialisation runs exactly once. That holds for a failure this
    function swallows AND for one it re-raises — either way the decision is
    latched for the life of the process, so either way it is announced.
    """
    global _analyzer, _anonymizer, _PRESIDIO_OK
    if _INIT_DONE.is_set():
        return _PRESIDIO_OK
    with _INIT_LOCK:
        if _INIT_DONE.is_set():
            return _PRESIDIO_OK
        try:
            _analyzer, _anonymizer = _build_presidio_engines()
            _PRESIDIO_OK = True
        # SystemExit is listed explicitly: it is a BaseException, so it is not
        # covered by `except Exception`, and it is exactly what an in-process
        # spaCy download raises. KeyboardInterrupt is deliberately NOT caught.
        except (Exception, SystemExit) as e:  # noqa: BLE001 -- init failure -> regex
            log.warning(
                "Presidio unavailable (%s); PII scrubbing will use regex fallback only",
                e,
            )
            _analyzer = None
            _anonymizer = None
            _PRESIDIO_OK = False
        # Everything the clause above does not cover — KeyboardInterrupt,
        # asyncio.CancelledError — still lands on the `finally` below, which
        # latches the decision for the life of the process: PII coverage
        # narrows to regex and never widens again. That must not happen in
        # silence. Announce it with the same sentence, then re-raise, so a
        # KeyboardInterrupt keeps killing the process as the user asked.
        except BaseException as e:
            log.warning(
                "Presidio unavailable (%s); PII scrubbing will use regex fallback only",
                e,
            )
            raise
        finally:
            # Unconditional, so a BaseException that propagates still releases
            # every waiter instead of deadlocking them.
            _INIT_DONE.set()
    return _PRESIDIO_OK


def start_background_init() -> threading.Thread | None:
    """Kick initialisation off on a daemon thread; return it, or ``None``.

    For a process that must answer something else first — the MCP server's
    ``initialize`` handshake — and would rather pay the model load in parallel
    with serving than in front of it. Idempotent: repeated calls return the
    same thread, and a call made after initialisation has already finished
    returns ``None`` and starts nothing. Daemon, because a warm-up must never
    hold the process open at exit.
    """
    global _warm_thread
    if _INIT_DONE.is_set():
        return None
    with _WARM_LOCK:
        if _warm_thread is None:
            _warm_thread = threading.Thread(
                target=ensure_presidio_ready,
                name="presidio-warmup",
                daemon=True,
            )
            _warm_thread.start()
        return _warm_thread


# --- Secret patterns
# Order matters: more-specific prefixes must run BEFORE more-general ones,
# because we mutate `text` sequentially. Concretely: anthropic keys (`sk-ant-`)
# would otherwise match the generic openai `sk-…` pattern first and be
# mis-classified.
SECRET_PATTERNS: dict[str, re.Pattern] = {
    # Anthropic variants first, most specific first, all BEFORE the generic
    # anthropic and openai `sk-` patterns so each is counted under its own
    # label. OAuth (Claude subscription login) and admin keys are separate
    # credential types; api03 is the exact console-key shape.
    "anthropic_oauth": re.compile(r"sk-ant-oat\d{2}-[A-Za-z0-9_-]{20,}"),
    "anthropic_admin": re.compile(r"sk-ant-admin\d{2}-[A-Za-z0-9_-]{20,}"),
    "anthropic_api03": re.compile(r"sk-ant-api03-[A-Za-z0-9_-]{93}AA"),
    # Anthropic: sk-ant-… . Match this BEFORE the generic openai `sk-` pattern
    # so anthropic keys are counted under the correct label.
    "anthropic_key": re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    # OpenAI: sk-… (optionally sk-proj-…). Now safe to run — anthropic already
    # consumed its variant above.
    "openai_key": re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    # GitHub fine-grained PAT: github_pat_<22>_<59> = 82 word chars. `{30,}`
    # rather than `{82}`, measured against the live store (2026-10-11): all
    # 14 `github_pat_11…` strings in it had 43 or 75 chars after the prefix —
    # truncated or hand-made, which gitleaks' exact-length rule never
    # flags — while every identifier that merely contains the prefix
    # (`test_github_pat_redacted`) has under 30.
    "github_fine_grained_pat": re.compile(r"github_pat_[A-Za-z0-9_]{30,}"),
    # GitHub PAT / OAuth / server token / user token / refresh token.
    "github_pat": re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    # Slack bot / app / user / refresh / legacy tokens.
    "slack_token": re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    # Tavily: tvly-[dev-|prod-]<body>.
    "tavily_key": re.compile(r"tvly-[A-Za-z0-9_-]{20,}"),
    # Cloudflare scannable tokens: user/account API token, global API key.
    "cloudflare_token": re.compile(r"cf(?:ut|at|k)_[A-Za-z0-9]{40,}"),
    # Google OAuth client secret.
    "google_oauth_secret": re.compile(r"GOCSPX-[A-Za-z0-9_-]{20,}"),
    # JWT: header.payload.signature — three base64url segments, first starts
    # with "eyJ" (base64 of `{"`).
    "jwt": re.compile(r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
    # AWS access key ID.
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    # PEM private key. When the matching END line follows, the WHOLE block goes: redacting
    # only the header would leave the key material in place AND hide it from
    # the gitleaks pass, whose private-key rule needs the header to fire.
    # Without an END (a header quoted in code, a truncated paste) only the
    # header goes, rather than everything after it.
    "generic_pem_start": re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----"
        r"(?:[\s\S]{0,20000}?-----END [A-Z ]*PRIVATE KEY-----)?"
    ),
}

# --- PII regex patterns (always run; Presidio is additive)
#
# Advisor round-1 MEDIUM: added credit_card (Luhn-shape via a custom check
# below), ipv4, ipv6, iban. These are not exhaustive — Presidio adds broader
# coverage when installed. Order matters: `iban` must run BEFORE `phone`
# because the phone regex would otherwise nibble the trailing digits of an
# IBAN (verified via the failing test we wrote first).
#
# `credit_card` is NOT in this dict because it uses Luhn-check-gated
# substitution rather than a bare regex sub. See ``_scrub_credit_cards``.
PII_PATTERNS: dict[str, re.Pattern] = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    # IBAN: 2 letters + 2 digits + 11..30 alphanumerics. Boundaries prevent
    # partial matches inside longer alphanumeric runs.
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
    # IPv6: full 8-group form (misses ``::`` shorthand — good enough for
    # v1). Round-2 MEDIUM: pre-fix ``\b`` anchors let the pattern fire on
    # benign hex-and-colon chains like ``sha256:aa:bb:cc:dd:ee:ff:11:22``
    # because ``\b`` sits between ``:`` (non-word) and hex (word), so the
    # prefix boundary matched inside the chain. Post-fix uses lookarounds
    # that reject any hex-or-colon character on either side of the run,
    # so the ``sha256:`` prefix disqualifies the match.
    "ipv6": re.compile(
        r"(?<![0-9A-Fa-f:])(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}(?![0-9A-Fa-f:])"
    ),
    # IPv4: four dotted octets, each 0-255.
    "ipv4": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\b"
    ),
    # Marker so PII_PATTERNS test can assert the credit-card path is present.
    # The actual substitution is Luhn-gated in ``_scrub_credit_cards``; this
    # regex would be too noisy on its own (would flag any 16-digit run).
    "credit_card": re.compile(
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
    ),
    # Loose international phone: optional +CC, then 3-3-4 with -/space/nothing.
    "phone": re.compile(
        r"\+?\d{1,3}[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{4}"
    ),
}


def _luhn_ok(digits: str) -> bool:
    """Standard Luhn checksum on a digits-only string."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _scrub_credit_cards(text: str, counts: dict[str, int]) -> str:
    """Regex-scan for 16-digit runs, then redact only those that pass Luhn.

    Prevents flagging arbitrary 16-digit strings (order numbers, hashes) that
    would false-positive under a bare regex.
    """
    pat = PII_PATTERNS["credit_card"]

    def _repl(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        if len(digits) == 16 and _luhn_ok(digits):
            counts["credit_card"] = counts.get("credit_card", 0) + 1
            return "[REDACTED:credit_card]"
        return m.group(0)

    return pat.sub(_repl, text)


@dataclass
class ScrubResult:
    """Return shape of ``scrub()``. Counts are metadata only — the redacted
    values themselves are intentionally not surfaced."""

    text: str
    counts: dict[str, int] = field(default_factory=dict)


def _run_patterns(
    text: str, patterns: dict[str, re.Pattern], counts: dict[str, int]
) -> str:
    for label, pat in patterns.items():
        # Use a default-arg closure to capture `label` correctly (avoids the
        # classic late-binding trap in for-loops).
        def _repl(_m: re.Match, _l: str = label) -> str:
            counts[_l] = counts.get(_l, 0) + 1
            return f"[REDACTED:{_l}]"

        text = pat.sub(_repl, text)
    return text


# --- gitleaks layer ----------------------------------------------------------

#: The vendored rules: gitleaks' defaults plus the formats it lacks. A
#: byte-identical copy of ``hooks/secret-rules.toml`` in the ``~/.claude``
#: config repo, which redacts tool output with the same rules; the copies are
#: pinned to one checksum by ``tests/core/test_scrub_gitleaks.py``.
GITLEAKS_RULES = Path(__file__).with_name("secret-rules.toml")

#: Names the gitleaks binary. Set by the Nix wrapper to a store path, because
#: a systemd unit's PATH is minimal and must not decide whether text is
#: scanned. Unset: ``PATH`` lookup (dev shells, CI).
GITLEAKS_ENV = "AGGREGATOR_GITLEAKS"

#: Bytes per scanned FILE, and it is a CORRECTNESS bound, not a tuning knob.
#: gitleaks 8.30 reads a file in 100,000-byte reads (``sources/file.go``
#: ``defaultBufferSize``) and, when no blank line follows within 25,000 bytes
#: (``sources/common.go`` ``maxPeekSize``), cuts the fragment at an arbitrary
#: byte: a token on the cut is never seen. Measured through stdin: 12,000
#: tokens on one line, 8 missed. A regular file under one read is exactly one
#: fragment, so nothing is cut. Anything longer is split by us into
#: overlapping windows (``_WINDOW_OVERLAP``), one file each.
_PIECE_BYTES = 96_000

#: How much one gitleaks call is handed, summed over its files. Bounds the
#: tmpfs footprint and the work lost to a timeout, nothing else.
_GITLEAKS_CALL_BYTES = 8 * 1024 * 1024

#: How far consecutive windows of one long text overlap, in characters. A
#: secret shorter than this that straddles one window edge lies wholly inside
#: the neighbouring window. Longer than any single-token rule matches; PEM
#: blocks and JWTs, which can be longer, are also caught by the regex floor.
_WINDOW_OVERLAP = 16_000

#: Go threads per gitleaks call. Unbounded it takes every core and burns ~3x
#: the CPU for the same scan (8.6 MB of real rows: 17.6 s CPU on 24 threads,
#: 6.1 s on 2, 5.5 s on 1); 2 keeps most of the wall-clock win at near-minimal
#: CPU, which matters for a job that runs on a timer in the background.
_GITLEAKS_THREADS = "2"

#: Wall-clock bound for one call; only a hung scanner reaches it.
_GITLEAKS_TIMEOUT_S = 300

#: Prepended to every scanned file. gitleaks SKIPS a file whose first read
#: sniffs as an ``application/*`` type (``sources/file.go``, h2non/filetype):
#: text starting ``%PDF``, ``MZ``, ``{\rtf``, ``SQLite format 3``, or holding
#: ``ustar`` at byte 257, was reported clean with zero bytes scanned. ``BM``
#: is the BMP magic; image types are matched before application and archive
#: ones, so every file sniffs as ``image/bmp`` and is scanned whatever
#: follows. The blank line keeps the text on lines of its own. Backed by the
#: scanned-bytes check in ``_run_gitleaks``, which refuses a call that did not
#: read everything it was given.
_HEADER = "BM aggregator-scrub\n\n"
_HEADER_NEWLINES = _HEADER.count("\n")

#: Where each call's files and working directory go: tmpfs when the host has
#: it (unscrubbed text then never touches a disk), else the default temp dir.
_RAM_DIR = "/dev/shm" if os.access("/dev/shm", os.W_OK) else None

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_SCANNED = re.compile(r"scanned ~(\d+) bytes")

#: gitleaks' JSON report key for the full matched text. A JSON key, not SQL —
#: declared in tests/test_fts5_match_site_enumeration.py's ``_NOT_SQL``.
_REPORT_MATCH_KEY = "Match"
_NON_SPACE_RUN = re.compile(r"\S+")


def gitleaks_binary() -> str:
    """The gitleaks executable to run, or :class:`ScrubUnavailableError`."""
    explicit = os.environ.get(GITLEAKS_ENV, "").strip()
    if explicit:
        if os.path.isfile(explicit) and os.access(explicit, os.X_OK):
            return explicit
        raise ScrubUnavailableError(
            f"{GITLEAKS_ENV}={explicit} is not an executable file"
        )
    found = _which("gitleaks")
    if found is None:
        raise ScrubUnavailableError(
            f"gitleaks not found: set {GITLEAKS_ENV} or put gitleaks on PATH"
        )
    return found


@dataclass(frozen=True)
class _Finding:
    """One gitleaks finding. Lines are 1-based; columns are 1-based BYTE
    offsets into their line, end inclusive (normalised in ``_run_gitleaks``)."""

    rule: str
    secret: str
    start_line: int
    end_line: int
    start_col: int
    end_col: int
    decoded: bool


def _run_gitleaks(texts: Sequence[str]) -> list[list[_Finding]]:
    """Scan each text as its OWN file in one gitleaks call; findings per text.

    ONE FILE PER TEXT, NOT ONE STREAM. gitleaks applies a rule only when one
    of its keywords occurs in the same fragment, so texts concatenated into
    one stream switch rules on for each other: a ``sourcegraph`` in one text
    made every 40-hex git SHA in another a "sourcegraph-access-token"
    (measured on real rows). A file per text keeps each verdict the one that
    text gets alone. Each text must fit ``_PIECE_BYTES``.

    Raises ScrubUnavailableError on any scanner fault, including a scan that
    did not read every byte it was given.
    """
    cmd = [
        gitleaks_binary(),
        "dir",
        "--no-banner",
        "--no-color",
        "--config", str(GITLEAKS_RULES),
        "--report-path", "-",
        "--report-format", "json",
        # info, for the "scanned ~N bytes" line checked below.
        "--log-level", "info",
        # Findings are the expected outcome, not an error. With the default
        # (1) a finding and a crash share an exit code; with 0, any non-zero
        # status is a failure.
        "--exit-code", "0",
        # Scanned text must not be able to opt itself out: a transcript line
        # containing `gitleaks:allow` would otherwise be skipped...
        "--ignore-gitleaks-allow",
        # ...nor may a .gitleaksignore. This flag alone is NOT enough:
        # cmd/root.go also loads `./.gitleaksignore` and
        # `<scan dir>/.gitleaksignore`, hence the fresh, empty cwd below and a
        # scan dir holding nothing but our numbered files.
        "--gitleaks-ignore-path", os.devnull,
        "in",
    ]
    total = 0
    try:
        with tempfile.TemporaryDirectory(prefix="aggregator-scrub-", dir=_RAM_DIR) as work:
            os.mkdir(os.path.join(work, "in"), 0o700)
            for i, text in enumerate(texts):
                data = (_HEADER + text).encode("utf-8", "replace")
                if len(data) > _PIECE_BYTES + len(_HEADER):
                    # A programming error, but one that would silently reopen
                    # the cut hole, so it fails closed like any scanner fault.
                    raise ScrubUnavailableError("gitleaks input piece exceeds one read")
                with open(os.path.join(work, "in", f"{i}.txt"), "wb") as fh:
                    fh.write(data)
                total += len(data)
            proc = _subprocess_run(  # noqa: S603 - fixed argv, no shell
                cmd,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=_GITLEAKS_TIMEOUT_S,
                check=False,
                cwd=work,
                env={**os.environ, "GOMAXPROCS": _GITLEAKS_THREADS},
            )
    except subprocess.TimeoutExpired as e:
        raise ScrubUnavailableError(
            f"gitleaks timed out after {_GITLEAKS_TIMEOUT_S}s"
        ) from e
    except OSError as e:
        raise ScrubUnavailableError(
            f"gitleaks could not be started ({type(e).__name__})"
        ) from e
    stderr = _ANSI.sub("", proc.stderr.decode("utf-8", "replace"))
    if proc.returncode != 0:
        # gitleaks' own diagnostic (config/IO errors), never scanned text —
        # and passed through the regex floor anyway, so even a scanner that
        # one day echoes input cannot put a secret in a log line.
        lines = stderr.strip().splitlines()
        first = next((ln for ln in lines if " FTL " in ln or " ERR " in ln), lines[0] if lines else "")
        detail = _run_patterns(first[:200], SECRET_PATTERNS, {})
        raise ScrubUnavailableError(
            f"gitleaks exited {proc.returncode}: {detail or '(no diagnostic)'}"
        )
    # An empty report is only "clean" if gitleaks actually READ the input.
    # Its skip paths (binary sniffing, path allowlists) exit 0 with `[]`,
    # which must never pass for a clean scan.
    scanned = _SCANNED.search(stderr)
    if scanned is None or int(scanned.group(1)) < total:
        raise ScrubUnavailableError(
            f"gitleaks scanned {scanned.group(1) if scanned else 'an unknown number of'}"
            f" of {total} bytes"
        )
    try:
        report = json.loads(proc.stdout or b"[]")
    except ValueError as e:
        raise ScrubUnavailableError("gitleaks report was not valid JSON") from e
    out: list[list[_Finding]] = [[] for _ in texts]
    for f in report or ():
        name = os.path.basename(str(f.get("File") or ""))
        stem = name[: -len(".txt")] if name.endswith(".txt") else ""
        if not stem.isdigit() or int(stem) >= len(texts):
            raise ScrubUnavailableError("gitleaks reported a finding in an unknown file")
        match = str(f.get(_REPORT_MATCH_KEY) or "")
        secret = f.get("Secret") or match
        start_line = int(f.get("StartLine") or 1)
        # EndLine is NOT trustworthy for a multi-line match: 8.30 reports a
        # whole PEM block as StartLine == EndLine (measured). The match text
        # itself says how many lines it spans.
        spanned = max(match.count("\n"), secret.count("\n"))
        end_line = max(int(f.get("EndLine") or start_line), start_line + spanned)
        # gitleaks 8.30 counts the preceding newline into the column on every
        # line but the first (measured: the same token at the start of line 1
        # and of line 3 reports column 1 and column 2), so take it back out.
        # Every finding is past the header, so never on line 1.
        start_col = int(f.get("StartColumn") or 1) - (start_line > 1)
        end_col = int(f.get("EndColumn") or start_col) - (end_line > 1)
        out[int(stem)].append(
            _Finding(
                rule=str(f.get("RuleID") or "gitleaks"),
                secret=secret,
                # Text-local: the header's lines come off.
                start_line=start_line - _HEADER_NEWLINES,
                end_line=end_line - _HEADER_NEWLINES,
                start_col=start_col,
                end_col=end_col,
                decoded=any(
                    str(t).startswith("decoded:") for t in (f.get("Tags") or ())
                ),
            )
        )
    return out


def _redact_at(lines: list[str], f: _Finding, marker: str) -> None:
    """Redact a finding by POSITION, in place on ``lines`` (0-based list).

    For the findings a string replace cannot reach: DECODED ones (gitleaks
    reports the decoded secret, which does not occur in the text — the
    encoded run does) and any whose secret is not in the text verbatim. One
    line: every whitespace-delimited run touching the reported columns.
    Several lines: the whole span, which is coarse and certain.
    """
    first, last = f.start_line - 1, f.end_line - 1
    if not (0 <= first <= last < len(lines)):
        return
    if first != last:
        lines[first : last + 1] = [marker]
        return
    line = lines[first]
    raw = line.encode("utf-8", "replace")
    # Byte columns -> character offsets on this line.
    lo = len(raw[: max(f.start_col - 1, 0)].decode("utf-8", "ignore"))
    hi = len(raw[: max(f.end_col, f.start_col)].decode("utf-8", "ignore"))
    runs = [m for m in _NON_SPACE_RUN.finditer(line) if m.start() < hi and m.end() > lo]
    if not runs:
        lines[first] = marker
        return
    lines[first] = line[: runs[0].start()] + marker + line[runs[-1].end() :]


def _apply_findings(text: str, findings: Sequence[_Finding], counts: dict[str, int]) -> str:
    """Redact ``findings`` (line numbers local to ``text``) from ``text``."""
    if not findings:
        return text
    by_string: list[_Finding] = []
    by_position: list[_Finding] = []
    for f in findings:
        if not f.decoded and f.secret and f.secret in text:
            by_string.append(f)
        else:
            by_position.append(f)
    if by_position:
        lines = text.split("\n")
        # Bottom-up, right-to-left, so an edit never moves a later target.
        for f in sorted(by_position, key=lambda f: (f.start_line, f.start_col), reverse=True):
            _redact_at(lines, f, f"[REDACTED:{f.rule}]")
            counts[f.rule] = counts.get(f.rule, 0) + 1
        text = "\n".join(lines)
    if by_string:
        # ONE pass for every secret, longest first in the alternation, so a
        # shorter secret never cuts into a longer one and leaves part of it
        # behind. (A replace per secret was quadratic: 12,000 findings in one
        # text took ~40 s.)
        rule_of: dict[str, str] = {}
        for f in sorted(by_string, key=lambda f: len(f.secret), reverse=True):
            rule_of.setdefault(f.secret, f.rule)
        pattern = re.compile("|".join(map(re.escape, rule_of)))

        def _repl(m: re.Match) -> str:
            rule = rule_of[m.group(0)]
            counts[rule] = counts.get(rule, 0) + 1
            return f"[REDACTED:{rule}]"

        text = pattern.sub(_repl, text)
    return text


@dataclass(frozen=True)
class _Piece:
    """A slice of one member text that fits a single scanned file.

    ``line0`` is the member line the slice starts on (1-based) and ``col0`` the
    byte column it starts at on that line, so a finding at piece line ``l``,
    column ``c`` sits at member line ``line0 + l - 1`` — column ``col0 + c``
    on the first line, ``c`` on every other.
    """

    member: int
    text: str
    size: int  # encoded bytes
    line0: int
    col0: int


def _pieces_of(k: int, text: str) -> list[_Piece]:
    """``text`` as one piece, or as overlapping windows when it is too big."""
    size = len(text.encode("utf-8", "replace"))
    if size <= _PIECE_BYTES:
        return [_Piece(k, text, size, 1, 0)]
    out: list[_Piece] = []
    start = 0
    while True:
        end = min(len(text), start + _PIECE_BYTES)
        while (size := len(text[start:end].encode("utf-8", "replace"))) > _PIECE_BYTES:
            end = start + max((end - start) * _PIECE_BYTES // size - 1, 1)
        line_start = text.rfind("\n", 0, start) + 1
        out.append(
            _Piece(
                k,
                text[start:end],
                size,
                text.count("\n", 0, start) + 1,
                len(text[line_start:start].encode("utf-8", "replace")),
            )
        )
        if end >= len(text):
            return out
        start = max(end - _WINDOW_OVERLAP, start + 1)


def _gitleaks_redact_many(texts: list[str], counts: list[dict[str, int]]) -> list[str]:
    """Run the gitleaks layer over every text, in as few calls as possible.

    Each text is scanned as its own file (a long one as overlapping windows,
    one file each), many files per call. Findings come back per file and are
    moved to their text's coordinates, so each text's result is exactly what
    scanning it alone gives, whatever it was batched with.
    """
    pieces = [p for k, t in enumerate(texts) if t for p in _pieces_of(k, t)]
    found: dict[int, set[_Finding]] = {}
    call: list[_Piece] = []
    size = 0

    def flush() -> None:
        nonlocal size
        if call:
            for p, fs in zip(call, _run_gitleaks([p.text for p in call]), strict=True):
                found.setdefault(p.member, set()).update(_rebase(p, f) for f in fs)
        call.clear()
        size = 0

    for p in pieces:
        if call and size + p.size > _GITLEAKS_CALL_BYTES:
            flush()
        call.append(p)
        size += p.size
    flush()
    out = list(texts)
    for k, fs in found.items():
        out[k] = _apply_findings(texts[k], list(fs), counts[k])
    return out


def _rebase(p: _Piece, f: _Finding) -> _Finding:
    """A finding at piece-local lines/columns, moved to member coordinates."""
    return replace(
        f,
        start_line=p.line0 + f.start_line - 1,
        end_line=p.line0 + f.end_line - 1,
        start_col=f.start_col + (p.col0 if f.start_line == 1 else 0),
        end_col=f.end_col + (p.col0 if f.end_line == 1 else 0),
    )


# Presidio entity allowlist: real PII we want scrubbed at the boundary.
# Explicitly excluded are the noisy recognizers that flag ordinary prose:
# PERSON, DATE_TIME, LOCATION, NRP, URL, ORGANIZATION. Those are useful in
# medical/legal domains but corrupt benign session/PR text (e.g. "released
# on Tuesday" → DATE_TIME; "the /rate_limit endpoint" → URL). Regex handles
# email/SSN/phone in this codebase; Presidio adds coverage for the rest.
PRESIDIO_ENTITIES: list[str] = [
    "CREDIT_CARD",
    "CRYPTO",
    "EMAIL_ADDRESS",
    "IBAN_CODE",
    "IP_ADDRESS",
    "MEDICAL_LICENSE",
    "PHONE_NUMBER",
    "US_BANK_NUMBER",
    "US_DRIVER_LICENSE",
    "US_ITIN",
    "US_PASSPORT",
    "US_SSN",
]


def _scrub_pii_presidio(text: str, counts: dict[str, int]) -> str:
    if _analyzer is None or _anonymizer is None:
        return text
    try:
        results = _analyzer.analyze(
            text=text,
            language="en",
            entities=PRESIDIO_ENTITIES,
            # Score threshold: Presidio's US_DRIVER_LICENSE recognizer fires at
            # ~0.3 on short alphanumeric tokens like "v2"; PHONE_NUMBER on a real
            # US-shaped phone lands at ~0.4. 0.4 is the elbow that filters
            # false positives on prose without dropping real hits.
            score_threshold=0.4,
        )
    except Exception as e:  # noqa: BLE001 -- degrade to what regex caught
        log.warning("Presidio analyze failed (%s); skipping PII pass", e)
        return text
    if not results:
        return text
    for r in results:
        label = f"presidio_{r.entity_type.lower()}"
        counts[label] = counts.get(label, 0) + 1
    try:
        anon = _anonymizer.anonymize(text=text, analyzer_results=results)
    except Exception as e:  # noqa: BLE001
        log.warning("Presidio anonymize failed (%s); returning regex-scrubbed text", e)
        return text
    return anon.text


def scrub(text: str) -> ScrubResult:
    """Apply secret + PII scrubbing. Deterministic and idempotent.

    Contract:
    * Same input → same output (deterministic), for a fixed Presidio state.
    * ``scrub(scrub(x).text).text == scrub(x).text`` (idempotent).
    * An analyse/anonymize failure MID-CALL is logged and swallowed: coverage
      degrades to regex-only rather than the call failing.

    NOT SIDE-EFFECT-FREE, AND IT CAN RAISE. Both were claimed here and neither
    survived the warm-up work, so they are spelled out rather than deleted —
    a caller that believed the old contract would have got the two surprises
    below in production:

    * The first call in the process — and any call that lands while a warm-up
      is still building the engines — runs ``ensure_presidio_ready()``, which
      BLOCKS for the model load (~50 s on this host) and latches the outcome in
      module globals. That is why ``aggregator/mcp.py``'s ``main()`` kicks the
      load off on a daemon thread before serving stdio: so the cost is paid in
      parallel with the handshake instead of in front of the first result.
      After the latch it is one atomic ``Event`` read.
    * ``ensure_presidio_ready()`` folds ``Exception`` and ``SystemExit`` into
      the regex fallback but RE-RAISES everything else — ``KeyboardInterrupt``,
      ``asyncio.CancelledError`` — after announcing the degradation. Those
      propagate straight out of this function. Deliberate: a ctrl-c during a
      50-second model load has to keep killing the process. It simply is not
      "never raises".
    * A non-empty text needs gitleaks. When it cannot run this raises
      :class:`ScrubUnavailableError` — fail closed, by design: the caller
      must not store or return the text.
    """
    return scrub_many([text])[0]


def scrub_secrets_many(texts: Sequence[str]) -> list[ScrubResult]:
    """The SECRET layers only — regex floor, then gitleaks — over every text.

    For re-scrubbing text that already went through the full ``scrub`` once
    (``aggregator rescrub``): its PII pass is already applied, and Presidio is
    ~70 ms a row, so running it again over the whole store would cost hours
    for nothing. Fails closed exactly like :func:`scrub`.
    """
    counts: list[dict[str, int]] = [{} for _ in texts]
    floor = [
        _run_patterns(t, SECRET_PATTERNS, c) if t else (t or "")
        for t, c in zip(texts, counts, strict=True)
    ]
    cleaned = _gitleaks_redact_many(floor, counts)
    return [ScrubResult(text=t, counts=c) for t, c in zip(cleaned, counts, strict=True)]


def scrub_many(texts: Sequence[str]) -> list[ScrubResult]:
    """:func:`scrub` over many texts, with ONE gitleaks call per ~4 MB.

    ``scrub_many(xs)[i] == scrub(xs[i])`` for every i — batching changes the
    cost, never the result (see :func:`_gitleaks_redact_many`). Raises
    :class:`ScrubUnavailableError` when the secret scanner cannot run, and
    then NO text in the batch may be stored or returned.
    """
    results = scrub_secrets_many(texts)
    return [ScrubResult(text=_scrub_pii(r.text, r.counts), counts=r.counts) for r in results]


def _scrub_pii(text: str, counts: dict[str, int]) -> str:
    """The PII layers, after the secret layers have run."""
    if not text:
        return text
    # Credit-card Luhn-gated pass runs BEFORE the generic PII loop so a
    # Luhn-valid 16-digit sequence isn't fragmented by another pattern first
    # (e.g. the phone regex).
    text = _scrub_credit_cards(text, counts)
    # PII regex always runs — Presidio doesn't catch our fixture formats
    # (SSN, +1-555-… phones) reliably. See module docstring.
    # ``credit_card`` is excluded from the generic loop: it's handled above
    # with Luhn validation so we don't false-positive on 16-digit non-cards.
    _generic_pii = {k: v for k, v in PII_PATTERNS.items() if k != "credit_card"}
    text = _run_patterns(text, _generic_pii, counts)
    # Blocks on the first call in the process (and on any call that lands while
    # a warm-up is still building the engines) rather than racing a half-built
    # one. After that it is one atomic Event read.
    if ensure_presidio_ready():
        text = _scrub_pii_presidio(text, counts)
    return text
