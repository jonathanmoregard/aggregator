"""Scrubber: applied pre-store AND pre-return (spec §Security constraint 3).

Two layers:

1. **Secrets**, in two passes, both mandatory:

   a. **Regex floor** (``SECRET_PATTERNS``) — the token formats most likely
      to appear in this user's history (OpenAI, Anthropic incl. OAuth/admin,
      GitHub classic + fine-grained PATs, Slack, Tavily, Cloudflare v2, Google
      OAuth client secret, JWT, AWS access key, PEM headers). Pure Python,
      always on.
   b. **gitleaks**, through the vendored ``gitleaks_driver`` with
      ``secret-rules.toml`` (every default rule — ~220 providers — plus the
      custom formats gitleaks lacks). Each target string the driver returns
      is replaced by ``[REDACTED:<rule-id>]``. FAIL CLOSED: if the scan does
      not complete, ``scrub``/``scrub_many`` raise
      :class:`ScrubUnavailableError` and the caller must neither store nor
      return the text.

   A gitleaks call costs ~100 ms of process start, so callers with many
   texts use :func:`scrub_many`: one call, and the same per-text result as
   ``scrub`` on each.

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

import logging
import re
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from aggregator.core import gitleaks_driver
from aggregator.core.gitleaks_driver import GitleaksError

#: The scrubber's fail-closed error: the scan did not complete, so the text
#: must be neither stored nor returned. An alias, not a subclass, so it is the
#: same class after ``importlib.reload`` of this module (tests reload it).
ScrubUnavailableError = GitleaksError

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
# ORDER MATTERS: patterns run in sequence over the same text, so a more
# specific prefix must come before a more general one — every Anthropic
# variant (OAuth login, admin, exact api03 console key) before the generic
# `sk-ant-` pattern, and all of them before OpenAI's `sk-` — or a key is
# counted under the wrong label.
SECRET_PATTERNS: dict[str, re.Pattern] = {
    "anthropic_oauth": re.compile(r"sk-ant-oat\d{2}-[A-Za-z0-9_-]{20,}"),
    "anthropic_admin": re.compile(r"sk-ant-admin\d{2}-[A-Za-z0-9_-]{20,}"),
    "anthropic_api03": re.compile(r"sk-ant-api03-[A-Za-z0-9_-]{93}AA"),
    "anthropic_key": re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
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
#: scanned. Unset: PATH, then the Nix profile dirs (see gitleaks_driver).
GITLEAKS_ENV = "AGGREGATOR_GITLEAKS"

#: Go threads per gitleaks call. Unbounded it takes every core and burns ~3x
#: the CPU for the same scan (8.6 MB of real rows: 17.6 s CPU on 24 threads,
#: 6.1 s on 2), and ingest runs on a timer in the background.
_GITLEAKS_CHILD_ENV = {"GOMAXPROCS": "2"}


def _gitleaks_redact_many(texts: list[str], counts: list[dict[str, int]]) -> list[str]:
    """Redact every gitleaks finding in the text it was found in. Fails closed.

    All gitleaks handling (one scanned file per text, header, piece bounds,
    hermetic run, read-everything check) lives in ``gitleaks_driver``; this
    only applies the target strings it returns.
    """
    if not any(texts):
        return list(texts)
    binary = gitleaks_driver.gitleaks_binary(GITLEAKS_ENV)
    if binary is None:
        raise ScrubUnavailableError(
            f"gitleaks not found: set {GITLEAKS_ENV} or put gitleaks on PATH"
        )
    found = gitleaks_driver.find_secrets(
        list(texts), rules=str(GITLEAKS_RULES), binary=binary, env=_GITLEAKS_CHILD_ENV
    )
    return [
        _redact_targets(t, targets, c)
        for t, targets, c in zip(texts, found, counts, strict=True)
    ]


def _redact_targets(text: str, targets: dict[str, str], counts: dict[str, int]) -> str:
    """Replace every target string with ``[REDACTED:<rule>]`` in ONE pass.

    At each position the LONGEST target wins, so a shorter target never cuts
    into a longer one and leaves part of it behind.
    """
    if not targets:
        return text

    def _repl(m: re.Match) -> str:
        rule = targets[m.group(0)]
        counts[rule] = counts.get(rule, 0) + 1
        return f"[REDACTED:{rule}]"

    return _union_pattern(targets).sub(_repl, text)


def _union_pattern(strings: Iterable[str]) -> re.Pattern:
    """One regex matching any of ``strings``, longest first, built as a trie.

    A flat ``a|b|c`` alternation is tried branch by branch at every position:
    12,000 distinct targets over 3.7 MB took 49 s. Nested by shared prefix
    (runs without a branch collapsed to one literal), each position costs one
    pass down the trie instead: 0.9 s for the same input.
    """
    trie: dict = {}
    for s in strings:
        node = trie
        for ch in s:
            node = node.setdefault(ch, {})
        node[""] = {}  # a string ends here

    def build(node: dict) -> str:
        branches = []
        for ch in sorted(k for k in node if k):
            literal, child = ch, node[ch]
            while len(child) == 1 and "" not in child:
                ((nxt, child),) = child.items()
                literal += nxt
            branches.append(re.escape(literal) + build(child))
        if not branches:
            return ""
        body = branches[0] if len(branches) == 1 else "(?:" + "|".join(branches) + ")"
        # A string ends here AND longer ones continue: try longer first.
        return f"(?:{body})?" if "" in node else body

    return re.compile(build(trie))


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
        _run_patterns(t, SECRET_PATTERNS, c) for t, c in zip(texts, counts, strict=True)
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
