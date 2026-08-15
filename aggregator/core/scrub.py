"""Scrubber: applied pre-store AND pre-return (spec §Security constraint 3).

Two layers:

1. **Secret regex patterns** — gitleaks-style, covering API-key shapes
   (OpenAI, Anthropic, GitHub PAT, JWT, AWS access key, PEM private key
   headers). Runs unconditionally; the fastest layer and the strictest
   security gate (secrets must never round-trip through the store).

2. **PII detection**:
   - If `presidio_analyzer` + `presidio_anonymizer` are importable, use them
     for entity-based PII detection (names, orgs, locations, credit cards, ...).
   - Regardless of Presidio availability, always run the explicit regex
     patterns for SSN / email / phone — Presidio's stock recognizers do NOT
     cover these formats reliably out of the box (verified: Presidio 2.2.x
     with en_core_web_lg detects EMAIL_ADDRESS/URL but misses SSN and
     +1-555-… phones). Redundancy here is the point.

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
from dataclasses import dataclass, field

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


try:
    if not _spacy_model_present():
        raise RuntimeError(
            "spaCy model for Presidio is not installed; install it with "
            "`python -m spacy download en_core_web_lg`"
        )
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine

    _analyzer: AnalyzerEngine | None = AnalyzerEngine()
    _anonymizer: AnonymizerEngine | None = AnonymizerEngine()
    _PRESIDIO_OK = True
# SystemExit is listed explicitly: it is a BaseException, so it is not covered
# by `except Exception`, and it is exactly what an in-process spaCy download
# raises. KeyboardInterrupt is deliberately NOT caught.
except (Exception, SystemExit) as e:  # noqa: BLE001 -- init failure -> regex path
    log.warning(
        "Presidio unavailable (%s); PII scrubbing will use regex fallback only", e
    )
    _analyzer = None
    _anonymizer = None
    _PRESIDIO_OK = False


# --- Secret patterns
# Order matters: more-specific prefixes must run BEFORE more-general ones,
# because we mutate `text` sequentially. Concretely: anthropic keys (`sk-ant-`)
# would otherwise match the generic openai `sk-…` pattern first and be
# mis-classified.
SECRET_PATTERNS: dict[str, re.Pattern] = {
    # Anthropic: sk-ant-… . Match this BEFORE the generic openai `sk-` pattern
    # so anthropic keys are counted under the correct label.
    "anthropic_key": re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    # OpenAI: sk-… (optionally sk-proj-…). Now safe to run — anthropic already
    # consumed its variant above.
    "openai_key": re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    # GitHub PAT / OAuth / server token / user token / refresh token.
    "github_pat": re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    # JWT: header.payload.signature — three base64url segments, first starts
    # with "eyJ" (base64 of `{"`).
    "jwt": re.compile(r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
    # AWS access key ID.
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    # PEM private-key header (redact just the header; body will collapse in
    # downstream FTS since the delimiter is gone).
    "generic_pem_start": re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----"),
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
    """Apply secret + PII scrubbing. Idempotent and side-effect-free.

    Contract:
    * Same input → same output (deterministic).
    * ``scrub(scrub(x).text).text == scrub(x).text`` (idempotent).
    * Never raises on well-formed ``str``; degrades to regex-only if Presidio
      fails at analyse/anonymize time.
    """
    counts: dict[str, int] = {}
    text = _run_patterns(text, SECRET_PATTERNS, counts)
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
    if _PRESIDIO_OK:
        text = _scrub_pii_presidio(text, counts)
    return ScrubResult(text=text, counts=counts)
