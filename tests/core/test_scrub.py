"""Tests for pre-store / pre-return scrubbing (M2).

Fixtures live under ``tests/fixtures/scrub/`` and are constructed so that this
test file itself never contains a substring gitleaks would flag — assertions
concatenate the guard prefixes with ``+`` at runtime.

Presidio may or may not be importable on a given machine; the scrub module
falls back to regex-only PII patterns when it isn't. The assertions below hold
under BOTH regimes (the regex fallback covers every fixture case), so this test
file is Presidio-agnostic.
"""
import importlib
from pathlib import Path

from aggregator.core.scrub import ScrubResult, scrub

FIX = Path(__file__).parent.parent / "fixtures" / "scrub"


def test_openai_key_redacted():
    txt = (FIX / "api_keys.txt").read_text()
    result = scrub(txt)
    # Split literals so this file itself is not gitleaks-flaggable.
    assert "sk-" + "proj-" not in result.text
    assert result.counts.get("openai_key", 0) >= 1


def test_anthropic_key_redacted():
    txt = (FIX / "api_keys.txt").read_text()
    result = scrub(txt)
    assert "sk-" + "ant-api03-" not in result.text
    assert result.counts.get("anthropic_key", 0) >= 1


def test_github_pat_redacted():
    txt = (FIX / "api_keys.txt").read_text()
    result = scrub(txt)
    assert "gh" + "p_" not in result.text
    assert result.counts.get("github_pat", 0) >= 1


def test_jwt_redacted():
    txt = (FIX / "api_keys.txt").read_text()
    result = scrub(txt)
    # JWTs begin with "ey" + "J" (base64 of `{"`); assert none survive.
    assert "ey" + "J" not in result.text
    assert result.counts.get("jwt", 0) >= 1


def test_aws_key_redacted():
    txt = (FIX / "api_keys.txt").read_text()
    result = scrub(txt)
    assert "AKIA" + "IOSFODNN7EXAMPLE" not in result.text
    assert result.counts.get("aws_access_key", 0) >= 1


def test_ssn_redacted():
    txt = (FIX / "pii.txt").read_text()
    result = scrub(txt)
    assert "123-45-6789" not in result.text


def test_email_redacted():
    txt = (FIX / "pii.txt").read_text()
    result = scrub(txt)
    assert "aggregator@example.com" not in result.text


def test_phone_redacted():
    txt = (FIX / "pii.txt").read_text()
    result = scrub(txt)
    assert "555-123-4567" not in result.text


def test_benign_text_untouched():
    txt = (FIX / "benign.txt").read_text()
    result = scrub(txt)
    assert result.text.strip() == txt.strip()
    assert sum(result.counts.values()) == 0


def test_scrub_returns_structured_result():
    r = scrub("hello world")
    assert isinstance(r, ScrubResult)
    assert isinstance(r.counts, dict)
    assert isinstance(r.text, str)


def test_scrub_is_idempotent():
    """Scrubbing an already-scrubbed string must not further mutate it.

    Defense in depth (spec §Security constraint 3): scrub runs pre-store AND
    pre-return. If it were non-idempotent, the second pass would corrupt the
    replacement tokens.
    """
    txt = (FIX / "api_keys.txt").read_text()
    once = scrub(txt).text
    twice = scrub(once).text
    assert once == twice


# --- MEDIUM: additional regex fallbacks (advisor round-1) ------------------
# These formats aren't reliably covered by Presidio's stock recognizers under
# the score threshold we use, so we add explicit regex fallbacks. Presidio
# still runs on top when available for broader coverage.


def test_credit_card_luhn_shape_redacted():
    """A Luhn-valid card number in text must be redacted.

    Uses a well-known Visa test PAN (4111111111111111). Written as split
    literals + hyphenated form to test both variants."""
    hyphenated = "4111-1111-1111-1111"
    contiguous = "4111" + "111111111111"
    for text in (hyphenated, contiguous):
        result = scrub(f"Card: {text}")
        assert hyphenated not in result.text
        assert contiguous not in result.text
        assert result.counts.get("credit_card", 0) >= 1


def test_credit_card_luhn_invalid_not_flagged_by_regex_layer():
    """A 16-digit run that FAILS the Luhn check must not be redacted by the
    regex layer (Presidio may still catch it independently; that's a
    separate layer). We invoke the private helper directly so this test is
    Presidio-independent.
    """
    from aggregator.core.scrub import _scrub_credit_cards

    counts: dict[str, int] = {}
    bad = "1234-5678-1234-5678"  # Luhn-invalid
    out = _scrub_credit_cards(f"num {bad} end", counts)
    assert bad in out
    assert counts.get("credit_card", 0) == 0


def test_ipv4_address_redacted():
    result = scrub("connect to 192.168.1.42 on port 8080")
    assert "192.168.1.42" not in result.text
    assert result.counts.get("ipv4", 0) >= 1


def test_ipv6_address_redacted():
    result = scrub("host at 2001:0db8:85a3:0000:0000:8a2e:0370:7334 lives")
    assert "2001:0db8:85a3:0000:0000:8a2e:0370:7334" not in result.text
    assert result.counts.get("ipv6", 0) >= 1


def test_iban_shape_redacted():
    """IBAN example (GB Nat West test IBAN, publicly documented format)."""
    iban = "GB82WEST12345698765432"
    result = scrub(f"send to {iban} thanks")
    assert iban not in result.text
    assert result.counts.get("iban", 0) >= 1


# --- MEDIUM (round-2): ipv6 regex must not over-match hash chains ---------


def test_ipv6_regex_does_not_match_sha256_hash_chain():
    """Round-2 MEDIUM: pre-fix, the ipv6 regex (7 hex groups + colon + hex
    tail) fired on benign chains like ``sha256:aa:bb:cc:dd:ee:ff:11:22``
    that share the same shape. Post-fix, boundaries on both sides require
    a non-hex/colon context so the ``sha256:`` prefix disqualifies the run.
    """
    chain = "sha256:aa:bb:cc:dd:ee:ff:11:22"
    result = scrub(f"integrity: {chain}")
    assert chain in result.text, (
        f"ipv6 pattern over-matched sha256 chain; text: {result.text!r}"
    )
    assert result.counts.get("ipv6", 0) == 0


def test_ipv6_full_length_still_redacted():
    """Regression guard: tighter regex must still catch a real IPv6."""
    addr = "2001:0db8:85a3:0000:0000:8a2e:0370:7334"
    result = scrub(f"host at {addr} lives")
    assert addr not in result.text
    assert result.counts.get("ipv6", 0) >= 1


# --- Presidio degradation: absence must fall back, never abort -------------
#
# Both tests reload the module, because the Presidio decision is made once at
# import time. Each restores the real module in a `finally` so ordering with
# the rest of the suite cannot matter.


def _reload_scrub():
    import aggregator.core.scrub as mod

    return importlib.reload(mod)


def test_missing_spacy_model_degrades_to_regex_instead_of_exiting():
    """A machine without the spaCy model must degrade, not kill the process.

    Presidio builds its NLP engine on construction and, when the model is
    absent, calls ``spacy.cli.download`` — which on failure calls
    ``sys.exit(1)``. ``SystemExit`` is a ``BaseException``, so the module's
    ``except Exception`` never saw it and the import took the interpreter
    down: CI aborted during collection with ``INTERNALERROR> SystemExit: 1``
    on every run from 2026-08-08 onward.
    """
    import spacy.util

    original = spacy.util.get_installed_models
    spacy.util.get_installed_models = lambda: []
    try:
        mod = _reload_scrub()
        assert mod._spacy_model_present() is False
        assert mod._PRESIDIO_OK is False
        # The regex layer is unaffected by Presidio's absence.
        assert mod.scrub("write to bob@example.com").counts.get("email", 0) >= 1
    finally:
        spacy.util.get_installed_models = original
        _reload_scrub()


def test_systemexit_from_engine_construction_is_caught():
    """Directly guard the ``except (Exception, SystemExit)`` widening.

    Pins the failure mode rather than the path to it: if someone narrows the
    clause back to ``except Exception``, this fails loudly here instead of as
    an INTERNALERROR in CI. Written to hold whether or not the real model is
    installed on the machine running it.
    """
    import presidio_analyzer
    import spacy.util
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    configured = [
        m["model_name"]
        for m in NlpEngineProvider().nlp_configuration.get("models", [])
    ]
    original_models = spacy.util.get_installed_models
    original_engine = presidio_analyzer.AnalyzerEngine

    def _exit_like_spacy_download(*_args, **_kwargs):
        raise SystemExit(1)

    # Claim the model IS present so the guard passes and construction runs.
    spacy.util.get_installed_models = lambda: configured
    presidio_analyzer.AnalyzerEngine = _exit_like_spacy_download
    try:
        mod = _reload_scrub()  # must not propagate SystemExit
        assert mod._PRESIDIO_OK is False
        assert mod.scrub("write to bob@example.com").counts.get("email", 0) >= 1
    finally:
        spacy.util.get_installed_models = original_models
        presidio_analyzer.AnalyzerEngine = original_engine
        _reload_scrub()
