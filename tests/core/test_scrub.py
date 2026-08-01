"""Tests for pre-store / pre-return scrubbing (M2).

Fixtures live under ``tests/fixtures/scrub/`` and are constructed so that this
test file itself never contains a substring gitleaks would flag — assertions
concatenate the guard prefixes with ``+`` at runtime.

Presidio may or may not be importable on a given machine; the scrub module
falls back to regex-only PII patterns when it isn't. The assertions below hold
under BOTH regimes (the regex fallback covers every fixture case), so this test
file is Presidio-agnostic.
"""
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
