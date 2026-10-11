"""The gitleaks secret layer of ``scrub``: coverage, batching, fail-closed.

NO TOKEN LITERALS. Every secret-shaped string below is generated at test time
from ``secrets`` — a literal would be a real-looking credential committed to a
public repo, and the pre-commit gitleaks hook would (rightly) refuse it. The
prefixes are split with ``+`` for the same reason.

These tests need the gitleaks binary (``AGGREGATOR_GITLEAKS`` or PATH). That
is deliberate: the layer fails closed, so a machine without it cannot ingest
at all, and CI installs it through the flake's nixpkgs.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import secrets
import stat
import string
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from aggregator.core import scrub as scrub_mod
from aggregator.core.scrub import (
    GITLEAKS_ENV,
    GITLEAKS_RULES,
    SECRET_PATTERNS,
    ScrubUnavailableError,
    _gitleaks_redact_many,
    _run_patterns,
    scrub,
    scrub_many,
)

#: sha256 of ``hooks/secret-rules.toml`` in the ~/.claude config repo, which
#: redacts agent tool output with the same rules. Update BOTH copies and this
#: constant together.
SHARED_RULES_SHA256 = "5eeb852b8109a33d63ebd748f4577082248be76b284b2e7f3b9d9628598d9206"

_ALNUM = string.ascii_letters + string.digits


def _r(n: int, alphabet: str = _ALNUM) -> str:
    return "".join(secrets.choice(alphabet) for _ in range(n))


def _tokens() -> dict[str, str]:
    """One fresh synthetic token per format, keyed by a short name."""
    return {
        "github_fine_grained": "github" + "_pat_" + _r(22) + "_" + _r(59),
        "github_classic": "gh" + "p_" + _r(36),
        "tavily": "tvly" + "-dev-" + _r(32),
        "slack": "xo" + "xb-" + _r(11, string.digits) + "-" + _r(13, string.digits) + "-" + _r(24),
        "cloudflare_v2": "cf" + "ut_" + _r(40) + _r(8, "0123456789abcdef"),
        "google_oauth": "GOC" + "SPX-" + _r(28),
        "anthropic_oauth": "sk-" + "ant-oat01-" + _r(80),
        "anthropic_admin": "sk-" + "ant-admin01-" + _r(80),
        "anthropic_api03": "sk-" + "ant-api03-" + _r(93) + "AA",
        # Not in the regex floor at all: only gitleaks' default rules know it.
        "gitlab_pat": "gl" + "pat-" + _r(20),
    }


def _mask(s: str) -> str:
    return f"{s[:4]}...({len(s)} chars)"


# --- the vendored rules ------------------------------------------------------


def test_vendored_rules_file_is_the_shared_copy():
    digest = hashlib.sha256(GITLEAKS_RULES.read_bytes()).hexdigest()
    assert digest == SHARED_RULES_SHA256, (
        "aggregator/core/secret-rules.toml drifted from the ~/.claude copy "
        "(hooks/secret-rules.toml); update both and SHARED_RULES_SHA256"
    )


def test_vendored_rules_match_the_local_claude_copy_when_present():
    other = Path.home() / ".claude" / "hooks" / "secret-rules.toml"
    if not other.exists():
        pytest.skip("no ~/.claude/hooks/secret-rules.toml on this machine")
    assert other.read_bytes() == GITLEAKS_RULES.read_bytes()


def test_rules_file_ships_inside_the_package():
    """Packaged next to scrub.py, so the wheel (and the Nix build) carries it."""
    assert GITLEAKS_RULES.parent == Path(scrub_mod.__file__).parent
    assert GITLEAKS_RULES.is_file()


# --- coverage ----------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_tokens()))
def test_each_format_is_redacted(name):
    token = _tokens()[name]
    out = scrub(f"config line: value {token} trailing words")
    assert token not in out.text, f"{name} survived: {_mask(token)}"
    assert "[REDACTED:" in out.text
    assert "trailing words" in out.text


def test_regex_floor_alone_covers_the_listed_formats():
    """The floor must stand without gitleaks for everything but gitlab."""
    for name, token in _tokens().items():
        if name == "gitlab_pat":
            continue
        out = _run_patterns(f"x {token} y", SECRET_PATTERNS, {})
        assert token not in out, f"floor missed {name}: {_mask(token)}"


def test_base64_encoded_secret_is_redacted():
    token = "gh" + "p_" + _r(36)
    encoded = base64.b64encode(f"user:{token}".encode()).decode()
    out = scrub(f"Authorization: Basic {encoded}\nnext line")
    assert encoded not in out.text
    assert token not in out.text
    assert out.text.endswith("\nnext line")


def test_private_key_block_is_redacted_whole():
    body = "\n".join(_r(64) for _ in range(12))
    block = "-----BEGIN " + "PRIVATE KEY-----\n" + body + "\n-----END " + "PRIVATE KEY-----"
    out = scrub(f"before\n{block}\nafter")
    assert body.splitlines()[3] not in out.text
    assert out.text.startswith("before\n") and out.text.endswith("\nafter")


def test_gitleaks_allow_comment_cannot_opt_text_out():
    token = "gl" + "pat-" + _r(20)
    out = scrub(f"{token}  # gitleaks" + ":allow")
    assert token not in out.text


# --- invariants --------------------------------------------------------------


def _mixed_corpus() -> list[str]:
    t = _tokens()
    return [
        "Nothing secret here, just prose about tokens and keys.",
        f"export GITHUB_TOKEN={t['github_fine_grained']}\necho done",
        f"tavily={t['tavily']} and slack {t['slack']}",
        "",
        f"line1\nline2\n  key: {_r(32)}\nline4",
        f"{t['gitlab_pat']}\n{t['cloudflare_v2']}\n{t['anthropic_oauth']}",
        "def get_token():\n    return os.environ['API_TOKEN']\n",
        f"same twice {t['github_classic']} and {t['github_classic']}",
    ]


def test_scrub_is_idempotent_over_new_formats():
    for text in _mixed_corpus():
        once = scrub(text).text
        assert scrub(once).text == once


def test_batch_gives_exactly_the_single_results():
    corpus = _mixed_corpus()
    many = scrub_many(corpus)
    for text, batched in zip(corpus, many, strict=True):
        single = scrub(text)
        assert batched.text == single.text
        assert batched.counts == single.counts


def test_a_multiline_finding_straddling_members_is_rescanned():
    """A PEM block whose header is in one member and whose END is in the next
    must not swallow the next member's own complete key. The regex floor is
    bypassed here on purpose — it would remove the headers first."""
    body = "\n".join(_r(64) for _ in range(12))
    head = "-----BEGIN RSA " + "PRIVATE KEY-----"
    tail = "-----END RSA " + "PRIVATE KEY-----"
    dangling = f"quoted header only: {head}\nno end here"
    complete = f"{head}\n{body}\n{tail}"
    texts = [dangling, complete]
    batched = _gitleaks_redact_many(list(texts), [{}, {}])
    singles = [_gitleaks_redact_many([t], [{}])[0] for t in texts]
    assert batched == singles
    assert body.splitlines()[5] not in batched[1]


def test_non_secret_text_is_unchanged():
    benign = [
        "The rate limit is 5000 requests per hour.",
        "Use the api_key parameter; the token is read from the keyring.",
        "def get_token():\n    return os.environ['API_TOKEN']\n",
        "commit 3f2a9c1 fixed the key rotation script",
        "see https://github.com/owner/repo/pull/42 for the token refactor",
    ]
    for text in benign:
        out = scrub(text)
        assert out.text == text, out.counts
        assert out.counts == {}


# --- fail closed -------------------------------------------------------------


def test_missing_binary_raises(monkeypatch, tmp_path):
    monkeypatch.setenv(GITLEAKS_ENV, str(tmp_path / "no-such-gitleaks"))
    with pytest.raises(ScrubUnavailableError):
        scrub("anything at all")


def test_no_binary_on_path_raises(monkeypatch, tmp_path):
    monkeypatch.delenv(GITLEAKS_ENV, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(ScrubUnavailableError):
        scrub_many(["one", "two"])


def test_erroring_binary_raises_without_echoing_the_text(monkeypatch, tmp_path):
    fake = tmp_path / "gitleaks"
    fake.write_text("#!/bin/sh\necho 'FTL config broken' >&2\nexit 2\n")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv(GITLEAKS_ENV, str(fake))
    canary = "canary-" + _r(16)
    with pytest.raises(ScrubUnavailableError) as exc:
        scrub(f"text with {canary}")
    assert canary not in str(exc.value)
    assert "exited 2" in str(exc.value)


def _obs_store(tmp_path):
    from aggregator.core.store import Store
    from aggregator.sources.base import ObservationRow, SessionRow

    store = Store(tmp_path / "cache.db")
    store.migrate()
    ts = datetime(2026, 10, 1, tzinfo=UTC)
    session = SessionRow(
        session_id="s1", root_session_id="s1", parent_session_id=None,
        kind="session", agent_id=None, agent_type=None,
        spawned_by_tool_use_id=None, cwd="/w", git_branch="main",
        first_ts=ts, last_ts=ts, jsonl_path="/x.jsonl",
    )

    def obs(i: int, body: str) -> ObservationRow:
        return ObservationRow(
            obs_id=f"o{i}", session_id="s1", root_session_id="s1",
            parent_obs_id=None, type="user", ts=ts + timedelta(seconds=i),
            model=None, input_tokens=None, output_tokens=None,
            tool_name=None, tool_use_id=None, body=body,
        )

    return store, session, obs


def test_ingest_stores_nothing_when_the_scanner_is_missing(monkeypatch, tmp_path):
    store, session, obs = _obs_store(tmp_path)
    token = "tvly" + "-dev-" + _r(32)
    monkeypatch.setenv(GITLEAKS_ENV, str(tmp_path / "absent"))
    with pytest.raises(ScrubUnavailableError):
        store.upsert_entities([session, obs(1, f"key {token}")])
    count = store._c().execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    assert count == 0
    store.close()


def test_ingest_redacts_new_formats(tmp_path):
    store, session, obs = _obs_store(tmp_path)
    t = _tokens()
    store.upsert_entities(
        [session, obs(1, f"pat {t['github_fine_grained']}"), obs(2, t["gitlab_pat"])]
    )
    bodies = [r[0] for r in store._c().execute("SELECT body FROM observations")]
    assert len(bodies) == 2
    for body in bodies:
        assert "[REDACTED:" in body
        assert t["github_fine_grained"] not in body and t["gitlab_pat"] not in body
    store.close()


def test_runner_does_not_quarantine_rows_when_the_scanner_is_down(monkeypatch, tmp_path):
    """Scanner down is a broken sink, not poison. Without the runner's special
    case, unchanged rows "survive" the isolation pass and the changed ones are
    parked in the quarantine as if they were bad data."""
    from aggregator.core.store import Store
    from aggregator.imports.ingest_state import PoisonLedger, Watermarks
    from aggregator.imports.runner import run_imports
    from aggregator.imports.store_sink import StoreSink
    from aggregator.sources.base import Record

    now = datetime(2026, 10, 1, tzinfo=UTC)

    def rec(n: int, body: str) -> Record:
        return Record(
            stable_id=f"dropbox:{n}", source="dropbox", subject=f"s{n}",
            body=body, updated_at=now - timedelta(days=n),
        )

    class Adapter:
        name = "dropbox"

        def __init__(self, records):
            self._records = records

        async def get_data(self) -> AsyncIterator:
            for r in self._records:
                yield r

    store = Store(tmp_path / "cache.db")
    store.migrate()
    first = [rec(i, f"b{i}") for i in range(4)]
    asyncio.run(run_imports([Adapter(first)], StoreSink(store), batch_size=10))

    monkeypatch.setenv(GITLEAKS_ENV, str(tmp_path / "absent"))
    changed = first[:2] + [rec(2, "edited two"), rec(3, "edited three")]
    ledger = PoisonLedger(store)
    report = asyncio.run(
        run_imports(
            [Adapter(changed)], StoreSink(store), batch_size=10,
            watermarks=Watermarks(store), poison=ledger,
        )
    )
    adapter = report.adapters["dropbox"]
    assert not adapter.ok
    assert adapter.quarantined == 0
    assert ledger.held("dropbox") == {}
    assert any("ScrubUnavailableError" in e for e in adapter.errors)
    bodies = {r[0] for r in store._c().execute("SELECT body FROM records")}
    assert bodies == {"b0", "b1", "b2", "b3"}
    store.close()
