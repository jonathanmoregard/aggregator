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
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from aggregator.core import gitleaks_driver
from aggregator.core import scrub as scrub_mod
from aggregator.core.scrub import (
    GITLEAKS_ENV,
    GITLEAKS_RULES,
    SECRET_PATTERNS,
    ScrubUnavailableError,
    _gitleaks_redact_many,
    _redact_targets,
    _run_patterns,
    scrub,
    scrub_many,
)

#: sha256 of ``hooks/secret-rules.toml`` in the ~/.claude config repo, which
#: redacts agent tool output with the same rules. Update BOTH copies and this
#: constant together.
#: sha256 of ``hooks/gitleaks_driver.py`` in the same repo, vendored here
#: byte-identical as ``aggregator/core/gitleaks_driver.py``.
SHARED_DRIVER_SHA256 = "019133e174e15d462f0d0143a8946f694cd8bf350051c0843354e884a3023c27"

SHARED_RULES_SHA256 = "763782827890d286f4f0b9a34b1b38e95a4da33ecd0fef0fa53f57e39ff9b540"

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


@pytest.mark.parametrize(
    ("vendored", "digest", "claude_copy"),
    [
        (GITLEAKS_RULES, SHARED_RULES_SHA256, "secret-rules.toml"),
        (Path(gitleaks_driver.__file__), SHARED_DRIVER_SHA256, "gitleaks_driver.py"),
    ],
    ids=["rules", "driver"],
)
def test_vendored_files_are_the_shared_copies(vendored, digest, claude_copy):
    assert hashlib.sha256(vendored.read_bytes()).hexdigest() == digest, (
        f"{vendored.name} drifted from the ~/.claude copy (hooks/{claude_copy}); "
        "update both and the pinned digest"
    )
    other = Path.home() / ".claude" / "hooks" / claude_copy
    if other.exists():
        assert other.read_bytes() == vendored.read_bytes()


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


@pytest.mark.parametrize("name", sorted(set(_tokens()) - {"gitlab_pat"}))
def test_regex_floor_alone_covers_the_listed_formats(name):
    """The floor must stand without gitleaks for everything but gitlab."""
    token = _tokens()[name]
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


@pytest.mark.parametrize("i", range(len(_mixed_corpus())))
def test_scrub_is_idempotent_over_new_formats(i):
    once = scrub(_mixed_corpus()[i]).text
    assert scrub(once).text == once


def test_batch_gives_exactly_the_single_results():
    corpus = _mixed_corpus()
    many = scrub_many(corpus)
    for text, batched in zip(corpus, many, strict=True):
        single = scrub(text)
        assert batched.text == single.text
        assert batched.counts == single.counts


def test_batched_texts_do_not_switch_rules_on_for_each_other():
    """gitleaks applies a rule only when one of its keywords is in the same
    fragment. Streamed together, a text mentioning sourcegraph made a git SHA
    in ANOTHER text a "sourcegraph-access-token" (seen on real rows)."""
    sha = _r(40, "0123456789abcdef")
    alone = scrub(f"commit {sha} merged").text
    batched = scrub_many(["notes about sourcegraph search", f"commit {sha} merged"])[1].text
    assert batched == alone


def test_a_multiline_secret_next_to_a_dangling_header_is_still_found():
    """A PEM block whose header is in one text and whose END is in the next
    must not swallow the next text's own complete key. The regex floor is
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


@pytest.mark.parametrize(
    "text",
    [
        "The rate limit is 5000 requests per hour.",
        "Use the api_key parameter; the token is read from the keyring.",
        "def get_token():\n    return os.environ['API_TOKEN']\n",
        "commit 3f2a9c1 fixed the key rotation script",
        "see https://github.com/owner/repo/pull/42 for the token refactor",
    ],
)
def test_non_secret_text_is_unchanged(text):
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
    monkeypatch.setattr(gitleaks_driver, "FALLBACK_DIRS", ())
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
    assert any(ScrubUnavailableError.__name__ in e for e in adapter.errors)
    bodies = {r[0] for r in store._c().execute("SELECT body FROM records")}
    assert bodies == {"b0", "b1", "b2", "b3"}
    store.close()


# --- gitleaks input-handling holes (close-out review) ------------------------
#
# Each of these was a real leak against gitleaks 8.30.1 driven naively; the
# tests pin, end to end through ``scrub``, that the driver closes them. The token format (GitLab PAT) is one
# only gitleaks knows, so the regex floor cannot mask a regression.


@pytest.mark.parametrize(
    "prefix",
    ["%PDF-1.4", "%!PS-Adobe", "{\\rtf1", "MZ", "SQLite format 3", "y" * 257 + "ustar"],
)
def test_text_that_sniffs_as_a_binary_type_is_still_scanned(prefix):
    """gitleaks skips input whose first bytes look like application/* and
    reports it clean with zero bytes scanned."""
    token = _tokens()["gitlab_pat"]
    assert token not in scrub(f"{prefix}\n{token}").text


def test_no_token_is_lost_to_a_fragment_cut():
    """gitleaks cuts long blank-line-free input at arbitrary bytes; tokens on
    the cut were missed (8 of 12,000 here, timing-dependent)."""
    tokens = [_tokens()["gitlab_pat"] for _ in range(12000)]
    out = scrub(" ".join(tokens)).text
    # About 1 random token in 240,000 is rejected by gitleaks' own rule and
    # allowlists wherever it stands (measured: missed when scanned alone too).
    # Those are not cut losses; anything else surviving is.
    survivors = [t for t in tokens if t in out]
    assert [t for t in survivors if t not in scrub(t).text] == []


def test_long_text_is_windowed_and_redacted_in_place():
    """A text far larger than one piece: a base64-encoded secret deep inside,
    on a line that starts mid-piece, is redacted and nothing else moves."""
    token = "gh" + "p_" + _r(36)
    encoded = base64.b64encode(f"user:{token}".encode()).decode()
    filler = "\n".join("plain words " * 8 for _ in range(3000))
    text = f"{filler}\nAuthorization: Basic {encoded} tail\n{filler}"
    out = scrub(text).text
    assert encoded not in out
    assert "Authorization: Basic [REDACTED:" in out and "] tail\n" in out
    assert out.count("plain words") == text.count("plain words")
    batched = scrub_many(["x", text, "y"])[1].text
    assert batched == out


def test_a_gitleaksignore_in_the_working_directory_cannot_suppress(monkeypatch, tmp_path):
    token = _tokens()["gitlab_pat"]
    (tmp_path / ".gitleaksignore").write_text(":gitlab-pat:1\n:gitlab-pat:3\n")
    monkeypatch.chdir(tmp_path)
    assert token not in scrub(token).text


def test_a_scan_that_reads_nothing_fails_closed(monkeypatch, tmp_path):
    """An empty report only counts as clean if gitleaks read the input."""
    fake = tmp_path / "gitleaks"
    fake.write_text(
        "#!/bin/sh\ncat >/dev/null\necho '[]'\n"
        "echo 'INF scanned ~0 bytes (0 bytes) in 1ms' >&2\nexit 0\n"
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv(GITLEAKS_ENV, str(fake))
    with pytest.raises(ScrubUnavailableError, match="read 0 of"):
        scrub("some text")


# --- applying targets ----------------------------------------------------------


def test_the_longest_target_wins_where_targets_overlap():
    counts: dict[str, int] = {}
    targets = {"abc": "short", "abcdef": "long", "bcd": "mid"}
    out = _redact_targets("abcdef abc bcd", targets, counts)
    assert out == "[REDACTED:long] [REDACTED:short] [REDACTED:mid]"
    assert counts == {"long": 1, "short": 1, "mid": 1}


def test_many_distinct_targets_over_megabytes_apply_quickly():
    """A flat alternation of 12,000 distinct targets over ~3.7 MB took 49 s;
    the trie-shaped pattern takes about 1 s. Bound generous for slow CI."""
    tokens = [_r(25) + "7" for _ in range(12000)]
    words = ("the", "rate", "limit", "server", "request", "value", "file")
    parts = [words[i % len(words)] for i in range(600_000)]
    for i, t in enumerate(tokens):
        parts[i * 50] = t
    text = " ".join(parts)
    counts: dict[str, int] = {}
    t0 = time.monotonic()
    out = _redact_targets(text, dict.fromkeys(tokens, "generic-api-key"), counts)
    elapsed = time.monotonic() - t0
    assert elapsed < 15, f"{elapsed:.1f}s"
    assert counts == {"generic-api-key": 12000}
    assert not any(t in out for t in tokens)
