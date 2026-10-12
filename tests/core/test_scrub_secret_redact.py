"""The secret layer of ``scrub``: one ``secret-redact`` call, fail closed.

The suite runs against the stub CLI (tests/secret_redact_stub.py, put on PATH
by tests/conftest.py); the real CLI's detection is tested in nixos-config.
``test_the_real_cli`` runs the real one when this machine has it.

NO TOKEN LITERALS: every secret-shaped string is generated at test time, and
prefixes are split with ``+``.
"""
from __future__ import annotations

import asyncio
import json
import secrets
import string
import subprocess
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest

from aggregator.core import scrub as scrub_mod
from aggregator.core.scrub import scrub, scrub_many
from aggregator.core.scrub_error import ScrubUnavailableError
from tests.conftest import REAL_SECRET_REDACT

_ALNUM = string.ascii_letters + string.digits


def _r(n: int) -> str:
    return "".join(secrets.choice(_ALNUM) for _ in range(n - 1)) + secrets.choice(string.digits)


def _fake_cli(tmp_path, body: str) -> str:
    fake = tmp_path / "secret-redact"
    fake.write_text(f"#!/bin/sh\n{body}\n")
    fake.chmod(0o755)
    return str(fake)


# --- the call --------------------------------------------------------------------


def test_one_call_redacts_and_counts_per_text():
    token = "gh" + "p_" + _r(36)
    a, b = scrub_many([f"key {token} end", "nothing here"])
    assert token not in a.text and "[REDACTED:" in a.text and a.text.endswith(" end")
    assert b.text == "nothing here" and b.counts == {}
    assert sum(a.counts.values()) == 1


def test_scrub_is_idempotent():
    once = scrub(f"x {'sk-' + 'ant-' + _r(40)} y").text
    assert scrub(once).text == once


def test_empty_texts_make_no_call(monkeypatch, tmp_path):
    monkeypatch.setattr(scrub_mod, "SECRET_REDACT", str(tmp_path / "absent"))
    assert [r.text for r in scrub_many(["", ""])] == ["", ""]


# --- fail closed --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "body"),
    [
        ("exits non-zero", "echo 'secret-redact: gitleaks exited 2' >&2; exit 1"),
        ("writes garbage", "echo 'not json'"),
        ("returns too few texts", """echo '{"texts": [], "counts": []}'"""),
    ],
)
def test_a_failing_scanner_raises_without_echoing_the_text(monkeypatch, tmp_path, label, body):
    monkeypatch.setattr(scrub_mod, "SECRET_REDACT", _fake_cli(tmp_path, body))
    canary = "canary-" + _r(16)
    with pytest.raises(ScrubUnavailableError) as exc:
        scrub(f"text with {canary}")
    assert canary not in str(exc.value), label


def test_a_missing_scanner_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(scrub_mod, "SECRET_REDACT", str(tmp_path / "absent"))
    with pytest.raises(ScrubUnavailableError, match="could not run"):
        scrub_many(["one", "two"])


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
    monkeypatch.setattr(scrub_mod, "SECRET_REDACT", str(tmp_path / "absent"))
    with pytest.raises(ScrubUnavailableError):
        store.upsert_entities([session, obs(1, "key " + "tvly" + "-dev-" + _r(32))])
    assert store._c().execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0
    store.close()


def test_ingest_redacts(tmp_path):
    store, session, obs = _obs_store(tmp_path)
    token = "github" + "_pat_" + _r(22) + "_" + _r(59)
    store.upsert_entities([session, obs(1, f"pat {token}")])
    (body,) = [r[0] for r in store._c().execute("SELECT body FROM observations")]
    assert token not in body and "[REDACTED:" in body
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

    monkeypatch.setattr(scrub_mod, "SECRET_REDACT", str(tmp_path / "absent"))
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


# --- the real CLI, when present --------------------------------------------------------


@pytest.mark.skipif(REAL_SECRET_REDACT is None, reason="secret-redact is not on this machine")
def test_the_real_cli(monkeypatch):
    """The contract the stub imitates, checked against the real thing."""
    monkeypatch.setattr(scrub_mod, "SECRET_REDACT", REAL_SECRET_REDACT)
    token = "glp" + "at-" + _r(20)  # a format only gitleaks knows; the stub does not
    key = ("-----BEGIN RSA " + "PRIVATE KEY-----\n" + "\n".join(_r(64) for _ in range(5)))
    a, b, c = scrub_many([f"A={token}", key, "plain prose"])
    assert token not in a.text and a.counts == {"gitlab-pat": 1}
    assert "PRIVATE KEY" not in b.text and b.counts
    assert c.text == "plain prose" and c.counts == {}
    raw = subprocess.run([REAL_SECRET_REDACT], input=json.dumps({"texts": [], "timeout": 5}).encode(),
                         capture_output=True, check=True).stdout
    assert json.loads(raw) == {"texts": [], "counts": []}
