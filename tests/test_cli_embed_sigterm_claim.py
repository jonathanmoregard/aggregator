"""A routine SIGTERM must not get an innocent row condemned as poison.

Round 1 gave the embed worker a durable claim: the row about to be embedded
is written and committed first, so a process that dies with no exception —
OOM kill, segfault in a native tokenizer, SIGKILL — leaves a trace, and the
next run blames that row through the quarantine ledger instead of picking it
up and dying identically twice an hour forever.

That mechanism could not tell a crash from a REQUEST TO STOP. ``systemctl
stop``, a reboot, a deploy, or ``TimeoutStartSec`` on a 25-30 day backfill
all send SIGTERM, whose default disposition ends the process just as
abruptly. The claim survived, and the next run held a perfectly good row in
the ledger, marked it ``embedding_state='error'``, printed it to stderr,
exited non-zero and fired a CRITICAL desktop notification. Every reboot,
forever — and the held row was permanently poisoned on the way through.

The fix is the shape ``imports/runner.graceful_shutdown`` already uses for
ingest: SIGTERM sets a flag, the row in flight finishes and releases its
claim, the loop stops at the next boundary, and the process exits cleanly. A
real kill still leaves a claim, so the crash-blame path is untouched — the
two are told apart by which signal the process was able to handle, which is
exactly the distinction being made.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys

import pytest

from aggregator.core.store import Store
from aggregator.imports.ingest_state import PoisonLedger

# Signals itself from inside ``embed_documents``, at a deterministic row
# rather than at a wall-clock moment, so the interruption always lands in the
# same place: after the claim is committed, before it is released.
_SIGNAL_RUNNER = '''\
import os
import signal
import sys

import numpy as np

import aggregator.cli as cli

SENTINEL = sys.argv[1]
SIG = int(sys.argv[2])


class SelfSignallingStub:
    def __init__(self, *a, **kw):
        pass

    def embed_documents(self, docs):
        if any(SENTINEL in d for d in docs):
            os.kill(os.getpid(), SIG)
        return np.zeros((len(docs), 768), dtype=np.float32)

    def embed_query(self, query):
        return np.zeros(768, dtype=np.float32)


cli.Embedder = SelfSignallingStub
sys.exit(cli.main(["embed", "--catchup", "--batch-size", "1"]))
'''

# Same runner without the signal: the NEXT run, which is the one that used to
# blame the innocent row.
_PLAIN_RUNNER = '''\
import sys

import numpy as np

import aggregator.cli as cli


class Stub:
    def __init__(self, *a, **kw):
        pass

    def embed_documents(self, docs):
        return np.zeros((len(docs), 768), dtype=np.float32)

    def embed_query(self, query):
        return np.zeros(768, dtype=np.float32)


cli.Embedder = Stub
sys.exit(cli.main(["embed", "--catchup", "--batch-size", "1"]))
'''

_SENTINEL = "THE-ROW-THE-DEPLOY-INTERRUPTED"


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    db = tmp_path / "aggregator" / "cache.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    s = Store(db_path=db)
    s.migrate()
    c = s._c()
    c.execute(
        "INSERT INTO sessions(session_id, root_session_id, kind, first_ts, "
        "last_ts, jsonl_path) VALUES ('sid','sid','session','2026-01-01',"
        "'2026-01-01','/tmp/x.jsonl')"
    )
    # Newest first, so ``ts DESC`` hands the worker the sentinel row FIRST.
    rows = [
        ("interrupted", _SENTINEL, "2026-07-09"),
        ("good1", "an ordinary body", "2026-07-08"),
        ("good2", "another ordinary body", "2026-07-07"),
    ]
    c.executemany(
        "INSERT INTO observations(obs_id, session_id, root_session_id, type, "
        "ts, body) VALUES (?, 'sid', 'sid', 'user', ?, ?)",
        [(i, ts, body) for i, body, ts in rows],
    )
    c.commit()
    s.close()
    return db


def _run(tmp_path, repo_root, script, name, argv):
    runner = tmp_path / name
    runner.write_text(script, encoding="utf-8")
    env = {**os.environ, "XDG_DATA_HOME": str(tmp_path)}
    return subprocess.run(  # noqa: S603
        [sys.executable, str(runner), *argv],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _signalled_run(tmp_path, repo_root, sig):
    return _run(
        tmp_path,
        repo_root,
        _SIGNAL_RUNNER,
        "signal_runner.py",
        [_SENTINEL, str(int(sig))],
    )


def _plain_run(tmp_path, repo_root):
    return _run(tmp_path, repo_root, _PLAIN_RUNNER, "plain_runner.py", [])


def _states(db):
    s = Store(db_path=db)
    try:
        return {
            r["obs_id"]: r["embedding_state"]
            for r in s._c().execute("SELECT obs_id, embedding_state FROM observations")
        }
    finally:
        s.close()


def _ledger(db):
    s = Store(db_path=db)
    try:
        return PoisonLedger(s).entries("embed:observations")
    finally:
        s.close()


def test_sigterm_mid_batch_does_not_poison_the_row_in_flight(
    cache, tmp_path, repo_root
):
    """THE REPRO. Interrupt one run; the next must not blame anybody.

    Without a handler this is a false CRITICAL alarm on every reboot, and the
    row it names is permanently held out of the index.
    """
    _signalled_run(tmp_path, repo_root, signal.SIGTERM)

    second = _plain_run(tmp_path, repo_root)

    assert second.returncode == 0, (
        "the run after a routine SIGTERM exited non-zero — that is a CRITICAL "
        f"desktop notification for a reboot.\nstdout={second.stdout!r}\n"
        f"stderr={second.stderr!r}"
    )
    assert "interrupted" not in _ledger(cache), (
        "the interrupted row was held in the quarantine ledger: a clean "
        "shutdown was read as a crash"
    )
    assert _states(cache)["interrupted"] == "ok", (
        "the interrupted row was marked 'error' and is out of the index "
        f"until a human intervenes; states={_states(cache)!r}"
    )


def test_a_sigterm_run_stops_cleanly_and_says_so(cache, tmp_path, repo_root):
    """The interrupted run itself exits 0 and leaves no claim behind.

    Exit 0 because a requested stop is not a failure — the ingest path reports
    the same event the same way. The claim is what would otherwise be read as
    a crash by the next run.
    """
    first = _signalled_run(tmp_path, repo_root, signal.SIGTERM)

    assert first.returncode == 0, (
        f"stdout={first.stdout!r}\nstderr={first.stderr!r}"
    )
    assert "INTERRUPTED" in (first.stdout + first.stderr), (
        "an interrupted run is indistinguishable from a completed one: "
        f"stdout={first.stdout!r} stderr={first.stderr!r}"
    )
    s = Store(db_path=cache)
    try:
        assert s.pending_embed_claim() is None
    finally:
        s.close()


def test_a_real_kill_is_still_blamed(cache, tmp_path, repo_root):
    """THE OTHER HALF, and a regression guard on round 1's crash blame.

    A SIGKILL cannot be handled, so the claim survives and the row is held —
    which is the whole point of the claim. Telling that apart from a SIGTERM
    is the fix; removing it would be a different bug.
    """
    killed = _signalled_run(tmp_path, repo_root, signal.SIGKILL)
    assert killed.returncode == -signal.SIGKILL

    second = _plain_run(tmp_path, repo_root)

    assert second.returncode == 1
    assert "interrupted" in _ledger(cache)
    assert _states(cache)["interrupted"] == "error"
