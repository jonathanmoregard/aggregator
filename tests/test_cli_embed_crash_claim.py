"""A row that KILLS the worker must not be retried forever.

Chunk N made a raising row cost one row instead of the run, by catching it,
holding it in the quarantine ledger and marking it ``'error'``. Every part of
that depends on an exception being raised and caught.

A row that ends the process outright raises nothing. An OOM kill, a segfault
in a native tokenizer or torch kernel, a SIGKILL — the handler never runs, so
there is no ledger entry, no ``embedding_state`` change and no stderr line.
``select_unembedded`` orders ``ts DESC``, so the very next tick selects the
same row first and dies the same way, twice an hour, forever.

Nothing detects it. The health probe cannot: it only runs from inside an
``except`` block that is never entered. ``aggregator status`` shows an empty
ledger. The unit's ``OnFailure=`` does fire, but on a 25-30 day backfill a
failing run is not by itself distinguishable from a working one. The only
visible symptom is ``vector_index`` counts that stop moving — which is exactly
the signal the rollout notes say to watch, and exactly the signal a human is
not watching twice an hour for a month.

The fix is a claim written and COMMITTED before the attempt. A run that finds
a claim left behind by a previous process knows that row ended it, and can
blame it through the ledger machinery that already exists — backoff first,
terminal only after ``POISON_MAX_ATTEMPTS`` sightings, so a SIGTERM during a
deploy costs one delayed row rather than a condemned one.
"""

import os
import signal
import subprocess
import sys

import pytest

from aggregator.core.store import Store

# Kills the process from inside ``embed_documents`` — no exception, no
# unwinding, no ``finally``. The closest thing to an OOM kill that a test can
# arrange deterministically, and unlike a timing-based kill it always lands on
# the same row.
_SUICIDE_RUNNER = '''\
import os
import signal
import sys

import numpy as np

import aggregator.cli as cli

POISON = sys.argv[1]


class SuicidalStub:
    def __init__(self, *a, **kw):
        pass

    def embed_documents(self, docs):
        if any(POISON in d for d in docs):
            os.kill(os.getpid(), signal.SIGKILL)
        return np.zeros((len(docs), 768), dtype=np.float32)

    def embed_query(self, query):
        return np.zeros(768, dtype=np.float32)


cli.Embedder = SuicidalStub
sys.exit(cli.main(["embed", "--catchup", "--batch-size", "1"]))
'''

_POISON = "THIS-ROW-KILLS-THE-WORKER"


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
    # Newest first, so ``ts DESC`` hands the worker the poison row FIRST.
    rows = [
        ("poison", _POISON, "2026-07-09"),
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


def _run_worker(tmp_path, repo_root, env_home):
    runner = tmp_path / "suicide_runner.py"
    runner.write_text(_SUICIDE_RUNNER, encoding="utf-8")
    env = {**os.environ, "XDG_DATA_HOME": str(env_home)}
    return subprocess.run(  # noqa: S603
        [sys.executable, str(runner), _POISON],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _states(db):
    s = Store(db_path=db)
    try:
        return {
            r["obs_id"]: r["embedding_state"]
            for r in s._c().execute("SELECT obs_id, embedding_state FROM observations")
        }
    finally:
        s.close()


def test_the_first_run_is_killed_by_the_poison_row(cache, tmp_path, repo_root):
    """Baseline for the two below: the kill really is a kill."""
    proc = _run_worker(tmp_path, repo_root, tmp_path)
    assert proc.returncode == -signal.SIGKILL


def test_a_row_that_kills_the_worker_does_not_wedge_the_backlog(
    cache, tmp_path, repo_root
):
    """THE REPRO. Run 1 dies on the poison row. Run 2 must make progress.

    Without a claim, run 2 selects the same row first — ``ts DESC``, state
    untouched — and dies identically, and so does every run after it. The
    backlog never drains and nothing anywhere records why.
    """
    first = _run_worker(tmp_path, repo_root, tmp_path)
    assert first.returncode == -signal.SIGKILL

    second = _run_worker(tmp_path, repo_root, tmp_path)

    assert second.returncode != -signal.SIGKILL, (
        "the second run was killed by the same row: the backlog is wedged"
    )
    states = _states(cache)
    assert states["good1"] == "ok"
    assert states["good2"] == "ok"


def test_the_killing_row_is_set_aside_and_named(cache, tmp_path, repo_root):
    _run_worker(tmp_path, repo_root, tmp_path)
    second = _run_worker(tmp_path, repo_root, tmp_path)

    assert _states(cache)["poison"] == "error"
    assert "poison" in (second.stderr + second.stdout)


def test_the_crash_is_recorded_in_the_quarantine_ledger(cache, tmp_path, repo_root):
    """So ``aggregator status`` can show it, like every other held row."""
    from aggregator.imports.ingest_state import PoisonLedger

    _run_worker(tmp_path, repo_root, tmp_path)
    _run_worker(tmp_path, repo_root, tmp_path)

    s = Store(db_path=cache)
    try:
        entries = PoisonLedger(s).entries("embed:observations")
    finally:
        s.close()
    assert "poison" in entries


def test_a_clean_run_leaves_no_claim_behind(cache, tmp_path, repo_root):
    """A stale claim would blame an innocent row on the next run."""
    _run_worker(tmp_path, repo_root, tmp_path)
    _run_worker(tmp_path, repo_root, tmp_path)

    s = Store(db_path=cache)
    try:
        assert s.pending_embed_claim() is None
    finally:
        s.close()
