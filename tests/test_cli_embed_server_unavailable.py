"""The embed server being down is an environment fault, never a row's fault.

The ``server`` backend moves the encoder out of the worker's process into a
systemd unit that offline-AI mode stops for hours at a time. So "the encoder
is not there" goes from a rare deployment accident to a routine state, and
the worker's answer to it has to be the one the rest of the file gives an
environment fault: exit non-zero with a message that names the fix, blame no
row, hold no row, leave no claim, and leave the backlog where it was.

Both moments it can happen are covered: before the run (the unit is already
stopped when the timer fires) and during it (the unit is stopped under a
running catchup). Real HTTP against ``tests/embed_server_stub.py``; the only
thing replaced is the socket on the other end.
"""

from __future__ import annotations

import argparse

import pytest

import aggregator.core.embed as embed_mod
from aggregator.core.store import Store
from tests.embed_server_stub import EmbedServerStub, short_socket_path

ROWS = 6


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.delenv("AGGREGATOR_EMBED_BACKEND", raising=False)
    db = tmp_path / "cache.db"
    s = Store(db_path=db)
    s.migrate()
    c = s._c()
    c.execute(
        "INSERT INTO sessions(session_id, root_session_id, kind, first_ts, "
        "last_ts, jsonl_path) VALUES ('sid', 'sid', 'session', "
        "'2026-01-01', '2026-01-01', '/tmp/x.jsonl')"
    )
    for i in range(ROWS):
        c.execute(
            "INSERT INTO observations(obs_id, session_id, root_session_id, "
            "type, ts, body) VALUES (?, 'sid', 'sid', 'user', ?, ?)",
            (f"o{i}", f"2026-01-0{i + 1}", f"distinct body number {i}"),
        )
    c.commit()
    s.close()
    return db


def _ns() -> argparse.Namespace:
    return argparse.Namespace(
        catchup=True,
        once=False,
        source="observations",
        batch_size=500,
        reindex=False,
        yes=False,
    )


def _run(cache) -> int:
    from aggregator.cli import _cmd_embed

    store = Store(db_path=cache)
    try:
        return _cmd_embed(_ns(), _store=store)
    finally:
        store.close()


def _ledger_and_states(cache):
    s = Store(db_path=cache)
    try:
        c = s._c()
        held = [dict(r) for r in c.execute("SELECT * FROM quarantine")]
        states = {
            r["obs_id"]: r["embedding_state"]
            for r in c.execute("SELECT obs_id, embedding_state FROM observations")
        }
        claim = s.embed_claim_path.exists()
    finally:
        s.close()
    return held, states, claim


def test_a_stopped_server_fails_the_run_and_touches_nothing(cache, monkeypatch, capsys):
    # The deployed transport, stopped: systemd has removed the socket.
    monkeypatch.setenv(embed_mod.EMBED_URL_ENV, f"unix://{short_socket_path()}")

    rc = _run(cache)

    err = capsys.readouterr().err
    assert rc != 0, "a run that embedded nothing reported success"
    assert "aggregator-embed-server" in err, f"no remedy named: {err!r}"
    assert "Traceback" not in err
    held, states, claim = _ledger_and_states(cache)
    assert held == []
    assert set(states.values()) == {None}, "the backlog moved"
    assert not claim


def test_a_server_stopped_mid_run_blames_no_row(cache, monkeypatch, capsys):
    """One request served, then the unit goes away under the worker."""
    with EmbedServerStub(unix_path=short_socket_path()) as stub:
        monkeypatch.setenv(embed_mod.EMBED_URL_ENV, stub.url)
        # A row is one request here (each body is a single chunk), plus the
        # health probe after the failure. Serving exactly one means one row
        # can finish and the rest meet a dead socket.
        stub.die_after = 1

        rc = _run(cache)

    err = capsys.readouterr().err
    assert rc != 0
    assert "aggregator-embed-server" in err, f"no remedy named: {err!r}"
    held, states, claim = _ledger_and_states(cache)
    assert held == [], f"rows were put in the poison ledger: {held}"
    assert "error" not in states.values(), f"rows were marked failed: {states}"
    assert not claim, "a claim survived, so the next run would blame a good row"
    # The row that was served is real work and must have been kept.
    assert list(states.values()).count("ok") == 1
    assert list(states.values()).count(None) == ROWS - 1
