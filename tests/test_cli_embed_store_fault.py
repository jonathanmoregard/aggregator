"""A database lock is not a bad row, and must not be ledgered as one.

Round 3's M3. Round 2's chunk N established the discriminator for embedder
faults: when a row fails, re-run the embedder on a known-good probe. If the
probe works the failure told two bodies apart and is a property of the ROW; if
the probe fails too the failure discriminates between nothing, so nothing may
be blamed on a record and the run aborts holding no one responsible.

That reasoning was applied to the embedder and stopped there. The same ``try``
block also contains two STORE calls — ``claim_embed_row`` and
``release_embed_claim`` — and ``_embedder_is_healthy`` probes only the
embedder. So a ``sqlite3.OperationalError: database is locked``, which the
30-minute ingest timer can produce at any moment, arrived at a handler that
asked the embedder whether the ROW was bad, got "the embedder is fine", and
condemned a perfectly good row: a ledger entry, ``embedding_state='error'``, a
stderr line naming it, and a non-zero exit blaming data for an environment
fault.

The row then sits out of the index behind a backoff, and after
``POISON_MAX_ATTEMPTS`` sightings it is terminal — permanently missing from the
vector arm because two writers overlapped once.

A store fault does not discriminate between rows. It is the machine's, like a
cold model or an OOM, and it gets the same answer: abort, blame nobody, leave
the backlog exactly where it was.
"""

import argparse
import sqlite3

import numpy as np
import pytest

from aggregator.cli import _cmd_embed
from aggregator.core.store import Store
from aggregator.imports.ingest_state import PoisonLedger


@pytest.fixture
def cache(tmp_path):
    db = tmp_path / "cache.db"
    s = Store(db_path=db)
    s.migrate()
    c = s._c()
    c.execute(
        "INSERT INTO sessions(session_id, root_session_id, kind, first_ts, "
        "last_ts, jsonl_path) VALUES ('sid','sid','session','2026-01-01',"
        "'2026-01-01','/tmp/x.jsonl')"
    )
    for i in range(3):
        c.execute(
            "INSERT INTO observations(obs_id, session_id, root_session_id, "
            "type, ts, body) VALUES (?, 'sid', 'sid', 'user', ?, ?)",
            (f"o{i}", f"2026-01-0{i + 1}", f"a perfectly good body {i}"),
        )
    c.commit()
    s.close()
    return db


def _stub_embedder(monkeypatch):
    class StubEmbedder:
        def __init__(self, *a, **kw):
            pass

        def embed_documents(self, docs):
            return np.array(
                [[float(i)] * 768 for i in range(len(docs))], dtype=np.float32
            )

    monkeypatch.setattr("aggregator.cli.Embedder", StubEmbedder)


def _ns(**kw):
    ns = argparse.Namespace(
        catchup=True,
        once=False,
        seed_models=False,
        source="observations",
        batch_size=500,
        reindex=False,
        yes=False,
    )
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


class _LockedOnClaim(Store):
    """The lock lands where the ingest timer would put it: on the claim write."""

    def claim_embed_row(self, kind, row_id):
        raise sqlite3.OperationalError("database is locked")


class _LockedOnBatchCommit(Store):
    """The contended write that is still a database write.

    SUCCESSOR TO ``_LockedOnRelease``, and the reason for the swap is a real
    contract change rather than a test tidy-up. This class used to override
    ``release_embed_claim``, because that was the second of two store calls
    inside the same ``try`` block and could therefore lose to the ingest
    timer's lock like any other write.

    It cannot any more. The in-flight claim moved out of the database and next
    to it (``Store.embed_claim_path``) precisely because a clean exit must be
    able to disown its row while sqlite is unavailable — an override that makes
    the release raise ``sqlite3.OperationalError`` now models a world that no
    longer exists, and a test built on it asserts nothing about this code.

    So the injection point moves to the write that IS still contended and still
    inside a guard: the one-transaction ``commit_embed_batch`` at the end of the
    batch. That is also the call the deployed worker actually died on — the
    production line reads "the cache could not be written while embedding
    records (OperationalError: database is locked)".
    """

    def commit_embed_batch(self, *a, **kw):
        raise sqlite3.OperationalError("database is locked")


def _held(db):
    s = Store(db_path=db)
    try:
        return PoisonLedger(s).entries("embed:observations")
    finally:
        s.close()


def _states(db):
    s = Store(db_path=db)
    try:
        return {
            r["obs_id"]: r["embedding_state"]
            for r in s._c().execute(
                "SELECT obs_id, embedding_state FROM observations"
            )
        }
    finally:
        s.close()


def test_a_locked_database_does_not_condemn_a_row(cache, monkeypatch, capsys):
    """THE M3 REGRESSION: an innocent row held in the poison ledger."""
    _stub_embedder(monkeypatch)

    rc = _cmd_embed(_ns(), _store=_LockedOnClaim(db_path=cache))
    err = capsys.readouterr().err

    assert rc == 1, "a store fault must still fail the run loudly"
    assert _held(cache) == {}, (
        f"blamed a row for a database lock: {_held(cache)}"
    )
    assert set(_states(cache).values()) == {None}, (
        "a row was marked out of the backlog for an environment fault"
    )
    assert "database is locked" in err
    assert "not bad data" in err


def test_the_error_names_the_store_not_the_row(cache, monkeypatch, capsys):
    _stub_embedder(monkeypatch)
    _cmd_embed(_ns(), _store=_LockedOnClaim(db_path=cache))
    err = capsys.readouterr().err

    assert "could not be embedded and was set aside" not in err, (
        "reported an environment fault in the vocabulary of a poison row"
    )


def test_a_lock_on_the_batch_commit_is_treated_the_same_way(
    cache, monkeypatch, capsys
):
    """Every store call in the path gets the same answer, not just the first.

    Was ``test_a_lock_on_the_release_is_treated_the_same_way`` — see
    ``_LockedOnBatchCommit`` for why the injection point moved. The behaviour
    under test is unchanged: a lock anywhere in the store path aborts the run
    without blaming a row for it.
    """
    _stub_embedder(monkeypatch)

    rc = _cmd_embed(_ns(), _store=_LockedOnBatchCommit(db_path=cache))
    capsys.readouterr()

    assert rc == 1
    assert _held(cache) == {}
    assert set(_states(cache).values()) == {None}, (
        "a row left the backlog for a lock on the batch commit"
    )


def test_a_genuinely_bad_row_is_still_blamed(cache, monkeypatch, capsys):
    """The discriminator must keep working in the direction it was built for."""

    class PickyEmbedder:
        def __init__(self, *a, **kw):
            pass

        def embed_documents(self, docs):
            if any("body 2" in d for d in docs):
                raise RuntimeError("this row kills the tokenizer")
            return np.array(
                [[float(i)] * 768 for i in range(len(docs))], dtype=np.float32
            )

    monkeypatch.setattr("aggregator.cli.PickyEmbedder", PickyEmbedder, raising=False)
    monkeypatch.setattr("aggregator.cli.Embedder", PickyEmbedder)

    rc = _cmd_embed(_ns(), _store=Store(db_path=cache))
    capsys.readouterr()

    assert rc == 1
    assert "o2" in _held(cache), "a reproducibly-bad row must still be set aside"
    assert _states(cache)["o2"] == "error"
    # And the innocent rows around it still embedded.
    assert _states(cache)["o0"] == "ok"
