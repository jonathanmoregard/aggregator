"""A lock must not leave behind a claim that condemns a good row NEXT run.

OBSERVED IN PRODUCTION, 2026-08-27, four times in three hours. The worker
embedded a row successfully, ``release_embed_claim`` then raised
``database is locked`` because the 30-minute ingest timer overlapped, the run
exited 1 — and the NEXT run read the claim still sitting on disk, concluded a
previous process had died on that row, and booked it into the poison ledger:

    embed: records row 'dropbox:…/CCL polska.docx' killed a previous worker
    process and was set aside — EmbedWorkerKilledError: … no exception was
    raised, so this is a kill (OOM, segfault …)

Nothing had been killed. The row had embedded fine.

WHY THE EXISTING COVERAGE MISSED IT. ``test_cli_embed_store_fault.py`` already
has ``test_a_lock_on_the_release_is_treated_the_same_way``, and it passes: it
asserts that THIS run holds nobody. The damage is entirely in the NEXT run, and
nothing asserted across that boundary. The ``sqlite3.Error`` handler's own
comment says "abort, blame nobody, leave the backlog untouched" — and it does
all three, while leaving on disk the one artifact that makes the next process
blame somebody. The intent was defeated one function call away.

THE ROOT CAUSE IS WHERE THE CLAIM LIVED, not which handler forgot to clear it.
The claim is a crash detector, and it was stored inside the very resource whose
unavailability is the most common NON-crash failure. Any clean-exit path that
has to write to a locked database in order to say "I exited cleanly" cannot say
it at exactly the moment it most needs to. Adding a retry around the DELETE
narrows the window; it cannot close it, because the whole failure mode is that
the database is unavailable for longer than the process is willing to wait.

So the claim moved next to the lock file the worker already holds. Removing a
file needs no database, so a clean exit can always disown its row; a SIGKILL
still cannot run code, so a real crash still leaves the claim exactly as
before. The detector keeps the evidence it was built for and loses the evidence
it was fabricating.
"""

import argparse
import sqlite3

import numpy as np
import pytest

from aggregator.cli import _cmd_embed
from aggregator.core.store import EMBED_CLAIM_KEY, Store
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


class _LockedOnBatchCommit(Store):
    """The lock lands on the write the deployed worker actually died on.

    Production, 2026-08-27: "the cache could not be written while embedding
    records (OperationalError: database is locked)" — that is
    ``commit_embed_batch``, the one-transaction write at the end of the batch,
    losing to the 30-minute ingest timer.

    NOT an override of ``release_embed_claim``, though that is where the old
    damage came from. The release is a file unlink now, so making it raise
    ``sqlite3.OperationalError`` would test a world this code no longer has.
    The interesting question after the fix is not "what if the release fails"
    — it cannot — but "does an aborted run still leave a claim behind", which
    is what the test below asks.
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


def test_a_run_that_dies_on_a_lock_leaves_no_claim_for_the_NEXT_run(
    cache, monkeypatch, capsys
):
    """The production regression, asserted across the run boundary.

    Two runs. The first loses to a database lock and exits 1, exactly as the
    deployed worker did four times in three hours on 2026-08-27. The second is
    an ordinary healthy run — and it must not discover a crash that never
    happened.

    THE BOUNDARY IS THE WHOLE TEST. ``test_cli_embed_store_fault.py`` already
    asserted that the FIRST run holds nobody, and it passed throughout the
    outage: every row of the damage was in the second run. A claim is the only
    thing that carries state across that boundary, so it is the only thing
    that can carry the bug across it.
    """
    _stub_embedder(monkeypatch)

    rc_first = _cmd_embed(_ns(), _store=_LockedOnBatchCommit(db_path=cache))
    capsys.readouterr()
    assert rc_first == 1, "a store fault must still fail the run loudly"

    claim_file = Store(db_path=cache).embed_claim_path
    assert not claim_file.exists(), (
        f"an aborted run left {claim_file.name} behind. Every exit path that "
        f"runs code must disown its row, or the next run reads the leftover "
        f"as a kill — which is exactly what condemned four good rows on "
        f"2026-08-27."
    )

    rc_second = _cmd_embed(_ns(), _store=Store(db_path=cache))
    err = capsys.readouterr().err

    assert _held(cache) == {}, (
        f"the next run condemned a row for a lock the previous run hit: "
        f"{_held(cache)}. The claim outlived the process that could no longer "
        f"clear it, and _blame_crashed_row read it as a kill."
    )
    assert "killed a previous worker process" not in err, (
        "reported a crash that did not happen"
    )
    assert rc_second == 0
    assert set(_states(cache).values()) == {"ok"}, (
        "the innocent rows did not go on to embed"
    )


def test_the_claim_is_released_even_when_every_database_write_fails(
    cache, monkeypatch
):
    """Disowning a row must not depend on the resource that just failed.

    This is the property the fix buys, stated on its own: ``release`` is
    reachable when sqlite is not. A test that only went through ``_cmd_embed``
    could pass on a lucky retry; this one refuses the database outright.
    """
    s = Store(db_path=cache)
    try:
        s.claim_embed_row("observations", "o1")
        assert s.pending_embed_claim() == ("observations", "o1")

        def refuse(*_a, **_kw):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(Store, "_c", refuse)
        s.release_embed_claim()
        monkeypatch.undo()

        assert s.pending_embed_claim() is None, (
            "the claim survived a release that could not reach the database"
        )
    finally:
        s.close()


def test_a_real_kill_still_leaves_a_claim_to_blame(cache):
    """THE DETECTOR MUST NOT BE TRADED AWAY for the false positives.

    A SIGKILL runs no code, so nothing removes the file — which is exactly the
    evidence ``_blame_crashed_row`` exists to read. If this ever goes green by
    the claim simply never persisting, the wedge is back: a row that reliably
    kills the worker is handed to the next tick, forever, with an empty ledger
    and nothing on stderr.
    """
    s = Store(db_path=cache)
    try:
        s.claim_embed_row("observations", "o2")
    finally:
        s.close()

    # A new process, as after a kill: nothing ran in the old one.
    s2 = Store(db_path=cache)
    try:
        assert s2.pending_embed_claim() == ("observations", "o2")
    finally:
        s2.close()


def test_a_claim_left_in_meta_by_an_older_build_blames_nobody(cache, capsys):
    """The upgrade path, and it must open by NOT condemning anything.

    A cache written by the previous build can carry ``meta.embed_inflight`` —
    the live one did, naming a 96 MB PDF, at the moment this was written. That
    claim is exactly the artifact whose meaning is unknowable: it is what a
    crash leaves AND what a lock-on-release leaves, and nothing on disk tells
    the two apart. Unknowable evidence convicts nobody.
    """
    s = Store(db_path=cache)
    try:
        c = s._c()
        c.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            (EMBED_CLAIM_KEY, '{"kind": "observations", "row_id": "o0"}'),
        )
        c.commit()

        assert s.pending_embed_claim() is None, (
            "a legacy in-database claim was read as a crash. It cannot be: "
            "the build that wrote it could not distinguish a kill from a "
            "locked release, so the claim carries no information."
        )
        # And it is cleaned up rather than left to be re-examined forever.
        left = s._c().execute(
            "SELECT count(*) FROM meta WHERE key = ?", (EMBED_CLAIM_KEY,)
        ).fetchone()[0]
        assert left == 0, "the legacy claim was left in meta as litter"
    finally:
        s.close()


def test_an_unreadable_claim_file_blames_nobody(cache):
    """Corruption names no row, so it can condemn no row."""
    s = Store(db_path=cache)
    try:
        s.claim_embed_row("observations", "o1")
        s.embed_claim_path.write_text("{not json at all")
        assert s.pending_embed_claim() is None
    finally:
        s.close()


def test_releasing_without_a_claim_is_not_an_error(cache):
    """Called on every clean row, including the ones that never claimed."""
    s = Store(db_path=cache)
    try:
        s.release_embed_claim()
        s.release_embed_claim()
        assert s.pending_embed_claim() is None
    finally:
        s.close()
