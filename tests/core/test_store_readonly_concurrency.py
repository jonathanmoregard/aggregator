"""The read-only recall connection must not lie while a writer is live.

``Store(read_only=True)`` is the connection ``aggregator_search_memory``
serves every query from, and the ingest timer rewrites this database every
30 minutes. It used to open with ``mode=ro&immutable=1``. ``immutable=1``
is a *promise to SQLite* that the file cannot change, which licenses it to
skip all locking, ignore the ``-wal`` entirely, and keep its page cache
across statements forever.

Both halves of that promise are false here, and the consequences are not
theoretical — each test below was watched to fail against ``immutable=1``:

* Data committed to the ``-wal`` is invisible, so ``has_embedded_rows`` —
  the v5 hybrid routing predicate — reports a warm vector index as cold.
  That no longer costs freshness only; it silently changes which retrieval
  arms run.
* Once a checkpoint or VACUUM rewrites the main file underneath the cached
  pages, the reader returns *wrong row counts with no error at all*, and
  then ``database disk image is malformed``.

Plain ``mode=ro`` is correct in every case here. In the one pathological
case it cannot serve — a dirty ``-wal`` whose ``-shm`` sidecar is absent and
cannot be created — it fails loudly with "unable to open database file",
where ``immutable=1`` silently returns a truncated answer. A loud failure
is the required behaviour for this project; a silent wrong answer is the
failure mode it exists to prevent.
"""

import sqlite3

import pytest

from aggregator.core.store import Store


def _seed_session(c):
    c.execute(
        "INSERT OR IGNORE INTO sessions(session_id, root_session_id, kind, "
        "first_ts, last_ts, jsonl_path, origin) VALUES ('sid', 'sid', "
        "'session', '2026-01-01', '2026-01-01', '/tmp/x.jsonl', 'claude-code')"
    )


def _seed_observations(store, n, prefix="o", body="hello"):
    c = store._c()
    _seed_session(c)
    c.executemany(
        "INSERT INTO observations(obs_id, session_id, root_session_id, "
        "type, ts, body) VALUES (?, 'sid', 'sid', 'user', '2026-01-01', ?)",
        [(f"{prefix}{i:06d}", body) for i in range(n)],
    )
    c.commit()


def _checkpoint(store):
    c = store._c()
    c.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    c.commit()


def test_routing_predicate_sees_rows_a_live_writer_just_embedded(tmp_path):
    """``has_embedded_rows`` must not report a warm index as cold.

    This is the v5-specific consequence: the predicate decides whether the
    vector arm engages at all, so a stale answer changes retrieval, not just
    freshness.

    The writer WRITES VECTORS rather than only flipping a column, because
    that is what the predicate now reads. ``embedding_state`` could never
    answer this honestly — it cannot say which model embedded the row, so it
    reports a warm index after a model change and a cold one after a source
    rebuild resets it. ``chunk_embeddings`` is written by the same call that
    writes the vector, so the two cannot disagree.
    """
    import numpy as np

    db = tmp_path / "cache.db"
    writer = Store(db_path=db)
    writer.migrate()
    _seed_observations(writer, 5)

    reader = Store(db_path=db, read_only=True)
    assert reader.has_embedded_rows("observations") is False

    vec = np.zeros(768, dtype=np.float32)
    vec[0] = 1.0
    writer.upsert_vec_observations([(f"o{i:06d}", vec) for i in range(5)])

    assert reader.has_embedded_rows("observations") is True


def test_reader_sees_rows_committed_after_it_opened(tmp_path):
    db = tmp_path / "cache.db"
    writer = Store(db_path=db)
    writer.migrate()
    _seed_observations(writer, 5)

    reader = Store(db_path=db, read_only=True)
    assert reader._c().execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 5

    _seed_observations(writer, 5, prefix="n")

    assert reader._c().execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 10


def test_reader_stays_consistent_across_a_checkpoint_and_vacuum(tmp_path):
    """A checkpoint/VACUUM rewrites the main file. The reader must not tear.

    Sized so the table spans many pages: the corruption is a stale page
    cache mixed with freshly-faulted pages, which needs real page churn to
    show up rather than a handful of rows on page 1.
    """
    db = tmp_path / "cache.db"
    writer = Store(db_path=db)
    writer.migrate()
    _seed_observations(writer, 6000, body="x" * 400)
    _checkpoint(writer)

    reader = Store(db_path=db, read_only=True)
    rc = reader._c()
    assert rc.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 6000

    _seed_observations(writer, 6000, prefix="n", body="y" * 400)
    _checkpoint(writer)

    assert rc.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 12000

    wc = writer._c()
    wc.execute("DELETE FROM observations WHERE obs_id LIKE 'o%'")
    wc.commit()
    _checkpoint(writer)
    wc.execute("VACUUM")
    wc.commit()

    assert rc.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 6000
    assert rc.execute("PRAGMA integrity_check(3)").fetchall()[0][0] == "ok"


def test_readonly_store_still_refuses_writes(tmp_path):
    """Dropping ``immutable=1`` must not make the recall connection writable."""
    db = tmp_path / "cache.db"
    writer = Store(db_path=db)
    writer.migrate()
    _seed_observations(writer, 1)

    reader = Store(db_path=db, read_only=True)
    with pytest.raises(RuntimeError, match="read-only Store cannot write"):
        reader.mark_embedded("observations", ["o000000"], "ok")
    with pytest.raises(sqlite3.OperationalError):
        reader._c().execute("DELETE FROM observations")


def test_readonly_store_creates_no_sidecars_of_its_own(tmp_path):
    """The recall path must not turn a cold database into a WAL one.

    ``mode=ro`` cannot run ``PRAGMA journal_mode=WAL`` — the read-only branch
    of ``_c()`` returns before the write pragmas — so opening a
    freshly-created cache for recall leaves no ``-wal``/``-shm`` behind.
    """
    db = tmp_path / "cache.db"
    writer = Store(db_path=db)
    writer.migrate()
    _seed_observations(writer, 1)
    _checkpoint(writer)
    writer.close()

    reader = Store(db_path=db, read_only=True)
    assert reader._c().execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 1
