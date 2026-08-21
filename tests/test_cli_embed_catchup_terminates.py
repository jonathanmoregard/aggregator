"""``--catchup`` must be able to prove it stops.

Round 3's M2. ``_embed_backlog`` drains with ``while True``, and its only exit
was "``select_unembedded`` came back empty". That was a complete argument while
every batch necessarily emptied itself: rows left the backlog as ``ok``,
``skip`` or ``error``, so the next SELECT could not return them again.

Round 2's S4 removed that guarantee. ``commit_embed_batch`` is now a
compare-and-swap, so a batch whose CAS all fails writes ZERO rows — everything
stays at ``embedding_state IS NULL``, and the identical batch is selected
again. Nothing in the loop notices.

THE VALIDATOR NARROWED THE TRIGGER AND IT IS WORTH KEEPING STRAIGHT: each pass
re-reads a fresh ``src_hash`` alongside the body, so one failed CAS does not
spin — the next pass compares against the new value and succeeds. A real spin
needs rewrites landing continuously, faster than the worker can embed. That is
rare. It is also not the point: the loop has no termination argument at all,
and "chunked, with committed checkpoints" (2026-08-16) exists precisely so that
every pass either advances the watermark or ends the run. A loop that cannot
say why it stops is one that can fail to.

So: a pass that moves no row out of the backlog is a STALL, a bounded number of
consecutive stalls ends the run, and the run says so. Not a failure — the rows
are intact, at NULL, and the next tick re-reads them from their current bodies.
"""

import argparse

import numpy as np
import pytest

from aggregator.cli import _cmd_embed
from aggregator.core.store import Store

_RUNAWAY = 40


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
    for i in range(4):
        c.execute(
            "INSERT INTO observations(obs_id, session_id, root_session_id, "
            "type, ts, body) VALUES (?, 'sid', 'sid', 'user', ?, ?)",
            (f"o{i}", f"2026-01-0{i + 1}", f"body text {i}"),
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
        batch_size=2,
        reindex=False,
        yes=False,
    )
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


class _NeverCommits(Store):
    """A store whose compare-and-swap never matches — the S4 zero-write batch.

    Models sustained concurrent rewrites: ingest keeps moving the bodies, so
    every guarded write finds a fingerprint that has already changed. Vectors
    are still deleted and re-inserted; nothing leaves the backlog.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.selects = 0

    def select_unembedded(self, kind, limit=500, model=None, source=None):
        # ``source`` and ``model`` are forwarded rather than dropped: the
        # worker walks the priority groups (EMBED_BACKLOG_ORDER), so swallowing
        # the scope here would hand every group the whole backlog and the
        # runaway counter would be measuring a different loop than the one
        # under test.
        self.selects += 1
        if self.selects > _RUNAWAY:
            raise AssertionError(
                f"--catchup re-selected the same backlog {self.selects} times "
                f"without writing a row: the loop has no termination argument"
            )
        return super().select_unembedded(
            kind, limit=limit, model=model, source=source
        )

    def commit_embed_batch(self, kind, **kw):
        # Everything the worker offers is rejected by the guard.
        return [], []


def test_catchup_stops_when_no_batch_can_make_progress(cache, monkeypatch, capsys):
    """THE M2 REGRESSION: without a stall bound this never returns."""
    _stub_embedder(monkeypatch)
    store = _NeverCommits(db_path=cache)

    rc = _cmd_embed(_ns(), _store=store)
    out = capsys.readouterr()

    assert rc == 0
    assert store.selects < _RUNAWAY, "the loop kept re-reading the same rows"
    said = out.out + out.err
    assert "STOPPED WITHOUT PROGRESS" in said, said
    assert "every row is still queued" in said, said


def test_the_stall_leaves_every_row_in_the_backlog(cache, monkeypatch, capsys):
    """Stopping early must not cost a row. They are re-read next tick."""
    _stub_embedder(monkeypatch)
    _cmd_embed(_ns(), _store=_NeverCommits(db_path=cache))
    capsys.readouterr()

    after = Store(db_path=cache)
    assert len(after.select_unembedded("observations", limit=10)) == 4


def test_a_healthy_catchup_still_drains_the_whole_backlog(
    cache, monkeypatch, capsys
):
    """The bound must not cut a run that IS making progress short."""
    _stub_embedder(monkeypatch)
    store = Store(db_path=cache)

    rc = _cmd_embed(_ns(), _store=store)
    capsys.readouterr()

    assert rc == 0
    assert Store(db_path=cache).select_unembedded("observations", limit=10) == []
