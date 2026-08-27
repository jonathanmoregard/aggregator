"""A stop must be reachable INSIDE a row, not only between rows.

THE PHASE-3 HIGH THIS CLOSES. The stop flag was read at a row boundary, and a
row's encode was one ``embed_documents`` call over every chunk of that row —
so the flag was not read again until the call returned. Measured read-only
against the live cache at the chunker's ``chunk-4000-400`` geometry and ~20 s
per chunk: **1,348 rows exceed 300 s in that single call and the largest is 257
chunks, about 86 minutes**, against a ``TimeoutStopSec`` of 5 minutes.

Stop the unit while one of those is encoding and the sequence is: SIGTERM sets
the flag, nobody reads it, systemd escalates to SIGKILL, the on-disk claim
survives — and the next run's ``_blame_crashed_row`` books a perfectly good row
into the poison ledger. Three sightings make it terminal, i.e. permanently
absent from the vector arm. Reproduced against a real spawned worker replaying
systemd's actual SIGTERM → wait → SIGKILL, with a control proving the victim
was a good row.

NOTHING FINITE FIXES THIS FROM THE UNIT SIDE, which is why the fix is here.
Raising ``TimeoutStopSec`` past 86 minutes would make a manual stop no longer
bound a wedged worker — and with ``TimeoutStartSec=infinity`` that stop is the
LAST bound there is. Shortening it spreads the same gap from the long tail to
every routine stop. The window cannot cover an unbounded call; the call has to
stop being unbounded.

So the encode is sliced and the flag is read between slices. The row is still
the unit the CLAIM covers and still the unit that COMMITS — a partially
embedded row must never reach ``commit_embed_batch``, because
``chunk_embeddings`` rows under its ``owner_id`` would make
``select_unembedded``'s LEFT JOIN read it as embedded and nothing would ever
come back for the rest of it. What changes is only that the worker can now put
the row down between slices instead of being killed holding it.
"""

from __future__ import annotations

import argparse

import numpy as np
import pytest

from aggregator import cli
from aggregator.cli import _embed_batch, _EmbedOutcome
from aggregator.core.chunk import chunk_body
from aggregator.core.store import Store
from aggregator.imports.ingest_state import PoisonLedger

#: Long enough that the chunker makes many chunks of it. The exact count does
#: not matter to any assertion below — every one of them is expressed in terms
#: of what the recording embedder actually saw.
_LONG_BODY = ("a paragraph that is long enough to matter. " * 400 + "\n\n") * 12


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
    c.execute(
        "INSERT INTO observations(obs_id, session_id, root_session_id, "
        "type, ts, body) VALUES ('long', 'sid', 'sid', 'user', "
        "'2026-01-02', ?)",
        (_LONG_BODY,),
    )
    c.execute(
        "INSERT INTO observations(obs_id, session_id, root_session_id, "
        "type, ts, body) VALUES ('short', 'sid', 'sid', 'user', "
        "'2026-01-01', 'a short one')"
    )
    c.commit()
    s.close()
    return db


class _Recorder:
    """Counts encoder calls and can trip the stop flag partway through one."""

    def __init__(self, stop_after_calls: int | None = None):
        self.calls: list[int] = []
        self.stopped = False
        self._stop_after = stop_after_calls

    def embed_documents(self, docs):
        self.calls.append(len(docs))
        if self._stop_after is not None and len(self.calls) >= self._stop_after:
            self.stopped = True
        return np.array(
            [[float(i)] * 768 for i in range(len(docs))], dtype=np.float32
        )

    def stop(self) -> bool:
        return self.stopped

    @property
    def chunks_encoded(self) -> int:
        return sum(self.calls)


def _rows(db, kind="observations"):
    s = Store(db_path=db)
    try:
        return s.select_unembedded(kind, limit=500, source=None)
    finally:
        s.close()


def _run_batch(db, embedder, stop=None):
    s = Store(db_path=db)
    try:
        outcome = _EmbedOutcome()
        rows = s.select_unembedded("observations", limit=500, source=None)
        moved = _embed_batch(
            s, embedder, "observations", rows, PoisonLedger(s), outcome, stop
        )
        return s, outcome, moved
    finally:
        s.close()


def test_a_long_row_is_encoded_in_more_than_one_call(cache):
    """The premise. One call per row is what made the stop unreachable."""
    rec = _Recorder()
    _run_batch(cache, rec)

    assert len(rec.calls) > 1, (
        "the whole row still goes to the encoder in a single call, so the "
        "stop flag cannot be read until it returns"
    )
    assert max(rec.calls) <= cli._MAX_CHUNKS_PER_ENCODE, (
        f"a slice of {max(rec.calls)} chunks exceeds the "
        f"{cli._MAX_CHUNKS_PER_ENCODE}-chunk bound, so the worst-case latency "
        f"to a stop is longer than the slice size promises"
    )


def test_the_slice_is_short_enough_to_answer_inside_the_stop_window(cache):
    """The number has to be justified against the window it exists to fit.

    ``TimeoutStopSec`` is 5 minutes and a chunk is ~20 s, so a slice must be
    well under 15 chunks or the stop is unreachable again by arithmetic. The
    assertion is deliberately loose: it pins the ORDER of magnitude, which is
    what went wrong, not the exact constant.
    """
    worst_case_seconds = cli._MAX_CHUNKS_PER_ENCODE * cli._SECONDS_PER_CHUNK
    assert worst_case_seconds <= 120, (
        f"a slice can take {worst_case_seconds}s, which leaves too little of "
        f"the 300s TimeoutStopSec for the flush and commit that follow it"
    )


def test_a_stop_partway_through_a_row_stops_encoding_it(cache):
    """The flag is read between slices, not only between rows."""
    rec = _Recorder(stop_after_calls=1)
    _run_batch(cache, rec, stop=rec.stop)

    unbounded = _Recorder()
    _run_batch(cache, unbounded)

    assert rec.chunks_encoded < unbounded.chunks_encoded, (
        f"the stop was set after the first slice but the row still encoded "
        f"{rec.chunks_encoded} chunks, the same work an uninterrupted run "
        f"does ({unbounded.chunks_encoded}). The flag is not being read "
        f"inside the row."
    )


def test_a_row_interrupted_partway_is_not_marked_embedded(cache):
    """A PARTIAL ROW MUST NOT COMMIT — the LEFT JOIN would call it done.

    ``chunk_embeddings`` rows written under this row's ``owner_id`` make
    ``select_unembedded`` skip it forever, so the missing chunks would never
    be asked for again. The row stays at NULL and the next run re-embeds it
    from scratch; re-doing one row is the cost, and it is the right one.
    """
    rec = _Recorder(stop_after_calls=1)
    _run_batch(cache, rec, stop=rec.stop)

    s = Store(db_path=cache)
    try:
        states = {
            r["obs_id"]: r["embedding_state"]
            for r in s._c().execute(
                "SELECT obs_id, embedding_state FROM observations"
            )
        }
        owners = {
            r[0]
            for r in s._c().execute(
                "SELECT DISTINCT owner_id FROM chunk_embeddings"
            )
        }
    finally:
        s.close()

    assert states["long"] is None, (
        f"the interrupted row was marked {states['long']!r}; it must stay in "
        f"the backlog"
    )
    assert "long" not in owners, (
        "chunks of a partially-encoded row were committed — select_unembedded "
        "is a LEFT JOIN against chunk_embeddings, so the row now reads as "
        "embedded and the rest of it is unreachable"
    )


def test_an_interrupted_row_leaves_no_claim_to_blame_it(cache):
    """The whole point: put the row down rather than be killed holding it."""
    rec = _Recorder(stop_after_calls=1)
    _run_batch(cache, rec, stop=rec.stop)

    s = Store(db_path=cache)
    try:
        assert s.pending_embed_claim() is None, (
            "the interrupted row is still claimed, so the next run's "
            "_blame_crashed_row will book it as poison — which is the defect "
            "this change exists to close"
        )
    finally:
        s.close()


def test_the_interruption_is_reported_as_one(cache):
    rec = _Recorder(stop_after_calls=1)
    _s, outcome, _moved = _run_batch(cache, rec, stop=rec.stop)

    assert outcome.interrupted, (
        "a stop inside a row must set the same flag a stop between rows does, "
        "or the run reports a clean finish over an unfinished backlog"
    )


def test_slicing_hands_the_model_the_same_chunks_in_the_same_order(cache):
    """Correctness of the split itself, and the reason it is asserted exactly.

    A slice that dropped, duplicated or reordered chunks passes every other
    assertion in this file: the counts still look plausible, the row still
    marks ``'ok'``, the stop still works. It would corrupt the index silently
    — each chunk stored under the wrong ``chunk_id``, so retrieval returns
    text that is not what matched. Worst failure mode here, least visible.

    Compared against ``chunk_body`` rather than against a second unsliced run,
    because a second run over the same cache has nothing left to embed. That
    also makes the oracle independent of this module: if slicing ever changed
    the chunking, the chunker still says what the chunking should have been.
    """
    seen: list[list[str]] = []

    class Echo:
        def embed_documents(self, docs):
            seen.append(list(docs))
            return np.array(
                [[float(len(d))] * 768 for d in docs], dtype=np.float32
            )

    _run_batch(cache, Echo())
    sliced = [d for call in seen for d in call]

    # ``select_unembedded`` is ts DESC, so the long row (2026-01-02) is handed
    # over before the short one (2026-01-01).
    expected = chunk_body(_LONG_BODY) + chunk_body("a short one")

    assert sliced == expected, (
        f"the sliced encode handed the model {len(sliced)} chunks where an "
        f"unsliced one would hand it {len(expected)}, or handed them over in a "
        f"different order. Chunk N of a row must reach the model as chunk N, "
        f"because that position is what its chunk_id is built from."
    )


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


def test_an_uninterrupted_long_row_still_embeds_completely(cache):
    """Slicing must not cost the ordinary path anything."""
    rec = _Recorder()
    _s, outcome, moved = _run_batch(cache, rec)

    s = Store(db_path=cache)
    try:
        states = {
            r["obs_id"]: r["embedding_state"]
            for r in s._c().execute(
                "SELECT obs_id, embedding_state FROM observations"
            )
        }
        n_chunks = s._c().execute(
            "SELECT count(*) FROM chunk_embeddings WHERE owner_id = 'long'"
        ).fetchone()[0]
    finally:
        s.close()

    assert states["long"] == "ok"
    assert states["short"] == "ok"
    assert moved == 2
    assert not outcome.interrupted
    assert n_chunks == rec.chunks_encoded - 1, (
        "every chunk the encoder was given for the long row should have been "
        "written (the short row accounts for the remaining one)"
    )
