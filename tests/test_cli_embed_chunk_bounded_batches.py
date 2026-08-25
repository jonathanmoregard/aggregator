"""A batch is bounded by CHUNKS, because chunks are what cost time.

``--batch-size`` bounded the batch by ROWS while the bill is per CHUNK, and
rows differ in chunk count by two orders of magnitude. The checkpoint interval
that fell out of that mismatch was nobody's decision:

Measured read-only against the live cache on 2026-08-25, at the chunker's own
``chunk-4000-400`` geometry — dropbox holds 1536 records ≈ 3838 chunks, and the
FIRST 500-row batch of it is ≈1149 chunks. ``docs/embedding-throughput.md``
measures the encoder at ~40 tok/s flat, i.e. ~19-20 seconds per 4000-character
chunk. So the first durable checkpoint of a dropbox backfill was **~6.4 hours**
away, and it was confirmed live: twenty minutes into a run,
``SELECT count(*) FROM chunk_embeddings`` was still 0.

The user's standing directive of 2026-08-16 says a batch must be bounded "so a
kill at any moment loses at most one chunk and the next run starts where this
one stopped". 6.4 unprotected hours is not that. And because the bound was on
rows, the window moved silently with the source: the same ``--batch-size 500``
is ~1149 chunks on dropbox and a completely different number on ``sessions``.

So the batch stops at ``cli._MAX_BATCH_CHUNKS`` chunks. Rows remain the unit of
claim, blame and compare-and-swap — none of that machinery moves — but how many
of them are pulled into one batch is now chosen from what they will cost.

THE ONE EXEMPTION, and it is load-bearing rather than a concession: the first
candidate row is ALWAYS taken, whatever it costs. A row is indivisible at the
commit boundary (committing half its chunks would put ``chunk_embeddings`` rows
under its ``owner_id``, and ``select_unembedded``'s LEFT JOIN would then read
the row as embedded and never come back for the rest). A packer that refused an
oversized row would return an empty batch, which writes nothing, which is a
stall — three of those end the run, and since ``ORDER BY ts DESC`` hands back
the same row next tick, that row and everything behind it starve forever. A cap
that can strand a row is a worse bug than the one being fixed.
"""

from __future__ import annotations

import argparse

import numpy as np
import pytest

import aggregator.cli as cli
from aggregator.core.chunk import CHUNK_MAX_LEN, CHUNK_OVERLAP, chunk_body
from aggregator.core.store import Store

_DIM = 768
_STRIDE = CHUNK_MAX_LEN - CHUNK_OVERLAP


def _body_of(n_chunks: int, tag: str) -> str:
    """A body the chunker splits into exactly ``n_chunks``.

    No blank line anywhere in it, so ``chunk_body`` takes the hard-window
    branch and the count is arithmetic rather than a property of the prose:
    the first window is ``CHUNK_MAX_LEN`` and every later one advances by
    ``CHUNK_MAX_LEN - CHUNK_OVERLAP``. Every token is distinct so that no two
    chunks hash alike — a body of repeated filler would be deduplicated by
    ``reusable_chunk_vectors`` and the test would measure the reuse cache
    instead of the bound.
    """
    if n_chunks < 1:
        raise ValueError("n_chunks must be >= 1")
    length = CHUNK_MAX_LEN + (n_chunks - 1) * _STRIDE
    parts: list[str] = []
    total = 0
    i = 0
    while total < length:
        word = f"{tag}{i:07d} "
        parts.append(word)
        total += len(word)
        i += 1
    body = "".join(parts)[:length]
    # The fixture is not allowed to lie about its own size.
    assert len(chunk_body(body)) == n_chunks, (
        f"fixture built {len(chunk_body(body))} chunks, wanted {n_chunks}"
    )
    return body


class _StubEmbedder:
    """Returns one distinct-but-cheap vector per document. No model loads."""

    def __init__(self, *a, **kw):
        pass

    def embed_documents(self, docs):
        return np.zeros((len(docs), _DIM), dtype=np.float32)

    def embed_query(self, query):
        return np.zeros(_DIM, dtype=np.float32)


def _new_store(tmp_path) -> Store:
    store = Store(db_path=tmp_path / "cache.db")
    store.migrate()
    if not store.vector_available:  # pragma: no cover - environment guard
        store.close()
        pytest.skip("sqlite-vec unavailable; the vector arm cannot be exercised")
    return store


def _add_session(store: Store, session_id: str = "sid") -> None:
    store._c().execute(
        "INSERT INTO sessions(session_id, root_session_id, kind, origin, "
        "first_ts, last_ts, jsonl_path) VALUES (?, ?, 'session', "
        "'claude-code', '2020-01-01', '2020-01-01', '/tmp/x.jsonl')",
        (session_id, session_id),
    )
    store._c().commit()


def _add_obs(store: Store, obs_id: str, body: str, ts: str) -> None:
    store._c().execute(
        "INSERT INTO observations(obs_id, session_id, root_session_id, type, "
        "ts, body) VALUES (?, 'sid', 'sid', 'user', ?, ?)",
        (obs_id, ts, body),
    )
    store._c().commit()


def _add_record(store: Store, stable_id: str, source: str, body: str) -> None:
    store._c().execute(
        "INSERT INTO records(stable_id, source, subject, body, tags, "
        "created_at, updated_at) VALUES (?, ?, ?, ?, '[]', '2020-01-01', "
        "'2020-01-01')",
        (stable_id, source, f"subject for {stable_id}", body),
    )
    store._c().commit()


def _ns(**kw) -> argparse.Namespace:
    ns = argparse.Namespace(
        catchup=True,
        once=False,
        seed_models=False,
        source="both",
        batch_size=500,
        reindex=False,
        yes=False,
    )
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def _spy_on_batches(monkeypatch, kind_filter: str | None = None):
    """Record, per batch, the ids handed over and what they cost in chunks.

    Wraps ``_embed_batch``, which is downstream of whatever chooses the batch —
    so the numbers here are what the worker actually committed against, not
    what a packer claimed it would.
    """
    batches: list[dict] = []
    real = cli._embed_batch

    def spy(store, embedder, kind, rows, ledger, outcome, stop=None):
        if kind_filter is None or kind == kind_filter:
            id_key = "obs_id" if kind == "observations" else "stable_id"
            bodies = [
                (r["body"] or "")
                if kind == "observations"
                else f"{r['subject']}\n\n{r['body']}"
                for r in rows
            ]
            batches.append(
                {
                    "kind": kind,
                    "ids": [r[id_key] for r in rows],
                    "chunks": sum(len(chunk_body(b)) for b in bodies),
                }
            )
        return real(store, embedder, kind, rows, ledger, outcome, stop)

    monkeypatch.setattr(cli, "_embed_batch", spy)
    return batches


@pytest.fixture
def embedder(monkeypatch):
    monkeypatch.setattr(cli, "Embedder", _StubEmbedder)


# --- A. the cap holds on every batch ----------------------------------------


def test_no_batch_exceeds_the_chunk_cap_on_a_wildly_uneven_backlog(
    tmp_path, embedder, monkeypatch
):
    """THE DEFECT. A backlog mixing 1-chunk and 200-chunk rows used to arrive
    as ONE batch of 230 chunks — over an hour of encoder time with no
    checkpoint in it, and on the real corpus 6.4 hours."""
    store = _new_store(tmp_path)
    _add_session(store)
    # ``ts DESC``: thirty small rows, then the giant, then thirty more small.
    for i in range(30):
        _add_obs(store, f"late{i:02d}", _body_of(1, f"L{i}"), f"2026-03-{i + 1:02d}")
    _add_obs(store, "giant", _body_of(200, "G"), "2026-02-01")
    for i in range(30):
        _add_obs(store, f"early{i:02d}", _body_of(1, f"E{i}"), f"2026-01-{i + 1:02d}")
    batches = _spy_on_batches(monkeypatch, "observations")

    assert cli._cmd_embed(_ns(source="observations"), _store=store) == 0

    assert len(batches) > 1, (
        "the whole uneven backlog went into a single batch, so nothing is "
        "bounding it by chunks"
    )
    for n, batch in enumerate(batches):
        assert batch["chunks"] <= cli._MAX_BATCH_CHUNKS or len(batch["ids"]) == 1, (
            f"batch {n} carried {batch['chunks']} chunks over "
            f"{len(batch['ids'])} rows, above the {cli._MAX_BATCH_CHUNKS}-chunk "
            f"cap. Only a SINGLE row may exceed it — a row is indivisible at "
            f"the commit boundary. ids={batch['ids']!r}"
        )


def test_the_uneven_backlog_still_drains_completely(tmp_path, embedder):
    """Bounding the batch must not cost a row. Everything still gets embedded."""
    store = _new_store(tmp_path)
    _add_session(store)
    for i in range(30):
        _add_obs(store, f"late{i:02d}", _body_of(1, f"L{i}"), f"2026-03-{i + 1:02d}")
    _add_obs(store, "giant", _body_of(200, "G"), "2026-02-01")

    assert cli._cmd_embed(_ns(source="observations"), _store=store) == 0

    assert store.select_unembedded("observations", limit=100) == []


# --- B. an oversized single row still embeds --------------------------------


def test_a_row_bigger_than_the_cap_is_taken_alone_and_committed(
    tmp_path, embedder, monkeypatch
):
    """THE STARVATION CASE, and the reason the first row is always taken.

    ``ORDER BY ts DESC`` puts the newest row at the head of every SELECT. A
    packer that refused a row costing more than the cap would hand back an
    empty batch, forever, for this row and for everything queued behind it.
    """
    store = _new_store(tmp_path)
    _add_session(store)
    # The giant is NEWEST, so it is the head of the very first SELECT.
    _add_obs(store, "giant", _body_of(200, "G"), "2026-09-01")
    for i in range(30):
        _add_obs(store, f"small{i:02d}", _body_of(1, f"S{i}"), f"2026-01-{i + 1:02d}")
    batches = _spy_on_batches(monkeypatch, "observations")

    assert cli._cmd_embed(_ns(source="observations"), _store=store) == 0

    assert batches[0]["ids"] == ["giant"], (
        "the first batch was not the oversized row on its own — either it was "
        f"packed with rows it cannot fit beside, or it was skipped. "
        f"first batch={batches[0]['ids']!r}"
    )
    state = store._c().execute(
        "SELECT embedding_state FROM observations WHERE obs_id = 'giant'"
    ).fetchone()[0]
    assert state == "ok", f"the oversized row never left the backlog: {state!r}"
    assert store.count_vec_rows("observations") == 200 + 30


def test_an_oversized_row_does_not_starve_the_rows_behind_it(tmp_path, embedder):
    """It takes a batch of its own; the queue behind it moves on the next one."""
    store = _new_store(tmp_path)
    _add_session(store)
    _add_obs(store, "giant", _body_of(200, "G"), "2026-09-01")
    for i in range(30):
        _add_obs(store, f"small{i:02d}", _body_of(1, f"S{i}"), f"2026-01-{i + 1:02d}")

    assert cli._cmd_embed(_ns(source="observations"), _store=store) == 0

    assert store.select_unembedded("observations", limit=100) == []


# --- C. the user's priority order is untouched ------------------------------


def test_a_source_needing_many_batches_still_finishes_before_the_next(
    tmp_path, embedder, monkeypatch
):
    """The 2026-08-21 directive, under the new bound.

    Chunk-bounded batches mean a source now takes SEVERAL passes where it used
    to take one, and the failure this guards is the walk interleaving them: a
    dropbox that needs three batches must still be finished before the first
    substack row is looked at.
    """
    store = _new_store(tmp_path)
    _add_session(store)
    for i in range(4):
        _add_record(store, f"dropbox:{i}", "dropbox", _body_of(20, f"D{i}"))
    _add_record(store, "substack:0", "substack", _body_of(1, "B"))
    _add_obs(store, "o0", _body_of(1, "O"), "2026-01-01")
    batches = _spy_on_batches(monkeypatch)

    assert cli._cmd_embed(_ns(), _store=store) == 0

    dropbox_batches = [
        b for b in batches if any(i.startswith("dropbox:") for i in b["ids"])
    ]
    assert len(dropbox_batches) > 1, (
        "dropbox is 80 chunks and fitted in one batch, so this test is not "
        "exercising a multi-batch source at all"
    )
    seen: list[str] = []
    for batch in batches:
        seen.extend(batch["ids"])
    assert all(i.startswith("dropbox:") for i in seen[:4]), (
        f"a non-dropbox row was embedded before dropbox was drained: {seen!r}"
    )
    assert seen.index("substack:0") < seen.index("o0")


def test_a_batch_never_mixes_two_sources(tmp_path, embedder, monkeypatch):
    """Packing chooses rows from ONE group's SELECT, so it cannot reorder the
    walk — asserted rather than assumed, because a packer that pulled ahead to
    fill its budget would break the user's directive silently."""
    store = _new_store(tmp_path)
    _add_session(store)
    _add_record(store, "dropbox:0", "dropbox", _body_of(1, "D"))
    _add_record(store, "substack:0", "substack", _body_of(1, "B"))
    batches = _spy_on_batches(monkeypatch)

    cli._cmd_embed(_ns(), _store=store)

    for batch in batches:
        prefixes = {i.split(":")[0] for i in batch["ids"]}
        assert len(prefixes) <= 1, f"one batch spanned two sources: {batch!r}"


# --- D. termination, re-derived for the new bound ---------------------------


class _NeverCommits(Store):
    """Every compare-and-swap fails, so no batch moves a row. Round 2's S4.

    The stall bound's argument is that a zero-write batch re-selects THE SAME
    batch, so more attempts inside one run cannot help. Chunk bounding changes
    what "the same batch" means — a prefix of the SELECT rather than all of it
    — so the argument is re-checked here rather than assumed to carry over. It
    does carry: packing is a pure function of the rows the SELECT returned and
    a constant cap, and nothing moved, so the same rows come back and the same
    prefix is taken.
    """

    _RUNAWAY = 40

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.selects = 0

    def select_unembedded(self, kind, limit=500, model=None, source=None):
        self.selects += 1
        if self.selects > self._RUNAWAY:
            raise AssertionError(
                f"--catchup re-selected the backlog {self.selects} times "
                f"without writing a row: the loop has no termination argument"
            )
        return super().select_unembedded(
            kind, limit=limit, model=model, source=source
        )

    def commit_embed_batch(self, kind, **kw):
        return [], []


def test_a_stalled_catchup_still_terminates_with_uneven_rows(
    tmp_path, embedder, capsys
):
    store = _new_store(tmp_path)
    _add_session(store)
    _add_obs(store, "giant", _body_of(200, "G"), "2026-09-01")
    for i in range(10):
        _add_obs(store, f"small{i:02d}", _body_of(1, f"S{i}"), f"2026-01-{i + 1:02d}")
    store.close()
    stalling = _NeverCommits(db_path=tmp_path / "cache.db")

    rc = cli._cmd_embed(_ns(source="observations"), _store=stalling)
    said = capsys.readouterr()

    assert rc == 0
    assert stalling.selects < _NeverCommits._RUNAWAY
    assert "STOPPED WITHOUT PROGRESS" in said.out + said.err


def test_the_packer_never_hands_back_an_empty_batch(tmp_path, embedder):
    """The termination argument depends on it. An empty batch writes nothing,
    which reads as a stall, and the stall bound then ends a run that had a full
    backlog in front of it."""
    store = _new_store(tmp_path)
    _add_session(store)
    _add_obs(store, "giant", _body_of(200, "G"), "2026-09-01")
    rows = store.select_unembedded("observations", limit=500)

    assert cli._pack_batch("observations", rows, chunk_cap=1) == rows[:1]


# --- rows that cost the encoder nothing -------------------------------------


def test_rows_with_nothing_to_embed_do_not_consume_the_chunk_budget(
    tmp_path, embedder, monkeypatch
):
    """~35% of the corpus has an empty body (``docs/embedding-throughput.md``).
    Those rows are marked ``'skip'`` and cost the model nothing, so a chunk cap
    must not throttle them — which is exactly what ``--batch-size`` is still
    here to bound."""
    store = _new_store(tmp_path)
    _add_session(store)
    for i in range(120):
        _add_obs(store, f"empty{i:03d}", "", f"2026-01-01T00:{i:02d}:00")
    batches = _spy_on_batches(monkeypatch, "observations")

    assert cli._cmd_embed(_ns(source="observations"), _store=store) == 0

    assert batches[0]["chunks"] == 0
    assert len(batches[0]["ids"]) == 120, (
        "rows with no chunks were rationed by the chunk cap, which bounds "
        f"encoder time they do not spend: first batch={len(batches[0]['ids'])}"
    )


def test_the_cost_of_a_record_is_measured_on_the_text_that_gets_embedded(
    tmp_path, embedder
):
    """Records are embedded as ``subject + "\\n\\n" + body``, not body alone.

    If the packer counted a different string than ``_embed_batch`` encodes, the
    cap would be measured against text nobody embeds — right for observations,
    quietly wrong for every record. Both read the body through the same
    helper, and this is what pins that.
    """
    store = _new_store(tmp_path)
    _add_record(store, "dropbox:0", "dropbox", _body_of(3, "D"))
    row = store.select_unembedded("records", limit=1)[0]

    row_id, text = cli._embed_row_text("records", row)

    assert row_id == "dropbox:0"
    assert text == f"{row['subject']}\n\n{row['body']}"
    assert cli._row_chunk_cost("records", row) == len(chunk_body(text))


# --- E. the operator can see the bound --------------------------------------


def _embed_help(flag: str) -> str:
    parser = cli.build_parser()
    embed = parser._subparsers._group_actions[0].choices["embed"]
    for action in embed._actions:
        if flag in (action.option_strings or []):
            return action.help or ""
    raise AssertionError(f"embed has no {flag} flag")


def test_the_batch_size_help_says_it_is_no_longer_the_only_bound():
    """A flag whose meaning narrowed has to say so where the operator looks.

    ``--batch-size`` still means rows per batch — it was not redefined — and it
    is no longer the ONLY bound: the chunk cap is a ceiling the flag cannot
    raise. It can still lower the interval, though, whenever a row count binds
    before the cap does, so "tuning it does nothing" would be the opposite
    error and is not what this asserts.
    """
    help_text = _embed_help("--batch-size")

    assert str(cli._MAX_BATCH_CHUNKS) in help_text, help_text
    assert "chunk" in help_text, help_text


def test_the_help_states_the_cap_in_wall_clock_terms():
    """"45 chunks" means nothing to an operator; "about 15 minutes" means
    everything. The rate is the measured one from
    ``docs/embedding-throughput.md``."""
    help_text = _embed_help("--batch-size")

    assert "minutes" in help_text, help_text
    assert "20 s" in help_text or "20 seconds" in help_text, help_text


def test_the_cap_is_defensible_at_the_measured_rate():
    """The constant and the sentence describing it must not drift apart."""
    seconds = cli._MAX_BATCH_CHUNKS * cli._SECONDS_PER_CHUNK
    assert 10 * 60 <= seconds <= 20 * 60, (
        f"the cap is {seconds / 60:.0f} minutes of encoder time; --help "
        f"promises about 15"
    )
