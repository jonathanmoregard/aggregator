"""Per-record failure isolation in the embed worker.

THE DEFECT THIS FILE PINS. ``_embed_batch`` used to call ``embed_documents``
outside any handler, so one row whose body killed the tokenizer aborted the
whole ``aggregator embed --catchup`` run. ``select_unembedded`` orders
``ts DESC``, so the same row was re-selected on the next 30-minute tick and
re-failed — no forward progress, a traceback twice an hour, and nothing
anywhere naming the row responsible. Exactly the 2026-08-15 ingest doom loop,
one layer down.

THE TWO AXES, because "transient or permanent?" cannot be answered from an
exception type. A cold model, an OOM and a body that deterministically kills
the tokenizer all arrive as ``RuntimeError``.

* **Does the failure discriminate between rows?** — answered by embedding a
  known-good probe string the moment a row fails. The probe embeds -> the
  embedder works and the fault belongs to that row. The probe fails too ->
  the environment is broken, so nothing is attributed, nothing is marked, and
  the backlog is left exactly where it was.
* **Does it reproduce?** — answered by the existing ``quarantine`` ledger. A
  row-attributed failure is HELD with backoff, not condemned; only a row that
  fails ``POISON_MAX_ATTEMPTS`` times across separate runs goes terminal, and
  a row that succeeds on retry leaves the ledger entirely.

Neither axis alone is enough. Axis 1 alone condemns a row on the single blip
that happened to hit it; axis 2 alone condemns the entire 372k-row corpus
three ticks after somebody's model failed to load.
"""

import argparse
import json

import numpy as np
import pytest

from aggregator.core.store import Store
from aggregator.imports.ingest_state import POISON_MAX_ATTEMPTS

POISON = "ZZPOISONZZ"


@pytest.fixture
def cache(tmp_path):
    """Four observations, newest first: three good and one poisonous."""
    db = tmp_path / "cache.db"
    s = Store(db_path=db)
    s.migrate()
    c = s._c()
    c.execute(
        "INSERT INTO sessions(session_id, root_session_id, kind, first_ts, "
        "last_ts, jsonl_path) VALUES ('sid', 'sid', 'session', "
        "'2026-01-01', '2026-01-01', '/tmp/x.jsonl')"
    )
    bodies = {
        "good-a": "harmless body a",
        "good-b": "harmless body b",
        "bad": f"a body that {POISON} the tokenizer",
        "good-c": "harmless body c",
    }
    for i, (obs_id, body) in enumerate(bodies.items()):
        c.execute(
            "INSERT INTO observations(obs_id, session_id, root_session_id, "
            "type, ts, body) VALUES (?, 'sid', 'sid', 'user', ?, ?)",
            (obs_id, f"2026-01-0{i + 1}", body),
        )
    c.commit()
    s.close()
    return db


class PoisonEmbedder:
    """Healthy except on bodies carrying the marker. Counts calls per body.

    ``dead=True`` makes it fail on EVERYTHING, including the health probe —
    the shape of a model that has not loaded, an OOM, or an I/O blip.
    """

    def __init__(self, *, dead: bool = False, poison: str = POISON):
        self.dead = dead
        self.poison = poison
        self.documents: list[str] = []
        self.calls = 0

    def embed_documents(self, docs):
        self.calls += 1
        if self.dead:
            raise RuntimeError("model is not loaded")
        if any(self.poison in d for d in docs):
            raise RuntimeError("tokenizer died on this body")
        self.documents.extend(docs)
        return np.array([[float(i)] * 768 for i in range(len(docs))], dtype=np.float32)

    def embed_query(self, q):
        return np.zeros(768, dtype=np.float32)


def ns(**kw):
    n = argparse.Namespace(
        catchup=True,
        once=False,
        source="observations",
        batch_size=500,
        reindex=False,
        yes=False,
    )
    for k, v in kw.items():
        setattr(n, k, v)
    return n


def run_embed(cache, embedder, monkeypatch, **kw) -> int:
    from aggregator.cli import _cmd_embed

    monkeypatch.setattr("aggregator.cli.Embedder", lambda *a, **k: embedder)
    store = Store(db_path=cache)
    try:
        return _cmd_embed(ns(**kw), _store=store)
    finally:
        store.close()


def states(cache) -> dict[str, str | None]:
    s = Store(db_path=cache)
    try:
        return {
            r["obs_id"]: r["embedding_state"]
            for r in s._c().execute("SELECT obs_id, embedding_state FROM observations")
        }
    finally:
        s.close()


def quarantine(cache) -> list[dict]:
    s = Store(db_path=cache)
    try:
        return [dict(r) for r in s._c().execute("SELECT * FROM quarantine")]
    finally:
        s.close()


def _set_retry(cache, when: str) -> int:
    """Move every non-terminal held row's retry time. Returns how many moved.

    Drives the clock by editing the STORED decision rather than by patching
    ``datetime``, because the stored decision is what the worker actually
    honours — the same reason ``ingest_state`` writes ``next_attempt_at`` down
    instead of re-deriving a jittered delay on every read. Setting it
    explicitly also keeps the "not due yet" tests off the jitter: a real hold
    rolls ``uniform(0, 30min)``, which is *almost* always longer than a test,
    and a test that fails one run in two thousand is worse than no test.
    """
    s = Store(db_path=cache)
    try:
        cur = s._c().execute(
            "UPDATE quarantine SET next_retry_at = ? WHERE next_retry_at IS NOT NULL",
            (when,),
        )
        s._c().commit()
        return cur.rowcount
    finally:
        s.close()


def make_retry_due(cache) -> int:
    return _set_retry(cache, "2000-01-01T00:00:00+00:00")


def defer_retry(cache) -> int:
    return _set_retry(cache, "2999-01-01T00:00:00+00:00")


# --- one poison row among many ----------------------------------------------


def test_one_poison_row_costs_one_row_and_not_the_run(cache, monkeypatch):
    """The backlog drains PAST the bad row. That is the whole fix."""
    embedder = PoisonEmbedder()
    rc = run_embed(cache, embedder, monkeypatch)

    assert rc != 0, "a new failure must be loud"
    assert states(cache) == {
        "good-a": "ok",
        "good-b": "ok",
        "good-c": "ok",
        "bad": "error",
    }


def test_the_good_rows_really_reached_the_vector_index(cache, monkeypatch):
    """Marked ``'ok'`` has to mean a vector exists, poison row or not."""
    run_embed(cache, PoisonEmbedder(), monkeypatch)

    s = Store(db_path=cache)
    try:
        vecs = {r["obs_id"] for r in s._c().execute("SELECT obs_id FROM vec_observations")}
    finally:
        s.close()
    assert vecs == {"good-a", "good-b", "good-c"}


def test_the_failing_row_is_named_on_stderr_the_first_time(cache, monkeypatch, capsys):
    """Loud once means the operator learns WHICH row and WHY, not that a run died."""
    run_embed(cache, PoisonEmbedder(), monkeypatch)

    err = capsys.readouterr().err
    assert "bad" in err
    assert "tokenizer died on this body" in err


def test_the_failing_row_is_recorded_in_the_ledger(cache, monkeypatch):
    run_embed(cache, PoisonEmbedder(), monkeypatch)

    rows = quarantine(cache)
    assert len(rows) == 1
    held = rows[0]
    assert held["source"] == "embed:observations"
    assert held["record_key"] == "bad"
    assert held["error_type"] == "RuntimeError"
    assert "tokenizer died" in held["error_detail"]
    assert held["attempts"] == 1
    assert held["next_retry_at"] is not None, "attempt 1 must still be retryable"


# --- the same poison row, run after run --------------------------------------


def test_a_known_poison_row_goes_quiet_and_stops_failing_the_run(cache, monkeypatch, capsys):
    """The known-poison-ledger bargain: reported loudly ONCE, then noted.

    A traceback twice an hour about a file that has been broken since March is
    how an operator learns to ignore the notifier, which costs the next real
    failure its audience.
    """
    assert run_embed(cache, PoisonEmbedder(), monkeypatch) != 0
    capsys.readouterr()
    assert make_retry_due(cache) == 1

    rc = run_embed(cache, PoisonEmbedder(), monkeypatch)

    assert rc == 0, "a failure already reported must not fail the run again"
    assert capsys.readouterr().err == "", "and must not be re-reported on stderr"


def test_a_known_poison_row_is_still_counted_out_loud_somewhere(cache, monkeypatch, capsys):
    """Quiet is not the same as forgotten: the run still says how many."""
    run_embed(cache, PoisonEmbedder(), monkeypatch)
    capsys.readouterr()
    make_retry_due(cache)

    run_embed(cache, PoisonEmbedder(), monkeypatch)

    out = capsys.readouterr().out
    assert "1 known-bad row(s)" in out
    assert "aggregator status" in out, "quiet must still say where to look"


def test_a_poison_row_is_not_retried_forever(cache, monkeypatch):
    """Bounded attempts. An unbounded retry is an alert that never stops."""
    for _ in range(POISON_MAX_ATTEMPTS):
        run_embed(cache, PoisonEmbedder(), monkeypatch)
        make_retry_due(cache)

    held = quarantine(cache)[0]
    assert held["attempts"] == POISON_MAX_ATTEMPTS
    assert held["next_retry_at"] is None, "must be terminal by now"

    # ...and terminal means the model is never handed that body again.
    final = PoisonEmbedder()
    assert run_embed(cache, final, monkeypatch) == 0
    assert final.calls == 0


def test_a_terminal_row_is_never_requeued_into_the_backlog(cache, monkeypatch):
    for _ in range(POISON_MAX_ATTEMPTS):
        run_embed(cache, PoisonEmbedder(), monkeypatch)
        make_retry_due(cache)
    assert make_retry_due(cache) == 0  # nothing left that is retryable

    run_embed(cache, PoisonEmbedder(), monkeypatch)

    assert states(cache)["bad"] == "error"


# --- a transient failure is not a permanent one ------------------------------


def test_an_embedder_that_cannot_embed_anything_marks_nothing(cache, monkeypatch):
    """A cold model, an OOM or an I/O blip is not bad data.

    The health probe fails alongside the row, so the failure does not
    discriminate between rows and nothing may be attributed to one. The backlog
    is left exactly where it was — the same call the worker already makes when
    sqlite-vec is missing.
    """
    rc = run_embed(cache, PoisonEmbedder(dead=True), monkeypatch)

    assert rc != 0
    assert set(states(cache).values()) == {None}, "the backlog must be untouched"
    assert quarantine(cache) == [], "nothing may be blamed on a row"


def test_the_dead_embedder_says_so_on_stderr(cache, monkeypatch, capsys):
    run_embed(cache, PoisonEmbedder(dead=True), monkeypatch)

    err = capsys.readouterr().err
    assert "model is not loaded" in err
    assert "backlog" in err


def test_a_transient_failure_is_fully_recovered_by_the_next_run(cache, monkeypatch):
    """Nothing was marked, so the next healthy run embeds everything."""
    assert run_embed(cache, PoisonEmbedder(dead=True), monkeypatch) != 0

    rc = run_embed(cache, PoisonEmbedder(poison="never-appears"), monkeypatch)

    assert rc == 0
    assert set(states(cache).values()) == {"ok"}


class PoisonThenDeadEmbedder(PoisonEmbedder):
    """Poisons one row while healthy, then stops working entirely.

    The compound case both axes have to survive at once: a row correctly
    attributed while the embedder still worked, followed by the embedder
    dying. ``die_after`` counts the calls that behave normally, the health
    probe among them.
    """

    def __init__(self, die_after: int):
        super().__init__()
        self.die_after = die_after

    def embed_documents(self, docs):
        if self.calls >= self.die_after:
            self.calls += 1
            raise RuntimeError("model is not loaded")
        return super().embed_documents(docs)


def test_a_row_blamed_before_the_model_died_is_still_announced(cache, monkeypatch, capsys):
    """An abort must not swallow an attribution it has already committed.

    The ledger entry for that row is written the moment it is blamed, so a run
    that recorded it and then returned early on the environment fault would
    leave it known-but-never-reported: quiet on this run, and quiet on every
    run after it, because the next run sees it as already-known. That is
    precisely the silence the ledger's report-once bargain is paid for.
    """
    rc = run_embed(cache, PoisonThenDeadEmbedder(die_after=3), monkeypatch)

    assert rc != 0
    err = capsys.readouterr().err
    assert "tokenizer died on this body" in err, "the blamed row went unannounced"
    assert "model is not loaded" in err, "and the abort must be reported too"
    assert [r["record_key"] for r in quarantine(cache)] == ["bad"]


# --- a poison row that stops being poisonous ---------------------------------


def test_a_row_that_stops_failing_leaves_the_ledger(cache, monkeypatch):
    """A fault that no longer reproduces describes no gap at all.

    Same rule as ``reconcile_faults``' released set: leaving the row in the
    ledger would have ``aggregator status`` overstating the damage forever.
    """
    assert run_embed(cache, PoisonEmbedder(), monkeypatch) != 0
    make_retry_due(cache)

    rc = run_embed(cache, PoisonEmbedder(poison="never-appears"), monkeypatch)

    assert rc == 0
    assert states(cache)["bad"] == "ok"
    assert quarantine(cache) == []


def test_the_recovered_row_is_actually_in_the_vector_index(cache, monkeypatch):
    """Released has to mean retrievable, not merely un-flagged."""
    run_embed(cache, PoisonEmbedder(), monkeypatch)
    make_retry_due(cache)
    run_embed(cache, PoisonEmbedder(poison="never-appears"), monkeypatch)

    s = Store(db_path=cache)
    try:
        vecs = {r["obs_id"] for r in s._c().execute("SELECT obs_id FROM vec_observations")}
    finally:
        s.close()
    assert "bad" in vecs


def test_a_held_row_is_not_retried_before_its_time(cache, monkeypatch):
    """Backoff is a real delay, not a formality — otherwise a deterministic
    row burns all three of its attempts inside one ``--catchup``."""
    run_embed(cache, PoisonEmbedder(), monkeypatch)
    assert defer_retry(cache) == 1

    second = PoisonEmbedder(poison="never-appears")
    assert run_embed(cache, second, monkeypatch) == 0
    assert second.calls == 0, "the row was retried before its backoff expired"
    assert states(cache)["bad"] == "error"


# --- an 'error' row must stay visible ----------------------------------------


def test_status_names_the_rows_the_embed_worker_set_aside(cache, monkeypatch, capsys):
    """``aggregator status`` is where the existing ledger surfaces. Follow it."""
    from aggregator.cli import _cmd_status

    run_embed(cache, PoisonEmbedder(), monkeypatch)
    capsys.readouterr()

    store = Store(db_path=cache)
    try:
        assert _cmd_status(argparse.Namespace(json=False), store) == 0
    finally:
        store.close()
    out = capsys.readouterr().out
    assert "embed:observations" in out
    assert "RuntimeError" in out


def test_status_json_carries_the_held_embed_rows(cache, monkeypatch, capsys):
    from aggregator.cli import _cmd_status

    run_embed(cache, PoisonEmbedder(), monkeypatch)
    capsys.readouterr()

    store = Store(db_path=cache)
    try:
        _cmd_status(argparse.Namespace(json=True), store)
    finally:
        store.close()
    payload = json.loads(capsys.readouterr().out)
    assert any(
        e["source"] == "embed:observations" for e in payload["held_records"]
    )


def test_a_backlog_with_errors_in_it_is_not_reported_complete(cache, monkeypatch):
    """``pending == 0`` is not "everything is reachable" once rows can error.

    A cache whose only unembedded rows are errors would otherwise report
    ``state='complete'`` — the index rotting while every count says it is fine,
    which is the one failure mode this project exists to prevent.
    """
    run_embed(cache, PoisonEmbedder(), monkeypatch)

    s = Store(db_path=cache)
    try:
        vi = s.capabilities()["vector_index"]
    finally:
        s.close()
    assert vi["observations"]["error"] == 1
    assert vi["observations"]["pending"] == 0
    assert vi["state"] == "degraded"


def test_a_clean_backlog_is_still_reported_complete(cache, monkeypatch):
    """The new state must not swallow the old one."""
    run_embed(cache, PoisonEmbedder(poison="never-appears"), monkeypatch)

    s = Store(db_path=cache)
    try:
        assert s.capabilities()["vector_index"]["state"] == "complete"
    finally:
        s.close()


def test_the_timer_invocation_isolates_and_retries_too(cache, monkeypatch):
    """``--once`` is what the 30-minute timer runs, so it is where the defect lived.

    Every other test here drives ``--catchup``. They share ``_embed_batch``, but
    "shared today" is not a contract, and the entry point that actually runs
    unattended is the one whose isolation has to be pinned.
    """
    once = {"catchup": False, "once": True, "batch_size": 10}
    assert run_embed(cache, PoisonEmbedder(), monkeypatch, **once) != 0
    assert states(cache)["bad"] == "error"
    assert states(cache)["good-a"] == "ok", "one batch, and it drained past the bad row"
    assert make_retry_due(cache) == 1

    healed = PoisonEmbedder(poison="never-appears")
    assert run_embed(cache, healed, monkeypatch, **once) == 0
    assert states(cache)["bad"] == "ok"


# --- records, not just observations ------------------------------------------


def test_the_records_ontology_isolates_failures_too(cache, monkeypatch):
    """Both ontologies run through the same worker; both must isolate."""
    s = Store(db_path=cache)
    for stable_id, body in (("github:1", "fine"), ("github:2", f"{POISON} here")):
        s._c().execute(
            "INSERT INTO records(stable_id, source, subject, body, tags, "
            "created_at, updated_at) VALUES (?, 'github', 'subj', ?, '[]', "
            "'2026-01-01', '2026-01-01')",
            (stable_id, body),
        )
    s.commit()
    s.close()

    rc = run_embed(cache, PoisonEmbedder(), monkeypatch, source="records")

    assert rc != 0
    s2 = Store(db_path=cache)
    try:
        got = {
            r["stable_id"]: r["embedding_state"]
            for r in s2._c().execute("SELECT stable_id, embedding_state FROM records")
        }
    finally:
        s2.close()
    assert got == {"github:1": "ok", "github:2": "error"}
    assert quarantine(cache)[0]["source"] == "embed:records"


def test_the_two_ontologies_do_not_share_a_ledger_key(cache, monkeypatch):
    """An obs_id and a stable_id could collide; the ledger source keeps them apart."""
    s = Store(db_path=cache)
    s._c().execute(
        "INSERT INTO records(stable_id, source, subject, body, tags, "
        "created_at, updated_at) VALUES ('bad', 'github', 'subj', ?, '[]', "
        "'2026-01-01', '2026-01-01')",
        (f"{POISON} here",),
    )
    s.commit()
    s.close()

    run_embed(cache, PoisonEmbedder(), monkeypatch, source="both")

    assert {r["source"] for r in quarantine(cache)} == {
        "embed:observations",
        "embed:records",
    }
