"""The embed worker fills the vector arm in the order the user chose.

USER DIRECTIVE, 2026-08-21: ``dropbox -> blog -> llm -> claude code``, then
everything unranked. The store half is in
``tests/core/test_store_embed_priority.py``; this file is about the worker
actually walking it, and about the operator being able to see how far it got.

WHY THE WORKER HAD TO CHANGE SHAPE. It drained by ONTOLOGY — all observations,
then all records — and the user's order cuts across that: two records sources
come before every observation and four more come after. No ordering inside
``select_unembedded`` could have produced it, because the two ontologies are
different queries against different tables. So the loop is now over the
priority plan, and the ontology is a property of each step rather than the
thing being iterated.

WHAT A HALF-EMBEDDED CORPUS HAS TO LOOK LIKE, which is the other half of the
directive: a source nobody has reached yet returns zero vector hits, and so
does a source that is finished and simply has nothing on the topic. Those are
opposite answers to "can I search my notes yet", and ``aggregator status`` has
to be able to tell them apart per source.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

import numpy as np
import pytest

from aggregator import cli
from aggregator.core.store import EMBED_BACKLOG_ORDER, EMBED_REST, Store
from aggregator.sources.base import ObservationRow, Record, SessionRow

_DIM = 768


class _StubEmbedder:
    """Never names a real model, never loads one."""

    def __init__(self, *a, **kw):
        pass

    def embed_documents(self, docs):
        return np.zeros((len(docs), _DIM), dtype=np.float32)

    def embed_query(self, q):
        return np.zeros(_DIM, dtype=np.float32)


@pytest.fixture(autouse=True)
def _stub_embedder(monkeypatch):
    monkeypatch.setattr("aggregator.core.embed.Embedder", _StubEmbedder)
    monkeypatch.setattr("aggregator.cli.Embedder", _StubEmbedder)


def _session(session_id, kind="session", origin="claude-code"):
    ts = datetime(2026, 7, 1, 8, 0, tzinfo=UTC)
    return SessionRow(
        session_id=session_id,
        root_session_id=session_id,
        parent_session_id=None,
        kind=kind,
        agent_id=None,
        agent_type=None,
        spawned_by_tool_use_id=None,
        cwd="/x",
        git_branch="main",
        first_ts=ts,
        last_ts=ts,
        jsonl_path=f"/tmp/{session_id}.jsonl",
        origin=origin,
    )


def _obs(obs_id, session_id):
    return ObservationRow(
        obs_id=obs_id,
        session_id=session_id,
        root_session_id=session_id,
        parent_obs_id=None,
        type="user",
        ts=datetime(2026, 7, 1, 8, 0, tzinfo=UTC),
        model=None,
        input_tokens=None,
        output_tokens=None,
        tool_name=None,
        tool_use_id=None,
        body=f"body of {obs_id}, long enough to chunk",
    )


def _record(stable_id, source):
    ts = datetime(2026, 7, 1, tzinfo=UTC)
    return Record(
        stable_id=stable_id,
        source=source,
        subject=f"subject {stable_id}",
        body=f"body of {stable_id}",
        tags=[],
        created_at=ts,
        updated_at=ts,
    )


@pytest.fixture
def cache(tmp_path):
    db = tmp_path / "cache.db"
    s = Store(db_path=db)
    s.migrate()
    s.upsert_entities(
        [
            _session("s-code"),
            _session("s-sub", kind="subagent"),
            _session("s-web", origin="claude-web"),
            _obs("o-code", "s-code"),
            _obs("o-sub", "s-sub"),
            _obs("o-web", "s-web"),
        ]
    )
    s.upsert(
        [
            _record("dropbox:a", "dropbox"),
            _record("substack:a", "substack"),
            _record("github:a", "github"),
        ]
    )
    s.close()
    return db


def _ns(**kw):
    ns = argparse.Namespace(
        catchup=True,
        once=False,
        source="both",
        batch_size=500,
        reindex=False,
        yes=False,
    )
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def _spy_on_selection(monkeypatch, store) -> list[tuple[str, str | None]]:
    """Record the (kind, source) of every backlog query the run makes."""
    seen: list[tuple[str, str | None]] = []
    real = store.select_unembedded

    def spy(kind, limit=500, model=None, source=None):
        seen.append((kind, source))
        return real(kind, limit=limit, model=model, source=source)

    monkeypatch.setattr(store, "select_unembedded", spy)
    return seen


# --- the order the worker walks ---------------------------------------------


def test_the_worker_walks_the_users_priority_order(cache, monkeypatch):
    """THE REPRO. The worker drained ontology by ontology, so every records
    source came after every observation -- dropbox, the source the user put
    FIRST, was last in line behind 505k observations, i.e. weeks away."""
    store = Store(db_path=cache)
    seen = _spy_on_selection(monkeypatch, store)

    assert cli._cmd_embed(_ns(), _store=store) == 0

    # Each group may be queried more than once (drain until empty), so compare
    # the sequence of DISTINCT groups in first-seen order. ``source=None`` is
    # excluded here and asserted on its own below: that is the completion
    # check, which asks a different question.
    walked: list[tuple[str, str | None]] = []
    for pair in seen:
        if pair[1] is not None and pair not in walked:
            walked.append(pair)
    assert walked == list(EMBED_BACKLOG_ORDER)


def test_the_completion_check_still_asks_about_the_whole_backlog(
    cache, monkeypatch
):
    """``completed_at`` publishes the index, and it may only flip when EVERY
    group is drained. Asking it per source would publish a half-built index the
    moment the first source finished -- which on this ordering is dropbox, 1500
    rows into a 483k-row corpus."""
    store = Store(db_path=cache)
    seen = _spy_on_selection(monkeypatch, store)

    cli._cmd_embed(_ns(), _store=store)

    assert any(source is None for _, source in seen), (
        "nothing asked for the unscoped backlog, so 'is the whole index "
        "built?' was answered from a per-source question"
    )


def test_dropbox_is_embedded_before_any_session_observation(cache, monkeypatch):
    """The behavioural version of the same claim, and the one that matters:
    over a 25-30 day backfill, ORDER is when each source becomes searchable."""
    store = Store(db_path=cache)
    order: list[str] = []
    real = cli._embed_batch

    def spy(store_, embedder, kind, rows, ledger, outcome, stop=None):
        order.extend(
            r["stable_id"] if kind == "records" else r["obs_id"] for r in rows
        )
        return real(store_, embedder, kind, rows, ledger, outcome, stop)

    monkeypatch.setattr(cli, "_embed_batch", spy)
    cli._cmd_embed(_ns(), _store=store)

    assert order.index("dropbox:a") < order.index("o-code")
    assert order.index("substack:a") < order.index("o-code")
    assert order.index("o-web") < order.index("o-code")
    assert order.index("o-code") < order.index("o-sub")
    assert order.index("o-sub") < order.index("github:a")


def test_a_source_scope_keeps_the_order_within_that_ontology(cache, monkeypatch):
    """``--source records`` is still a priority walk, just a shorter one."""
    store = Store(db_path=cache)
    seen = _spy_on_selection(monkeypatch, store)

    cli._cmd_embed(_ns(source="records"), _store=store)

    walked = [p for i, p in enumerate(seen) if p not in seen[:i]]
    assert walked == [g for g in EMBED_BACKLOG_ORDER if g[0] == "records"]


def test_once_still_means_one_batch_not_one_per_source(cache, monkeypatch):
    """``--once`` is a hand-run probe. Eight groups must not turn it into eight
    batches -- the operator asked for one unit of work and one is what a
    time-boxed probe can afford."""
    store = Store(db_path=cache)
    batches: list[int] = []
    real = cli._embed_batch

    def spy(store_, embedder, kind, rows, ledger, outcome, stop=None):
        batches.append(len(rows))
        return real(store_, embedder, kind, rows, ledger, outcome, stop)

    monkeypatch.setattr(cli, "_embed_batch", spy)
    cli._cmd_embed(_ns(catchup=False, once=True, batch_size=1), _store=store)

    assert len(batches) == 1


def test_once_skips_past_sources_that_are_already_done(cache, monkeypatch):
    """An empty group is not a unit of work. If --once stopped at the first
    group regardless, a finished dropbox would starve every source behind it
    forever -- the priority order turned into a deadlock."""
    store = Store(db_path=cache)
    store.upsert_vec_records(
        [
            ("dropbox:a", np.zeros(_DIM, dtype=np.float32)),
            ("substack:a", np.zeros(_DIM, dtype=np.float32)),
        ]
    )
    store.mark_embedded(
        "records",
        ["dropbox:a", "substack:a"],
        "ok",
        expected={"dropbox:a": None, "substack:a": None},
    )
    seen: list[str] = []
    real = cli._embed_batch

    def spy(store_, embedder, kind, rows, ledger, outcome, stop=None):
        seen.extend(
            r["stable_id"] if kind == "records" else r["obs_id"] for r in rows
        )
        return real(store_, embedder, kind, rows, ledger, outcome, stop)

    monkeypatch.setattr(cli, "_embed_batch", spy)
    cli._cmd_embed(_ns(catchup=False, once=True, batch_size=1), _store=store)

    assert seen == ["o-web"]


# --- what the operator can see ----------------------------------------------


def test_the_run_reports_progress_per_source(cache, monkeypatch, capsys):
    """The question a user asks of a multi-week backfill is 'which sources are
    done', and no global percentage answers it."""
    store = Store(db_path=cache)
    cli._cmd_embed(_ns(), _store=store)
    out = capsys.readouterr().out + capsys.readouterr().err

    assert "dropbox" in out and "substack" in out and "sessions" in out


def test_status_names_which_sources_are_fully_embedded(cache, capsys):
    """THE LOOKALIKE. A source nobody has reached and a source that is
    finished both answer every query with zero vector hits."""
    store = Store(db_path=cache)
    store.migrate()
    store.upsert_vec_records([("dropbox:a", np.zeros(_DIM, dtype=np.float32))])
    store.mark_embedded(
        "records", ["dropbox:a"], "ok", expected={"dropbox:a": None}
    )

    cli.main(["status"], _store=store)
    out = " ".join(capsys.readouterr().out.split())

    assert "embedding progress" in out.lower()
    assert "dropbox: complete" in out
    assert "substack: not_started" in out


def test_status_json_carries_the_same_per_source_progress(cache, capsys):
    store = Store(db_path=cache)
    store.migrate()

    cli.main(["status", "--json"], _store=store)

    import json

    payload = json.loads(capsys.readouterr().out)
    rows = payload["embedding_progress"]
    assert [(r["kind"], r["source"]) for r in rows] == [
        list(pair) and (pair[0], pair[1]) for pair in EMBED_BACKLOG_ORDER
    ]
    by_group = {(r["kind"], r["source"]): r for r in rows}
    assert by_group[("records", "dropbox")]["state"] == "not_started"
    assert by_group[("observations", EMBED_REST)]["state"] == "empty"
