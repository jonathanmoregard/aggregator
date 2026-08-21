"""The last window: between ``select_unembedded`` and the fingerprint re-read.

``tests/test_cli_embed_toctou.py`` closed the WIDE window — an edit landing
while the embedder is busy. It did so by re-reading each row's ``src_hash``
immediately before the writes and comparing it with a snapshot taken at the
top of the batch.

That snapshot is one statement later than the SELECT that handed over the
body, and the gap is enough. Interleave an ingest write there and BOTH reads
see the new hash, so the row compares equal — "unmoved" — while the body the
worker holds is the old one. Reproduced end to end: the old body's vector
stored as current with the row marked ``'ok'``, which ``select_unembedded``
(``WHERE embedding_state IS NULL``) never returns again. The vector arm then
answers with text the row no longer contains, forever, while every count
reports the index complete.

The close is to stop comparing snapshots at all: ``select_unembedded`` returns
``src_hash`` alongside the body, so the "before" value comes from the same
statement as the text, and every write carries ``AND src_hash IS ?`` in the
statement that does the writing. There is then no interval to land in —
``Store.commit_embed_batch``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

import aggregator.cli as cli
from aggregator.core.store import Store
from aggregator.sources.base import ObservationRow, SessionRow

_OLD = "ORIGINAL BODY the worker was handed by the SELECT"
_NEW = "EDITED BODY ingest wrote before the worker looked again"
_OTHER = "an ordinary body nobody touches"
_MARKERS = {_OLD: 1.0, _NEW: 2.0, _OTHER: 3.0}


def _session():
    ts = datetime(2026, 7, 1, tzinfo=UTC)
    return SessionRow(
        session_id="sid",
        root_session_id="sid",
        parent_session_id=None,
        kind="session",
        agent_id=None,
        agent_type=None,
        spawned_by_tool_use_id=None,
        cwd=None,
        git_branch=None,
        first_ts=ts,
        last_ts=ts,
        jsonl_path="/tmp/x.jsonl",
    )


def _obs(obs_id: str, body: str, day: int):
    return ObservationRow(
        obs_id=obs_id,
        session_id="sid",
        root_session_id="sid",
        parent_obs_id=None,
        type="user",
        ts=datetime(2026, 7, day, tzinfo=UTC),
        model=None,
        input_tokens=None,
        output_tokens=None,
        tool_name=None,
        tool_use_id=None,
        body=body,
    )


def _vector_for(doc: str) -> np.ndarray:
    """A vector that says which body produced it, so the two are tellable
    apart. A zero vector would make "old text indexed" and "new text indexed"
    look identical, and that is the entire question."""
    vec = np.zeros(768, dtype=np.float32)
    for body, marker in _MARKERS.items():
        if body in doc:
            vec[0] = marker
    return vec


class _Stub:
    def __init__(self, *a, **kw):
        pass

    def embed_documents(self, docs):
        return np.array([_vector_for(d) for d in docs], dtype=np.float32)

    def embed_query(self, query):
        return np.zeros(768, dtype=np.float32)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    db = tmp_path / "aggregator" / "cache.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    s = Store(db_path=db)
    s.migrate()
    if not s.vector_available:  # pragma: no cover - environment guard
        pytest.skip("sqlite-vec unavailable; the vector arm cannot be exercised")
    # ``ts DESC`` hands the worker "target" first.
    s.upsert_entities(
        [_session(), _obs("target", _OLD, 9), _obs("other", _OTHER, 8)]
    )
    monkeypatch.setattr(cli, "Embedder", _Stub)
    yield s
    s.close()


def _states(s):
    return {
        r["obs_id"]: r["embedding_state"]
        for r in s._c().execute("SELECT obs_id, embedding_state FROM observations")
    }


def _stored_marker(s, obs_id: str):
    row = s._c().execute(
        "SELECT embedding FROM vec_observations WHERE obs_id = ?", (obs_id,)
    ).fetchone()
    if row is None:
        return None
    return float(np.frombuffer(row[0], dtype=np.float32)[0])


def _race_in_the_select_window(store, monkeypatch):
    """Edit the row the instant ``select_unembedded`` hands it over.

    That is the window itself: after the SELECT returned the body, before the
    worker can read any fingerprint. Injected here rather than timed, so the
    interleaving lands in exactly one place.
    """
    real = store.select_unembedded
    fired: list[int] = []

    def racing_select(kind, limit=500, model=None, source=None):
        # The scope is forwarded, not swallowed: the worker drains one priority
        # group at a time, and a stub that ignored ``source`` would hand every
        # group the whole backlog and race a different row than the one named.
        rows = real(kind, limit=limit, model=model, source=source)
        if rows and not fired:
            fired.append(1)
            store.upsert_entities([_obs("target", _NEW, 9)])
        return rows

    monkeypatch.setattr(store, "select_unembedded", racing_select)
    return fired


def test_a_row_edited_inside_the_select_window_is_not_marked(store, monkeypatch):
    fired = _race_in_the_select_window(store, monkeypatch)

    rc = cli.main(["embed", "--once", "--batch-size", "5"], _store=store)

    assert fired, "the interleaving never happened; this test proves nothing"
    assert rc == 0
    assert _states(store)["target"] is None, (
        "the worker marked a row 'ok' whose body had already changed — "
        "``select_unembedded`` only looks at NULL, so nothing will ever come "
        f"back for it. states={_states(store)!r}"
    )


def test_the_stale_vector_is_not_written_in_the_select_window(store, monkeypatch):
    fired = _race_in_the_select_window(store, monkeypatch)

    cli.main(["embed", "--once", "--batch-size", "5"], _store=store)

    assert fired
    assert _stored_marker(store, "target") is None, (
        "the OLD body's vector was written as current for a row whose body "
        "had already changed"
    )


def test_a_catchup_run_ends_with_the_current_bodys_vector(store, monkeypatch):
    """Stated as what a search returns, which is the thing that matters."""
    fired = _race_in_the_select_window(store, monkeypatch)

    rc = cli.main(["embed", "--catchup", "--batch-size", "5"], _store=store)

    assert fired
    assert rc == 0
    assert _stored_marker(store, "target") == _MARKERS[_NEW]
    assert _states(store)["target"] == "ok"


def test_the_rest_of_the_batch_is_unaffected(store, monkeypatch):
    """The guard costs the edited row only, not the batch around it."""
    _race_in_the_select_window(store, monkeypatch)

    cli.main(["embed", "--once", "--batch-size", "5"], _store=store)

    assert _states(store)["other"] == "ok"
    assert _stored_marker(store, "other") == _MARKERS[_OTHER]


def test_a_superseded_row_is_reported_as_superseded_not_failed(
    store, monkeypatch, capsys
):
    """It self-heals on the next pass, so a ledger entry would only make
    ``aggregator status`` overstate the damage."""
    _race_in_the_select_window(store, monkeypatch)

    rc = cli.main(["embed", "--once", "--batch-size", "5"], _store=store)

    assert rc == 0
    out = capsys.readouterr()
    assert "edited by ingest while being embedded" in out.out
    # Not a failure: the run stays green and the ledger holds nothing.
    assert out.err == ""


def test_a_quiet_run_is_still_marked_ok(store):
    """The guard must not read an ordinary run as a race."""
    rc = cli.main(["embed", "--catchup", "--batch-size", "5"], _store=store)

    assert rc == 0
    assert _states(store) == {"target": "ok", "other": "ok"}
    assert store.count_vec_rows("observations") == 2


# --- the store primitives the close is built from ---------------------------


def test_select_unembedded_returns_the_fingerprint_with_the_body(store):
    """From ONE statement. A later read is a different point in time."""
    for row in store.select_unembedded("observations"):
        assert set(row.keys()) >= {"obs_id", "body", "src_hash"}


def test_mark_embedded_skips_a_row_whose_hash_moved(store):
    before = {r["obs_id"]: r["src_hash"] for r in store.select_unembedded(
        "observations"
    )}
    store.upsert_entities([_obs("target", _NEW, 9)])

    written = store.mark_embedded(
        "observations", ["target", "other"], "skip", before
    )

    assert written == ["other"]
    assert _states(store)["target"] is None


def test_mark_embedded_still_marks_a_row_with_a_null_fingerprint(store):
    """A legacy row, or one a test inserted by hand, is unchanged and must
    still leave the backlog — ``IS`` rather than ``=`` is what makes that
    work, and ``=`` would strand it at NULL forever."""
    store._c().execute("UPDATE observations SET src_hash = NULL WHERE obs_id='target'")
    store._c().commit()

    written = store.mark_embedded(
        "observations", ["target"], "skip", {"target": None}
    )

    assert written == ["target"]
    assert _states(store)["target"] == "skip"


def test_mark_embedded_does_not_mark_a_row_that_vanished(store):
    before = {r["obs_id"]: r["src_hash"] for r in store.select_unembedded(
        "observations"
    )}
    store._c().execute("DELETE FROM observations WHERE obs_id='target'")
    store._c().commit()

    assert store.mark_embedded(
        "observations", ["target"], "skip", before
    ) == []


def test_the_error_state_is_deliberately_not_guarded(store):
    """``'error'`` is not a claim about content — it says the worker could not
    embed the row — and the quarantine ledger owns when it comes back. Guarding
    it would leave a failed row at NULL, which the very next batch of the same
    run re-selects: the abort loop this path exists to prevent."""
    store.upsert_entities([_obs("target", _NEW, 9)])

    assert store.mark_embedded("observations", ["target"], "error") == ["target"]
    assert _states(store)["target"] == "error"


def test_mark_embedded_without_expected_behaves_as_before(store):
    assert store.mark_embedded("observations", ["target", "other"], "skip") == [
        "target",
        "other",
    ]
    assert _states(store) == {"target": "skip", "other": "skip"}


def test_commit_embed_batch_writes_vectors_and_watermark_together(store):
    rows = store.select_unembedded("observations")
    expected = {r["obs_id"]: r["src_hash"] for r in rows}
    vecs = [("target", "target", _vector_for(_OLD))]

    ok, skip = store.commit_embed_batch(
        "observations",
        vectors=vecs,
        ok_ids=["target"],
        skip_ids=["other"],
        error_ids=[],
        expected=expected,
    )

    assert (ok, skip) == (["target"], ["other"])
    assert _states(store) == {"target": "ok", "other": "skip"}
    assert _stored_marker(store, "target") == _MARKERS[_OLD]


def test_commit_embed_batch_refuses_the_vector_of_a_moved_row(store):
    rows = store.select_unembedded("observations")
    expected = {r["obs_id"]: r["src_hash"] for r in rows}
    store.upsert_entities([_obs("target", _NEW, 9)])

    ok, _ = store.commit_embed_batch(
        "observations",
        vectors=[("target", "target", _vector_for(_OLD))],
        ok_ids=["target"],
        skip_ids=[],
        error_ids=[],
        expected=expected,
    )

    assert ok == []
    assert _stored_marker(store, "target") is None
    assert _states(store)["target"] is None


def test_commit_embed_batch_guards_every_chunk_of_a_moved_row(store):
    """A long body is stored as ``<id>:0 .. <id>:N-1``, so the guard has to be
    expressed against the OWNER while the write targets the chunk."""
    rows = store.select_unembedded("observations")
    expected = {r["obs_id"]: r["src_hash"] for r in rows}
    store.upsert_entities([_obs("target", _NEW, 9)])

    store.commit_embed_batch(
        "observations",
        vectors=[
            ("target", f"target:{i}", _vector_for(_OLD)) for i in range(3)
        ],
        ok_ids=["target"],
        skip_ids=[],
        error_ids=[],
        expected=expected,
    )

    assert store.count_vec_rows("observations") == 0
