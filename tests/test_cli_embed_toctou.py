"""A body edited while the worker holds it must not be marked embedded.

The embed worker selects a batch, embeds each row, and then marks the whole
batch ``embedding_state='ok'``. Ingest, meanwhile, sets ``embedding_state``
back to NULL and drops the row's vectors whenever a body changes — that is
how an edited row gets re-embedded at all.

Interleave the two and the invalidation is undone by the worker that raced
it: the worker embeds the OLD body, ingest edits the row and drops its
vectors, and the worker then upserts the old body's vector back and marks the
row ``'ok'`` unconditionally. ``select_unembedded`` only ever looks at NULL,
so nothing comes back for that row until the body is edited AGAIN. Until
then the vector arm answers with text the row no longer contains while the
keyword arm — which has a trigger — answers with the text it does. The two
arms of one hybrid query disagree, and the cache reports itself fully
embedded.

The fix is to make the mark conditional on the row still being what the
worker read: re-check ``src_hash`` before writing vectors, and leave anything
that moved in the backlog for the next run.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

import aggregator.cli as cli
from aggregator.core.store import Store
from aggregator.sources.base import ObservationRow, SessionRow

_TARGET_BODY = "ORIGINAL BODY of the row the worker is holding"
_EDITED_BODY = "EDITED BODY that ingest wrote while the worker was busy"
_OTHER_BODY = "an ordinary body nobody touches"


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
    s.upsert_entities([_session(), _obs("target", _TARGET_BODY, 9),
                       _obs("other", _OTHER_BODY, 8)])
    yield s
    s.close()


def _states(s):
    return {
        r["obs_id"]: r["embedding_state"]
        for r in s._c().execute("SELECT obs_id, embedding_state FROM observations")
    }


#: Which body a stored vector was made from. A zero vector would make the two
#: outcomes — old text indexed, new text indexed — indistinguishable, and
#: "which text is the vector arm answering with" is the whole question.
_MARKERS = {_TARGET_BODY: 1.0, _EDITED_BODY: 2.0, _OTHER_BODY: 3.0}


def _marked_vector(doc: str) -> np.ndarray:
    vec = np.zeros(768, dtype=np.float32)
    for body, marker in _MARKERS.items():
        if body in doc:
            vec[0] = marker
    return vec


def _stored_marker(s, obs_id: str):
    row = s._c().execute(
        "SELECT embedding FROM vec_observations WHERE obs_id = ?", (obs_id,)
    ).fetchone()
    if row is None:
        return None
    return float(np.frombuffer(row[0], dtype=np.float32)[0])


def _install_editing_embedder(monkeypatch, s):
    """An embedder that lets ingest edit the row while the worker holds it.

    The edit is triggered from inside ``embed_documents``, at the row it
    concerns, so the interleaving lands in exactly one place instead of
    depending on a clock.
    """
    edits: list[str] = []

    class EditingStub:
        def __init__(self, *a, **kw):
            pass

        def embed_documents(self, docs):
            if not edits and any(_TARGET_BODY in d for d in docs):
                edits.append("done")
                s.upsert_entities([_obs("target", _EDITED_BODY, 9)])
            return np.array([_marked_vector(d) for d in docs], dtype=np.float32)

        def embed_query(self, query):
            return np.zeros(768, dtype=np.float32)

    monkeypatch.setattr(cli, "Embedder", EditingStub)
    return edits


def test_the_vector_kept_for_an_edited_row_is_not_the_old_bodys(store, monkeypatch):
    """THE REPRO, stated as the thing a search actually returns.

    A full ``--catchup`` run: the row is edited mid-embed, and what the vector
    arm ends up holding for it must describe the body the row has now.
    """
    edits = _install_editing_embedder(monkeypatch, store)

    rc = cli.main(["embed", "--catchup", "--batch-size", "2"], _store=store)

    assert edits, "the interleaving never happened; this test proves nothing"
    assert rc == 0
    assert _stored_marker(store, "target") == _MARKERS[_EDITED_BODY], (
        "the vector index holds the OLD body's vector for a row whose body "
        "changed: the worker marked it 'ok' after ingest had invalidated it, "
        "so nothing comes back for it and the two arms of one hybrid query "
        "disagree about what the row says"
    )


def test_a_row_edited_mid_batch_stays_in_the_backlog(store, monkeypatch):
    """One batch, stopped there: the row must still be owed an embedding."""
    edits = _install_editing_embedder(monkeypatch, store)

    rc = cli.main(["embed", "--once", "--batch-size", "2"], _store=store)

    assert edits, "the interleaving never happened; this test proves nothing"
    assert rc == 0
    states = _states(store)
    assert states["target"] is None, (
        "the worker marked a row 'ok' whose body changed underneath it — "
        "``select_unembedded`` only looks at NULL, so nothing will ever come "
        f"back for it. states={states!r}"
    )
    assert _stored_marker(store, "target") is None, (
        "the old body's vector was upserted back after ingest dropped it"
    )


def test_the_untouched_row_in_the_same_batch_is_still_embedded(store, monkeypatch):
    """The guard costs the edited row only — not the batch around it."""
    _install_editing_embedder(monkeypatch, store)

    cli.main(["embed", "--once", "--batch-size", "2"], _store=store)

    assert _states(store)["other"] == "ok"
    assert _stored_marker(store, "other") == _MARKERS[_OTHER_BODY]


def test_a_quiet_batch_is_marked_ok_as_before(store, monkeypatch):
    """The guard must not read an ordinary run as a race."""

    class Stub:
        def __init__(self, *a, **kw):
            pass

        def embed_documents(self, docs):
            return np.array([_marked_vector(d) for d in docs], dtype=np.float32)

        def embed_query(self, query):
            return np.zeros(768, dtype=np.float32)

    monkeypatch.setattr(cli, "Embedder", Stub)

    rc = cli.main(["embed", "--catchup", "--batch-size", "2"], _store=store)

    assert rc == 0
    assert _states(store) == {"target": "ok", "other": "ok"}
    assert store.count_vec_rows("observations") == 2
