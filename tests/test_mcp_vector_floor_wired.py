"""Criterion D's per-arm distance floor must run on the DEFAULT query path.

THIS FILE EXISTS BECAUSE A PASSING UNIT TEST IS NOT EVIDENCE THAT A GUARD IS
LIVE. ``hybrid.vector_floor`` shipped fully tested and with no production
caller: ``Store._vec_obs_ids`` ordered by distance and then selected the id
column alone, so the number the floor reads was computed by sqlite-vec and
discarded one layer below ``aggregator.mcp``. Three independent reviewers found
the same shape on an earlier round's ``migrate(embedder=)``. So the assertions
here are deliberately about the PRODUCTION CALL, not about the function:

* ``test_the_floor_runs_on_a_plain_default_query`` spies on the symbol
  ``aggregator.mcp`` actually calls and fails if a default ``aggregator_query``
  — no flags, no ``search_mode``, no fixtures that force the path — never
  reaches it. Unwire the call in ``_fused_id_scope`` and this test goes red.
* the behaviour tests below then prove the call CHANGES THE ANSWER, because a
  guard that is invoked and ignored is the same defect one layer up.

A DISTANT NEIGHBOUR IS NOT A MISS AT THE TOOL BOUNDARY, and the tests say so:
when the floor empties the vector arm the query still answers from FTS5, so the
result stays a superset of keyword search. The floor can only ever remove
vector-ONLY candidates.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

import aggregator.mcp as mcp
from aggregator.core.hybrid import VECTOR_FLOOR_MIN_SAMPLE
from aggregator.core.store import Store
from aggregator.sources.base import ObservationRow, SessionRow

_TS = datetime(2026, 7, 1, 8, 0, tzinfo=UTC)
_DIM = 768

# 20 far neighbours + 1 near one. ``VECTOR_FLOOR_MIN_SAMPLE`` is 20, so a
# smaller set would fail open and the floor's decision would never be visible.
_FAR = VECTOR_FLOOR_MIN_SAMPLE


def _unit(axis: int) -> np.ndarray:
    v = np.zeros(_DIM, dtype=np.float32)
    v[axis] = 1.0
    return v


class StubEmbedder:
    """No real model, ever. An earlier round named one in a test and pulled
    15 GB off a CDN; nothing here is about embedding quality.

    Every query embeds to ``e0``, which is exactly the near document's vector
    and orthogonal to every far one — so the candidate distances are ``0.0``
    once and ``sqrt(2)`` twenty times. Those 21 values put the near document at
    z = 4.47 and every far one at z = -0.22 against ``VECTOR_FLOOR_Z = 3.0``:
    one survivor, twenty dropped, and no reliance on a real embedding space.
    """

    def __init__(self):
        self.query_calls = 0

    def embed_query(self, query: str) -> np.ndarray:
        self.query_calls += 1
        return _unit(0)

    def embed_documents(self, docs: list[str]) -> np.ndarray:
        return np.array([_unit(0) for _ in docs], dtype=np.float32)


@pytest.fixture
def store(tmp_path):
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    return s


@pytest.fixture
def embedder(monkeypatch):
    stub = StubEmbedder()
    monkeypatch.setattr("aggregator.mcp._get_embedder", lambda: stub)
    return stub


def _add(store: Store, obs_id: str, body: str) -> None:
    sid = f"s-{obs_id}"
    store.upsert_entities(
        [
            SessionRow(
                session_id=sid, root_session_id=sid, parent_session_id=None,
                kind="session", agent_id=None, agent_type=None,
                spawned_by_tool_use_id=None, cwd="/x", git_branch="main",
                first_ts=_TS, last_ts=_TS, jsonl_path=f"/tmp/{sid}.jsonl",
            ),
            ObservationRow(
                obs_id=obs_id, session_id=sid, root_session_id=sid,
                parent_obs_id=None, type="user", ts=_TS, model=None,
                input_tokens=None, output_tokens=None, tool_name=None,
                tool_use_id=None, body=body,
            ),
        ]
    )


def _seed(store: Store) -> None:
    """One near neighbour and ``_FAR`` distant ones, none of them lexical hits.

    No document contains the query word, so FTS5 contributes nothing and every
    id in the answer arrived through the vector arm. That is what makes the
    floor's effect readable at the tool boundary rather than masked by a
    keyword match.
    """
    _add(store, "o-near", "quadratic voting rollout")
    store.upsert_vec_observations([("o-near", _unit(0))])
    store.mark_embedded("observations", ["o-near"], state="ok")
    for i in range(_FAR):
        obs_id = f"o-far-{i:02d}"
        _add(store, obs_id, f"unrelated pigeon husbandry note {i}")
        store.upsert_vec_observations([(obs_id, _unit(1 + i))])
        store.mark_embedded("observations", [obs_id], state="ok")


def _ids(result) -> set[str]:
    return {r["stable_id"] for r in result.get("records", [])}


# --- the proof: the floor is on the default path ----------------------------


def test_the_floor_runs_on_a_plain_default_query(store, embedder, monkeypatch):
    """THE ACCEPTANCE BAR FOR CRITERION D. Not "the unit test passes"."""
    _seed(store)
    calls: list[list[tuple[str, float]]] = []
    real = mcp.vector_floor

    def spy(scored, **kw):
        calls.append(list(scored))
        return real(scored, **kw)

    monkeypatch.setattr(mcp, "vector_floor", spy)
    result = mcp.aggregator_query("source:sessions governance", _store=store)
    assert result["ok"] is True, result
    assert calls, (
        "hybrid.vector_floor was never called on a default aggregator_query; "
        "criterion D's abstention rule has no production caller"
    )
    # It has to be handed DISTANCES, not ids: a caller that passed the id list
    # would satisfy the spy and threshold nothing.
    assert all(
        isinstance(doc_id, str) and isinstance(dist, float)
        for doc_id, dist in calls[0]
    ), calls[0]


def test_the_floor_drops_the_distant_neighbours_from_the_answer(store, embedder):
    """Invoked AND obeyed: the 20 orthogonal neighbours must not come back."""
    _seed(store)
    result = mcp.aggregator_query("source:sessions governance", _store=store)
    assert result["ok"] is True, result
    assert _ids(result) == {"s-o-near"}


def test_a_small_candidate_set_still_answers(store, embedder):
    """FAILS OPEN, and the default path inherits that.

    Below ``VECTOR_FLOOR_MIN_SAMPLE`` there is no distribution to judge, and a
    floor that fired on noise would produce "search got smarter and stopped
    finding the thing I know is in there".
    """
    _add(store, "o-near", "quadratic voting rollout")
    store.upsert_vec_observations([("o-near", _unit(0))])
    store.mark_embedded("observations", ["o-near"], state="ok")
    _add(store, "o-far", "unrelated pigeon husbandry note")
    store.upsert_vec_observations([("o-far", _unit(1))])
    store.mark_embedded("observations", ["o-far"], state="ok")

    result = mcp.aggregator_query("source:sessions governance", _store=store)
    assert result["ok"] is True, result
    assert _ids(result) == {"s-o-near", "s-o-far"}


def test_a_floored_out_arm_still_returns_the_keyword_rows(store, embedder):
    """The floor may only ever remove vector-ONLY candidates.

    ``o-lex`` contains the query word, so FTS5 reaches it. Its vector sits with
    the far cluster and the floor drops it from the vector arm — and it must
    still be in the answer, because the fused set is a superset of the keyword
    arm's by construction.
    """
    _seed(store)
    _add(store, "o-lex", "governance working notes")
    store.upsert_vec_observations([("o-lex", _unit(1 + _FAR))])
    store.mark_embedded("observations", ["o-lex"], state="ok")

    result = mcp.aggregator_query("source:sessions governance", _store=store)
    assert result["ok"] is True, result
    assert _ids(result) == {"s-o-near", "s-o-lex"}


def test_the_page_token_freezes_the_survivors_not_the_raw_knn(store, embedder):
    """A continuation must reproduce the page it is continuing.

    The token freezes the vector arm's hits so the candidate set cannot move
    mid-pagination. Freezing the pre-floor list would mean page 2 was cut from
    a wider set than page 1 — the floor would apply to the first page only.
    """
    _seed(store)
    page1 = mcp.aggregator_query(
        "source:sessions governance", page_size=1, _store=store
    )
    assert page1["ok"] is True, page1
    token = page1.get("next_page_token")
    if token is None:
        # One survivor, so there is no second page — which is itself the
        # assertion: the frozen set held the survivor alone.
        assert _ids(page1) == {"s-o-near"}
        return
    page2 = mcp.aggregator_query(
        "source:sessions governance", page_size=1, page_token=token, _store=store
    )
    assert page2["ok"] is True, page2
    assert _ids(page1) | _ids(page2) == {"s-o-near"}


# --- the store reads the floor needs -----------------------------------------


def test_the_observation_arm_reports_distances_in_ascending_order(store, embedder):
    _seed(store)
    scored = store._vec_obs_scored(_unit(0), 5)
    assert scored[0][0] == "o-near"
    assert scored[0][1] == pytest.approx(0.0, abs=1e-6)
    assert [d for _, d in scored] == sorted(d for _, d in scored)


def test_the_record_arm_reports_distances_too(store):
    """Records are the other ontology and have their own vec table; a floor
    wired for one arm and not the other is half a rule."""
    from aggregator.sources.base import Record

    store.upsert(
        [
            Record(
                stable_id="github:1", source="github", subject="pr one",
                body="body one", updated_at=_TS,
            )
        ]
    )
    store.upsert_vec_records([("github:1", _unit(0))])
    store.mark_embedded("records", ["github:1"], state="ok")
    scored = store._vec_record_scored(_unit(0), 5)
    assert scored == [("github:1", pytest.approx(0.0, abs=1e-6))]
