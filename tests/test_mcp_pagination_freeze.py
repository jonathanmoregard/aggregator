"""A page token must freeze the candidate SET, not merely the arm.

The v5 token records which arm minted it (``"h40"`` vs ``"40"``), which
stops a continuation flipping between FTS5-only and hybrid. That fixed a
real hazard and it is not the whole hazard.

Inside a pinned ``h`` token the vector arm is still RE-EVALUATED on every
page: ``_fused_id_scope`` re-runs the KNN top-``_VECTOR_ARM_K``. The embed
worker is on a 30-minute timer and the backfill runs for 25-30 days, so
vectors land between page 1 and page 2 routinely. A new neighbour joins the
fused set, the set the offset addresses is no longer the set page 1 was cut
from, and the caller silently re-reads some rows and never sees others —
for weeks, on a warm-up that only happens once and cannot be re-run.

The fix: the token carries the vector arm's own hits, and a continuation
reuses them instead of asking the index again. Bounded by construction —
at most ``_VECTOR_ARM_K`` ids, ~1.5 KB packed. The FTS5 arm is NOT frozen:
it drifts with ingest exactly as it did before v5, which is a pre-existing
property of a stateless offset API and not something hybrid introduced.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from aggregator.core.store import Store
from aggregator.mcp import _parse_page_token, aggregator_query
from aggregator.sources.base import ObservationRow, SessionRow

_AXES = {"voting": 0, "governance": 0, "quadratic": 0, "pigeon": 1, "roost": 1}


class StubEmbedder:
    def __init__(self):
        self.query_calls = 0

    @staticmethod
    def _vec_for(text):
        v = np.zeros(768, dtype=np.float32)
        lowered = (text or "").lower()
        for word, axis in _AXES.items():
            if word in lowered:
                v[axis] = 1.0
                return v
        v[2] = 1.0
        return v

    def embed_query(self, query):
        self.query_calls += 1
        return self._vec_for(query)

    def embed_documents(self, docs):
        return np.array([self._vec_for(d) for d in docs], dtype=np.float32)


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


def _session(session_id, day):
    ts = datetime(2026, 7, day, 8, 0, tzinfo=UTC)
    return SessionRow(
        session_id=session_id,
        root_session_id=session_id,
        parent_session_id=None,
        kind="session",
        agent_id=None,
        agent_type=None,
        spawned_by_tool_use_id=None,
        cwd="/x",
        git_branch="main",
        first_ts=ts,
        last_ts=ts,
        jsonl_path=f"/tmp/{session_id}.jsonl",
    )


def _obs(obs_id, session_id, body, day):
    return ObservationRow(
        obs_id=obs_id,
        session_id=session_id,
        root_session_id=session_id,
        parent_obs_id=None,
        type="user",
        ts=datetime(2026, 7, day, 8, 0, tzinfo=UTC),
        model=None,
        input_tokens=None,
        output_tokens=None,
        tool_name=None,
        tool_use_id=None,
        body=body,
    )


def _seed(store, docs):
    entities = []
    for obs_id, body, day in docs:
        sid = f"s-{obs_id}"
        entities.append(_session(sid, day))
        entities.append(_obs(obs_id, sid, body, day))
    store.upsert_entities(entities)


def _embed(store, pairs):
    stub = StubEmbedder()
    vecs = stub.embed_documents([t for _, t in pairs])
    store.upsert_vec_observations([(i, vecs[n]) for n, (i, _) in enumerate(pairs)])
    store.mark_embedded("observations", [i for i, _ in pairs], state="ok")


def _ids(result):
    return [r["stable_id"] for r in result["records"]]


def test_backfill_between_pages_does_not_duplicate_or_skip_rows(store, embedder):
    """THE REPRO. Page 1 is already hybrid, so the arm never flips — and the
    rows still shift, because the KNN is re-run against a bigger index."""
    docs = [(f"o{i}", f"voting note {i}", i + 1) for i in range(4)]
    _seed(store, docs)
    _embed(store, [(i, b) for i, b, _ in docs])

    page1 = aggregator_query("governance", page_size=2, _store=store)
    assert page1["total"] == 4
    token = page1["next_page_token"]
    assert token.startswith("h"), "page 1 must already be hybrid for this repro"

    # The embed worker lands one more row, NEWER than everything on page 1.
    _seed(store, [("o9", "quadratic backfill arrival", 30)])
    _embed(store, [("o9", "quadratic backfill arrival")])

    page2 = aggregator_query(
        "governance", page_size=2, page_token=token, _store=store
    )
    assert page2["ok"] is True

    seen = _ids(page1) + _ids(page2)
    assert len(seen) == len(set(seen)), f"row served twice across pages: {seen}"


def test_a_continuation_reuses_the_frozen_hits_instead_of_re_running_knn(
    store, embedder
):
    """Proof the freeze is real: the candidate set cannot have moved, because
    the index was never consulted for page 2."""
    docs = [(f"o{i}", f"voting note {i}", i + 1) for i in range(4)]
    _seed(store, docs)
    _embed(store, [(i, b) for i, b, _ in docs])

    page1 = aggregator_query("governance", page_size=2, _store=store)
    token = page1["next_page_token"]

    _seed(store, [("o9", "quadratic backfill arrival", 30)])
    _embed(store, [("o9", "quadratic backfill arrival")])

    before = embedder.query_calls
    page2 = aggregator_query(
        "governance", page_size=2, page_token=token, _store=store
    )
    assert page2["ok"] is True
    assert embedder.query_calls == before, (
        "a continuation re-embedded the query, so it also re-ran the KNN"
    )
    assert "o9" not in {i.removeprefix("s-") for i in _ids(page2)}


def test_the_frozen_set_survives_a_whole_pagination(store, embedder):
    docs = [(f"o{i}", f"voting note {i}", i + 1) for i in range(6)]
    _seed(store, docs)
    _embed(store, [(i, b) for i, b, _ in docs])

    seen: list[str] = []
    token = None
    for page_no in range(4):
        page = aggregator_query(
            "governance", page_size=2, page_token=token, _store=store
        )
        assert page["ok"] is True
        seen += _ids(page)
        # A batch lands between every single page.
        _seed(store, [(f"n{page_no}", "quadratic later arrival", 20 + page_no)])
        _embed(store, [(f"n{page_no}", "quadratic later arrival")])
        token = page.get("next_page_token")
        if not token:
            break

    assert len(seen) == len(set(seen)), f"rows served twice: {seen}"
    assert set(seen) == {f"s-o{i}" for i in range(6)}


def test_a_token_carrying_frozen_ids_round_trips(store):
    from aggregator.mcp import _mint_page_token

    frozen = {"observations": ["a-1", "b-2:3"], "records": ["github:x:1"]}
    token = _mint_page_token(40, True, "fingerprint01", frozen)
    cursor = _parse_page_token(token)
    assert cursor.offset == 40
    assert cursor.hybrid is True
    assert cursor.frozen == frozen
    assert cursor.fingerprint == "fingerprint01"


def test_legacy_tokens_keep_working(store, embedder):
    """Tokens already in flight predate the payload and must still page."""
    for token, offset, hybrid in (("2", 2, False), ("h2", 2, True), (None, 0, None)):
        cursor = _parse_page_token(token)
        assert (cursor.offset, cursor.hybrid) == (offset, hybrid)
        assert cursor.frozen is None


def test_a_garbage_payload_is_refused_rather_than_raising(store, embedder):
    """Still no traceback — but no longer a silent fallback either.

    This used to assert ``ok is True``: an unreadable payload dropped the
    frozen set, re-ran the KNN and answered from offset 0. That is the failure
    the freeze exists to prevent, arrived at by a different route, and the
    caller had no way to see it. The contract now is a structured refusal;
    ``tests/test_mcp_page_token_hardening.py`` owns the detail.
    """
    _seed(store, [("o1", "quadratic voting", 1)])
    _embed(store, [("o1", "quadratic voting")])
    result = aggregator_query(
        "governance", page_token="h0~fingerprint01.!!!not-base64!!!", _store=store
    )
    assert result["ok"] is False
    assert "page_token" in result["reason"]
    assert result["remediation"]


def test_a_fresh_query_still_gets_the_current_index(store, embedder):
    """Freezing must not make new results unreachable — only continuations."""
    docs = [(f"o{i}", f"voting note {i}", i + 1) for i in range(2)]
    _seed(store, docs)
    _embed(store, [(i, b) for i, b, _ in docs])
    aggregator_query("governance", page_size=1, _store=store)

    _seed(store, [("o9", "quadratic backfill arrival", 30)])
    _embed(store, [("o9", "quadratic backfill arrival")])

    fresh = aggregator_query("governance", page_size=10, _store=store)
    assert "s-o9" in _ids(fresh)


def test_union_mode_freezes_both_ontologies_independently(store, embedder):
    """Records and observations backfill at different speeds; one token
    has to pin both or the faster table drifts underneath the slower one."""
    from aggregator.mcp import _mint_page_token

    token = _mint_page_token(
        10, True, "fingerprint01", {"observations": ["o1"], "records": ["github:x:1"]}
    )
    cursor = _parse_page_token(token)
    assert set(cursor.frozen) == {"observations", "records"}
