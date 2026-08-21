"""A page token is only meaningful to the query that minted it.

Round 1 packed the vector arm's hit ids INTO the token so a continuation stops
re-running the KNN — the right fix for a set that moved underneath a caller.
It also made the token carry query-specific state without saying so anywhere.

Nothing bound that state to a query. ``_parse_page_token`` decoded an offset
and a frozen id list; ``_fused_id_scope`` then substituted those ids for the
KNN and RRF-unioned them with the NEW dsl's FTS ids. So a caller that reused a
token against a changed ``dsl`` silently received the PREVIOUS query's 50
frozen vector hits inside a different query's result set, at an offset cut from
a set that never existed for it — and got ``ok: True``.

The caller here is an LLM driving an MCP tool. It cannot ask a follow-up
question and it cannot tell a plausible page from a correct one, so a
wrong-but-plausible page is strictly worse than an error.

The contract these tests pin: every token carries a fingerprint of the query
that minted it, a token whose fingerprint disagrees with the current call is
refused structurally, and frozen hits may not travel without one.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from aggregator.core.store import Store
from aggregator.mcp import _VECTOR_ARM_K, aggregator_query
from aggregator.sources.base import ObservationRow, SessionRow

# Two orthogonal topics. The stub embedder puts each on its own axis, so a
# query about one is at distance 0 from its own topic and orthogonal to the
# other — which is what makes "the wrong query's neighbours" visible.
_AXES = {"voting": 0, "governance": 0, "quadratic": 0, "pigeon": 1, "roost": 1}

# DERIVED FROM THE ARM'S DEPTH, not a literal beside it. The seed has to put
# the CORPUS (two topics) above the cap while leaving ONE topic exactly at it:
# below the cap the KNN returns the whole corpus for any query, so both
# queries' vector arms would be the same set and the injection would be
# invisible. Criterion G raised the depth from 50 to 150 and this file was the
# only place in the suite where a hardcoded 50 quietly voided the premise —
# every assertion still passed, against a test that no longer tested anything.
_PER_TOPIC = _VECTOR_ARM_K


class StubEmbedder:
    """No real model, ever. An earlier round named one in a test and pulled
    15 GB off a CDN; nothing here is about embedding quality."""

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


def _session(session_id: str, ts: datetime) -> SessionRow:
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


def _obs(obs_id: str, session_id: str, body: str, ts: datetime) -> ObservationRow:
    return ObservationRow(
        obs_id=obs_id,
        session_id=session_id,
        root_session_id=session_id,
        parent_obs_id=None,
        type="user",
        ts=ts,
        model=None,
        input_tokens=None,
        output_tokens=None,
        tool_name=None,
        tool_use_id=None,
        body=body,
    )


def _seed_two_topics(store: Store) -> None:
    """Two disjoint topics, the voting one strictly NEWER than the pigeon one.

    Recency ordering is what the offset addresses, so putting the wrong
    query's rows at the top is what makes a contaminated page 2 land on them.
    """
    base = datetime(2026, 7, 1, tzinfo=UTC)
    docs: list[tuple[str, str, datetime]] = []
    for i in range(_PER_TOPIC):
        docs.append((f"p{i}", f"pigeon roost note {i}", base + timedelta(minutes=i)))
    for i in range(_PER_TOPIC):
        docs.append(
            (f"v{i}", f"voting note {i}", base + timedelta(days=10, minutes=i))
        )
    entities = []
    for obs_id, body, ts in docs:
        entities.append(_session(f"s-{obs_id}", ts))
        entities.append(_obs(obs_id, f"s-{obs_id}", body, ts))
    store.upsert_entities(entities)

    stub = StubEmbedder()
    vecs = stub.embed_documents([b for _, b, _ in docs])
    store.upsert_vec_observations(
        [(obs_id, vecs[n]) for n, (obs_id, _, _) in enumerate(docs)]
    )
    store.mark_embedded("observations", [d[0] for d in docs], state="ok")


def _ids(result) -> list[str]:
    # ``.get``: a refusal carries no ``records`` key, and these ids are only
    # ever read to build a failure message.
    return [r["stable_id"] for r in result.get("records", [])]


def test_a_token_minted_for_one_query_is_refused_by_another(store, embedder):
    """THE REPRO.

    ``governance`` mints a token whose payload freezes a full arm's worth of
    voting-topic vector hits. Handing it back with ``dsl='pigeon'`` fused those
    ids into the pigeon query's candidate set and answered at an offset cut
    from the voting query's ordering — ``ok: True``, no notice, rows a pigeon
    query cannot reach.
    """
    _seed_two_topics(store)

    page1 = aggregator_query("governance", page_size=1, _store=store)
    assert page1["ok"] is True, page1
    token = page1["next_page_token"]
    assert token.startswith("h") and "." in token, (
        f"page 1 must be hybrid AND carry frozen hits for this repro: {token!r}"
    )

    fresh = aggregator_query("pigeon", page_size=200, _store=store)
    fresh_ids = set(_ids(fresh))

    reused = aggregator_query(
        "pigeon", page_size=1, page_token=token, _store=store
    )

    smuggled = [i for i in _ids(reused) if i not in fresh_ids]
    assert reused["ok"] is False, (
        f"a token minted for 'governance' was accepted for 'pigeon': "
        f"returned {_ids(reused)}, of which {smuggled} are rows a fresh "
        f"'pigeon' query never returns — the previous query's frozen vector "
        f"hits, served as page 2 of this one"
    )
    assert "page_token" in reused["reason"], reused
    assert reused["remediation"], reused


def test_the_previous_querys_frozen_hits_cannot_reach_the_new_scope(
    store, embedder
):
    """The narrow claim, stated separately from the refusal.

    Even with the offset ignored, the fused scope for the reused token was the
    UNION of this query's FTS ids and the other query's KNN ids. Asking for
    page 1 (offset 0) isolates that: the offset is trivially right, and only
    the membership can be wrong.
    """
    _seed_two_topics(store)
    minted = aggregator_query("governance", page_size=1, _store=store)
    frozen_payload = minted["next_page_token"].partition(".")[2]
    assert frozen_payload

    fresh = aggregator_query("pigeon", page_size=200, _store=store)
    reused = aggregator_query(
        "pigeon",
        page_size=200,
        page_token=f"h0.{frozen_payload}",
        _store=store,
    )

    assert reused["ok"] is False, (
        f"the voting query's frozen hits widened the pigeon query's scope by "
        f"{sorted(set(_ids(reused)) - set(_ids(fresh)))}"
    )


def test_the_same_query_still_pages_through_its_own_token(store, embedder):
    """The refusal must not cost the feature it protects."""
    _seed_two_topics(store)
    seen: list[str] = []
    token = None
    # Bounded by the corpus rather than by a literal page count: the page size
    # is fixed and the seed size follows ``_VECTOR_ARM_K``, so a hardcoded
    # number of iterations stops covering the set the moment the depth moves.
    for _ in range(2 * _PER_TOPIC // 20 + 2):
        page = aggregator_query(
            "governance", page_size=20, page_token=token, _store=store
        )
        assert page["ok"] is True, page
        seen += _ids(page)
        token = page.get("next_page_token")
        if not token:
            break
    assert token is None, "pagination did not terminate within the corpus"
    assert len(seen) == len(set(seen)), f"row served twice: {seen}"
    assert set(seen) == {f"s-v{i}" for i in range(_PER_TOPIC)}


def test_an_equivalent_dsl_keeps_its_token(store, embedder):
    """The fingerprint is over the PARSED query, not the string.

    ``source:github pr`` and ``pr source:github`` are the same query written
    two ways. Binding to the raw text would refuse a caller that did nothing
    wrong, and a refusal a caller cannot act on is its own failure mode.
    """
    _seed_records(store, 3)
    page1 = aggregator_query("source:github pr", page_size=1, _store=store)
    token = page1["next_page_token"]
    page2 = aggregator_query(
        "pr   source:github", page_size=1, page_token=token, _store=store
    )
    assert page2["ok"] is True, page2


def _seed_records(store: Store, n: int = 3) -> None:
    from aggregator.sources.base import Record

    store.upsert(
        [
            Record(
                stable_id=f"github:acme/api:{i}",
                source="github",
                subject=f"pr {i}",
                body=f"body of pull request {i}",
                tags=["pr"],
                created_at=datetime(2026, 7, 20 + i, tzinfo=UTC),
                updated_at=datetime(2026, 7, 20 + i, tzinfo=UTC),
            )
            for i in range(n)
        ]
    )


def test_a_narrowed_filter_invalidates_the_token(store, embedder):
    """No vector arm anywhere in this one: the OFFSET alone is enough.

    ``source:github`` and ``source:github state:open`` order different row
    sets, so row 1 of the first is not row 1 of the second. The frozen-hit
    injection is the sharp end of the finding; this is the blunt end, and it
    reaches every FTS5-only token too.
    """
    _seed_records(store, 3)
    token = aggregator_query("source:github", page_size=1, _store=store)[
        "next_page_token"
    ]
    narrowed = aggregator_query(
        "source:github state:open", page_size=1, page_token=token, _store=store
    )
    assert narrowed["ok"] is False, narrowed
    assert "page_token" in narrowed["reason"], narrowed


def test_a_drilldown_flip_invalidates_the_token(store, embedder):
    """``drilldown`` selects WHICH TABLE the offset indexes.

    Session cards and observation rows are different sets of different sizes;
    an offset carried across the flip addresses neither. Same hazard, same
    mechanism, one boolean away — so it is bound with the dsl rather than left
    as the one argument that can still hand back a plausible wrong page.
    """
    _seed_two_topics(store)
    token = aggregator_query(
        "source:sessions", page_size=1, _store=store
    )["next_page_token"]
    flipped = aggregator_query(
        "source:sessions",
        page_size=1,
        page_token=token,
        drilldown=True,
        _store=store,
    )
    assert flipped["ok"] is False, flipped


def test_frozen_hits_cannot_travel_without_a_binding(store, embedder):
    """Structural, like round 2's refusals: this server never mints one.

    Stripping the fingerprint off a token is not a downgrade path back to the
    unbound behaviour — a payload with no query binding is refused outright.
    """
    _seed_two_topics(store)
    token = aggregator_query("governance", page_size=1, _store=store)[
        "next_page_token"
    ]
    head, _, payload = token.partition(".")
    unbound = f"{head.partition('~')[0]}.{payload}"

    result = aggregator_query("governance", page_token=unbound, _store=store)
    assert result["ok"] is False, result
    assert "page_token" in result["reason"], result
