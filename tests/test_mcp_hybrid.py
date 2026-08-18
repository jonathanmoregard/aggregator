"""Task I — ``aggregator_search_memory`` routes text through the hybrid arm.

THE SHAPE OF THE FEATURE, because it constrains every test below. When the
vector index has rows and the caller passed free text, the retriever runs two
arms — FTS5 and vector KNN — fuses them with RRF, and hands the store the
fused id set as ``id_scope``. When the vector index is empty, the embedder is
never constructed and the query takes the pre-v5 FTS5 path byte for byte.

TWO INVARIANTS THAT THE TESTS EXIST TO PIN DOWN, because both are easy to
break and neither is visible in a passing happy-path test:

* **Hybrid only ever ADDS.** The FTS5 arm is passed to RRF untruncated, so the
  fused set is a superset of the FTS5 result set. A hybrid query can never
  return fewer keyword matches than the same query without a vector index —
  "we made search smarter and it stopped finding the thing I know is in
  there" is the failure that would kill trust in the tool.
* **Ordering is unchanged.** The store still orders by recency, so a page
  token keeps meaning what it meant. See ``test_page_token_...`` below for the
  one genuine hazard this creates and how it is closed.

Sad paths get the same weight as happy ones: empty index, half-embedded
corpus, missing extension, dead embedder, dead reranker, single-arm matches,
and the filter-only query that must not embed anything at all.
"""

import sqlite3
from datetime import UTC, datetime

import numpy as np
import pytest

from aggregator.core import store as store_mod
from aggregator.core.store import Store
from aggregator.mcp import aggregator_query
from aggregator.sources.base import ObservationRow, Record, SessionRow

# --- stub models -------------------------------------------------------------
#
# A real Qwen3 embedding would make these tests slow and non-deterministic.
# The stub maps a keyword to one axis of a 768-d unit basis, so "which document
# is nearest" is decided by the test author rather than by a model, and the
# vector arm's behaviour is exactly reproducible.

_AXES = {"voting": 0, "governance": 0, "quadratic": 0, "pigeon": 1, "roost": 1}


class StubEmbedder:
    """Deterministic keyword→axis embedder. Counts its own calls so a test
    can assert the embedder was never even asked."""

    def __init__(self):
        self.query_calls = 0

    @staticmethod
    def _vec_for(text: str) -> np.ndarray:
        v = np.zeros(768, dtype=np.float32)
        lowered = (text or "").lower()
        for word, axis in _AXES.items():
            if word in lowered:
                v[axis] = 1.0
                return v
        v[2] = 1.0
        return v

    def embed_query(self, query: str) -> np.ndarray:
        self.query_calls += 1
        return self._vec_for(query)

    def embed_documents(self, docs: list[str]) -> np.ndarray:
        return np.array([self._vec_for(d) for d in docs], dtype=np.float32)


class ExplodingEmbedder:
    """Every embed attempt fails — stands in for a corrupt or missing model."""

    def embed_query(self, query):
        raise RuntimeError("model weights are corrupt")


class StubReranker:
    """Scores by a caller-supplied ranking of substrings; counts calls."""

    def __init__(self, prefer: str):
        self.prefer = prefer
        self.calls = 0

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        self.calls += 1
        return np.array(
            [1.0 if self.prefer in d else 0.0 for d in docs], dtype=np.float32
        )


class ExplodingReranker:
    def score(self, query, docs):
        raise RuntimeError("cross-encoder failed to load")


# --- fixtures ----------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    return s


@pytest.fixture
def embedder(monkeypatch):
    """Install the stub embedder and hand the test its call counter."""
    stub = StubEmbedder()
    monkeypatch.setattr("aggregator.mcp._get_embedder", lambda: stub)
    return stub


@pytest.fixture
def no_vec_store(tmp_path, monkeypatch):
    def _boom(conn):
        raise sqlite3.OperationalError("simulated sqlite-vec ABI mismatch")

    monkeypatch.setattr(store_mod, "_load_sqlite_vec", _boom)
    monkeypatch.setattr(store_mod, "_VEC_LOAD_WARNED", False)
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    assert s.vector_available is False
    return s


def _session(session_id, day=1):
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


def _obs(obs_id, session_id, body, day=1):
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


def _record(stable_id, subject, body, day=1):
    ts = datetime(2026, 7, day, 8, 0, tzinfo=UTC)
    return Record(
        stable_id=stable_id,
        source="github",
        subject=subject,
        body=body,
        tags=[],
        created_at=ts,
        updated_at=ts,
    )


def _seed_sessions(store, docs):
    """``docs`` = [(obs_id, body, day)] — one session per observation."""
    entities = []
    for obs_id, body, day in docs:
        sid = f"s-{obs_id}"
        entities.append(_session(sid, day=day))
        entities.append(_obs(obs_id, sid, body, day=day))
    store.upsert_entities(entities)


def _embed(store, kind, pairs):
    """Write stub vectors for ``pairs`` = [(id, text)] and mark them ok."""
    stub = StubEmbedder()
    vecs = stub.embed_documents([text for _, text in pairs])
    rows = [(id_, vecs[i]) for i, (id_, _) in enumerate(pairs)]
    if kind == "observations":
        store.upsert_vec_observations(rows)
    else:
        store.upsert_vec_records(rows)
    store.mark_embedded(kind, [id_ for id_, _ in pairs], state="ok")


def _ids(result):
    return {r["stable_id"] for r in result["records"]}


# --- falling through to FTS5 when there is no vector index -------------------


def test_text_query_falls_through_to_fts_when_the_vec_table_is_empty(
    store, embedder
):
    _seed_sessions(store, [("o1", "quadratic voting is a governance mechanism", 1)])
    result = aggregator_query("voting", _store=store)
    assert result["ok"] is True
    assert result["total"] == 1


def test_an_empty_vec_table_never_constructs_the_embedder(store, embedder):
    """Not merely "the answer is the same" — the model must not be touched.
    Every session that never runs a vector query pays for it otherwise."""
    _seed_sessions(store, [("o1", "quadratic voting", 1)])
    aggregator_query("voting", _store=store)
    assert embedder.query_calls == 0


def test_a_filter_only_query_never_embeds_anything(store, embedder):
    """A query with no free text has nothing to embed. Even with a fully
    warm vector index, the embedder must not be called."""
    _seed_sessions(store, [("o1", "quadratic voting", 1)])
    _embed(store, "observations", [("o1", "quadratic voting")])
    result = aggregator_query("source:sessions", _store=store)
    assert result["ok"] is True
    assert embedder.query_calls == 0


def test_a_date_only_query_never_embeds_anything(store, embedder):
    _seed_sessions(store, [("o1", "quadratic voting", 1)])
    _embed(store, "observations", [("o1", "quadratic voting")])
    aggregator_query("from:2026-01-01 to:2026-12-31", _store=store)
    assert embedder.query_calls == 0


# --- the vector arm earns its keep -------------------------------------------


def test_hybrid_surfaces_a_row_that_fts_alone_would_miss(store, embedder):
    """The whole point. "governance" shares no token with "quadratic voting",
    so FTS5 cannot reach it; the vector arm can."""
    _seed_sessions(store, [("o1", "quadratic voting explained", 1)])

    # Before the backfill: the same query, the same store, FTS5 alone.
    fts_only = aggregator_query("governance", _store=store)
    assert fts_only["total"] == 0, "precondition: FTS5 alone finds nothing"
    assert embedder.query_calls == 0

    _embed(store, "observations", [("o1", "quadratic voting explained")])
    hybrid = aggregator_query("governance", _store=store)
    assert hybrid["ok"] is True
    assert hybrid["total"] == 1
    assert "s-o1" in _ids(hybrid)
    assert embedder.query_calls == 1


def test_hybrid_never_loses_an_fts_hit(store, embedder):
    """Hybrid only ever adds. A keyword match that the pre-v5 path returned
    must still be returned once the vector arm is warm — including when the
    vector arm's own top-K is full of other documents."""
    docs = [(f"o{i}", f"voting notes number {i}", 1 + i) for i in range(12)]
    _seed_sessions(store, docs)
    # Embed only the pigeon-ish decoys, so the vector arm returns rows that
    # are NOT the FTS hits and cannot be mistaken for them.
    _seed_sessions(store, [("p1", "pigeons roost on ledges", 20)])
    _embed(store, "observations", [("p1", "pigeons roost on ledges")])

    result = aggregator_query("voting", _store=store)
    assert result["ok"] is True
    for i in range(12):
        assert f"s-o{i}" in _ids(result), f"lost FTS hit o{i}"


def test_a_query_matching_only_the_vector_arm_still_returns_ok(store, embedder):
    _seed_sessions(store, [("p1", "pigeons roost on ledges", 1)])
    _embed(store, "observations", [("p1", "pigeons roost on ledges")])
    result = aggregator_query("roost", _store=store)
    assert result["ok"] is True
    assert "s-p1" in _ids(result)


def test_a_query_matching_only_the_fts_arm_still_returns_ok(store, embedder):
    """Vector index warm, but the nearest neighbour is a different document.
    The keyword hit must survive the fusion."""
    _seed_sessions(
        store,
        [("o1", "a note about kubernetes ingress", 1), ("p1", "pigeons roost", 2)],
    )
    _embed(store, "observations", [("p1", "pigeons roost")])
    result = aggregator_query("kubernetes", _store=store)
    assert result["ok"] is True
    assert "s-o1" in _ids(result)


def test_both_arms_empty_returns_an_ok_empty_result(store, embedder):
    """Nothing to keyword-match and no vector index to fall back on."""
    _seed_sessions(store, [("o1", "quadratic voting", 1)])
    result = aggregator_query("nonexistentterm", _store=store)
    assert result["ok"] is True
    assert result["records"] == []
    assert result["total"] == 0


def test_a_warm_vector_arm_returns_neighbours_even_for_an_unrelated_query(
    store, embedder
):
    """KNOWN BEHAVIOUR, PINNED HERE SO IT IS A DECISION AND NOT A SURPRISE.

    KNN has no distance floor: ``ORDER BY distance LIMIT k`` returns the k
    nearest rows however far away they are. So once the vector index is warm,
    a query that matches nothing at all still comes back with up to
    ``_VECTOR_ARM_K`` semantic neighbours, where the FTS5-only path returned
    zero. Recall goes up and precision goes down, which is the trade hybrid
    retrieval IS, but the floor of "no results" disappears with it.

    Deliberately not "fixed" here. A similarity cutoff is a tuned constant,
    the spec defers tuning until there are labelled query pairs, and a
    threshold picked to make a synthetic test go green would be worse than
    none — with real Qwen3 embeddings almost every pair has positive cosine,
    so any cutoff chosen against stub vectors would filter nothing in
    production while looking principled. Task M measures the real
    precision/recall delta against a copy of the live cache; that measurement
    is what should set a cutoff, if one is wanted.
    """
    _seed_sessions(store, [("o1", "quadratic voting", 1)])
    _embed(store, "observations", [("o1", "quadratic voting")])
    result = aggregator_query("nonexistentterm", _store=store)
    assert result["ok"] is True
    assert result["total"] == 1


def test_a_half_embedded_corpus_still_serves_the_unembedded_rows(store, embedder):
    """The realistic steady state: the worker is partway through a 372k-row
    backlog. Rows with ``embedding_state IS NULL`` have no vector, so only
    FTS5 can reach them — and it still must."""
    _seed_sessions(
        store,
        [("o1", "quadratic voting", 1), ("o2", "voting reform proposal", 2)],
    )
    _embed(store, "observations", [("o1", "quadratic voting")])
    unembedded = store.count_embedding_states("observations")["pending"]
    assert unembedded == 1, "precondition: corpus is half embedded"

    result = aggregator_query("voting", _store=store)
    assert result["ok"] is True
    assert {"s-o1", "s-o2"} <= _ids(result)


# --- degradation: the vector arm is broken -----------------------------------


def test_missing_extension_falls_through_to_fts_and_never_reaches_the_caller(
    no_vec_store, embedder
):
    """``VectorIndexUnavailableError`` must not escape to the tool boundary.
    A caller asking a keyword question on a machine with no sqlite-vec gets
    keyword answers, not an error."""
    _seed_sessions(no_vec_store, [("o1", "quadratic voting", 1)])
    result = aggregator_query("voting", _store=no_vec_store)
    assert result["ok"] is True
    assert result["total"] == 1
    assert embedder.query_calls == 0


def test_a_dead_embedder_degrades_to_fts_rather_than_failing_the_query(
    store, monkeypatch
):
    _seed_sessions(store, [("o1", "quadratic voting", 1)])
    _embed(store, "observations", [("o1", "quadratic voting")])
    monkeypatch.setattr("aggregator.mcp._get_embedder", ExplodingEmbedder)
    result = aggregator_query("voting", _store=store)
    assert result["ok"] is True
    assert result["total"] == 1


# --- rerank ------------------------------------------------------------------


def test_rerank_reorders_the_page(store, embedder, monkeypatch):
    _seed_sessions(
        store,
        [("o1", "voting alpha", 1), ("o2", "voting beta", 2), ("o3", "voting gamma", 3)],
    )
    reranker = StubReranker(prefer="alpha")
    monkeypatch.setattr("aggregator.mcp._get_reranker", lambda: reranker)

    plain = aggregator_query("voting", fields="full", _store=store)
    reranked = aggregator_query("voting", fields="full", rerank=True, _store=store)

    assert reranker.calls == 1
    assert plain["ok"] is reranked["ok"] is True
    assert _ids(plain) == _ids(reranked), "rerank reorders, it must not refilter"
    assert "alpha" in reranked["records"][0]["subject"]
    assert plain["records"][0]["subject"] != reranked["records"][0]["subject"]


def test_rerank_defaults_to_off_and_never_loads_the_model(
    store, embedder, monkeypatch
):
    _seed_sessions(store, [("o1", "voting alpha", 1)])
    called = []
    monkeypatch.setattr(
        "aggregator.mcp._get_reranker", lambda: called.append(1) or StubReranker("x")
    )
    aggregator_query("voting", _store=store)
    assert called == []


def test_a_dead_reranker_returns_the_unreranked_page_rather_than_an_error(
    store, embedder, monkeypatch
):
    """Rerank is a nice-to-have on top of a result the caller already has.
    Losing it must cost the ordering, never the answer."""
    _seed_sessions(store, [("o1", "voting alpha", 1), ("o2", "voting beta", 2)])
    monkeypatch.setattr("aggregator.mcp._get_reranker", ExplodingReranker)
    result = aggregator_query("voting", rerank=True, _store=store)
    assert result["ok"] is True
    assert result["total"] == 2


def test_rerank_on_a_filter_only_query_loads_neither_model(
    store, embedder, monkeypatch
):
    """There is no query string to score a document against, so a
    cross-encoder has nothing to do. ``rerank=True`` on a pure-filter query
    must therefore cost nothing — not the 2 GB reranker, not the embedder."""
    _seed_sessions(store, [("o1", "quadratic voting", 1)])
    _embed(store, "observations", [("o1", "quadratic voting")])
    loaded = []
    monkeypatch.setattr(
        "aggregator.mcp._get_reranker",
        lambda: loaded.append(1) or StubReranker("x"),
    )
    result = aggregator_query("source:sessions", rerank=True, _store=store)
    assert result["ok"] is True
    assert loaded == []
    assert embedder.query_calls == 0


def test_rerank_works_on_the_records_ontology(store, embedder, monkeypatch):
    store.upsert(
        [
            _record("github:1", "voting alpha", "body about voting", 1),
            _record("github:2", "voting beta", "body about voting", 2),
        ]
    )
    reranker = StubReranker(prefer="alpha")
    monkeypatch.setattr("aggregator.mcp._get_reranker", lambda: reranker)
    result = aggregator_query(
        "source:github voting", fields="full", rerank=True, _store=store
    )
    assert result["ok"] is True
    assert "alpha" in result["records"][0]["subject"]


def test_rerank_on_a_drilldown_query(store, embedder, monkeypatch):
    _seed_sessions(store, [("o1", "voting alpha", 1), ("o2", "voting beta", 2)])
    reranker = StubReranker(prefer="beta")
    monkeypatch.setattr("aggregator.mcp._get_reranker", lambda: reranker)
    result = aggregator_query(
        "source:sessions voting", fields="full", drilldown=True, rerank=True,
        _store=store,
    )
    assert result["ok"] is True
    assert result["mode"] == "observations"
    assert reranker.calls == 1


def test_rerank_on_an_empty_page_does_not_call_the_model(
    store, embedder, monkeypatch
):
    _seed_sessions(store, [("o1", "voting alpha", 1)])
    reranker = StubReranker(prefer="alpha")
    monkeypatch.setattr("aggregator.mcp._get_reranker", lambda: reranker)
    result = aggregator_query("nonexistentterm", rerank=True, _store=store)
    assert result["ok"] is True
    assert reranker.calls == 0


# --- the other two routing paths --------------------------------------------


def test_hybrid_runs_on_the_records_path(store, embedder):
    store.upsert([_record("github:1", "quadratic voting", "voting body", 1)])
    _embed(store, "records", [("github:1", "quadratic voting")])
    result = aggregator_query("source:github governance", _store=store)
    assert result["ok"] is True
    assert result["mode"] == "records"
    assert "github:1" in _ids(result)


def test_hybrid_handles_record_ids_that_end_in_a_colon_and_digits(store, embedder):
    """A trap in the id format. The embed worker suffixes MULTI-chunk rows as
    ``<id>:<n>``, so a naive ``split(':')`` looks like the way to recover the
    base id — but ``stable_id_for`` mints ``github:<owner>/<repo>:<number>``,
    which already ends in ``:<digits>``. Stripping a suffix that was never
    added turns ``github:acme/api:42`` into ``github:acme/api`` and the row
    becomes unreachable through the vector arm. Every GitHub PR and issue in
    the cache has this shape.
    """
    store.upsert(
        [_record("github:acme/api:42", "quadratic voting", "voting body", 1)]
    )
    _embed(store, "records", [("github:acme/api:42", "quadratic voting")])
    result = aggregator_query("source:github governance", _store=store)
    assert result["ok"] is True
    assert "github:acme/api:42" in _ids(result)


def test_hybrid_runs_on_the_union_path(store, embedder):
    """A bare text query hits both ontologies. Both get their vector arm."""
    store.upsert([_record("github:1", "quadratic voting", "voting body", 1)])
    _embed(store, "records", [("github:1", "quadratic voting")])
    _seed_sessions(store, [("o1", "quadratic voting notes", 2)])
    _embed(store, "observations", [("o1", "quadratic voting notes")])
    result = aggregator_query("governance", _store=store)
    assert result["ok"] is True
    assert result["mode"] == "union"
    assert {"github:1", "s-o1"} <= _ids(result)


def test_routing_does_not_run_the_linear_vec_row_count(store, embedder):
    """``count_vec_rows`` is O(n) over the vec0 table — ~70 ms at the live
    cache's 400k vectors. Deciding hybrid-vs-FTS5 happens on every text query
    and on both ontologies, so it must not be the thing that asks. Booby-trap
    the expensive call and require the query to succeed anyway.
    """
    _seed_sessions(store, [("o1", "quadratic voting", 1)])
    _embed(store, "observations", [("o1", "quadratic voting")])

    def _too_slow(kind):
        raise AssertionError(
            f"routing called the linear count_vec_rows({kind!r})"
        )

    store.count_vec_rows = _too_slow
    result = aggregator_query("governance", _store=store)
    assert result["ok"] is True
    assert "s-o1" in _ids(result)


def test_the_union_path_embeds_the_query_exactly_once(store, embedder):
    """Union is the DEFAULT path — a bare text query with no ``source:`` key
    lands here — and it drives two ontologies with two separate vector
    tables. The embedding is the same vector for both, and embedding a query
    with Qwen3 on CPU costs real milliseconds, so computing it once per
    ontology doubles the cost of the most common search there is.
    """
    store.upsert([_record("github:1", "quadratic voting", "voting body", 1)])
    _embed(store, "records", [("github:1", "quadratic voting")])
    _seed_sessions(store, [("o1", "quadratic voting notes", 2)])
    _embed(store, "observations", [("o1", "quadratic voting notes")])

    result = aggregator_query("governance", _store=store)
    assert result["ok"] is True
    assert embedder.query_calls == 1


def test_hybrid_runs_on_the_drilldown_path(store, embedder):
    _seed_sessions(store, [("o1", "quadratic voting notes", 1)])
    _embed(store, "observations", [("o1", "quadratic voting notes")])
    result = aggregator_query(
        "source:sessions governance", drilldown=True, _store=store
    )
    assert result["ok"] is True
    assert result["mode"] == "observations"
    assert {r["obs_id"] for r in result["records"]} == {"o1"}


# --- the contract the surface must keep --------------------------------------


def test_hybrid_does_not_change_the_result_shape(store, embedder):
    _seed_sessions(store, [("o1", "quadratic voting", 1)])
    plain = aggregator_query("voting", fields="full", _store=store)
    _embed(store, "observations", [("o1", "quadratic voting")])
    hybrid = aggregator_query("voting", fields="full", _store=store)
    assert set(plain) == set(hybrid)
    assert set(plain["records"][0]) == set(hybrid["records"][0])
    assert plain["mode"] == hybrid["mode"]


def test_hybrid_still_wraps_bodies_in_external_content(store, embedder):
    """The untrusted-data delimiters are a security invariant, not a
    formatting choice — they must survive whichever arm answered."""
    _seed_sessions(store, [("o1", "quadratic voting", 1)])
    _embed(store, "observations", [("o1", "quadratic voting")])
    result = aggregator_query("governance", fields="full", _store=store)
    assert "<ExternalContent" in result["records"][0]["content"]


def test_hybrid_respects_the_dsl_filters(store, embedder):
    """The fused id set narrows the candidates; it must never smuggle a row
    past a filter the caller asked for."""
    _seed_sessions(
        store, [("o1", "quadratic voting", 1), ("o2", "quadratic voting", 20)]
    )
    _embed(
        store,
        "observations",
        [("o1", "quadratic voting"), ("o2", "quadratic voting")],
    )
    result = aggregator_query(
        "source:sessions governance from:2026-07-15", _store=store
    )
    assert _ids(result) == {"s-o2"}


# --- pagination: the one real hazard hybrid introduces -----------------------


def test_page_token_pins_the_retrieval_arm_across_a_backfill_landing(
    store, embedder
):
    """THE HAZARD. Page 1 is served FTS5-only because the vector index is
    empty. The embed timer then lands rows between the two calls. Without a
    pinned arm, page 2 is computed over a LARGER candidate set, so the offset
    the caller was handed no longer points where it pointed — rows shift and
    the caller silently sees some twice and, in the general case, misses
    others. Backfill landing mid-pagination is not an edge case here: the
    worker is on a timer and the corpus is 372k rows.

    The fix: the page token records which arm minted it, and a continuation
    reproduces that arm.
    """
    docs = [(f"o{i}", f"voting note {i}", 1 + i) for i in range(4)]
    _seed_sessions(store, docs)

    page1 = aggregator_query("voting", page_size=2, _store=store)
    assert page1["total"] == 4
    token = page1["next_page_token"]

    # The backfill lands, and it adds a row only the vector arm can reach.
    _seed_sessions(store, [("p1", "pigeons roost on ledges", 30)])
    _embed(store, "observations", [("p1", "pigeons roost on ledges")])

    page2 = aggregator_query("voting", page_size=2, page_token=token, _store=store)
    assert page2["ok"] is True
    seen = list(_ids(page1)) + list(_ids(page2))
    assert len(seen) == len(set(seen)), f"duplicate rows across pages: {seen}"
    assert set(seen) == {f"s-o{i}" for i in range(4)}


def test_a_fresh_query_after_backfill_does_use_the_vector_arm(store, embedder):
    """The other half of the pinning rule: pinning must not make hybrid
    unreachable. A caller who starts a NEW query gets the warm arm."""
    _seed_sessions(store, [("o1", "quadratic voting", 1)])
    _embed(store, "observations", [("o1", "quadratic voting")])
    result = aggregator_query("governance", _store=store)
    assert result["total"] == 1


def test_a_hybrid_page_token_keeps_the_vector_arm_on(store, embedder):
    docs = [("o1", "quadratic voting one", 1), ("o2", "quadratic voting two", 2)]
    _seed_sessions(store, docs)
    _embed(store, "observations", docs and [(i, b) for i, b, _ in docs])
    page1 = aggregator_query("governance", page_size=1, _store=store)
    assert page1["total"] == 2
    page2 = aggregator_query(
        "governance", page_size=1, page_token=page1["next_page_token"], _store=store
    )
    assert page2["ok"] is True
    assert _ids(page1) | _ids(page2) == {"s-o1", "s-o2"}


def test_legacy_integer_page_tokens_are_still_accepted(store, embedder):
    """Tokens are opaque, but a plain integer is what every pre-v5 page
    handed out. One must still page correctly rather than resetting to 0."""
    docs = [(f"o{i}", f"voting note {i}", 1 + i) for i in range(4)]
    _seed_sessions(store, docs)
    page2 = aggregator_query("voting", page_size=2, page_token="2", _store=store)
    assert page2["ok"] is True
    assert len(page2["records"]) == 2


def test_a_garbage_page_token_still_falls_back_to_the_first_page(
    store, embedder
):
    _seed_sessions(store, [("o1", "quadratic voting", 1)])
    result = aggregator_query("voting", page_token="not-a-token", _store=store)
    assert result["ok"] is True
    assert result["total"] == 1
