"""The search callables the harness measures.

The harness is deliberately id-list-in: it takes a ``(query, limit) -> [ids]``
callable and knows nothing about how the ids were produced. That is what lets
the same golden set and the same drift metric measure a lexical-only arm today
and the fused pipeline once criterion H exposes the modes.

ONE RULE THESE FUNCTIONS DO NOT SHARE WITH THE MCP SERVER: they never degrade.
The server degrades to FTS5 when the vector arm is missing, because a user
would rather have keyword results than an error. An eval harness that did the
same would silently measure a different system and report "no drift" — the
exact "empty-result-looks-like-success" failure the project bans.
"""

import sqlite3
from datetime import UTC, datetime

import numpy as np
import pytest

from aggregator.core import store as store_mod
from aggregator.core.store import Store, VectorIndexUnavailableError
from aggregator.evals.search import (
    SEARCH_MODES,
    hybrid_search_fn,
    lexical_search_fn,
    resolve_search_fn,
)
from aggregator.sources.base import ObservationRow, SessionRow

_DIM = 768
_AXES = {"quadratic": 0, "pigeon": 1}


class StubEmbedder:
    """Deterministic keyword->axis embedder. Never names a real model."""

    @staticmethod
    def _vec_for(text: str) -> np.ndarray:
        v = np.zeros(_DIM, dtype=np.float32)
        lowered = (text or "").lower()
        for word, axis in _AXES.items():
            if word in lowered:
                v[axis] = 1.0
                return v
        v[2] = 1.0
        return v

    def embed_query(self, query: str) -> np.ndarray:
        return self._vec_for(query)


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


def _seed(store, docs):
    entities = []
    for obs_id, body, day in docs:
        sid = f"s-{obs_id}"
        entities.append(_session(sid, day=day))
        entities.append(_obs(obs_id, sid, body, day=day))
    store.upsert_entities(entities)


DOCS = [
    ("o1", "quadratic voting is the governance mechanism we picked", 1),
    ("o2", "quadratic funding differs from quadratic voting", 2),
    ("o3", "the pigeon will roost on the balcony", 3),
]


@pytest.fixture
def store(tmp_path):
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    _seed(s, DOCS)
    return s


@pytest.fixture
def no_vec_store(tmp_path, monkeypatch):
    def _boom(conn):
        raise sqlite3.OperationalError("simulated sqlite-vec ABI mismatch")

    monkeypatch.setattr(store_mod, "_load_sqlite_vec", _boom)
    monkeypatch.setattr(store_mod, "_VEC_LOAD_WARNED", False)
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    assert s.vector_available is False
    _seed(s, DOCS)
    return s


# --- lexical ----------------------------------------------------------------


def test_lexical_search_returns_the_observations_that_match(store):
    hits = lexical_search_fn(store)("quadratic", 50)
    assert set(hits) == {"o1", "o2"}


def test_lexical_search_respects_the_limit(store):
    assert len(lexical_search_fn(store)("quadratic", 1)) == 1


def test_lexical_search_returns_nothing_for_an_absent_term(store):
    assert lexical_search_fn(store)("axolotl", 50) == []


def test_a_query_fts5_could_not_parse_now_searches_for_its_words(store):
    """Criterion B, seen from here, AFTER b4eab9b.

    This assertion used to be ``== []`` and the reason was the bug: ``power-on``
    raised ``no such column: on`` and the store abstained. Now
    ``fts5_match_query`` rewrites it to ``"power" "on"``, so it is an ordinary
    two-term search — and the corpus is asked a real question rather than
    handed a syntax error. Seeded so the answer is non-empty, because an empty
    list here would pass under either regime and would therefore be worth
    nothing as evidence.
    """
    _seed(store, [("o4", "the power-on self test writes to the console", 4)])
    assert lexical_search_fn(store)("power-on", 50) == ["o4"]


def test_lexical_search_needs_no_vector_index(no_vec_store):
    assert set(lexical_search_fn(no_vec_store)("quadratic", 50)) == {"o1", "o2"}


# --- hybrid -----------------------------------------------------------------


def _embed_all(store, embedder, docs):
    store.upsert_vec_observations(
        [(obs_id, embedder._vec_for(body)) for obs_id, body, _ in docs]
    )


def test_hybrid_search_fuses_both_arms(store):
    embedder = StubEmbedder()
    _embed_all(store, embedder, DOCS)
    hits = hybrid_search_fn(store, embedder)("pigeon", 50)
    assert "o3" in hits


def test_hybrid_search_returns_a_relevance_ordering_not_a_set(store):
    """RRF's ranking is kept here. The MCP path discards it; the eval needs it."""
    embedder = StubEmbedder()
    _embed_all(store, embedder, DOCS)
    hits = hybrid_search_fn(store, embedder)("quadratic", 50)
    assert hits[0] in {"o1", "o2"}
    assert len(hits) == len(set(hits)), "fusion must not emit duplicates"


def test_hybrid_search_refuses_to_run_without_a_vector_index(no_vec_store):
    """Never degrade silently: that would measure the lexical arm and say hybrid."""
    with pytest.raises(VectorIndexUnavailableError):
        hybrid_search_fn(no_vec_store, StubEmbedder())("quadratic", 50)


# --- the swallow that outlived the bug it was for ---------------------------
#
# ``hybrid_search_fn`` caught ``sqlite3.OperationalError`` from the FTS5 arm
# and fused the vector arm alone, under a comment naming "Criterion B's bug: an
# unescaped MATCH". b4eab9b fixed that bug at the five MATCH binding sites, so
# no user text can produce a syntax error any more — which does NOT make the
# except clause harmless. It makes it a trap: the only OperationalErrors left
# are a locked cache, a corrupt index, and an FTS5 that changed under us, and
# every one of those is a reason to stop rather than a reason to quietly
# measure half the pipeline and file it under "hybrid". That is the module
# docstring's own NEVER DEGRADE rule, violated by the module.
#
# Both halves are asserted below, because deleting a swallow on the strength of
# "it cannot trigger" needs the "cannot" demonstrated, not asserted.


def test_no_frozen_golden_query_can_make_the_lexical_arm_raise(store):
    """The swallow's STATED cause, checked against the whole frozen set.

    86 real queries, 25 of which used to raise. Run through the shipped
    ``_fts_obs_ids`` — the same call ``hybrid_search_fn`` makes — against a
    real index. Anything that raises here is a syntax error the whitelist
    missed, and would mean the except clause still has work to do.
    """
    from aggregator.evals.golden import load_golden_queries

    raised: list[tuple[str, str]] = []
    for query in load_golden_queries():
        try:
            store._fts_obs_ids(query.query)
        except sqlite3.OperationalError as e:  # pragma: no cover - the assertion
            raised.append((query.id, str(e)))
    assert raised == [], (
        "the FTS5 arm still raises for frozen golden queries, so the swallow "
        f"in evals/search.py is not dead after all: {raised}"
    )


def test_a_broken_lexical_arm_is_not_quietly_measured_as_hybrid(store):
    """THE REPRO for what the swallow would do now that the bug is gone.

    A locked cache raises ``OperationalError`` exactly like a malformed MATCH
    used to. Swallowed, the run fuses the vector arm alone, reports a full
    result list and drifts by however much the missing arm was contributing —
    an eval that measured a different system and said nothing.
    """
    embedder = StubEmbedder()
    _embed_all(store, embedder, DOCS)

    def locked(_text):
        raise sqlite3.OperationalError("database is locked")

    store._fts_obs_ids = locked  # type: ignore[method-assign]

    with pytest.raises(sqlite3.OperationalError, match="locked"):
        hybrid_search_fn(store, embedder)("quadratic", 50)


# --- mode resolution --------------------------------------------------------


def test_the_modes_on_offer_are_named(store):
    assert set(SEARCH_MODES) == {"lexical", "hybrid"}


def test_resolving_an_unknown_mode_fails_loudly(store):
    with pytest.raises(ValueError, match="telepathy"):
        resolve_search_fn("telepathy", store)


def test_resolving_hybrid_without_an_embedder_fails_loudly(store):
    with pytest.raises(ValueError, match="embedder"):
        resolve_search_fn("hybrid", store)


def test_resolving_lexical_gives_a_working_callable(store):
    assert set(resolve_search_fn("lexical", store)("quadratic", 50)) == {"o1", "o2"}
