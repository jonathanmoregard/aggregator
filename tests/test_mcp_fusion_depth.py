"""Criterion G: each arm is retrieved to ``FUSION_ARM_DEPTH`` before fusion.

WHY THIS IS A TEST AND NOT A CONSTANT NOBODY CHECKS. The depth is the one
parameter that bounds what RRF is able to do: below roughly 50 per arm too few
documents appear in both lists, the cross-arm agreement signal never fires, and
the fused ranking is two single-arm rankings glued together. A regression here
is invisible — the pipeline still returns results, still returns them in a
plausible order, and only the quality moves.

THE ASYMMETRY IS DELIBERATE AND IS ALSO PINNED HERE. "150 per arm" is a FLOOR,
and the keyword arm is uncapped, which satisfies it the other way: capping FTS5
at 150 would make a warm vector index REMOVE keyword matches the same query
returned yesterday, and the superset invariant this file's sibling
(``test_mcp_hybrid.py``) exists to protect is worth more than a symmetric
number.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from aggregator.core.hybrid import FUSION_ARM_DEPTH
from aggregator.core.store import Store
from aggregator.mcp import _VECTOR_ARM_K, aggregator_query
from aggregator.sources.base import ObservationRow, Record, SessionRow

_TS = datetime(2026, 7, 1, 8, 0, tzinfo=UTC)


class _StubEmbedder:
    def embed_query(self, query: str) -> np.ndarray:
        v = np.zeros(768, dtype=np.float32)
        v[0] = 1.0
        return v

    def embed_documents(self, docs: list[str]) -> np.ndarray:
        return np.array([self.embed_query(d) for d in docs], dtype=np.float32)


@pytest.fixture
def store(tmp_path):
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    return s


@pytest.fixture
def embedder(monkeypatch):
    stub = _StubEmbedder()
    monkeypatch.setattr("aggregator.mcp._get_embedder", lambda: stub)
    return stub


def _seed(store: Store) -> None:
    sid = "s1"
    store.upsert_entities(
        [
            SessionRow(
                session_id=sid, root_session_id=sid, parent_session_id=None,
                kind="session", agent_id=None, agent_type=None,
                spawned_by_tool_use_id=None, cwd="/x", git_branch="main",
                first_ts=_TS, last_ts=_TS, jsonl_path="/tmp/s1.jsonl",
            ),
            ObservationRow(
                obs_id="o1", session_id=sid, root_session_id=sid,
                parent_obs_id=None, type="user", ts=_TS, model=None,
                input_tokens=None, output_tokens=None, tool_name=None,
                tool_use_id=None, body="quadratic voting is a governance idea",
            ),
        ]
    )
    store.upsert(
        [
            Record(
                stable_id="github:acme/api:1", source="github",
                subject="voting", body="quadratic voting", tags=[],
                created_at=_TS, updated_at=_TS,
            )
        ]
    )
    vec = _StubEmbedder().embed_documents(["quadratic voting"])
    store.upsert_vec_observations([("o1", vec[0])])
    store.mark_embedded("observations", ["o1"], state="ok")
    store.upsert_vec_records([("github:acme/api:1", vec[0])])
    store.mark_embedded("records", ["github:acme/api:1"], state="ok")


def test_the_mcp_vector_arm_uses_the_shared_fusion_depth(store, embedder):
    """One constant, two call sites. ``aggregator.mcp`` used to carry its own
    50 and the eval harness its own 150, so the pipeline being measured was
    not the pipeline being served."""
    assert _VECTOR_ARM_K == FUSION_ARM_DEPTH


@pytest.mark.parametrize(
    ("dsl", "method"),
    [
        ("source:sessions voting", "_vec_obs_ids"),
        ("source:github voting", "_vec_record_ids"),
    ],
)
def test_the_vector_arm_is_asked_for_150_candidates(
    store, embedder, monkeypatch, dsl, method
):
    seen: list[int] = []
    original = getattr(store, method)

    def _spy(embedding, k):
        seen.append(k)
        return original(embedding, k)

    _seed(store)
    monkeypatch.setattr(store, method, _spy)
    result = aggregator_query(dsl, _store=store)
    assert result["ok"] is True, result
    assert seen == [FUSION_ARM_DEPTH]


def test_the_keyword_arm_is_never_truncated(store, embedder, monkeypatch):
    """"150 per arm" is a floor, and an uncapped arm satisfies it. Pinned
    because the obvious symmetric edit — slice the FTS5 arm to the same depth —
    would silently delete keyword matches a warm vector index must never cost.
    """
    seen: list[int] = []
    original = store._fts_obs_ids

    def _spy(text):
        out = original(text)
        seen.append(len(out))
        return out

    entities: list = []
    for i in range(FUSION_ARM_DEPTH + 25):
        sid = f"s{i}"
        entities.append(
            SessionRow(
                session_id=sid, root_session_id=sid, parent_session_id=None,
                kind="session", agent_id=None, agent_type=None,
                spawned_by_tool_use_id=None, cwd="/x", git_branch="main",
                first_ts=_TS, last_ts=_TS, jsonl_path=f"/tmp/{sid}.jsonl",
            )
        )
        entities.append(
            ObservationRow(
                obs_id=f"o{i}", session_id=sid, root_session_id=sid,
                parent_obs_id=None, type="user", ts=_TS, model=None,
                input_tokens=None, output_tokens=None, tool_name=None,
                tool_use_id=None, body=f"quadratic voting note {i}",
            )
        )
    store.upsert_entities(entities)
    vec = _StubEmbedder().embed_documents(["quadratic voting"])
    store.upsert_vec_observations([("o0", vec[0])])
    store.mark_embedded("observations", ["o0"], state="ok")

    monkeypatch.setattr(store, "_fts_obs_ids", _spy)
    result = aggregator_query("source:sessions voting", _store=store)
    assert result["ok"] is True, result
    assert seen and seen[0] > FUSION_ARM_DEPTH
