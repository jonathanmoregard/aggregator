"""A page token is only meaningful to the query that minted it.

An offset indexes one ordered result set. Handed back with a different query —
a changed text, a narrowed filter, a drilldown flip — it addresses a position
in a set that never existed for that query, and the caller (an LLM driving an
MCP tool) cannot tell a plausible wrong page from a correct one. So a
wrong-but-plausible page is strictly worse than an error.

The contract these tests pin: every token carries a fingerprint of the query
that minted it, and a token whose fingerprint disagrees with the current call
is refused structurally.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from aggregator.core.store import Store
from aggregator.mcp import aggregator_query
from aggregator.sources.base import ObservationRow, Record, SessionRow

_PER_TOPIC = 30


@pytest.fixture
def store(tmp_path):
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    return s


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
    """Two disjoint topics, the voting one strictly NEWER than the pigeon one."""
    base = datetime(2026, 7, 1, tzinfo=UTC)
    docs: list[tuple[str, str, datetime]] = []
    for i in range(_PER_TOPIC):
        docs.append((f"p{i}", f"pigeon roost note {i}", base + timedelta(minutes=i)))
    for i in range(_PER_TOPIC):
        docs.append((f"v{i}", f"voting note {i}", base + timedelta(days=10, minutes=i)))
    entities = []
    for obs_id, body, ts in docs:
        entities.append(_session(f"s-{obs_id}", ts))
        entities.append(_obs(obs_id, f"s-{obs_id}", body, ts))
    store.upsert_entities(entities)


def _seed_records(store: Store, n: int = 3) -> None:
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


def _ids(result) -> list[str]:
    # ``.get``: a refusal carries no ``records`` key.
    return [r["stable_id"] for r in result.get("records", [])]


def test_a_token_minted_for_one_query_is_refused_by_another(store):
    _seed_two_topics(store)
    token = aggregator_query("voting", page_size=1, _store=store)["next_page_token"]

    reused = aggregator_query("pigeon", page_size=1, page_token=token, _store=store)

    assert reused["ok"] is False, (
        f"a token minted for 'voting' was accepted for 'pigeon': {_ids(reused)}"
    )
    assert "page_token" in reused["reason"], reused
    assert reused["remediation"], reused


def test_the_same_query_still_pages_through_its_own_token(store):
    """The refusal must not cost the feature it protects."""
    _seed_two_topics(store)
    seen: list[str] = []
    token = None
    for _ in range(_PER_TOPIC // 7 + 2):
        page = aggregator_query("voting", page_size=7, page_token=token, _store=store)
        assert page["ok"] is True, page
        seen += _ids(page)
        token = page.get("next_page_token")
        if not token:
            break
    assert token is None, "pagination did not terminate within the corpus"
    assert len(seen) == len(set(seen)), f"row served twice: {seen}"
    assert set(seen) == {f"s-v{i}" for i in range(_PER_TOPIC)}


def test_an_equivalent_dsl_keeps_its_token(store):
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


def test_a_narrowed_filter_invalidates_the_token(store):
    """``source:github`` and ``source:github state:open`` order different row
    sets, so row 1 of the first is not row 1 of the second."""
    _seed_records(store, 3)
    token = aggregator_query("source:github", page_size=1, _store=store)[
        "next_page_token"
    ]
    narrowed = aggregator_query(
        "source:github state:open", page_size=1, page_token=token, _store=store
    )
    assert narrowed["ok"] is False, narrowed
    assert "page_token" in narrowed["reason"], narrowed


def test_a_drilldown_flip_invalidates_the_token(store):
    """``drilldown`` selects WHICH TABLE the offset indexes.

    Session cards and observation rows are different sets of different sizes;
    an offset carried across the flip addresses neither.
    """
    _seed_two_topics(store)
    token = aggregator_query("source:sessions", page_size=1, _store=store)[
        "next_page_token"
    ]
    flipped = aggregator_query(
        "source:sessions",
        page_size=1,
        page_token=token,
        drilldown=True,
        _store=store,
    )
    assert flipped["ok"] is False, flipped
