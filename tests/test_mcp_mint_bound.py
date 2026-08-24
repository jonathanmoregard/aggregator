"""The server must not mint a page token it will itself refuse.

``_MAX_FROZEN_ID_CHARS`` derives the payload budget from "the longest id we
mint, a dropbox path" — an assumption about today's sources, not a bound
anything enforces. Round 2 taught ``_unpack_frozen`` to reject an over-budget
payload, which is right, and left ``_pack_frozen`` and ``_mint_page_token``
asserting nothing at all. So the failure surfaced one round trip away from its
cause: page 1 succeeded and handed back a token, and page 2 refused it, from
the same process, for a payload that process had just produced.

That is the worst place to find out. The caller did nothing wrong, the
remediation it is handed ("pass back the exact token, unmodified") is exactly
what it did, and the pagination cannot be completed by any means available to
it.

The ids here are ~2 000 characters — four times the derived per-id budget, well
inside PATH_MAX, and entirely reachable for a deeply nested dropbox path. They
are also highly compressible, which is the sharp case rather than the lenient
one: the packed token sails past the base64 length gate and only dies at the
inflate gate, so nothing about its size is visible at mint time unless mint
time goes looking.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from aggregator.core.store import Store
from aggregator.mcp import (
    _MAX_FROZEN_ID_CHARS,
    _VECTOR_ARM_K,
    _mint_page_token,
    _parse_page_token,
    aggregator_query,
)
from aggregator.sources.base import ObservationRow, SessionRow

#: Past the derived per-id budget, under PATH_MAX. A dropbox record's id is
#: its filesystem path, so this shape is a deployment away, not a fantasy.
_LONG_ID_CHARS = 2000


class StubEmbedder:
    """No real model is named anywhere in this suite, on purpose."""

    def embed_query(self, query):
        return np.zeros(768, dtype=np.float32)

    def embed_documents(self, docs):
        return np.zeros((len(docs), 768), dtype=np.float32)


@pytest.fixture
def store(tmp_path):
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    return s


@pytest.fixture(autouse=True)
def embedder(monkeypatch):
    monkeypatch.setattr("aggregator.mcp._get_embedder", StubEmbedder)


def _long_id(i: int) -> str:
    padding = "a" * (_LONG_ID_CHARS - 40)
    return f"dropbox:/Archive/{padding}/note-{i}.md"


def _seed_long_ids(store: Store, n: int = _VECTOR_ARM_K) -> None:
    base = datetime(2026, 7, 1, tzinfo=UTC)
    entities: list[object] = []
    docs = []
    for i in range(n):
        ts = base + timedelta(minutes=i)
        obs_id = _long_id(i)
        entities.append(
            SessionRow(
                session_id=f"s{i}",
                root_session_id=f"s{i}",
                parent_session_id=None,
                kind="session",
                agent_id=None,
                agent_type=None,
                spawned_by_tool_use_id=None,
                cwd="/x",
                git_branch="main",
                first_ts=ts,
                last_ts=ts,
                jsonl_path=f"/tmp/s{i}.jsonl",
            )
        )
        entities.append(
            ObservationRow(
                obs_id=obs_id,
                session_id=f"s{i}",
                root_session_id=f"s{i}",
                parent_obs_id=None,
                type="user",
                ts=ts,
                model=None,
                input_tokens=None,
                output_tokens=None,
                tool_name=None,
                tool_use_id=None,
                body=f"governance voting note {i}",
            )
        )
        docs.append(obs_id)
    store.upsert_entities(entities)
    vecs = np.zeros((len(docs), 768), dtype=np.float32)
    store.upsert_vec_observations(list(zip(docs, vecs, strict=True)))
    store.mark_embedded("observations", docs, state="ok")


def test_the_server_never_mints_a_token_it_will_refuse(store):
    """THE REPRO. Page 1 mints; page 2 refuses what page 1 minted."""
    _seed_long_ids(store)

    page1 = aggregator_query("governance", page_size=1, _store=store)
    assert page1["ok"] is True, page1
    token = page1["next_page_token"]

    page2 = aggregator_query(
        "governance", page_size=1, page_token=token, _store=store
    )

    assert page2["ok"] is True, (
        f"the server refused a {len(token)} character token it had just "
        f"minted: {page2.get('reason')}"
    )
    assert page2["records"], page2


def test_the_over_budget_page_says_so_rather_than_degrading_quietly(store):
    """Dropping the frozen hits re-opens the drift round 1 closed.

    Correct-but-drifty beats a pagination the caller cannot finish, so the
    token is still minted — with the arm and the query pinned, and the KNN
    re-run. What must not happen is that trade being made silently.
    """
    _seed_long_ids(store)
    page1 = aggregator_query("governance", page_size=1, _store=store)

    notice = page1.get("notice", "")
    assert "page token" in notice.lower(), (
        f"the frozen hits were dropped with nothing said about it: {notice!r}"
    )
    assert str(_MAX_FROZEN_ID_CHARS) in notice, (
        f"the notice does not name the budget that was blown: {notice!r}"
    )


def test_a_normal_corpus_still_freezes_its_hits(store, monkeypatch):
    """The bound must not cost the freeze on ids of a sane length.

    Same corpus, same query, ordinary ids: the payload is still there and the
    response carries no degradation notice.
    """
    monkeypatch.setattr(
        "tests.test_mcp_mint_bound._long_id", lambda i: f"dropbox:/notes/{i}.md"
    )
    _seed_long_ids(store)

    page1 = aggregator_query("governance", page_size=1, _store=store)

    assert "." in page1["next_page_token"], (
        f"the frozen hits were dropped from a perfectly ordinary token: "
        f"{page1['next_page_token']!r}"
    )
    assert "page token" not in page1.get("notice", "").lower()


def test_mint_and_parse_agree_for_every_id_length(store):
    """The invariant, stated directly: anything mint produces, parse accepts.

    Sizes chosen either side of the derived budget and far past it, all at the
    full ``_VECTOR_ARM_K`` count on both ontologies — the worst legitimate
    case round 2's ceiling was derived against, plus three that break it.
    """
    for id_chars in (16, _MAX_FROZEN_ID_CHARS, _MAX_FROZEN_ID_CHARS + 1, 2000, 60000):
        frozen = {
            kind: [
                f"{i}".rjust(id_chars, "a")  # exactly id_chars long, all distinct
                for i in range(_VECTOR_ARM_K)
            ]
            for kind in ("observations", "records")
        }
        token = _mint_page_token(7, True, "fingerprint01", frozen)
        cursor = _parse_page_token(token)  # must not raise
        assert cursor.offset == 7
        assert cursor.hybrid is True
        if id_chars <= _MAX_FROZEN_ID_CHARS:
            assert cursor.frozen == frozen, (
                f"a legitimate {id_chars}-char id family lost its freeze"
            )
        else:
            assert cursor.frozen is None, (
                f"{id_chars}-char ids were minted into a token anyway"
            )


def test_more_hits_than_the_arm_can_return_are_never_minted(store):
    """The count ceiling holds at mint too, not only at parse."""
    frozen = {"observations": [f"o{i}" for i in range(_VECTOR_ARM_K + 1)]}
    token = _mint_page_token(7, True, "fingerprint01", frozen)
    assert _parse_page_token(token).frozen is None
