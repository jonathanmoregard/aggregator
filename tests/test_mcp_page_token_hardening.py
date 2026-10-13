"""A page token is caller-controlled input, and it is parsed like one.

``page_token`` is the only argument on this surface that the server itself
minted, so it reads like server state — and it is not. It arrives over MCP
from whatever is driving the tool. Every malformation used to resolve to
"offset 0": a caller that mangled a token silently re-read page 1 while
believing it had advanced, which is the silent-data-loss shape the project's
fail-loudly rule exists to forbid.

The contract these tests pin: a token this server would not have minted is
refused with a structured error naming the remedy, never silently restarted.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from aggregator.core.store import Store
from aggregator.mcp import _mint_page_token, _parse_page_token, aggregator_query
from aggregator.sources.base import Record


@pytest.fixture
def store(tmp_path):
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    return s


def _seed_records(store, n=3):
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


def test_a_mangled_token_is_refused_not_silently_restarted(store):
    """THE REPRO. A caller that corrupts a token in transit re-read page 1
    while believing it had advanced — and nothing in the response said so."""
    _seed_records(store)
    page1 = aggregator_query("source:github", page_size=1, _store=store)
    token = page1["next_page_token"]
    assert token

    result = aggregator_query(
        "source:github", page_size=1, page_token="x" + token, _store=store
    )

    assert result["ok"] is False, (
        f"page 1 served again as if it were page 2: {result.get('records')}"
    )
    assert "page_token" in result["reason"]
    assert result["remediation"]


def test_a_token_that_is_not_a_number_is_refused(store):
    result = aggregator_query(
        "source:github", page_token="not-a-real-token", _store=store
    )
    assert result["ok"] is False
    assert "page_token" in result["reason"]


def test_a_token_from_the_retired_hybrid_format_is_refused(store):
    """``h40~<fp>.<payload>`` was minted while a vector arm existed. It is not
    a page of anything this server can serve, so it is refused rather than
    read as some other offset."""
    result = aggregator_query(
        "source:github", page_token="h0~fingerprint01.eJwLz0kt", _store=store
    )
    assert result["ok"] is False
    assert "page_token" in result["reason"]


def test_the_tokens_this_server_mints_still_page_normally(store):
    """The refusal must not cost the feature it protects."""
    _seed_records(store)
    seen: list[str] = []
    token = None
    for _ in range(5):
        page = aggregator_query(
            "source:github", page_size=1, page_token=token, _store=store
        )
        assert page["ok"] is True, page
        seen += [r["stable_id"] for r in page["records"]]
        token = page.get("next_page_token")
        if not token:
            break
    assert sorted(seen) == [f"github:acme/api:{i}" for i in range(3)]


def test_a_bare_offset_is_still_accepted(store):
    """A bare offset carries only a position, which a caller could type anyway."""
    _seed_records(store)
    page = aggregator_query("source:github", page_size=1, page_token="0", _store=store)
    assert page["ok"] is True, page


def test_mint_and_parse_round_trip():
    cursor = _parse_page_token(_mint_page_token(40, "fingerprint01"))
    assert cursor.offset == 40
    assert cursor.fingerprint == "fingerprint01"
