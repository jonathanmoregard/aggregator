"""A page token is caller-controlled input, and it was trusted like state.

``page_token`` is the only argument on this surface that the server itself
minted, so it reads like server state — and it is not. It arrives over MCP
from whatever is driving the tool, and until this module's fixes it was
decompressed, decoded and believed with no bound anywhere:

* ``zlib.decompress`` with no ``max_length``. Deflate reaches ~1000:1, and
  the MCP server is an in-process child of the user's editor, so a token the
  caller can type expands into the editor's address space until the OOM
  killer picks a victim. Three separate reviewers flagged this one.
* No cap on the number of ids inside, even though the arm that mints them
  returns at most ``_VECTOR_ARM_K``.
* Every malformation resolved to "offset 0, free arm choice" — a caller that
  mangled a token silently re-read page 1 while believing it had advanced.
  That is the exact silent-data-loss shape the project's fail-loudly rule
  exists to forbid, and the caller has no way to notice.
* ``"40.<payload>"`` — no ``h``, but frozen hits present — parsed to
  ``hybrid=False`` while ``pin_for`` pinned the arms ON from ``frozen``. The
  token said two incompatible things and the code quietly followed one.

The contract these tests pin: a token this server would not have minted is
refused with a structured error naming the remedy, never truncated, never
silently restarted.
"""

from __future__ import annotations

import base64
import tracemalloc
import zlib

import pytest

from aggregator.core.store import Store
from aggregator.mcp import (
    _MAX_FROZEN_PAYLOAD_BYTES,
    _VECTOR_ARM_K,
    _mint_page_token,
    _pack_frozen,
    _parse_page_token,
    aggregator_query,
)


@pytest.fixture
def store(tmp_path):
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    return s


@pytest.fixture(autouse=True)
def _no_real_model(monkeypatch):
    """A pinned token engages the vector arm, which would build a real
    Embedder. Nothing here is about embedding quality, and a test that names
    a real model is how an earlier round pulled 15 GB off a CDN."""

    class _StubEmbedder:
        def embed_query(self, query):
            import numpy as np

            return np.zeros(768, dtype=np.float32)

    monkeypatch.setattr("aggregator.mcp._get_embedder", _StubEmbedder)


def _bomb_payload(expanded_bytes: int) -> str:
    """A token whose compressed form is small and whose expansion is not."""
    return (
        base64.urlsafe_b64encode(zlib.compress(b"\0" * expanded_bytes, 9))
        .decode()
        .rstrip("=")
    )


# --- M1: unbounded decompress on caller-supplied input ----------------------


def test_a_decompression_bomb_token_is_never_expanded(store):
    """THE REPRO. 64 MB of expansion from a ~90 KB token, and the pre-fix code
    materialised every byte of it inside the editor's own process."""
    expanded = 64 * 1024 * 1024
    token = "h0~fingerprint01." + _bomb_payload(expanded)

    tracemalloc.start()
    try:
        result = aggregator_query("voting", page_token=token, _store=store)
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 8 * 1024 * 1024, (
        f"the caller-supplied token was expanded: {peak / 1e6:.1f} MB of heap "
        f"for a {len(token)} byte token"
    )
    assert result["ok"] is False
    assert "page_token" in result["reason"]


def test_the_bomb_is_refused_loudly_and_not_truncated_into_a_wrong_page(store):
    """Truncating at the cap would hand back a plausible, wrong frozen set."""
    token = "h40~fingerprint01." + _bomb_payload(8 * 1024 * 1024)
    result = aggregator_query("voting", page_token=token, _store=store)
    assert result["ok"] is False
    assert result["remediation"]


def test_an_oversized_payload_is_rejected_before_it_is_decoded(store):
    """A huge token must cost O(1), not a base64 decode of the whole thing."""
    token = "h0~fingerprint01." + ("A" * (4 * _MAX_FROZEN_PAYLOAD_BYTES))
    result = aggregator_query("voting", page_token=token, _store=store)
    assert result["ok"] is False


def test_more_ids_than_the_vector_arm_can_ever_return_is_refused(store):
    """``_VECTOR_ARM_K`` is the real maximum; the token claimed more."""
    payload = _pack_frozen({"observations": [f"o{i}" for i in range(_VECTOR_ARM_K + 1)]})
    result = aggregator_query("voting", page_token=f"h0~fingerprint01.{payload}", _store=store)
    assert result["ok"] is False
    assert "page_token" in result["reason"]


# --- M3: a token the server cannot read must not look like progress --------


def _seed_records(store, n=3):
    from datetime import UTC, datetime

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


def test_a_mangled_token_is_refused_not_silently_restarted(store):
    """THE REPRO. A caller that corrupts a token in transit re-read page 1
    while believing it had advanced — and nothing in the response said so."""
    _seed_records(store)
    page1 = aggregator_query(
        "source:github", page_size=1, _store=store
    )
    token = page1["next_page_token"]
    assert token

    result = aggregator_query(
        "source:github", page_size=1, page_token=token + "junk", _store=store
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


def test_an_undecodable_payload_is_refused(store):
    result = aggregator_query(
        "source:github", page_token="h0~fingerprint01.!!!not-base64!!!", _store=store
    )
    assert result["ok"] is False


def test_a_payload_that_is_not_a_frozen_set_is_refused(store):
    payload = (
        base64.urlsafe_b64encode(zlib.compress(b'["not", "a", "dict"]', 9))
        .decode()
        .rstrip("=")
    )
    result = aggregator_query(
        "source:github", page_token=f"h0~fingerprint01.{payload}", _store=store
    )
    assert result["ok"] is False


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


def test_legacy_tokens_are_not_collateral_damage(store):
    """``40`` and ``h40`` predate the frozen payload and are still valid."""
    _seed_records(store)
    for token in ("0", "h0"):
        page = aggregator_query(
            "source:github", page_size=1, page_token=token, _store=store
        )
        assert page["ok"] is True, f"{token!r} rejected: {page}"


# --- M4: a token that says two incompatible things -------------------------


def test_a_token_cannot_claim_fts5_only_and_carry_frozen_hits(store):
    """THE REPRO. ``40.<payload>`` parses to ``hybrid=False`` — "this page
    came from the FTS5-only arm" — while ``pin_for`` reads ``self.frozen``
    first and pins the vector arm ON. Two contradictory claims, resolved
    silently in favour of whichever field the code happened to read first."""
    payload = _pack_frozen({"observations": ["o1", "o2"]})
    result = aggregator_query(
        "voting", page_token=f"40~fingerprint01.{payload}", _store=store
    )
    assert result["ok"] is False
    assert "page_token" in result["reason"]


def test_the_cursor_cannot_even_represent_the_contradiction():
    """Closing the instance is not closing the bug: nothing may construct a
    cursor that pins no arm and freezes one anyway."""
    from aggregator.mcp import _PageCursor

    _PageCursor(
        offset=0, hybrid=True, frozen={"records": ["r1"]}, fingerprint="fp01"
    )  # fine
    with pytest.raises(ValueError, match="frozen"):
        _PageCursor(
            offset=0, hybrid=False, frozen={"records": ["r1"]}, fingerprint="fp01"
        )
    # Same closure, one field over: frozen hits with no query to bind them to.
    with pytest.raises(ValueError, match="fingerprint"):
        _PageCursor(offset=0, hybrid=True, frozen={"records": ["r1"]})


def test_a_maximum_legitimate_token_still_round_trips():
    """The cap is derived from the real maximum, so the real maximum fits."""
    long_id = "dropbox:" + ("a" * 500)
    frozen = {
        "observations": [f"{long_id}:{i}" for i in range(_VECTOR_ARM_K)],
        "records": [f"{long_id}:{i}" for i in range(_VECTOR_ARM_K)],
    }
    cursor = _parse_page_token(_mint_page_token(40, True, "fingerprint01", frozen))
    assert cursor.frozen == frozen
    assert cursor.offset == 40
