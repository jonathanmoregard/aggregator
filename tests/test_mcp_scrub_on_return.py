"""Pre-return scrubbing: new token formats, every route, and fail-closed.

Rows are written RAW, bypassing the ingest scrub, to stand in for a row
stored before its token format was known. Tokens are generated at test time.
"""
from __future__ import annotations

import secrets
import string

import pytest

from aggregator.core.store import Store
from aggregator.mcp import aggregator_query

from .test_mcp_query import _sess

_ALNUM = string.ascii_letters + string.digits


def _token() -> tuple[str, str]:
    body = "".join(secrets.choice(_ALNUM) for _ in range(32))
    return "tvly" + "-dev-" + body, body


@pytest.fixture()
def leaky(tmp_data_home):
    store = Store()
    store.migrate()
    store.upsert_entities([_sess("sess-leak")])
    token, secret = _token()
    c = store._c()
    # The secret sits in the first 120 characters on purpose: the observation
    # card's SUBJECT is cut from there, and used to be cut from the raw body.
    c.execute(
        "INSERT INTO observations(obs_id, session_id, root_session_id, type, ts, body) "
        "VALUES ('o-leak', 'sess-leak', 'sess-leak', 'user', "
        "'2026-07-25T10:00:00+00:00', ?)",
        (f"key {token} leaked here",),
    )
    c.execute(
        "INSERT INTO records(stable_id, source, subject, body, tags, updated_at) "
        "VALUES ('github:1', 'github', ?, ?, '[]', '2026-07-25T10:00:00+00:00')",
        (f"issue {token}", f"body {token}"),
    )
    c.commit()
    yield store, secret
    store.close()


@pytest.mark.parametrize(
    "dsl",
    [
        "source:sessions",  # session cards: subject + preview
        "leaked",  # union route
        "source:github",  # records route
    ],
)
@pytest.mark.parametrize("fields", ["summary", "full"])
def test_no_route_returns_the_secret(leaky, dsl, fields):
    store, secret = leaky
    result = aggregator_query(dsl=dsl, fields=fields, _store=store)
    assert result["ok"] is True
    assert secret not in repr(result)


@pytest.mark.parametrize("fields", ["summary", "full"])
def test_observation_drilldown_subject_and_body_are_scrubbed(leaky, fields):
    store, secret = leaky
    result = aggregator_query(
        dsl="source:sessions", fields=fields, drilldown=True, _store=store
    )
    assert result["ok"] is True
    assert result["records"], "the leaky observation must be on the page"
    assert secret not in repr(result)


def test_results_are_withheld_when_the_scanner_is_missing(leaky, monkeypatch, tmp_path):
    store, secret = leaky
    monkeypatch.setenv("AGGREGATOR_GITLEAKS", str(tmp_path / "absent"))
    result = aggregator_query(dsl="source:sessions", fields="full", _store=store)
    assert result["ok"] is False
    assert "secret scanner unavailable" in result["reason"]
    assert secret not in repr(result)
