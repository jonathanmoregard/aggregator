"""The tool description is the contract, and it did not describe the default.

``aggregator_query.__doc__`` IS the MCP tool description — ``build_server``
hands it straight to ``server.tool(description=...)``, so it is what the model
reads before deciding how to call this and how to read the answer back.

It documented ``mode`` as ``sessions``, ``observations`` or ``records``. But
``_route_mode`` returns ``"union"`` for any query with no source hint and no
ontology-specific keys — text-only, date-only, tag-only — which is the COMMON
case and the one the description's own first example (``dsl="quadratic
voting"``) produces. ``_query_union_path`` then emits ``mode="union"`` and a
``records`` list holding BOTH record-shaped and session-shaped items, which
carry different keys.

So the documented contract did not describe what the caller would usually
receive, and the one shape it never mentioned is the one that needs mentioning
most, because it is the only heterogeneous one.

These are doc guards with teeth: the mode list is read out of the production
source, not restated here. A hand-maintained list in a test goes stale in the
same way, at the same moment, and for the same reason as the one in the
docstring — which is what happened.
"""

from __future__ import annotations

import inspect
import re
from datetime import UTC, datetime

import pytest

import aggregator.mcp as mcp
from aggregator.core.store import Store
from aggregator.mcp import aggregator_query
from aggregator.sources.base import ObservationRow, Record, SessionRow

_TS = datetime(2026, 7, 1, tzinfo=UTC)


def _modes_the_code_can_emit() -> set[str]:
    """Every literal ``mode`` value the query tool can put in a response."""
    src = inspect.getsource(mcp)
    return set(re.findall(r'"mode":\s*"([a-z_]+)"', src)) | set(
        re.findall(r'\bmode="([a-z_]+)"', src)
    )


@pytest.fixture
def store(tmp_path):
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert(
        [
            Record(
                stable_id="github:acme/api:1", source="github",
                subject="pr 1 about quadratic voting", body="body",
                tags=["pr"], created_at=_TS, updated_at=_TS,
            )
        ]
    )
    s.upsert_entities(
        [
            SessionRow(
                session_id="s1", root_session_id="s1", parent_session_id=None,
                kind="session", agent_id=None, agent_type=None,
                spawned_by_tool_use_id=None, cwd="/x", git_branch="main",
                first_ts=_TS, last_ts=_TS, jsonl_path="/tmp/s1.jsonl",
            ),
            ObservationRow(
                obs_id="o1", session_id="s1", root_session_id="s1",
                parent_obs_id=None, type="user", ts=_TS, model=None,
                input_tokens=None, output_tokens=None, tool_name=None,
                tool_use_id=None, body="a discussion of quadratic voting",
            ),
        ]
    )
    return s


def test_the_mode_extraction_has_teeth():
    """Without this the guard below can pass by finding nothing."""
    modes = _modes_the_code_can_emit()
    assert modes >= {"records", "sessions", "observations", "union"}, (
        f"extraction found only {sorted(modes)}"
    )


def test_every_mode_the_code_can_emit_is_documented():
    """THE REPRO. ``union`` was emitted and never mentioned."""
    doc = aggregator_query.__doc__ or ""
    missing = sorted(m for m in _modes_the_code_can_emit() if m not in doc)
    assert not missing, f"undocumented response modes: {missing}"


def test_the_common_case_really_does_return_the_undocumented_mode(store):
    """Not a hypothetical gap: the description's own first example hits it."""
    result = aggregator_query("quadratic voting", _store=store)
    assert result["mode"] == "union"
    assert result["mode"] in (aggregator_query.__doc__ or "")


def test_a_date_only_query_takes_the_same_route(store):
    result = aggregator_query("from:2026-01-01 to:2026-12-31", _store=store)
    assert result["mode"] == "union"


def test_the_mixed_shape_of_a_union_page_is_described(store):
    """``union`` items are not one shape, and a caller told only the mode name
    would still not know how to read them."""
    result = aggregator_query("quadratic voting", _store=store)
    kinds = result["records"]
    assert any("stable_id" in i and "kind" in i for i in kinds), (
        "expected a session-shaped item in the union page"
    )
    assert any("kind" not in i for i in kinds), (
        "expected a record-shaped item in the union page"
    )

    doc = (aggregator_query.__doc__ or "").lower()
    assert "mixed" in doc or "both shapes" in doc, (
        "the union mode's heterogeneous item shape is not described"
    )


def test_the_description_the_server_publishes_is_this_docstring():
    """If these ever come apart, the guards above stop guarding anything."""
    assert mcp._tool_aggregator_query.__doc__ == aggregator_query.__doc__


# --- the batch facility the docs describe now has somewhere to point -------


def _rerank_docs() -> str:
    _, _, tail = (aggregator_query.__doc__ or "").partition("rerank:")
    return tail


def _cli_help(subcommand: str, capsys) -> str:
    from aggregator import cli

    with pytest.raises(SystemExit):
        cli.main([subcommand, "--help"])
    return capsys.readouterr().out


def test_the_rerank_docs_name_the_batch_surface(capsys):
    """The docs called rerank a batch/offline facility while naming no batch
    surface, so the only actionable reading of "do not use this interactively"
    was "do not use this". ``aggregator query --rerank`` now exists."""
    assert "aggregator query --rerank" in _rerank_docs()


def test_the_named_batch_surface_actually_exists(capsys):
    """A tool description pointing at a command that is not there is worse
    than one pointing at nothing: it is checkable and wrong."""
    assert "--rerank" in _cli_help("query", capsys)


def test_the_weight_fetch_command_named_in_a_remediation_exists(capsys):
    """``_maybe_rerank``'s failure notice tells the operator to run this."""
    from aggregator.mcp import _maybe_rerank

    class Dead:
        def score(self, q, docs):
            raise RuntimeError("no weights")

    mcp._get_reranker, saved = (lambda: Dead()), mcp._get_reranker
    try:
        _items, reranked, notice, standout = _maybe_rerank(
            [{"content": "x"}], "q", True
        )
    finally:
        mcp._get_reranker = saved

    # A count now, not a flag: it is how far down the page the ranking got,
    # and ``rerank_applied`` is a view of it. Nothing was ranked here.
    assert reranked == 0
    assert "aggregator embed --seed-models" in notice
    # Criterion D's fourth value. ``None``, not ``False``: the model produced
    # no scores, so it has no opinion about whether anything was relevant, and
    # reading a dead model as "nothing stood out" would invent an answer out
    # of a failure.
    assert standout is None
    assert "--seed-models" in _cli_help("embed", capsys)
