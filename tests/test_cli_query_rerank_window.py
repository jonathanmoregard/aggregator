"""``--rerank`` ranks the head of a page, and must say where the head ends.

Round 3's M4. The cross-encoder reorders at most ``mcp._RERANK_WINDOW`` = 20
items; ``--page-size`` defaults to 50. So the default invocation of the flag
returns a page that is 40% ranked and 60% in the ordinary recency order, with
nothing marking the seam.

Two things made that worse than a missing detail:

* the note printed on every ``--rerank`` run claimed it "scores every hit",
  which is not true at any page size above the window — the one line an
  operator reads while waiting 47 seconds actively told them the wrong thing;
* the rows below the boundary look exactly like the rows above it. A reader
  scrolling a 50-row page has no way to tell that row 21 onwards was never
  scored, so the recency ordering reads as a relevance ordering.

The window itself is not changed here — it lives in ``mcp.py`` and the cost it
bounds (47 s median for 20 pairs on this CPU) is real. What changes is that the
page stops lying about which part of it was ranked.
"""

import argparse

import pytest

from aggregator.cli import _cmd_query, build_parser
from aggregator.mcp import _RERANK_WINDOW


def _ns(**kw):
    ns = argparse.Namespace(
        dsl="body",
        fields=None,
        page_size=50,
        page_token=None,
        json=False,
        drilldown=False,
        rerank=True,
    )
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


@pytest.fixture
def ranked_page(monkeypatch):
    """A page of 50 hits that the reranker really did reorder the head of."""
    records = [
        {
            "source": "sessions",
            "subject": f"subject {i}",
            "stable_id": f"s{i}",
            "content": f"body {i}",
        }
        for i in range(50)
    ]
    result = {
        "ok": True,
        "mode": "records",
        "records": records,
        "total": 50,
        "rerank_applied": True,
    }
    monkeypatch.setattr("aggregator.cli._mcp_query", lambda **kw: result)
    monkeypatch.setattr("aggregator.cli._mcp_get_reranker", lambda: object())
    return result


def test_the_note_no_longer_claims_every_hit_is_scored(ranked_page, capsys):
    """THE M4 REGRESSION: a false claim on the line the operator waits on."""
    _cmd_query(_ns(), store=None)
    err = capsys.readouterr().err

    assert "scores every hit" not in err, err
    assert str(_RERANK_WINDOW) in err, (
        "the note does not say how many hits are actually scored"
    )


def test_the_boundary_is_marked_in_the_output(ranked_page, capsys):
    """Row 21 of 50 was never scored and must not read as if it were."""
    _cmd_query(_ns(), store=None)
    out = capsys.readouterr().out

    lines = out.splitlines()
    marker = [i for i, ln in enumerate(lines) if "ranked by relevance" in ln]
    assert marker, f"no boundary marker in the page:\n{out}"

    before = "\n".join(lines[: marker[0]])
    after = "\n".join(lines[marker[0] :])
    # The window's worth of hits sit above the seam, the rest below it.
    assert "# sessions :: subject 0" in before
    assert f"# sessions :: subject {_RERANK_WINDOW}" in after


def test_no_marker_when_the_page_fits_inside_the_window(monkeypatch, capsys):
    """Nothing to disclaim: every hit on the page really was scored."""
    records = [
        {
            "source": "sessions",
            "subject": f"subject {i}",
            "stable_id": f"s{i}",
            "content": f"body {i}",
        }
        for i in range(5)
    ]
    monkeypatch.setattr(
        "aggregator.cli._mcp_query",
        lambda **kw: {
            "ok": True,
            "mode": "records",
            "records": records,
            "total": 5,
            "rerank_applied": True,
        },
    )
    monkeypatch.setattr("aggregator.cli._mcp_get_reranker", lambda: object())

    _cmd_query(_ns(page_size=5), store=None)
    out = capsys.readouterr().out
    assert "ranked by relevance" not in out


def test_no_marker_when_the_rerank_did_not_apply(monkeypatch, capsys):
    """A degraded rerank must not claim a ranked prefix it does not have."""
    records = [
        {
            "source": "sessions",
            "subject": f"subject {i}",
            "stable_id": f"s{i}",
            "content": f"body {i}",
        }
        for i in range(50)
    ]
    monkeypatch.setattr(
        "aggregator.cli._mcp_query",
        lambda **kw: {
            "ok": True,
            "mode": "records",
            "records": records,
            "total": 50,
            "rerank_applied": False,
            "notice": "rerank did NOT apply: the cross-encoder failed",
        },
    )
    monkeypatch.setattr("aggregator.cli._mcp_get_reranker", lambda: object())

    _cmd_query(_ns(), store=None)
    out = capsys.readouterr().out
    assert "ranked by relevance" not in out


def test_the_help_text_states_the_window(capsys):
    """A reader choosing --page-size needs the number before they spend 47 s."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["query", "--help"])
    helptext = capsys.readouterr().out

    assert "scores every hit" not in helptext
    assert str(_RERANK_WINDOW) in helptext
