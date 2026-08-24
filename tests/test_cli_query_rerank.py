"""``aggregator query --rerank`` — the batch surface the docs already claim.

``mcp.aggregator_query``'s own documentation calls ``rerank=True`` a
background/batch facility and tells callers not to hold an interactive turn
open for it — **273 s median, 304 s at p95** per call on this machine against
0.65 s without. Every surface that exposed it was interactive, and the only
non-interactive entry point, the CLI, had no flag at all. So the measured
advice named a place to use the feature that did not exist.

AND IT MUST NOT DEGRADE QUIETLY. ``_maybe_rerank`` still catches a rerank
failure and returns the page in its fused order, which is right for the MCP
tool — a lost ordering should not cost a caller their answer. It no longer
does so in silence: since ``04acf86`` the response carries ``rerank_applied:
False`` and leads with a ``notice`` naming the exception and pointing at
``aggregator embed --seed-models``, and this command prints that notice.

On a command whose ONLY purpose is to rerank, reporting after the fact is
still the wrong trade — measured here with a reranker that loads and then
raises while scoring: exit 0, the full page printed, and the notice on the
line after the last row, so the operator reads it only after scrolling past
the results it disclaims and a caller checking ``$?`` sees success. So the
CLI loads the model up front and refuses out loud when it cannot, turning a
degradation reported afterwards into a refusal that costs nothing and exits
non-zero. It narrows the window rather than closing it — scoring can still
fail once the model is loaded — and the notice is what covers the remainder.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from aggregator import cli
from aggregator.core.store import Store
from aggregator.sources.base import Record


def _seed(store: Store) -> None:
    store.migrate()
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
            for i in range(3)
        ]
    )


def test_the_flag_exists_on_the_batch_surface(tmp_data_home, capsys):
    """THE REPRO for the missing surface."""
    with pytest.raises(SystemExit):
        cli.main(["query", "--help"])
    assert "--rerank" in capsys.readouterr().out


def test_rerank_reaches_the_query(tmp_data_home, monkeypatch, capsys):
    """A flag that parses and then does nothing would be worse than none."""
    store = Store()
    _seed(store)
    seen: dict = {}

    def fake_query(**kwargs):
        seen.update(kwargs)
        return {"ok": True, "records": [], "total": 0, "mode": "records"}

    monkeypatch.setattr(cli, "_mcp_query", fake_query)
    monkeypatch.setattr(cli, "_mcp_get_reranker", lambda: object())

    rc = cli.main(["query", "source:github", "--rerank"], _store=store)

    assert rc == 0
    assert seen.get("rerank") is True


def test_a_reranker_that_cannot_load_is_loud_and_non_zero(
    tmp_data_home, monkeypatch, capsys
):
    """THE REPRO for the silent degrade: absent weights must not read as success."""
    store = Store()
    _seed(store)

    def no_weights():
        raise OSError(
            "Qwen/Qwen3-Reranker-0.6B is not in the local cache and "
            "local_files_only=True"
        )

    monkeypatch.setattr(cli, "_mcp_get_reranker", no_weights)

    rc = cli.main(["query", "source:github", "--rerank"], _store=store)

    err = capsys.readouterr().err
    assert rc != 0, (
        "the command exited 0 having silently returned results in their "
        "fused order — the operator has no way to know the cross-encoder "
        f"never ran. stderr={err!r}"
    )
    assert "--rerank" in err
    assert "seed-models" in err, (
        f"the refusal does not name the command that fixes it: {err!r}"
    )


def test_a_load_failure_does_not_print_a_page_of_results(
    tmp_data_home, monkeypatch, capsys
):
    """Refusing means refusing: no recency-ordered page dressed up as ranked."""
    store = Store()
    _seed(store)
    monkeypatch.setattr(
        cli, "_mcp_get_reranker", lambda: (_ for _ in ()).throw(OSError("nope"))
    )

    cli.main(["query", "source:github", "--rerank"], _store=store)

    assert "pull request" not in capsys.readouterr().out


def test_no_reranker_is_built_without_the_flag(tmp_data_home, monkeypatch):
    """~2 GB RSS and a model load that a plain query must never pay."""
    store = Store()
    _seed(store)

    def must_not_run():  # pragma: no cover - the assertion is that it does not
        raise AssertionError("the reranker was constructed for a plain query")

    monkeypatch.setattr(cli, "_mcp_get_reranker", must_not_run)

    assert cli.main(["query", "source:github"], _store=store) == 0


# --- --rerank implies --fields full -----------------------------------------
#
# THE REPRO. ``rerank=True`` refuses under ``fields='summary'`` because the
# cross-encoder would score empty bodies (measured: three documents collapsing
# to the one literal string ``'user\n\n'``) for the same multi-minute pass.
# That refusal is
# right and stands. Its side effect was not: ``--fields`` defaulted to
# ``summary``, so ``aggregator query "..." --rerank`` — the batch surface's
# most obvious invocation, the one the MCP tool description points callers at —
# exited 1 against itself. Observed before the fix:
#
#     $ aggregator query "source:github voting" --rerank
#       -> exit 1
#       | error: rerank=True needs document bodies to score, and
#       |        fields='summary' does not return them ...
#
# So ``--rerank`` now supplies the ``--fields full`` it needs, and ONLY when
# the operator left ``--fields`` off. Passing ``--fields summary --rerank``
# still refuses: that is two incompatible things asked for out loud, and
# quietly overriding one of them is the failure mode this whole round removed.


def test_rerank_alone_asks_for_the_bodies_it_needs(
    tmp_data_home, monkeypatch, capsys
):
    """THE REPRO: the batch surface's most obvious invocation refused itself."""
    store = Store()
    _seed(store)
    seen: dict = {}

    def fake_query(**kwargs):
        seen.update(kwargs)
        return {"ok": True, "records": [], "total": 0, "mode": "records"}

    monkeypatch.setattr(cli, "_mcp_query", fake_query)
    monkeypatch.setattr(cli, "_mcp_get_reranker", lambda: object())

    rc = cli.main(["query", "source:github voting", "--rerank"], _store=store)

    assert rc == 0, (
        "`--rerank` with no --fields exited non-zero: the flag refuses its own "
        f"default. stderr={capsys.readouterr().err!r}"
    )
    assert seen.get("fields") == "full", (
        "--rerank must supply the document bodies it scores; got "
        f"fields={seen.get('fields')!r}"
    )


def test_an_explicit_fields_full_is_unchanged(tmp_data_home, monkeypatch, capsys):
    """The invocation that already worked must keep working, identically."""
    store = Store()
    _seed(store)
    seen: dict = {}

    def fake_query(**kwargs):
        seen.update(kwargs)
        return {"ok": True, "records": [], "total": 0, "mode": "records"}

    monkeypatch.setattr(cli, "_mcp_query", fake_query)
    monkeypatch.setattr(cli, "_mcp_get_reranker", lambda: object())

    rc = cli.main(
        ["query", "source:github voting", "--rerank", "--fields", "full"],
        _store=store,
    )

    assert rc == 0, capsys.readouterr().err
    assert seen.get("fields") == "full"


def test_explicitly_asking_for_summary_and_rerank_still_refuses(
    tmp_data_home, monkeypatch, capsys
):
    """Two incompatible things, asked for out loud. Silently dropping one of
    them is exactly the degradation this command exists to refuse.

    And it refuses on argv alone — neither the ~2 GB cross-encoder load nor any
    retrieval may be spent reaching a conclusion argparse already had. Both are
    wired to fail here, so a refusal that arrives late fails this test.
    """
    store = Store()
    _seed(store)

    def must_not_load():  # pragma: no cover - the assertion is that it does not
        raise AssertionError("the cross-encoder was loaded for a doomed query")

    def must_not_query(**kwargs):  # pragma: no cover - asserted not to run
        raise AssertionError("the refusal must land before any retrieval")

    monkeypatch.setattr(cli, "_mcp_get_reranker", must_not_load)
    monkeypatch.setattr(cli, "_mcp_query", must_not_query)

    rc = cli.main(
        ["query", "source:github voting", "--rerank", "--fields", "summary"],
        _store=store,
    )
    err = capsys.readouterr().err

    assert rc != 0, (
        "--fields summary --rerank exited 0 — the caller asked for an ordering "
        "over bodies the mode does not return, and was not told"
    )
    assert "--rerank" in err and "--fields summary" in err, (
        f"the refusal must name both flags the operator typed: {err!r}"
    )
    assert "--fields full" in err, (
        f"the refusal does not name the one-step fix: {err!r}"
    )


def test_a_plain_query_still_summarises(tmp_data_home, monkeypatch, capsys):
    """The implication rides on --rerank and nothing else: a query without it
    keeps the cheap summary default and its 200-row pages."""
    store = Store()
    _seed(store)
    seen: dict = {}

    def fake_query(**kwargs):
        seen.update(kwargs)
        return {"ok": True, "records": [], "total": 0, "mode": "records"}

    monkeypatch.setattr(cli, "_mcp_query", fake_query)

    assert cli.main(["query", "source:github voting"], _store=store) == 0
    assert seen.get("fields") == "summary"


def test_help_says_the_implication_changes_what_is_printed(tmp_data_home, capsys):
    """--fields full returns document BODIES. The implication therefore changes
    the payload, and an operator must read that in --help, not discover it."""
    with pytest.raises(SystemExit):
        cli.main(["query", "--help"])
    out = capsys.readouterr().out

    assert "--fields full" in out, (
        f"--help does not say what --rerank implies: {out!r}"
    )
    assert "bodies" in out, (
        "--help does not state the payload consequence of the implication — "
        f"that full mode prints document bodies: {out!r}"
    )


# --- the cost figure the operator decides on --------------------------------
#
# "47 s median / 59 s worst" was measured while the reranker was scoring
# SESSION CARDS AGAINST THEIR OWN SUBJECTS — near-empty documents — and while
# the three pages holding a real long document were being OOM-killed rather
# than timed. Both are fixed, and the figure was re-measured on a read-only
# snapshot of the live 505k-observation cache: **273 s median, 304 s p95**,
# ~13.7 s per (query, document) pair. ``aggregator/mcp.py`` and its tests were
# corrected in b8c00ea; these three CLI surfaces were left quoting the old
# number, which is worse than quoting none — an operator who reads "47 s"
# decides to wait, and then waits four and a half minutes.
#
# PINNED AS A RANGE, NOT A DIGIT. What must survive re-measurement is the
# ORDER OF MAGNITUDE and the unit: a three-digit seconds figure reads as small
# at a glance, and "minutes" is what actually stops someone reaching for this
# mid-turn. Same contract as ``test_mcp_discoverability`` — if it is measured
# again, update both.


def _rerank_surfaces(store, monkeypatch, capsys) -> dict[str, str]:
    """Every string the CLI shows an operator about what ``--rerank`` costs.

    Whitespace-collapsed, because argparse rewraps ``--help`` to the terminal
    width and a phrase that happens to straddle a line break is not a different
    claim. The assertions are about what an operator READS.
    """
    surfaces = {"refusal": cli._rerank_needs_full_fields()}

    with pytest.raises(SystemExit):
        cli.main(["query", "--help"])
    surfaces["help"] = capsys.readouterr().out

    monkeypatch.setattr(cli, "_mcp_get_reranker", lambda: object())
    monkeypatch.setattr(
        cli,
        "_mcp_query",
        lambda **kw: {"ok": True, "records": [], "total": 0, "mode": "records"},
    )
    cli.main(["query", "source:github", "--rerank"], _store=store)
    surfaces["note"] = capsys.readouterr().err
    return {k: " ".join(v.split()) for k, v in surfaces.items()}


def test_no_cli_surface_quotes_the_falsified_rerank_cost(
    tmp_data_home, monkeypatch, capsys
):
    """THE REPRO for a stale measurement outliving its measurement."""
    store = Store()
    _seed(store)

    for name, text in _rerank_surfaces(store, monkeypatch, capsys).items():
        assert "47 s" not in text and "47-second" not in text, (
            f"the --rerank {name} still quotes the falsified 47 s median. It "
            "was measured over session cards scored against their own "
            "subjects, and over pages whose expensive members were OOM-killed "
            f"rather than timed. Re-measured: 273 s median, 304 s p95. {text!r}"
        )


def test_the_cli_names_a_wait_measured_in_minutes(
    tmp_data_home, monkeypatch, capsys
):
    """A number is not the point; the unit is. 273 reads small, 4.5 minutes
    does not, and the decision this text exists to inform is "do I wait?"."""
    surfaces = _rerank_surfaces(Store(), monkeypatch, capsys)

    for name in ("help", "note"):
        text = surfaces[name]
        assert "273 s" in text, (
            f"the --rerank {name} does not carry the measured median "
            f"(273 s): {text!r}"
        )
        assert "minute" in text, (
            f"the --rerank {name} states seconds only. A three-digit seconds "
            "figure reads as small at a glance; the unit is what stops a "
            f"caller reaching for this mid-turn: {text!r}"
        )
