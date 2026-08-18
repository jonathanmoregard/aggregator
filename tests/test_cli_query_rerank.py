"""``aggregator query --rerank`` — the batch surface the docs already claim.

``mcp.aggregator_query``'s own documentation calls ``rerank=True`` a
background/batch facility and tells callers not to hold an interactive turn
open for it — 47 s median per call on this machine against 0.65 s without.
Every surface that exposed it was interactive, and the only non-interactive
entry point, the CLI, had no flag at all. So the measured advice named a
place to use the feature that did not exist.

AND IT MUST NOT DEGRADE QUIETLY. ``_maybe_rerank`` swallows a rerank failure
and returns the page in its fused order, which is right for the MCP tool — a
lost ordering should not cost a caller their answer. On a command whose ONLY
purpose was to rerank it is the wrong trade: the operator waited, got
recency-ordered output, and nothing said the cross-encoder never ran. So the
CLI loads the model up front and refuses out loud when it cannot.
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
