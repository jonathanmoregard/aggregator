"""The CLI must be able to consume the continuation token it prints.

``_cmd_query`` ends by printing ``# next_page_token: …`` whenever a result
set is longer than one page, and the ``query`` subparser accepted no way to
pass one back. So the only advertised route through a large result set was a
token the tool then refused as an unrecognised argument — page 2 was
unreachable from the CLI, and the operator's evidence for that was an
argparse usage error, not a message.

The MCP path already takes ``page_token`` and the token format is stable, so
the fix is to thread the existing parameter rather than to stop printing it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from aggregator import cli
from aggregator.core.store import Store
from aggregator.sources.base import Record


def _seed(store: Store, n: int = 3) -> None:
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
            for i in range(n)
        ]
    )


def _page(capsys, argv, store):
    rc = cli.main(argv, _store=store)
    payload = json.loads(capsys.readouterr().out)
    return rc, payload


def test_the_printed_token_is_accepted_back(tmp_data_home, capsys):
    """THE REPRO. Take the token the CLI just printed and hand it back."""
    store = Store()
    _seed(store)

    rc, first = _page(
        capsys, ["query", "source:github", "--page-size", "1", "--json"], store
    )
    assert rc == 0
    token = first.get("next_page_token")
    assert token, f"no continuation token to test with: {first!r}"

    rc, second = _page(
        capsys,
        ["query", "source:github", "--page-size", "1", "--json",
         "--page-token", token],
        store,
    )

    assert rc == 0
    assert second["ok"] is True
    first_ids = [r["stable_id"] for r in first["records"]]
    second_ids = [r["stable_id"] for r in second["records"]]
    assert second_ids and second_ids != first_ids, (
        f"the second page repeated the first: {second_ids!r}"
    )


def test_a_bad_token_degrades_to_the_first_page(tmp_data_home, capsys):
    """The CLI inherits the MCP path's contract for a token it cannot read.

    ``_parse_page_token`` resets an unparseable token to the first page rather
    than raising — deliberately, so tokens minted by older builds keep
    working. The CLI passes the value straight through, so it must behave the
    same way and not crash; asserted here so the two surfaces cannot drift.
    """
    store = Store()
    _seed(store)

    rc, page = _page(
        capsys,
        ["query", "source:github", "--json", "--page-token", "not-a-real-token"],
        store,
    )

    assert rc == 0
    assert page["ok"] is True


def test_the_flag_appears_in_help(tmp_data_home, capsys):
    """Discoverable, or the token printed at the bottom of a page is a dead end."""
    with pytest.raises(SystemExit):
        cli.main(["query", "--help"])
    assert "--page-token" in capsys.readouterr().out
