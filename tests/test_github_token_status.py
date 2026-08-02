"""Tests for the GH_TOKEN detection + ``aggregator github-token-status`` CLI.

Part B of the aggregator source-alignment work:

* ``token_status()`` in ``sources/github.py`` resolves which token the ingest
  path would use (env ``GH_TOKEN`` verbatim, else fall back to ``gh auth
  token``) and reports its scopes + a recommendation. Read-only, no writes.
* ``aggregator github-token-status`` CLI wraps the same call and prints
  either a human-readable summary or ``--json`` for scripting.

The read-only contract stays intact: even with ``GH_TOKEN`` set, the ingest
path still scope-checks the token and refuses write-capable scopes unless
``AGGREGATOR_ALLOW_WRITE_TOKEN=1`` is set. The new surface just makes the
current state visible + actionable.
"""
from __future__ import annotations

import json

import pytest

from aggregator.sources.github import (
    GitHubSource,
    TokenStatus,
    token_status,
)

# --- token_status: pure resolver ------------------------------------------


def test_token_status_uses_gh_token_env_when_set(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "ghp_readonly_from_env")
    monkeypatch.delenv("AGGREGATOR_ALLOW_WRITE_TOKEN", raising=False)
    status = token_status(
        _scope_fetcher=lambda: ["public_repo", "read:org"],
        _gh_token_fetcher=lambda: "ghp_from_gh_cli_should_not_be_used",
    )
    assert isinstance(status, TokenStatus)
    assert status.source == "env"
    assert status.scopes == ["public_repo", "read:org"]
    assert status.write_capable is False
    assert status.override_active is False


def test_token_status_falls_back_to_gh_auth_when_env_unset(monkeypatch):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("AGGREGATOR_ALLOW_WRITE_TOKEN", raising=False)
    status = token_status(
        _scope_fetcher=lambda: ["public_repo"],
        _gh_token_fetcher=lambda: "ghp_from_gh_cli",
    )
    assert status.source == "gh-cli"
    assert status.scopes == ["public_repo"]
    assert status.write_capable is False


def test_token_status_reports_write_scopes(monkeypatch):
    """Even without the override, ``token_status`` REPORTS scopes rather
    than raising — it's a diagnostic surface, not an enforcement point."""
    monkeypatch.setenv("GH_TOKEN", "ghp_full_repo")
    monkeypatch.delenv("AGGREGATOR_ALLOW_WRITE_TOKEN", raising=False)
    status = token_status(
        _scope_fetcher=lambda: ["repo", "gist", "workflow"],
        _gh_token_fetcher=lambda: None,
    )
    assert status.write_capable is True
    # The offending scopes should surface in the recommendation.
    assert "repo" in status.recommendation
    assert "GH_TOKEN" in status.recommendation


def test_token_status_flags_override_when_env_set(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "x")
    monkeypatch.setenv("AGGREGATOR_ALLOW_WRITE_TOKEN", "1")
    status = token_status(
        _scope_fetcher=lambda: ["repo"],
        _gh_token_fetcher=lambda: None,
    )
    assert status.override_active is True
    # With override active AND write scopes, recommendation mentions both.
    assert (
        "AGGREGATOR_ALLOW_WRITE_TOKEN" in status.recommendation
        or "override" in status.recommendation.lower()
    )


def test_token_status_handles_no_token_at_all(monkeypatch):
    """Neither env nor gh-cli produced a token: recommend logging in / setting."""
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("AGGREGATOR_ALLOW_WRITE_TOKEN", raising=False)
    status = token_status(
        _scope_fetcher=lambda: [],
        _gh_token_fetcher=lambda: None,
    )
    assert status.source == "none"
    # Recommendation should point to setting GH_TOKEN or running gh auth login.
    assert (
        "GH_TOKEN" in status.recommendation
        or "gh auth" in status.recommendation.lower()
    )


def test_token_status_handles_scope_fetch_failure(monkeypatch):
    """Scope fetch failing shouldn't raise from token_status — surface a
    diagnostic recommendation instead."""
    monkeypatch.setenv("GH_TOKEN", "x")
    monkeypatch.delenv("AGGREGATOR_ALLOW_WRITE_TOKEN", raising=False)

    def boom() -> list[str]:
        raise FileNotFoundError("gh missing")

    status = token_status(
        _scope_fetcher=boom,
        _gh_token_fetcher=lambda: None,
    )
    # source is still 'env' — we know a token is set — but scopes are
    # unknown and the recommendation calls that out.
    assert status.source == "env"
    assert status.scopes == []
    assert status.scope_error is not None
    assert (
        "cannot" in status.recommendation.lower()
        or "unverif" in status.recommendation.lower()
        or "gh" in status.recommendation.lower()
    )


# --- GitHubSource honours GH_TOKEN env token source ----------------------


def test_github_source_reports_gh_token_as_env_source(monkeypatch):
    """When ``GH_TOKEN`` is set, ingest resolves it as the token source.

    The scope check is still performed against whichever token gh api
    picks up — same read-only contract as before. This test just pins
    the resolution behaviour so a future refactor can't accidentally
    prefer gh's stored credential over the explicit env var.
    """
    monkeypatch.setenv("GH_TOKEN", "ghp_readonly")
    monkeypatch.delenv("AGGREGATOR_ALLOW_WRITE_TOKEN", raising=False)
    src = GitHubSource(
        _scope_fetcher=lambda: ["public_repo"],
        _api_fetcher=lambda p: [],
    )
    # ingest must not raise (readonly scopes) AND must use the env token.
    result = src.ingest(since=None)
    assert result.errors == []
    # Resolved source should be reported via the same helper.
    status = src.token_status()
    assert status.source == "env"


def test_github_source_token_status_uses_gh_cli_when_env_unset(monkeypatch):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("AGGREGATOR_ALLOW_WRITE_TOKEN", raising=False)
    src = GitHubSource(
        _scope_fetcher=lambda: ["public_repo"],
        _api_fetcher=lambda p: [],
        _gh_token_fetcher=lambda: "ghp_from_gh_cli",
    )
    status = src.token_status()
    assert status.source == "gh-cli"


# --- CLI subcommand ------------------------------------------------------


def test_cli_github_token_status_prints_human_summary(tmp_data_home, capsys, monkeypatch):
    """``aggregator github-token-status`` prints a plain-text summary."""
    from aggregator import cli
    from aggregator.core.store import Store

    monkeypatch.setenv("GH_TOKEN", "ghp_readonly")
    monkeypatch.delenv("AGGREGATOR_ALLOW_WRITE_TOKEN", raising=False)

    class StubSource:
        name = "github"

        def token_status(self) -> TokenStatus:
            return TokenStatus(
                source="env",
                scopes=["public_repo", "read:org"],
                write_capable=False,
                override_active=False,
                scope_error=None,
                recommendation="Token looks good — scopes are read-only.",
            )

    store = Store()
    store.migrate()
    rc = cli.main(
        ["github-token-status"],
        _store=store,
        _sources={"github": StubSource()},
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "env" in out
    assert "public_repo" in out
    assert "read-only" in out.lower() or "read:org" in out


def test_cli_github_token_status_json_output(
    tmp_data_home, capsys, monkeypatch
):
    from aggregator import cli
    from aggregator.core.store import Store

    monkeypatch.setenv("GH_TOKEN", "ghp_x")
    monkeypatch.delenv("AGGREGATOR_ALLOW_WRITE_TOKEN", raising=False)

    class StubSource:
        name = "github"

        def token_status(self) -> TokenStatus:
            return TokenStatus(
                source="env",
                scopes=["repo", "workflow"],
                write_capable=True,
                override_active=False,
                scope_error=None,
                recommendation=(
                    "Token has write scopes ['repo', 'workflow']. Either "
                    "export GH_TOKEN=<readonly PAT> or set "
                    "AGGREGATOR_ALLOW_WRITE_TOKEN=1."
                ),
            )

    store = Store()
    store.migrate()
    rc = cli.main(
        ["github-token-status", "--json"],
        _store=store,
        _sources={"github": StubSource()},
    )
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["source"] == "env"
    assert data["write_capable"] is True
    assert "repo" in data["scopes"]
    assert "recommendation" in data


def test_cli_github_token_status_is_idempotent_and_side_effect_free(
    tmp_data_home, capsys, monkeypatch
):
    """Running the subcommand twice must produce identical output and touch
    no store state — pure diagnostic."""
    from aggregator import cli
    from aggregator.core.store import Store

    monkeypatch.setenv("GH_TOKEN", "ghp_x")
    monkeypatch.delenv("AGGREGATOR_ALLOW_WRITE_TOKEN", raising=False)

    class StubSource:
        name = "github"

        def __init__(self):
            self.calls = 0

        def token_status(self) -> TokenStatus:
            self.calls += 1
            return TokenStatus(
                source="env",
                scopes=["public_repo"],
                write_capable=False,
                override_active=False,
                scope_error=None,
                recommendation="ok",
            )

    stub = StubSource()
    store = Store()
    store.migrate()
    # First call.
    cli.main(
        ["github-token-status", "--json"],
        _store=store,
        _sources={"github": stub},
    )
    out1 = capsys.readouterr().out
    # Second call.
    cli.main(
        ["github-token-status", "--json"],
        _store=store,
        _sources={"github": stub},
    )
    out2 = capsys.readouterr().out
    assert out1 == out2
    # And the sub was consulted twice — no cached global state.
    assert stub.calls == 2


def test_cli_github_token_status_unknown_source(
    tmp_data_home, capsys
):
    """If the github source isn't registered (weird test config), exit nonzero
    with a friendly message rather than crash."""
    from aggregator import cli
    from aggregator.core.store import Store

    store = Store()
    store.migrate()
    rc = cli.main(
        ["github-token-status"],
        _store=store,
        _sources={"sessions": object()},
    )
    assert rc != 0
    err = capsys.readouterr().err
    assert "github" in err


# --- token detection integration ----------------------------------------


@pytest.mark.parametrize(
    "gh_token, ghauth, expected_source",
    [
        ("ghp_env", None, "env"),
        ("", "ghp_gh", "gh-cli"),
        ("", None, "none"),
    ],
)
def test_token_status_resolution_matrix(
    monkeypatch, gh_token, ghauth, expected_source
):
    monkeypatch.delenv("AGGREGATOR_ALLOW_WRITE_TOKEN", raising=False)
    if gh_token:
        monkeypatch.setenv("GH_TOKEN", gh_token)
    else:
        monkeypatch.delenv("GH_TOKEN", raising=False)
    status = token_status(
        _scope_fetcher=lambda: ["public_repo"] if (gh_token or ghauth) else [],
        _gh_token_fetcher=lambda: ghauth,
    )
    assert status.source == expected_source
