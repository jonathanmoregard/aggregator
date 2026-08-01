"""GitHub source: uses `gh api` under the hood to cache PRs + issues.

Read-only credential enforcement: refuses to run if `gh auth` scopes include
any write-capable scope, unless AGGREGATOR_ALLOW_WRITE_TOKEN=1 is set.

See spec §Security constraint 1 and plan M1b for the contract.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from collections.abc import Callable, Iterator
from datetime import datetime

# Reserved seam: LLM wrapper for future ingest enrichment (see spec §Error
# handling). v1 makes no LLM calls, so the import previously wired here was
# dropped (advisor round-2 BLOCKER: sessions.py had already dropped its
# equivalent; github.py still carried the dead import). When enrichment
# lands, wire the runner at the call site, not as an unused module-level
# import.
from aggregator.sources.base import IngestResult, QueryAST, Record, stable_id_for

log = logging.getLogger(__name__)


# Scopes that grant write capability. Presence of any of these = refuse.
# See https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/scopes-for-oauth-apps
WRITE_SCOPES = {
    "repo",  # full repo access includes write
    "delete_repo",
    "admin:repo_hook",
    "admin:org",
    "admin:public_key",
    "admin:org_hook",
    "gist",  # can create gists
    "write:packages",
    "write:discussion",
    "workflow",  # can modify workflows
}
# Read-only equivalents that are FINE:
#   repo:status, public_repo, read:org, read:user, read:discussion, read:packages


class WriteCapableTokenError(RuntimeError):
    """Raised when the gh token has write scopes and the override env var is
    not set — OR when scope verification failed (advisor round-1 HIGH-2:
    fail-closed on unverifiable scopes, not fail-open)."""


class ScopeFetchError(RuntimeError):
    """Raised by ``_default_scope_fetcher`` when it cannot determine scopes.

    Kept distinct from ``WriteCapableTokenError`` so custom fetchers can
    signal "I don't know" without pretending to have parsed a response.
    ``_check_scopes`` catches this AND arbitrary exceptions from the fetcher
    and re-raises as ``WriteCapableTokenError`` (unless the operator has
    set ``AGGREGATOR_ALLOW_WRITE_TOKEN=1``).
    """


def _parse_scopes(scopes_header: str) -> list[str]:
    """Parse an 'X-Oauth-Scopes: a, b, c' header line into a list.

    Accepts either the raw header line ('X-Oauth-Scopes: a, b') or a bare CSV
    ('a, b'). Case-insensitive on the header prefix.
    """
    line = scopes_header.strip()
    lower = line.lower()
    if lower.startswith("x-oauth-scopes:"):
        _, _, val = line.partition(":")
    else:
        val = line
    return [s.strip() for s in val.split(",") if s.strip()]


def _has_write_scope(scopes: list[str]) -> bool:
    return any(s in WRITE_SCOPES for s in scopes)


def _default_scope_fetcher() -> list[str]:
    """Call `gh api -i /rate_limit` and parse X-Oauth-Scopes from headers.

    On failure (missing gh, non-zero exit, timeout) raises
    ``ScopeFetchError``. ``_check_scopes`` converts that to a
    ``WriteCapableTokenError`` unless ``AGGREGATOR_ALLOW_WRITE_TOKEN=1`` is
    set. This is a change from the pre-HIGH-2 behaviour where failure
    returned ``[]`` and was silently treated as read-only.

    An empty ``X-Oauth-Scopes`` header (no scopes assigned) is still
    returned as ``[]`` — that's a valid response from GitHub, not a fetch
    failure.
    """
    try:
        result = subprocess.run(
            ["gh", "api", "-i", "/rate_limit"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        out = result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        log.warning("gh api -i /rate_limit failed: %s", e)
        raise ScopeFetchError(str(e)) from e
    for line in out.splitlines():
        if line.lower().startswith("x-oauth-scopes:"):
            return _parse_scopes(line)
    # Header absent from response: gh returned something but not the scopes
    # header we expected. Treat as unverifiable — same fail-closed policy.
    raise ScopeFetchError("no X-Oauth-Scopes header in gh api response")


def _default_api_fetcher(path: str) -> list[dict]:
    """Call `gh api <path> --paginate` and return the JSON-decoded list.

    Returns [] on failure so ingest can continue with partial data. The caller
    records the resulting empty page as zero adds without error.
    """
    try:
        result = subprocess.run(
            ["gh", "api", "--paginate", path],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        log.warning("gh api %s failed: %s", path, e)
        return []
    try:
        data = json.loads(out)
    except json.JSONDecodeError as e:
        log.warning("gh api %s returned non-JSON: %s", path, e)
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("items", [])
    return []


class GitHubSource:
    """Source implementation for GitHub PRs + issues via `gh api`.

    Instances take injectable `_scope_fetcher` and `_api_fetcher` seams so unit
    tests never hit the real gh CLI. The default fetchers shell out via
    subprocess.
    """

    name = "github"

    def __init__(
        self,
        _scope_fetcher: Callable[[], list[str]] = _default_scope_fetcher,
        _api_fetcher: Callable[[str], list[dict]] = _default_api_fetcher,
    ):
        self._scope_fetcher = _scope_fetcher
        self._api_fetcher = _api_fetcher

    def record_shape(self) -> dict[str, str]:
        """Return the DSL-facing field surface. Consumed by M2 help generator."""
        return {
            "repo": "str (owner/name)",
            "number": "int",
            "state": "'open'|'closed'",
            "mergeable": "bool | None",
            "checks": "'pass'|'fail'|'pending'|None",
            "author": "str (@login)",
            "url": "str",
            "body_excerpt": "str (first 500 chars)",
        }

    def _check_scopes(self) -> None:
        override = os.environ.get("AGGREGATOR_ALLOW_WRITE_TOKEN") == "1"
        try:
            scopes = self._scope_fetcher()
        except Exception as e:  # noqa: BLE001 -- ANY fetch failure is fail-closed
            # Advisor round-1 HIGH-2: pre-fix, subprocess errors returned []
            # which the check treated as read-only. Post-fix, unverifiable
            # scopes are fail-closed unless the operator has explicitly opted
            # out via AGGREGATOR_ALLOW_WRITE_TOKEN=1.
            if override:
                log.warning(
                    "scope check bypassed (AGGREGATOR_ALLOW_WRITE_TOKEN=1); "
                    "scope fetch failed: %s",
                    e,
                )
                return
            raise WriteCapableTokenError(
                f"cannot verify GitHub token scope: {e}. "
                "Set AGGREGATOR_ALLOW_WRITE_TOKEN=1 to proceed anyway."
            ) from e
        if _has_write_scope(scopes) and not override:
            offending = sorted(set(scopes) & WRITE_SCOPES)
            raise WriteCapableTokenError(
                f"gh token has write-capable scopes {offending}. "
                "Set AGGREGATOR_ALLOW_WRITE_TOKEN=1 to override, or re-scope the token "
                "(recommended: public_repo, repo:status, read:org)."
            )

    def _pr_to_record(self, pr: dict) -> Record:
        repo = pr.get("base", {}).get("repo", {}).get("full_name") or "unknown/unknown"
        number = pr.get("number", 0)
        checks = (pr.get("_checks") or {}).get("summary")
        body = pr.get("body") or ""
        return Record(
            stable_id=stable_id_for("github", f"{repo}:{number}"),
            source="github",
            subject=(pr.get("title") or "")[:280],
            body=body,
            tags=[repo, "pr", pr.get("state", "unknown")],
            created_at=_parse_iso(pr.get("created_at")),
            updated_at=_parse_iso(pr.get("updated_at")),
            extra={
                "repo": repo,
                "number": number,
                "state": pr.get("state"),
                "mergeable": pr.get("mergeable"),
                "mergeable_state": pr.get("mergeable_state"),
                "checks": checks,
                "author": (pr.get("user") or {}).get("login"),
                "url": pr.get("html_url"),
                "body_excerpt": body[:500],
                "kind": "pr",
            },
        )

    def _issue_to_record(self, issue: dict, *, kind: str) -> Record:
        # repo comes from `repository_url` on the issues endpoint:
        #   https://api.github.com/repos/owner/name  →  owner/name
        repo_url = issue.get("repository_url", "")
        if repo_url:
            parts = repo_url.rsplit("/", 2)
            repo = f"{parts[-2]}/{parts[-1]}" if len(parts) >= 2 else "unknown/unknown"
        else:
            repo = "unknown/unknown"
        number = issue.get("number", 0)
        body = issue.get("body") or ""
        return Record(
            stable_id=stable_id_for("github", f"{repo}:{number}"),
            source="github",
            subject=(issue.get("title") or "")[:280],
            body=body,
            tags=[repo, "issue", issue.get("state", "unknown"), kind],
            created_at=_parse_iso(issue.get("created_at")),
            updated_at=_parse_iso(issue.get("updated_at")),
            extra={
                "repo": repo,
                "number": number,
                "state": issue.get("state"),
                "author": (issue.get("user") or {}).get("login"),
                "assignees": [a.get("login") for a in issue.get("assignees", [])],
                "url": issue.get("html_url"),
                "body_excerpt": body[:500],
                "kind": f"issue-{kind}",
            },
        )

    # Four search endpoints fanned across (kind, subkind, path). Class-level so
    # ``iter_records`` and ``ingest`` share the definition.
    _ENDPOINTS: list[tuple[str, str, str]] = [
        ("pr", "authored", "/search/issues?q=is:pr+author:@me"),
        ("pr", "review-requested", "/search/issues?q=is:pr+review-requested:@me"),
        ("issue", "authored", "/search/issues?q=is:issue+author:@me"),
        ("issue", "assigned", "/search/issues?q=is:issue+assignee:@me"),
    ]

    def iter_records(self, since: datetime | None) -> Iterator[Record]:
        """Yield PR + issue records from all four search endpoints.

        Enforces the read-only scope check up front (same fail-closed policy
        as ``ingest``). Per-endpoint transport failures are logged and
        skipped — partial ingest beats total loss (spec §Error handling).
        The caller (``cli._cmd_ingest``) is responsible for persistence.

        Note: ``since`` is not applied here because the search endpoints
        already return only records the user cares about; the store's
        upsert path handles freshness by stable-ID overwrite.
        """
        self._check_scopes()
        _ = since
        for kind, subkind, path in self._ENDPOINTS:
            try:
                rows = self._api_fetcher(path)
            except Exception as e:  # noqa: BLE001 -- degrade gracefully
                log.warning("gh api %s failed during iter_records: %s", path, e)
                continue
            for row in rows:
                if kind == "pr":
                    yield self._pr_to_record(row)
                else:
                    yield self._issue_to_record(row, kind=subkind)

    def ingest(self, since: datetime | None) -> IngestResult:
        """Count-only path retained for protocol compat + integration tests.

        Persistence happens in ``cli._cmd_ingest`` via ``iter_records`` +
        ``store.upsert``. This method fans out over the same endpoints so
        per-endpoint transport failures still land in ``errors`` (which
        ``iter_records`` cannot surface — it just logs and skips).

        Raises ``WriteCapableTokenError`` before any fetch if the token is
        write-capable and the override is unset (fail-closed on credential
        shape).
        """
        self._check_scopes()
        added = 0
        errors: list[str] = []
        for kind, subkind, path in self._ENDPOINTS:
            try:
                for row in self._api_fetcher(path):
                    if kind == "pr":
                        _ = self._pr_to_record(row)
                    else:
                        _ = self._issue_to_record(row, kind=subkind)
                    added += 1
            except Exception as e:  # noqa: BLE001 -- ingest degrades gracefully
                errors.append(f"{path}: {e}")
        return IngestResult(added=added, updated=0, skipped=0, errors=errors)

    def search(self, ast: QueryAST) -> list[Record]:
        """M1b passthrough: iterate PRs from the authored endpoint and apply
        the subset of AST filters this source understands. Real dispatch lives
        in M2's Store (SQLite/FTS5)."""
        out: list[Record] = []
        for pr in self._api_fetcher("/search/issues?q=is:pr+author:@me"):
            r = self._pr_to_record(pr)
            wanted_state = ast.extra.get("state")
            if wanted_state and r.extra["state"] != wanted_state:
                continue
            wanted_author = ast.extra.get("author")
            if wanted_author and r.extra["author"] != wanted_author.lstrip("@"):
                continue
            out.append(r)
        return out


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
