# GitHub Keyring GET-only Ingest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (default) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore unattended GitHub ingest permanently by using the existing GitHub CLI keyring credential behind an explicit GET-only, search-endpoint capability.

**Architecture:** `GitHubSource` stops treating broad OAuth scopes as the safety boundary. Its subprocess adapter validates the exact search paths it accepts and always passes `--method GET`; token scopes remain diagnostic only. NixOS stops exporting an expiring agenix PAT and lets the user service use the same keyring credential already proven available inside the user manager.

**Tech Stack:** Python 3.11, pytest, `subprocess.run` argv execution, GitHub CLI, Nix/Home Manager, systemd user units, NixOS VM tests.

---

### Task 1: Pin GET-only GitHub API behavior with failing tests

**Files:**
- Modify: `tests/sources/test_github.py`
- Modify: `tests/test_github_token_status.py`

- [ ] **Step 1: Replace obsolete scope-refusal tests with capability tests**

Add tests equivalent to:

```python
def test_ingest_accepts_write_scoped_keyring_token_without_override(monkeypatch):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("AGGREGATOR_ALLOW_WRITE_TOKEN", raising=False)
    src = GitHubSource(
        _scope_fetcher=lambda: ["repo", "gist", "workflow"],
        _api_fetcher=lambda path: [],
        _gh_token_fetcher=lambda: "keyring-token",
    )
    assert src.ingest(since=None).errors == []


def test_default_api_fetcher_uses_explicit_get():
    fake = subprocess.CompletedProcess(args=["gh"], returncode=0, stdout="", stderr="")
    with patch("aggregator.sources.github.subprocess.run", return_value=fake) as run:
        _default_api_fetcher("/search/issues?q=is:pr+author:@me")
    assert run.call_args.args[0][0:4] == ["gh", "api", "--method", "GET"]


@pytest.mark.parametrize("path", [
    "/repos/o/r/issues/1",
    "--method=DELETE",
    "/search/issues?q=is:pr+author:@me\n--method DELETE",
])
def test_default_api_fetcher_rejects_paths_outside_capability(path):
    with patch("aggregator.sources.github.subprocess.run") as run:
        with pytest.raises(GhApiError, match="refusing"):
            _default_api_fetcher(path)
    run.assert_not_called()
```

Update token-status expectations so broad scopes produce an informational
recommendation naming GET-only enforcement, with no override field or PAT
recommendation.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
uv run pytest -q tests/sources/test_github.py tests/test_github_token_status.py
```

Expected: new write-scope ingest test raises `WriteCapableTokenError`; argv
test lacks `--method GET`; invalid-path tests start mocked subprocess or fail
to raise; status shape still exposes override behavior.

### Task 2: Implement minimal GET-only capability

**Files:**
- Modify: `aggregator/sources/github.py`
- Modify: `aggregator/cli.py`

- [ ] **Step 1: Validate exact generated search paths**

Add compiled full-match expression covering four existing endpoint shapes
plus optional UTC date qualifier:

```python
_SEARCH_PATH = re.compile(
    r"/search/issues\?q=is:"
    r"(?:pr\+(?:author|review-requested)|issue\+(?:author|assignee))"
    r":@me(?:\+updated:>=\d{4}-\d{2}-\d{2})?\Z"
)


def _validate_search_path(path: str) -> None:
    if _SEARCH_PATH.fullmatch(path) is None:
        raise GhApiError(
            "refusing GitHub API path outside GET-only search capability"
        )
```

- [ ] **Step 2: Make every API subprocess explicitly GET-only**

Call validation before `subprocess.run`, then use:

```python
[
    "gh", "api", "--method", "GET", "--paginate",
    "--jq", ".items[]", path,
]
```

Keep `shell=False` default, timeouts, JSONL parsing, and loud `GhApiError`
behavior unchanged. Add `--method GET` to fixed `/rate_limit` diagnostic call.

- [ ] **Step 3: Remove scope enforcement, retain diagnostics**

Delete `WriteCapableTokenError`, `GitHubSource._check_scopes()`, and its call
from `iter_records`. Remove `TokenStatus.override_active`; make broad-scope
recommendations say adapter is safe because operations are constrained to
GET-only search. Remove CLI `override:` output row.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run Task 1 command. Expected: all tests pass.

- [ ] **Step 5: Run fastest file checks**

```bash
uv run ruff check aggregator/sources/github.py aggregator/cli.py \
  tests/sources/test_github.py tests/test_github_token_status.py
uv run python -m py_compile aggregator/sources/github.py aggregator/cli.py
```

Expected: exit 0.

### Task 3: Remove stale PAT contract from aggregator docs and Nix module

**Files:**
- Modify: `README.md`
- Modify: `nix/README.md`
- Modify: `nix/aggregator.nix`

- [ ] **Step 1: Simplify module execution path**

Change `mkExecStart` to accept only `source` and `since`, always returning the
store-pinned command directly. Remove `sources.github.githubTokenFile`; call
GitHub with plain keyring-backed execution path.

- [ ] **Step 2: Update credential documentation**

Replace PAT creation/rotation guidance with: authenticate once through
`gh auth login`; systemd user service consumes GitHub CLI keyring; adapter
permits only explicit GET requests to fixed search endpoints.

- [ ] **Step 3: Check Nix and docs immediately**

```bash
git diff --check
nix flake check --no-build
rg -n "AGGREGATOR_ALLOW_WRITE_TOKEN|githubTokenFile|read-only PAT flow" \
  README.md nix/README.md nix/aggregator.nix
```

Expected: diff check and flake evaluation exit 0; `rg` returns no matches.

- [ ] **Step 4: Run full aggregator gate**

```bash
uv run pytest -q
uv run ruff check .
```

Expected: full suite green, ruff exit 0.

- [ ] **Step 5: Commit aggregator implementation**

```bash
git add aggregator/sources/github.py aggregator/cli.py \
  tests/sources/test_github.py tests/test_github_token_status.py \
  README.md nix/README.md nix/aggregator.nix
git commit -m "fix(github): use keyring behind GET-only capability"
```

### Task 4: Publish aggregator PR

**Files:**
- Modify: PR body only

- [ ] **Step 1: Push branch and open PR against main**

```bash
git push -u origin fix/github-keyring-get-only
gh pr create --base main --head fix/github-keyring-get-only \
  --title "Fix GitHub ingest with keyring-backed GET-only access" \
  --body-file /tmp/aggregator-github-keyring-pr-body.md
```

PR body must include root-cause evidence, explicit capability boundary,
focused/full test results, and required deployment order.

- [ ] **Step 2: Run exact completion gate**

```bash
python3 /home/jonathan/.claude/scripts/dod-check.py --repo /home/jonathan/Repos/aggregator
```

Expected before CI settles: only `CHECKS_PENDING`; after CI: green and
mergeable except Jonathan's merge click.

### Task 5: Write failing NixOS regression assertions

**Files:**
- Modify: `tests/base.nix`
- Modify: `tests/secrets-no-dead-credentials.nix`

- [ ] **Step 1: Create dedicated worktree from current origin/main**

```bash
git -C ~/Repos/nixos-config-worktrees/main fetch origin main
git -C ~/Repos/nixos-config-worktrees/main worktree add \
  ~/Repos/nixos-config-worktrees/aggregator-github-keyring \
  -b feat/aggregator-github-keyring origin/main
```

- [ ] **Step 2: Add RED assertions**

Add `github-readonly-pat` to `deadCredentials`, with owner text naming GitHub
CLI keyring. Replace old secret-guard assertions in `tests/base.nix` with
assertions that rendered unit/wrapper contains neither `github-readonly-pat`
nor `GH_TOKEN`, while still containing store-pinned `gh` and aggregator CLI.

- [ ] **Step 3: Verify RED**

```bash
git add tests/base.nix tests/secrets-no-dead-credentials.nix
nix build .#checks.x86_64-linux.secrets-no-dead-credentials -L
nix build .#checks.x86_64-linux.vm-base -L
```

Expected: dead-credential check reports declared `github-readonly-pat`;
`vm-base` reports old secret/GH_TOKEN wiring.

### Task 6: Implement NixOS credential-owner fix

**Files:**
- Modify: `flake.nix`
- Modify: `flake.lock`
- Modify: `hosts/dellan/default.nix`
- Modify: `modules/nixos/aggregator-ingest-timer.nix`

- [ ] **Step 1: Pin reviewed aggregator commit**

Set `aggregator-src` URL revision to pushed implementation commit and run
`nix flake lock` (never `nix flake update`). Stage files before evaluation.

- [ ] **Step 2: Remove static-token consumption**

Remove host `age.secrets.github-readonly-pat` declaration. Simplify
`ingestScript` to validate CA bundle then exec store-pinned aggregator; retain
pinned `gh`, error propagation, notifications, timeouts, and cadence. Update
comments and failure toast text.

- [ ] **Step 3: Run cheap checks**

```bash
git add -A
nix eval --no-warn-dirty .#checks.x86_64-linux.vm-base.drvPath
nix build --no-link --print-out-paths \
  .#nixosConfigurations.dellan.config.home-manager.users.jonathan.home.path
nix build .#checks.x86_64-linux.secrets-no-dead-credentials -L
```

Expected: all exit 0; generated wrapper contains no secret read or GH_TOKEN.

- [ ] **Step 4: Verify GREEN in automated VM**

```bash
nix build .#checks.x86_64-linux.vm-base -L
```

Expected: `vm-base` passes including wrapper/unit assertions and notifier.

- [ ] **Step 5: Exercise interactive VM branches**

Start `nix run .#feature-vm`, connect over SSH, inspect unit, and start it with
no GitHub credential. Expected: GitHub source fails loudly, remaining sources
finish, watermark stays unchanged, unit exits non-zero, notifier journal edge
fires. Stop VM cleanly.

- [ ] **Step 6: Exercise real-host keyring success path**

Invoke built `aggregator-ingest` wrapper through transient user-manager unit.
Expected: GitHub source finishes with zero errors, unit exits 0, GitHub
watermark advances, token text absent from journal/argv.

- [ ] **Step 7: Commit NixOS change with required trailer**

Commit message must include `Type: risky`, current rebase proof, `vm-base` and
dead-credential gates, interactive smoke evidence, risky markers, unchanged
`feature-vm.nix`, and real-host behavioral evidence.

### Task 7: Publish NixOS PR and drive both ladders green

**Files:**
- Modify: PR body only

- [ ] **Step 1: Push and open NixOS PR**

Push `feat/aggregator-github-keyring`; open PR against default branch. Body
must state aggregator PR merges first, then NixOS PR triggers auto-deploy.

- [ ] **Step 2: Measure both repos**

```bash
python3 /home/jonathan/.claude/scripts/dod-check.py --repo /home/jonathan/Repos/aggregator
python3 /home/jonathan/.claude/scripts/dod-check.py --repo \
  /home/jonathan/Repos/nixos-config-worktrees/aggregator-github-keyring
```

Work every reported blocker. Pending checks get watch/follow-up; never merge.

- [ ] **Step 3: Handoff only fresh green state**

Report full clickable URLs, current checks, merge order, and post-deploy
verification. Jonathan performs merge clicks.
