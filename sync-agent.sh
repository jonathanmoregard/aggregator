#!/usr/bin/env bash
# Spins off a Claude agent to commit and push any changes in a git repo.
# Designed to be run from system cron — no Claude Code session required.
# Place this script inside the repo it should sync; path is inferred automatically.

set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
CLAUDE="$(command -v claude)"

cd "$REPO"

# Skip if nothing to commit
if [ -z "$(git status --porcelain)" ]; then
  exit 0
fi

# Abort if gitleaks finds secrets in the working tree (staged or unstaged).
# Without this guard, --dangerously-skip-permissions + auto-push could leak
# an accidentally-dropped secret to the remote.
if command -v gitleaks >/dev/null 2>&1; then
  if ! gitleaks detect --no-banner --no-git --redact --exit-code 1 >&2; then
    echo "sync-agent: gitleaks found secrets in $REPO — aborting auto-sync" >&2
    exit 1
  fi
else
  echo "sync-agent: gitleaks not installed — refusing to auto-push without secret scan" >&2
  exit 1
fi

BRANCH="$(git symbolic-ref --short HEAD)"

PROMPT="There are uncommitted changes in $REPO. Group the changes into logical commits using your own judgment — related files should go together, unrelated changes should be separate commits. Use concise conventional commit messages (e.g. 'chore: sync ghostty config'). After all commits are done, push origin $BRANCH once. Do nothing else."

"$CLAUDE" \
  --model claude-sonnet-4-6 \
  --allowedTools "Read Bash(git status:*) Bash(git diff:*) Bash(git add:*) Bash(git commit:*) Bash(git push:*) Bash(git log:*) Bash(git rm:*)" \
  -p "$PROMPT"
