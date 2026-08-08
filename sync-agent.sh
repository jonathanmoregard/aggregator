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

# The exact set of files this run could commit: tracked changes vs HEAD plus
# untracked-but-not-ignored. Both guards below scan only this set. Scanning
# the whole worktree instead would walk gitignored runtime dirs — in a repo
# like ~/.claude that means hundreds of MB of session transcripts, which is
# both far too slow for cron and a false-positive farm.
candidates=()
while IFS= read -r -d '' f; do
  [ -f "$f" ] || continue # skip deletions
  candidates+=("$f")
done < <(
  {
    git diff --name-only -z HEAD
    git ls-files --others --exclude-standard -z
  } | sort -zu
)

if [ ${#candidates[@]} -eq 0 ]; then
  exit 0
fi

# Abort if any pending file is oversized. Bulk data landing in a source repo
# (datasets, transcript dumps, DB snapshots, archives) is nearly always a
# misplacement, and once pushed it is only removable by a history rewrite —
# so this fails closed. Raise the ceiling per repo with SYNC_MAX_FILE_BYTES.
MAX_FILE_BYTES="${SYNC_MAX_FILE_BYTES:-5242880}" # 5 MiB
oversized=""
for f in "${candidates[@]}"; do
  size="$(stat -c %s "$f")"
  if [ "$size" -gt "$MAX_FILE_BYTES" ]; then
    oversized+="  $(numfmt --to=iec --suffix=B "$size")  $f"$'\n'
  fi
done

if [ -n "$oversized" ]; then
  {
    echo "sync-agent: oversized file(s) pending in $REPO — aborting auto-sync"
    echo "  ceiling: $(numfmt --to=iec --suffix=B "$MAX_FILE_BYTES") (override: SYNC_MAX_FILE_BYTES)"
    echo "$oversized"
    echo "Resolve by moving the data out of the repo or adding it to .gitignore."
  } >&2
  exit 1
fi

# Abort if gitleaks finds secrets in anything about to be committed. Without
# this guard an accidentally-dropped secret gets auto-pushed to the remote.
if ! command -v gitleaks >/dev/null 2>&1; then
  echo "sync-agent: gitleaks not installed — refusing to auto-push without secret scan" >&2
  exit 1
fi

leaked=""
for f in "${candidates[@]}"; do
  if ! gitleaks dir --no-banner --redact --exit-code 1 --log-level error "$f" >&2; then
    leaked+="  $f"$'\n'
  fi
done

if [ -n "$leaked" ]; then
  {
    echo "sync-agent: gitleaks found secrets in $REPO — aborting auto-sync"
    echo "$leaked"
  } >&2
  exit 1
fi

BRANCH="$(git symbolic-ref --short HEAD)"

PROMPT="There are uncommitted changes in $REPO. Group the changes into logical commits using your own judgment — related files should go together, unrelated changes should be separate commits. Use concise conventional commit messages (e.g. 'chore: sync ghostty config'). After all commits are done, push origin $BRANCH once. Do nothing else."

"$CLAUDE" \
  --model claude-sonnet-4-6 \
  --allowedTools "Read Bash(git status:*) Bash(git diff:*) Bash(git add:*) Bash(git commit:*) Bash(git push:*) Bash(git log:*) Bash(git rm:*)" \
  -p "$PROMPT"
