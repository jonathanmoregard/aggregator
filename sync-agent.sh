#!/usr/bin/env bash
# Spins off a Claude agent to commit and push any changes in a git repo.
# Designed to be run from system cron — no Claude Code session required.
# Place this script inside the repo it should sync; path is inferred automatically.

set -euo pipefail

# Normally the script lives in the repo it syncs and infers the path from its
# own location. SYNC_REPO overrides that, so a repo can be synced without
# planting this file inside it — required for public forks, where local
# automation must not land in the tree or leak into an upstream PR (the
# superpowers fork gitignores this script for exactly that reason, which is
# also how it went missing and left its cron erroring on every run).
REPO="${SYNC_REPO:-$(cd "$(dirname "$0")" && pwd)}"
CLAUDE="$(command -v claude)"

if [ ! -d "$REPO/.git" ]; then
  echo "sync-agent: $REPO is not a git repository — aborting" >&2
  exit 1
fi

cd "$REPO"

# Surface failures. A cron job has nobody watching its output: aggregator's
# sync failed on 4 consecutive runs (Claude usage limit, then a refused
# connection) and went unnoticed for six days because the only evidence was a
# line appended to sync.log. Same shape as the sota-watch runner sitting red
# for 11 days on an expired token.
#
# Pattern lifted from /etc/nixos/home/sota-watch.nix: write the log marker
# first (always works, headless included), then attempt the desktop
# notification best-effort. A missing notification daemon must not become a
# second failure, but the fallback is logged so it stays diagnosable.
notify_failure() {
  local reason="$1"
  echo "sync-agent: FAILED in $REPO — $reason" >&2

  if ! command -v notify-send >/dev/null 2>&1; then
    echo "sync-agent: notify-send not on PATH — failure recorded in log only" >&2
    return 0
  fi

  # cron inherits no session bus; point at the user bus the way the
  # home-manager units do (autodoro.nix uses this same address).
  export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/$(id -u)/bus}"

  if ! notify-send -u critical "repo-autosync FAILED: $(basename "$REPO")" \
    "$reason

Repo: $REPO
Log:  $REPO/sync.log"; then
    echo "sync-agent: notify-send failed (no notification daemon on session bus?) — failure recorded in log only" >&2
  fi
}

# Catch-all for anything set -e kills that isn't handled explicitly below
# (detached HEAD, unreadable file, git blowing up mid-run).
trap 'rc=$?; notify_failure "unexpected error (exit $rc) at line $LINENO"' ERR

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
  notify_failure "Oversized file(s) pending — nothing committed.
$oversized
Move the data out of the repo or .gitignore it."
  exit 1
fi

# Abort if gitleaks finds secrets in anything about to be committed. Without
# this guard an accidentally-dropped secret gets auto-pushed to the remote.
if ! command -v gitleaks >/dev/null 2>&1; then
  echo "sync-agent: gitleaks not installed — refusing to auto-push without secret scan" >&2
  notify_failure "gitleaks not on PATH — refusing to auto-push without a secret scan. Nothing committed."
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
  notify_failure "gitleaks found secrets — nothing committed.
$leaked
Remove the secret or add a gitleaks allowlist entry."
  exit 1
fi

BRANCH="$(git symbolic-ref --short HEAD)"

# Give the agent a way to ignore runtime state instead of committing it.
#
# Without this it had Read but no write affordance, so every generated file
# was a forced choice between committing it and leaving the worktree dirty —
# and it committed: 1496 files / 5.7 MB of session-reflection logs landed in
# ~/.claude in one run, with handoff.log mutating every session afterwards.
#
# The grant is a single generated command rather than the Edit tool because
# --allowedTools cannot path-scope Edit/Write (probed 2026-08-01), so "let it
# edit .gitignore" would in practice mean unrestricted edit on an unattended
# cron agent. This can only append ignore rules to one file in one repo.
#
# Regenerated every run so improvements propagate with the script, and kept
# under .git/ so it can never itself become a pending change.
case "${SYNC_IGNORE_TARGET:-gitignore}" in
  exclude)   IGNORE_FILE="$REPO/.git/info/exclude"; IGNORE_LABEL=".git/info/exclude (local-only, never committed)" ;;
  gitignore) IGNORE_FILE="$REPO/.gitignore";        IGNORE_LABEL=".gitignore (tracked — commit it)" ;;
  *) echo "sync-agent: SYNC_IGNORE_TARGET must be 'gitignore' or 'exclude'" >&2; exit 1 ;;
esac

IGNORE_TOOL="$REPO/.git/sync-agent-ignore"
{
  printf '#!/usr/bin/env bash\n'
  printf '# Generated by sync-agent.sh on every run — edits are overwritten.\n'
  printf 'set -euo pipefail\n'
  printf 'REPO=%q\n' "$REPO"
  printf 'TARGET=%q\n' "$IGNORE_FILE"
  cat <<'HELPER'
reason="runtime state"
patterns=()
while [ $# -gt 0 ]; do
  case "$1" in
    --reason) reason="${2-}"; shift 2 ;;
    --) shift; while [ $# -gt 0 ]; do patterns+=("$1"); shift; done ;;
    -*) echo "sync-agent-ignore: unknown flag: $1" >&2; exit 2 ;;
    *) patterns+=("$1"); shift ;;
  esac
done

if [ ${#patterns[@]} -eq 0 ]; then
  echo 'usage: sync-agent-ignore [--reason "why"] PATTERN [PATTERN...]' >&2
  exit 2
fi

# Fail the whole call on a bad pattern rather than applying it partially —
# a half-applied ignore set is harder to reason about than none.
for p in "${patterns[@]}"; do
  case "$p" in
    "" | " "*) echo "sync-agent-ignore: refusing empty/blank pattern" >&2; exit 1 ;;
    "!"*) echo "sync-agent-ignore: refusing negation '$p' — un-ignoring can expose files an existing rule deliberately hides" >&2; exit 1 ;;
    "*" | "**" | "/" | "." | "./" | "*/" | "**/")
      echo "sync-agent-ignore: refusing '$p' — that ignores the entire repo" >&2; exit 1 ;;
  esac
  if [ "$p" != "${p%$'\n'*}" ]; then
    echo "sync-agent-ignore: refusing pattern containing a newline" >&2; exit 1
  fi
  tracked="$(git -C "$REPO" ls-files -- "$p" 2>/dev/null | head -3 || true)"
  if [ -n "$tracked" ]; then
    {
      echo "sync-agent-ignore: refusing '$p' — it matches files already tracked in git:"
      printf '%s\n' "$tracked" | sed 's/^/    /'
      echo "  Ignoring a tracked file is a no-op; git keeps committing it."
      echo "  If it is genuinely runtime state, 'git rm --cached' it first, then re-run this."
    } >&2
    exit 1
  fi
done

new=()
for p in "${patterns[@]}"; do
  if [ -f "$TARGET" ] && grep -qxF -- "$p" "$TARGET"; then
    echo "sync-agent-ignore: '$p' already present — skipping"
  else
    new+=("$p")
  fi
done

if [ ${#new[@]} -eq 0 ]; then
  exit 0
fi

mkdir -p "$(dirname "$TARGET")"
{
  printf '\n# %s (repo-autosync, %s)\n' "$reason" "$(date -u +%Y-%m-%d)"
  printf '%s\n' "${new[@]}"
} >> "$TARGET"

echo "sync-agent-ignore: added to $TARGET:"
printf '  %s\n' "${new[@]}"
HELPER
} > "$IGNORE_TOOL"
chmod 755 "$IGNORE_TOOL"

PROMPT="There are uncommitted changes in $REPO.

FIRST, triage them. Generated and runtime files must NOT be committed: logs, caches, build output, per-run or per-session state, PID and lock files, timestamped .bak-* backups, coverage and profiling output, downloaded artifacts — anything a tool rewrites on its own. To ignore those instead of committing them, run:

  $IGNORE_TOOL --reason \"<short why>\" <pattern> [<pattern>...]

It appends to $IGNORE_LABEL. It refuses a pattern that matches already-tracked files; when a tracked file is genuinely runtime state, 'git rm --cached' it first and then ignore it. Check your work with 'git check-ignore -v <path>'.

THEN commit what is left — source, config, docs, notes. Group into logical commits using your own judgment: related files together, unrelated changes separate. Use concise conventional commit messages (e.g. 'chore: sync ghostty config'). Commit the ignore-rule change too when you made one.

If you cannot tell whether something is runtime state or real content, leave it untracked and uncommitted rather than guessing — that outcome is logged for human review, not treated as a failure.

After all commits are done, push origin $BRANCH once. Do nothing else."

# Capture the agent's output as well as tee it to the log. Claude Code prints
# "You've hit your org's monthly usage limit" and "API Error: ..." to stdout
# and still exits 0, so the exit code alone cannot tell an API failure apart
# from a finished sync — aggregator logged three usage-limit runs and one
# ConnectionRefused, every one of them reported as "worktree still dirty".
# The text is kept only to explain a failure the checks below detect, never to
# declare one, so a commit message containing "API Error" cannot fake a fail.
claude_rc=0
claude_out="$(
  "$CLAUDE" \
    --model claude-sonnet-4-6 \
    --allowedTools "Read Bash(git status:*) Bash(git diff:*) Bash(git add:*) Bash(git commit:*) Bash(git push:*) Bash(git log:*) Bash(git rm:*) Bash(git check-ignore:*) Bash($IGNORE_TOOL:*)" \
    -p "$PROMPT" 2>&1
)" || claude_rc=$?
printf '%s\n' "$claude_out"

api_hint="$(printf '%s\n' "$claude_out" |
  grep -iE 'usage limit|rate limit|API Error|Invalid API key|Credit balance|/login' |
  head -3 || true)"
if [ -n "$api_hint" ]; then
  api_hint="
Claude reported:
$api_hint"
fi

if [ "$claude_rc" -ne 0 ]; then
  notify_failure "claude exited $claude_rc — changes still uncommitted.
Usual causes: usage limit reached, expired auth (claude /login), or API unreachable.$api_hint"
  exit 1
fi

# The agent is prompted to commit everything and push, but it is an agent —
# it can stop early, or push can fail. Verify rather than assume, so a
# half-done sync is not indistinguishable from a clean one.
#
# Only *tracked* changes count as incomplete. The agent is deliberately given
# judgment over what belongs in the repo, and it exercises it: aggregator's
# sync succeeded and pushed on every run for days while being reported FAILED,
# because one ephemeral `mission.md.bak-*` was left untracked. That is a
# permanent red — nothing ever removes the file — so an untracked-only
# leftover is logged, not escalated.
tracked_dirty="$(git status --porcelain --untracked-files=no)"
unpushed="$(git log --oneline "origin/$BRANCH..$BRANCH" 2>/dev/null || true)"

if [ -n "$tracked_dirty" ]; then
  notify_failure "claude exited 0 but tracked changes are still uncommitted — sync incomplete.
$(printf '%s\n' "$tracked_dirty" | head -20)$api_hint"
  exit 1
fi

if [ -n "$unpushed" ]; then
  notify_failure "Commits were made but not pushed — origin/$BRANCH is behind.
$(printf '%s\n' "$unpushed" | head -10)$api_hint"
  exit 1
fi

skipped="$(git ls-files --others --exclude-standard)"
if [ -n "$skipped" ]; then
  {
    echo "sync-agent: synced $REPO; left $(printf '%s\n' "$skipped" | wc -l) file(s) untracked by agent judgment:"
    printf '%s\n' "$skipped" | head -20 | sed 's/^/  ?? /' || true
    echo "  (.gitignore them if this is recurring runtime state)"
  } >&2
fi
