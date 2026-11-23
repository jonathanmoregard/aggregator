#!/usr/bin/env bash
# Commits any changes in a git repo with a confined Claude agent, then scans and pushes.
# Designed to be run from system cron — no Claude Code session required.
# Place this script inside the repo it should sync; path is inferred automatically.

set -euo pipefail
set -E # without errtrace the ERR trap below never fires inside functions

# Normally the script lives in the repo it syncs and infers the path from its
# own location. SYNC_REPO overrides that, so a repo can be synced without
# planting this file inside it — required for public forks, where local
# automation must not land in the tree or leak into an upstream PR (the
# superpowers fork gitignores this script for exactly that reason, which is
# also how it went missing and left its cron erroring on every run).
REPO="${SYNC_REPO:-$(cd "$(dirname "$0")" && pwd)}"
# Checked only where the agent is needed: pushing commits already made does
# not need it, and a lookup failing here would exit before the ERR trap exists.
CLAUDE="$(command -v claude || true)"

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
# set -E makes command substitutions inherit this trap; only the main shell
# notifies, or one failure would raise two notifications.
trap 'rc=$?; if [ "$BASHPID" = "$$" ]; then notify_failure "unexpected error (exit $rc) at line $LINENO"; fi' ERR

# Asked of git rather than tested as a .git directory: in a linked worktree or
# a submodule .git is a file.
if ! git -C "$REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "sync-agent: $REPO is not a git repository — aborting" >&2
  notify_failure "not a git repository — nothing synced."
  exit 1
fi

cd "$REPO"

# What origin already has, read before the agent runs. The agent can write
# files under .git (and so move refs/remotes/origin/*), so the outgoing range
# is computed from these SHAs, held in this shell, never from refs read later.
# Space-joined "^sha" excludes with no trailing blank: gitleaks splits
# --log-opts on spaces, and a trailing one made it scan nothing (measured).
origin_refs="$(git for-each-ref --format='^%(objectname)' refs/remotes/origin)"
mapfile -t origin_excludes <<<"$origin_refs"
ORIGIN_EXCLUDES="${origin_excludes[*]}"

BRANCH="$(git symbolic-ref --short HEAD)"
# Where origin has this branch, read now and held in this shell for the same
# reason: push_branch uses it to decide whether a push is needed at all, and
# an agent that writes refs/remotes/origin/$BRANCH could otherwise make its
# own commit look already-pushed and skip the scan-and-push entirely.
# Empty (missing or unreadable ref) means "push", never "in sync".
ORIGIN_BRANCH_SHA="$(git rev-parse --verify -q "refs/remotes/origin/$BRANCH^{commit}" || true)"

# Raise the ceiling per repo with SYNC_MAX_FILE_BYTES.
MAX_FILE_BYTES="${SYNC_MAX_FILE_BYTES:-5242880}" # 5 MiB
# A non-integer would make every [ -gt ] below error, read as "not oversized".
# Over 18 digits it no longer fits bash's 64-bit test and errors the same way.
case "$MAX_FILE_BYTES" in
  "" | *[!0-9]* | ???????????????????*)
    notify_failure "SYNC_MAX_FILE_BYTES must be a whole number of bytes, got '$MAX_FILE_BYTES' — nothing synced."
    exit 1 ;;
esac

require_gitleaks() {
  if ! command -v gitleaks >/dev/null 2>&1; then
    echo "sync-agent: gitleaks not installed — refusing to auto-push without secret scan" >&2
    notify_failure "gitleaks not on PATH — refusing to auto-push without a secret scan. Nothing was pushed."
    exit 1
  fi
}

# Bulk data landing in a source repo (datasets, transcript dumps, DB snapshots,
# archives) is nearly always a misplacement, and once pushed it is only
# removable by a history rewrite — so oversized files fail closed.
# $1 is the formatted list of oversized files, $2 what did not happen ("committed"/"pushed").
abort_oversized() {
  {
    echo "sync-agent: oversized file(s) in $REPO — aborting auto-sync"
    echo "  ceiling: $(numfmt --to=iec --suffix=B "$MAX_FILE_BYTES") (override: SYNC_MAX_FILE_BYTES)"
    echo "$1"
    echo "Resolve by moving the data out of the repo or adding it to .gitignore."
  } >&2
  notify_failure "Oversized file(s) — nothing $2.
$1
Move the data out of the repo or .gitignore it."
  exit 1
}

# An origin ref that cannot be read must not look like "not ahead": the push
# then runs and its own error surfaces.
ahead_of_origin() {
  local count
  git rev-parse --verify -q "refs/remotes/origin/$1" >/dev/null || return 0
  count="$(git rev-list --count "origin/$1..HEAD" 2>/dev/null)" || return 0
  [ "$count" -gt 0 ]
}

# The agent only commits; pushing happens here. Every push, including commits
# that were never scanned (hand-made work in progress, an earlier failed push),
# is scanned and size-checked over its outgoing range before it goes out.
# The range ends at one resolved SHA and that SHA is what gets pushed, so a
# commit landing on the branch mid-scan cannot ride along unscanned.
push_branch() {
  local branch sha push_out range leaks_out objects sizes blob_oids oversized="" type size oid path
  branch="$(git symbolic-ref --short HEAD)"
  sha="$(git rev-parse --verify 'HEAD^{commit}')"

  require_gitleaks
  range="$sha${ORIGIN_EXCLUDES:+ $ORIGIN_EXCLUDES}"

  # Whether a push is needed is decided by where origin has this branch, never
  # by "are there outgoing objects": ff-merging main to a commit that only
  # origin/side carries leaves nothing to scan while origin/main still has to
  # be moved. The SHA is the one held since before the agent ran, and it is
  # only an answer while HEAD is still that branch; anything else pushes.
  if [ "$branch" = "$BRANCH" ] && [ "$ORIGIN_BRANCH_SHA" = "$sha" ]; then
    return 0 # origin already has this branch here: nothing to scan or push
  fi

  # $range is deliberately unquoted: it is several rev-list arguments.
  if ! objects="$(git rev-list --objects $range 2>&1)"; then
    notify_failure "could not list the commits about to be pushed — nothing pushed.
$(printf '%s\n' "$objects" | head -10)"
    exit 1
  fi

  # Only the scans are skipped when origin already has every object (the ref
  # still moves below); with nothing outgoing there is nothing to scan.
  if [ -n "$objects" ]; then
    # Assigned, not fed through < <(…): a failing cat-file there would leave the
    # loop reading nothing and the size check passing.
    sizes="$(printf '%s\n' "$objects" | git cat-file --batch-check='%(objecttype) %(objectsize) %(objectname) %(rest)')"
    blob_oids=""
    while read -r type size oid path; do
      [ "$type" = blob ] || continue
      blob_oids+="$oid"$'\n'
      if [ "$size" -gt "$MAX_FILE_BYTES" ]; then
        oversized+="  $(numfmt --to=iec --suffix=B "$size")  $path"$'\n'
      fi
    done <<<"$sizes"
    [ -z "$oversized" ] || abort_oversized "$oversized" pushed

    # -m: without it git log shows no diff for merge commits, so a secret added
    # only in a merge resolution would go out unscanned.
    # gitleaks exits 1 for a config error as well as for leaks, so the message
    # cannot tell them apart; its output below does.
    # -v: without it gitleaks prints no findings and the notification names no file.
    if ! leaks_out="$(gitleaks git -v --no-color --no-banner --redact --exit-code 1 --log-level error --log-opts="-m $range" . 2>&1)"; then
      notify_failure "gitleaks failed or found secrets in commits about to be pushed — nothing pushed.
$(printf '%s\n' "$leaks_out" | head -15)"
      exit 1
    fi
    # The scan above reads `git log -p`, which prints "Binary files differ"
    # instead of content for anything git calls binary (a NUL byte in the blob,
    # or -diff in .gitattributes), so a key in such a file never reaches it.
    # The outgoing blobs are therefore scanned as content as well; cat-file's
    # own "<oid> blob <size>" header lines are just extra text to the scanner.
    if [ -n "$blob_oids" ] &&
      ! leaks_out="$(printf '%s' "$blob_oids" | git cat-file --batch |
        gitleaks stdin -v --no-color --no-banner --redact --exit-code 1 --log-level error 2>&1)"; then
      notify_failure "gitleaks failed or found secrets in file content about to be pushed — nothing pushed.
$(printf '%s\n' "$leaks_out" | head -15)"
      exit 1
    fi
    # gitleaks git reads diffs only; commit messages go out too (measured: a key
    # only in a message passed the scan above).
    # $range is deliberately unquoted, as above.
    if ! leaks_out="$(git log --format=%B $range | gitleaks stdin -v --no-color --no-banner --redact --exit-code 1 --log-level error 2>&1)"; then
      notify_failure "gitleaks failed or found secrets in commit messages about to be pushed — nothing pushed.
$(printf '%s\n' "$leaks_out" | head -15)"
      exit 1
    fi
  fi

  if ! push_out="$(git push origin "$sha:refs/heads/$branch" 2>&1)"; then
    notify_failure "git push origin $branch failed — commits are local only.
$(printf '%s\n' "$push_out" | head -10)"
    exit 1
  fi
  # A branch with no upstream gets origin's; an existing upstream (a fork
  # tracking another remote, even one whose ref is gone) is left alone. Written
  # as config, not --set-upstream-to, which needs a tracking ref that a
  # single-branch clone never creates.
  if ! git config --get "branch.$branch.remote" >/dev/null; then
    git config "branch.$branch.remote" origin
    git config "branch.$branch.merge" "refs/heads/$branch"
  fi
}

BRANCH="$(git symbolic-ref --short HEAD)"

# Nothing to commit. Commits left behind by an earlier failed push still go out.
# Assigned rather than tested inline: a failing git status inside [ -z "$(…)" ]
# reads as "clean" and set -e never sees it.
status="$(git status --porcelain)"
if [ -z "$status" ]; then
  if ahead_of_origin "$BRANCH"; then
    push_branch
  fi
  exit 0
fi

# A dirty tree must not strand commits an earlier run left behind. Every exit
# between here and the agent (oversized candidate, leaked candidate, no claude,
# bad SYNC_IGNORE_TARGET) happens before any push, so without this those
# commits would wait for a clean tree that may never arrive — the tree is
# dirty precisely because something keeps failing. push_branch scans whatever
# it sends, so pushing before the pre-commit checks weakens nothing.
if ahead_of_origin "$BRANCH"; then
  push_branch
fi

# The exact set of files this run could commit: tracked changes vs HEAD plus
# untracked-but-not-ignored. Both guards below scan only this set. Scanning
# the whole worktree instead would walk gitignored runtime dirs — in a repo
# like ~/.claude that means hundreds of MB of session transcripts, which is
# both far too slow for cron and a false-positive farm.
#
# Listed into a file first: inside < <(…) a failing git diff or ls-files is
# invisible to set -e and reads as "nothing to commit".
cand_file="$(mktemp)"
# .sorted too: a failing sort exits the script with that file already created.
trap 'rm -f "$cand_file" "$cand_file.sorted"' EXIT
{
  git diff --name-only -z HEAD
  git ls-files --others --exclude-standard -z
} >"$cand_file"
sort -zu "$cand_file" >"$cand_file.sorted"
mv "$cand_file.sorted" "$cand_file"
candidates=()
while IFS= read -r -d '' f; do
  [ -f "$f" ] || continue # deletions add no content to scan
  candidates+=("$f")
done <"$cand_file"
# No candidates with a dirty tree means only deletions: they still go to the
# agent to be committed; the scans below have nothing to look at.

oversized=""
for f in "${candidates[@]}"; do
  size="$(stat -c %s "$f")"
  if [ "$size" -gt "$MAX_FILE_BYTES" ]; then
    oversized+="  $(numfmt --to=iec --suffix=B "$size")  $f"$'\n'
  fi
done
[ -z "$oversized" ] || abort_oversized "$oversized" committed

# Abort if gitleaks finds secrets in anything about to be committed. Without
# this guard an accidentally-dropped secret gets auto-pushed to the remote.
require_gitleaks

leaked=""
for f in "${candidates[@]}"; do
  if ! gitleaks dir --no-banner --redact --exit-code 1 --log-level error "$f" >&2; then
    leaked+="  $f"$'\n'
  fi
done

if [ -n "$leaked" ]; then
  {
    echo "sync-agent: gitleaks failed or found secrets in $REPO — aborting auto-sync"
    echo "$leaked"
  } >&2
  notify_failure "gitleaks failed or found secrets — nothing committed.
$leaked
Remove the secret or add a gitleaks allowlist entry."
  exit 1
fi

if [ -z "$CLAUDE" ]; then
  notify_failure "claude not on PATH — nothing committed."
  exit 1
fi

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
# cron agent. The agent has no Edit/Write tool at all (--tools) and no user
# settings (--setting-sources ''); its Bash surface is the git subcommands and
# this command, which can only append ignore rules to one file in one repo.
# git log and git diff are left out on purpose: their --output=<file> writes
# any file, .git/config and this helper included.
#
# Regenerated every run so improvements propagate with the script, and kept
# under the git dir so it can never itself become a pending change. Paths come
# from git rev-parse --git-path: in a linked worktree .git is a file.
case "${SYNC_IGNORE_TARGET:-gitignore}" in
  exclude)   IGNORE_FILE="$(git rev-parse --path-format=absolute --git-path info/exclude)"; IGNORE_LABEL=".git/info/exclude (local-only, never committed)" ;;
  gitignore) IGNORE_FILE="$REPO/.gitignore";        IGNORE_LABEL=".gitignore (tracked — commit it)" ;;
  *)
    echo "sync-agent: SYNC_IGNORE_TARGET must be 'gitignore' or 'exclude'" >&2
    notify_failure "SYNC_IGNORE_TARGET must be 'gitignore' or 'exclude' — nothing committed."
    exit 1 ;;
esac

IGNORE_TOOL="$(git rev-parse --path-format=absolute --git-path sync-agent-ignore)"
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

# The reason is written as a comment line; a newline in it would start new
# ignore rules that skip every pattern check below.
case "$reason" in
  *$'\n'* | *$'\r'*) echo "sync-agent-ignore: refusing --reason containing a newline" >&2; exit 1 ;;
esac

# Random names no sane runtime-state rule matches: a rule that ignores one of
# them ignores (nearly) everything.
probe_root="sync-agent-probe-$RANDOM$RANDOM"
probe_dir="sync-agent-probe-dir-$RANDOM$RANDOM"

# The probes are matched in a scratch repo, not in $REPO. check-ignore applies
# the repo's own .gitignore and info/exclude on top of core.excludesFile, so a
# whitelist repo (a "/*" rule plus "!/src/") makes every probe ignored no
# matter what $p says, and every pattern gets refused as too broad. An empty
# repo puts only $p in play. Pattern semantics are path-based, so matching
# there answers the same question.
probe_repo="$(mktemp -d)"
rule_file=""
trap 'rm -rf "$probe_repo"; [ -z "$rule_file" ] || rm -f "$rule_file"' EXIT
if ! git init -q "$probe_repo" >/dev/null 2>&1; then
  echo "sync-agent-ignore: could not create a scratch repo to test patterns in" >&2
  exit 1
fi

# Fail the whole call on a bad pattern rather than applying it partially —
# a half-applied ignore set is harder to reason about than none.
for p in "${patterns[@]}"; do
  case "$p" in
    "" | " "*) echo "sync-agent-ignore: refusing empty/blank pattern" >&2; exit 1 ;;
    "!"*) echo "sync-agent-ignore: refusing negation '$p' — un-ignoring can expose files an existing rule deliberately hides" >&2; exit 1 ;;
    "*" | "**" | "/" | "." | "./" | "*/" | "**/" | "/*" | "/**" | "/*/" | "/**/")
      echo "sync-agent-ignore: refusing '$p' — too broad: that ignores the entire repo" >&2; exit 1 ;;
  esac
  case "$p" in
    *$'\n'* | *$'\r'*) echo "sync-agent-ignore: refusing pattern containing a newline" >&2; exit 1 ;;
  esac
  # What the rule would do is asked of git's own ignore matcher, never
  # re-derived here: pathspecs and hand-kept lists disagree with gitignore
  # semantics (anchors, trailing blanks, '**', basename matches at any depth).
  rule_file="$(mktemp)"
  printf '%s\n' "$p" >"$rule_file"
  tracked="$(git -C "$REPO" ls-files -ci --exclude-from="$rule_file")" || exit 1
  broad=""
  for probe in "$probe_root" "$probe_dir/$probe_root" ".gitignore"; do
    probe_rc=0
    git -C "$probe_repo" -c core.excludesFile="$rule_file" \
      check-ignore -q --no-index -- "$probe" || probe_rc=$?
    case "$probe_rc" in
      0) broad="$probe"; break ;;
      1) ;; # not ignored: this probe is fine
      # 128 and anything else is git failing, not an answer. Reading it as
      # "not ignored" would wave the pattern through unchecked.
      *) echo "sync-agent-ignore: git check-ignore failed (exit $probe_rc) while testing '$p' — refusing it" >&2
         exit 1 ;;
    esac
  done
  rm -f "$rule_file"
  rule_file=""
  if [ -n "$broad" ]; then
    echo "sync-agent-ignore: refusing '$p' — it would ignore '$broad': too broad (the whole repo or the ignore file itself)" >&2
    exit 1
  fi
  # sed, not head: head exits early and pipefail turns that into a silent 141.
  tracked="$(printf '%s\n' "$tracked" | sed -n 1,3p)"
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

Inspect them with 'git status -vv' (shows the diffs) and the Read tool.

FIRST, triage them. Generated and runtime files must NOT be committed: logs, caches, build output, per-run or per-session state, PID and lock files, timestamped .bak-* backups, coverage and profiling output, downloaded artifacts — anything a tool rewrites on its own. To ignore those instead of committing them, run:

  $IGNORE_TOOL --reason \"<short why>\" <pattern> [<pattern>...]

It appends to $IGNORE_LABEL. It refuses a pattern that matches already-tracked files; when a tracked file is genuinely runtime state, 'git rm --cached' it first and then ignore it. Check your work with 'git check-ignore -v <path>'.

THEN commit what is left — source, config, docs, notes. Group into logical commits using your own judgment: related files together, unrelated changes separate. Use concise conventional commit messages (e.g. 'chore: sync ghostty config'). Commit the ignore-rule change too when you made one.

If you cannot tell whether something is runtime state or real content, leave it untracked and uncommitted rather than guessing — that outcome is logged for human review, not treated as a failure.

After all commits are done, stop. Do nothing else."

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
    --setting-sources '' \
    --tools "Read,Bash" \
    --disallowedTools "Bash(git push:*)" \
    --allowedTools "Read Bash(git status:*) Bash(git add:*) Bash(git commit:*) Bash(git rm:*) Bash(git check-ignore:*) Bash($IGNORE_TOOL:*)" \
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
  # What it did commit is scanned and pushed now: while claude keeps failing
  # the tree stays dirty, every run ends here, and commits would never leave.
  push_branch
  left="$(git status --porcelain --untracked-files=no)"
  if [ -n "$left" ]; then left="tracked changes remain uncommitted"; else left="no tracked changes remain"; fi
  notify_failure "claude exited $claude_rc — sync incomplete. Anything it committed was scanned and pushed; $left.
Usual causes: usage limit reached, expired auth (claude /login), or API unreachable.$api_hint"
  exit 1
fi

# The agent is prompted to commit everything, but it is an agent — it can
# stop early. Verify rather than assume, so a half-done sync is not
# indistinguishable from a clean one.
#
# Only *tracked* changes count as incomplete. The agent is deliberately given
# judgment over what belongs in the repo, and it exercises it: aggregator's
# sync succeeded and pushed on every run for days while being reported FAILED,
# because one ephemeral `mission.md.bak-*` was left untracked. That is a
# permanent red — nothing ever removes the file — so an untracked-only
# leftover is logged, not escalated.
tracked_dirty="$(git status --porcelain --untracked-files=no)"

if [ -n "$tracked_dirty" ]; then
  # Same reason as the claude-failed path above: the agent may have committed
  # part of its work and left the rest. Exiting here without pushing kept
  # those commits local for as long as the leftovers stayed uncommitted.
  push_branch
  notify_failure "claude exited 0 but tracked changes are still uncommitted — sync incomplete.
$(printf '%s\n' "$tracked_dirty" | head -20)$api_hint"
  exit 1
fi

push_branch

skipped="$(git ls-files --others --exclude-standard)"
if [ -n "$skipped" ]; then
  {
    echo "sync-agent: synced $REPO; left $(printf '%s\n' "$skipped" | wc -l) file(s) untracked by agent judgment:"
    printf '%s\n' "$skipped" | head -20 | sed 's/^/  ?? /' || true
    echo "  (.gitignore them if this is recurring runtime state)"
  } >&2
fi
