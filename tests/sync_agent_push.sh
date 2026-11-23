#!/usr/bin/env bash
# Runs sync-agent.sh against scratch repos with a stub `claude` that only
# commits, proving the script itself does the push. Usage: tests/sync_agent_push.sh
set -uo pipefail

SCRIPT="$(cd "$(dirname "$0")/.." && pwd)/sync-agent.sh"
WORK="$(mktemp -d /tmp/sync-agent-test.XXXXXX)"
fail=0

check() {
  if [ "$2" = "$3" ]; then
    echo "PASS $1"
  else
    echo "FAIL $1: expected '$3', got '$2'"
    fail=1
  fi
}

mkdir "$WORK/bin"
cat >"$WORK/bin/claude" <<'EOF'
#!/usr/bin/env bash
printf '<%s>' "$@" >>"$CLAUDE_ARGS"
echo >>"$CLAUDE_ARGS"
git add -A && git commit -q -m "stub: sync"
[ -z "${STUB_DETACH:-}" ] || git checkout -q --detach
[ -z "${STUB_AFTER:-}" ] || bash -c "$STUB_AFTER"
exit "${STUB_RC:-0}"
EOF
cat >"$WORK/bin/gitleaks" <<'EOF'
#!/usr/bin/env bash
[ -z "${GITLEAKS_REAL:-}" ] || exec "$GITLEAKS_REAL" "$@"
printf '<%s>' "$@" >>"$GITLEAKS_LOG"
echo >>"$GITLEAKS_LOG"
if [ "$1" = git ]; then
  [ -z "${GITLEAKS_GIT_HOOK:-}" ] || bash -c "$GITLEAKS_GIT_HOOK"
  exit "${GITLEAKS_GIT_RC:-0}"
fi
if [ "$1" = stdin ]; then
  cat >/dev/null
  exit "${GITLEAKS_STDIN_RC:-0}"
fi
exit "${GITLEAKS_DIR_RC:-0}"
EOF
cat >"$WORK/bin/notify-send" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >>"$NOTIFY_LOG"
EOF
chmod +x "$WORK/bin/"*

new_pair() {
  git init -q --bare -b main "$WORK/$1-origin.git"
  git clone -q "$WORK/$1-origin.git" "$WORK/$1" 2>/dev/null
  git -C "$WORK/$1" config user.email t@example.com
  git -C "$WORK/$1" config user.name t
  git -C "$WORK/$1" config core.hooksPath /dev/null
  echo base >"$WORK/$1/base.txt"
  git -C "$WORK/$1" add -A
  git -C "$WORK/$1" commit -q -m base
  git -C "$WORK/$1" push -q -u origin main 2>/dev/null
}

local_commit() {
  echo local >"$WORK/$1/local.txt"
  git -C "$WORK/$1" add -A
  git -C "$WORK/$1" commit -q -m "$2"
}

run_sync() {
  : >"$WORK/notify.log"
  : >"$WORK/gitleaks.log"
  : >"$WORK/claude-args.log"
  PATH="${RUN_PATH:-$WORK/bin:$PATH}" NOTIFY_LOG="$WORK/notify.log" CLAUDE_ARGS="$WORK/claude-args.log" \
    GITLEAKS_LOG="$WORK/gitleaks.log" \
    DBUS_SESSION_BUS_ADDRESS=unix:path=/nonexistent \
    SYNC_REPO="$WORK/$1" bash "$SCRIPT" >"$WORK/$1.out" 2>&1
}

notified() {
  grep -c -F -e "$1" "$WORK/notify.log" 2>/dev/null || true
}

new_pair dirty
echo change >"$WORK/dirty/new.txt"
run_sync dirty
check "dirty tree: exit 0" "$?" "0"
check "dirty tree: commit reaches origin" \
  "$(git --git-dir="$WORK/dirty-origin.git" log -1 --format=%s main)" "stub: sync"
check "agent argv: user settings not loaded (empty --setting-sources)" \
  "$(grep -c -F -e '<--setting-sources><>' "$WORK/claude-args.log")" "1"
check "agent argv: git push disallowed" \
  "$(grep -c -F -e '<--disallowedTools><Bash(git push:*)>' "$WORK/claude-args.log")" "1"
check "agent argv: tool surface limited to Read and Bash" \
  "$(grep -c -F -e '<--tools><Read,Bash>' "$WORK/claude-args.log")" "1"
# The whole Bash surface, in order: a grant added by accident (git log, a bare
# Bash, anything beyond these git subcommands) breaks this, where a grep for
# one subcommand would not.
check "agent argv: allowed tools recorded" \
  "$(grep -c -F -e '<--allowedTools><Read Bash(git status:*) Bash(git add:*) Bash(git commit:*) Bash(git rm:*) Bash(git check-ignore:*) Bash(' "$WORK/claude-args.log")" "1"
check "agent argv: the only non-git grant is the generated ignore helper" \
  "$(grep -c -E -e 'Bash\(/[^ >]*/sync-agent-ignore:\*\)>' "$WORK/claude-args.log")" "1"
check "agent argv: no git log/diff, whose --output writes files" \
  "$(grep -c -E -e 'Bash\(git (log|diff):' "$WORK/claude-args.log")" "0"

new_pair ahead
local_commit ahead "local only"
run_sync ahead
check "clean but ahead: exit 0" "$?" "0"
check "clean but ahead: commit reaches origin" \
  "$(git --git-dir="$WORK/ahead-origin.git" log -1 --format=%s main)" "local only"
check "clean but ahead: gitleaks scans the pushed SHA minus what origin had" \
  "$(grep -c -F -e "--log-opts=-m $(git -C "$WORK/ahead" rev-parse HEAD) ^$(git -C "$WORK/ahead" rev-parse HEAD~1)" "$WORK/gitleaks.log")" "1"

new_pair badpush
git -C "$WORK/badpush" remote set-url origin "$WORK/does-not-exist.git"
echo change >"$WORK/badpush/new.txt"
run_sync badpush
check "push failure: exit 1" "$?" "1"
check "push failure: notify_failure fired" \
  "$(notified 'git push origin main failed')" "1"

new_pair nobranch
git -C "$WORK/nobranch" checkout -q -b feature
local_commit nobranch "feature commit"
run_sync nobranch
check "no upstream, clean: exit 0" "$?" "0"
check "no upstream, clean: commit reaches origin" \
  "$(git --git-dir="$WORK/nobranch-origin.git" log -1 --format=%s feature 2>&1)" "feature commit"
check "no upstream, clean: upstream is set" \
  "$(git -C "$WORK/nobranch" rev-parse --abbrev-ref 'feature@{u}' 2>&1)" "origin/feature"
check "no upstream, clean: gitleaks scans commits not on any origin ref" \
  "$(grep -c -F -e "--log-opts=-m $(git -C "$WORK/nobranch" rev-parse HEAD) ^$(git -C "$WORK/nobranch" rev-parse main)" "$WORK/gitleaks.log")" "1"

new_pair aheadfail
local_commit aheadfail "local only"
git -C "$WORK/aheadfail" remote set-url origin "$WORK/does-not-exist.git"
run_sync aheadfail
check "clean but ahead, push failure: exit 1" "$?" "1"
check "clean but ahead, push failure: notify_failure fired" \
  "$(notified 'git push origin main failed')" "1"

new_pair leaky
local_commit leaky "leaky commit"
GITLEAKS_GIT_RC=1 run_sync leaky
check "ahead, gitleaks finds leak: exit 1" "$?" "1"
check "ahead, gitleaks finds leak: nothing pushed" \
  "$(git --git-dir="$WORK/leaky-origin.git" log -1 --format=%s main)" "base"
check "ahead, gitleaks finds leak: notify_failure fired" \
  "$(notified 'gitleaks failed or found secrets')" "1"

new_pair detach
echo change >"$WORK/detach/new.txt"
STUB_DETACH=1 run_sync detach
rc=$?
check "failure inside a function: exit non-zero" "$([ "$rc" -ne 0 ] && echo yes)" "yes"
check "failure inside a function: ERR trap notifies exactly once" \
  "$(notified 'unexpected error')" "1"

new_pair bigahead
head -c 6291456 /dev/zero >"$WORK/bigahead/big.bin"
git -C "$WORK/bigahead" add -A
git -C "$WORK/bigahead" commit -q -m "big commit"
run_sync bigahead
check "clean but ahead, 6 MiB commit: exit 1" "$?" "1"
check "clean but ahead, 6 MiB commit: nothing pushed" \
  "$(git --git-dir="$WORK/bigahead-origin.git" log -1 --format=%s main)" "base"
check "clean but ahead, 6 MiB commit: notify_failure fired" \
  "$(notified 'Oversized')" "1"

new_pair corrupt
local_commit corrupt "local only"
echo 1111111111111111111111111111111111111111 >"$WORK/corrupt/.git/refs/remotes/origin/main"
run_sync corrupt
check "unreadable origin ref: not silently treated as in sync (exit 1)" "$?" "1"
check "unreadable origin ref: failure notified" \
  "$([ "$(wc -l <"$WORK/notify.log")" -ge 1 ] && echo yes)" "yes"

new_pair badtarget
echo change >"$WORK/badtarget/new.txt"
SYNC_IGNORE_TARGET=bogus run_sync badtarget
check "bad SYNC_IGNORE_TARGET: exit 1" "$?" "1"
check "bad SYNC_IGNORE_TARGET: notify_failure fired" \
  "$(notified 'SYNC_IGNORE_TARGET')" "1"

new_pair nogitleaks
local_commit nogitleaks "local only"
mkdir "$WORK/bin-nogl"
for t in bash git dirname basename id head tr; do ln -s "$(command -v "$t")" "$WORK/bin-nogl/$t"; done
ln -s "$WORK/bin/claude" "$WORK/bin/notify-send" "$WORK/bin-nogl/"
RUN_PATH="$WORK/bin-nogl" run_sync nogitleaks
check "gitleaks missing on the push path: exit 1" "$?" "1"
check "gitleaks missing on the push path: message names the real consequence" \
  "$(notified 'refusing to auto-push without a secret scan')" "1"
check "gitleaks missing on the push path: no false claim about commits" \
  "$(notified 'Nothing committed')" "0"

if REAL_GITLEAKS="$(command -v gitleaks)"; then
  new_pair merged
  git -C "$WORK/merged" checkout -q -b side
  echo side >"$WORK/merged/g.txt"
  git -C "$WORK/merged" add -A
  git -C "$WORK/merged" commit -q -m side
  git -C "$WORK/merged" checkout -q main
  echo main >"$WORK/merged/h.txt"
  git -C "$WORK/merged" add -A
  git -C "$WORK/merged" commit -q -m main
  git -C "$WORK/merged" merge -q --no-ff --no-commit side >/dev/null 2>&1
  printf 'aws_access_key_id = %s%s\n' AKIAIOSFODNN7 EXAMPLF >"$WORK/merged/base.txt"
  git -C "$WORK/merged" add -A
  git -C "$WORK/merged" commit -q -m "merge with secret in resolution"
  RUN_PATH="$WORK/bin:$PATH" GITLEAKS_REAL="$REAL_GITLEAKS" run_sync merged
  check "secret added only in a merge resolution: exit 1" "$?" "1"
  check "secret added only in a merge resolution: nothing pushed" \
    "$(git --git-dir="$WORK/merged-origin.git" log -1 --format=%s main)" "base"
else
  echo "SKIP merge-resolution secret scan: real gitleaks not on PATH"
fi

new_pair reftamper
printf 'big.bin\n' >"$WORK/reftamper/.gitignore"
git -C "$WORK/reftamper" add .gitignore && git -C "$WORK/reftamper" commit -q -m ignore
git -C "$WORK/reftamper" push -q origin main 2>/dev/null
head -c 6291456 /dev/zero >"$WORK/reftamper/big.bin"
echo change >"$WORK/reftamper/new.txt"
STUB_AFTER='git add -f big.bin && git commit -q -m sneak && git log -1 --format=%H --output=.git/refs/remotes/origin/main' \
  run_sync reftamper
check "agent moves origin ref: exit 1" "$?" "1"
check "agent moves origin ref: oversized commit not pushed" \
  "$(git --git-dir="$WORK/reftamper-origin.git" log -1 --format=%s main)" "ignore"
check "agent moves origin ref: notify_failure fired" "$(notified 'Oversized')" "1"

new_pair noclaude
local_commit noclaude "local only"
mkdir "$WORK/bin-noclaude"
for t in bash git dirname basename id head numfmt stat sort tr mktemp rm mv; do ln -s "$(command -v "$t")" "$WORK/bin-noclaude/$t"; done
ln -s "$WORK/bin/gitleaks" "$WORK/bin/notify-send" "$WORK/bin-noclaude/"
RUN_PATH="$WORK/bin-noclaude" run_sync noclaude
check "claude missing, clean but ahead: exit 0" "$?" "0"
check "claude missing, clean but ahead: commit reaches origin" \
  "$(git --git-dir="$WORK/noclaude-origin.git" log -1 --format=%s main)" "local only"
echo change >"$WORK/noclaude/new.txt"
RUN_PATH="$WORK/bin-noclaude" run_sync noclaude
check "claude missing, dirty: exit 1" "$?" "1"
check "claude missing, dirty: notify_failure fired once" "$(notified 'claude not on PATH')" "1"

new_pair race
echo change >"$WORK/race/new.txt"
GITLEAKS_GIT_HOOK='echo late >late.txt && git add late.txt && git commit -q -m late' run_sync race
check "commit lands during scan: scanned commit pushed, late one not" \
  "$(git --git-dir="$WORK/race-origin.git" log -1 --format=%s main)" "stub: sync"

new_pair badindex
local_commit badindex "local only"
echo garbage >"$WORK/badindex/.git/index"
run_sync badindex
check "git status fails: exit non-zero" "$([ "$?" -ne 0 ] && echo yes)" "yes"
check "git status fails: nothing pushed" \
  "$(git --git-dir="$WORK/badindex-origin.git" log -1 --format=%s main)" "base"
check "git status fails: notified exactly once" "$(notified 'repo-autosync FAILED')" "1"
check "git status fails: a notification fired" "$(notified 'unexpected error')" "1"

new_pair gl2
local_commit gl2 "local only"
GITLEAKS_GIT_RC=2 run_sync gl2
check "gitleaks errors: message does not claim secrets were found" "$(notified 'gitleaks found secrets')" "0"
check "gitleaks errors: message says scan failed or found secrets" "$(notified 'gitleaks failed or found secrets')" "1"

new_pair agentfail
echo change >"$WORK/agentfail/new.txt"
STUB_RC=1 run_sync agentfail
check "agent commits then exits 1: exit 1" "$?" "1"
check "agent commits then exits 1: no false 'still uncommitted'" "$(notified 'still uncommitted')" "0"
check "agent commits then exits 1: notified once" "$(notified 'repo-autosync FAILED')" "1"
check "agent commits then exits 1: says claude failed" "$(notified 'claude exited 1')" "1"
check "agent commits then exits 1: its commit is scanned and pushed" \
  "$(git --git-dir="$WORK/agentfail-origin.git" log -1 --format=%s main)" "stub: sync"

new_pair agentfail2
echo change >"$WORK/agentfail2/new.txt"
STUB_RC=1 STUB_AFTER='echo more >>base.txt' run_sync agentfail2
STUB_RC=1 STUB_AFTER='echo more >>base.txt' run_sync agentfail2
check "agent keeps failing with a dirty tree: its commits still reach origin" \
  "$(git --git-dir="$WORK/agentfail2-origin.git" rev-list --count main)" "3"

new_pair nothingahead
echo scratch >"$WORK/nothingahead/scratch.txt"
git -C "$WORK/nothingahead" remote set-url origin "$WORK/does-not-exist.git"
STUB_AFTER='git reset -q --soft HEAD~1 && git reset -q' run_sync nothingahead
check "dirty, agent commits nothing, origin down: exit 0" "$?" "0"
check "dirty, agent commits nothing, origin down: no false 'commits are local'" "$(notified 'local only')" "0"

new_pair msgleak
git -C "$WORK/msgleak" commit -q --allow-empty -m "harmless"
GITLEAKS_STDIN_RC=1 run_sync msgleak
check "secret in a commit message: exit 1" "$?" "1"
check "secret in a commit message: nothing pushed" \
  "$(git --git-dir="$WORK/msgleak-origin.git" log -1 --format=%s main)" "base"
check "secret in a commit message: notified" "$(notified 'commit messages')" "1"

new_pair sortfail
echo change >"$WORK/sortfail/new.txt"
mkdir "$WORK/bin-sortfail"
printf '#!/usr/bin/env bash\nexit 1\n' >"$WORK/bin-sortfail/sort"
chmod +x "$WORK/bin-sortfail/sort"
RUN_PATH="$WORK/bin-sortfail:$WORK/bin:$PATH" run_sync sortfail
check "sort fails while listing candidates: exit non-zero" "$([ "$?" -ne 0 ] && echo yes)" "yes"
check "sort fails while listing candidates: agent never ran" "$(wc -c <"$WORK/claude-args.log" | tr -d ' ')" "0"

new_pair hugeceiling
local_commit hugeceiling "local only"
SYNC_MAX_FILE_BYTES=99999999999999999999 run_sync hugeceiling
check "ceiling beyond 64 bits: exit 1" "$?" "1"
check "ceiling beyond 64 bits: notified" "$(notified 'SYNC_MAX_FILE_BYTES')" "1"

new_pair upstream
git -C "$WORK/upstream" remote add upstream "$WORK/upstream-origin.git"
git -C "$WORK/upstream" fetch -q upstream
git -C "$WORK/upstream" branch -q --set-upstream-to=upstream/main
local_commit upstream "local only"
run_sync upstream
check "existing upstream on another remote: kept" \
  "$(git -C "$WORK/upstream" rev-parse --abbrev-ref 'main@{u}')" "upstream/main"

new_pair goneup
git -C "$WORK/goneup" config branch.main.remote upstream
git -C "$WORK/goneup" config branch.main.merge refs/heads/gone
local_commit goneup "local only"
run_sync goneup
check "configured upstream without a tracking ref: remote kept" \
  "$(git -C "$WORK/goneup" config branch.main.remote)" "upstream"
check "configured upstream without a tracking ref: merge kept" \
  "$(git -C "$WORK/goneup" config branch.main.merge)" "refs/heads/gone"

new_pair single
git clone -q --single-branch "$WORK/single-origin.git" "$WORK/single1" 2>/dev/null
git -C "$WORK/single1" config user.email t@example.com
git -C "$WORK/single1" config user.name t
git -C "$WORK/single1" config core.hooksPath /dev/null
git -C "$WORK/single1" checkout -q -b f
local_commit single1 "on f"
run_sync single1
check "single-branch clone, new branch: exit 0" "$?" "0"
run_sync single1
check "single-branch clone, second run: exit 0" "$?" "0"
check "single-branch clone, second run: no notification" "$(notified 'repo-autosync FAILED')" "0"
check "single-branch clone: commit reaches origin" \
  "$(git --git-dir="$WORK/single-origin.git" log -1 --format=%s f 2>&1)" "on f"

new_pair badceiling
head -c 6291456 /dev/zero >"$WORK/badceiling/big.bin"
git -C "$WORK/badceiling" add -A && git -C "$WORK/badceiling" commit -q -m big
SYNC_MAX_FILE_BYTES=5M run_sync badceiling
check "non-numeric SYNC_MAX_FILE_BYTES: exit 1" "$?" "1"
check "non-numeric SYNC_MAX_FILE_BYTES: nothing pushed" \
  "$(git --git-dir="$WORK/badceiling-origin.git" log -1 --format=%s main)" "base"
check "non-numeric SYNC_MAX_FILE_BYTES: notified" "$(notified 'SYNC_MAX_FILE_BYTES')" "1"

new_pair catfail
head -c 6291456 /dev/zero >"$WORK/catfail/big.bin"
git -C "$WORK/catfail" add -A && git -C "$WORK/catfail" commit -q -m big
mkdir "$WORK/bin-catfail"
REAL_GIT="$(command -v git)"
printf '#!/usr/bin/env bash\n[ "${1:-}" = cat-file ] && exit 1\nexec %q "$@"\n' "$REAL_GIT" >"$WORK/bin-catfail/git"
chmod +x "$WORK/bin-catfail/git"
RUN_PATH="$WORK/bin-catfail:$WORK/bin:$PATH" run_sync catfail
check "git cat-file fails: exit non-zero" "$([ "$?" -ne 0 ] && echo yes)" "yes"
check "git cat-file fails: nothing pushed" \
  "$(git --git-dir="$WORK/catfail-origin.git" log -1 --format=%s main)" "base"

# The ignore helper is generated on every run that reaches the agent.
new_pair helper
echo change >"$WORK/helper/new.txt"
run_sync helper
HELPER="$WORK/helper/.git/sync-agent-ignore"
(cd "$WORK/helper" && "$HELPER" --reason $'x\n!.env\n#' foo.tmp) >/dev/null 2>&1
check "helper: newline in --reason refused" "$?" "1"
check "helper: newline in --reason writes nothing" \
  "$(cat "$WORK/helper/.gitignore" 2>/dev/null | grep -c -x -F -e '!.env')" "0"
helper_says() { (cd "$WORK/helper" && "$HELPER" "$@") 2>&1; }
for p in '/*' '**/*' '* ' '.gitignore'; do
  out="$(helper_says "$p")"; rc=$?
  check "helper: '$p' refused as too broad" "$rc:$(grep -c 'too broad' <<<"$out")" "1:1"
done
mkdir -p "$WORK/helper/sub" && echo a >"$WORK/helper/sub/config.json"
git -C "$WORK/helper" add sub/config.json && git -C "$WORK/helper" commit -q -m sub
out="$(helper_says 'config.json')"; rc=$?
check "helper: basename rule matching a nested tracked file refused" "$rc:$(grep -c 'already tracked' <<<"$out")" "1:1"
helper_says '/build/' >/dev/null
check "helper: anchored directory rule accepted" "$?" "0"
helper_says --reason scratch 'scratch.tmp' >/dev/null
check "helper: ordinary pattern still accepted" "$?" "0"
out="$(helper_says '!keep.txt')"; rc=$?
check "helper: negation refused" "$rc:$(grep -c 'refusing negation' <<<"$out")" "1:1"
out="$(helper_says $'foo.tmp\n!.env')"; rc=$?
check "helper: newline in a pattern refused" "$rc:$(grep -c 'newline' <<<"$out")" "1:1"
check "helper: newline in a pattern writes nothing" \
  "$(cat "$WORK/helper/.gitignore" 2>/dev/null | grep -c -x -F -e '!.env')" "0"
# git failing is not an answer: read as "not ignored" it waves patterns through.
mkdir "$WORK/bin-ci128"
printf '#!/usr/bin/env bash\nfor a in "$@"; do [ "$a" = check-ignore ] && exit 128; done\nexec %q "$@"\n' \
  "$REAL_GIT" >"$WORK/bin-ci128/git"
chmod +x "$WORK/bin-ci128/git"
out="$(cd "$WORK/helper" && PATH="$WORK/bin-ci128:$PATH" "$HELPER" ci128.tmp 2>&1)"; rc=$?
check "helper: check-ignore exit 128 refuses the pattern" \
  "$rc:$(grep -c 'check-ignore failed' <<<"$out")" "1:1"
check "helper: check-ignore exit 128 writes nothing" \
  "$(cat "$WORK/helper/.gitignore" 2>/dev/null | grep -c -x -F -e 'ci128.tmp')" "0"
# Over 64 KB of tracked matches: the refusal must still explain itself.
blob="$(git -C "$WORK/helper" hash-object -w --stdin </dev/null)"
for i in $(seq 1 4000); do printf '100644 %s\tmany/f_%04d_padding_padding_padding\n' "$blob" "$i"; done |
  git -C "$WORK/helper" update-index --index-info
out="$(helper_says 'f_*')"; rc=$?
check "helper: refusal over many tracked matches still explains itself" "$rc:$(grep -c 'already tracked' <<<"$out")" "1:1"

new_pair deleted
git -C "$WORK/deleted" rm -q --cached base.txt && rm "$WORK/deleted/base.txt"
git -C "$WORK/deleted" commit -q -m "drop base" && git -C "$WORK/deleted" push -q origin main 2>/dev/null
echo two >"$WORK/deleted/two.txt"
git -C "$WORK/deleted" add two.txt && git -C "$WORK/deleted" commit -q -m two
git -C "$WORK/deleted" push -q origin main 2>/dev/null
rm "$WORK/deleted/two.txt"
run_sync deleted
check "deletion-only change: exit 0" "$?" "0"
check "deletion-only change: agent ran and the deletion reached origin" \
  "$(git --git-dir="$WORK/deleted-origin.git" ls-tree --name-only main two.txt)" ""

new_pair difffail
echo change >"$WORK/difffail/new.txt"
mkdir "$WORK/bin-difffail"
printf '#!/usr/bin/env bash\n[ "${1:-}" = diff ] && exit 1\nexec %q "$@"\n' "$REAL_GIT" >"$WORK/bin-difffail/git"
chmod +x "$WORK/bin-difffail/git"
RUN_PATH="$WORK/bin-difffail:$WORK/bin:$PATH" run_sync difffail
check "git diff fails while listing candidates: exit non-zero" "$([ "$?" -ne 0 ] && echo yes)" "yes"
check "git diff fails while listing candidates: notified once" "$(notified 'repo-autosync FAILED')" "1"

new_pair linked
git -C "$WORK/linked" worktree add -q -b wt "$WORK/linked-wt" 2>/dev/null
git -C "$WORK/linked-wt" config core.hooksPath /dev/null
echo change >"$WORK/linked-wt/new.txt"
run_sync linked-wt
check "linked worktree: exit 0" "$?" "0"
check "linked worktree: no 'not a git repository'" "$(notified 'not a git repository')" "0"
check "linked worktree: commit reaches origin" \
  "$(git --git-dir="$WORK/linked-origin.git" log -1 --format=%s wt 2>&1)" "stub: sync"

new_pair glerr
echo change >"$WORK/glerr/new.txt"
GITLEAKS_DIR_RC=1 run_sync glerr
check "pre-commit gitleaks exit 1: log line does not claim secrets were found" \
  "$(grep -c -F 'gitleaks found secrets' "$WORK/glerr.out")" "0"

new_pair leftover
echo change >"$WORK/leftover/new.txt"
STUB_AFTER='echo x >>base.txt' run_sync leftover
check "agent exits 0 leaving tracked changes: exit 1" "$?" "1"
check "agent exits 0 leaving tracked changes: notified" "$(notified 'still uncommitted')" "1"
check "agent exits 0 leaving tracked changes: what it did commit reaches origin" \
  "$(git --git-dir="$WORK/leftover-origin.git" log -1 --format=%s main)" "stub: sync"

# Nothing is outgoing object-wise (origin/side already has the commit), but
# origin/main still has to move, so "no objects" must not mean "no push".
new_pair ffside
git -C "$WORK/ffside" checkout -q -b side
local_commit ffside "on side"
git -C "$WORK/ffside" push -q origin side 2>/dev/null
git -C "$WORK/ffside" fetch -q origin 2>/dev/null
git -C "$WORK/ffside" checkout -q main
git -C "$WORK/ffside" merge -q --ff-only side
run_sync ffside
check "main ff'd to a commit origin has on another branch: exit 0" "$?" "0"
check "main ff'd to a commit origin has on another branch: origin/main advances" \
  "$(git --git-dir="$WORK/ffside-origin.git" log -1 --format=%s main)" "on side"

# Exits on the dirty path happen before the agent runs; commits from an
# earlier run must still get out, or a tree that stays dirty strands them.
new_pair stranded
local_commit stranded "earlier work"
head -c 6291456 /dev/zero >"$WORK/stranded/big.bin"
run_sync stranded
check "oversized dirty file with an older commit: exit 1" "$?" "1"
check "oversized dirty file with an older commit: notified once" "$(notified 'Oversized')" "1"
check "oversized dirty file with an older commit: the older commit reaches origin" \
  "$(git --git-dir="$WORK/stranded-origin.git" log -1 --format=%s main)" "earlier work"

new_pair stranded2
local_commit stranded2 "earlier work"
echo change >"$WORK/stranded2/new.txt"
RUN_PATH="$WORK/bin-noclaude" run_sync stranded2
check "claude missing, dirty, with an older commit: exit 1" "$?" "1"
check "claude missing, dirty, with an older commit: the older commit reaches origin" \
  "$(git --git-dir="$WORK/stranded2-origin.git" log -1 --format=%s main)" "earlier work"

# A whitelist repo ignores everything but one directory. The probes must be
# matched against the candidate pattern alone, not against those rules too.
new_pair whitelist
printf '/*\n!/src/\n' >"$WORK/whitelist/.gitignore"
git -C "$WORK/whitelist" add -f .gitignore && git -C "$WORK/whitelist" commit -q -m whitelist
echo change >>"$WORK/whitelist/base.txt"
run_sync whitelist
check "whitelist repo: exit 0" "$?" "0"
HELPER_W="$WORK/whitelist/.git/sync-agent-ignore"
whitelist_says() { (cd "$WORK/whitelist" && "$HELPER_W" "$@") 2>&1; }
whitelist_says --reason runtime 'logs/*.log' >/dev/null
check "whitelist repo: ordinary pattern still accepted" "$?" "0"
out="$(whitelist_says '**/*')"; rc=$?
check "whitelist repo: the probes still catch a too-broad pattern" \
  "$rc:$(grep -c 'too broad' <<<"$out")" "1:1"

if [ -n "${REAL_GITLEAKS:-}" ]; then
  new_pair binleak
  printf 'aws_access_key_id = %s%s\n' AKIAIOSFODNN7 EXAMPLF >"$WORK/binleak/blob.bin"
  printf '\000binary\000' >>"$WORK/binleak/blob.bin"
  git -C "$WORK/binleak" add -A
  git -C "$WORK/binleak" commit -q -m "binary-looking file"
  RUN_PATH="$WORK/bin:$PATH" GITLEAKS_REAL="$REAL_GITLEAKS" run_sync binleak
  check "secret in a NUL-containing file: exit 1" "$?" "1"
  check "secret in a NUL-containing file: nothing pushed" \
    "$(git --git-dir="$WORK/binleak-origin.git" log -1 --format=%s main)" "base"
else
  echo "SKIP binary-blob secret scan: real gitleaks not on PATH"
fi

[ "$fail" -eq 0 ] && rm -r "$WORK"
exit "$fail"
