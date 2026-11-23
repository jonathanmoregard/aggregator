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
exit 0
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
check "agent argv: allowed tools recorded" \
  "$(grep -c -F -e 'Bash(git commit:*)' "$WORK/claude-args.log")" "1"
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
for t in bash git dirname basename id head numfmt stat sort tr; do ln -s "$(command -v "$t")" "$WORK/bin-noclaude/$t"; done
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
check "agent commits then exits 1: says commits are local" "$(notified 'claude exited 1')" "1"

new_pair upstream
git -C "$WORK/upstream" remote add upstream "$WORK/upstream-origin.git"
git -C "$WORK/upstream" fetch -q upstream
git -C "$WORK/upstream" branch -q --set-upstream-to=upstream/main
local_commit upstream "local only"
run_sync upstream
check "existing upstream on another remote: kept" \
  "$(git -C "$WORK/upstream" rev-parse --abbrev-ref 'main@{u}')" "upstream/main"

[ "$fail" -eq 0 ] && rm -r "$WORK"
exit "$fail"
