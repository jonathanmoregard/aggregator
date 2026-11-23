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
git add -A && git commit -q -m "stub: sync"
EOF
printf '#!/usr/bin/env bash\nexit 0\n' >"$WORK/bin/gitleaks"
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
  echo base >"$WORK/$1/base.txt"
  git -C "$WORK/$1" add -A
  git -C "$WORK/$1" commit -q -m base
  git -C "$WORK/$1" push -q -u origin main 2>/dev/null
}

run_sync() {
  PATH="$WORK/bin:$PATH" NOTIFY_LOG="$WORK/notify.log" DBUS_SESSION_BUS_ADDRESS=unix:path=/nonexistent \
    SYNC_REPO="$WORK/$1" bash "$SCRIPT" >"$WORK/$1.out" 2>&1
}

new_pair dirty
echo change >"$WORK/dirty/new.txt"
run_sync dirty
check "dirty tree: exit 0" "$?" "0"
check "dirty tree: commit reaches origin" \
  "$(git --git-dir="$WORK/dirty-origin.git" log -1 --format=%s main)" "stub: sync"

new_pair ahead
echo local >"$WORK/ahead/local.txt"
git -C "$WORK/ahead" add -A
git -C "$WORK/ahead" commit -q -m "local only"
run_sync ahead
check "clean but ahead: exit 0" "$?" "0"
check "clean but ahead: commit reaches origin" \
  "$(git --git-dir="$WORK/ahead-origin.git" log -1 --format=%s main)" "local only"

new_pair badpush
git -C "$WORK/badpush" remote set-url origin "$WORK/does-not-exist.git"
echo change >"$WORK/badpush/new.txt"
run_sync badpush
check "push failure: exit 1" "$?" "1"
check "push failure: notify_failure fired" \
  "$(grep -c 'git push origin main failed' "$WORK/notify.log" 2>/dev/null || true)" "1"

[ "$fail" -eq 0 ] && rm -r "$WORK"
exit "$fail"
