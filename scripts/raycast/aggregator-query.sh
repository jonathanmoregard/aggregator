#!/usr/bin/env bash
# @raycast.title Aggregator Query
# @raycast.mode fullOutput
# @raycast.packageName Aggregator
# @raycast.icon 🗂
# @raycast.argument1 { "type": "text", "placeholder": "source:sessions from:2026-07-25" }
# @raycast.description Query personal aggregator; result is printed AND copied to clipboard.

set -euo pipefail

QUERY="${1:-}"
if [[ -z "$QUERY" ]]; then
  echo "usage: aggregator-query.sh 'DSL string'" >&2
  exit 2
fi

if ! command -v aggregator >/dev/null 2>&1; then
  echo "aggregator CLI not on PATH. Enable the home-manager module or run 'nix run .#'" >&2
  exit 3
fi

OUT=$(aggregator query "$QUERY" --fields full)
printf "%s" "$OUT" | pbcopy 2>/dev/null || printf "%s" "$OUT" | wl-copy 2>/dev/null || true
printf "%s\n" "$OUT"
