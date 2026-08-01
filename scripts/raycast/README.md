# Raycast wrapper for aggregator

Place `aggregator-query.sh` in a directory Raycast scans for scripts (default:
`~/.raycast/scripts`) or link it:

```bash
mkdir -p ~/.raycast/scripts
ln -sf $(pwd)/scripts/raycast/aggregator-query.sh ~/.raycast/scripts/
chmod +x scripts/raycast/aggregator-query.sh
```

Raycast picks it up on next reload. The DSL is passed as `argument1`; result is
printed in Raycast and copied to clipboard (`pbcopy` on macOS, `wl-copy` on
Wayland Linux).

The script requires `aggregator` on PATH — enable via the home-manager module
(`nix/README.md`) or `nix run .#`.
