# Aggregator Nix module

Home-manager module that installs the aggregator package, wires up systemd
user timers for the sessions + GitHub ingesters, and (optionally) registers
the MCP with Claude Code.

## Enable in your home-manager config

```nix
{ inputs, pkgs, ... }:
{
  imports = [ inputs.aggregator.homeManagerModules.default ];

  services.aggregator = {
    enable = true;
    package = inputs.aggregator.packages.${pkgs.system}.default;

    sources.sessions = {
      enable = true;
      interval = "*:0/30";          # every 30 min
      since = "";                   # empty = all-time; e.g. "2026-07-01T00:00:00Z"
    };

    sources.github = {
      enable = true;
      interval = "*:0/30";
      # Read-only PAT via agenix (see "Read-only PAT flow" below):
      githubTokenFile = "/run/agenix/github-readonly-pat";
    };

    mcp.autoRegister = false;       # true = run `claude mcp add` on activation
  };
}
```

Add the flake input in your top-level `flake.nix`:

```nix
inputs.aggregator.url = "path:/home/jonathan/Repos/aggregator";
# or: inputs.aggregator.url = "github:jonathan/aggregator";
```

## Per-source toggles

Each source has an explicit `enable` flag. Disable a source and *both* its
service and timer are omitted from the generated home-manager config — no
`systemctl --user mask` gymnastics needed:

```nix
services.aggregator.sources.github.enable = false;
```

## Read-only PAT flow (github source)

The github ingester refuses to run if the token in scope has any
write-capable scopes (`repo`, `gist`, `workflow`, `delete_repo`, …). The
supported production flow is:

1. Create a **read-only** PAT at github.com/settings/tokens/new. Scopes:
   `public_repo`, `repo:status`, `read:org`. Nothing else. 90-day
   expiration recommended.
2. Store it as an agenix-managed secret. In your NixOS config, add
   `age.secrets.github-readonly-pat.file = ./secrets/github-readonly-pat.age`
   (owned by your user, mode 0400). agenix decrypts at boot to
   `/run/agenix/github-readonly-pat`.
3. Point the module at that file:

   ```nix
   services.aggregator.sources.github.githubTokenFile =
     "/run/agenix/github-readonly-pat";
   ```

The systemd unit wraps its `ExecStart` in a small `bash -c` shim that reads
the file at unit start (`export GH_TOKEN="$(cat …)"`), so agenix rotation
is picked up automatically on the next timer tick — no home-manager
rebuild needed. If the file is missing the unit fails loudly (`set -e`)
rather than silently ingesting anonymously and hitting rate limits.

Leave `githubTokenFile` unset (default `null`) to fall back to whatever
`gh auth` has cached — fine for interactive machines, blocked by the
write-scope refusal on tokens with `repo`/`gist`/`workflow`.

## First-run cost

Sessions ingest walks every JSONL in `~/.claude/projects` on first run. On
a real dev workstation this measured **~3.1h wall time** on 2026-08-02:
5678 sessions + 1170 subagents + 348168 observations. Steady-state ticks
are much shorter (idempotent on stable_id, only re-parses changed files).

If you're recovering from a schema migration (v1 → v2 Schema B), don't
wait for the timer — run the one-shot reingest script under systemd-run
so it survives your terminal:

```bash
systemd-run --user --unit=aggregator-reingest --scope \
  uv run python /home/jonathan/Repos/aggregator/scripts/reingest_v2.py
```

The `--rebuild` flag on `aggregator ingest sessions` is guarded — it
refuses to wipe a nonempty store if the iterator yields zero rows (round-3
HIGH: transient-failure wipe pattern). Safe to schedule.

## Timer behaviour: missed windows

Both timers set `OnBootSec = "5min"` and `Persistent = true`:

- `Persistent = true` — if the last `OnCalendar` tick was missed
  (laptop closed), fire immediately on unit activation, then continue on
  the normal schedule.
- `OnBootSec = "5min"` — also fire once 5 minutes after boot,
  independent of the calendar schedule. Combined with `Persistent`, this
  means resuming a closed laptop reliably triggers a catch-up ingest
  within a few minutes.

These are the two knobs `systemd.time(7)` gives us for "run soon after
the machine wakes up". Verified against systemd v256 timer docs.

## Register the MCP with Claude Code

Default (`mcp.autoRegister = false`) prints the command in the activation
output. Run it once after activation:

```bash
claude mcp add aggregator $(which aggregator-mcp)
```

Set `mcp.autoRegister = true` to have home-manager run `claude mcp add`
for you at activation. The activation script:

- checks `claude mcp list` first (idempotent — skips if `aggregator` is
  already registered);
- never writes to `~/.claude.json` directly, always goes through the
  `claude` CLI so Claude Code's own validation runs;
- degrades gracefully if `claude` isn't on `PATH` (prints the command
  instead).

The legacy `mcpRegistration = "manual" | "activation-script"` option is
still accepted for backward compatibility (maps onto `mcp.autoRegister`).

## Verify

```bash
systemctl --user list-timers | grep aggregator
journalctl --user -u aggregator-sessions -n 50
journalctl --user -u aggregator-github -n 50
aggregator status
```

## Python deps that may need overlays (follow-up)

`fastmcp`, `presidio-analyzer`, and `presidio-anonymizer` are PyPI-only
and may not be packaged in nixpkgs. Consequences:

- `packages.default` from this flake is intentionally thin
  (`propagatedBuildInputs` is empty); a plain `nix build .#default` will
  succeed at build-graph time but the resulting binary will fail at import
  time until those deps are available.
- **Production path**: run the CLI/MCP inside a `uv run` shell that
  resolves the PyPI deps at runtime, wrapping via `nix run .#default --
  ...` once an overlay is added; or invoke `uv run aggregator ...` inside
  the devShell.
- **Overlay follow-up**: package the PyPI-only deps as an overlay under
  `nix/overlays/` and add them to `propagatedBuildInputs`. Track
  separately; this module ships the timer + option scaffolding, not the
  dependency closure.
