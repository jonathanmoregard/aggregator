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

## GitHub credential flow

Authenticate once with `gh auth login`. The systemd user service consumes
that same GitHub CLI keyring credential; it does not copy the token into Nix,
an environment file, or an agenix secret.

The source constrains authority at its subprocess boundary. It accepts only
the four fixed `/search/issues` query shapes it generates and always invokes
`gh api` with explicit HTTP `GET`. `aggregator github-token-status` reports
the credential's scopes for diagnosis, but broad OAuth scopes do not require
a second, expiring PAT.

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

## Reranker weights

`rerank=True` (and `aggregator query --rerank`) loads the
`Qwen/Qwen3-Reranker-0.6B` cross-encoder inside the MCP server, pinned to
`rerank.QWEN3_RERANKER_REVISION`. Model loads are offline unless
`AGGREGATOR_ALLOW_MODEL_DOWNLOAD=1` is set, so the weights are fetched once, by
hand, with network access:

```bash
AGGREGATOR_ALLOW_MODEL_DOWNLOAD=1 aggregator seed-reranker
```

Without them `rerank=True` degrades to the recency order and says so in the
response's `notice`; keyword search is unaffected.

## Verifying the units without deploying

```bash
nix build .#checks.x86_64-linux.aggregator-unit-hygiene
cat result/aggregator-tag.service
```

The check instantiates the module through home-manager, renders the real unit
files, and asserts on them. Its output *is* the rendered units, so `result/`
is the artifact to read when you want to know what will be installed.

What it enforces, and why each one is there:

- **`ExecStart` is a `/nix/store` path, and so is every script it names.**
  The 2026-08-16 rule is that a unit executes a deployed artifact pinned to a
  committed revision, never a developer checkout. The bug that produced that
  rule was invisible at the unit-file level: `aggregator-ingest.service` had a
  store-path `ExecStart` whose wrapper script ended in
  `exec uv run --directory <checkout>`. So the check *follows* `ExecStart`
  into the script and greps that too, for `/home/` and for `uv run`.
  Deployment: new revision → new store path → `home-manager switch`. There is
  no in-place update path, by design.
- **No unit enables `AGGREGATOR_ALLOW_MODEL_DOWNLOAD`**, so nothing unattended
  can start a GB-scale download.
- **`OnFailure=` is wired, and the tag unit names its OWN notifier** — which
  cites its own journal.
- **The failure-notify debounce arms only on a delivered popup** — asserted by
  executing the generated script with a stubbed `notify-send`.
- **The tag unit carries its sandbox set**, and sets none of the directives
  that would stop `claude -p` from working.

`nix flake check` runs it.

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

`fastmcp`, `presidio-analyzer`, `presidio-anonymizer` and
`sentence-transformers` are PyPI-only and may not be packaged in nixpkgs.
Consequences:

- `packages.default` from this flake is intentionally thin
  (`propagatedBuildInputs` is empty). Corrected 2026-08-18: `nix build
  .#default` does **not** succeed — it fails at `pythonRuntimeDepsCheckHook`,
  which enumerates the gap for you. That is why
  `checks.<system>.aggregator-unit-hygiene` renders its units against a
  store-path stub rather than against `packages.default`: the check is about
  the shape of the generated units, and depending on the real package would
  make it fail for an unrelated reason.
- **Production path**: run the CLI/MCP inside a `uv run` shell that
  resolves the PyPI deps at runtime, wrapping via `nix run .#default --
  ...` once an overlay is added; or invoke `uv run aggregator ...` inside
  the devShell.
- **Overlay follow-up**: package the PyPI-only deps as an overlay under
  `nix/overlays/` and add them to `propagatedBuildInputs`. Track
  separately; this module ships the timer + option scaffolding, not the
  dependency closure.
