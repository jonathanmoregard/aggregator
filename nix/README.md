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

## Background embed worker (vector index)

`services.aggregator.embed` installs `aggregator-embed.service` +
`aggregator-embed.timer`, which fill the v5 sqlite-vec index that the hybrid
retriever's vector arm reads. It is never on the recall path: a query falls
through to FTS5 for anything not yet embedded, so a cold or half-full index
costs ranking quality, never availability.

```nix
services.aggregator.embed = {
  enable = true;            # default
  interval = "*:15/30";     # default — :15 and :45, offset from ingest
  batchSize = 500;          # rows per checkpoint
  timeoutStartSec = "8h";   # sized for a first full backfill
};
```

### Model weights: pre-seeded, never fetched by the timer

The embedder is Qwen3-Embedding-0.6B — about 1.2 GB of safetensors. Three
ways to get that to an unattended unit, and they fail differently:

| Approach | Upkeep | Manual work | Failure mode |
|---|---|---|---|
| Bake into a store path | new hash on every model bump; 1.2 GB in every closure | none | none at runtime, but every CI build pulls 1.2 GB |
| Fetch at runtime | none | none | an outage, a rate limit or a renamed repo turns a background indexer into a fetcher retrying a 1.2 GB download every 30 min |
| **Pre-seed a cache (chosen)** | **none** | **one command, once per machine** | **loud, specific, and names its own fix** |

So the timer-driven unit runs with `HF_HUB_OFFLINE=1` and opens no sockets.
`HF_HOME` is `%C/huggingface` — the systemd cache-directory specifier, which
for a user manager is `$XDG_CACHE_HOME`, i.e. the same `~/.cache/huggingface`
the interactive tooling already populates. No second copy, and no home path
baked into the unit.

If the weights are absent the worker refuses the run, non-zero, and prints
the one command that fixes it:

```bash
systemctl --user start aggregator-embed-seed.service
```

That unit is the only thing here allowed to touch the network. It has no
`[Install]` section and no timer — it runs when a human starts it, downloads
the weights, and then embeds exactly one row, so the seeding step also proves
weights + torch + sqlite-vec + a real database write in one go.

### Failure is loud, but only once a day

`OnFailure=aggregator-embed-failure-notify.service`, mirroring the ingest
unit: a journal line plus a CRITICAL `notify-send` popup. The popup — not the
journal line — is debounced to once per 24h, because the two standing failure
modes (weights absent, sqlite-vec absent) persist until a human acts, and 48
identical popups a day is how a loud system trains you to ignore it. Same
reasoning as `60a931d`. The debounce fails open: any error reading the stamp
results in a notification.

### Why `--catchup`, and why the timeout is hours

The unit runs `aggregator embed --catchup`, not `--once`. `--once` does one
batch per tick; at 500 rows against a ~376k-observation corpus that is ~380
ticks — over a week before the index is useful. `--catchup` drains the backlog
in the same bounded, per-batch-committed chunks and is a fast no-op once warm.

Overlapping runs are already handled by the worker itself: it takes an
OS-level `flock` on `<cache>.embed.lock`, and a tick that finds a catchup
still running prints one line and exits 0.

`TimeoutStartSec` defaults to `8h` — a runaway guard, not a deadline the
backfill has to beat. Being killed at the timeout is safe: each batch commits
its vectors and *then* advances `embedding_state`, so a kill at any instant
costs at most one batch and can never leave the watermark ahead of the data.
The vec writes are delete-then-insert, so a repeated batch is idempotent. Note
that the worker installs no SIGTERM handler — a stop is immediate, and the
safety comes from the commit ordering rather than from graceful shutdown.

### Contention with ingest

`interval` defaults to `*:15/30`, deliberately offset from the ingest timers'
`*:0/30`. That is an optimisation for the common case, not the correctness
story — an ingest run can last hours, so overlap is inevitable eventually.
What makes overlap safe is WAL journal mode plus a 30s `busy_timeout` on the
store, and one short write transaction per batch instead of one long one per
run. `checks.<system>.aggregator-embed-unit-hygiene` fails the build if the
embed timer's `OnCalendar` ever collides with an ingest timer's.

### Runtime dependencies

The embed path needs `sentence-transformers`, `torch` and `sqlite-vec` in the
package's Python closure, on top of the deps listed under "Python deps that
may need overlays" below. They are **not** in the deployed `aggregator-env`
today. Until they are, the unit fails loudly on every tick rather than
degrading quietly — see `tasks/pending_for_human.md`.

## Verifying the units without deploying

```bash
nix build .#checks.x86_64-linux.aggregator-embed-unit-hygiene
cat result/aggregator-embed.service
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
- **`SSL_CERT_FILE` and `NIX_SSL_CERT_FILE` are set.** A missing trust store
  makes every HTTPS host look like it is serving a self-signed certificate,
  which sends you on exactly the wrong investigation. It cost the TickTick
  source a day once.
- **`OnFailure=` is wired.**
- **`HF_HUB_OFFLINE=1`** on the timer-driven unit.
- **The embed timer's `OnCalendar` differs from every ingest timer's.**
- **`aggregator-embed-seed.service` has no `[Install]` section**, so it can
  never be pulled in by a target and start a 1.2 GB download unattended.

Each assertion has been red-tested by breaking the module on purpose and
confirming the check fails; `nix flake check` runs it.

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
journalctl --user -u aggregator-embed -n 50
aggregator status
```

## Python deps that may need overlays (follow-up)

`fastmcp`, `presidio-analyzer`, `presidio-anonymizer`, `sentence-transformers`
and `sqlite-vec` are PyPI-only and may not be packaged in nixpkgs.
Consequences:

- `packages.default` from this flake is intentionally thin
  (`propagatedBuildInputs` is empty). Corrected 2026-08-18: `nix build
  .#default` does **not** succeed — it fails at `pythonRuntimeDepsCheckHook`,
  which enumerates the gap for you. That is why
  `checks.<system>.aggregator-embed-unit-hygiene` renders its units against a
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
