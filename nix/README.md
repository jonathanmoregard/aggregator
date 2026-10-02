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
  timeoutStartSec = "infinity";  # default — see "Why there is no start timeout"
};
```

### Model weights: pre-seeded, never fetched by the timer

Two models are deployed — the embedding GGUF (~640 MB) and the reranker
(~1.2 GB of safetensors):

| Model | Loaded by | Used for | Pinned |
|---|---|---|---|
| `Qwen/Qwen3-Embedding-0.6B-GGUF`, file `Qwen3-Embedding-0.6B-Q8_0.gguf` | `aggregator-embed-server.service` (llama.cpp, Vulkan) | every embedding — the worker's backfill and every query | sha, `embed.QWEN3_EMBEDDING_GGUF_REVISION`, **in the path the unit opens** |
| `Qwen/Qwen3-Reranker-0.6B` | the MCP server, lazily | `rerank=True` on a search | sha, `rerank.QWEN3_RERANKER_REVISION` |
| `Qwen/Qwen3-Embedding-0.6B` | nothing deployed — opt-in `AGGREGATOR_EMBED_BACKEND=st` only | the in-process CPU backend | sha, `embed.QWEN3_EMBEDDING_REVISION` |

### The embed server

The worker and the MCP server no longer compute embeddings. They talk HTTP,
over a **unix socket**, to `aggregator-embed-server.service`: an always-on
user unit running `llama-server --embedding --pooling last` on the pinned
Q8_0 file, bound to `$XDG_RUNTIME_DIR/aggregator-embed-server/embed.sock`,
fully offloaded to the GPU. Measured on tuxedo (Radeon 890M): 2495 tokens/s
against ~40 for the CPU backend it replaces, with vectors agreeing with the
fp32 ones at cosine 0.9994 mean / 0.9990 min — see
`docs/embedding-throughput.md`. The index is stamped
`Qwen/Qwen3-Embedding-0.6B-GGUF-q8_0@768/chunk-4000-400/norm-l2`; switching to
it starts a new backfill beside the old index rather than deleting it.

* **It is resident at ~3 GiB, not ~20.** The 2026-10-01 flags (`-ub 8192`,
  no flash attention, llama-server's default 8 GiB prompt cache) pinned
  RSS 11 GiB + 8.7 GiB of GPU-pinned RAM for a 640 MB model, on the belief
  that an embedding input must fit one micro-batch. It need not: the model
  is a causal decoder, the KV cache carries an input across micro-batches,
  and a 5997-token input embeds at `-ub 512`. The unit now runs
  `-b 1024 -ub 1024 -fa on --cache-ram 0` — measured 2026-10-02 at 3.0 GiB
  resident after a 64-chunk batch, the old throughput, vectors at cosine
  ≥ 0.9997 against the old flags (sweep in `docs/embedding-throughput.md`
  and `embedServerRunner`'s comment). The hygiene check executes the
  launcher and asserts those three flags.
* **A socket, not a TCP port, because of the worker's sandbox.** The worker
  reads the whole untrusted corpus and stays at
  `RestrictAddressFamilies=AF_UNIX AF_NETLINK` — no IP at all. TCP would have
  needed `AF_INET`, and the loopback fence for it
  (`IPAddressDeny=any` + `IPAddressAllow=localhost`) is cgroup BPF, which
  tuxedo's user manager does not get: measured under those directives, a TCP
  connect to the host's own LAN address succeeded. The server has no IP
  family either. The hygiene check asserts neither unit permits `AF_INET`.
* **The path is not an option.** The MCP server is registered bare and dials
  `unix://$XDG_RUNTIME_DIR/` + `embed.EMBED_SOCKET_NAME` (falling back to
  `/run/user/<uid>`); the unit binds the same path, and the hygiene check
  asserts the two agree. The directory is the unit's `RuntimeDirectory`, mode
  0700 (only this user reaches the encoder), removed by systemd on stop — so
  "stopped" is "no socket file" to every client; the launcher also clears a
  stale socket after an unclean exit. `AGGREGATOR_EMBED_URL`
  (`unix:///path.sock` or `http://host:port`) can point a process at another
  instance, but the backend checks what that instance serves (file name,
  width, quantization via `/v1/models`) and refuses anything but the pinned
  file, so a URL can never change what the stamp vouches for. Pooling is not
  visible on any endpoint, so `--pooling last` is pinned by the unit's command
  line and asserted by executing its launcher in the check.
* **Stopped server = FTS5, not an error.** Offline-AI mode stops the unit for
  hours. A query then answers from the keyword arm (the construction failure
  is not cached, so the vector arm returns with the server); the worker's
  next tick exits non-zero naming the unit, embeds nothing, blames no row and
  leaves the backlog where it was. A server stopped mid-run is caught by the
  worker's health probe the same way.
* **Sandbox, verified by execution.** The server runs under
  `NoNewPrivileges`, `PrivateTmp`, `ProtectSystem=full`,
  `ProtectHome=read-only` (which also makes `/run/user` read-only — hence the
  `RuntimeDirectory`, the one writable spot), `RestrictAddressFamilies=AF_UNIX
  AF_NETLINK` and the rest of the worker's set, and **not** `PrivateDevices`,
  which would hide `/dev/dri` and silently put the model on the CPU. Run under
  exactly that set with `systemd-run --user` on tuxedo, `/v1/models` and
  `/v1/embeddings` answered over the socket at 2270 tokens/s.

### Sandboxing the two units that run torch

**Since the embed server (2026-10-01) the worker no longer loads torch** on
the deployed backend — the encoding moved to `aggregator-embed-server`, whose
sandbox is described and verified in "The embed server" above — and it
now also uses `AF_UNIX` to reach that server's socket; it still has no IP
family.
The rest of this section still holds for the seeder, which loads the
reranker, and for the worker on the opt-in `st` backend.

Both `aggregator-embed.service` and `aggregator-embed-seed.service` load
third-party weights into torch and a native tokenizer — a large C++ attack
surface. The worker additionally feeds it the corpus (web pages, PDFs, chat
exports, GitHub bodies), none of which the user wrote. They share one Nix
binding, `embedSandboxCommon`, so they cannot drift apart on anything except
the axis they are genuinely meant to differ on: the network.

Measured with `systemd-analyze security --offline=true --user` against the
rendered files:

| Unit | Before | After |
|---|---|---|
| `aggregator-embed.service` | 9.4 UNSAFE | **6.3 MEDIUM** |
| `aggregator-embed-seed.service` | 9.2 UNSAFE | **6.8 MEDIUM** |

The seeder had *zero* sandbox directives, despite being the one unit that
reaches the public internet. Being human-triggered bounds how often that
happens, not what it can do when it does.

Exactly two directives are relaxed for the seeder, and the 0.5 point gap above
is precisely that relaxation:

- **`RestrictAddressFamilies` gains `AF_INET` + `AF_INET6`.** The directive is
  not dropped — keeping it still bars `AF_PACKET`, `AF_BLUETOOTH`, `AF_VSOCK`
  and the other exotic families that carry most of the socket-family kernel
  attack surface. A downloader needs TCP over IP and nothing else.
- **`IPAddressDeny` is omitted**, not set. `any` would block the download
  outright, and there is no useful allowlist to put in its place: huggingface.co
  resolves to a CDN whose address set changes without notice, and a stale
  allowlist becomes a mystery failure on the one unit a human runs by hand.

Reproduce either score with:

```bash
systemd-analyze security --offline=true --user \
  "$(nix build --no-link --print-out-paths \
     .#checks.x86_64-linux.aggregator-embed-unit-hygiene)/aggregator-embed-seed.service"
```

**Deliberately absent, and each absence is load-bearing:**

- **`MemoryDenyWriteExecute`** — torch's JIT and the OpenMP runtime allocate
  W|X pages; setting it makes `import torch` die. It is the most likely
  directive for a future hardening pass to reach for, so the check asserts it
  stays absent from both units.
- **`ProtectSystem=strict`** — would need an explicit `ReadWritePaths` for the
  HF cache, and getting that list wrong fails at runtime on units this host
  cannot start. `full` leaves `$HOME` (and so the cache) writable while `/usr`,
  `/boot` and `/etc` go read-only.
- **`PrivateDevices`, `ProtectKernelModules`, `SystemCallFilter`** — not
  applied, because nothing has ever executed this sandbox (see below), and
  widening it on a host that cannot run it would be guesswork.

The check asserts the first two absences directly, so a future hardening pass
that adds them fails the build with a reason instead of producing a unit that
cannot start.

### What remains unproven about the sandbox

**No process has ever run under these directives.** This is a known gap, not an
oversight, and it is stated here so nobody reads the table above as an
all-clear.

The deployed `aggregator-env` on this host has no `sentence-transformers`,
`torch-bin`, `transformers` or `sqlite-vec`, so neither unit can start here at
all; closing that lives in a different repo. "Start the unit and see" was
therefore never available while this was written.

What **is** verified:

- The rendered unit text — every directive that must be present, and the three
  that must be absent, are asserted at eval time by
  `checks.<system>.aggregator-embed-unit-hygiene`, and every one of those
  assertions was watched to fail against a deliberately-broken module before
  being trusted.
- The exposure scores in the table, measured with `systemd-analyze security
  --offline=true --user` against the rendered files on systemd 261.
- `systemd-analyze verify --user` is silent on both rendered units, i.e.
  systemd recognises every key and would not silently ignore one.
- The generated shell scripts, executed directly with stubbed binaries across
  every cache state that matters.

What is **not** verified, and what would settle it:

| Claim | Status | How to settle it |
|---|---|---|
| llama-server keeps the GPU under the embed server's directive set | **verified 2026-10-01** on tuxedo via `systemd-run --user` (2448 vs 2500 tokens/s unsandboxed) | — |
| `IPAddressDeny=any` + `IPAddressAllow=localhost` would confine an `AF_INET` unit to loopback | **measured FALSE on tuxedo** — no cgroup BPF in the user manager; a connect to the host's LAN address succeeded | settled by design: the embed units use a unix socket and permit no IP family |
| torch imports and runs under this directive set | **UNPROVEN** | run `aggregator-embed.service` on a host whose closure has torch, and read the journal |
| the seeder can actually reach huggingface.co through `RestrictAddressFamilies=AF_UNIX AF_NETLINK AF_INET AF_INET6` | **UNPROVEN** | `systemctl --user start aggregator-embed-seed.service` on such a host |
| `MemoryDenyWriteExecute` would break torch | **asserted from documented behaviour, not measured here** | as above, plus one run with the directive added |

Neither `systemd-analyze security` nor `systemd-analyze verify` can run inside
the Nix build sandbox — on systemd 261 both print

```
Failed to lookup RuntimeDirectory path: No such device or address
Failed to initialize manager: No such device or address
```

and then **exit 0**, so wiring either into the check would have produced a gate
that always passes while asserting nothing. Both are documented as manual steps
instead:

```bash
out=$(nix build --no-link --print-out-paths \
  .#checks.x86_64-linux.aggregator-embed-unit-hygiene)
systemd-analyze security --offline=true --user "$out/aggregator-embed-seed.service"
systemd-analyze verify --user "$out"/*.service
```

**First person to run either unit on a host with torch: watch the journal for
an early exit before any aggregator output.** That is what a sandbox
incompatibility looks like, and it is the one failure mode nothing above can
rule out.

### Failure is loud, but only once a day

Each unit that can fail has **its own** `OnFailure=` notifier, mirroring the
ingest unit: a journal line plus a CRITICAL `notify-send` popup.

| failing unit | notifier | debounce stamp |
| --- | --- | --- |
| `aggregator-embed.service` | `aggregator-embed-failure-notify.service` | `embed-failure-notified` |
| `aggregator-embed-seed.service` | `aggregator-embed-seed-failure-notify.service` | `embed-seed-failure-notified` |

One notifier used to serve both, written as though only the worker could fire
it. A **seed** failure therefore sent you to `journalctl -u
aggregator-embed.service` — a unit that had not run — and named `systemctl
--user start aggregator-embed-seed.service` as the remedy, i.e. the unit whose
failure you were being told about. They also shared one stamp, so a worker
failure silenced an unrelated seed failure for 24h. Both are generated from one
`mkFailureNotify`, so they cannot drift apart on anything but the text.

Not the templated `OnFailure=notify@%n.service` idiom. Probed here on systemd
261: `%i` expands correctly, but `%I` mangles `aggregator-embed.service` into
`aggregator/embed.service`, and the half that matters — `%n` inside
`OnFailure=` — could not be verified on this host at all, because
`systemd-analyze verify` ignores `OnFailure=` targets entirely (a control
naming a nonexistent unit draws no complaint) and these units cannot be started
here. `$SERVICE_RESULT` / `$EXIT_CODE` are not available to a separate
`OnFailure=` unit either. With two statically known units, generating two
notifiers needs none of that.

The popup — not the journal line — is debounced to once per 24h, because the
two standing failure modes (weights absent, sqlite-vec absent) persist until a
human acts, and 48 identical popups a day is how a loud system trains you to
ignore it. Same reasoning as `60a931d`.

The debounce fails **open on both halves**:

- *reading* the stamp — any error leaves it looking un-notified, so the run
  notifies rather than assuming it already did;
- *writing* the stamp — it is armed only after `notify-send` exits 0. A send
  that fails (no notification daemon on the session bus, the normal state of a
  headless or freshly-booted session) buys no silence at all; the next failing
  tick tries again. The debounce is a record that *the human was told*, and a
  failed send is precisely the case where they were not.

That cannot become a popup storm: while `notify-send` keeps failing there is no
daemon to show anything, so the cost is one extra journal line per tick.
`checks.<system>.aggregator-embed-unit-hygiene` pins both halves by *executing*
each generated script twice with a stubbed `notify-send` — once exiting 1 (tick
2 must still notify) and once exiting 0 (tick 2 must be suppressed) — and then
runs the two scripts against one shared `XDG_STATE_HOME`, in both orders, to
prove neither unit's failure silences the other.

### Why `--catchup`

The unit runs `aggregator embed --catchup`, not `--once`. `--once` does one
batch per tick; at 500 rows against the measured 483k-observation corpus that
is ~970 ticks — roughly three weeks before the last row is even attempted.
`--catchup` drains the backlog in the same bounded, per-batch-committed chunks
and is a fast no-op once warm.

Overlapping runs are already handled by the worker itself: it takes an
OS-level `flock` on `<cache>.embed.lock`, and a tick that finds a catchup
still running prints one line and exits 0.

### Why there is no start timeout

`timeoutStartSec` defaults to `infinity`. The reason is measurement, not
optimism: the real corpus is 483,193 observations / 422,261 chunks / 609M
chars, and the worker embeds at 249.6 chars per wall-second on CPU, so a first
full backfill is **25-30 days of continuous work** — not the ~3.5h the design
assumed.

Being SIGTERMed is safe. Each batch commits its vectors and *then* advances
`embedding_state`, so a kill at any instant costs at most one batch and can
never leave the watermark ahead of the data; the vec writes are
delete-then-insert, so a repeated batch is idempotent. The worker installs no
SIGTERM handler — a stop is immediate, and the safety comes from the commit
ordering, not from graceful shutdown.

Safe is not the same as free. A timeout puts the unit in `failed`, which fires
`OnFailure=`. At the old `8h` a *correctly progressing* backfill needed ~85
consecutive runs, so it would have raised ~28 CRITICAL popups over a month
(debounced to one a day), each announcing that the vector index is not being
filled and naming two causes that did not apply. An alarm that fires on
success is how you learn to ignore the alarm — the same failure the debounce
above exists to prevent.

**What guards a genuinely wedged worker, then?** Not a clock: no wall-clock
value can separate "wedged" from "working" when working legitimately takes a
month. Instead —

- `Nice=19` and `IOSchedulingClass=idle` bound a spinning worker to
  otherwise-idle capacity.
- Batch-sized write transactions plus the per-batch checkpoint bound what a
  wedge can lose to one batch; it can never park a long transaction on the
  cache.
- The `flock` means later ticks exit 0 as no-ops instead of stacking workers.
- `TimeoutStopSec=5min` stays finite, so
  `systemctl --user stop aggregator-embed.service` is always a bounded kill.
- Detection is by **progress**: `aggregator status` and
  `aggregator_capabilities()['vector_index']` report embedded / pending /
  error counts, and a wedged worker is one whose counts stop moving.

`checks.<system>.aggregator-embed-unit-hygiene` asserts both halves — that the
start timeout is disabled, and that the stop timeout is not.

`timeoutStartSec` is typed as a **systemd time span**, not a bare string. It
used to be `types.str`, so `"infinty"` evaluated, built and deployed happily;
systemd then rejected the span at unit start and applied its ~90 second
default, truncating every tick of a 25-30 day backfill. The parse error goes to
the journal, but nothing reads that journal until someone already suspects a
problem, and the only other symptom is a progress counter that stops moving.
Now the typo fails where it was typed:

```
error: A definition for option `services.aggregator.embed.timeoutStartSec' is
not of type `systemd time span per systemd.time(7) — e.g. "8h", "90min",
"1h 30min", or bare seconds "3600" — or the literal "infinity" to disable the
timeout'. Definition values:
- In `<unknown-file>': "infinty"
```

The grammar is from `systemd.time(7)`, and the check evaluates the module
against 8 spans `systemd-analyze timespan` accepts and 6 it rejects, failing
the build in **either** direction — a type that is too loose lets the typo
through, and one that is too tight blocks a legitimate config.

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
- **`OnFailure=` is wired, and each unit names its OWN notifier** — which
  cites its own journal and no other unit's.
- **`HF_HUB_OFFLINE=1`** on the timer-driven unit.
- **The embed timer's `OnCalendar` differs from every ingest timer's.**
- **`aggregator-embed-seed.service` has no `[Install]` section**, so it can
  never be pulled in by a target and start a 2.4 GB download unattended.
- **The seed unit names both model repos — as read from the Python defaults
  that decide the fetch — runs `embed --seed-models`, exports the download
  opt-in, and embeds no rows.**
- **`timeoutStartSec` accepts exactly the time spans `systemd-analyze timespan`
  accepts** — checked in both directions against 14 cases.
- **The failure-notify debounce arms only on a delivered popup, and is
  per-unit** — asserted by executing both generated scripts with a stubbed
  `notify-send`, including against one shared state dir in both orders.
- **Both torch units carry the shared sandbox set**, the seeder can still open
  an IP socket, and neither unit sets `MemoryDenyWriteExecute` or
  `ProtectHome`.

Each assertion has been red-tested by breaking the module on purpose and
confirming the check fails; `nix flake check` runs it. An assertion that could
not be made to go red was removed rather than shipped — see the
`ProtectSystem=strict` note in `flake.nix`.

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
