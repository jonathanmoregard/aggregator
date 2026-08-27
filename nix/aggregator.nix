{ config, lib, pkgs, ... }:
let
  cfg = config.services.aggregator;
  aggregatorBin = "${cfg.package}/bin/aggregator";
  aggregatorMcpBin = "${cfg.package}/bin/aggregator-mcp";

  # Build the ExecStart command for a source, threading in optional --since.
  # Written as `sh -c '...'` so ``GH_TOKEN=$(cat …)`` (see githubTokenFile
  # branch below) is expanded by the shell at unit-start, not by systemd's
  # own env parser (which does no command substitution).
  mkExecStart = { source, since, tokenFile }:
    let
      base = "${aggregatorBin} ingest ${source}"
        + lib.optionalString (since != "") " --since ${lib.escapeShellArg since}";
    in
      if tokenFile == null then
        # No token wiring — plain exec, no shell wrapper needed.
        base
      else
        # Read the token from disk at unit start so agenix rotation is
        # picked up without a rebuild. Fail loudly (`set -e`) if the file
        # is missing rather than silently ingesting anonymously and
        # hitting rate limits.
        #
        # Round-1 MEDIUM: keep the assignment and the export on separate
        # lines. `export FOO=$(cmd)` masks `cmd`'s exit — the outer
        # `export` builtin returns 0 regardless — so `set -e` never
        # trips on a missing token file. Splitting into `token=$(cat ...)`
        # then `export GH_TOKEN="$token"` lets `set -e` see the cat
        # failure and abort the unit before we hand off to the CLI.
        "${pkgs.bash}/bin/bash -c '"
          + "set -e; "
          + "token=\"$(${pkgs.coreutils}/bin/cat "
          + lib.escapeShellArg tokenFile + ")\"; "
          + "export GH_TOKEN=\"$token\"; "
          + "exec ${base}"
          + "'";

  # ---- embed worker plumbing --------------------------------------------
  #
  # DEPLOYMENT CONSTRAINT (2026-08-16, non-negotiable): every executable this
  # unit touches is a /nix/store path pinned to the revision home-manager was
  # built from. Nothing here may reach into a developer checkout. The bug that
  # produced this rule was invisible in the unit file — `ExecStart=` pointed at
  # a store path whose *wrapper script* ended in
  # `exec uv run --directory <checkout>`, so the timer ran whatever branch
  # happened to be checked out. That is why the ExecStart here is a
  # `writeShellScript` whose own text is asserted clean by
  # `checks.<system>.aggregator-embed-unit-hygiene` in flake.nix: the check
  # follows ExecStart into the script and greps the script too, not just the
  # unit. New version → new rev → new store path → `home-manager switch`.
  # There is no in-place update path, by design.

  # NixOS' system-wide CA bundle. Same path the deployed ingest unit uses.
  # A missing trust store is not a subtle failure — it makes every HTTPS host
  # on the internet look like it is serving a self-signed certificate, which
  # sends a human on exactly the wrong investigation. TickTick lost a day to
  # this once.
  caBundle = "/etc/ssl/certs/ca-bundle.crt";

  # Hugging Face cache root. `%C` is the systemd cache-directory specifier —
  # for a *user* manager it expands to $XDG_CACHE_HOME (i.e. ~/.cache), so the
  # unit text carries no home path while still resolving to the cache the
  # interactive tooling already populates. Specifier expansion in
  # `Environment=` is documented in systemd.unit(5) "Specifiers" and verified
  # against systemd v261 on this host.
  #
  # DELIBERATE deviation from plan step K2, which proposed
  # `HF_HOME=%C/aggregator/huggingface`. A private cache dir would force a
  # fresh 1.2 GB download even on a machine that already holds the weights,
  # and would keep a second copy forever. Pointing at the shared default is
  # strictly less manual work and less disk, and the seeding path below makes
  # the "cache is empty" case loud rather than silent.
  hfHome = "%C/huggingface";

  # The sentence-transformers repo id the Embedder loads by default
  # (aggregator/core/embed.py::_DEFAULT_MODEL_ST) and the on-disk directory
  # name the huggingface_hub cache gives it.
  embedModelRepo = "Qwen/Qwen3-Embedding-0.6B";
  embedModelDir = "models--Qwen--Qwen3-Embedding-0.6B";

  # The cross-encoder the MCP server loads on `rerank=True`
  # (aggregator/core/rerank.py::_DEFAULT_MODEL).
  #
  # Round-2 MEDIUM: nothing, anywhere, used to fetch these weights. The seed
  # unit ran `embed --once`, which constructs only the Embedder;
  # `Reranker()` is built in exactly one place (aggregator/mcp.py), and it
  # passes `local_files_only=not downloads_allowed()` while the MCP server is
  # registered bare — no `AGGREGATOR_ALLOW_MODEL_DOWNLOAD`, by design, since
  # a query must never start a GB-scale download inside the editor's process.
  # So the cache was never populated by any path, every `rerank=True` raised
  # inside the constructor, and `_maybe_rerank` caught it and returned the
  # page in its original order. `rerank=True` degraded to unranked FOREVER,
  # and the response carried no notice saying so.
  #
  # Naming the repo here is what gives the weights a fetch path at all.
  rerankModelRepo = "Qwen/Qwen3-Reranker-0.6B";
  rerankModelDir = "models--Qwen--Qwen3-Reranker-0.6B";

  # HOW MANY CORES BACKGROUND EMBEDDING MAY TAKE.
  #
  # ONE binding, used for BOTH the thread pools and the cgroup quota below, so
  # the two cannot drift. That pairing is the whole point: a pool sized larger
  # than the quota is the worst of the three configurations, because all of
  # those threads still get scheduled and are then throttled together — full
  # context-switching and cache-thrash for a fraction of the throughput. Sized
  # together they are just fewer, faster threads.
  #
  # FOUR, on a 12-core machine, because this is an interactive laptop and the
  # operator can hear it. Measured from the unit's own accounting on
  # 2026-08-27: `1d 7h 51min` of CPU over `4h 3s` of wall clock — ~8x
  # parallelism, sustained, which is a fan that never spins down. The
  # instruction already in place was `Nice=19`, and nice is the wrong
  # instrument for this: it orders who runs FIRST, not how many run AT ONCE,
  # so twelve threads at nice 19 still saturate twelve cores and still make
  # the same heat. It stays, because it is the right instrument for the
  # different question of who yields when the operator starts typing.
  #
  # The backfill is measured in weeks either way; the operator's directive on
  # 2026-08-27 was explicit that wall clock is the currency to spend here
  # ("limit it a bit more, even if it takes a bit longer"). Not exposed as a
  # module option: this is the behaviour of a background indexer on a personal
  # machine, not a dial anyone should have to find.
  embedThreads = 4;

  # Environment shared by the embed worker and its one-shot seeding sibling.
  # `AGGREGATOR_EMBED_BACKEND=st` pins the safetensors loader; the `gguf`
  # backend needs an optional extra that is not in the deployed closure, and a
  # backend that silently changes under a unit is a debugging trap.
  #
  # THE THREE THREAD VARIABLES ARE THREE DIFFERENT POOLS, not belt-and-braces
  # spellings of one. torch's kernels run on OpenMP (`OMP_NUM_THREADS`), its
  # BLAS calls can go to MKL (`MKL_NUM_THREADS`), and HuggingFace's fast
  # tokenizers run a rayon pool of their own (`RAYON_NUM_THREADS`) — which is
  # why `systemd-cgls` showed 46 tasks under a unit doing one encode at a
  # time. Capping one leaves the others at the core count.
  #
  # They are READ AT IMPORT, so setting them in the unit is what makes them
  # effective; `aggregator.core.embed._pin_thread_pools` then resizes torch's
  # separate intra-op pool, which several builds size from the core count
  # regardless of `OMP_NUM_THREADS`.
  embedBaseEnvironment = [
    "SSL_CERT_FILE=${caBundle}"
    "NIX_SSL_CERT_FILE=${caBundle}"
    "HF_HOME=${hfHome}"
    "AGGREGATOR_EMBED_BACKEND=st"
    "OMP_NUM_THREADS=${toString embedThreads}"
    "MKL_NUM_THREADS=${toString embedThreads}"
    "RAYON_NUM_THREADS=${toString embedThreads}"
  ];

  # Shell fragment resolving the HF cache root from whatever HF_HOME the unit
  # actually got, falling back to huggingface_hub's own default so a hand-run
  # of the script outside systemd reports the same diagnosis, plus a
  # `have_model <cache-dir-name>` predicate over it. Two units and two models
  # ask this question now, so it is one helper rather than four copies of a
  # glob. POSIX-only: no coreutils on PATH is assumed.
  modelPresenceCheck = ''
    hf_home="''${HF_HOME:-''${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
    # 0 when the named repo has at least one materialised snapshot in the
    # cache, 1 otherwise. A bare `snapshots/` with nothing under it is what an
    # interrupted download leaves behind, so the directory existing is not
    # enough — hence the glob.
    have_model() {
      _snapshots="$hf_home/hub/$1/snapshots"
      [ -d "$_snapshots" ] || return 1
      for _entry in "$_snapshots"/*; do
        [ -e "$_entry" ] && return 0
      done
      return 1
    }
  '';

  # ExecStart for the timer-driven worker.
  #
  # `--catchup`, not the plan's `--once`. `--once` does a single batch per
  # tick, and a batch is bounded at `cli._MAX_BATCH_CHUNKS` chunks — about
  # fifteen minutes of encoder time — so against a backfill measured in weeks
  # a 30-minute tick would run at half speed at best, which is a backfill that
  # never finishes for any practical purpose. `--catchup` drains the backlog in
  # the same bounded, per-batch-committed chunks and is a fast no-op once the
  # index is warm.
  # Overlap is already handled by the worker's own flock on
  # `<cache>.embed.lock`: a tick that finds a catchup still running prints one
  # line and exits 0.
  #
  # A catchup does NOT finish inside one tick, and is not meant to: Task M
  # measured the first full backfill at 25-30 days of continuous CPU. It runs
  # to completion across ticks because `timeoutStartSec` no longer cuts it
  # off — see that option's description for why a finite one had to go.
  embedRunner = pkgs.writeShellScript "aggregator-embed" ''
    set -uo pipefail

    # Not fatal here, unlike the ingest wrapper: this unit runs with
    # HF_HUB_OFFLINE=1 and opens no sockets, so an unusable bundle cannot
    # affect the run. Say so anyway — the variable exists so that a
    # deliberate seeding run inherits a trust store, and a human reading the
    # journal should not have to infer that.
    ca_bundle="''${SSL_CERT_FILE:-}"
    if [ -z "$ca_bundle" ] || [ ! -s "$ca_bundle" ]; then
      echo "aggregator-embed: warning: SSL_CERT_FILE ('$ca_bundle') is unset or empty. Harmless for this offline unit, but aggregator-embed-seed.service will fail against an empty trust store." >&2
    fi

    ${modelPresenceCheck}
    # Only the EMBEDDING weights gate this unit. It never reranks — the
    # cross-encoder is loaded lazily by the MCP server — so a missing
    # reranker must not stop the index from filling.
    if ! have_model "${embedModelDir}"; then
      snapshots="$hf_home/hub/${embedModelDir}/snapshots"
      echo "aggregator-embed: ${embedModelRepo} weights are not in the Hugging Face cache (looked in $snapshots)." >&2
      echo "aggregator-embed: this unit runs OFFLINE by design (HF_HUB_OFFLINE=1) and will NOT pull ~1.2 GB unattended." >&2
      echo "aggregator-embed: seed the cache once, then this timer takes over:" >&2
      echo "aggregator-embed:     systemctl --user start aggregator-embed-seed.service" >&2
      echo "aggregator-embed: refusing to run rather than no-op silently; the embedding backlog is untouched." >&2
      exit 1
    fi

    # `exec` so the CLI's exit status IS the unit's, with no wrapper in
    # between. `aggregator embed` exits non-zero rather than advancing
    # embedding_state when sqlite-vec is unavailable; that status has to
    # reach systemd unaltered for OnFailure to fire.
    exec ${aggregatorBin} embed --catchup --source both --batch-size ${toString cfg.embed.batchSize}
  '';

  # One-shot, human-triggered, never on a timer. The only place in this module
  # that is allowed to touch the network for model weights.
  #
  # `embed --seed-models`, NOT the previous `embed --once --source
  # observations --batch-size 1`. Round-2 MEDIUM, three problems with that
  # command and one of them was load-bearing:
  #
  #   1. It constructed only the `Embedder`. The `Reranker` weights were
  #      fetched by nothing, anywhere, so `rerank=True` degraded to unranked
  #      forever and said nothing about it. Seeding has to cover every model
  #      the product actually loads, or "seeded" is a claim about one of them.
  #   2. It embedded a REAL, UNTRUSTED CORPUS ROW purely to warm a cache —
  #      running attacker-influenced text through torch as a side effect of a
  #      download.
  #   3. It touched the database at all. A weight-seeding step that opens the
  #      cache can contend with an ingest run and can advance
  #      `embedding_state`, which is state that a download has no business
  #      moving.
  #
  # The contract `--seed-models` is written against: construct BOTH the
  # Embedder and the Reranker, touch no database rows, permit downloads only
  # under the `AGGREGATOR_ALLOW_MODEL_DOWNLOAD` opt-in exported below, and
  # exit non-zero naming the remedy when weights are absent and downloads are
  # disallowed. Constructing both models is still a live proof that the
  # weights are complete and loadable by torch — it just no longer proves it
  # by writing to the corpus.
  embedSeeder = pkgs.writeShellScript "aggregator-embed-seed" ''
    set -uo pipefail

    ca_bundle="''${SSL_CERT_FILE:-}"
    if [ -z "$ca_bundle" ] || [ ! -s "$ca_bundle" ]; then
      echo "aggregator-embed-seed: no usable CA bundle (SSL_CERT_FILE='$ca_bundle') — the download would fail CERTIFICATE_VERIFY_FAILED against an empty trust store" >&2
      exit 1
    fi

    ${modelPresenceCheck}
    for spec in "${embedModelRepo}|${embedModelDir}|the embed worker's vector index" \
                "${rerankModelRepo}|${rerankModelDir}|the MCP server's rerank=True path"; do
      repo="''${spec%%|*}"
      rest="''${spec#*|}"
      dir="''${rest%%|*}"
      what="''${rest#*|}"
      if have_model "$dir"; then
        echo "aggregator-embed-seed: $repo already present under $hf_home/hub/$dir/snapshots — nothing to download."
      else
        echo "aggregator-embed-seed: downloading $repo (~1.2 GB) into $hf_home — feeds $what. One-time cost."
      fi
    done

    # THE ONLY OPT-IN IN THIS MODULE. The Python loaders pass
    # `local_files_only=True` unless this is set, so every other caller — the
    # timer, the MCP server, an ad-hoc CLI run — refuses to fetch weights
    # rather than pulling GBs from wherever it happens to be running.
    # HF_HUB_OFFLINE cannot express that on its own: huggingface_hub reads it
    # into a constant at import time, and the MCP server has already imported
    # it (via the scrubber's spaCy probe) before any aggregator code could set
    # it. This unit is human-triggered and never timer-driven, which is
    # exactly the property that makes the download consented to.
    export AGGREGATOR_ALLOW_MODEL_DOWNLOAD=1
    exec ${aggregatorBin} embed --seed-models
  '';

  # OnFailure target, generated PER FAILING UNIT. Mirrors the deployed
  # aggregator-ingest-failure-notify unit (journal line + CRITICAL libnotify
  # popup), with one addition: the popup is debounced to once per day.
  #
  # ONE NOTIFIER PER UNIT, and that is round 3's M3. There used to be a single
  # script wired as `OnFailure=` for BOTH `aggregator-embed.service` and
  # `aggregator-embed-seed.service`, and it was written as though only the
  # first could ever fire it. Two concrete defects fell out, both reproduced
  # by running the rendered script twice:
  #
  #   1. WRONG JOURNAL AND CIRCULAR ADVICE. When the SEED unit failed, the
  #      operator was handed `journalctl --user -u aggregator-embed.service`
  #      — which contains nothing about the download that just died — and was
  #      told the fix was to run `systemctl --user start
  #      aggregator-embed-seed.service`, i.e. the very unit whose failure they
  #      were being notified about.
  #   2. ONE STAMP SILENCED THE OTHER UNIT. Both shared
  #      `embed-failure-notified`, so a worker failure armed the 24h debounce
  #      and a seed failure minutes later was suppressed outright. Observed:
  #      tick 1 (worker) notified, tick 2 (seed) printed "suppressed".
  #
  # NOT the templated `OnFailure=notify@%n.service` idiom, deliberately.
  # Verified on this host with systemd 261: `%i` on an instance named
  # `aggregator-embed.service` does expand to `aggregator-embed.service`, and
  # `%I` mangles it to `aggregator/embed.service` (the `-`→`/` unescape), so
  # the idiom works but has a live footgun in it. The half that MATTERS —
  # that `%n` inside `OnFailure=` expands to the failing unit's own name —
  # could not be verified here at all: `systemd-analyze verify` does not
  # inspect `OnFailure=` targets, confirmed with a control naming a unit that
  # does not exist and drawing no complaint, and the embed units cannot be
  # started on this host. With exactly two statically-known units, a
  # parameterised generator needs no specifier semantics, no instance
  # escaping, and nothing that has to be taken on trust.
  #
  # The embed timer fires every 30 minutes. Its two standing failure modes —
  # weights absent, sqlite-vec absent — are both *persistent* until a human
  # acts, so an undebounced popup would fire 48 times a day and be muted,
  # which is how a loud system becomes a silent one. Same reasoning as
  # `60a931d` (report permanently-bad input once, not twice an hour). The
  # journal line is NOT debounced; only the desktop popup is.
  #
  # The debounce fails OPEN on BOTH halves, which is the whole point:
  #
  #   read  — any error stat-ing the stamp leaves `recent` empty, so we
  #           notify rather than assume we already did.
  #   write — the stamp is armed ONLY after notify-send exits 0. Round-2 LOW:
  #           it used to be touched *before* the send, so a popup that failed
  #           to reach any daemon still bought 24 hours of silence and the
  #           user was never told the vector index had stopped filling. A
  #           debounce is a record of "the human was told", and a failed send
  #           is precisely the case where they were not. Reproduced by running
  #           this script twice with a notify-send stub exiting 1: run 1
  #           printed "notify-send failed", run 2 printed "suppressed".
  #
  # Failing open here cannot become a popup storm: if notify-send keeps
  # failing there is no daemon to show anything, so the cost is one extra
  # journal line per tick, and the moment a daemon does appear the user gets
  # told once and the debounce arms normally.
  #
  # `$SERVICE_RESULT` / `$EXIT_CODE` are NOT available to a separate
  # `OnFailure=` unit (round 2 established this), which is why the failing
  # unit's identity has to be baked in at generation time rather than read
  # from the environment at runtime.
  mkFailureNotify = { name, unit, stamp, summary, body }:
    pkgs.writeShellScript name ''
      set -uo pipefail

      echo "${unit} FAILED — inspect: journalctl --user -u ${unit} -n 200"

      stamp_dir="''${XDG_STATE_HOME:-$HOME/.local/state}/aggregator"
      # Per-unit stamp. A shared one makes either unit's failure buy silence
      # for the other, which is the same "loud system becomes silent" bug the
      # debounce exists to avoid, arrived at from the other direction.
      stamp="$stamp_dir/${stamp}"
      recent="$(${pkgs.findutils}/bin/find "$stamp" -mmin -1440 2>/dev/null)"
      if [ -n "$recent" ]; then
        echo "desktop notification suppressed — already notified within 24h (stamp: $stamp). Failure is in the journal above."
        exit 0
      fi

      if ${pkgs.libnotify}/bin/notify-send -u critical -a aggregator \
        "${summary}" \
        "${body} Details: journalctl --user -u ${unit} -n 200"; then
        # Delivered. Arm the 24h debounce, and only now.
        ${pkgs.coreutils}/bin/mkdir -p "$stamp_dir" 2>/dev/null
        ${pkgs.coreutils}/bin/touch "$stamp" 2>/dev/null
      else
        echo "notify-send failed (no notification daemon on session bus?) — failure recorded in journal only. NOT arming the 24h debounce: an undelivered popup must not buy silence, so the next failing tick will try again."
      fi
    '';

  embedFailureNotify = mkFailureNotify {
    name = "aggregator-embed-failure-notify";
    unit = "aggregator-embed.service";
    stamp = "embed-failure-notified";
    summary = "aggregator embed FAILED";
    body =
      "The background embed worker exited non-zero, so the vector index is"
      + " not being filled. Likely: Qwen3 weights missing from the HF cache"
      + " (fix: systemctl --user start aggregator-embed-seed.service), or the"
      + " sqlite-vec extension did not load. Keyword search is unaffected.";
  };

  # The seeder's own notification. Pointing this one at the worker's journal
  # told the operator to read a unit that had not run, and naming the seed
  # unit as the remedy told them to run the thing that had just failed.
  #
  # Re-running IS the right move here — but only after the cause is fixed,
  # and the cause is in this unit's own journal. The causes named are the
  # ones this unit can actually hit: it is the only unit permitted to reach
  # the network, it has a 4h start timeout, and it writes ~2.4 GB to disk.
  embedSeedFailureNotify = mkFailureNotify {
    name = "aggregator-embed-seed-failure-notify";
    unit = "aggregator-embed-seed.service";
    stamp = "embed-seed-failure-notified";
    summary = "aggregator model download FAILED";
    body =
      "The one-time Qwen3 weight download exited non-zero, so the embedding"
      + " and reranker weights are NOT in the cache: the embed worker will"
      + " refuse on every tick and rerank=True stays degraded. Likely: no"
      + " network, an empty CA bundle, not enough disk for ~2.4 GB, a hub"
      + " rate limit, or the 4h start timeout. Keyword search is unaffected."
      + " Fix the cause below, then re-run: systemctl --user start"
      + " aggregator-embed-seed.service.";
  };

  # ---- systemd time-span option type ------------------------------------
  #
  # Round-2 LOW. `timeoutStartSec` was `lib.types.str`, so `"infinty"`
  # evaluated clean, built, deployed, and was only then rejected — by
  # systemd, at unit start, which falls back to its ~90 second default.
  # Against a backfill measured at 25-30 days that truncates every single
  # tick. systemd does log a parse error, so it is not literally silent, but
  # nobody reads that journal until they already suspect a problem, and the
  # only other symptom is a progress counter that stops moving. Catch it
  # where the typo is typed instead.
  #
  # Grammar from systemd.time(7) "Parsing Time Spans": one or more
  # `<number><unit>` terms, optional whitespace between them, a bare number
  # meaning seconds, or the literal `infinity`. The accept/reject split this
  # produces was checked term-by-term against `systemd-analyze timespan` on
  # systemd 261, and `checks.<system>.aggregator-embed-unit-hygiene` pins it.
  systemdTimeSpanUnits = lib.concatStringsSep "|" [
    "usec" "usecs" "microsecond" "microseconds" "us"
    "msec" "msecs" "millisecond" "milliseconds" "ms"
    "seconds" "second" "sec" "s"
    "minutes" "minute" "min" "m"
    "hours" "hour" "hr" "h"
    "days" "day" "d"
    "weeks" "week" "w"
    "months" "month" "M"
    "years" "year" "y"
  ];
  systemdTimeSpan =
    lib.types.strMatching
      "(infinity|([0-9]+(\\.[0-9]+)?[[:space:]]*(${systemdTimeSpanUnits})?[[:space:]]*)+)"
    // {
      description =
        "systemd time span per systemd.time(7) — e.g. \"8h\", \"90min\","
        + " \"1h 30min\", or bare seconds \"3600\" — or the literal"
        + " \"infinity\" to disable the timeout";
    };

  # ---- shared sandbox for the two units that run torch -------------------
  #
  # WHAT THESE UNITS ACTUALLY DO: pull ~2.4 GB of third-party weights off the
  # internet, and feed the corpus — web pages, PDFs, chat exports, GitHub
  # bodies, none of it authored by the user — through torch and a native
  # tokenizer. That is a large C++ attack surface chewing on untrusted bytes,
  # and until round 1 it did so with the user's full ambient authority.
  #
  # One binding rather than two copies, so the worker and the seeder cannot
  # drift apart on everything except the axis they are genuinely supposed to
  # differ on, which is the network.
  #
  # DELIBERATELY ABSENT, and each absence is load-bearing for torch:
  #
  #   MemoryDenyWriteExecute — torch's JIT and the OpenMP runtime allocate
  #     W|X pages. Setting it makes `import torch` die, and it is the single
  #     most likely directive for a future hardening pass to reach for. There
  #     is a check asserting it stays absent, precisely because adding it
  #     looks like an improvement.
  #   ProtectSystem=strict — would need an explicit ReadWritePaths for the HF
  #     cache; getting that list wrong fails at runtime, on units this branch
  #     cannot start on this host. `full` leaves $HOME writable, which is
  #     where the cache lives, and still makes /usr, /boot and /etc read-only.
  #   PrivateDevices, ProtectKernelModules, SystemCallFilter — not applied in
  #     round 1 and not added here. Nothing has ever executed this sandbox
  #     (see nix/README.md, "what remains unproven"), so widening it further
  #     on a host that cannot run it would be guesswork dressed as rigour.
  embedSandboxCommon = {
    NoNewPrivileges = true;
    # Its own /tmp. torch and huggingface both scribble there, and a shared
    # /tmp is a trivial channel between this and everything else the user runs.
    PrivateTmp = true;
    RestrictNamespaces = true;
    RestrictRealtime = true;
    RestrictSUIDSGID = true;
    LockPersonality = true;
    SystemCallArchitectures = "native";
    # `full`, NOT `strict` — see above.
    ProtectSystem = "full";
    ProtectKernelTunables = true;
    ProtectControlGroups = true;
  };

  # user-timer schema notes (verified against `man systemd.timer`, systemd v256):
  #   OnCalendar    — realtime calendar spec (`*:0/30` = every 30min of every hour).
  #   OnBootSec     — offset from boot; triggers once per boot after the delay.
  #                   Useful when the laptop was closed at the last OnCalendar tick.
  #   Persistent    — for OnCalendar timers only: on activation, if the last
  #                   scheduled run was missed (laptop closed / system off),
  #                   fire the unit immediately, then resume normal schedule.
  #                   User timers require the state file under $XDG_STATE_HOME —
  #                   home-manager sets this up correctly by default.
in {
  options.services.aggregator = {
    enable = lib.mkEnableOption "personal aggregator (sessions + GitHub cache)";

    package = lib.mkOption {
      type = lib.types.package;
      description = "The aggregator package (built from this flake).";
    };

    sources.sessions = {
      enable = lib.mkOption {
        type = lib.types.bool;
        default = true;
        description = ''
          Whether to install and start the sessions ingest timer.
          Set to false to disable this source entirely without editing
          the module (e.g. on machines that never run Claude Code
          locally, so ~/.claude/projects is empty).
        '';
      };

      interval = lib.mkOption {
        type = lib.types.str;
        default = "*:0/30";
        example = "hourly";
        description = ''
          systemd OnCalendar spec for the sessions ingest timer. Default
          is every 30min (`*:0/30`). Any calendar spec accepted by
          `systemd.time(7)` works.
        '';
      };

      since = lib.mkOption {
        type = lib.types.str;
        default = "";
        example = "2026-07-01T00:00:00Z";
        description = ''
          ISO-8601 timestamp to bound the sessions ingest window. Passed
          as `--since ISO` to the CLI. Default empty = ingest all time
          (correct for the first ~3h full-scan run; also correct for
          steady-state since the source is idempotent on stable_id).
          Set this to trim expensive walks on machines with a huge
          ~/.claude/projects backlog you don't care to reingest.
        '';
      };
    };

    sources.github = {
      enable = lib.mkOption {
        type = lib.types.bool;
        default = true;
        description = ''
          Whether to install and start the GitHub ingest timer. Set to
          false to disable this source entirely without editing the
          module (e.g. on machines with no `gh` auth configured).
        '';
      };

      interval = lib.mkOption {
        type = lib.types.str;
        default = "*:0/30";
        example = "hourly";
        description = "systemd OnCalendar spec for the GitHub ingest timer.";
      };

      githubTokenFile = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        example = "/run/agenix/github-readonly-pat";
        description = ''
          Absolute path to a file whose contents are a **read-only**
          GitHub PAT (scopes `public_repo`, `repo:status`, `read:org`
          only). Typically an agenix-managed secret. When set, the
          github ingest service reads the file at unit start and
          exports its contents as `GH_TOKEN` for the CLI invocation,
          overriding whatever `gh auth` has cached.

          When null (default), the CLI uses the existing `gh auth`
          token, which will refuse to ingest if the token has any
          write-capable scopes (see `pending_for_human.md`).
        '';
      };
    };

    embed = {
      enable = lib.mkOption {
        type = lib.types.bool;
        default = true;
        description = ''
          Whether to install the background embed worker (timer + service)
          that fills the v5 sqlite-vec index. Off means keyword-only
          (FTS5) recall: `aggregator_search_memory` still works, it just
          never gains the vector arm.

          Requires the aggregator package's Python closure to carry
          `sentence-transformers`, `torch` and `sqlite-vec`. Without them
          the unit fails loudly on every tick rather than degrading
          quietly — see `nix/README.md`.
        '';
      };

      interval = lib.mkOption {
        type = lib.types.str;
        default = "*:15/30";
        example = "hourly";
        description = ''
          systemd OnCalendar spec for the embed timer. Default `*:15/30`
          is every 30 minutes at :15 and :45 — deliberately offset from
          the ingest timers' `*:0/30`, so the common case is that an
          embed run and an ingest run are not opening the same SQLite
          database at the same moment.

          The offset is an optimisation, not the correctness story: an
          ingest run can last hours (TimeoutStartSec=4h on the deployed
          unit), so overlap is inevitable eventually. What makes overlap
          safe is on the store side — WAL journal mode plus a 30s
          busy_timeout — and on the worker side, one short write
          transaction per batch rather than one long one per run.
        '';
      };

      batchSize = lib.mkOption {
        type = lib.types.ints.positive;
        default = 500;
        description = ''
          Rows per embed batch, and NO LONGER THE BOUND THAT DECIDES
          WHAT A KILL COSTS. Each batch is still a checkpoint — its
          vectors and its watermark land in one transaction, so the
          watermark can never get ahead of the data — but the size of
          that checkpoint is bounded by CHUNKS as well as by rows
          (`cli._MAX_BATCH_CHUNKS`, about fifteen minutes of encoder time
          at the measured ~20 s per 4000-character chunk). Chunks are
          what the encoder is billed for, and rows differ in chunk count
          by two orders of magnitude: at 500 rows of dropbox that
          mismatch put the first durable checkpoint 6.4 hours away.

          What this option still bounds is rows the encoder never sees.
          An empty body costs no chunks, and about a third of the corpus
          is empty bodies, so without a row bound a batch of those would
          be unbounded. The chunk cap is deliberately not exposed here:
          it is derived from a measured rate, and a deployment that
          raised it would re-create the defect it closes.
        '';
      };

      timeoutStartSec = lib.mkOption {
        # NOT `types.str`. A mistyped span (`"infinty"`) used to evaluate,
        # build and deploy, and systemd would then reject it and apply its
        # ~90s default — truncating every tick of a month-long backfill,
        # with a stalled progress counter as the only visible symptom.
        # `systemdTimeSpan` rejects it at eval time, where the typo is.
        type = systemdTimeSpan;
        default = "infinity";
        example = "8h";
        description = ''
          `TimeoutStartSec` for the embed service. Disabled by default,
          because on this corpus no finite value can distinguish a healthy
          run from a broken one.

          Measured 2026-08-17 against the real cache: 483,193 observations
          / 422,261 chunks / 609M chars, embedding at 249.6 chars per
          wall-second on CPU. A first full backfill is therefore **25-30
          days of continuous work**, not the ~3.5h the design assumed.

          Being SIGTERMed is safe — the worker checkpoints per batch and
          the next tick resumes from the watermark — but it is not free:
          a timeout puts the unit in `failed`, which fires
          `OnFailure=aggregator-embed-failure-notify.service`. At the
          previous `8h` a *correctly progressing* backfill needed ~85
          consecutive runs and would have raised ~28 CRITICAL desktop
          notifications over a month (the popup is debounced to one a
          day), each saying the vector index is not being filled and
          naming two causes that did not apply. An alarm that fires on
          success is how a human learns to ignore the alarm, which costs
          the next real failure its audience.

          What guards a genuinely wedged worker instead, since this no
          longer does — a wall clock cannot tell "wedged" from "working"
          when working legitimately takes a month:

          - `Nice=19` + `IOSchedulingClass=idle` bound the cost of a
            spinning worker to otherwise-idle capacity.
          - Batch-sized write transactions plus the per-batch checkpoint
            bound what a wedge can lose or corrupt to one batch; it can
            never park a long transaction on the cache.
          - The worker's `flock` on `<cache>.embed.lock` means later timer
            ticks exit 0 as no-ops instead of stacking workers up.
          - `TimeoutStopSec` stays finite, so
            `systemctl --user stop aggregator-embed.service` is always a
            bounded kill.
          - Detection is by PROGRESS, not by clock: `aggregator status`
            and `aggregator_capabilities()['vector_index']` report
            embedded / pending / error counts, and a wedged worker is one
            whose counts stop moving. That is the only signal that can
            actually tell the two apart.

          Set a finite value (e.g. `"8h"`) only if you would rather cap
          the wall time than keep the notifier truthful.
        '';
      };
    };

    mcp.autoRegister = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = ''
        If true, run `claude mcp add aggregator …` on home-manager
        activation (idempotent — checks whether the entry exists first
        via `claude mcp list`). If false (default), just print the
        command in the activation output so you can run it yourself
        after inspecting.

        The activation script never writes directly to `~/.claude.json`
        — always goes through the `claude` CLI so Claude Code's own
        validation runs.
      '';
    };

    # Compatibility shim: prior module exposed `mcpRegistration = "manual"
    # | "activation-script"`. Keep it working so existing home-manager
    # configs don't break on upgrade — map onto `mcp.autoRegister`.
    mcpRegistration = lib.mkOption {
      type = lib.types.nullOr (lib.types.enum [ "manual" "activation-script" ]);
      default = null;
      description = ''
        Deprecated: use `mcp.autoRegister` instead. When set, overrides
        `mcp.autoRegister` (`manual` → false, `activation-script` → true).
      '';
    };
  };

  config = lib.mkIf cfg.enable (
    let
      autoRegister =
        if cfg.mcpRegistration == "activation-script" then true
        else if cfg.mcpRegistration == "manual" then false
        else cfg.mcp.autoRegister;
    in
    {
      home.packages = [ cfg.package ];

      # ---- sessions ------------------------------------------------------
      systemd.user.services.aggregator-sessions = lib.mkIf cfg.sources.sessions.enable {
        Unit.Description = "Aggregator: sessions ingest";
        Service = {
          Type = "oneshot";
          ExecStart = mkExecStart {
            source = "sessions";
            since = cfg.sources.sessions.since;
            tokenFile = null;  # sessions ingest reads only local filesystem
          };
          StandardOutput = "journal";
          StandardError = "journal";
        };
      };
      systemd.user.timers.aggregator-sessions = lib.mkIf cfg.sources.sessions.enable {
        Unit.Description = "Aggregator: sessions ingest timer";
        Timer = {
          OnCalendar = cfg.sources.sessions.interval;
          # First-run cost is ~3.1h against a real ~/.claude/projects tree
          # (measured 2026-08-02, 5678 sessions / 348168 observations).
          # OnBootSec fires 5min after boot so a laptop that was closed
          # across the OnCalendar window ingests soon after resume,
          # without racing early boot / VPN / net-online.
          OnBootSec = "5min";
          # If the last OnCalendar tick was missed (laptop closed), fire
          # immediately on activation, then continue on schedule.
          Persistent = true;
        };
        Install.WantedBy = [ "timers.target" ];
      };

      # ---- github --------------------------------------------------------
      systemd.user.services.aggregator-github = lib.mkIf cfg.sources.github.enable {
        Unit.Description = "Aggregator: github ingest";
        Service = {
          Type = "oneshot";
          ExecStart = mkExecStart {
            source = "github";
            since = "";  # github source paginates by /search/issues, --since not wired end-to-end
            tokenFile = cfg.sources.github.githubTokenFile;
          };
          StandardOutput = "journal";
          StandardError = "journal";
        };
      };
      systemd.user.timers.aggregator-github = lib.mkIf cfg.sources.github.enable {
        Unit.Description = "Aggregator: github ingest timer";
        Timer = {
          OnCalendar = cfg.sources.github.interval;
          # Codex Phase 2 MEDIUM: stagger against aggregator-sessions.
          # Both timers default to `*:0/30` + OnBootSec=5min; without a
          # delay they land in the same tick and both open a writer
          # against the same cache.db. Sessions ingest is by far the
          # heavier writer, so we jitter github by up to 3 min.
          # busy_timeout=30s on the store side absorbs the residual
          # overlap. This does NOT protect against a full sessions
          # rebuild (--rebuild) which holds a savepoint for hours; for
          # that case, disable this timer temporarily.
          OnBootSec = "5min";
          RandomizedDelaySec = "3min";
          Persistent = true;
        };
        Install.WantedBy = [ "timers.target" ];
      };

      # ---- embed worker --------------------------------------------------
      # Fills the v5 vector index in the background. Never on the recall
      # path: queries fall through to FTS5 for anything not yet embedded,
      # so a cold or half-full index degrades ranking, never availability.
      systemd.user.services.aggregator-embed = lib.mkIf cfg.embed.enable {
        Unit = {
          Description = "Aggregator: background embed worker (vector index)";
          OnFailure = "aggregator-embed-failure-notify.service";
        };
        Service = {
          Type = "oneshot";
          ExecStart = "${embedRunner}";
          Environment = embedBaseEnvironment ++ [
            # Offline by construction. The weights are seeded once by
            # aggregator-embed-seed.service; after that this unit must never
            # reach the network, so a Hugging Face outage, a rate limit or a
            # renamed repo cannot turn a background indexer into a fetcher
            # that retries a 1.2 GB download every 30 minutes. A missing
            # cache is caught by the preflight above and reported with the
            # exact command that fixes it.
            "HF_HUB_OFFLINE=1"
            # huggingface_hub reads the newer var; transformers still checks
            # the legacy one on some paths. Set both so "offline" cannot be
            # half-true.
            "TRANSFORMERS_OFFLINE=1"
          ];
          # Background work on an interactive laptop. The embedder is
          # CPU-bound and will happily take every core otherwise.
          #
          # NICE ALONE DID NOT DO THIS, and the unit ran for weeks as if it
          # had. Nice orders who the scheduler picks FIRST; it says nothing
          # about how many run AT ONCE, so twelve threads at nice 19 saturate
          # twelve cores whenever nothing else wants them — which, on a
          # backfill measured in weeks, is most of the time. The measurement
          # that settled it is in `embedThreads` above: 1d 7h 51min of CPU
          # over 4h of wall clock.
          Nice = 19;
          IOSchedulingClass = "idle";
          # THE HARD BOUND, paired with the thread pools in
          # `embedBaseEnvironment` and derived from the same binding so they
          # cannot drift apart. The environment variables are a REQUEST — any
          # library that ignores them, and any subprocess spawned along the
          # way, is outside them. This is the boundary: it holds whatever the
          # process does to itself, exactly as `RestrictAddressFamilies` does
          # for the offline sandbox further down.
          CPUQuota = "${toString (embedThreads * 100)}%";
          # And when the operator IS using the machine, yield rather than
          # merely queue behind them. `CPUWeight` is the cgroup-v2 successor to
          # nice for contention between cgroups, which is the axis that
          # actually decides whether typing stutters; nice only orders threads
          # within one.
          CPUWeight = 10;
          TimeoutStartSec = cfg.embed.timeoutStartSec;
          # The window between SIGTERM and SIGKILL, and round 2 changed what
          # it buys. This used to say the worker installed no SIGTERM handler
          # and that correctness came from committing vectors before the
          # watermark. BOTH premises are now false: `cli.py` wraps the embed
          # loop in `graceful_shutdown()`, and `Store.commit_embed_batch` is a
          # single transaction rather than two ordered commits.
          #
          # What the timeout is for NOW. SIGTERM sets a flag — no work in the
          # handler — which the loop reads at a ROW boundary, not a batch one.
          # The rows already embedded are flushed through the same one-shot
          # commit, the in-flight claim on the current row is released, and
          # the process exits cleanly.
          #
          # A LONG ROW OUTLIVES THIS WINDOW, and nothing finite would change
          # that. A row's encode is a single `embed_documents` call over all of
          # that row's chunks, and nothing inside the call looks at the stop
          # flag — the flag is only read once it returns. Measured read-only
          # against the live cache at the chunker's `chunk-4000-400` geometry
          # and the measured ~20 s per chunk: 1348 rows (1298 observations + 50
          # records) each exceed 300 s in one call, and the largest single row
          # is 257 chunks, about 86 minutes. So 5min is roughly three orders of
          # magnitude under that tail. It is sized to bound a wedged worker
          # (see FINITE below), never to let a long row finish.
          #
          # That leaves a KNOWN, CURRENTLY UNFIXED gap. Stop the unit while one
          # of those rows is encoding and systemd escalates to SIGKILL: the
          # on-disk claim survives, the next run's `_blame_crashed_row` reads
          # it as a crash and attributes it to that row, and the row is booked
          # into the poison ledger. Three such stops make it terminal — a good
          # row dropped from the vector arm permanently. Reproduced against a
          # real spawned worker and a real SIGTERM→SIGKILL sequence, and it
          # reproduces identically on `main`, so the batch-bounding work
          # neither introduced it nor closed it. Changing this number does not
          # close it either: nothing finite covers an unbounded call. The fix
          # has to be inside the worker — a stop the encoder itself can reach,
          # or a crash attribution that can tell a SIGKILL-at-stop from a row
          # that genuinely killed the process.
          #
          # What the window still PREVENTS. The claim a row leaves on disk is
          # the worker's crash detector: only code that runs can clear it, so a
          # claim found at startup means the previous worker died on that row
          # and it gets set aside as poison. A stop that reaches its boundary
          # clears the claim and is therefore invisible to that logic; a
          # SIGKILL is not. Shortening this window would spread the gap above
          # from the long tail to EVERY routine `systemctl --user stop`, reboot
          # and deploy, on a backlog measured in weeks. Losing the batch would
          # be cheap; losing the row from the index quietly is not.
          #
          # And it stays FINITE. With TimeoutStartSec=infinity above, a manual
          # stop is the last bound on a wedged worker, so it must itself
          # complete — asserted by `aggregator-embed-unit-hygiene` step 6.
          TimeoutStopSec = "5min";
          StandardOutput = "journal";
          StandardError = "journal";

        } // embedSandboxCommon // {
          # ---- sandbox: the OFFLINE half ----------------------------------
          # This unit's "offline" used to be two environment variables. An env
          # var is a request, not a boundary: any library that ignores it, or
          # any subprocess spawned along the way, had the entire network.
          #
          # RestrictAddressFamilies is the load-bearing line here. seccomp,
          # supported in a USER manager, and it makes an AF_INET socket()
          # fail outright — so "does not talk to the network" stops resting
          # on every library agreeing to read HF_HUB_OFFLINE. AF_UNIX stays
          # for journal and dbus; AF_NETLINK because glibc probes interfaces
          # during resolver setup even when nothing ever connects.
          RestrictAddressFamilies = "AF_UNIX AF_NETLINK";
          # Belt to that brace, and deliberately second: IP filtering is BPF
          # and a user manager only gets it with cgroup delegation, so this
          # may be a no-op here. It costs nothing when unsupported and covers
          # the case where a future revision has to re-widen the address
          # families above.
          IPAddressDeny = "any";
        };
      };

      systemd.user.timers.aggregator-embed = lib.mkIf cfg.embed.enable {
        Unit.Description = "Aggregator: background embed worker timer";
        Timer = {
          OnCalendar = cfg.embed.interval;
          # 15min after boot, so a resumed laptop does ingest (5min) first
          # and embeds what that produced, rather than racing it.
          OnBootSec = "15min";
          # Small on purpose. The point of `interval` is the offset from
          # the ingest ticks; a wide jitter would smear it back over them.
          RandomizedDelaySec = "1min";
          Persistent = true;
        };
        Install.WantedBy = [ "timers.target" ];
      };

      # One-time weight seeding. Deliberately has no [Install] section and
      # no timer: it is started by hand, exactly once per machine, and the
      # embed worker's failure message names it verbatim.
      systemd.user.services.aggregator-embed-seed = lib.mkIf cfg.embed.enable {
        Unit = {
          Description = "Aggregator: one-time download of the Qwen3 embedding + reranker weights (~2.4 GB)";
          # ITS OWN notifier, not the worker's. See `mkFailureNotify`: sharing
          # one sent the operator to a journal this unit never wrote to, told
          # them to run the unit that had just failed, and let either unit's
          # failure silence the other for 24h through a shared stamp.
          OnFailure = "aggregator-embed-seed-failure-notify.service";
        };
        Service = {
          Type = "oneshot";
          ExecStart = "${embedSeeder}";
          Environment = embedBaseEnvironment ++ [ "HF_HUB_OFFLINE=0" ];
          # Two ~1.2 GB downloads on a slow link. Was `2h` when this unit
          # fetched one model; the payload doubled with the reranker, so the
          # budget does too. A timeout here throws away a partial download
          # that would restart from scratch, and the unit is human-triggered
          # and one-shot — there is nothing to protect by cutting it short.
          TimeoutStartSec = "4h";
          StandardOutput = "journal";
          StandardError = "journal";
        } // embedSandboxCommon // {
          # ---- sandbox: the ONLINE half -----------------------------------
          # Round-2 LOW: this unit had ZERO sandbox directives. Measured with
          # `systemd-analyze security --offline=true --user` on the rendered
          # files: aggregator-embed.service scored 6.3 MEDIUM after round 1,
          # this one 9.2 UNSAFE. It is the unit that reaches the public
          # internet and then loads ~2.4 GB of third-party weights into torch,
          # so "it only runs when a human starts it" bounds how often that
          # happens, not what it can do when it does.
          #
          # It shares `embedSandboxCommon` with the worker. Exactly two
          # directives are relaxed, and only because downloading is the
          # unit's entire purpose:
          #
          #   RestrictAddressFamilies gains AF_INET + AF_INET6. It cannot be
          #     dropped to "no restriction": keeping the directive still bars
          #     AF_PACKET (raw frames), AF_BLUETOOTH, AF_VSOCK and the rest of
          #     the exotic families that carry most of the socket-family
          #     kernel attack surface. A downloader needs TCP over IP and
          #     nothing else.
          #
          #   IPAddressDeny is omitted rather than set. `any` would block the
          #     download outright; there is no useful allowlist to put here
          #     either, because huggingface.co resolves to a CDN whose address
          #     set changes without notice, and an allowlist that goes stale
          #     turns into a mystery failure on the one unit a human runs by
          #     hand and watches. `HF_HUB_OFFLINE=0` in Environment above is
          #     what marks this unit as the network one.
          #
          # Everything else in the common set survives unchanged: fetching
          # weights needs no new privileges, no namespaces, no realtime
          # scheduling, no setuid, no writable /usr, and no shared /tmp.
          #
          # NOT PROVEN BY EXECUTION — see nix/README.md. This host's
          # aggregator-env lacks torch, so no process has run under these
          # directives. What is verified is the rendered file and its score.
          RestrictAddressFamilies = "AF_UNIX AF_NETLINK AF_INET AF_INET6";
        };
      };

      systemd.user.services.aggregator-embed-failure-notify =
        lib.mkIf cfg.embed.enable {
          Unit.Description = "Desktop notification: aggregator embed run failed";
          Service = {
            Type = "oneshot";
            ExecStart = "${embedFailureNotify}";
            StandardOutput = "journal";
            StandardError = "journal";
          };
        };

      systemd.user.services.aggregator-embed-seed-failure-notify =
        lib.mkIf cfg.embed.enable {
          Unit.Description =
            "Desktop notification: aggregator model download failed";
          Service = {
            Type = "oneshot";
            ExecStart = "${embedSeedFailureNotify}";
            StandardOutput = "journal";
            StandardError = "journal";
          };
        };

      # ---- MCP registration ---------------------------------------------
      # Manual mode: print the command. `autoRegister` mode: run it via
      # the `claude` CLI (never touch ~/.claude.json ourselves).
      home.activation.aggregatorMcpRegister =
        lib.hm.dag.entryAfter [ "writeBoundary" ] (
          if autoRegister then ''
            if command -v claude >/dev/null 2>&1; then
              if ! claude mcp list 2>/dev/null | ${pkgs.gnugrep}/bin/grep -q '^aggregator[[:space:]]'; then
                echo "aggregator: registering MCP with Claude Code"
                claude mcp add aggregator ${aggregatorMcpBin} || \
                  echo "aggregator: claude mcp add failed; register manually"
              else
                echo "aggregator: MCP already registered, skipping"
              fi
            else
              echo "aggregator: 'claude' not on PATH; skipping MCP auto-register"
              echo "aggregator: run manually — claude mcp add aggregator ${aggregatorMcpBin}"
            fi
          '' else ''
            echo "aggregator: to register the MCP with Claude Code, run:"
            echo "  claude mcp add aggregator ${aggregatorMcpBin}"
          ''
        );
    }
  );
}
