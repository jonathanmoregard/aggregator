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

  # Environment shared by the embed worker and its one-shot seeding sibling.
  # `AGGREGATOR_EMBED_BACKEND=st` pins the safetensors loader; the `gguf`
  # backend needs an optional extra that is not in the deployed closure, and a
  # backend that silently changes under a unit is a debugging trap.
  embedBaseEnvironment = [
    "SSL_CERT_FILE=${caBundle}"
    "NIX_SSL_CERT_FILE=${caBundle}"
    "HF_HOME=${hfHome}"
    "AGGREGATOR_EMBED_BACKEND=st"
  ];

  # Shell fragment resolving the model snapshot dir from whatever HF_HOME the
  # unit actually got, falling back to huggingface_hub's own default so a
  # hand-run of the script outside systemd reports the same diagnosis.
  # POSIX-only: no coreutils on PATH is assumed.
  modelPresenceCheck = ''
    hf_home="''${HF_HOME:-''${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
    snapshots="$hf_home/hub/${embedModelDir}/snapshots"
    set -- "$snapshots"/*
    if [ ! -d "$snapshots" ] || [ ! -e "$1" ]; then
      have_weights=0
    else
      have_weights=1
    fi
  '';

  # ExecStart for the timer-driven worker.
  #
  # `--catchup`, not the plan's `--once`. `--once` does a single batch per
  # tick; at batch-size 500 against a ~376k-observation corpus that is ~380
  # ticks, i.e. over a week of wall time to reach a usable index, which is a
  # backfill that never finishes for any practical purpose. `--catchup`
  # drains the backlog in the same bounded, per-batch-committed chunks and is
  # a fast no-op once the index is warm. Overlap is already handled by the
  # worker's own flock on `<cache>.embed.lock`: a tick that finds a catchup
  # still running prints one line and exits 0.
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
    if [ "$have_weights" -ne 1 ]; then
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
  # `embed --once --batch-size 1` is deliberate: `_cmd_embed` constructs the
  # Embedder before it looks at the backlog, so this downloads the weights
  # even on an already-embedded corpus, and then does exactly one batch — a
  # live end-to-end proof of weights + torch + sqlite-vec + a real DB write,
  # rather than a download that is only proven when the timer next fires.
  embedSeeder = pkgs.writeShellScript "aggregator-embed-seed" ''
    set -uo pipefail

    ca_bundle="''${SSL_CERT_FILE:-}"
    if [ -z "$ca_bundle" ] || [ ! -s "$ca_bundle" ]; then
      echo "aggregator-embed-seed: no usable CA bundle (SSL_CERT_FILE='$ca_bundle') — the download would fail CERTIFICATE_VERIFY_FAILED against an empty trust store" >&2
      exit 1
    fi

    ${modelPresenceCheck}
    if [ "$have_weights" -eq 1 ]; then
      echo "aggregator-embed-seed: ${embedModelRepo} already present under $snapshots — nothing to download; running one batch as a live check."
    else
      echo "aggregator-embed-seed: downloading ${embedModelRepo} (~1.2 GB) into $hf_home. This is a one-time cost."
    fi

    exec ${aggregatorBin} embed --once --source observations --batch-size 1
  '';

  # OnFailure target. Mirrors the deployed aggregator-ingest-failure-notify
  # unit (journal line + CRITICAL libnotify popup), with one addition: the
  # popup is debounced to once per day.
  #
  # The embed timer fires every 30 minutes. Its two standing failure modes —
  # weights absent, sqlite-vec absent — are both *persistent* until a human
  # acts, so an undebounced popup would fire 48 times a day and be muted,
  # which is how a loud system becomes a silent one. Same reasoning as
  # `60a931d` (report permanently-bad input once, not twice an hour). The
  # journal line is NOT debounced; only the desktop popup is. The stamp logic
  # fails OPEN — any error reading it results in a notification.
  embedFailureNotify = pkgs.writeShellScript "aggregator-embed-failure-notify" ''
    set -uo pipefail

    echo "aggregator embed run FAILED — inspect: journalctl --user -u aggregator-embed.service -n 200"

    stamp_dir="''${XDG_STATE_HOME:-$HOME/.local/state}/aggregator"
    stamp="$stamp_dir/embed-failure-notified"
    recent="$(${pkgs.findutils}/bin/find "$stamp" -mmin -1440 2>/dev/null)"
    if [ -n "$recent" ]; then
      echo "desktop notification suppressed — already notified within 24h (stamp: $stamp). Failure is in the journal above."
      exit 0
    fi

    ${pkgs.coreutils}/bin/mkdir -p "$stamp_dir" 2>/dev/null
    ${pkgs.coreutils}/bin/touch "$stamp" 2>/dev/null

    if ! ${pkgs.libnotify}/bin/notify-send -u critical -a aggregator \
      "aggregator embed FAILED" \
      "The background embed worker exited non-zero, so the vector index is not being filled. Likely: Qwen3 weights missing from the HF cache (fix: systemctl --user start aggregator-embed-seed.service), or the sqlite-vec extension did not load. Keyword search is unaffected. Details: journalctl --user -u aggregator-embed.service -n 200"; then
      echo "notify-send failed (no notification daemon on session bus?) — failure recorded in journal only"
    fi
  '';

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
          Rows per embed batch. Each batch is a checkpoint: vectors are
          committed, then `embedding_state` advances, in that order. A
          kill at any instant therefore costs at most this many rows of
          recomputation and can never leave the watermark ahead of the
          data. Bigger batches amortise model overhead; smaller ones
          shorten the window a SIGTERM can waste.
        '';
      };

      timeoutStartSec = lib.mkOption {
        type = lib.types.str;
        default = "8h";
        example = "infinity";
        description = ''
          `TimeoutStartSec` for the embed service. Sized for a first full
          backfill of a ~376k-observation corpus on CPU, which is hours,
          not minutes. Being SIGTERMed at the timeout is safe and
          expected — the worker checkpoints per batch and the next tick
          resumes from the watermark — so this is a runaway guard, not a
          deadline the backfill must beat.
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
          Nice = 19;
          IOSchedulingClass = "idle";
          TimeoutStartSec = cfg.embed.timeoutStartSec;
          # Give an in-flight batch room to commit before SIGKILL. Note the
          # worker installs no SIGTERM handler, so a stop is immediate
          # regardless; correctness comes from the commit ordering (vectors
          # first, watermark second), which makes the worst case a repeated
          # batch — the vec upserts are delete-then-insert, hence idempotent
          # — and never a row marked embedded with no vector behind it.
          TimeoutStopSec = "5min";
          StandardOutput = "journal";
          StandardError = "journal";
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
          Description = "Aggregator: one-time download of the Qwen3 embedding weights (~1.2 GB)";
          OnFailure = "aggregator-embed-failure-notify.service";
        };
        Service = {
          Type = "oneshot";
          ExecStart = "${embedSeeder}";
          Environment = embedBaseEnvironment ++ [ "HF_HUB_OFFLINE=0" ];
          # A 1.2 GB download on a slow link, plus one batch.
          TimeoutStartSec = "2h";
          StandardOutput = "journal";
          StandardError = "journal";
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
