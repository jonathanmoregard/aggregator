{ config, lib, pkgs, ... }:
let
  cfg = config.services.aggregator;
  aggregatorBin = "${cfg.package}/bin/aggregator";
  aggregatorMcpBin = "${cfg.package}/bin/aggregator-mcp";

  # Build the store-pinned ExecStart command for a source, threading in
  # optional --since. GitHub authentication stays owned by the gh CLI keyring;
  # the source constrains its subprocess boundary to GET-only search requests.
  mkExecStart = { source, since }:
    let
      base = "${aggregatorBin} ingest ${source}"
        + lib.optionalString (since != "") " --since ${lib.escapeShellArg since}";
    in
      base;

  # ---- LLM record tagging -----------------------------------------------
  #
  # Daily backfill of LLM topic tags onto the records-shaped sources
  # (`aggregator tag`). Needs the network — the `claude` CLI talks to the
  # API — so it is shaped like aggregator-github (network-permitted oneshot
  # + timer + OnFailure notifier).
  #
  # HOW `claude` IS FOUND, and why that satisfies the 2026-08-16 deployment
  # constraint: the CLI subprocesses the bare name `claude` from PATH,
  # exactly as the github ingest finds `gh` — and PATH is an environment
  # fact no comment can pin, so the preflight below ENFORCES it at unit
  # start instead of asserting it. On this machine the user manager's PATH
  # is the NixOS profile chain (/etc/profiles/per-user/<user>/bin,
  # /run/current-system/sw/bin, ...), whose `claude` is a symlink into the
  # Nix store — a pinned artifact. What the guard exists to refuse is the
  # OTHER install: ~/.local/bin's self-updating native build, which is a
  # working-tree-shaped dependency by another name (unpinned, mutates
  # underneath the unit). No /home/ path is ever written into the unit or
  # this script; the guard resolves whatever PATH offers and requires the
  # real binary to live in the store.
  tagRunner = pkgs.writeShellScript "aggregator-tag" ''
    set -uo pipefail

    if ! command -v claude >/dev/null 2>&1; then
      echo "aggregator-tag: no 'claude' CLI on the unit's PATH ($PATH)." >&2
      echo "aggregator-tag: install claude-code into the system or home-manager profile; the tag backfill cannot run without it." >&2
      exit 1
    fi

    resolved=$(command -v claude)
    real=$(readlink -f "$resolved")
    case "$real" in
      /nix/store/*) ;;
      *)
        echo "aggregator-tag: 'claude' resolved to $resolved -> $real, which is not a Nix store artifact." >&2
        echo "aggregator-tag: a service must run a deployed, pinned binary — not a self-updating install. Put claude-code in the system or home-manager profile and make sure the unit's PATH prefers it." >&2
        exit 1
        ;;
    esac

    # `exec` so the CLI's exit status IS the unit's: `aggregator tag` exits
    # non-zero whenever any record failed, and that status must reach
    # systemd unaltered for OnFailure= to fire (constraint: fail loudly).
    exec ${aggregatorBin} tag
  '';

  # OnFailure target. Mirrors the deployed aggregator-ingest-failure-notify
  # unit (journal line + CRITICAL libnotify popup), with one addition: the
  # popup is debounced to once per day, because a failure that persists until
  # a human acts would otherwise raise a popup every tick, be muted, and turn
  # a loud system into a silent one. The journal line is NOT debounced.
  #
  # ONE NOTIFIER PER FAILING UNIT, each with its own stamp, so one unit's
  # failure can never buy silence for another's.
  #
  # The debounce fails OPEN on both halves:
  #
  #   read  — any error stat-ing the stamp leaves `recent` empty, so we
  #           notify rather than assume we already did.
  #   write — the stamp is armed ONLY after notify-send exits 0. A debounce is
  #           a record of "the human was told", and a failed send is precisely
  #           the case where they were not.
  #
  # `$SERVICE_RESULT` / `$EXIT_CODE` are NOT available to a separate
  # `OnFailure=` unit, which is why the failing unit's identity is baked in at
  # generation time rather than read from the environment at runtime.
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

  tagFailureNotify = mkFailureNotify {
    name = "aggregator-tag-failure-notify";
    unit = "aggregator-tag.service";
    stamp = "tag-failure-notified";
    summary = "aggregator tag FAILED";
    body =
      "The LLM record-tagging run exited non-zero, so some records are"
      + " missing topic tags and tag: search under-selects for them."
      + " Likely: the claude CLI is not on the unit's PATH, its auth"
      + " expired, a rate limit, or records the model kept mis-answering"
      + " (those are named on stderr). Free-text search is unaffected;"
      + " the next daily run retries what failed.";
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

    };

    tag = {
      enable = lib.mkOption {
        type = lib.types.bool;
        default = true;
        description = ''
          Whether to install the daily LLM record-tagging timer
          (`aggregator tag`). Off means records keep only their
          source-written tags: `tag:` search still works, it just never
          gains the LLM topic layer, and llm_tag_coverage reports
          not_started forever. Requires the `claude` CLI in the deployed
          profile (subscription auth — no API key is configured here).
        '';
      };

      interval = lib.mkOption {
        type = lib.types.str;
        default = "daily";
        example = "weekly";
        description = ''
          systemd OnCalendar spec for the tag timer. Daily is deliberate:
          the watermark (llm_tags_src_hash vs src_hash) makes a run over
          an already-tagged corpus a fast no-op, so the cost of a tick is
          proportional to what actually changed that day — a handful of
          records, one or two claude invocations.
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

      # ---- LLM record tagging -------------------------------------------
      # Network-permitted like the github ingest: the claude CLI needs the
      # API. It runs no model locally; it shells out to a CLI that must reach
      # its endpoint.
      systemd.user.services.aggregator-tag = lib.mkIf cfg.tag.enable {
        Unit = {
          Description = "Aggregator: LLM topic tags for records (fills llm_tags)";
          OnFailure = "aggregator-tag-failure-notify.service";
        };
        Service = {
          Type = "oneshot";
          # Type=oneshot defaults TimeoutStartSec to infinity, which is
          # right here: the FIRST backfill over ~4.4k records is hours of
          # claude invocations, and a wall clock cannot tell that from a
          # wedge. What bounds a wedge:
          # the CLI's per-invocation timeout (300s, bounded retries), the
          # per-batch committed checkpoint (a SIGTERM costs at most one
          # batch — graceful_shutdown stops at the boundary), and the timer
          # not stacking runs (a oneshot still `activating` is not
          # re-triggered).
          ExecStart = "${tagRunner}";
          StandardOutput = "journal";
          StandardError = "journal";
          # A sandbox, to the extent the claude CLI tolerates — every
          # directive below verified on this host (2026-09-04): `claude -p
          # --model haiku` under exactly this set via `systemd-run --user`
          # answered and exited 0. The prompt carries attacker-influenced
          # record bodies, so the unit's ambient surface is shrunk.
          #
          # DELIBERATELY ABSENT, each load-bearing for `claude -p`:
          #   ProtectHome — the CLI must READ ~/.claude config+credentials
          #     and WRITE its own state under $HOME. `ProtectSystem=full`
          #     still makes /usr, /boot and /etc read-only.
          #   PrivateNetwork / IPAddressDeny — the CLI exists to reach the
          #     API; this unit is network-permitted like aggregator-github.
          # The hygiene check asserts both directions (present + absent).
          NoNewPrivileges = true;
          PrivateTmp = true;
          RestrictRealtime = true;
          RestrictSUIDSGID = true;
          LockPersonality = true;
          ProtectSystem = "full";
          ProtectKernelTunables = true;
          ProtectControlGroups = true;
        };
      };

      systemd.user.timers.aggregator-tag = lib.mkIf cfg.tag.enable {
        Unit.Description = "Aggregator: LLM record tagging timer";
        Timer = {
          OnCalendar = cfg.tag.interval;
          # Wide jitter on purpose: this tick spends API rate-limit quota
          # from the same subscription the interactive tooling uses, so it
          # should not land at a predictable minute.
          RandomizedDelaySec = "45min";
          # A laptop closed over the scheduled time still tags that day's
          # records on resume.
          Persistent = true;
        };
        Install.WantedBy = [ "timers.target" ];
      };

      systemd.user.services.aggregator-tag-failure-notify =
        lib.mkIf cfg.tag.enable {
          Unit.Description = "Desktop notification: aggregator tag run failed";
          Service = {
            Type = "oneshot";
            ExecStart = "${tagFailureNotify}";
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
