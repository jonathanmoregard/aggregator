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
