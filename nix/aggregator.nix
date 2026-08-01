{ config, lib, pkgs, ... }:
let
  cfg = config.services.aggregator;
  aggregatorBin = "${cfg.package}/bin/aggregator";
  aggregatorMcpBin = "${cfg.package}/bin/aggregator-mcp";
in {
  options.services.aggregator = {
    enable = lib.mkEnableOption "personal aggregator (sessions + GitHub cache)";

    package = lib.mkOption {
      type = lib.types.package;
      description = "The aggregator package (built from this flake).";
    };

    sessions = {
      interval = lib.mkOption {
        type = lib.types.str;
        default = "30min";
        description = ''
          systemd OnCalendar interval for sessions ingest.
          The timer below uses "*:0/30" (every 30min) to match the github
          timer; set this option for documentation/override in downstream
          modules (advisor round-1 MEDIUM: prior default was a misleading
          "1h" that didn't match what the timer actually did).
        '';
      };
    };

    github = {
      interval = lib.mkOption {
        type = lib.types.str;
        default = "30min";
        description = "systemd OnCalendar interval for github ingest.";
      };
    };

    mcpRegistration = lib.mkOption {
      type = lib.types.enum [ "manual" "activation-script" ];
      default = "manual";
      description = ''
        How to register the aggregator MCP with Claude Code.
        'manual' prints the `claude mcp add` command in nix/README.md.
        'activation-script' runs a small script on home-manager activation
        that writes an entry into ~/.claude.json (only if not present).
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    home.packages = [ cfg.package ];

    systemd.user.services.aggregator-sessions = {
      Unit.Description = "Aggregator: sessions ingest";
      Service = {
        Type = "oneshot";
        ExecStart = "${aggregatorBin} ingest sessions";
        StandardOutput = "journal";
        StandardError = "journal";
      };
    };
    systemd.user.timers.aggregator-sessions = {
      Unit.Description = "Aggregator: sessions ingest timer";
      Timer = {
        # Both timers run every 30min ("*:0/30"). If you need a different
        # cadence per source, override this via a downstream module.
        OnCalendar = "*:0/30";
        Persistent = true;
      };
      Install.WantedBy = [ "timers.target" ];
    };

    systemd.user.services.aggregator-github = {
      Unit.Description = "Aggregator: github ingest";
      Service = {
        Type = "oneshot";
        ExecStart = "${aggregatorBin} ingest github";
        StandardOutput = "journal";
        StandardError = "journal";
      };
    };
    systemd.user.timers.aggregator-github = {
      Unit.Description = "Aggregator: github ingest timer";
      Timer = {
        OnCalendar = "*:0/30";
        Persistent = true;
      };
      Install.WantedBy = [ "timers.target" ];
    };

    # MCP registration: manual-first (documented in nix/README.md). The
    # activation-script variant is intentionally minimal; auto-registration
    # is opt-in per spec (M4 does not auto-register).
    home.activation.aggregatorMcpRegisterNote =
      lib.mkIf (cfg.mcpRegistration == "manual")
        (lib.hm.dag.entryAfter [ "writeBoundary" ] ''
          echo "aggregator: To register the MCP with Claude Code, run:"
          echo "  claude mcp add aggregator ${aggregatorMcpBin}"
        '');

    home.activation.aggregatorMcpRegisterScript =
      lib.mkIf (cfg.mcpRegistration == "activation-script")
        (lib.hm.dag.entryAfter [ "writeBoundary" ] ''
          if ! ${pkgs.jq}/bin/jq -e '.mcpServers.aggregator' "$HOME/.claude.json" >/dev/null 2>&1; then
            echo "aggregator: adding MCP entry to ~/.claude.json"
            tmp=$(${pkgs.coreutils}/bin/mktemp)
            ${pkgs.jq}/bin/jq --arg cmd "${aggregatorMcpBin}" \
              '.mcpServers.aggregator = {command: $cmd, args: []}' \
              "$HOME/.claude.json" > "$tmp" && mv "$tmp" "$HOME/.claude.json"
          fi
        '');
  };
}
