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
    sessions.interval = "1h";     # spec default (see note below)
    github.interval = "30min";
    mcpRegistration = "manual";   # or "activation-script"
  };
}
```

Add the flake input in your top-level `flake.nix`:

```nix
inputs.aggregator.url = "path:/home/jonathan/Repos/aggregator";
# or: inputs.aggregator.url = "github:jonathan/aggregator";
```

## Register the MCP with Claude Code

The `manual` mode (default) does not touch `~/.claude.json`. Run this once
after activation:

```bash
claude mcp add aggregator $(which aggregator-mcp)
```

The `activation-script` mode adds the entry on home-manager activation,
idempotently — only inserts `mcpServers.aggregator` if not already present.

## Verify

```bash
systemctl --user list-timers | grep aggregator
journalctl --user -u aggregator-sessions -n 50
journalctl --user -u aggregator-github -n 50
aggregator status
```

## Python deps that may need overlays (follow-up)

`fastmcp`, `presidio-analyzer`, `presidio-anonymizer`, and `claude-runner`
are PyPI-only and may not be packaged in nixpkgs. Consequences:

- `packages.default` from this flake is intentionally thin
  (`propagatedBuildInputs` is empty); a plain `nix build .#default` will
  succeed at build-graph time but the resulting binary will fail at import
  time until those deps are available.
- **Production path**: run the CLI/MCP inside a `uv run` shell that resolves
  the PyPI deps at runtime, wrapping via `nix run .#default -- ...` once an
  overlay is added; or invoke `uv run aggregator ...` inside the devShell.
- **Overlay follow-up**: package the four PyPI-only deps as an overlay under
  `nix/overlays/` and add them to `propagatedBuildInputs`. Track separately;
  M4 ships the module + timer scaffolding, not the dependency closure.

## Sessions interval note

The `sessions.interval` option currently documents the spec default (`1h`),
but the systemd timer in `nix/aggregator.nix` uses `OnCalendar = "*:0/30"`
per the plan's step-2 comment. To honour a strict `1h` value, override the
timer in your home-manager config or extend the module to derive
`OnCalendar` from the option.
