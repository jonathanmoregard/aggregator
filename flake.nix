{
  description = "Personal aggregator: sessions + GitHub cache, FastMCP + CLI surfaces";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.flake-utils.url = "github:numtide/flake-utils";
  # Eval-only dependency, used by `checks` to instantiate the home-manager
  # module and render its real unit files so they can be asserted on. Nothing
  # in `packages` or `devShells` depends on it.
  inputs.home-manager.url = "github:nix-community/home-manager";
  inputs.home-manager.inputs.nixpkgs.follows = "nixpkgs";
  outputs = { self, nixpkgs, flake-utils, home-manager }:
    let
      systemOutputs = flake-utils.lib.eachDefaultSystem (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          python = pkgs.python311;
          aggregatorPkg = python.pkgs.buildPythonApplication {
            pname = "aggregator";
            version = "0.0.1";
            src = ./.;
            format = "pyproject";
            nativeBuildInputs = [ python.pkgs.hatchling ];
            propagatedBuildInputs = with python.pkgs; [
              # NOTE: fastmcp / presidio / claude-runner may need overlays or
              # pip install in the devShell if not in nixpkgs. See nix/README.md.
            ];
            doCheck = false;
          };

          # Stand-in for `services.aggregator.package` in the check fixture.
          #
          # NOT `aggregatorPkg`: that derivation is a deliberate stub
          # (`propagatedBuildInputs` is empty, see nix/README.md) and its
          # build fails at `pythonRuntimeDepsCheckHook` because fastmcp,
          # presidio, sentence-transformers and sqlite-vec are not packaged
          # here. Depending on it would make this check fail for a reason
          # that has nothing to do with what it asserts. The check is about
          # the SHAPE of the generated units — store-path ExecStart, no home
          # path, trust store, OnFailure, stagger — and a store-path stub is
          # faithful for all of it.
          fixturePackage = pkgs.runCommand "aggregator-unit-fixture" { } ''
            mkdir -p "$out/bin"
            for b in aggregator aggregator-mcp; do
              printf '#!/bin/sh\nexit 0\n' > "$out/bin/$b"
              chmod +x "$out/bin/$b"
            done
          '';

          # A throwaway home-manager evaluation of ./nix/aggregator.nix, used
          # only to render the unit files the module generates.
          #
          # `homeDirectory` is deliberately NOT under /home. The whole point of
          # the check below is that a `/home/` string anywhere in the generated
          # units is a deployment bug, so the fixture must not manufacture one.
          hmFixture = home-manager.lib.homeManagerConfiguration {
            inherit pkgs;
            modules = [
              ./nix/aggregator.nix
              {
                home.username = "aggregator-check";
                home.homeDirectory = "/nonexistent/aggregator-check";
                home.stateVersion = "24.11";
                services.aggregator = {
                  enable = true;
                  package = fixturePackage;
                };
              }
            ];
          };
        in {
          devShells.default = pkgs.mkShell {
            packages = [
              python
              pkgs.uv
              pkgs.ruff
              pkgs.sqlite
              pkgs.gitleaks
              pkgs.gh
            ];
          };
          packages.default = aggregatorPkg;

          checks.aggregator-embed-unit-hygiene = pkgs.runCommand
            "aggregator-embed-unit-hygiene"
            { nativeBuildInputs = [ pkgs.gnugrep pkgs.gnused ]; }
            ''
              set -euo pipefail
              units=${hmFixture.config.home-files}/.config/systemd/user
              fail() { echo "FAIL: $*" >&2; exit 1; }

              for u in aggregator-embed.service aggregator-embed.timer \
                       aggregator-embed-seed.service \
                       aggregator-embed-failure-notify.service; do
                [ -e "$units/$u" ] || fail "$u was not generated"
              done

              svc="$units/aggregator-embed.service"

              # ---- 1. The 2026-08-16 constraint --------------------------
              # A unit must execute a deployed artifact from the Nix store,
              # pinned to a committed revision, never a developer checkout.
              #
              # The bug this encodes was INVISIBLE at the unit-file level:
              # aggregator-ingest.service had a store-path ExecStart whose
              # wrapper script ended in `exec uv run --directory <checkout>`.
              # So the check follows every ExecStart into the script it names
              # and greps that too. Grepping only the unit would have passed
              # on the broken deployment.
              for u in aggregator-embed.service aggregator-embed-seed.service \
                       aggregator-embed-failure-notify.service \
                       aggregator-embed.timer; do
                f="$units/$u"
                # Dereference: home-manager installs units as symlinks.
                real=$(readlink -f "$f")
                if grep -q '/home/' "$real"; then
                  grep -n '/home/' "$real" >&2
                  fail "$u references a home directory"
                fi
                for script in $(sed -n 's/^ExecStart=\([^ ]*\).*/\1/p' "$real"); do
                  case "$script" in
                    /nix/store/*) ;;
                    *) fail "$u: ExecStart is not a store path: $script" ;;
                  esac
                  if grep -q '/home/' "$script"; then
                    grep -n '/home/' "$script" >&2
                    fail "$u: ExecStart script $script references a home directory"
                  fi
                  if grep -qE 'uv run|--directory' "$script"; then
                    fail "$u: ExecStart script $script shells out via uv run — that runs a working tree, not a pinned artifact"
                  fi
                done
              done

              # ---- 2. Trust store ----------------------------------------
              # A missing CA bundle already cost this project a day on the
              # TickTick source. Both spellings: NIX_SSL_CERT_FILE is what
              # Nix-built OpenSSL consults, SSL_CERT_FILE what Python's ssl
              # and certifi-free requests stacks consult.
              grep -qE '^Environment="?SSL_CERT_FILE=' "$svc" \
                || fail "aggregator-embed.service does not set SSL_CERT_FILE"
              grep -qE '^Environment="?NIX_SSL_CERT_FILE=' "$svc" \
                || fail "aggregator-embed.service does not set NIX_SSL_CERT_FILE"

              # ---- 3. Fail loudly ----------------------------------------
              grep -q '^OnFailure=aggregator-embed-failure-notify.service$' "$svc" \
                || fail "aggregator-embed.service has no OnFailure notification"
              notify_unit=$(readlink -f "$units/aggregator-embed-failure-notify.service")
              notify_script=$(sed -n 's/^ExecStart=\([^ ]*\).*/\1/p' "$notify_unit")
              grep -q 'notify-send' "$notify_script" \
                || fail "the failure-notify unit does not call notify-send"

              # ---- 4. Weights are never fetched by the unattended unit ----
              grep -q 'HF_HUB_OFFLINE=1' "$svc" \
                || fail "aggregator-embed.service is not pinned offline"

              # ---- 5. Staggered against the ingest timers -----------------
              embed_cal=$(sed -n 's/^OnCalendar=//p' "$units/aggregator-embed.timer")
              [ -n "$embed_cal" ] || fail "embed timer has no OnCalendar"
              for t in aggregator-sessions.timer aggregator-github.timer; do
                other=$(sed -n 's/^OnCalendar=//p' "$units/$t")
                if [ "$embed_cal" = "$other" ]; then
                  fail "embed timer shares OnCalendar ($embed_cal) with $t"
                fi
              done

              # ---- 6. Backfill-sized timeout, safe to kill ---------------
              grep -q '^TimeoutStartSec=' "$svc" \
                || fail "aggregator-embed.service has no TimeoutStartSec"

              # ---- 7. The seeding unit is human-triggered only ------------
              if grep -q '^\[Install\]' "$units/aggregator-embed-seed.service"; then
                fail "aggregator-embed-seed.service must not be wanted by any target"
              fi

              echo "aggregator-embed unit hygiene: OK"

              # Keep the rendered units as the check's output, so a human can
              # read exactly what was asserted on without re-deriving it.
              mkdir -p "$out"
              for u in aggregator-embed.service aggregator-embed.timer \
                       aggregator-embed-seed.service \
                       aggregator-embed-failure-notify.service; do
                cp -L "$units/$u" "$out/$u"
              done
            '';
        });
    in
      systemOutputs // {
        homeManagerModules.default = import ./nix/aggregator.nix;
      };
}
