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
          # presidio and sentence-transformers are not packaged here.
          # Depending on it would make this check fail for a reason that has
          # nothing to do with what it asserts. The check is about the SHAPE
          # of the generated units — store-path ExecStart, no home path,
          # OnFailure, sandbox — and a store-path stub is faithful for all of
          # it.
          # The names mirror `[project.scripts]` in pyproject.toml. A unit that
          # execs `${cfg.package}/bin/<name>` can only be asserted on if the
          # fixture carries that name, so a console script missing here would
          # make the hygiene check pass by never reaching the assertion.
          fixturePackage = pkgs.runCommand "aggregator-unit-fixture" { } ''
            mkdir -p "$out/bin"
            for b in aggregator aggregator-mcp aggregator-schema-probe; do
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

          checks.aggregator-unit-hygiene = pkgs.runCommand
            "aggregator-unit-hygiene"
            { nativeBuildInputs = [ pkgs.gnugrep pkgs.gnused ]; }
            ''
              set -euo pipefail
              units=${hmFixture.config.home-files}/.config/systemd/user
              fail() { echo "FAIL: $*" >&2; exit 1; }

              all_units="aggregator-tag.service aggregator-tag.timer \
                         aggregator-tag-failure-notify.service"

              for u in $all_units; do
                [ -e "$units/$u" ] || fail "$u was not generated"
              done

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
              for u in $all_units; do
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
                  # No unattended unit may enable model downloads: the opt-in
                  # is what makes a GB-scale download consented to, and only
                  # means anything if a human types it.
                  if grep -q 'AGGREGATOR_ALLOW_MODEL_DOWNLOAD' "$script"; then
                    fail "$u: ExecStart script $script enables AGGREGATOR_ALLOW_MODEL_DOWNLOAD"
                  fi
                done
                if grep -q 'AGGREGATOR_ALLOW_MODEL_DOWNLOAD' "$real"; then
                  fail "$u sets AGGREGATOR_ALLOW_MODEL_DOWNLOAD"
                fi
              done

              # ---- 2. Fail loudly, about the unit that actually failed ----
              notify_script_for() {
                # $1 = notify unit name. Echoes its ExecStart script path.
                sed -n 's/^ExecStart=\([^ ]*\).*/\1/p' \
                  "$(readlink -f "$units/$1")"
              }

              tag_svc="$units/aggregator-tag.service"
              notifier=aggregator-tag-failure-notify.service
              grep -q "^OnFailure=$notifier$" "$tag_svc" \
                || fail "aggregator-tag.service does not name $notifier as its OnFailure target"
              notify_script=$(notify_script_for "$notifier")
              grep -q 'notify-send' "$notify_script" \
                || fail "$notifier does not call notify-send"
              grep -qF "journalctl --user -u aggregator-tag.service" "$notify_script" \
                || fail "$notifier never points at 'journalctl --user -u aggregator-tag.service' — the operator is sent to the wrong unit's journal"

              # ---- 3. The debounce must fail OPEN on an undelivered popup --
              # The popup is debounced to once per 24h via a stamp file. A
              # debounce records "the human was told"; a failed send is exactly
              # the case where they were not, so it must not arm the stamp.
              #
              # Executed, not pattern-matched: the real generated script is run
              # twice with notify-send swapped for a stub, once failing and once
              # succeeding, so the assertion tests the behaviour rather than the
              # shape of the source.
              work="$TMPDIR/notify-debounce"

              stub_notify() {
                # $1 = notify script, $2 = destination, $3 = stub exit status.
                local bin
                bin=$(grep -oE '/nix/store/[^ ]*/bin/notify-send' "$1" | head -1)
                [ -n "$bin" ] \
                  || fail "could not locate the notify-send binary in $1"
                printf '#!/bin/sh\nexit %s\n' "$3" > "$work/bin/notify-send"
                chmod +x "$work/bin/notify-send"
                sed "s|$bin|$work/bin/notify-send|" "$1" > "$2"
                chmod +x "$2"
              }

              tick() {
                # $1 = prepared script, $2 = log. Shares $work/state, so the
                # debounce sees the same stamp directory a real session would.
                HOME="$work/home" XDG_STATE_HOME="$work/state" "$1" \
                  > "$2" 2>&1 || true
              }

              run_notify() {
                # $1 = notify script, $2 = exit status for the stub.
                rm -rf "$work"
                mkdir -p "$work/bin" "$work/state" "$work/home"
                stub_notify "$1" "$work/notify.sh" "$2"
                # Two ticks inside the same 24h window.
                tick "$work/notify.sh" "$work/1.log"
                tick "$work/notify.sh" "$work/2.log"
              }

              run_notify "$notify_script" 1
              if grep -q 'suppressed' "$work/2.log"; then
                echo "--- tick 1 ---" >&2; cat "$work/1.log" >&2
                echo "--- tick 2 ---" >&2; cat "$work/2.log" >&2
                fail "$notifier's debounce fails CLOSED: notify-send exited non-zero on tick 1, yet tick 2 was suppressed. An undelivered popup must never buy 24h of silence — arm the stamp only after a successful send"
              fi

              # The mirror assertion, so "fail open" cannot be satisfied by
              # deleting the debounce outright: a DELIVERED popup must arm it.
              run_notify "$notify_script" 0
              grep -q 'suppressed' "$work/2.log" \
                || { cat "$work/2.log" >&2; \
                     fail "$notifier's popup is not debounced: a delivered notification must arm the 24h stamp"; }
              rm -rf "$work"

              # ---- 4. The tag unit's sandbox -------------------------------
              # `aggregator tag` shells out to `claude -p` with
              # attacker-influenced record bodies in the prompt, so the unit
              # carries a sandbox to the extent the claude CLI tolerates. Every
              # directive below was verified on the host (2026-09-04):
              # `claude -p --model haiku` under exactly this set via
              # `systemd-run --user` answered and exited 0.
              for directive in \
                'NoNewPrivileges=true' \
                'PrivateTmp=true' \
                'RestrictRealtime=true' \
                'RestrictSUIDSGID=true' \
                'LockPersonality=true' \
                'ProtectSystem=full' \
                'ProtectKernelTunables=true' \
                'ProtectControlGroups=true'; do
                grep -qxF "$directive" "$tag_svc" \
                  || fail "aggregator-tag.service is missing '$directive' — it feeds attacker-influenced record bodies to a subprocess and must not run with the user's full ambient authority"
              done

              # ...and the directives that would BREAK it.
              if grep -q '^ProtectHome=' "$tag_svc"; then
                fail "aggregator-tag.service sets ProtectHome — claude -p must read ~/.claude config+credentials and write its own state under \$HOME"
              fi
              if grep -q '^PrivateNetwork=' "$tag_svc"; then
                fail "aggregator-tag.service sets PrivateNetwork — the claude CLI exists to reach its API; this unit is network-permitted like aggregator-github"
              fi
              if grep -q '^IPAddressDeny=any$' "$tag_svc"; then
                fail "aggregator-tag.service sets IPAddressDeny=any — that blocks the API call this unit exists to make"
              fi

              echo "aggregator unit hygiene: OK"

              # Keep the rendered units as the check's output, so a human can
              # read exactly what was asserted on without re-deriving it.
              mkdir -p "$out"
              for u in $all_units; do
                cp -L "$units/$u" "$out/$u"
              done
            '';
        });
    in
      systemOutputs // {
        homeManagerModules.default = import ./nix/aggregator.nix;
      };
}
