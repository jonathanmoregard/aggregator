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
          # ---- eval-time probe for the timeoutStartSec option type --------
          #
          # Round-2 LOW: the option was `types.str`, so `"infinty"` evaluated,
          # built and deployed, and only systemd rejected it — falling back to
          # its ~90s default, which truncates every tick of a 25-30 day
          # backfill while the journal parse error goes unread.
          #
          # This asserts the module's type agrees with systemd's own parser.
          # The expected column is not guesswork: every string below was run
          # through `systemd-analyze timespan` on systemd 261 and the module
          # must reproduce that verdict exactly.
          timeoutStartSecAccepts = value:
            (builtins.tryEval (builtins.deepSeq
              (home-manager.lib.homeManagerConfiguration {
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
                      embed.timeoutStartSec = value;
                    };
                  }
                ];
              }).config.systemd.user.services.aggregator-embed.Service.TimeoutStartSec
              true)).success;

          # `systemd-analyze timespan "<x>"` exits 0 on these.
          validTimeSpans = [ "infinity" "8h" "90" "1h 30min" "500ms" "1d" "1M" "0" ];
          # …and non-zero on these ("Failed to parse time span").
          invalidTimeSpans = [ "infinty" "8hh" "eight hours" "" "h" "8 hourz" ];

          wronglyRejected = builtins.filter (v: !(timeoutStartSecAccepts v)) validTimeSpans;
          wronglyAccepted = builtins.filter timeoutStartSecAccepts invalidTimeSpans;
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

              # ---- 3b. The debounce must fail OPEN on an undelivered popup -
              # Round-2 LOW. The popup is debounced to once per 24h via a
              # stamp file, and the stamp used to be touched BEFORE
              # notify-send ran. So a send that failed — no notification
              # daemon on the session bus, which is the normal state of a
              # freshly-booted or headless session — still bought a full day
              # of silence, and the user was never told the vector index had
              # stopped filling. A debounce records "the human was told"; a
              # failed send is exactly the case where they were not.
              #
              # This is executed, not pattern-matched: the real generated
              # script is run twice with notify-send swapped for a stub, once
              # failing and once succeeding, so the assertion tests the
              # behaviour rather than the shape of the source. Line-order
              # greps would pass on any restructure that moved the touch out
              # of the success branch.
              work="$TMPDIR/notify-debounce"
              notify_bin=$(grep -oE '/nix/store/[^ ]*/bin/notify-send' \
                             "$notify_script" | head -1)
              [ -n "$notify_bin" ] \
                || fail "could not locate the notify-send binary in $notify_script"

              run_notify() {
                # $1 = exit status the notify-send stub should return.
                rm -rf "$work"
                mkdir -p "$work/bin" "$work/state" "$work/home"
                printf '#!/bin/sh\nexit %s\n' "$1" > "$work/bin/notify-send"
                chmod +x "$work/bin/notify-send"
                sed "s|$notify_bin|$work/bin/notify-send|" "$notify_script" \
                  > "$work/notify.sh"
                chmod +x "$work/notify.sh"
                # Two ticks inside the same 24h window.
                HOME="$work/home" XDG_STATE_HOME="$work/state" "$work/notify.sh" \
                  > "$work/1.log" 2>&1 || true
                HOME="$work/home" XDG_STATE_HOME="$work/state" "$work/notify.sh" \
                  > "$work/2.log" 2>&1 || true
              }

              run_notify 1
              if grep -q 'suppressed' "$work/2.log"; then
                echo "--- tick 1 ---" >&2; cat "$work/1.log" >&2
                echo "--- tick 2 ---" >&2; cat "$work/2.log" >&2
                fail "the failure-notify debounce fails CLOSED: notify-send exited non-zero on tick 1, yet tick 2 was suppressed. An undelivered popup must never buy 24h of silence — arm the stamp only after a successful send"
              fi

              # The mirror assertion, so "fail open" cannot be satisfied by
              # deleting the debounce outright: a DELIVERED popup must arm it,
              # or the 30-minute timer raises 48 CRITICAL popups a day and
              # trains the user to ignore all of them.
              run_notify 0
              grep -q 'suppressed' "$work/2.log" \
                || { cat "$work/2.log" >&2; \
                     fail "the failure-notify popup is not debounced: a delivered notification must arm the 24h stamp, otherwise the embed timer raises 48 CRITICAL popups a day"; }
              rm -rf "$work"

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

              # ---- 6. The start timeout must not fire on a healthy run ----
              # Task M measured the real corpus: 483,193 observations /
              # 422,261 chunks / 609M chars at 249.6 chars per wall-second,
              # CPU-only. A full backfill is ~25-30 days of continuous work.
              # Any finite TimeoutStartSec therefore SIGTERMs a *correctly
              # progressing* run, and each kill is a systemd failure that
              # fires OnFailure= — a CRITICAL popup saying the vector index
              # is not being filled while it is being filled. The previous
              # `8h` would have produced ~85 such kills over one backfill.
              #
              # A wall clock cannot separate a wedged worker from a working
              # one when the honest working time is a month, so the start
              # timeout was never the wedge guard and is disabled outright.
              # What bounds a wedge instead: Nice/idle-IO cap the blast
              # radius, the per-batch checkpoint caps the loss, the flock
              # stops workers piling up, and progress (aggregator status /
              # vector_index) is what a human actually reads to spot one.
              start_timeout=$(sed -n 's/^TimeoutStartSec=//p' "$svc")
              [ -n "$start_timeout" ] \
                || fail "aggregator-embed.service has no TimeoutStartSec"
              [ "$start_timeout" = "infinity" ] \
                || fail "aggregator-embed.service sets TimeoutStartSec=$start_timeout — a finite start timeout kills a healthy multi-week backfill and reports it as a failure"

              # With no start timeout, the only remaining bound on a wedged
              # worker is a human running `systemctl --user stop`. That path
              # must itself complete, so the STOP timeout stays finite.
              stop_timeout=$(sed -n 's/^TimeoutStopSec=//p' "$svc")
              [ -n "$stop_timeout" ] \
                || fail "aggregator-embed.service has no TimeoutStopSec"
              [ "$stop_timeout" != "infinity" ] \
                || fail "aggregator-embed.service sets TimeoutStopSec=infinity — the manual stop is the last bound on a wedged worker and must not hang"

              # ---- 6b. A mistyped start timeout must fail at EVAL time ----
              # Assertion 6 above only sees the fixture's default. A real
              # deployment sets this option in its own home-manager config,
              # where a typo never reaches this check — it reaches systemd,
              # which rejects the span and applies its ~90s default, cutting
              # off every tick of a month-long backfill. So the option type
              # itself has to be the gate, and this asserts the type agrees
              # with `systemd-analyze timespan` on systemd 261 case for case.
              wrongly_rejected=${pkgs.lib.escapeShellArg (builtins.toJSON wronglyRejected)}
              wrongly_accepted=${pkgs.lib.escapeShellArg (builtins.toJSON wronglyAccepted)}
              [ "$wrongly_rejected" = "[]" ] \
                || fail "services.aggregator.embed.timeoutStartSec rejects time spans systemd accepts: $wrongly_rejected — the option type is too tight and blocks a legitimate config"
              [ "$wrongly_accepted" = "[]" ] \
                || fail "services.aggregator.embed.timeoutStartSec accepts time spans systemd rejects: $wrongly_accepted — systemd would fall back to its ~90s default and truncate every tick of a 25-30 day backfill"

              # ---- 7. The seeding unit is human-triggered only ------------
              if grep -q '^\[Install\]' "$units/aggregator-embed-seed.service"; then
                fail "aggregator-embed-seed.service must not be wanted by any target"
              fi

              # ---- 7b. Seeding covers EVERY model the product loads -------
              # Round-2 MEDIUM. The seed unit used to run
              # `embed --once --source observations --batch-size 1`, which
              # constructs only the Embedder. `Reranker()` is built in exactly
              # one place — the MCP server — with
              # `local_files_only=not downloads_allowed()`, and that server is
              # registered bare so downloads are never allowed there. Net
              # effect: nothing anywhere fetched the reranker weights, every
              # `rerank=True` raised inside the constructor, and
              # `_maybe_rerank` swallowed it and returned the page unranked
              # with no notice. The feature was dead on arrival and silent
              # about it.
              #
              # A seeding step that covers one of the two models is a claim
              # about one of the two models, so this asserts both are named
              # and that the entry point is the dedicated one.
              seed_script=$(sed -n 's/^ExecStart=\([^ ]*\).*/\1/p' \
                              "$units/aggregator-embed-seed.service")
              grep -q 'embed --seed-models' "$seed_script" \
                || fail "aggregator-embed-seed.service does not run 'aggregator embed --seed-models' — that is the only entry point that constructs both the Embedder and the Reranker"
              for repo in 'Qwen/Qwen3-Embedding-0.6B' 'Qwen/Qwen3-Reranker-0.6B'; do
                grep -qF "$repo" "$seed_script" \
                  || fail "the seed unit never mentions $repo — a model the product loads at runtime has no fetch path, so it degrades forever the first time it is asked for"
              done

              # The seeder is a DOWNLOAD, not a workload. It used to embed a
              # real corpus row to warm the cache, which ran untrusted text
              # through torch and advanced embedding_state as a side effect of
              # a download. Nothing about fetching weights needs the database.
              if grep -qE 'embed .*(--once|--catchup)' "$seed_script"; then
                grep -nE 'embed .*(--once|--catchup)' "$seed_script" >&2
                fail "the seed unit embeds real corpus rows — a weight download must not touch the database, contend with an ingest run, or feed untrusted text to torch as a side effect"
              fi

              # The opt-in the Python loaders gate downloads on. Without it
              # this unit fetches nothing and the whole seeding story is a
              # no-op that exits 0.
              grep -q 'AGGREGATOR_ALLOW_MODEL_DOWNLOAD=1' "$seed_script" \
                || fail "the seed unit does not export AGGREGATOR_ALLOW_MODEL_DOWNLOAD=1 — the loaders pass local_files_only=True without it, so it would download nothing"

              # ...and it is the ONLY rendered artifact that enables it. The
              # opt-in is what makes a 2.4 GB download consented to, and that
              # only means anything if the one unit carrying it is the one a
              # human starts by hand. tests/core/test_model_offline_default.py
              # asserts this over the Nix source; this asserts it over what
              # that source actually renders to, including every Environment=
              # line and every ExecStart script, so the two cannot drift.
              for u in aggregator-embed.service aggregator-embed.timer \
                       aggregator-embed-failure-notify.service; do
                f=$(readlink -f "$units/$u")
                if grep -q 'AGGREGATOR_ALLOW_MODEL_DOWNLOAD' "$f"; then
                  fail "$u sets AGGREGATOR_ALLOW_MODEL_DOWNLOAD — only the human-triggered seed unit may enable model downloads"
                fi
                for s in $(sed -n 's/^ExecStart=\([^ ]*\).*/\1/p' "$f"); do
                  if grep -q 'AGGREGATOR_ALLOW_MODEL_DOWNLOAD' "$s"; then
                    grep -n 'AGGREGATOR_ALLOW_MODEL_DOWNLOAD' "$s" >&2
                    fail "$u: ExecStart script $s enables AGGREGATOR_ALLOW_MODEL_DOWNLOAD — an unattended unit must never be able to start a GB-scale download"
                  fi
                done
              done

              # ---- 8. Sandbox the unit that eats attacker-influenced text -
              # This unit feeds the corpus — web pages, PDFs, chat exports,
              # GitHub bodies, none of it authored by the user — through
              # torch and a native tokenizer, i.e. a large C++ attack surface
              # processing untrusted bytes. Until now it ran with the user's
              # full ambient authority and its "offline" was a pair of
              # environment variables, which is a request rather than a
              # boundary: anything that spawns a subprocess, or any library
              # that ignores them, had the whole network.
              #
              # RestrictAddressFamilies is the load-bearing line. It is
              # enforced by seccomp, works in a USER manager, and makes an
              # AF_INET socket() fail outright — so "this unit does not talk
              # to the network" stops depending on every library agreeing to
              # read HF_HUB_OFFLINE. AF_UNIX stays for journal/dbus,
              # AF_NETLINK for the glibc resolver paths that probe interfaces
              # on startup even when nothing connects.
              # The directives BOTH torch units carry. Round-2 LOW: the seeder
              # had none of them at all — `systemd-analyze security
              # --offline=true --user` rated the rendered files 6.3 MEDIUM for
              # the worker and 9.2 UNSAFE for the seeder — even though the
              # seeder is the one that reaches the public internet and then
              # loads ~2.4 GB of third-party weights into torch. Being
              # human-triggered bounds how often that happens, not what it can
              # do when it does. They share one Nix binding now, and this
              # loop asserts the sharing survived.
              seed_svc="$units/aggregator-embed-seed.service"
              for directive in \
                'NoNewPrivileges=true' \
                'PrivateTmp=true' \
                'RestrictNamespaces=true' \
                'RestrictRealtime=true' \
                'RestrictSUIDSGID=true' \
                'LockPersonality=true' \
                'SystemCallArchitectures=native' \
                'ProtectSystem=full' \
                'ProtectKernelTunables=true' \
                'ProtectControlGroups=true'; do
                grep -qxF "$directive" "$svc" \
                  || fail "aggregator-embed.service is missing '$directive'"
                grep -qxF "$directive" "$seed_svc" \
                  || fail "aggregator-embed-seed.service is missing '$directive' — it downloads 2.4 GB off the internet and loads it into torch, and it must not be less sandboxed than the offline worker on anything except the network"
              done

              # The worker's two network directives, which the seeder is the
              # one unit allowed to relax.
              for directive in \
                'RestrictAddressFamilies=AF_UNIX AF_NETLINK' \
                'IPAddressDeny=any'; do
                grep -qxF "$directive" "$svc" \
                  || fail "aggregator-embed.service is missing '$directive'"
              done

              # The seeder MUST still restrict address families — dropping the
              # directive entirely would re-admit AF_PACKET, AF_BLUETOOTH,
              # AF_VSOCK and the rest of the exotic families, which is most of
              # the socket-family kernel attack surface and none of what a
              # downloader needs. It must simply also permit IP.
              seed_raf=$(sed -n 's/^RestrictAddressFamilies=//p' "$seed_svc")
              [ -n "$seed_raf" ] \
                || fail "aggregator-embed-seed.service sets no RestrictAddressFamilies — a downloader needs TCP over IP, not AF_PACKET and AF_BLUETOOTH"
              for fam in AF_INET AF_INET6; do
                case " $seed_raf " in
                  *" $fam "*) ;;
                  *) fail "aggregator-embed-seed.service does not permit $fam (RestrictAddressFamilies=$seed_raf) — this is the documented download path and it could not open a socket" ;;
                esac
              done
              if grep -q '^IPAddressDeny=any$' "$seed_svc"; then
                fail "aggregator-embed-seed.service sets IPAddressDeny=any — that blocks the download this unit exists to perform"
              fi

              # ---- 8b. Directives that would BREAK these units -------------
              # Round-2 advisory. The sandbox above has never executed: this
              # host's aggregator-env has no torch, so nothing has ever run
              # under it (see nix/README.md, "what remains unproven"). The
              # standing risk is therefore not that a directive gets removed —
              # step 8 catches that — but that a future hardening pass ADDS
              # one that looks like an improvement and silently makes the unit
              # unstartable, on a branch where nobody can start it to find out.
              #
              # Each absence below is justified from something this module
              # itself sets, except the first, which is justified from torch's
              # documented behaviour and is NOT empirically verified here.
              for u in aggregator-embed.service aggregator-embed-seed.service; do
                f="$units/$u"

                # torch's JIT and the OpenMP runtime allocate W|X pages, so
                # this makes `import torch` die. It is the single most likely
                # directive for a well-meaning hardening pass to reach for.
                # NOT verified by execution on this host — asserted from
                # torch's documented behaviour, and recorded as such.
                if grep -q '^MemoryDenyWriteExecute=' "$f"; then
                  fail "$u sets MemoryDenyWriteExecute — torch's JIT and the OpenMP runtime allocate W|X pages, so import torch dies and this unit can never start"
                fi

                # HF_HOME is %C/huggingface, which for a user manager is
                # $XDG_CACHE_HOME under $HOME. Any ProtectHome= makes the
                # weights cache unreachable — unwritable for the seeder,
                # unreadable for the worker.
                if grep -q '^ProtectHome=' "$f"; then
                  fail "$u sets ProtectHome — HF_HOME resolves under \$HOME, so the weights cache becomes unreachable"
                fi
              done

              # NO assertion here for `ProtectSystem=strict`, which would also
              # break these units ($HOME read-only, and the HF cache lives
              # there). One was written and then removed, because it could not
              # be made to go red: home-manager renders exactly one value per
              # key, so setting `strict` REPLACES `ProtectSystem=full` rather
              # than shadowing it, and step 8's "missing ProtectSystem=full"
              # fires first. Watched: injecting `ProtectSystem = "strict"`
              # into the seeder's override block failed with
              #   "aggregator-embed-seed.service is missing 'ProtectSystem=full'"
              # and the dedicated assertion never ran. Shipping it anyway would
              # have been dead code that reads like coverage.

              # The seeder exists to download. PrivateNetwork is fine on the
              # offline worker and fatal here.
              if grep -q '^PrivateNetwork=' "$seed_svc"; then
                fail "aggregator-embed-seed.service sets PrivateNetwork — it exists to fetch 2.4 GB of weights over the internet"
              fi

              # ---- 9. The score behind step 8, and why it is not asserted -
              # `systemd-analyze security --offline=true --user <unit>` rates
              # the rendered file without loading it, and
              # `systemd-analyze verify --user <unit>` catches directives
              # systemd would silently ignore (a misspelled `ProtectSytem=`
              # prints "Unknown key ... ignoring" and is otherwise invisible —
              # the same failure shape as the timeoutStartSec typo in 6b).
              # Both are exactly the tools this check wants. NEITHER can run
              # in the Nix build sandbox. Observed on systemd 261, for both
              # subcommands:
              #
              #     Failed to lookup RuntimeDirectory path: No such device or address
              #     Failed to initialize manager: No such device or address
              #
              # and — worse — both still exit 0 after printing it, so a naive
              # gate would read as passing coverage while asserting nothing.
              # That was tried and deliberately not shipped.
              #
              # So these are recorded measurements plus a manual step, not
              # gates. Taken on this host against the rendered units:
              #
              #     aggregator-embed.service        9.4 UNSAFE -> 6.3 MEDIUM
              #     aggregator-embed-seed.service   9.2 UNSAFE -> 6.8 MEDIUM
              #
              # Reproduce with:
              #
              #     out=$(nix build --no-link --print-out-paths \
              #       .#checks.x86_64-linux.aggregator-embed-unit-hygiene)
              #     systemd-analyze security --offline=true --user \
              #       "$out/aggregator-embed-seed.service"
              #     systemd-analyze verify --user "$out"/*.service
              #
              # Steps 8 and 8b are what hold the line in CI: 8 names every
              # directive that must be present, 8b every one that must not be,
              # so drift in either direction fails the build even though the
              # number itself cannot be checked here.

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
