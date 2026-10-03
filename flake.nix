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

          # Stand-in for `services.aggregator.embed.server.package`. Not
          # `pkgs.llama-cpp-vulkan`: the check is about the unit and its
          # launcher, and pulling a GPU build of llama.cpp into the check's
          # closure would make it slow for nothing it asserts. The stub prints
          # its argv, so step 10 can EXECUTE the real launcher and read back
          # the exact command line the server would be started with.
          fixtureLlama = pkgs.runCommand "llama-server-unit-fixture" { } ''
            mkdir -p "$out/bin"
            printf '#!/bin/sh\nprintf "%%s\\n" "$@"\n' > "$out/bin/llama-server"
            chmod +x "$out/bin/llama-server"
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
                  embed.server.package = fixtureLlama;
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
                      embed.server.package = fixtureLlama;
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

          # ---- the repo ids that ACTUALLY determine what gets fetched ------
          #
          # Round-3 finding: step 7b used to grep the seed script for two
          # hardcoded repo ids, and those strings occur in that script only
          # inside an informational `echo`. The real fetch is
          # `aggregator embed --seed-models`, which constructs `Embedder()` and
          # `Reranker()` and resolves whatever the PYTHON defaults say. So the
          # assertion could not fail for the thing it appeared to protect:
          # changing `_DEFAULT_MODEL_ST` left the check green — verified, and
          # `nix build` even returned a byte-identical store path, because the
          # derivation did not depend on that file at all. Fake coverage, which
          # is worse than none: it reads as protection.
          #
          # Reading the ids OUT of the Python source fixes both halves. The
          # value asserted on is now the one that decides the download, and the
          # check's inputs include the file that decides it, so editing that
          # file re-runs this.
          #
          # Tolerates a type annotation (`NAME: str | None = "..."`), which is
          # how the GGUF revision pin is declared; the value is the last group.
          pythonDefaultModel = { file, const }:
            let
              lines = pkgs.lib.splitString "\n" (builtins.readFile file);
              hits = builtins.filter (m: m != null) (
                map (l: builtins.match "${const}(:[^=]*)? = \"([^\"]+)\".*" l) lines
              );
            in
              if builtins.length hits == 1
              then builtins.elemAt (builtins.head hits) 1
              else throw (
                "flake check: expected exactly one `${const} = \"...\"` in "
                + "${toString file}, found ${toString (builtins.length hits)}. "
                + "This is how the check knows which model is really fetched; "
                + "if the declaration moved, teach it the new shape rather "
                + "than deleting the assertion."
              );

          # huggingface_hub's on-disk mangling: `Org/Name` -> `models--Org--Name`.
          # This is what `have_model` globs for, so a repo id that changes
          # without its directory changing makes the worker's preflight test
          # for weights nobody will ever write there.
          hfCacheDirOf = repo: "models--" + builtins.replaceStrings ["/"] ["--"] repo;

          # THE SERVER BACKEND'S MODEL, because it is the source default
          # (`DEFAULT_BACKEND = "server"`): the repo, the pinned revision and
          # the one file, which together are the path the server unit opens.
          pyEmbedModel = pythonDefaultModel {
            file = ./aggregator/core/embed.py;
            const = "_DEFAULT_MODEL_GGUF";
          };
          pyEmbedRevision = pythonDefaultModel {
            file = ./aggregator/core/embed.py;
            const = "QWEN3_EMBEDDING_GGUF_REVISION";
          };
          pyEmbedFile = pythonDefaultModel {
            file = ./aggregator/core/embed.py;
            const = "QWEN3_EMBEDDING_GGUF_FILENAME";
          };
          pyEmbedBackend = pythonDefaultModel {
            file = ./aggregator/core/embed.py;
            const = "DEFAULT_BACKEND";
          };
          # The socket (relative to $XDG_RUNTIME_DIR) every process that does
          # not export AGGREGATOR_EMBED_URL dials — the MCP server is
          # registered bare, so this IS where its queries go. The unit must
          # bind exactly this.
          pyEmbedSocket = pythonDefaultModel {
            file = ./aggregator/core/embed.py;
            const = "EMBED_SOCKET_NAME";
          };
          pyRerankModel = pythonDefaultModel {
            file = ./aggregator/core/rerank.py;
            const = "_DEFAULT_MODEL";
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

            # Isolate the dev shell from the PRODUCTION cache.
            #
            # `Store` resolves its database as
            # `$XDG_DATA_HOME/aggregator/cache.db` (aggregator/core/store.py
            # :1297-1300), falling back to `~/.local/share` when the variable
            # is unset — and `migrate()` runs on EVERY subcommand, stamping
            # `PRAGMA user_version = SCHEMA_VERSION` unconditionally
            # (store.py, ~:1532). So before this hook existed, a bare
            # `uv run aggregator …` typed inside `nix develop` read and wrote
            # the live cache the systemd ingest timer owns, and re-stamped its
            # schema version at whatever the checkout happened to be. A
            # developer poking at a branch could therefore migrate production
            # state forward — or, running an older branch, tell the cache it
            # was older than it is. That is the write-side half of the
            # reader/writer schema-skew class defect (2026-08-30): the shared
            # mutable file is the coupling, and the fix is to stop sharing it.
            #
            # Isolation is the RESTING STATE, not an opt-in. There is
            # deliberately no enable flag: a guard that only works when someone
            # remembers to set a variable is not a guard. Touching the real
            # cache from a shell now requires saying so out loud, e.g.
            # `XDG_DATA_HOME="$HOME/.local/share" uv run aggregator status`.
            #
            # Why the whole XDG_DATA_HOME rather than an aggregator-specific
            # variable: there isn't one. Three separate call sites read
            # XDG_DATA_HOME directly — the cache (store.py:1299), the retrieval
            # eval DB (aggregator/evals/db.py:91) and the TickTick backup
            # archive (aggregator/sources/ticktick.py:69). Overriding the base
            # covers all three at once and matches what tests/conftest.py:8-9
            # already does per-test.
            #
            # Why repo-local: it follows the worktree. Every `~/worktrees/
            # aggregator-*` checkout gets its own cache, so two branches under
            # test cannot corrupt each other, and deleting a worktree reclaims
            # the space. `.gitignore` anchors `/.devshell-data/` so the
            # (potentially multi-GB) database is neither committed by the
            # auto-commit cron nor copied into the Nix store when `nix develop`
            # snapshots a dirty git tree.
            #
            # `git rev-parse --show-toplevel` rather than `$PWD` because
            # `nix develop` can be run from a subdirectory; the `||` fallback
            # keeps the hook working if the flake is ever evaluated outside a
            # git checkout. The value must never end up empty —
            # ticktick.py:65-68 records that an empty XDG_DATA_HOME silently
            # takes the spec default, i.e. production again — so the fallback
            # is a literal path, not an unset variable.
            # `uv` is the one other tool here that keys off XDG_DATA_HOME, and
            # moving its stores would be a worse bug than the one being fixed.
            # Measured: with only XDG_DATA_HOME set, `uv python dir` and
            # `uv tool dir` inside the shell relocate to
            # `<repo>/.devshell-data/uv/{python,tools}` — so every worktree
            # would re-download its own CPython (and, finding no managed
            # interpreter, uv falls back to the Nix `python` on PATH, whose
            # numpy manylinux wheel then dies on a missing `libstdc++.so.6`).
            # Interpreters are not project state; they are a shared cache. Pin
            # both back to the real user data dir, captured BEFORE the
            # override. Only aggregator's own state moves.
            shellHook = ''
              _agg_root="$(${pkgs.git}/bin/git rev-parse --show-toplevel 2>/dev/null || true)"
              if [ -z "$_agg_root" ]; then
                _agg_root="$PWD"
              fi
              _agg_xdg_real="''${XDG_DATA_HOME:-$HOME/.local/share}"
              export UV_PYTHON_INSTALL_DIR="$_agg_xdg_real/uv/python"
              export UV_TOOL_DIR="$_agg_xdg_real/uv/tools"
              export XDG_DATA_HOME="$_agg_root/.devshell-data"
              mkdir -p "$XDG_DATA_HOME/aggregator"
              unset _agg_root _agg_xdg_real
              echo "aggregator devShell: XDG_DATA_HOME=$XDG_DATA_HOME (production cache at ~/.local/share/aggregator is NOT in scope)" >&2
            '';
          };
          packages.default = aggregatorPkg;

          checks.aggregator-embed-unit-hygiene = pkgs.runCommand
            "aggregator-embed-unit-hygiene"
            { nativeBuildInputs = [ pkgs.gnugrep pkgs.gnused ]; }
            ''
              set -euo pipefail
              units=${hmFixture.config.home-files}/.config/systemd/user
              fail() { echo "FAIL: $*" >&2; exit 1; }

              # Every unit this module generates. The notifiers are separate
              # per failing unit on purpose — see step 3. The tag units ride
              # the same hygiene: store-artifact ExecStart, no /home/, no
              # uv run, their own notifier with their own debounce stamp.
              all_units="aggregator-embed.service aggregator-embed.timer \
                         aggregator-embed-seed.service \
                         aggregator-embed-server.service \
                         aggregator-embed-failure-notify.service \
                         aggregator-embed-seed-failure-notify.service \
                         aggregator-tag.service aggregator-tag.timer \
                         aggregator-tag-failure-notify.service"

              for u in $all_units; do
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

              # ---- 3. Fail loudly, AND about the unit that actually failed -
              # Round-3 MEDIUM. One notifier used to be the OnFailure= target
              # for both the worker and the seeder, written as though only the
              # worker could fire it. So a SEED failure sent the operator to
              # `journalctl -u aggregator-embed.service` — a unit that had not
              # run — and told them the fix was to start
              # aggregator-embed-seed.service, the very unit whose failure they
              # were being told about. There is one notifier per unit now.
              #
              # `$SERVICE_RESULT`/`$EXIT_CODE` are not available to a separate
              # OnFailure= unit, and `OnFailure=notify@%n.service` could not be
              # verified on this host (systemd-analyze verify does not inspect
              # OnFailure= targets at all — checked with a control naming an
              # absent unit, which drew no complaint). With two statically
              # known units, generating two notifiers needs neither.
              notify_script_for() {
                # $1 = notify unit name. Echoes its ExecStart script path.
                sed -n 's/^ExecStart=\([^ ]*\).*/\1/p' \
                  "$(readlink -f "$units/$1")"
              }

              for pair in \
                "aggregator-embed.service:aggregator-embed-failure-notify.service" \
                "aggregator-embed-seed.service:aggregator-embed-seed-failure-notify.service" \
                "aggregator-tag.service:aggregator-tag-failure-notify.service"; do
                failing=''${pair%%:*}
                notifier=''${pair##*:}

                grep -q "^OnFailure=$notifier$" "$units/$failing" \
                  || fail "$failing does not name $notifier as its OnFailure target — a failure notification that describes a different unit sends the operator to a journal that says nothing about what broke"

                script=$(notify_script_for "$notifier")
                grep -q 'notify-send' "$script" \
                  || fail "$notifier does not call notify-send"

                # It must name ITS OWN unit's journal...
                grep -qF "journalctl --user -u $failing" "$script" \
                  || fail "$notifier never points at 'journalctl --user -u $failing' — the operator is sent to the wrong unit's journal"

                # ...and must not send them to the other one's.
                for other in aggregator-embed.service \
                             aggregator-embed-seed.service \
                             aggregator-tag.service; do
                  [ "$other" = "$failing" ] && continue
                  if grep -qF "journalctl --user -u $other" "$script"; then
                    grep -nF "journalctl --user -u $other" "$script" >&2
                    fail "$notifier tells the operator to read $other's journal, but it fires for $failing"
                  fi
                done
              done

              # The seeder's notification must not be circular: "your download
              # failed, so run the download" is what the shared notifier said,
              # and it said it because it was written for the other unit.
              # Re-running is legitimate ONLY once the cause is fixed, so the
              # advice has to be conditioned on that.
              seed_notify=$(notify_script_for aggregator-embed-seed-failure-notify.service)
              grep -qF 'Fix the cause below, then re-run' "$seed_notify" \
                || fail "aggregator-embed-seed-failure-notify.service names the seed unit as the remedy without telling the operator to fix the cause first — that is the circular advice this split exists to remove"

              # Kept for the assertions further down that predate the split.
              notify_script=$(notify_script_for aggregator-embed-failure-notify.service)

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
              #
              # Run for BOTH notifiers, and then once more ACROSS them: round
              # 3's M3 second half. The two used to share one stamp file, so a
              # worker failure armed the debounce and a seed failure minutes
              # later was suppressed outright — the operator got no popup at
              # all for the second, different, problem.
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

              for notifier in aggregator-embed-failure-notify.service \
                              aggregator-embed-seed-failure-notify.service \
                              aggregator-tag-failure-notify.service; do
                script=$(notify_script_for "$notifier")

                run_notify "$script" 1
                if grep -q 'suppressed' "$work/2.log"; then
                  echo "--- tick 1 ---" >&2; cat "$work/1.log" >&2
                  echo "--- tick 2 ---" >&2; cat "$work/2.log" >&2
                  fail "$notifier's debounce fails CLOSED: notify-send exited non-zero on tick 1, yet tick 2 was suppressed. An undelivered popup must never buy 24h of silence — arm the stamp only after a successful send"
                fi

                # The mirror assertion, so "fail open" cannot be satisfied by
                # deleting the debounce outright: a DELIVERED popup must arm
                # it, or the 30-minute timer raises 48 CRITICAL popups a day
                # and trains the user to ignore all of them.
                run_notify "$script" 0
                grep -q 'suppressed' "$work/2.log" \
                  || { cat "$work/2.log" >&2; \
                       fail "$notifier's popup is not debounced: a delivered notification must arm the 24h stamp, otherwise the embed timer raises 48 CRITICAL popups a day"; }
              done

              # ---- 3c. One unit's failure must not silence the OTHER -------
              # Executed, not grepped: both real scripts are run against ONE
              # shared XDG_STATE_HOME, worker first and seeder second, with a
              # notify-send stub that always succeeds. The seeder's popup must
              # still be delivered. With the pre-round-3 shared stamp it was
              # suppressed, so the operator learned about the embed worker and
              # never learned the weight download had died.
              rm -rf "$work"
              mkdir -p "$work/bin" "$work/state" "$work/home"
              worker_script=$(notify_script_for aggregator-embed-failure-notify.service)
              seed_script_notify=$(notify_script_for aggregator-embed-seed-failure-notify.service)
              stub_notify "$worker_script" "$work/worker.sh" 0
              stub_notify "$seed_script_notify" "$work/seed.sh" 0
              tick "$work/worker.sh" "$work/worker.log"
              tick "$work/seed.sh" "$work/seed.log"
              if grep -q 'suppressed' "$work/seed.log"; then
                echo "--- worker notification ---" >&2; cat "$work/worker.log" >&2
                echo "--- seed notification ---" >&2; cat "$work/seed.log" >&2
                fail "a worker failure silenced the seed unit's notification for 24h — the two notifiers share a debounce stamp, so one unit's problem hides an unrelated one. Give each unit its own stamp file"
              fi
              # ...and the reverse direction, which a single shared stamp
              # would break just as thoroughly.
              rm -rf "$work"
              mkdir -p "$work/bin" "$work/state" "$work/home"
              stub_notify "$seed_script_notify" "$work/seed.sh" 0
              stub_notify "$worker_script" "$work/worker.sh" 0
              tick "$work/seed.sh" "$work/seed.log"
              tick "$work/worker.sh" "$work/worker.log"
              if grep -q 'suppressed' "$work/worker.log"; then
                echo "--- seed notification ---" >&2; cat "$work/seed.log" >&2
                echo "--- worker notification ---" >&2; cat "$work/worker.log" >&2
                fail "a seed failure silenced the embed worker's notification for 24h — see above"
              fi
              rm -rf "$work"

              # ---- 3d. Every notifier has its OWN debounce stamp ----------
              # The executed pairwise proof above covers the two embed units;
              # with a third notifier the pair count grows quadratically, so
              # the general property is asserted structurally instead: the
              # stamp filename each rendered script arms must be unique. A
              # shared stamp is exactly the round-3 bug (one unit's failure
              # buying 24h of silence for a different unit's problem).
              stamps=""
              for n in aggregator-embed-failure-notify.service \
                       aggregator-embed-seed-failure-notify.service \
                       aggregator-tag-failure-notify.service; do
                script=$(notify_script_for "$n")
                stamp_name=$(sed -n 's|.*stamp="$stamp_dir/\([^"]*\)".*|\1|p' "$script")
                [ -n "$stamp_name" ] \
                  || fail "$n: could not locate its debounce stamp assignment — the structural uniqueness check below would assert nothing"
                stamps="$stamps$stamp_name
"
              done
              dupes=$(printf '%s' "$stamps" | sort | uniq -d)
              [ -z "$dupes" ] \
                || fail "failure notifiers share a debounce stamp ($dupes) — one unit's failure would silence another unit's notification for 24h"

              # ---- 4. Weights are never fetched by the unattended unit ----
              grep -q 'HF_HUB_OFFLINE=1' "$svc" \
                || fail "aggregator-embed.service is not pinned offline"

              # ---- 4b. The CPU cap is expressed, and expressed COHERENTLY -
              # 2026-08-27. The unit consumed 1d 7h 51min of CPU over 4h 3s of
              # wall clock — ~8x parallelism on the operator's laptop, i.e. a
              # fan that never spins down. `Nice=19` was already set and had
              # been read for weeks as if it addressed this; it does not, since
              # nice orders who runs first rather than how many run at once.
              #
              # WHAT IS ASSERTED IS THE PAIRING, not either number. A thread
              # pool larger than the quota is the one configuration that is
              # worse than doing nothing: every thread is still scheduled and
              # then throttled together, so the process pays full
              # context-switching for a fraction of the throughput. They come
              # from one `embedThreads` binding in nix/aggregator.nix, and this
              # step is what stops someone editing one of the four sites and
              # not the others.
              quota=$(sed -n 's/^CPUQuota=//p' "$svc")
              [ -n "$quota" ] \
                || fail "aggregator-embed.service has no CPUQuota — Nice=19 alone does not bound how many cores the embedder takes"
              case "$quota" in
                *%) ;;
                *) fail "aggregator-embed.service CPUQuota=$quota is not a percentage" ;;
              esac
              quota_cores=$(( ''${quota%\%} / 100 ))
              for var in OMP_NUM_THREADS MKL_NUM_THREADS RAYON_NUM_THREADS; do
                n=$(sed -n "s/^Environment=\"\\?$var=\\([0-9]*\\).*/\\1/p" "$svc")
                [ -n "$n" ] \
                  || fail "aggregator-embed.service does not set $var — torch's OpenMP pool, its MKL calls and HuggingFace's rayon tokenizer pool are three separate pools, and an uncapped one saturates every core on its own"
                [ "$n" = "$quota_cores" ] \
                  || fail "aggregator-embed.service sets $var=$n but CPUQuota=$quota ($quota_cores cores). A pool wider than the quota is throttled rather than smaller: same context-switching, less throughput. Size them from the one embedThreads binding."
              done

              # ---- 5. Staggered against the ingest timers -----------------
              embed_cal=$(sed -n 's/^OnCalendar=//p' "$units/aggregator-embed.timer")
              [ -n "$embed_cal" ] || fail "embed timer has no OnCalendar"
              for t in aggregator-sessions.timer aggregator-github.timer; do
                other=$(sed -n 's/^OnCalendar=//p' "$units/$t")
                if [ "$embed_cal" = "$other" ]; then
                  fail "embed timer shares OnCalendar ($embed_cal) with $t"
                fi
              done

              # ---- 6. The worker must start in the background -------------
              # A full catch-up takes weeks. Type=oneshot keeps the start job
              # active for that whole run, so Home Manager's sd-switch waits
              # for it during activation and blocks nixos-rebuild switch.
              # Type=simple considers the unit started once the worker is
              # spawned while preserving its exit status and OnFailure path.
              service_type=$(sed -n 's/^Type=//p' "$svc")
              [ "$service_type" = "simple" ] \
                || fail "aggregator-embed.service Type=$service_type — a long-running oneshot blocks Home Manager activation instead of starting in the background"

              # ---- 7. The start timeout must not fire on a healthy run ----
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

              # ---- 7b. A mistyped start timeout must fail at EVAL time ----
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

              # THE IDS COME FROM THE PYTHON SOURCE, not from a literal typed
              # here. `--seed-models` builds `Embedder()` and `Reranker()`, so
              # `_DEFAULT_MODEL_ST` and `_DEFAULT_MODEL` are what decide which
              # bytes are fetched. The previous version of this loop compared
              # the unit against two hardcoded strings that appear only in the
              # seed script's informational `echo`, so changing the real model
              # left it green — the derivation did not even depend on those
              # files. Now it does, and every string below is derived.
              py_embed_model=${pkgs.lib.escapeShellArg pyEmbedModel}
              py_rerank_model=${pkgs.lib.escapeShellArg pyRerankModel}
              py_embed_dir=${pkgs.lib.escapeShellArg (hfCacheDirOf pyEmbedModel)}
              py_rerank_dir=${pkgs.lib.escapeShellArg (hfCacheDirOf pyRerankModel)}
              # The exact file the server opens, relative to HF_HOME: repo
              # directory, PINNED revision, pinned filename. A unit gating on
              # "some snapshot exists" would wave through a cache holding the
              # wrong revision; one gating on a stale sha would refuse forever
              # on a correctly seeded machine.
              py_embed_path=${pkgs.lib.escapeShellArg "hub/${hfCacheDirOf pyEmbedModel}/snapshots/${pyEmbedRevision}/${pyEmbedFile}"}
              py_embed_socket=${pkgs.lib.escapeShellArg pyEmbedSocket}

              for repo in "$py_embed_model" "$py_rerank_model"; do
                grep -qF "$repo" "$seed_script" \
                  || fail "the seed unit never mentions $repo, which is the repo id the Python default actually resolves to — the module and the code disagree about which model this deployment fetches, so the operator is told one thing and 'embed --seed-models' downloads another"
              done

              # The PRESENCE GATE has to agree too, and it is the half that
              # bites silently. `have_model <dir>` is what makes the worker
              # refuse when weights are absent; point it at the cache
              # directory of a model nobody fetches and it either refuses
              # forever on a correctly seeded machine, or waves through a
              # machine holding the wrong weights.
              worker_script=$(sed -n 's/^ExecStart=\([^ ]*\).*/\1/p' "$svc")
              server_script=$(sed -n 's/^ExecStart=\([^ ]*\).*/\1/p' \
                                "$units/aggregator-embed-server.service")
              for pair in "aggregator-embed.service:$worker_script" \
                          "aggregator-embed-server.service:$server_script" \
                          "aggregator-embed-seed.service:$seed_script"; do
                grep -qF "$py_embed_path" "''${pair#*:}" \
                  || fail "''${pair%%:*} does not look for $py_embed_path — the pinned file of $py_embed_model that embed.py fetches and the server serves. It is gating on some other path, so 'weights present' is a claim about the wrong bytes"
              done
              grep -qF "$py_rerank_dir" "$seed_script" \
                || fail "the seed unit's presence check never looks at $py_rerank_dir — it would report 'already present' or 'downloading' about a directory the loaders do not use"

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
                       aggregator-embed-failure-notify.service \
                       aggregator-embed-seed-failure-notify.service \
                       aggregator-tag.service aggregator-tag.timer \
                       aggregator-tag-failure-notify.service; do
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

              # The worker's network: NONE. It reads the whole untrusted corpus,
              # and it reaches the encoder over a unix socket, so it needs no
              # IP family at all; the seeder is the one unit allowed the
              # internet. Asserted twice over — the exact directive, and the
              # absence of either IP family on any RestrictAddressFamilies line
              # — because the second is the property: an AF_INET "for
              # loopback" was measured to reach the LAN on this host, whose
              # user manager has no cgroup BPF to fence it with.
              grep -qxF 'RestrictAddressFamilies=AF_UNIX AF_NETLINK' "$svc" \
                || fail "aggregator-embed.service is missing 'RestrictAddressFamilies=AF_UNIX AF_NETLINK' — it must reach the embed server's unix socket and nothing else"
              for u in aggregator-embed.service aggregator-embed-server.service; do
                if grep -E '^RestrictAddressFamilies=.*AF_INET' "$units/$u"; then
                  fail "$u permits an IP address family — it processes untrusted text, needs only the unix socket, and IPAddressDeny/Allow cannot fence IP in this user manager"
                fi
                grep -qE '^RestrictAddressFamilies=' "$units/$u" \
                  || fail "$u sets no RestrictAddressFamilies at all, i.e. every family including IP"
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

              # ---- 8c. The tag unit's sandbox ------------------------------
              # `aggregator tag` shells out to `claude -p` with
              # attacker-influenced record bodies in the prompt, so the unit
              # carries the sibling sandbox to the extent the claude CLI
              # tolerates. Every directive below was verified on the host
              # (2026-09-04): `claude -p --model haiku` under exactly this
              # set via `systemd-run --user` answered and exited 0.
              tag_svc="$units/aggregator-tag.service"
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

              # ...and the directives that would BREAK it, step-8b style.
              if grep -q '^ProtectHome=' "$tag_svc"; then
                fail "aggregator-tag.service sets ProtectHome — claude -p must read ~/.claude config+credentials and write its own state under \$HOME"
              fi
              if grep -q '^PrivateNetwork=' "$tag_svc"; then
                fail "aggregator-tag.service sets PrivateNetwork — the claude CLI exists to reach its API; this unit is network-permitted like aggregator-github"
              fi
              if grep -q '^IPAddressDeny=any$' "$tag_svc"; then
                fail "aggregator-tag.service sets IPAddressDeny=any — that blocks the API call this unit exists to make"
              fi

              # ---- 10. The embed server ------------------------------------
              # EXECUTED, not grepped: the real launcher is run against a
              # scratch HF_HOME with the stub llama-server from the fixture,
              # which prints its argv. So what is asserted is the command line
              # the server would actually get.
              server_svc="$units/aggregator-embed-server.service"
              work="$TMPDIR/embed-server"
              rm -rf "$work"; mkdir -p "$work/hf"

              # (a) Unseeded: refuse with EX_CONFIG and name the seed unit, so
              # Restart=on-failure does not loop on something a restart
              # cannot fix.
              rc=0
              HF_HOME="$work/hf" "$server_script" > "$work/unseeded.log" 2>&1 || rc=$?
              [ "$rc" = 78 ] \
                || { cat "$work/unseeded.log" >&2; fail "the server launcher exited $rc on an unseeded cache, not 78 — RestartPreventExitStatus=78 is what stops a restart loop there"; }
              grep -qF 'aggregator-embed-seed.service' "$work/unseeded.log" \
                || fail "the server launcher's refusal does not name aggregator-embed-seed.service, the only unit that can fix it"
              grep -qxF 'RestartPreventExitStatus=78' "$server_svc" \
                || fail "aggregator-embed-server.service does not stop restarting on 78 — an unseeded cache becomes a restart loop"

              # (b) Seeded: the exact pinned file, and the flags the stamp
              # depends on.
              mkdir -p "$(dirname "$work/hf/$py_embed_path")"
              : > "$work/hf/$py_embed_path"
              # A socket file left by a SIGKILLed predecessor: llama-server
              # refuses to bind over it, so the launcher must clear it.
              mkdir -p "$(dirname "$work/rt/$py_embed_socket")"
              : > "$work/rt/$py_embed_socket"
              XDG_RUNTIME_DIR="$work/rt" HF_HOME="$work/hf" "$server_script" > "$work/argv" 2>&1 \
                || { cat "$work/argv" >&2; fail "the server launcher failed on a seeded cache"; }
              [ ! -e "$work/rt/$py_embed_socket" ] \
                || fail "the server launcher leaves a stale socket file in place — llama-server then fails with 'couldn't bind HTTP server socket' after any unclean exit"
              argv_has() { grep -qxF -- "$1" "$work/argv"; }
              argv_pair() {
                # $1 flag, $2 value: the value must follow the flag.
                grep -A1 -xF -- "$1" "$work/argv" | tail -n 1 | grep -qxF -- "$2"
              }
              argv_pair -m "$work/hf/$py_embed_path" \
                || fail "the server is not started on $py_embed_path — it would serve bytes the stamp does not describe"
              argv_has --embedding || fail "the server is not started with --embedding"
              # Qwen3-Embedding is last-token pooled, and no endpoint reports
              # pooling, so embed.py cannot check it on connect. This is the
              # only place it is pinned.
              argv_pair --pooling last \
                || fail "the server is not started with --pooling last — any other pooling yields well-formed vectors in a different space, and nothing downstream can tell"
              # The memory flags. With -ub 8192 and no flash attention the
              # unit pinned ~20 GiB of RAM for a 640 MB model (2026-10-02,
              # sweep in embedServerRunner's comment). Asserted so a flag
              # edit cannot bring that back without failing here.
              argv_pair -fa on \
                || fail "the server is not started with -fa on — without flash attention the KQ compute buffer scales with ubatch x context, and at -ub 8192 the unit pinned ~20 GiB for a 640 MB model"
              argv_pair -ub 1024 \
                || fail "the server is not started with -ub 1024 — the measured knee: -ub 512 runs at a third of the throughput, every larger size only adds memory (2.3 GiB at 512, 3.0 at 1024, 4.3 at 2048, 6.7 at 4096, 9.3 at 8192)"
              argv_pair --cache-ram 0 \
                || fail "the server is not started with --cache-ram 0 — llama-server's prompt cache (8 GiB default) keeps the KV state of past inputs in host RAM, and an embedding corpus has no shared prefixes to reuse"
              argv_pair --host "$work/rt/$py_embed_socket" \
                || fail "the server does not bind \$XDG_RUNTIME_DIR/$py_embed_socket, the socket embed.py's EMBED_SOCKET_NAME dials — the bare-registered MCP server would find nothing and every query would silently lose its vector arm"
              # The socket's directory is the unit's own RuntimeDirectory:
              # private to the user, and removed on stop so that "stopped"
              # means "no socket" to every client.
              grep -qxF "RuntimeDirectory=''${py_embed_socket%%/*}" "$server_svc" \
                || fail "aggregator-embed-server.service's RuntimeDirectory is not ''${py_embed_socket%%/*} — the socket directory would not exist, or would not be writable under ProtectHome=read-only (which covers /run/user)"
              grep -qxF 'RuntimeDirectoryMode=0700' "$server_svc" \
                || fail "aggregator-embed-server.service's socket directory is not 0700 — another local user could reach the encoder"

              # (c) Always on, restarted on failure, IP-free sandbox, and
              # the GPU still visible.
              grep -qxF 'WantedBy=default.target' "$server_svc" \
                || fail "aggregator-embed-server.service is not wanted by default.target — queries would embed against a server nobody started"
              grep -qxF 'Restart=on-failure' "$server_svc" \
                || fail "aggregator-embed-server.service does not restart on failure"
              for directive in \
                'NoNewPrivileges=true' \
                'PrivateTmp=true' \
                'RestrictNamespaces=true' \
                'RestrictSUIDSGID=true' \
                'LockPersonality=true' \
                'ProtectSystem=full' \
                'ProtectHome=read-only' \
                'RestrictAddressFamilies=AF_UNIX AF_NETLINK'; do
                grep -qxF "$directive" "$server_svc" \
                  || fail "aggregator-embed-server.service is missing '$directive' — it tokenizes the whole untrusted corpus in C++"
              done
              if grep -q '^PrivateDevices=' "$server_svc"; then
                fail "aggregator-embed-server.service sets PrivateDevices — that hides /dev/dri, and llama.cpp then runs on the CPU at a sixtieth of the speed without failing"
              fi

              # (d) The worker is ordered after it and pulls it in.
              grep -qxF 'Wants=aggregator-embed-server.service' "$svc" \
                || fail "aggregator-embed.service does not Want the embed server"
              grep -qxF 'After=aggregator-embed-server.service' "$svc" \
                || fail "aggregator-embed.service is not ordered After the embed server"

              # (e) The backend is the SOURCE default, not a unit override.
              # `cli._would_start_a_second_index_by_accident` refuses a
              # backfill whose only author is AGGREGATOR_EMBED_BACKEND, and
              # the MCP server (registered bare) cannot see a unit's
              # environment — so an override here would split the worker and
              # the queries across two indexes.
              [ ${pkgs.lib.escapeShellArg pyEmbedBackend} = server ] \
                || fail "embed.py's DEFAULT_BACKEND is ${pyEmbedBackend}, but this module deploys the server backend"
              for u in aggregator-embed.service aggregator-embed-seed.service; do
                if grep -qE 'AGGREGATOR_EMBED_(BACKEND|URL)=' "$units/$u"; then
                  grep -nE 'AGGREGATOR_EMBED_(BACKEND|URL)=' "$units/$u" >&2
                  fail "$u exports the embed backend or URL — the deployed choice must be the source default, which is the only thing every process agrees on"
                fi
              done
              rm -rf "$work"

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
