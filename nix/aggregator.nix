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

  # The embedding model this deployment runs: the Q8_0 GGUF that
  # `aggregator-embed-server` serves on the GPU (aggregator/core/embed.py::
  # _DEFAULT_MODEL_GGUF, QWEN3_EMBEDDING_GGUF_REVISION and
  # QWEN3_EMBEDDING_GGUF_FILENAME), and where the huggingface_hub cache puts
  # that exact file at that exact revision.
  #
  # THE REVISION IS IN THE PATH, not just in the download. The server unit
  # opens `snapshots/<sha>/<file>` and nothing else, so a cache that also
  # holds some other revision of the repo — or a later `main` pulled by some
  # other tool — cannot change what is served. These four strings are copies
  # of the Python constants; `checks.<system>.aggregator-embed-unit-hygiene`
  # reads the constants OUT of embed.py and fails when they disagree, so a
  # re-pin in Python cannot leave the unit serving the old file.
  embedModelRepo = "Qwen/Qwen3-Embedding-0.6B-GGUF";
  embedModelDir = "models--Qwen--Qwen3-Embedding-0.6B-GGUF";
  embedModelRevision = "370f27d7550e0def9b39c1f16d3fbaa13aa67728";
  embedModelFile = "Qwen3-Embedding-0.6B-Q8_0.gguf";
  # Relative to HF_HOME. The worker's preflight, the server's launcher and the
  # seeder's report all test this one path.
  embedModelRelPath = "hub/${embedModelDir}/snapshots/${embedModelRevision}/${embedModelFile}";

  # The server's unix socket, relative to $XDG_RUNTIME_DIR. Its first
  # component is the unit's RuntimeDirectory (mode 0700, removed on stop).
  #
  # A SOCKET, NOT A TCP PORT, so the worker can keep `RestrictAddressFamilies=
  # AF_UNIX AF_NETLINK` — no IP at all — while it reads the untrusted corpus.
  # TCP was tried first and measured: the loopback-only fence for AF_INET
  # (`IPAddressDeny=any` + `IPAddressAllow=localhost`) is cgroup BPF, which
  # this host's user manager does not get, so a connect to the host's own LAN
  # address went straight through. TCP would have widened the worker from no
  # network to all of it.
  #
  # NOT an option, deliberately: the MCP server is registered bare and finds
  # the embedder at the source default (aggregator/core/embed.py::
  # EMBED_SOCKET_NAME), so a configurable path would silently split the
  # worker from every query. One value, asserted equal to the Python default
  # by the hygiene check.
  embedSocketName = "aggregator-embed-server/embed.sock";
  embedRuntimeDir = builtins.head (lib.splitString "/" embedSocketName);

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

  # HOW MANY CORES THE BACKGROUND EMBED WORKER MAY TAKE.
  #
  # ONE binding, used for BOTH the thread pools and the cgroup quota below, so
  # the two cannot drift. That pairing is the whole point: a pool sized larger
  # than the quota is the worst of the three configurations, because all of
  # those threads still get scheduled and are then throttled together — full
  # context-switching and cache-thrash for a fraction of the throughput. Sized
  # together they are just fewer, faster threads.
  #
  # ONE, since the worker stopped computing embeddings. The encoder runs in
  # `aggregator-embed-server` on the GPU; what is left in this process is
  # chunking, sha256 of each chunk, JSON over a unix socket and SQLite writes —
  # single-threaded Python that never loads torch at all on the `server`
  # backend. It was FOUR while the worker ran sentence-transformers on the
  # CPU, and that number was measured, not chosen: `1d 7h 51min` of CPU over
  # `4h 3s` of wall clock on 2026-08-27, i.e. a fan that never spun down, and
  # the operator's directive was to spend wall clock rather than heat. With
  # the arithmetic gone a one-core cap costs no throughput (the GPU is the
  # bottleneck, measured in docs/embedding-throughput.md) and bounds whatever
  # a regression back to CPU encoding would cost to one core rather than four.
  #
  # `Nice=19` stays for the reason it always had — who yields when the
  # operator starts typing — and so does `CPUWeight`. Not exposed as a module
  # option: this is the behaviour of a background indexer on a personal
  # machine, not a dial anyone should have to find.
  embedThreads = 1;

  # Environment shared by the embed worker and its one-shot seeding sibling.
  #
  # NO `AGGREGATOR_EMBED_BACKEND`, deliberately. It used to pin `st` here. The
  # backend is a SOURCE default now (aggregator/core/embed.py::
  # DEFAULT_BACKEND), for the rule `cli._would_start_a_second_index_by_accident`
  # enforces: a model change is made in source and deployed as a new store
  # path, and a backfill whose only author is an exported variable is refused.
  # Exporting it from the unit would make every deploy look like exactly that
  # accident — and would let the unit and the bare-registered MCP server
  # disagree about which index they are reading. No `AGGREGATOR_EMBED_URL`
  # either, for the same reason: the socket is the source default, see
  # `embedSocketName`.
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
    # The embedding model is checked by EXACT FILE at the pinned revision,
    # not by "some snapshot exists": that file is the only thing the server
    # will open, so any weaker test can wave through a cache that holds the
    # wrong revision and then fail one unit later with a worse message.
    embed_model="$hf_home/${embedModelRelPath}"
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
    #
    # This process no longer opens the file itself — aggregator-embed-server
    # does — but it is still the unit that fires every 30 minutes and has a
    # notifier, so it is where "the weights were never seeded" has to be
    # diagnosed by name. Without this the same fact would arrive as "the
    # embed server is not answering", which sends the operator to a unit
    # that is only failing for the reason below.
    if [ ! -e "$embed_model" ]; then
      echo "aggregator-embed: ${embedModelRepo} is not in the Hugging Face cache (looked for $embed_model)." >&2
      echo "aggregator-embed: this unit runs OFFLINE by design (HF_HUB_OFFLINE=1) and will NOT pull ~640 MB unattended." >&2
      echo "aggregator-embed: seed the cache once, then this timer and aggregator-embed-server take over:" >&2
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
    if [ -e "$embed_model" ]; then
      echo "aggregator-embed-seed: ${embedModelRepo}@${embedModelRevision} already present at $embed_model — nothing to download."
    else
      echo "aggregator-embed-seed: downloading ${embedModelFile} from ${embedModelRepo}@${embedModelRevision} (~640 MB) into $hf_home — served by aggregator-embed-server to fill the vector index. One-time cost."
    fi
    if have_model "${rerankModelDir}"; then
      echo "aggregator-embed-seed: ${rerankModelRepo} already present under $hf_home/hub/${rerankModelDir}/snapshots — nothing to download."
    else
      echo "aggregator-embed-seed: downloading ${rerankModelRepo} (~1.2 GB) into $hf_home — feeds the MCP server's rerank=True path. One-time cost."
    fi

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

  # The embedding model server. Always on; the worker and every MCP query
  # talk to it over a unix socket in the user's runtime directory.
  #
  # WHY A SERVER AT ALL. The worker used to run sentence-transformers on the
  # CPU at ~40 tokens/second — a 25-30 day backfill for this corpus. The same
  # model as a Q8_0 GGUF under llama.cpp's Vulkan backend on the Radeon 890M
  # iGPU measured 2495 tokens/second on this machine, with vectors agreeing
  # with the fp32 ones at cosine 0.9994 mean / 0.9990 min over 45 real chunks
  # (docs/embedding-throughput.md). Out of process, because the GPU runtime
  # is a C++ stack that has no business inside the editor's MCP process, and
  # one resident copy serves both the worker and queries.
  #
  # EVERY FLAG IS PART OF WHAT THE STAMP VOUCHES FOR, and the one that most
  # needs saying is `--pooling last`: Qwen3-Embedding is last-token pooled,
  # llama-server exposes the pooling mode on no endpoint, and mean pooling
  # would produce well-formed vectors in a different space with nothing
  # anywhere noticing. The Python side verifies the served FILE (name,
  # width, quantization) on connect; pooling can only be pinned here.
  #   -ngl 99           every layer on the GPU.
  #   -c 8192           context; a 4000-character chunk is ~1k tokens, and
  #                     the chunker's worst case (dense non-English text,
  #                     base64) stays well inside it. The ONLY size ceiling
  #                     on an input — see -ub.
  #   -b/-ub 1024       the micro-batch is a MEMORY/THROUGHPUT knob, not an
  #                     input limit. This block used to say "an embedding
  #                     input must fit ONE micro-batch, so the physical
  #                     batch is the context" and set both to 8192. That is
  #                     true of non-causal (BERT-style) embedders; Qwen3-
  #                     Embedding is a causal decoder with last-token
  #                     pooling, and llama-server carries an input across
  #                     micro-batches in the KV cache. Measured 2026-10-02
  #                     on tuxedo (build 10273, Vulkan, Radeon 890M): a
  #                     5997-token input embeds at every -ub from 512 to
  #                     8192, cosine ≥ 0.9999 between sizes. What -ub 8192
  #                     without flash attention DID do was pin ~20 GiB for a
  #                     640 MB model — the KQ compute buffer scales with
  #                     ubatch × context — and the unit sat at RSS 11.0 GiB
  #                     + GPU-pinned RAM (GTT) 8.7 GiB, cgroup peak 17.9
  #                     GiB, 6.4 GiB of it swapped. Sweep, with -fa on and
  #                     --cache-ram 0: 64 chunks of 828-1062 tokens, four
  #                     per request as the worker sends them, then one
  #                     5997-token input; resident = RSS + GTT after that;
  #                     tokens/s on the chunks, old and new under the same
  #                     background load (the worker was running):
  #                       -ub  512   2.3 GiB    638 tok/s
  #                       -ub 1024   3.0 GiB   2302 tok/s   <- this
  #                       -ub 2048   4.3 GiB   2269 tok/s
  #                       -ub 4096   6.7 GiB   2058 tok/s
  #                       -ub 8192   9.3 GiB   1250 tok/s   (fa on)
  #                       old flags 19.7 GiB   2033 tok/s   (fa off, cache 8 GiB)
  #                     1024 is the knee: the old throughput at a sixth of
  #                     the memory. -b is set EQUAL to -ub because
  #                     llama-server forces n_batch = n_ubatch for
  #                     embeddings ("setting n_batch = n_ubatch ... to avoid
  #                     assertion failure"); a larger -b in argv would be a
  #                     number that never runs. The flake's hygiene check
  #                     executes this launcher and asserts -ub, -fa and
  #                     --cache-ram, so an edit cannot bring the hog back
  #                     quietly.
  #   -fa on            flash attention: the KQ matrix is never materialised
  #                     (compute buffer 612 MiB at -ub 1024 instead of
  #                     gigabytes). Vectors agree with the non-FA ones at
  #                     cosine min 0.99978 / mean 0.99996 over the 64
  #                     chunks and 0.99992 on the 5997-token input — the
  #                     stamp stays valid.
  #   --cache-ram 0     llama-server's prompt cache (default 8192 MiB of
  #                     host RAM) keeps the KV state of past prompts to
  #                     reuse on a shared prefix. An embedding corpus has
  #                     no prefixes to reuse; the cache only grew RSS by
  #                     ~100 MiB per ~1k-token input until it hit the cap.
  #   --host <x>.sock   llama-server binds a unix socket when the host ends in
  #                     .sock (verified on build 10273, /v1/models and
  #                     /v1/embeddings both answer over it). The same path
  #                     rule as embed.py::default_embed_url, fallback included.
  #   --no-webui        nothing here wants a chat UI.
  #
  # Refuses, with the fix named, when the pinned file is absent — and exits
  # 78 (EX_CONFIG) so `RestartPreventExitStatus` stops Restart=on-failure
  # from turning an unseeded cache into a restart loop.
  embedServerRunner = pkgs.writeShellScript "aggregator-embed-server" ''
    set -uo pipefail

    ${modelPresenceCheck}
    if [ ! -e "$embed_model" ]; then
      echo "aggregator-embed-server: ${embedModelFile} (${embedModelRepo}@${embedModelRevision}) is not in the Hugging Face cache (looked for $embed_model)." >&2
      echo "aggregator-embed-server: seed it once — the only unit allowed to download weights:" >&2
      echo "aggregator-embed-server:     systemctl --user start aggregator-embed-seed.service" >&2
      echo "aggregator-embed-server: then: systemctl --user restart aggregator-embed-server.service" >&2
      exit 78
    fi

    runtime_dir="''${XDG_RUNTIME_DIR:-/run/user/$(${pkgs.coreutils}/bin/id -u)}"
    sock="$runtime_dir/${embedSocketName}"
    # llama-server will not bind over a socket file left by a SIGKILLed
    # predecessor ("couldn't bind HTTP server socket"). RuntimeDirectory= is
    # normally removed on stop, so this only matters for an unclean exit.
    ${pkgs.coreutils}/bin/rm -f "$sock"

    exec ${cfg.embed.server.package}/bin/llama-server \
      -m "$embed_model" \
      --embedding --pooling last \
      -ngl 99 -c 8192 -b 1024 -ub 1024 -fa on --cache-ram 0 \
      --host "$sock" \
      --no-webui
  '';

  # ---- LLM record tagging -----------------------------------------------
  #
  # Daily backfill of LLM topic tags onto the records-shaped sources
  # (`aggregator tag`). Needs the network — the `claude` CLI talks to the
  # API — so it is shaped like aggregator-github (network-permitted oneshot
  # + timer + OnFailure notifier), not like the sandboxed-offline embed
  # worker.
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
      + " not being filled. Likely: aggregator-embed-server is stopped"
      + " (offline-AI mode stops it; systemctl --user status"
      + " aggregator-embed-server.service), the Qwen3 GGUF is missing from the"
      + " HF cache (fix: systemctl --user start aggregator-embed-seed.service),"
      + " or the sqlite-vec extension did not load. Keyword search is"
      + " unaffected.";
  };

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
          is every 30 minutes at :15 and :45 — nominally offset from the
          ingest timers' `*:0/30`.

          THE OFFSET BUYS NOTHING AND IS NOT LOAD-BEARING. Measured on
          this machine 2026-08-30, embed runs took 15, 17, 47, 29, 29,
          30, 30 and 23 minutes while ingest runs took 3 to 17 minutes,
          both on 30-minute periods — so embed was already running
          back-to-back (21:03 -> 21:32 -> 22:02 -> 22:32 -> 23:03, with
          no gap at all) and every single embed tick overlapped an ingest
          run. Two jobs whose durations sum past their shared period
          cannot be separated by choosing a phase; there is no offset
          that works, and the corpus only grows.

          WHAT MAKES OVERLAP SAFE is `Store._CacheWriteLock`: both units
          take an OS `flock` on `<cache>.write.lock` before touching
          SQLite, for the length of a transaction, so the loser waits
          instead of failing. This replaced a claim that used to live
          here — "WAL plus a 30s busy_timeout, and one short write
          transaction per batch" — whose second half was true of the
          embed worker and false of ingest: `cli._ingest_entities` writes
          a whole source in ONE `upsert_entities` transaction, measured
          holding the write lock for 139 seconds in a single block. A 30s
          timeout against that loses every time, and it did, on every
          tick for at least five hours on 2026-08-30.

          Keeping the offset anyway: it costs nothing, and starting the
          two at the same instant would make one of them wait on the
          other for no reason.
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

      server.package = lib.mkOption {
        type = lib.types.package;
        default = pkgs.llama-cpp-vulkan;
        defaultText = lib.literalExpression "pkgs.llama-cpp-vulkan";
        description = ''
          The llama.cpp build whose `bin/llama-server` runs
          `aggregator-embed-server`. The Vulkan build is the one measured
          (2495 tokens/s on a Radeon 890M, see
          `docs/embedding-throughput.md`); any build with `llama-server`
          works, at its own speed. The model, the port and every flag are
          fixed by this module (see `embedServerRunner`) because they are
          part of what the index's embedding stamp vouches for.
        '';
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

      # ---- embed worker --------------------------------------------------
      # Fills the v5 vector index in the background. Never on the recall
      # path: queries fall through to FTS5 for anything not yet embedded,
      # so a cold or half-full index degrades ranking, never availability.
      systemd.user.services.aggregator-embed = lib.mkIf cfg.embed.enable {
        Unit = {
          Description = "Aggregator: background embed worker (vector index)";
          OnFailure = "aggregator-embed-failure-notify.service";
          # The encoder. Wants, not Requires: a stopped server must make THIS
          # run fail with its own message (the worker exits non-zero, names
          # the server unit and leaves the backlog untouched), not have
          # systemd refuse to start it with a dependency error that no
          # notifier describes. After, so a tick that starts both lets the
          # server fork first; the worker then waits out the model load.
          Wants = [ "aggregator-embed-server.service" ];
          After = [ "aggregator-embed-server.service" ];
        };
        Service = {
          # The catch-up worker is expected to stay alive for weeks while the
          # initial index fills. Type=oneshot leaves its start job active for
          # that whole run, which makes Home Manager's sd-switch wait during
          # activation and can block nixos-rebuild indefinitely. Type=simple
          # reports startup as soon as systemd spawns the worker; systemd still
          # owns the process and propagates its eventual failure to OnFailure.
          Type = "simple";
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
          # A LONG ROW STILL OUTLIVES THIS WINDOW, and that is now fine — but
          # only because of something the worker does, not something this
          # number does. Measured read-only against the live cache at the
          # chunker's `chunk-4000-400` geometry and the measured ~20 s per
          # chunk: 1348 rows (1298 observations + 50 records) each exceed 300 s
          # of encoding, and the largest single row is 257 chunks, about 86
          # minutes. So 5min is roughly three orders of magnitude under that
          # tail. It is sized to bound a wedged worker (see FINITE below),
          # never to let a long row finish.
          #
          # WHAT USED TO MAKE THAT A DATA-LOSS BUG. A row's encode was a single
          # `embed_documents` call over all of its chunks, and nothing inside
          # the call looked at the stop flag — it was only read once the call
          # returned. Stop the unit mid-row and systemd escalated to SIGKILL:
          # the claim survived, the next run's `_blame_crashed_row` read it as
          # a crash and attributed it to that row, and three such stops made it
          # terminal — a good row dropped from the vector arm permanently.
          # Reproduced against a real spawned worker and a real SIGTERM→SIGKILL
          # sequence.
          #
          # WHY THE FIX IS NOT HERE. Nothing finite covers an unbounded call:
          # raising this past 86 minutes would stop it bounding a wedged worker
          # at all, and with TimeoutStartSec=infinity above that makes a manual
          # stop no longer the last bound on one. Shortening it spreads the same
          # gap from the long tail to every routine stop. So the call stopped
          # being unbounded instead: `cli.py` slices each row's encode at
          # `_MAX_CHUNKS_PER_ENCODE` and reads the stop flag between slices, so
          # a SIGTERM is answered within ~80 seconds however long the row is,
          # and the row is put down — unmarked, uncommitted, still in the
          # backlog — rather than held into a SIGKILL.
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
          # on every library agreeing to read HF_HUB_OFFLINE. AF_UNIX is also
          # the transport to the encoder (`aggregator-embed-server`'s socket
          # in $XDG_RUNTIME_DIR), plus journal and dbus; AF_NETLINK because
          # glibc probes interfaces during resolver setup even when nothing
          # ever connects.
          #
          # NO IPAddressDeny/IPAddressAllow. They are cgroup BPF, and this
          # host's user manager does not get it: measured 2026-10-01 under
          # `systemd-run --user`, `IPAddressDeny=any` let a TCP connect to the
          # host's own LAN address through. With no IP family at all there is
          # nothing for them to add, and a directive that reads as a fence
          # but is not one is worse than none. The hygiene check asserts this
          # unit has no AF_INET/AF_INET6, which is the property that matters.
          RestrictAddressFamilies = "AF_UNIX AF_NETLINK";
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

      # ---- embed server --------------------------------------------------
      # The encoder, resident. See `embedServerRunner` for what it runs and
      # why every flag is fixed.
      systemd.user.services.aggregator-embed-server = lib.mkIf cfg.embed.enable {
        Unit = {
          Description = "Aggregator: Qwen3 embedding server (llama.cpp, unix socket %t/${embedSocketName})";
          # A crash loop must end somewhere a human can see it; the defaults
          # (5 starts in 10s) are too tight for a GPU driver that needs a
          # moment after resume.
          StartLimitIntervalSec = 300;
          StartLimitBurst = 5;
        };
        # ALWAYS ON. Queries embed through it, and a query must not wait for
        # a model load. The worker's `Wants=` also brings it back on the next
        # tick if it was stopped by hand and forgotten.
        Install.WantedBy = [ "default.target" ];
        Service = {
          Type = "simple";
          ExecStart = "${embedServerRunner}";
          Environment = [ "HF_HOME=${hfHome}" ];
          # $XDG_RUNTIME_DIR/aggregator-embed-server: where the socket lives.
          # 0700 so only this user can reach the encoder; removed by systemd
          # on stop, so "unit stopped" means "socket absent" to every client.
          # Also the one writable spot under /run/user: ProtectHome=read-only
          # below covers /run/user too, and llama-server could not bind there
          # without it (observed: "couldn't bind HTTP server socket").
          RuntimeDirectory = embedRuntimeDir;
          RuntimeDirectoryMode = "0700";
          Restart = "on-failure";
          RestartSec = "10s";
          # 78 is the launcher's "weights not seeded": restarting cannot fix
          # that, and the message already names the unit that does.
          RestartPreventExitStatus = "78";
          StandardOutput = "journal";
          StandardError = "journal";

          # ---- sandbox ------------------------------------------------------
          # It parses attacker-influenced text (every chunk of the corpus) in
          # a C++ tokenizer and hands it to a GPU driver, so it gets the same
          # "shrink the ambient surface" treatment as the worker. Verified by
          # running llama-server under exactly this set with
          # `systemd-run --user` on the deploying host (2026-10-01), unix
          # socket and all (RuntimeDirectory, no AF_INET): /v1/models and
          # /v1/embeddings answered over the socket at 2270 tokens/s, against
          # ~2500 for an unsandboxed TCP instance, i.e. still on the GPU.
          #
          # DELIBERATELY ABSENT:
          #   PrivateDevices: hides /dev/dri, so the model silently runs on
          #     the CPU at a sixtieth of the speed. The one directive a
          #     hardening pass would reach for first; the hygiene check
          #     asserts it stays off.
          #   MemoryDenyWriteExecute: not verified against the Vulkan
          #     driver's shader compilation, and the failure mode there is a
          #     quiet fallback to CPU, not an error.
          NoNewPrivileges = true;
          PrivateTmp = true;
          RestrictNamespaces = true;
          RestrictRealtime = true;
          RestrictSUIDSGID = true;
          LockPersonality = true;
          SystemCallArchitectures = "native";
          ProtectSystem = "full";
          # Read-only, not hidden: the model lives in the HF cache under
          # $HOME, and nothing else in $HOME is this process's business.
          ProtectHome = "read-only";
          ProtectKernelTunables = true;
          ProtectKernelModules = true;
          ProtectControlGroups = true;
          # Listens on a unix socket and dials nothing: no IP family at all.
          RestrictAddressFamilies = "AF_UNIX AF_NETLINK";
        };
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

      # ---- LLM record tagging -------------------------------------------
      # Network-permitted like the github ingest: the claude CLI needs the
      # API. Deliberately NOT under the torch sandbox — this unit runs no
      # model locally, it shells out to a CLI that must reach its endpoint.
      systemd.user.services.aggregator-tag = lib.mkIf cfg.tag.enable {
        Unit = {
          Description = "Aggregator: LLM topic tags for records (fills llm_tags)";
          OnFailure = "aggregator-tag-failure-notify.service";
        };
        Service = {
          Type = "oneshot";
          # Type=oneshot defaults TimeoutStartSec to infinity, which is
          # right here for the same reason as the embed worker: the FIRST
          # backfill over ~4.4k records is hours of claude invocations, and
          # a wall clock cannot tell that from a wedge. What bounds a wedge:
          # the CLI's per-invocation timeout (300s, bounded retries), the
          # per-batch committed checkpoint (a SIGTERM costs at most one
          # batch — graceful_shutdown stops at the boundary), and the timer
          # not stacking runs (a oneshot still `activating` is not
          # re-triggered).
          ExecStart = "${tagRunner}";
          StandardOutput = "journal";
          StandardError = "journal";
          # The sandbox its siblings carry, to the extent the claude CLI
          # tolerates — every directive below verified on this host
          # (2026-09-04): `claude -p --model haiku` under exactly this set
          # via `systemd-run --user` answered and exited 0. The prompt
          # carries attacker-influenced record bodies, so the unit gets the
          # same "shrink the ambient surface" treatment as the torch pair.
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
          # should not land at a predictable minute — and unlike the embed
          # timer there is no phase relationship with the ingest ticks worth
          # preserving at daily granularity.
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
