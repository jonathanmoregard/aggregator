"""Qwen3-Embedding-0.6B wrapper.

Three loader paths, one interface:

* ``AGGREGATOR_EMBED_BACKEND=server`` (the SOURCE default, see
  ``DEFAULT_BACKEND``) — a local ``llama-server --embedding`` serving the
  pinned Q8_0 GGUF on the GPU, spoken to over loopback HTTP. The worker and
  the MCP server compute nothing themselves; the ``aggregator-embed-server``
  user unit does. Measured ~60x the CPU backends on this machine
  (``docs/embedding-throughput.md``).
* ``AGGREGATOR_EMBED_BACKEND=st`` — sentence-transformers + safetensors on
  CPU. ~1.2 GB RAM, no extra runtime.
* ``AGGREGATOR_EMBED_BACKEND=gguf`` — llama-cpp-python + Q4_K_M GGUF,
  in-process on CPU. Requires the optional ``embed-gguf`` extra.

All paths return float32, L2-normalized, MRL-truncated to 768 dims.
The Qwen3 query instruction ("Instruct: ...\\nQuery:...") is applied by
``embed_query``; documents go through ``embed_documents`` unprefixed
(load-bearing per the Qwen3 model card — omitting the instruction loses
1–5% retrieval on the leaderboard).

THE INSTRUCTION IS READ OFF THE MODEL, NOT RETYPED HERE. See
``Embedder.query_prompt``: it comes from the checkpoint's own
``config_sentence_transformers.json``, and ``QWEN3_QUERY_PREFIX`` below is only
the fallback for a load that cannot expose one.
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import socket
import time
import urllib.parse
from collections.abc import Callable
from pathlib import Path

import numpy as np

from aggregator.core.chunk import CHUNKER_VERSION

log = logging.getLogger(__name__)

#: The task instruction Qwen3-Embedding expects on the QUERY side — VERBATIM
#: from the checkpoint's ``config_sentence_transformers.json``, not a
#: restatement of it.
#:
#: THIS IS A FALLBACK. ``Embedder.query_prompt`` prefers the registry the
#: loaded model exposes; this literal covers the loads that cannot expose one
#: (the gguf backend has no such file, and a stripped export may not carry it).
#:
#: WHY IT IS COPIED CHARACTER-FOR-CHARACTER. It used to read "Given a search
#: query" with a trailing space after "Query:" — a plausible paraphrase, and
#: wrong in two places at once. Microsoft's Olive recipe for this exact model
#: records what that costs in the limit: an export that drops
#: ``config_sentence_transformers.json`` loses ~20% on the retrieval
#: benchmarks, because these task prompts are trained-in state and not
#: documentation. A near-miss is the same bug with a smaller blast radius, and
#: it is invisible without putting the two strings side by side — which is what
#: ``tests/core/test_embed_query_instruction.py`` now does against the file on
#: disk.
#:
#: No trailing space, deliberately: sentence-transformers concatenates
#: ``prompt + text`` with no separator, and so does Qwen's own published
#: ``get_detailed_instruct`` helper. The space was ours.
#:
#: NOT in ``embedding_version``. Documents never see it, so no stored vector
#: can be invalidated by changing it — see that function's docstring.
QWEN3_QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that "
    "answer the query\nQuery:"
)
_EMBED_DIM = 768  # MRL truncation target
_NATIVE_DIM = 1024
_DEFAULT_MODEL_ST = "Qwen/Qwen3-Embedding-0.6B"
_DEFAULT_MODEL_GGUF = "Qwen/Qwen3-Embedding-0.6B-GGUF"

#: Commit sha of the weights this build was verified against.
#:
#: "PINNED ARTIFACT, NO IN-PLACE UPDATE" HAS TO COVER THE WEIGHTS. Without a
#: revision every load resolves ``main`` on the hub, so the bytes a
#: rev-pinned systemd unit executes can change with no commit anywhere in
#: this repository. A sha rather than a tag, because a tag is repointable by
#: the repo owner — which is the thing being defended against.
#:
#: Only applied to the DEFAULT model: a pin taken from one repository says
#: nothing about a model name a caller passed in.
QWEN3_EMBEDDING_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"

#: The same pin for the ``-GGUF`` repository — a DIFFERENT repository.
#:
#: ``QWEN3_EMBEDDING_REVISION`` above was read off the safetensors repo and is
#: not a valid ref in the ``-GGUF`` one; they are separate repositories with
#: separate histories. This sha was read off the hub download metadata of
#: ``QWEN3_EMBEDDING_GGUF_FILENAME`` on the deploying machine (2026-09-30), and
#: the file it resolved to hashed to
#: ``06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439``. It
#: closes the hole this constant used to name as ``None``: the ``gguf``
#: backend's download refusal (``_gguf_revision``) no longer has anything to
#: refuse, and the ``server`` backend's weights are fetched at this revision
#: and nowhere else.
#:
#: VERIFIED FOR THE Q8_0 FILE ONLY. Whether this revision carries the
#: ``gguf`` backend's ``Q4_K_M`` default was not established — the repo may
#: not publish one at all, in which case that backend fails with a
#: not-found at this revision rather than resolving ``main``, which is the
#: honest failure.
#:
#: A sha, never a tag: a tag is repointable by the repo owner, which is the
#: thing being defended against. Deliberately a source constant and NOT an
#: environment variable — an env-var pin is an in-place mutable knob, i.e. the
#: exact thing "pinned artifact, no in-place update" forbids. New sha → new
#: commit → new store path. ``nix/aggregator.nix`` builds the server unit's
#: model path from this same value, and the flake's hygiene check reads it out
#: of this file so the two cannot disagree.
QWEN3_EMBEDDING_GGUF_REVISION: str | None = "370f27d7550e0def9b39c1f16d3fbaa13aa67728"

#: The one file the ``server`` backend accepts the server serving, and the one
#: the seed step fetches. Q8_0 rather than the ``gguf`` backend's Q4_K_M:
#: measured against 45 real chunks from the live cache, Q8_0 on the Vulkan
#: backend agrees with the fp32 safetensors vectors at cosine mean 0.9994,
#: min 0.9990 — the quantization is not what anyone would be trading for the
#: speed.
QWEN3_EMBEDDING_GGUF_FILENAME = "Qwen3-Embedding-0.6B-Q8_0.gguf"


#: The backend a process gets when nothing is exported. A SOURCE constant,
#: because a model change is made in source and deployed as a new store path —
#: ``cli._would_start_a_second_index_by_accident`` refuses a backfill whose
#: only author is an exported variable.
DEFAULT_BACKEND = "server"

#: Override for where the ``server`` backend finds llama-server:
#: ``unix:///abs/path.sock`` or ``http://host:port``. An environment value
#: because WHERE is deployment plumbing — the test suite points it at a stub,
#: a developer at a scratch instance — while WHAT is served is not: the
#: backend refuses a server that is not serving
#: ``QWEN3_EMBEDDING_GGUF_FILENAME``, so this can move the socket and never
#: the stamp. Unset in every deployed unit; see :func:`default_embed_url`.
EMBED_URL_ENV = "AGGREGATOR_EMBED_URL"

#: The server's socket, relative to ``$XDG_RUNTIME_DIR``. A UNIX SOCKET, NOT A
#: TCP PORT, because of the worker's sandbox: it reads the whole untrusted
#: corpus, and "offline by design" there means ``RestrictAddressFamilies=
#: AF_UNIX AF_NETLINK`` — no IP at all. The loopback-only alternative,
#: ``AF_INET`` plus ``IPAddressDeny=any``/``IPAddressAllow=localhost``, was
#: measured to confine NOTHING in the deploying host's user manager (no cgroup
#: BPF there: a connect to the host's LAN address went through), so TCP would
#: have widened the worker from no network to the whole network.
#:
#: The directory is the server unit's ``RuntimeDirectory=`` — mode 0700, so
#: only this user can reach the socket, and removed by systemd when the unit
#: stops, so a stale socket cannot outlive it. ``nix/aggregator.nix`` binds
#: exactly this path; the flake check reads it out of this line, so the unit
#: and every process that does not export ``AGGREGATOR_EMBED_URL`` — the MCP
#: server is registered bare — agree without anybody configuring them to.
EMBED_SOCKET_NAME = "aggregator-embed-server/embed.sock"


def default_embed_url() -> str:
    """``unix://$XDG_RUNTIME_DIR/<EMBED_SOCKET_NAME>`` for this process.

    Falls back to ``/run/user/<uid>`` — which is what systemd-logind makes
    ``XDG_RUNTIME_DIR`` anyway — for a process started with a scrubbed
    environment, so an MCP server launched without it still finds the unit.
    """
    runtime = os.environ.get("XDG_RUNTIME_DIR", "").strip() or f"/run/user/{os.getuid()}"
    return f"unix://{runtime.rstrip('/')}/{EMBED_SOCKET_NAME}"


class _UnixHTTPConnection(http.client.HTTPConnection):
    """``http.client`` over ``AF_UNIX``. Stdlib only, no new dependency."""

    def __init__(self, path: str, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self._socket_path = path

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        try:
            sock.connect(self._socket_path)
        except BaseException:
            sock.close()
            raise
        self.sock = sock


class _HTTPStatusError(Exception):
    """A non-2xx reply, carried to the one place that classifies it."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail

#: The unit a "server is unreachable" message sends the operator to.
EMBED_SERVER_UNIT = "aggregator-embed-server.service"

#: How long construction waits out llama-server's 503 "Loading model". The
#: worker is ordered After= the server unit, and a Type=simple unit counts as
#: started the moment it forks — before the ~600 MB model is on the GPU. A
#: refused CONNECTION is not waited on at all: that is a stopped unit, and the
#: MCP path must fall back to FTS5 at once rather than hang a query.
_SERVER_LOADING_GRACE_S = 30.0

#: Per-request ceiling. A batch here is ``cli._MAX_CHUNKS_PER_ENCODE`` chunks,
#: about a second of GPU time; two minutes only ever expires on a wedged
#: server, and then the worker's health probe turns it into an environment
#: fault rather than a blamed row.
_SERVER_TIMEOUT_S = 120.0
#: A query has a human waiting on it; past this the MCP path answers from
#: FTS5 instead.
_SERVER_QUERY_TIMEOUT_S = 15.0


def configured_backend() -> str:
    """The backend ``Embedder()`` builds in this process.

    An exported-but-empty variable means unset, the same reading
    ``cli._would_start_a_second_index_by_accident`` gives it, so the two can
    never disagree about whether a variable is "doing something".
    """
    return os.environ.get("AGGREGATOR_EMBED_BACKEND", "").strip() or DEFAULT_BACKEND


class EmbedServerError(RuntimeError):
    """The ``server`` backend cannot be used. Never a property of a row."""


class EmbedServerUnavailableError(EmbedServerError):
    """Nothing usable is answering at the embed URL — the unit is stopped,
    still loading past the grace period, or died mid-request."""


class EmbedServerMismatchError(EmbedServerError):
    """Something answers, but it is not serving the pinned model, so vectors
    from it would be stamped with an identity they do not have."""


#: The ONE opt-in that lets a model load reach the network.
MODEL_DOWNLOAD_ENV = "AGGREGATOR_ALLOW_MODEL_DOWNLOAD"


def downloads_allowed() -> bool:
    """Whether this process may fetch model weights from the hub.

    FAIL CLOSED, BECAUSE THE HARDENED PATH WAS THE ONLY HARDENED PATH.
    ``HF_HUB_OFFLINE=1`` is set on the timer-driven embed unit and nowhere
    else; the MCP server is registered bare. So the first ``rerank=True``, or
    the first text query once the index is warm, would have resolved the hub
    and pulled GB-scale weights inside the editor's MCP process — from a tool
    whose annotations declare ``openWorldHint=False``.

    An env var cannot fix that from inside this package:
    ``huggingface_hub`` reads ``HF_HUB_OFFLINE`` into a module constant at
    import time, so setting the variable after that import has no effect, and
    nothing here controls when the import happens — the first model
    construction on any path triggers it, from a process this package does not
    own the environment of. (Until 2026-09 it was even worse: ``core.scrub`` →
    spaCy → thinc → transformers pulled ``huggingface_hub`` in before
    ``aggregator.mcp`` had finished importing, so the variable was already read
    before a single line of aggregator code had run. Presidio initialises
    lazily now, which removes that particular certainty and changes nothing
    about the argument.) Hence an explicit per-call ``local_files_only``, which
    no import order can defeat.

    ``aggregator-embed-seed.service`` — human-triggered, never on a timer — is
    the single place in the deployment that sets this.
    """
    return os.environ.get(MODEL_DOWNLOAD_ENV, "").strip().lower() in ("1", "true", "yes")


def _resolve_model_id(backend: str, model_name: str | None) -> str:
    """The repo id a given ``(backend, model_name)`` pair resolves to.

    THE SINGLE RESOLUTION. ``Embedder.__init__`` and ``configured_model_id``
    used to answer this question with two separate copies of the same
    if-statement, and the docstring below claimed they "mirror exactly" — a
    claim held up by nothing but the two being adjacent. They are one function
    now, so the mirror is structural rather than aspirational.
    """
    if model_name is not None:
        return model_name
    if backend in ("gguf", "server"):
        return _DEFAULT_MODEL_GGUF
    return _DEFAULT_MODEL_ST


#: What each backend does to the weights before they multiply anything.
#:
#: Two vectors from the same checkpoint at different precisions are close but
#: not equal, and a KNN compares them against each other with no way to tell —
#: so this belongs in the version string as surely as the repo id does.
#:
#: ``server`` and ``gguf`` are both GGUF files from the same repository but
#: different quantizations run by different runtimes, so they are different
#: entries — and ``tests/core/test_embed_server_backend.py`` fails if any two
#: backends ever collapse onto one stamp.
_QUANTIZATION = {"st": "fp32", "gguf": "q4_k_m", "server": "q8_0"}


def configured_quantization(embedder: Embedder | None = None) -> str:
    """The precision the vectors in this cache are supposed to be.

    Read off the embedder when there is one, for the same reason
    :func:`configured_model_id` is: the object that did the work knows, and
    ``AGGREGATOR_EMBED_BACKEND`` only guesses.
    """
    if embedder is not None:
        backend = getattr(embedder, "backend", None)
        if isinstance(backend, str):
            return _QUANTIZATION.get(backend, backend)
    backend = configured_backend()
    return _QUANTIZATION.get(backend, backend)


def embedding_version(embedder: Embedder | None = None) -> str:
    """The identity a stored vector is keyed on: EVERYTHING that changed it.

    ``<repo id>-<quantization>@<dim>/<chunker>/norm-l2``, e.g.
    ``Qwen/Qwen3-Embedding-0.6B-fp32@768/chunk-4000-400/norm-l2``.

    WHY A BARE REPO ID WAS NOT ENOUGH. The stamp exists so vectors written by
    one build are never compared against vectors written by another, and a
    repo id is silent about three things that each change the bytes:

    * **quantization** — the same checkpoint at fp32 and at Q4_K_M produces
      different vectors, and ``AGGREGATOR_EMBED_BACKEND`` switches between
      them with no other trace;
    * **dimension** — MRL truncation to 768 of a 1024-wide model is a
      different embedding space, and the two are not even the same width;
    * **chunker version** — the encoder sees the text the chunker handed it,
      so re-chunking the corpus invalidates every vector in it even though
      not one byte of the model moved.

    Normalization is named too. ``_truncate_and_normalize`` re-normalizes
    AFTER truncating, which is what lets sqlite-vec's L2 distance stand in for
    cosine; a build that stopped doing that would be silently incomparable.

    WHAT IT MUST NEVER CONTAIN is anything that moves per deploy. A git hash
    or a build date here would invalidate the whole index on every release —
    on this hardware, a multi-week re-embed (``docs/embedding-throughput.md``)
    triggered by a typo fix. Every component below is a named constant or a
    property of the model actually loaded.

    THE QUERY INSTRUCTION IS DELIBERATELY ABSENT, and that is a decision rather
    than an oversight. The retrieval research is right that Qwen3's task
    instruction is part of the embedding contract; it is wrong that it belongs
    in THIS string, because this string exists for one purpose — so a vector
    written by one build is never compared against a vector written by another.
    The instruction cannot make two stored vectors incomparable, because no
    stored vector has ever seen it: ``embed_documents`` applies nothing. It
    transforms the QUERY, which is embedded fresh on every search, so a reword
    takes effect immediately and uniformly against the index already on disk.

    Keying on it would mean rewording a sentence invalidates ~483k document
    vectors and starts a 25-30 day re-embed to recompute bytes that provably do
    not change — the paragraph above, in its purest form. What WOULD belong
    here is a DOCUMENT-side instruction; Qwen3 ships an empty one, and
    ``tests/core/test_embed_query_instruction.py`` fails if that ever stops
    being true, so the exclusion cannot outlive its justification.
    """
    return (
        f"{configured_model_id(embedder)}"
        f"-{configured_quantization(embedder)}"
        f"@{_EMBED_DIM}"
        f"/{CHUNKER_VERSION}"
        f"/norm-l2"
    )


def configured_model_id(embedder: Embedder | None = None) -> str:
    """The model id vectors should be stamped with.

    The vector index is only valid for the model that wrote it, so this is
    what ``Store`` stamps into the cache and compares against on every later
    run. Round 1's H1 (refuse a foreign index rather than silently reranking
    against it) and round 2's S1 (never delete one without explicit consent)
    are both decisions taken FROM that stamp, so a stamp that can disagree
    with reality undermines both.

    PASS THE EMBEDDER WHENEVER ONE EXISTS. That is round 3's M2. This function
    took no arguments and read ``AGGREGATOR_EMBED_BACKEND`` only, while
    ``Embedder`` resolves from its own ``backend=``/``model_name=`` arguments
    first. So ``Embedder(backend="gguf")`` in a process with the variable
    unset wrote vectors from one model and stamped them with another — the
    stamp vouching for exactly the thing it exists to catch. With an embedder
    in hand the answer is read off the object that did the work, and it cannot
    be wrong.

    The no-argument form is still correct and still needed: the read path asks
    "may this process trust the vectors on disk?" before any embedder is
    built, and there the honest answer is what ``Embedder()`` WOULD load.
    """
    if embedder is not None:
        model_id = getattr(embedder, "model_id", None)
        if isinstance(model_id, str) and model_id:
            return model_id
        # A DOUBLE THAT CANNOT SAY WHAT IT IS. ``Embedder`` sets ``model_id``
        # before it touches a single weight, so this is unreachable from any
        # real one — it means a duck-typed stand-in was passed. Said out loud
        # because a silent answer here would be indistinguishable from the bug
        # this argument exists to fix, and then answered with the process
        # default, which is what such a stand-in is standing in for.
        log.warning(
            "%s was passed as the embedder writing this index but exposes no "
            ".model_id; stamping what Embedder() would load in this process "
            "instead. A real Embedder always carries one.",
            type(embedder).__name__,
        )
    return _resolve_model_id(configured_backend(), None)


def _pin_thread_pools(setter: Callable[[int], None] | None = None) -> int | None:
    """Shrink torch's intra-op pool to the cap the environment asked for.

    Returns the cap applied, or ``None`` when none was.

    WHY THIS EXISTS, MEASURED. ``aggregator-embed.service`` reported **1d 7h
    51min of CPU time over 4h 3s of wall clock** on 2026-08-27 — ~8x
    parallelism, sustained, on the operator's daily-driver laptop. The audible
    result is a fan that never spins down. ``Nice=19`` was already set and is
    the wrong instrument: nice orders who runs FIRST, not how many run AT
    ONCE, so twelve threads at nice 19 still saturate twelve cores.

    TWO KNOBS, NOT ONE, AND THE SECOND IS THE REASON THIS FUNCTION IS NOT JUST
    AN ENVIRONMENT VARIABLE. ``OMP_NUM_THREADS`` sizes the OpenMP pool at
    import; torch ALSO keeps its own intra-op pool, and on several builds that
    one is sized from the core count regardless of the variable. Setting only
    the variable therefore caps some of the parallelism some of the time,
    which is the worst of the three outcomes because it looks like it worked.

    AND IT SHRINKS THE POOL RATHER THAN THROTTLING IT. Pairing a twelve-thread
    pool with ``CPUQuota=400%`` does not give a third of the work at a third of
    the heat: all twelve threads still get scheduled and are throttled
    together, so the process pays full context-switch and cache-thrash cost for
    a third of the throughput. Fewer threads under the same quota is strictly
    better, so the quota and the pool are sized to match in
    ``nix/aggregator.nix``.

    FAILS OPEN, DELIBERATELY, IN EVERY DIRECTION. An unset variable pins
    nothing — the MCP server builds an ``Embedder`` too and a query there is
    one short encode a human is waiting on, so the unit that wants the cap is
    the unit that sets it. A value that is not a positive integer pins nothing
    either, because ``set_num_threads(0)`` is not a no-op but undefined
    behaviour that has been seen to abort, and this runs unattended from a
    timer where a typo in a unit file must cost the cap, never the backfill.
    """
    raw = os.environ.get("OMP_NUM_THREADS")
    if raw is None:
        return None
    try:
        cap = int(raw.strip())
    except ValueError:
        log.warning(
            "OMP_NUM_THREADS=%r is not an integer; leaving the torch intra-op "
            "pool at its default. The CPU cap this was meant to apply is NOT "
            "in effect.",
            raw,
        )
        return None
    if cap < 1:
        log.warning(
            "OMP_NUM_THREADS=%r is not a positive thread count; leaving the "
            "torch intra-op pool at its default rather than passing it to "
            "set_num_threads, where it is undefined behaviour.",
            raw,
        )
        return None
    if setter is None:  # pragma: no cover - exercised via the real torch path
        try:
            import torch
        except ImportError:
            return None
        setter = torch.set_num_threads
    try:
        setter(cap)
    except Exception as e:  # noqa: BLE001 - the cap is never worth the run
        log.warning(
            "could not pin the torch intra-op pool to %d threads (%s: %s); "
            "the CPU cap is NOT in effect in-process, though a CPUQuota= on "
            "the unit still bounds it.",
            cap,
            type(e).__name__,
            e,
        )
        return None
    return cap


class Embedder:
    """Single-model embedder. Load once per process, share across writes."""

    def __init__(
        self,
        backend: str | None = None,
        model_name: str | None = None,
        gguf_filename: str = "Qwen3-Embedding-0.6B-Q4_K_M.gguf",
        cache_dir: str | Path | None = None,
    ):
        self.backend = backend or configured_backend()
        self.model_name = model_name
        #: The repo id THIS instance actually loaded, whatever the environment
        #: says. Vectors this embedder produces must be stamped with this and
        #: nothing else — see ``configured_model_id``. Set before any weights
        #: are touched, so it is readable even if the load below raises.
        self.model_id = _resolve_model_id(self.backend, model_name)
        self._st_model = None
        self._gguf_model = None
        self._server_url: str | None = None
        self._server_model: str | None = None
        if self.backend == "server":
            if model_name is not None:
                # The server serves ONE file and this backend verifies it is
                # the pinned one; there is no second model it could vouch for.
                raise ValueError(
                    "the server embed backend serves only the pinned "
                    f"{_DEFAULT_MODEL_GGUF} ({QWEN3_EMBEDDING_GGUF_FILENAME}); "
                    f"model_name={model_name!r} cannot be honoured"
                )
            url = os.environ.get(EMBED_URL_ENV, "").strip() or default_embed_url()
            self._server_url = url.rstrip("/")
            parsed = urllib.parse.urlsplit(self._server_url)
            if parsed.scheme not in ("unix", "http") or not (
                parsed.path if parsed.scheme == "unix" else parsed.netloc
            ):
                raise ValueError(
                    f"{EMBED_URL_ENV}={url!r} is neither unix:///path.sock nor "
                    f"http://host:port"
                )
            self._server_model = self._verify_server()
        elif self.backend == "st":
            from sentence_transformers import SentenceTransformer

            # AFTER the import, which is what makes it worth doing at all.
            # Importing sentence_transformers imports torch, and torch sizes
            # its intra-op pool at import time from the visible core count —
            # so this is the first moment the pool exists to be resized. It is
            # a no-op unless the environment asked for a cap.
            _pin_thread_pools()
            self._st_model = SentenceTransformer(
                self.model_id,
                cache_folder=str(cache_dir) if cache_dir else None,
                # No revision for a caller-supplied model: this pin was taken
                # from the default repository and vouches for nothing else.
                revision=(
                    QWEN3_EMBEDDING_REVISION if self.model_name is None else None
                ),
                local_files_only=not downloads_allowed(),
            )
        elif self.backend == "gguf":
            try:
                from llama_cpp import Llama
            except ImportError as e:
                raise RuntimeError(
                    "AGGREGATOR_EMBED_BACKEND=gguf requires the "
                    "'embed-gguf' optional extra: pip install "
                    "'aggregator[embed-gguf]'"
                ) from e
            # RESOLVE THE FILE FIRST, THEN LOAD IT. Not
            # ``Llama.from_pretrained``: it forwards ``**kwargs`` to the
            # ``Llama`` constructor rather than to ``hf_hub_download``, so
            # there is no argument that can carry ``local_files_only`` through
            # it — which is how this became the ONE model-construction path
            # with no offline gate while the other three had one. That gap is
            # not cosmetic: ``AGGREGATOR_EMBED_BACKEND=gguf`` is read inside
            # the MCP server too, and that process is registered bare, so a
            # single query could have started a hub fetch from a tool that
            # advertises ``openWorldHint=False``.
            #
            # ``hf_hub_download`` takes the flag by name, so the guard is
            # explicit and no import order can defeat it — the same reasoning
            # as ``downloads_allowed``. It resolves into the same hub cache
            # ``from_pretrained`` used, so an already-seeded machine loads
            # exactly the file it loaded before.
            #
            # THE PIN IS PASSED HERE TOO, and that is round 3's M1. This call
            # used to omit ``revision=`` entirely while the ``st`` path four
            # branches up passed one, so the two backends were not equally safe
            # on the single path that can reach the network: gguf resolved
            # ``main``, a moving target, under a deployment whose whole rule is
            # that a rev-pinned unit executes fixed bytes. A comment admitting
            # the gap is not the same as closing it — nothing enforced it, and
            # nothing would have noticed it widening.
            #
            # ``QWEN3_EMBEDDING_GGUF_REVISION`` holds a verified sha since the
            # ``server`` backend needed one (see its docstring). Were it ever
            # ``None`` again, ``revision=None`` would resolve exactly as
            # omitting the argument did, and what stops that moving the bytes
            # is ``_gguf_revision`` refusing the unpinned DOWNLOAD. The
            # argument is spelled out either way, so the pin is a wired,
            # greppable, testable thing rather than a missing keyword nobody
            # can assert on.
            repo_id = self.model_id
            revision = self._gguf_revision(repo_id)

            from huggingface_hub import hf_hub_download

            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=gguf_filename,
                revision=revision,
                cache_dir=str(cache_dir) if cache_dir else None,
                local_files_only=not downloads_allowed(),
            )
            self._gguf_model = Llama(
                model_path=model_path,
                embedding=True,
                n_ctx=8192,
                verbose=False,
            )
        else:
            raise ValueError(f"unknown embed backend: {self.backend!r}")

        #: The instruction ``embed_query`` puts in front of the query text.
        #: Resolved once, after the weights are in hand, because the honest
        #: answer depends on WHICH model got loaded — see
        #: :meth:`_resolve_query_prompt`.
        self.query_prompt = self._resolve_query_prompt()

    def _resolve_query_prompt(self) -> str:
        """The query-side instruction for the model THIS instance loaded.

        READ IT OFF THE MODEL. ``config_sentence_transformers.json`` ships a
        ``prompts`` registry with the exact strings the checkpoint was trained
        and benchmarked with, and sentence-transformers exposes it as
        ``.prompts``. Taking the value from there rather than from a constant
        in this file means a checkpoint bump cannot leave a stale instruction
        glued to every query, and it removes the transcription step that had
        already introduced two errors (see ``QWEN3_QUERY_PREFIX``).

        This is the local form of the failure Microsoft's Olive recipe for this
        model documents: an export that leaves
        ``config_sentence_transformers.json`` behind drops the retrieval
        benchmarks by ~20%, because the task prompts are trained-in state.
        Dropping the file and retyping it slightly wrong are the same bug.

        A CALLER-SUPPLIED MODEL WITH NO REGISTRY GETS NO INSTRUCTION, and that
        is the same rule ``QWEN3_EMBEDDING_REVISION`` follows: a value taken
        from one repository vouches for nothing else. BGE, E5 and GTE each want
        a different prefix, so pasting Qwen's onto one of them is not a milder
        error than omitting it — it is a different wrong answer. A Qwen3
        checkpoint passed by name still gets its own, because it carries its
        own registry.

        THE DEFAULT MODEL FALLS BACK TO THE LITERAL rather than to nothing.
        No registry on the model this package pins means an old
        sentence-transformers or a stripped export — the Olive case exactly —
        and 1–5% of retrieval is not worth losing to a missing config file when
        the string is known.
        """
        prompts = getattr(self._st_model, "prompts", None)
        if isinstance(prompts, dict):
            shipped = prompts.get("query")
            if isinstance(shipped, str) and shipped:
                return shipped
        if self.model_id in (_DEFAULT_MODEL_ST, _DEFAULT_MODEL_GGUF):
            return QWEN3_QUERY_PREFIX
        return ""

    @staticmethod
    def _gguf_revision(repo_id: str) -> str | None:
        """The revision to resolve the gguf repo at — or a loud refusal.

        A CALLER-SUPPLIED REPO GETS NO PIN AND NO REFUSAL, exactly as on the
        ``st`` path: a pin taken from one repository vouches for nothing else,
        and someone who names their own repo has already chosen it. The rule
        being enforced is only about the repo this package picks by default.

        REFUSING IS THE POINT, and only on the download path. While
        ``QWEN3_EMBEDDING_GGUF_REVISION`` is ``None`` the default gguf repo has
        no verified sha, so a fetch would resolve ``main`` — whatever the repo
        owner pushed most recently — into a deployment that claims every
        artifact is pinned to a commit. Loading an ALREADY-SEEDED cache is
        untouched: those bytes are on disk and not moving, and breaking a
        working offline load to protest a missing pin would help nobody.

        Loud rather than silent, and a raise rather than a warning, because
        this can only be reached by a human who opted into ``gguf`` AND into
        ``AGGREGATOR_ALLOW_MODEL_DOWNLOAD`` in the same breath — someone
        watching a terminal right now, who can act on a message that names the
        file, the constant and the command. The deployed units run the
        ``server`` backend and the ``embed-gguf`` extra is not in the closure,
        so no unit can reach this at all.
        """
        if repo_id != _DEFAULT_MODEL_GGUF:
            return None
        if QWEN3_EMBEDDING_GGUF_REVISION is not None:
            return QWEN3_EMBEDDING_GGUF_REVISION
        if not downloads_allowed():
            # Offline: nothing can move, so nothing to refuse.
            return None
        raise RuntimeError(
            f"refusing to DOWNLOAD {_DEFAULT_MODEL_GGUF} unpinned. "
            f"aggregator.core.embed.QWEN3_EMBEDDING_GGUF_REVISION is None, so "
            f"this fetch would resolve 'main' — whatever that repo holds right "
            f"now — while every other artifact in this deployment is pinned to "
            f"a commit. QWEN3_EMBEDDING_REVISION cannot be reused: it belongs "
            f"to the safetensors repo {_DEFAULT_MODEL_ST} and is not a valid "
            f"ref here. Either use the pinned default backend "
            f"(AGGREGATOR_EMBED_BACKEND=st), or set "
            f"QWEN3_EMBEDDING_GGUF_REVISION in aggregator/core/embed.py to a "
            f"sha you verified — "
            f"HfApi().model_info({_DEFAULT_MODEL_GGUF!r}).sha — and rebuild."
        )

    # -- the server backend ---------------------------------------------------

    def _server_request(
        self,
        path: str,
        payload: dict | None = None,
        timeout: float = _SERVER_TIMEOUT_S,
    ) -> dict:
        """One JSON round-trip to llama-server. Stdlib only, on purpose.

        EVERY WAY THE SOCKET CAN BE MISSING RAISES ONE TYPE. A refused
        connection (unit stopped), a dropped one (unit stopped mid-request), a
        timeout (wedged) and a 503 (still loading) are all "the encoder is not
        there", and callers decide on that one fact: the worker's health probe
        refuses to blame a row for it, and the MCP path answers from FTS5. Any
        OTHER HTTP status is a real answer about THIS input — a body the server
        rejects — and is raised as a plain ``RuntimeError`` so the worker's
        probe can find out whether it discriminates between rows.
        """
        assert self._server_url is not None
        where = f"{self._server_url}{path}"
        parsed = urllib.parse.urlsplit(self._server_url)
        if parsed.scheme == "unix":
            conn: http.client.HTTPConnection = _UnixHTTPConnection(parsed.path, timeout)
        else:
            conn = http.client.HTTPConnection(parsed.netloc, timeout=timeout)
        body = None if payload is None else json.dumps(payload).encode()
        try:
            try:
                conn.request(
                    "GET" if body is None else "POST",
                    path,
                    body=body,
                    headers={"Content-Type": "application/json"} if body else {},
                )
                resp = conn.getresponse()
                raw = resp.read()
            finally:
                conn.close()
            if resp.status >= 300:
                raise _HTTPStatusError(resp.status, raw[:500].decode("utf-8", "replace"))
            return json.loads(raw)
        except _HTTPStatusError as e:
            if e.status == 503:
                raise EmbedServerUnavailableError(
                    f"{where} answered 503 ({e.detail}); llama-server says this "
                    f"while its model is still loading"
                ) from e
            raise RuntimeError(f"{where} answered HTTP {e.status}: {e.detail}") from e
        except (OSError, ValueError, http.client.HTTPException) as e:
            # A missing socket file (unit stopped: systemd removed its runtime
            # directory), a refused or reset connection, a timeout, and a reply
            # cut off mid-body are all OSError or HTTPException; ValueError is
            # a body that is not JSON — a half-written reply from a process
            # that died.
            raise EmbedServerUnavailableError(
                f"the embed server at {where} is not answering "
                f"({type(e).__name__}: {e}). It is the {EMBED_SERVER_UNIT} "
                f"user unit — check `systemctl --user status "
                f"{EMBED_SERVER_UNIT}`; offline-AI mode stops it on purpose. "
                f"Set {EMBED_URL_ENV} only to point at a different instance "
                f"of the same model."
            ) from e

    def _verify_server(self) -> str:
        """Confirm the server serves the pinned file. Returns its model id.

        THE STAMP IS A SOURCE CONSTANT AND THE URL IS NOT. Without this, a
        variable pointed at any llama-server — the chat model on 8717, an f16
        export, some other embedder entirely — would produce vectors that get
        stamped ``…-GGUF-q8_0`` and mixed into the index as if they were.
        ``(chunk_id, model)`` keying cannot catch that: the key is exactly what
        would be wrong. So the identity is checked here, before anything is
        embedded, against what the server itself reports.

        WHAT IS CHECKED is what ``/v1/models`` exposes on llama.cpp build 10273:
        the served file's name (``data[].id`` is the ``-m`` path) and, when
        ``meta`` is present, its native width and its quantization. Pooling is
        NOT exposed by any endpoint, so ``--pooling last`` is enforced where it
        is decided — the unit's command line in ``nix/aggregator.nix`` — and
        the measured parity of that exact command line is recorded in
        ``docs/embedding-throughput.md``.
        """
        deadline = time.monotonic() + _SERVER_LOADING_GRACE_S
        while True:
            try:
                listing = self._server_request("/v1/models")
                break
            except EmbedServerUnavailableError as e:
                loading = isinstance(e.__cause__, _HTTPStatusError)
                if not loading or time.monotonic() >= deadline:
                    raise
                time.sleep(0.5)

        served = listing.get("data") or []
        for entry in served:
            name = str(entry.get("id") or "")
            if os.path.basename(name) != QWEN3_EMBEDDING_GGUF_FILENAME:
                continue
            meta = entry.get("meta") or {}
            n_embd = meta.get("n_embd")
            ftype = meta.get("ftype")
            if n_embd is not None and n_embd != _NATIVE_DIM:
                raise EmbedServerMismatchError(
                    f"{self._server_url} serves {name} with n_embd={n_embd}, "
                    f"but {_DEFAULT_MODEL_GGUF} is {_NATIVE_DIM} wide — this "
                    f"is not the model the index is stamped for"
                )
            if ftype is not None and str(ftype).upper() != "Q8_0":
                raise EmbedServerMismatchError(
                    f"{self._server_url} serves {name} as {ftype}, not Q8_0 — "
                    f"the stamp names q8_0 and these vectors would not be"
                )
            return name
        names = [str(e.get("id")) for e in served] or ["nothing"]
        raise EmbedServerMismatchError(
            f"refusing the embed server at {self._server_url}: it serves "
            f"{', '.join(names)}, not {QWEN3_EMBEDDING_GGUF_FILENAME} from "
            f"{_DEFAULT_MODEL_GGUF}@{QWEN3_EMBEDDING_GGUF_REVISION}. Vectors "
            f"from it would be stamped as that model and mixed into its index. "
            f"Point {EMBED_URL_ENV} at a server running the pinned file (the "
            f"{EMBED_SERVER_UNIT} unit does), or unset it."
        )

    def _server_encode(self, texts: list[str], timeout: float) -> np.ndarray:
        """All of ``texts`` in ONE request, rows put back by ``index``.

        One request per call, not per text: the GPU's advantage is the batch,
        and ``-b``/``-ub`` on the unit are sized so a full
        ``cli._MAX_CHUNKS_PER_ENCODE`` slice fits one micro-batch.
        """
        reply = self._server_request(
            "/v1/embeddings",
            {"input": list(texts), "model": self._server_model},
            timeout=timeout,
        )
        rows = reply.get("data") or []
        if len(rows) != len(texts):
            raise RuntimeError(
                f"the embed server returned {len(rows)} embedding(s) for "
                f"{len(texts)} input(s)"
            )
        out = np.empty((len(texts), _NATIVE_DIM), dtype=np.float32)
        seen: set[int] = set()
        for row in rows:
            i = int(row["index"])
            vec = np.asarray(row["embedding"], dtype=np.float32)
            if vec.shape != (_NATIVE_DIM,) or i in seen or not 0 <= i < len(texts):
                raise RuntimeError(
                    f"the embed server returned a malformed embedding "
                    f"(index {i}, shape {vec.shape})"
                )
            seen.add(i)
            out[i] = vec
        return out

    def _encode(self, texts: list[str], *, timeout: float = _SERVER_TIMEOUT_S) -> np.ndarray:
        """Backend-specific encode. Returns raw native-dim vectors."""
        if self._server_url is not None:
            arr = self._server_encode(texts, timeout)
        elif self._st_model is not None:
            arr = self._st_model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        elif self._gguf_model is not None:
            arr = np.array([self._gguf_model.embed(t) for t in texts], dtype=np.float32)
        else:
            raise RuntimeError("no embedder backend loaded")
        return arr.astype(np.float32)

    @staticmethod
    def _truncate_and_normalize(arr: np.ndarray) -> np.ndarray:
        """MRL truncation + L2 normalization.

        Qwen3-Embedding is trained with Matryoshka losses; truncating the
        first ``_EMBED_DIM`` dims of the native ``_NATIVE_DIM`` output
        preserves ranking quality within a few tenths of a point per the
        Qwen3 tech report. Renormalize after truncation so cosine ≡ dot
        product downstream (sqlite-vec + RRF both assume unit norm).
        """
        if arr.shape[1] > _EMBED_DIM:
            arr = arr[:, :_EMBED_DIM]
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (arr / norms).astype(np.float32)

    def embed_documents(self, docs: list[str]) -> np.ndarray:
        """Encode ``docs`` EXACTLY AS GIVEN — no instruction, ever.

        Qwen3 ships ``prompts['document'] == ''``, and this path applies
        nothing rather than applying an empty string, so the two cannot drift
        apart. That is what makes the query instruction safe to leave out of
        :func:`embedding_version`: no stored vector has ever seen one, so
        rewording it cannot invalidate a single row. Add a document-side
        instruction and that stops being true and the version string has to
        move with it — ``tests/core/test_embed_query_instruction.py`` fails on
        the day somebody tries.
        """
        if not docs:
            return np.zeros((0, _EMBED_DIM), dtype=np.float32)
        raw = self._encode(list(docs))
        return self._truncate_and_normalize(raw)

    def embed_query(self, query: str) -> np.ndarray:
        """Encode ``query`` with this model's own task instruction applied.

        Plain concatenation rather than ``encode(prompt_name='query')``, and
        the two are identical here: the checkpoint's pooling config sets
        ``include_prompt: true``, so instruction tokens are pooled either way.
        One code path then serves both backends — the gguf loader has no prompt
        registry to call through — instead of two that are quietly different.
        """
        raw = self._encode(
            [f"{self.query_prompt}{query}"], timeout=_SERVER_QUERY_TIMEOUT_S
        )
        return self._truncate_and_normalize(raw)[0]


def fetch_server_weights() -> str:
    """Resolve the pinned Q8_0 file into the hub cache. Returns its path.

    What the ``aggregator-embed-server`` unit serves, fetched at the pinned
    revision into the standard hub layout —
    ``$HF_HOME/hub/models--Qwen--Qwen3-Embedding-0.6B-GGUF/snapshots/<sha>/<file>``
    — which is the exact path the unit's launcher resolves. Offline unless
    ``AGGREGATOR_ALLOW_MODEL_DOWNLOAD`` is set, like every other load, so run
    without the opt-in it is a presence check that names the fix.
    """
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=_DEFAULT_MODEL_GGUF,
        filename=QWEN3_EMBEDDING_GGUF_FILENAME,
        revision=QWEN3_EMBEDDING_GGUF_REVISION,
        local_files_only=not downloads_allowed(),
    )


def seed_embedder() -> object:
    """Make the configured embedder's weights present. ``embed --seed-models``.

    FOR THE SERVER BACKEND THIS MUST NOT CONSTRUCT AN ``Embedder``. That would
    ask the server whether it is serving the right file — and on a fresh
    machine the server cannot be running yet, because the file it serves is
    the thing being seeded. So the weights are fetched directly, and the
    server is never contacted. The other backends load in-process, and
    constructing them is still the honest proof their weights load.
    """
    backend = configured_backend()
    if backend == "server":
        return fetch_server_weights()
    return Embedder(backend=backend)
