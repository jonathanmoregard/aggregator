"""Qwen3-Embedding-0.6B wrapper.

Two loader paths, one interface:

* ``AGGREGATOR_EMBED_BACKEND=st`` (default) — sentence-transformers +
  safetensors. ~1.2 GB RAM, no extra runtime.
* ``AGGREGATOR_EMBED_BACKEND=gguf`` — llama-cpp-python + Q4_K_M GGUF.
  ~400 MB RAM, tiny bit slower per query. Requires the optional
  ``embed-gguf`` extra installed.

Both paths return float32, L2-normalized, MRL-truncated to 768 dims.
The Qwen3 query prefix ("Instruct: ...\\nQuery: ...") is applied by
``embed_query``; documents go through ``embed_documents`` unprefixed
(load-bearing per the Qwen3 model card — omitting the prefix loses
1–5% retrieval on the leaderboard).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from aggregator.core.chunk import CHUNKER_VERSION

log = logging.getLogger(__name__)

QWEN3_QUERY_PREFIX = (
    "Instruct: Given a search query, retrieve relevant passages that answer the query\nQuery: "
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

#: The same pin for the ``-GGUF`` repository — **not yet established**.
#:
#: ``QWEN3_EMBEDDING_REVISION`` above was read off the safetensors repo and is
#: not a valid ref in the ``-GGUF`` one; they are separate repositories with
#: separate histories. Reusing it would not pin the download, it would break
#: it, and inventing a plausible-looking sha would be worse than either.
#:
#: So the hole is left open and NAMED rather than papered over. It is not a
#: silent one: ``Embedder`` refuses to construct the default gguf backend on
#: the download path while this is ``None`` (see ``_gguf_revision``), because
#: that path — and only that path — is where unpinned bytes would actually
#: move. An already-seeded machine loading from cache is unaffected.
#:
#: TO CLOSE IT: resolve the sha of ``Qwen/Qwen3-Embedding-0.6B-GGUF`` on a
#: machine with network access — ``huggingface_hub.HfApi().model_info(
#: "Qwen/Qwen3-Embedding-0.6B-GGUF").sha`` — verify the Q4_K_M file loads at
#: that revision, and put the 40-hex-character sha here. A sha, never a tag:
#: a tag is repointable by the repo owner, which is the thing being defended
#: against. Deliberately a source constant and NOT an environment variable —
#: an env-var pin is an in-place mutable knob, i.e. the exact thing
#: "pinned artifact, no in-place update" forbids. New sha → new commit → new
#: store path.
QWEN3_EMBEDDING_GGUF_REVISION: str | None = None


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
    import time, and it is already imported before ``aggregator.mcp`` finishes
    loading (``core.scrub`` → spaCy → thinc → transformers). Hence an explicit
    per-call ``local_files_only``, which no import order can defeat.

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
    if backend == "gguf":
        return _DEFAULT_MODEL_GGUF
    return _DEFAULT_MODEL_ST


#: What each backend does to the weights before they multiply anything.
#:
#: Two vectors from the same checkpoint at different precisions are close but
#: not equal, and a KNN compares them against each other with no way to tell —
#: so this belongs in the version string as surely as the repo id does.
_QUANTIZATION = {"st": "fp32", "gguf": "q4_k_m"}


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
    backend = os.environ.get("AGGREGATOR_EMBED_BACKEND", "st")
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
    backend = os.environ.get("AGGREGATOR_EMBED_BACKEND", "st")
    return _resolve_model_id(backend, None)


class Embedder:
    """Single-model embedder. Load once per process, share across writes."""

    def __init__(
        self,
        backend: str | None = None,
        model_name: str | None = None,
        gguf_filename: str = "Qwen3-Embedding-0.6B-Q4_K_M.gguf",
        cache_dir: str | Path | None = None,
    ):
        self.backend = backend or os.environ.get("AGGREGATOR_EMBED_BACKEND", "st")
        self.model_name = model_name
        #: The repo id THIS instance actually loaded, whatever the environment
        #: says. Vectors this embedder produces must be stamped with this and
        #: nothing else — see ``configured_model_id``. Set before any weights
        #: are touched, so it is readable even if the load below raises.
        self.model_id = _resolve_model_id(self.backend, model_name)
        self._st_model = None
        self._gguf_model = None
        if self.backend == "st":
            from sentence_transformers import SentenceTransformer

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
            # ``QWEN3_EMBEDDING_GGUF_REVISION`` is currently ``None`` because no
            # sha for the ``-GGUF`` repo has been verified (see its docstring).
            # ``revision=None`` resolves exactly as omitting the argument did,
            # so this line alone changes no behaviour — what changes behaviour
            # is ``_gguf_revision`` refusing the unpinned DOWNLOAD below. The
            # argument is spelled out regardless, so the pin is a wired,
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
        file, the constant and the command. The deployed units pin
        ``AGGREGATOR_EMBED_BACKEND=st`` and the ``embed-gguf`` extra is not in
        the closure, so no unit can reach this at all.
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

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Backend-specific encode. Returns raw native-dim vectors."""
        if self._st_model is not None:
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
        """Encode ``docs`` without the Qwen3 query prefix."""
        if not docs:
            return np.zeros((0, _EMBED_DIM), dtype=np.float32)
        raw = self._encode(list(docs))
        return self._truncate_and_normalize(raw)

    def embed_query(self, query: str) -> np.ndarray:
        """Encode ``query`` with the Qwen3 instruction prefix applied."""
        raw = self._encode([f"{QWEN3_QUERY_PREFIX}{query}"])
        return self._truncate_and_normalize(raw)[0]
