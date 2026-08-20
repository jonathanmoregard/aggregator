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


def configured_model_id() -> str:
    """Which model ``Embedder()`` would load right now, without loading it.

    The vector index is only valid for the model that wrote it, so this is
    what ``Store.migrate()`` stamps into the cache and compares against on
    every later run. It mirrors ``Embedder.__init__``'s default selection
    exactly — if the two ever disagree, the stamp starts vouching for vectors
    a different model produced, which is the failure the stamp exists to
    prevent. Kept next to those defaults for that reason.
    """
    backend = os.environ.get("AGGREGATOR_EMBED_BACKEND", "st")
    if backend == "gguf":
        return _DEFAULT_MODEL_GGUF
    return _DEFAULT_MODEL_ST


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
        self._st_model = None
        self._gguf_model = None
        if self.backend == "st":
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(
                self.model_name or _DEFAULT_MODEL_ST,
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
            repo_id = self.model_name or _DEFAULT_MODEL_GGUF
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
