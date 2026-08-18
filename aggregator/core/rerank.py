"""Qwen3-Reranker-0.6B cross-encoder wrapper.

DO NOT swap in bge-reranker-v2-m3: per Qwen's own eval it *degrades*
Qwen3-embedded English retrieval (MTEB-R 61.82 → 57.03). Qwen3-Reranker
is the vendor-tested-safe pairing.

Lazy load: instantiate ``Reranker()`` only when the caller passes
``rerank=True`` to the MCP tool. Adds ~2 GB RSS + ~300 ms/query.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "Qwen/Qwen3-Reranker-0.6B"

#: Commit sha of the weights this build was verified against. Same rule as
#: ``embed.QWEN3_EMBEDDING_REVISION``, and it binds harder here: this model is
#: loaded inside the long-lived MCP server, so an unpinned ``main`` is a
#: moving artifact executing in the process that holds the user's history.
QWEN3_RERANKER_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"


class Reranker:
    def __init__(self, model_name: str | None = None):
        from sentence_transformers import CrossEncoder

        pinned = model_name is None
        self.model_name = model_name or _DEFAULT_MODEL
        # NO ``trust_remote_code``. This constructor runs lazily inside the MCP
        # server process — the one holding the user's entire personal history —
        # and that server is registered bare, with no ``HF_HUB_OFFLINE`` wrapper
        # (only the timer-driven embed unit sets it). The flag would therefore
        # let a single ``rerank=True`` query fetch and execute
        # repository-controlled Python right there, while the tool advertises
        # ``openWorldHint=False``.
        # Nothing is given up: the Qwen3-Reranker repo ships no modeling code,
        # and the architecture is in-tree in transformers. Verified offline —
        # loads in 0.4 s as ``transformers.models.qwen3.Qwen3ForCausalLM``, and
        # ranks the relevant document first. Pinned by a test, not by comment.
        self._model = CrossEncoder(
            self.model_name,
            # No revision for a caller-supplied model: the pin was taken from
            # the default repository and vouches for nothing else.
            revision=QWEN3_RERANKER_REVISION if pinned else None,
        )

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        """Return one relevance score per doc (higher = more relevant)."""
        if not docs:
            return np.zeros((0,), dtype=np.float32)
        pairs = [(query, d) for d in docs]
        scores = self._model.predict(pairs, show_progress_bar=False)
        return np.asarray(scores, dtype=np.float32)
