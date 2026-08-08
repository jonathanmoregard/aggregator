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


class Reranker:
    def __init__(self, model_name: str | None = None):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name or _DEFAULT_MODEL
        self._model = CrossEncoder(self.model_name, trust_remote_code=True)

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        """Return one relevance score per doc (higher = more relevant)."""
        if not docs:
            return np.zeros((0,), dtype=np.float32)
        pairs = [(query, d) for d in docs]
        scores = self._model.predict(pairs, show_progress_bar=False)
        return np.asarray(scores, dtype=np.float32)
