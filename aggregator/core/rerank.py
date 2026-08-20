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

#: Hard cap on the tokens of one ``(query, document)`` pair.
#:
#: A LATENCY AND MEMORY BUDGET, NOT THE MODEL'S LIMIT. Saying so matters,
#: because the obvious justification — "512 is the cross-encoder's own maximum
#: anyway, so this costs nothing" — is true of the bge-reranker/XLM-R family
#: and FALSE here: Qwen3-Reranker-0.6B carries a 40960-token window, and
#: ``CrossEncoder`` inherits it, so the stage was configured to score whatever
#: it was handed.
#:
#: WHAT IT IS FOR, measured on a WAL-consistent read-only snapshot of the real
#: 505k-observation cache (``scripts/rag_rollout_smoke.py snapshot``), over the
#: golden query set's real pages at ``fields='full'``:
#:
#: * The candidate documents are SHORT IN THE MIDDLE AND ENORMOUS IN THE TAIL
#:   — 183 tokens at the median, 202 at p95, and 25941 at the maximum, with 3
#:   of 240 over 512.
#: * ``CrossEncoder.predict`` length-sorts and pads each batch to its longest
#:   member, so ONE oversized record sets the price of the whole 20-pair page.
#: * The first attempt to measure that page was OOM-KILLED: 20.2 GB RSS and
#:   9.4 GB of swap before the kernel stepped in. This model is constructed
#:   inside the MCP server — a stdio child of the editor, with no unit and no
#:   ``MemoryMax`` (see ``mcp.py``'s "KNOWN, ACCEPTED EXPOSURE") — so an
#:   untruncated pass is an availability failure the user experiences as their
#:   editor dying, arrived at by asking a legitimate question.
#:
#: So the cap does little for the MEDIAN page, whose documents are already
#: under it, and everything for the tail: unbounded becomes bounded.
#:
#: WHAT IS GIVEN UP: ranking signal past token 512 of a long document. On this
#: corpus that is 1.25% of candidates, and a cross-encoder's judgement is
#: dominated by the lead of a passage in any case.
MAX_PAIR_TOKENS = 512


class Reranker:
    def __init__(self, model_name: str | None = None):
        from sentence_transformers import CrossEncoder

        from aggregator.core.embed import downloads_allowed

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
            # OFFLINE UNLESS EXPLICITLY ALLOWED. This constructor runs inside
            # the MCP server on ``rerank=True``, so without it one query could
            # start a GB-scale download in the editor's process. See
            # ``embed.downloads_allowed``.
            local_files_only=not downloads_allowed(),
            # TRUNCATE. Without this the model's own 40960-token window
            # applies, a batch pads to its longest member, and a single long
            # record takes the editor's process down. See MAX_PAIR_TOKENS.
            max_length=MAX_PAIR_TOKENS,
        )

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        """Return one relevance score per doc (higher = more relevant)."""
        if not docs:
            return np.zeros((0,), dtype=np.float32)
        pairs = [(query, d) for d in docs]
        scores = self._model.predict(pairs, show_progress_bar=False)
        return np.asarray(scores, dtype=np.float32)
