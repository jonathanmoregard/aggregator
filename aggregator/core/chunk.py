"""Paragraph-aware windowing for embedder ingest.

Called only for observation bodies over the embedder's context ceiling
(~8000 chars ≈ 2000 tokens fits inside Qwen3's 32K comfortably, but
we cap at 4000 chars for retrieval-locality: a paragraph-scale chunk
matches the user's semantic query granularity better than a whole-doc
embed). Splits on paragraph boundaries where possible; falls back to
hard windowing with a fixed overlap so retrieval hits near a boundary
still see full context.
"""

from __future__ import annotations


def chunk_body(text: str, max_len: int = 4000, overlap: int = 400) -> list[str]:
    """Return zero-or-more chunks of ``text``.

    Empty / whitespace-only input returns ``[]`` so the caller writes no
    vec rows for empty observations (there are ~half the corpus in the
    p<50 bucket). Short input returns ``[text]`` unchanged.
    """
    if not text or not text.strip():
        return []
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    paragraphs = text.split("\n\n")
    # Greedy paragraph packing when paragraphs help.
    if len(paragraphs) > 1 and all(len(p) <= max_len for p in paragraphs):
        cur = ""
        for p in paragraphs:
            candidate = f"{cur}\n\n{p}" if cur else p
            if len(candidate) <= max_len:
                cur = candidate
            else:
                if cur:
                    chunks.append(cur)
                cur = p
        if cur:
            chunks.append(cur)
        return chunks
    # Hard windowing fallback with exact overlap.
    i = 0
    stride = max_len - overlap
    while i < len(text):
        chunks.append(text[i : i + max_len])
        if i + max_len >= len(text):
            break
        i += stride
    return chunks
