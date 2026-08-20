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

#: The window geometry every stored vector was produced under.
CHUNK_MAX_LEN = 4000
CHUNK_OVERLAP = 400

#: The chunker's contribution to the embedding version string.
#:
#: DERIVED FROM THE CONSTANTS ABOVE, not written out beside them, because the
#: whole job of this string is to change when the geometry changes. A version
#: someone has to remember to bump is one that silently stops describing the
#: index — and two chunkings of the same document are not interchangeable
#: vectors even when the model and the weights are identical: the text that
#: went into the encoder was different text.
#:
#: What it must NOT contain is anything that moves per deploy. A git hash here
#: would re-embed the entire corpus on every release, which on this hardware is
#: a multi-week operation (see ``docs/embedding-throughput.md``).
CHUNKER_VERSION = f"chunk-{CHUNK_MAX_LEN}-{CHUNK_OVERLAP}"


def chunk_body(
    text: str, max_len: int = CHUNK_MAX_LEN, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Return zero-or-more chunks of ``text``.

    Empty / whitespace-only input returns ``[]`` so the caller writes no
    vec rows for empty observations (there are ~half the corpus in the
    p<50 bucket). Short input returns ``[text]`` unchanged.

    Raises ``ValueError`` on parameters that cannot terminate. The hard-window
    fallback advances by ``stride = max_len - overlap``, so ``overlap >=
    max_len`` makes ``stride <= 0`` and the loop never moves ``i`` — it exits
    only on ``i + max_len >= len(text)``, which a non-advancing ``i`` never
    reaches. Measured with a bounded probe rather than by running it out: at
    ``max_len=overlap=4000`` it took 200,000 iterations without ``i`` leaving
    0, holding 800 MB of duplicate chunks and still growing. On the embed
    worker that is an unkillable-by-timeout hang inside one row, then an OOM
    kill, then — via the in-flight claim — a good row blamed for it.

    CHECKED BEFORE THE EARLY RETURNS, deliberately. A misconfiguration that
    only surfaced on bodies over ``max_len`` would pass every short row for
    weeks and then take the process down on one long one.
    """
    if max_len < 1:
        raise ValueError(f"max_len must be >= 1, got {max_len}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= max_len:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than max_len ({max_len}): "
            f"the hard-window fallback advances by max_len - overlap, so this "
            f"would never terminate"
        )
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
