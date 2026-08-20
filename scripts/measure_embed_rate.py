"""Measure the real per-chunk embedding rate on THIS machine.

Criterion F's instrument. The project's backfill estimate (25-30 days for ~483k
rows) was ~100x worse than the arithmetic in the SOTA research report, and a
100x gap is normally a bug — so it was measured instead of argued. The findings
live in ``docs/embedding-throughput.md``; this is what produced them, kept in
the tree so the numbers can be re-derived on other hardware rather than
believed.

Three questions, in order of how much they were suspected:

1. **Is it batching?** The worker calls ``embed_documents`` once per ROW, so the
   model usually sees a batch of one. Measured: worth under 10%, and negative
   past batch 8.
2. **Is it chunk size?** ``chunk-4000-400`` is ~804 tokens against the report's
   512. Measured: throughput is flat in sequence length, so this changes
   ms-per-chunk but not the cost of the corpus.
3. **Is it threading?** A hybrid 2 P-core + 8 E-core laptop part is exactly the
   shape that mis-schedules. Measured: it scales, so the cores are being used.

NOTHING IS DOWNLOADED. This loads whatever is already in the local Hugging Face
cache under the offline default; an earlier round pulled 15 GB by naming an
uncached model in a probe. Run it detached — a full sweep is ~30 minutes on the
machine it was written for:

    XDG_DATA_HOME=$(mktemp -d) python scripts/measure_embed_rate.py
"""
from __future__ import annotations

import json
import os
import sys
import time

WORD = "the quick brown fox jumps over the lazy dog and then writes some code "


def body(chars: int) -> str:
    """Filler of an exact character length, so the x-axis is the chunker's."""
    return (WORD * (chars // len(WORD) + 1))[:chars]


def main() -> int:
    # Never against the real cache: this builds nothing, but the embedder's
    # module-level environment reads are shared with code that would.
    os.environ.setdefault("XDG_DATA_HOME", "/tmp/aggregator-measure-xdg")

    import torch

    from aggregator.core.embed import Embedder

    print("threads default:", torch.get_num_threads(), flush=True)
    t0 = time.perf_counter()
    embedder = Embedder()
    print("load_s", round(time.perf_counter() - t0, 2), flush=True)
    tok = embedder._st_model.tokenizer
    print("max_seq_length", embedder._st_model.max_seq_length, flush=True)

    # Warm-up, so the first timed run is not paying lazy allocator init.
    embedder.embed_documents([body(200)])

    runs: list[dict] = []

    def run(label: str, chars: int, batch: int) -> None:
        docs = [body(chars) + f" #{i}" for i in range(batch)]
        ntok = len(tok(docs[0])["input_ids"])
        t = time.perf_counter()
        embedder.embed_documents(docs)
        dt = time.perf_counter() - t
        rec = {
            "label": label,
            "chars": chars,
            "tokens_per_chunk": ntok,
            "batch": batch,
            "wall_s": round(dt, 3),
            "ms_per_chunk": round(dt / batch * 1000, 1),
            "tokens_per_s": round(ntok * batch / dt, 1),
            "threads": torch.get_num_threads(),
        }
        runs.append(rec)
        print("RUN " + json.dumps(rec), flush=True)

    for chars in (4000, 2000, 1000, 500, 250):
        run("size-curve", chars, 1)
    for batch in (2, 4, 8, 16):
        run("batch-curve-4000", 4000, batch)
    for batch in (1, 8, 16, 32):
        run("batch-curve-2000", 2000, batch)
    for n in (2, 4, 6, 12):
        torch.set_num_threads(n)
        run(f"threads-{n}", 2000, 8)

    print("SUMMARY " + json.dumps(runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
