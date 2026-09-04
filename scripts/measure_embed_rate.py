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

NOTHING IS DOWNLOADED by the bare invocation. It loads whatever is already in
the local Hugging Face cache under the offline default; an earlier round pulled
15 GB by naming an uncached model in a probe. Run it detached — a full sweep is
~30 minutes on the machine it was written for:

    XDG_DATA_HOME=$(mktemp -d) python scripts/measure_embed_rate.py

THE GGUF BENCH is the one mode that may touch the network, and only because a
human asked for it three times over: ``--backend gguf``, plus
``AGGREGATOR_ALLOW_MODEL_DOWNLOAD=1``, plus either ``--resolve-and-pin`` or an
explicit ``--gguf-revision``/``AGGREGATOR_GGUF_BENCH_REVISION`` sha. It exists
to close the hole ``embed.py`` names: ``QWEN3_EMBEDDING_GGUF_REVISION`` is
``None`` because no sha of the separate ``-GGUF`` repo has ever been verified,
and the loader refuses an unpinned download. This mode resolves the sha,
downloads the Q4_K_M file at it, benches gguf against the shipping st fp32
backend on the doc's own size curve, reports per-text cosine between the two
backends' vectors, and ends by printing the exact source line to paste into
``aggregator/core/embed.py``:

    AGGREGATOR_ALLOW_MODEL_DOWNLOAD=1 \
        uv run --extra embed-gguf python scripts/measure_embed_rate.py \
        --backend gguf --resolve-and-pin

The revision override is deliberately a HARNESS knob, not an ``embed.py`` one:
the production pin must stay a source constant (see that constant's docstring —
an env-var pin is an in-place mutable knob), while validating a candidate sha
has to happen BEFORE the constant is edited. So the harness injects the
candidate into ``aggregator.core.embed.QWEN3_EMBEDDING_GGUF_REVISION`` for the
lifetime of the load — exercising the exact code path the hardcoded pin will
take — and restores it afterwards.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Mapping

import numpy as np

WORD = "the quick brown fox jumps over the lazy dog and then writes some code "

#: Env-var form of ``--gguf-revision``, for unit files and detached runs. The
#: flag wins when both are set. Bench-only — the production loader never reads
#: it (see the module docstring for why that is a feature).
GGUF_REVISION_ENV = "AGGREGATOR_GGUF_BENCH_REVISION"

#: The probe geometry of the 40 tok/s table in ``docs/embedding-throughput.md``.
#: The gguf bench reuses it verbatim so its numbers land in the same table.
SIZE_CURVE_CHARS = (4000, 2000, 1000, 500, 250)

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure embedding throughput. Bare: the legacy full st sweep "
            "behind docs/embedding-throughput.md. --backend st: the size "
            "curve only. --backend gguf: the size curve on BOTH gguf and st "
            "(st is the fp32 reference the cos-sim and speedup are against)."
        )
    )
    parser.add_argument(
        "--backend",
        choices=("st", "gguf"),
        default=None,
        help="bench this backend; omit for the legacy full sweep",
    )
    parser.add_argument(
        "--gguf-revision",
        default=None,
        metavar="SHA",
        help=(
            "40-hex commit sha of Qwen/Qwen3-Embedding-0.6B-GGUF to load at, "
            f"so a pin can be validated before it is hardcoded; env "
            f"{GGUF_REVISION_ENV} is the fallback"
        ),
    )
    parser.add_argument(
        "--resolve-and-pin",
        action="store_true",
        help=(
            "resolve the -GGUF repo's current sha via huggingface_hub (needs "
            "network + AGGREGATOR_ALLOW_MODEL_DOWNLOAD=1), bench at it, and "
            "print the QWEN3_EMBEDDING_GGUF_REVISION line for embed.py"
        ),
    )
    return parser.parse_args(argv)


def validate_sha(sha: str) -> str:
    """A 40-hex commit sha or nothing.

    Same rule as the constant this feeds: a tag (or ``main``) is repointable
    by the repo owner, which is the thing the pin defends against — so the
    harness refuses to bench one, exactly as ``embed.py`` would refuse to
    hold one.
    """
    if not _SHA_RE.match(sha):
        raise ValueError(
            f"{sha!r} is not a 40-character lowercase hex commit sha; refusing "
            f"to bench it — a tag or branch name is a moving target and could "
            f"not be pinned in aggregator/core/embed.py anyway"
        )
    return sha


def effective_revision(flag_value: str | None, environ: Mapping[str, str]) -> str | None:
    """The revision override to bench at: flag first, then env, validated.

    Validation happens HERE, before any model loads, so a typo'd pin costs
    milliseconds rather than surfacing after a multi-minute download.
    """
    raw = flag_value if flag_value is not None else environ.get(GGUF_REVISION_ENV)
    if raw is None:
        return None
    return validate_sha(raw)


def pin_line(sha: str) -> str:
    """The EXACT source line that closes the pin hole in ``embed.py``.

    Formatted to be a drop-in replacement for the line
    ``QWEN3_EMBEDDING_GGUF_REVISION: str | None = None`` — the unit test
    reads ``embed.py`` off disk and fails if the two ever drift apart.
    """
    return f'QWEN3_EMBEDDING_GGUF_REVISION: str | None = "{validate_sha(sha)}"'


def cosine_stats(ref: np.ndarray, cand: np.ndarray) -> dict:
    """Row-wise cosine similarity between two stacks of vectors.

    True cosine, not a dot product: ``embed_documents`` output is already
    L2-normalized so the two coincide in practice, but a fidelity number must
    not silently depend on that.
    """
    if ref.shape != cand.shape:
        raise ValueError(f"shape mismatch: {ref.shape} vs {cand.shape}")
    ref = ref.astype(np.float64)
    cand = cand.astype(np.float64)
    norms = np.linalg.norm(ref, axis=1) * np.linalg.norm(cand, axis=1)
    norms = np.where(norms == 0, 1.0, norms)
    per_text = (np.sum(ref * cand, axis=1) / norms).tolist()
    return {
        "per_text": per_text,
        "mean": float(np.mean(per_text)),
        "min": float(np.min(per_text)),
    }


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
