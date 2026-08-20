# Embedding throughput on this machine

Measured 2026-08-21 on `feat/rag-hybrid-v2`. This file exists because the
project's own backfill estimate — 25-30 days of continuous CPU for ~483k rows —
was ~100x worse than the arithmetic in the SOTA research report, and a 100x gap
is normally a bug. It was measured rather than argued. **The estimate was
right.** The gap is real, and it is not a bug in this pipeline.

Everything below is wall-clock, from the model already resident in the local
Hugging Face cache (`Qwen/Qwen3-Embedding-0.6B` @ `97b0c614…`, 1.2 GB, present
before this work started — nothing was downloaded). Probe:
`scripts/measure_embed_rate.py`.

## Hardware

| | |
|---|---|
| CPU | 13th Gen Intel Core i7-1365U — 15 W laptop part, 2 P-cores + 8 E-cores |
| SIMD | AVX2. **No AVX-512** (disabled on Raptor Lake consumer parts) |
| RAM | 30 GiB |
| torch threads | 10 (default) |
| Backend | `AGGREGATOR_EMBED_BACKEND=st` — sentence-transformers, **fp32** |
| Model | Qwen3-Embedding-0.6B, MRL-truncated to 768 dims, L2-normalized |

## The number

**~40 tokens per second.** Flat.

| chars/chunk | tokens/chunk | batch | wall (s) | ms/chunk | tokens/s |
|---|---|---|---|---|---|
| 4000 | 804 | 1 | 20.08 | 20082 | 40.0 |
| 2000 | 404 | 1 | 10.06 | 10058 | 40.2 |
| 1000 | 204 | 1 | 4.51 | 4511 | 45.2 |
| 500 | 104 | 1 | 3.11 | 3111 | 33.4 |
| 250 | 54 | 1 | 1.18 | 1184 | 45.6 |

At the chunker's own geometry (`chunk-4000-400`, i.e. 4000 characters ≈ 804
tokens) **one chunk costs ~19-20 seconds**.

## Batching does not help

This was the leading hypothesis for the gap — the worker calls
`embed_documents` once per ROW, so the model sees a batch of about 1. Measured,
it is worth **under 10%**:

| chars | tokens | batch | ms/chunk | tokens/s | note |
|---|---|---|---|---|---|
| 4000 | 804 | 1 | 20082 | 40.0 | clean |
| 4000 | 804 | 2 | 18845 | 42.7 | clean |
| 4000 | 804 | 4 | 18461 | 43.6 | clean |
| 4000 | 804 | 8 | 18360 | 43.8 | contended |
| 4000 | 804 | 16 | 24418 | 32.9 | contended |
| 2000 | 404 | 8 | 9951 | 40.6 | contended |
| 2000 | 404 | 16 | 9700 | 41.6 | contended |
| 2000 | 404 | 32 | 15898 | 25.4 | contended |

**Honesty about the method:** the test suite was running on the same 12 threads
for everything marked "contended", so those absolute numbers are depressed and
the apparent regression past batch 8 is **not** attributable — it may be
scheduling rather than memory pressure. The clean points are enough for the
conclusion: batch 1 → 4 buys 9%, and the curve is already flat there.

So the worker's per-row call is **not** the bug it looked like. Enlarging it
would have bought ~9% at best while making one bad row cost a whole batch of
crash attribution — the per-row `claim_embed_row` is what lets a segfault be
blamed on the row that caused it. It was left alone deliberately.

## Threading is not it either

A 2 P-core + 8 E-core hybrid part is exactly the shape that mis-schedules, so
the sweep was run. It scales, which is the only conclusion it can support — the
whole sweep was contended, so the absolute values are depressed:

| torch threads | tokens/s (contended) |
|---|---|
| 2 | 11.6 |
| 4 | 16.7 |
| 6 | 12.2 |
| 12 | 20.8 |
| 10 (default) | 40.0 (clean, from the size curve above) |

(The 6-thread point is out of order because the contention was not constant —
another reason not to read anything into these beyond the trend.)

The cores are being used. There is no thread-count setting that recovers a
factor of 100.

The other two candidates measure as noise against 19 seconds: the per-row
`claim_embed_row` / `release_embed_claim` commits are sub-millisecond on a WAL
connection with `synchronous=NORMAL`, and the model is loaded once per process
(3.5 s), not per call.

## Where the ~100x actually goes

The report's arithmetic is 45 ms per 512-token chunk on "an ordinary x86 CPU",
i.e. **~11,400 tokens/s**. We measure **~40**. That is 285x, not 100x — and the
report flags its own source as a single blog post, so it should be read as an
order of magnitude, not a target.

Decomposed:

1. **Model size — the dominant term, and a deliberate choice.**
   Qwen3-Embedding-0.6B has ~0.44B non-embedding parameters. A forward pass is
   roughly `2 · params · tokens` FLOPs, so 11,400 tokens/s would need
   ~10 TFLOP/s of fp32 — which no laptop CPU does, with or without a bug. At
   11,400 tokens/s the report's figure implies a MiniLM-class model of ~20-30M
   parameters (~0.5 TFLOP/s, entirely ordinary for a CPU). We are ~20x larger
   **on purpose**: 75.41 MTEB-Code is why this model was picked, and this
   corpus is saturated with identifiers.
2. **Hardware.** A 15 W two-P-core laptop part with AVX2 only. Effective
   throughput here works out at ~35 GFLOP/s for this workload.
3. **Precision.** fp32. The `gguf` (Q4_K_M) backend exists but its repository
   revision is unpinned (`QWEN3_EMBEDDING_GGUF_REVISION is None`) and it has
   never been benchmarked here, so it is not a number this file can quote.

None of the three is a defect in the ingest path. **Criterion F's outcome is
therefore "documented with the real number and the reason", not "fixed".**

## What that means for the backfill

At ~19 s per full-size chunk and ~40 tokens/s:

* ~483k observation rows, of which roughly half have empty or whitespace-only
  bodies and are marked `skip` without touching the model;
* so ~240k embeddable rows, at ≥1 chunk each;
* **≈ 55 days** if every row fills a 4000-character chunk, **≈ 14 days** if the
  mean body is nearer 200 tokens.

The project's 25-30 day figure sits inside that range. It was not a bad
estimate and there is no missing 100x to recover.

## What was actually made cheaper

Since the encoder's rate is fixed, the only lever is *not calling it*. Three
changes, all from criterion E:

* **`content_sha256` reuse.** A chunk whose exact bytes have already been
  embedded under this model is copied, not recomputed. Chat transcripts repeat
  themselves constantly — quoted text, re-pasted tool output, the same file
  read twice — and each repeat is ~19 seconds not spent. This also carries
  vectors across a re-chunk.
* **A source rebuild no longer re-embeds.** The embedding index is keyed on
  `(chunk_id, model)` and is not touched when a row is rewritten with identical
  content, so `ingest --rebuild` costs zero encoder time. Under the previous
  `embedding_state` column it returned ~483k rows to the backlog — a repeatable
  multi-week bill, found by review rather than by anyone noticing.
* **A model change no longer deletes anything.** The previous model's vectors
  keep their own key and go on serving while the new model's index is built
  behind `completed_at`. The old shape made a model swap a delete plus a
  multi-week outage, which is why three review rounds were spent building
  consent gates in front of the delete instead.

## Open, and the user's to decide

Not taken here, because each is a dependency or quality decision rather than a
bug fix, and all three have the right order of magnitude:

* a smaller embedding model (the 20x term) — costs MTEB-Code, which is why this
  one was chosen;
* int8 dynamic quantization or an ONNX Runtime export — typically 2-4x on CPU,
  and would change the version string, so it is a full re-embed to adopt;
* the `gguf` Q4_K_M backend already in the tree — needs its repository
  revision pinned and verified first.
