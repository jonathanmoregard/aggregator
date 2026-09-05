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

**Revised 2026-08-24.** The rate below (~40 tok/s) held up. The *backfill
estimate* built on it did not: it assumed the embeddable-row count and the mean
body length, and both were wrong. Those two sections now carry counted numbers
from `scripts/embedding_token_bill.py`, which exists so that every figure in
this file is a command a reader can re-run rather than a number they have to
trust. The int8 lever is closed by measurement in the same pass.

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

**This section used to be an estimate built on two assumptions, and both were
wrong in the direction that made the backfill look tractable.** It said ~240k
embeddable rows ("roughly half" of 483k) and offered "≈14 days if the mean body
is nearer 200 tokens" as the optimistic bound. The bill has since been COUNTED
rather than assumed — tokenizer only, no forward passes, over the read-only
snapshot of the live cache — by `scripts/embedding_token_bill.py total`:

| | measured | previously assumed |
|---|---|---|
| rows | 509,411 | ~483k |
| embeddable | **332,007 (65.2%)** | ~240k ("roughly half") |
| chunks | 440,198 | — |
| tokens | **189,453,589** | — |
| mean tokens/chunk | 423 observations, 866 records | ~200 (guessed) |
| **at the measured 40 tok/s** | **54.8 days** | 14-55 days |

So the true figure is the TOP of the old range, not the middle: there is no
14-day case. The 25-30 day estimate this file was written to check is now the
one that looks optimistic, and the reason is the mean chunk — 423 tokens, not
the ~200 that was guessed.

### The 35% that is not embeddable is not a coverage gap

65.2% invites the question "what happens to the other third?", so it is
answered here rather than left to be re-derived. Measured per row, the skipped
177,404 rows break down as:

| bucket | rows | what it is |
|---|---|---|
| `attachment` | 112,385 | structural markers. `body` is `''` and **no other column holds text** — the content was never in the cache to begin with |
| `assistant` | 49,236 | turns whose entire payload was a tool call. The call itself is a separate `tool_use` row, and those are **100% embeddable** |
| `system` | 12,889 | system/meta turns, empty body |
| `tool_result` | 2,665 | empty results |
| `progress` | 228 | progress markers |
| `user` | 1 | one empty prompt |

**Every row that contains text is embedded.** The count of rows whose body is
non-empty but which the chunker rejects is **zero** — there is no content being
silently dropped between the cache and the model. And there is no asymmetry
between the arms: `obs_fts` indexes `body` and nothing else, so an empty-body
observation is invisible to the keyword arm too. A row skipped here was never
findable by any means.

**Records are 100% embeddable** (4,293 of 4,293), because the embedder feeds the
model `subject + "\n\n" + body` for that table — so a TickTick task with a title
and no notes is indexed on its title. That is worth stating because it is easy
to get wrong when counting: measuring the record bill from `body` alone reports
1,242 of them as skipped and undercounts the record tokens by 48k.

The one genuinely open question this leaves is upstream of retrieval: whether
`attachment` observations *should* be carrying text that ingest is not
capturing. That is an ingest question, not an indexing one, and nothing in the
retrieval path can answer it.

### The order is what matters, not the total

54.8 days is the wrong number to plan against, because the backfill runs in a
priority order the user set (`dropbox -> blog -> llm -> claude code`) and the
FTS5 arm serves the whole corpus throughout. What matters is when the vector arm
becomes useful, which is a PREFIX of this table.
`scripts/embedding_token_bill.py by-source`, day columns **cumulative**:

| group | tokens | @40 t/s (measured) | @176 t/s (300M, est) | @530 t/s (33M, est) |
|---|---|---|---|---|
| dropbox | 3.8M | 1.09 | 0.25 | 0.08 |
| substack | 0.2M | 1.15 | 0.26 | 0.09 |
| claude-web | 9.8M | 3.99 | 0.91 | 0.30 |
| sessions | 116.8M | 37.80 | 8.59 | 2.85 |
| subagents | 55.8M | 53.93 | 12.26 | 4.07 |
| github/research/sota-watch/ticktick | 3.0M | 54.80 | 12.46 | 4.14 |

**91% of the 54.8 days is `sessions` + `subagents` — the two sources the user
deliberately ranked LAST.** Everything ranked ahead of them finishes in about 4
days on the model that ships today, unchanged. Only the 40 tok/s column is
measured; the other two are scaled by non-embedding parameter count and are
estimates, quoted to size a decision and not to be planned against.

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

## int8 dynamic quantization is dead on this hardware — measured

This was listed below as "typically 2-4x on CPU". It is not, here. Measured with
`scripts/embedding_token_bill.py bench-int8` over the CACHED weights (torch
2.13, quantized engine `x86`, no download, no new packages):

| chunk size | speedup | cos(fp32, int8) |
|---|---|---|
| 1800 chars (~420 tokens) | **1.01x** | ~1.0 |
| 4000 chars (~804 tokens) | **1.06x** | ~1.0 |

The geometry does not move, which would have been the thing to worry about —
`hybrid.VECTOR_FLOOR_MAX_DISTANCE` is calibrated in this embedding space — but
that only matters if the speedup is real, and 1.06x is not. Dynamic
quantization helps when the model is memory-bandwidth-bound in its `Linear`
layers; at 804 tokens on a 15 W part this workload is compute-bound instead.

**This falsifies the cheap lever and leaves only the expensive one.** The three
open items below were meant to be three independent shots at the problem; two of
them are now closed by measurement, so the model itself is the only remaining
lever on the 40 tok/s.

## gguf Q4_K_M — pending bench

The Q4_K_M backend in `aggregator/core/embed.py` has never produced a number
in this file, and as of 2026-09-04 it cannot: `QWEN3_EMBEDDING_GGUF_REVISION`
is `None` — no sha of the separate `-GGUF` repository has been verified — the
loader refuses an unpinned download by design, and agent sessions on this
machine have no network. What exists is the harness.
`scripts/measure_embed_rate.py --backend gguf` runs the same size curve as
the table at the top (4000/2000/1000/500/250 chars, batch 1) on both gguf and
the shipping st fp32 backend, with token counts from the st tokenizer for
both so the tok/s columns divide cleanly, and reports per-text cosine between
the two backends' vectors — 768-dim, L2-normalized, the same space the
abstention floor is calibrated in.

One human run with network closes the loop, from the repo root:

    AGGREGATOR_ALLOW_MODEL_DOWNLOAD=1 uv run --extra embed-gguf python scripts/measure_embed_rate.py --backend gguf --resolve-and-pin

It resolves the `-GGUF` repo's current commit sha via `huggingface_hub`,
downloads the Q4_K_M file at that sha, benches both backends, and ends by
printing tok/s per backend, the speedup ratio, the cos-sim stats, and the
exact `QWEN3_EMBEDDING_GGUF_REVISION` line to paste over the `= None` line in
`aggregator/core/embed.py`. A candidate sha can also be validated *before*
being hardcoded: `--gguf-revision <sha>` (env
`AGGREGATOR_GGUF_BENCH_REVISION`) benches at a named revision without
touching the constant — the harness injects it into the loader's own pin
path for the load and restores it after.

**Decision rule: adopt gguf for the backfill if it measures ≥2.5× the st
tok/s at this geometry with an acceptable cos-sim/retrieval delta.**
Otherwise the pin stays `None` and st remains the only benched backend. The
int8 section above is the prior to weigh, not a prediction: the same
compute-bound argument applies to any quantization on this part, so the 3-5x
sometimes quoted for Q4 is a claim the bench must earn.

## Open, and the user's to decide

* **A smaller embedding model — now the ONLY lever.** Costs MTEB-Code, which is
  why the current one was chosen. Note the trap: the 20x-class models
  (MiniLM-L6 and friends) have a **256-token maximum sequence** while our chunks
  are ~804 tokens, so they would silently embed the first third of every chunk.
  A model that keeps this chunk geometry (≥804-token window, 768 dims) is
  100-140M parameters, i.e. ~4-5x, not 20x. Any swap is a full re-embed and
  requires re-deriving the abstention floor, since `_QUANTIZATION` and the model
  id are both in the version string.
* **The `gguf` Q4_K_M backend already in the tree** — still needs its repository
  revision pinned (`QWEN3_EMBEDDING_GGUF_REVISION is None`) and verified. Its
  plausible 3-5x overlaps with what int8 was supposed to deliver, and int8
  measured 1.06x, so the same compute-bound argument applies and this should be
  weighed accordingly rather than assumed. The bench harness and the
  one-command human run now exist — see "gguf Q4_K_M — pending bench" above.
* ~~int8 dynamic quantization or an ONNX Runtime export~~ — **closed by
  measurement**, see the section above. ONNX Runtime is untested and would be a
  separate measurement, not a continuation of this one.
