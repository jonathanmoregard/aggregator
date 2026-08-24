<!-- written 2026-08-24, at the moment PR #1 was opened -->
# RAG: what happens after the merge

This file exists because merging the hybrid-retrieval PR indexes **nothing**,
and that is easy to assume otherwise. It is the ordered list of what remains,
with the measurements each step depends on, so the work can be picked up cold.

## The state this file was written in

- `feat/rag-hybrid-v2` is open against `main` as PR #1. Gate green on a settled
  tree at `e2f57e8`: 2055 passed / 0 failed (612 s), `ruff` clean,
  `aggregator-embed-unit-hygiene` force-rebuilt to "OK", `nix flake check` rc 0.
- The **deployed** aggregator is a store path pinned to `65fdec3` in
  nixos-config's `flake.lock`. That build has no vector code at all: no `embed`
  subcommand, no vec DDL, `SCHEMA_VERSION = 4`.
- There is **no `aggregator-embed` systemd unit on the machine**. Only the
  30-minute ingest timer.
- The live cache is `user_version 4`, vec tables present but **empty**, no
  `embedding_versions` row, zero vectors.

So nothing in the vector arm can be exercised against real data yet, and two
open questions below are blocked on that rather than on any code in the PR.

## Embedding model: decided

**Ship as-is on `Qwen3-Embedding-0.6B`** (decided 2026-08-24). No swap.

The alternative was `embeddinggemma-300m` — ~12.5 days instead of 54.8, 768-dim
native so no schema change, 2048-token window — but it costs a license-gated
HuggingFace download and a pre-registered A/B, and the argument that settles it
is ordering, not totals: **91% of the 54.8 days is `sessions` + `subagents`,
the two sources ranked last.** Everything ahead of them finishes in ~4 days on
the model already in the cache.

The ~33M class was rejected on measurement, not taste: a 512-token context
truncates our chunks (mean 423 tokens for observations, 866 for records), so it
would silently embed the first fraction of the large ones.

## The chain, in order

1. **Merge PR #1.** Repo owner's click.
2. **nixos-config worktree:** bump `aggregator-src` to the new `main` rev.
3. **nixos-config:** add the embed timer (`aggregator embed --catchup`) and the
   human-triggered `--seed-models` unit. `--seed-models` is the only path that
   fetches the reranker weights and it needs
   `AGGREGATOR_ALLOW_MODEL_DOWNLOAD=1`; without it, it reports what is missing
   and exits non-zero.
4. **VM gate + nixos-config PR + merge**, then auto-deploy lands it. The embed
   worker adds a second writer against `cache.db`; the store is built for that
   (WAL, `busy_timeout = 30000`), but the units are new and the VM gate is where
   that gets asserted rather than hoped.
5. **Confirm the backfill is really running** — `aggregator status` reports
   per-source progress, and `empty` is never folded into `complete`.

Backfill order is a standing directive:
`dropbox → substack → claude-web (+chatgpt) → sessions → subagents → the rest`,
each finished before the next begins. Cumulative at the measured 40 tok/s:
dropbox ~1.1 days, + substack ~1.15, + claude-web ~4.0, + sessions 37.8,
+ subagents 53.9, everything 54.8.

## Post-processing, once the index is populated

**The day `dropbox` completes (~1.1 days of CPU) is the first moment any of this
is possible.**

- **Calibrate `VECTOR_FLOOR_MAX_DISTANCE = 1.00` against a real populated
  index.** It never has been. The present evidence is Monte Carlo over a
  modelled window plus spot checks that embed real cache text on demand. Do a
  freeze/run pair.
- **Run the eval harness in mcp mode.** It refuses a cold cache with exit 2
  rather than scoring the FTS5 fallback and calling it hybrid, so mcp-mode drift
  becomes measurable here for the first time.
- **Measure whether RRF equal weighting is wrong for this corpus.** A
  620k-chunk benchmark has BM25 beating dense on named entities 0.961 vs 0.712
  and on error codes 0.974 vs 0.681, calls the gap structural, and finds hybrid
  *losing* to BM25-alone on both. We have an 86-query golden set to settle it
  on, which is the part nobody else in the surveyed cohort could do.
- **Revisit sqlite-vec ANN (`rescore`)** with a real index under it.
  `rescore(quantizer=bit, oversample=8)` benchmarks ~6x at recall@10 0.988, but
  every ANN index is pre-release (v0.1.10-alpha.4). Deliberately out of scope
  for the PR; it is a dependency decision, not a retrieval one.

## Open questions that do not depend on the index

- **The attachment gap.** 112,385 rows — 22% of the corpus — carry no text in
  any column, verified by dumping full rows. Nothing is *lost* between cache and
  model (rows with a non-blank body that the chunker rejects number zero, and
  the keyword arm is equally blind to them since `obs_fts` indexes `body` and
  nothing else). The question is upstream: **should ingest be capturing
  attachment content at all?** Retrieval cannot answer it.
- **The FTS5 tokenizer change is one decision, not two.**
  `tokenize="unicode61 tokenchars '-'"` must land *with* `-` added to the
  sanitizer whitelist. Measured: with `-` in the index but not the whitelist,
  `wifi-6E` matches "the wifi is up and the 6E band" but not "we got wifi-6E
  working", and `read-only` matches nothing at all. A cheaper intermediate worth
  measuring first is `-` in the whitelist only, no reindex — every hyphenated
  query raises today, so nothing previously-working can break.
- **Hand-labelling ~30 golden queries.** A human step. Blocks nothing: the
  labelled metrics report "no labels" and store SQL NULL rather than a
  misleading number.
- **The ~12 pre-existing read-path `sqlite3.OperationalError` swallows in
  `store.py`.** This diff added none. Whether they deserve their own branch is
  undecided.
