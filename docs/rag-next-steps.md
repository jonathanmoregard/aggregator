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

## What "CI green" does and does not cover

The first CI run on this branch failed one test, and chasing it turned up
something structural worth stating.

**The failure.** `test_reindex_with_nothing_to_delete_does_not_prompt` built a
real `Embedder`, which resolves weights through the HuggingFace cache. It passed
on any machine that had run a backfill and failed in CI, where the cache is
empty and downloads are refused by design. Fixed by stubbing the embedder, as
every sibling on that path already did. Not a production defect: building the
embedder before `migrate()` is deliberate, because the provenance stamp carries
the quantization and chunker version and neither is knowable from the
environment.

**The structural part.** With a cold model cache, **11 tests skip rather than
run** — all of `tests/core/test_embed.py`, `test_embed_query_instruction.py` and
`test_rerank.py`. They skip loudly, with the reason attached, which is the right
call: downloading weights in CI is exactly the failure this repo has a guard
against. But the consequence is that **CI has never executed the embedder or the
reranker**, and a green tick there says nothing about either. Only a local run
against a warm cache covers them.

Two practical consequences. Run the gate locally with `HF_HOME` pointed at an
empty directory when the question is "does this pass in CI" — a warm cache
silently answers a different question, which is how the bug above survived a
green local gate. And run it warm when the question is "does the embedder still
work", because cold skips 11 of the tests that would tell you.

**A known flake, pre-existing.** `tests/core/test_store.py::
test_two_processes_concurrent_writes_succeed` joins two spawned writers with a
fixed `timeout=30`. On a loaded box the child has not finished importing, so
`exitcode` is `None` and the assertion reports a concurrency failure that did
not happen. It predates this branch (last touched by `847ac74`), passes 3/3 when
run idle, and passed in CI and in the warm gate. It conflates "the writers
deadlocked" with "the machine was busy"; worth separating those, in its own
commit.

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

- **The attachment subtypes ingest throws away.** Scoped below — it is a small
  ingest change, not the 22%-of-the-corpus question it first looked like.
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

## The `attachment` rows: keep them, fix 2.5% of them

This started as "22% of the corpus carries no text — is that a coverage gap?"
It is not, and the framing was the error. Recorded here because the number is
alarming and invites the wrong fix.

**Where they come from.** All 116,862 of them are Claude Code session and
subagent JSONL, `origin='claude-code'`, lines with `type: "attachment"`. The
payload sits under a top-level `attachment` key; ingest maps nothing from that
key to `body`, so the cache keeps the row's provenance and drops the content.

**They are edges, not documents.** Measured on the live cache: **115,613 rows
name an attachment row as their `parent_obs_id`, and 37,712 of those are real
messages** — 24,618 `assistant`, 9,725 `user`, 3,127 `tool_use`, 242 `system`.
Deleting the attachment rows severs the thread between those messages and what
preceded them. They carry no `model`, no token counts, no `tool_name`, no
`tool_use_id`: structural nodes with a timestamp, and they cost **~21.5 MB of
~708 MB — 3% of storage for 22.4% of the rows**, because the bodies are empty.

Collapsing them transitively (re-pointing each child at the attachment's own
parent) is possible, and is the wrong trade: it rewrites history and erases the
fact that context *was injected at that point in the turn*, which is precisely
the signal wanted when reconstructing why an agent did something. Not worth 3%
of disk.

**What is actually inside them.** Full scan, all 10,956 transcripts, 0
unreadable, 148,437 attachment lines (`scripts/attachment_payload_census.py
--all`). Not a sample — two sampled runs disagreed on the headline ratio (2.4%
at 400 files, 4.7% at 120), which is why this was run whole:

| `attachment.type` | rows | % | MB raw | mean text |
|---|---|---|---|---|
| `hook_success` | 63,903 | 43.1% | 113.0 | 1.2 KB |
| `hook_additional_context` | 30,288 | 20.4% | 32.8 | 0.6 KB |
| `skill_listing` | 11,654 | 7.9% | **240.6** | 20 KB |
| `deferred_tools_delta` | 8,940 | 6.0% | 36.7 | 3.7 KB |
| `task_reminder` | 8,586 | 5.8% | 7.4 | 0.4 KB |
| `total_tokens_reminder` | 6,972 | 4.7% | 3.5 | 49 B |
| `mcp_instructions_delta` | 5,600 | 3.8% | 9.1 | 1.2 KB |
| `agent_listing_delta` | 3,115 | 2.1% | 12.1 | 3.4 KB |
| `async_hook_response` | 1,875 | 1.3% | 1.3 | 213 B |
| **`queued_command`** | 1,433 | 1.0% | 5.9 | 3.6 KB |
| **`edited_text_file`** | 864 | 0.6% | 5.7 | 5.8 KB |
| **`nested_memory`** | 416 | 0.3% | 6.5 | 15.3 KB |
| **`file`** | 388 | 0.3% | 2.4 | 5.7 KB |
| **`invoked_skills`** | 54 | 0.0% | 1.1 | 21.1 KB |
| **`plan_file_reference`** | 3 | 0.0% | 0.0 | 6.3 KB |
| *(20 further plumbing subtypes)* | | | | |

**97.9% is harness plumbing repeated near-verbatim across every session.**
`skill_listing` alone is **240.6 MB — 11,654 copies of the same catalogue**, and
the single largest payload class by bytes in the whole set. Embedding that bulk
would not close a coverage gap, it would **poison the vector arm**: tens of
thousands of near-duplicate chunks, after which semantic queries start returning
the skills listing. Near-duplicate contamination is a known dense-retrieval
failure and this corpus is unusually rich in it.

**The actual defect, and it is small.** The six content-bearing subtypes are
**3,155 rows — 2.1% of attachments**, and their payload is discarded rather than
written to `body`. A pasted file, an edited buffer, a memory file, a referenced
plan: currently unfindable by *either* arm. `plan_file_reference` is in on
payload rather than frequency — 3 rows, 6.3 KB each.

That is the change worth making, in `aggregator/sources/sessions.py`, as its own
ingest commit — deliberately **not** folded into the retrieval branch, whose
diff is settled and green. Everything else stays exactly as it is: empty
structural nodes, excluded from both arms, doing their job.

The census refuses to bucket a subtype it does not recognise, on the same
principle as the FTS5 MATCH scanner. It earned that on its first full run,
surfacing five subtypes no sample had shown — `plan_file_reference` (promoted to
content-bearing) plus `auto_mode`, `plan_mode`, `plan_mode_exit` and `directory`
(mode markers, 5–195 bytes, classified as plumbing). Re-run it before writing
the ingest filter; new harness releases mint new subtypes.

**One loose end, deliberately not chased here.** The scan found 148,437
attachment lines on disk against 116,862 rows in the cache — a ~31.6k gap. That
is an ingest-coverage question (transcripts not yet ingested, or `src_hash`
dedup), not a retrieval one, and it belongs with the ingest commit above.
