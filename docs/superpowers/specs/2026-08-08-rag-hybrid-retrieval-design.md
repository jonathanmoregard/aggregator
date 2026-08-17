# RAG / Hybrid Retrieval — design spec (2026-08-08)

## Purpose

Add local semantic retrieval to the aggregator alongside the existing FTS5
index. Query surface stays one tool (`aggregator_search_memory`), one DSL.
Callers get better recall on conceptual queries ("that thing about voting
theory") without losing exact-match queries ("session:abc123").

Corpus at spec time: **359,068 observations** (avg 1KB, p90 1.6KB, p99
25KB, ~9K > 8KB), **1,032 records** (avg 7.4KB), **7,911 sessions**.
Realistic vector count after filtering empty-body rows and chunking p99
outliers: **~200-250K vectors**.

Reference research reports:
- `~/Repos/research-agent/reports/8f2d9fec0fd140848706ad488d988e31.md` —
  SOTA embedder/vector-store landscape Q3 2026.
- `~/Repos/research-agent/reports/0bf2ced7a1aa4d5d861bf05d610c24f4.md` —
  Head-to-head ranking of gte-modernbert-base / Qwen3-Embedding-0.6B /
  nomic-embed-text-v1.5 / BGE-small-v1.5. Includes the load-bearing
  finding that `bge-reranker-v2-m3` **degrades** Qwen3 English retrieval
  per vendor eval (MTEB-R 61.82 → 57.03).

## Non-goals (YAGNI, documented)

- **LanceDB / Qdrant / DuckDB VSS.** sqlite-vec fits 250K vectors in the
  existing cache.db with zero index maintenance. Revisit if the corpus
  crosses 1M or brute-force scan p95 exceeds 500ms on target hardware.
- **ANN indexing.** sqlite-vec exact brute-force. Adds when
  sqlite-vec ships ANN (issue #25) or corpus scale demands it.
- **Convex-combination fusion.** RRF k=60 is the SOTA cheap floor.
  Upgrade to tuned convex combination requires 50-100 labeled query
  pairs; deferred until a hit-quality eval exists.
- **ColBERT / late-interaction reranking.** Cross-encoder rerank
  (Qwen3-Reranker-0.6B) is the practical ceiling for v1.
- **API embedders** (OpenAI, Voyage, Cohere). Aggregator is local-first
  and never egresses; runtime dollar cost stays at $0.
- **Auto-tuning K, alpha, k parameters.** Fixed constants for v1
  (top-K=50 per arm, RRF k=60).
- **Query-side answer synthesis.** Aggregator remains a retriever; the
  MCP calling agent is the LLM.
- **Multi-language embeddings.** Corpus is English-dominant; add
  multilingual embedder (bge-m3, arctic-l-v2) only if non-English
  material shows up in ingest.

## Architecture

Two new subsystems, three new files, one migration.

1. **Embed worker** (`aggregator/core/embed.py`, new)
   - Loads Qwen3-Embedding-0.6B via `sentence-transformers` (safetensors)
     or `llama-cpp-python` (GGUF q4). Selectable via env var; GGUF is the
     default for smaller RAM footprint.
   - Applies Qwen3's mandatory prefix template:
     - Document embed: raw text, no prefix.
     - Query embed: `Instruct: Given a search query, retrieve relevant passages\nQuery: <text>` — encoded in one wrapper function, never touched by callers. Load-bearing (1-5% retrieval loss if omitted, per Qwen model card).
   - Matryoshka truncation from native 1024d → **768d** stored. Halves
     scan cost at negligible quality loss (Qwen3 supports MRL 32-1024d).
   - Batches with `batch_size=32`, `normalize_embeddings=True` (unit L2
     norm → cosine ≡ dot product).

2. **Vector store** (schema addition in `aggregator/core/store.py`)
   - New table `vec_observations` (sqlite-vec `vec0` virtual table),
     shape: `obs_id TEXT PRIMARY KEY, embedding float[768]`.
   - New table `vec_records` (same shape, `stable_id` primary key).
   - **NULL-vector watermark**: `observations.embedding_state` TEXT NULL
     column added. Values: `NULL` = not embedded, `'ok'` = embedded,
     `'skip'` = body too short/empty, `'error'` = last attempt failed
     (with retry backoff logic in the worker). Same column on `records`.
   - Chunking rule: observation body > 8000 chars → split into ≤4000
     char windows with 400-char overlap (paragraph-boundary aware). One
     `vec_observations` row per chunk with synthetic
     `obs_id = f"{original}:{n}"`. `count(*)` at spec time gives us
     ~9,329 rows crossing the threshold; total chunk count ≈ 360K.

3. **Background indexer** (`aggregator/cli.py` subcommand + nix timer)
   - `aggregator embed --catchup` — SELECT rows WHERE embedding_state IS
     NULL, embed in batches, UPDATE state to 'ok' / 'skip' / 'error'.
     Idempotent, resumable, single-process (holds file lock via
     `flock(cache.db + '.embed.lock')`). Wall-time budget for initial
     backfill: **~3.5 hours** single-thread CPU for 250K chunks.
   - `aggregator embed --once` — one batch (e.g. 500 rows) then exits.
     Used by the systemd timer for incremental catchup after each
     ingest cycle.
   - Nix module (`nix/aggregator.nix`) adds
     `services.aggregator.embed` timer, fires 5 min after each ingest
     tick, `Type=oneshot`, `nice=19` + `IOSchedulingClass=idle` so it
     never contends with query load.

4. **Query fusion** (`aggregator/core/hybrid.py`, new + hooks in
   `aggregator/mcp.py`)
   - `hybrid_retrieve(store, ast, k=50) -> list[(obs_id, rrf_score)]`:
     runs FTS5 arm (existing `_fts_obs_ids` / `_fts_ids`) + vector arm
     (new `_vec_obs_ids(query_embedding, k)`) → both return top-K
     ordered lists → RRF fusion `1 / (60 + rank)` per doc → return
     top-K merged.
   - Called from `_query_sessions_path` and `_query_records_path` when
     `ast.text` is present. When only DSL filters present (no text),
     bypass entirely (RRF over an empty query is undefined).
   - Backwards-compat: when `vec_observations` is empty (fresh cache
     before backfill), fall through to FTS5-only. Result shape
     unchanged; RRF is invisible.

5. **Reranker** (`aggregator/core/rerank.py`, new, flag-gated)
   - Qwen3-Reranker-0.6B loaded on demand (lazy import).
   - `rerank=True` param added to `aggregator_search_memory`. When set,
     take fused top-20 → score each `(query, doc_body[:2048])` pair
     jointly → reorder → return.
   - **Do NOT use bge-reranker-v2-m3** — Qwen's own eval shows it
     degrades Qwen3-embedded results on English (MTEB-R 61.82 → 57.03).
     Qwen3-Reranker-0.6B is the vendor-tested-safe pairing.

## Data flow

Ingest path (additive to existing):

```
Source.iter_entities()
  → store.upsert_entities(entities)   [existing, unchanged]
    → INSERT observations rows
    → embedding_state = NULL (default)
Later, on embed timer:
  aggregator embed --once
    → SELECT obs_id, body FROM observations WHERE embedding_state IS NULL LIMIT 500
    → chunk(body) if len > 8000
    → Qwen3-Embedding-0.6B(chunks) → float[768][]
    → INSERT INTO vec_observations(obs_id, embedding) VALUES ...
    → UPDATE observations SET embedding_state='ok' WHERE obs_id IN (...)
```

Query path (additive; existing FTS path preserved):

```
aggregator_search_memory("quadratic voting", rerank=False)
  → parse DSL → AST
  → ast.text present + vec_observations non-empty:
      hybrid_retrieve(store, ast, k=50)
        → FTS5 arm: obs_fts MATCH "quadratic voting" → top 50 obs_ids
        → vector arm: qwen3_embed(query) → sqlite-vec KNN(vec_observations) → top 50 obs_ids
        → RRF k=60 merge → top 50 obs_ids fused
      → downstream: existing session-projection logic (roots, cards)
  → optional rerank=True:
      → fused top-20 → Qwen3-Reranker-0.6B(query, doc) → reorder → top-K
```

Zero-latency-blocking guarantee (round-1 invariant): the query path
NEVER waits for backfill. Rows with `embedding_state IS NULL` still
surface via the FTS5 arm of the fusion. Backfill catches up eventually;
queries degrade gracefully to FTS5-only until then.

## Schema migration (v4 → v5)

`SCHEMA_VERSION = 5`.

**Renumbered 2026-08-17.** This spec was written on 2026-08-08 against a
schema that was then at v3, so the RAG schema was numbered 4. `main` has
since shipped its own v4 — the incremental-ingest watermarks
(`ingest_state`), the dead-letter table and `poison_faults` — so the RAG
schema takes 5. Read every "v3 → v4" below as "v4 → v5"; the migration
shape is unchanged, and it must not disturb any v4 artefact.

Additive, no rebuild required (parity with v2→v3 pattern):
- `ALTER TABLE observations ADD COLUMN embedding_state TEXT` (nullable,
  default NULL — existing rows count as "not embedded yet").
- `ALTER TABLE records ADD COLUMN embedding_state TEXT` (same).
- `CREATE INDEX obs_embedding_state ON observations(embedding_state)`
  (queried by embed worker to find backlog).
- `CREATE INDEX rec_embedding_state ON records(embedding_state)`.
- `CREATE VIRTUAL TABLE vec_observations USING vec0(obs_id TEXT PRIMARY KEY, embedding float[768])`.
- `CREATE VIRTUAL TABLE vec_records USING vec0(stable_id TEXT PRIMARY KEY, embedding float[768])`.

Ensure-column probe pattern (`_ensure_sessions_origin_column`)
generalises to `_ensure_column(table, column, ddl)` so migrations remain
idempotent under half-applied state.

`Store.rebuild_all()` gains the two vec tables in `_DROP_ALL`. Nothing
in the vec tables is source-of-truth (embeddings are derived from
observations.body); a rebuild triggers `--catchup` on next embed timer
tick.

## Components (delta from existing)

| File | Change |
|---|---|
| `aggregator/core/store.py` | v4 DDL; `embedding_state` column + index; `vec_observations` / `vec_records` DDL; `_vec_obs_ids(embedding, k)` and `_vec_record_ids(embedding, k)` reader methods; `select_unembedded(kind, limit)` writer-worker helper; `mark_embedded(kind, ids, state)` batch UPDATE. |
| `aggregator/core/embed.py` **new** | `Embedder` class: `embed_query(str) -> np.ndarray`, `embed_documents(list[str]) -> np.ndarray[N,768]`. Handles Qwen3 prefix, MRL truncation, GGUF vs safetensors loader selection. Module-level singleton with lazy load. |
| `aggregator/core/hybrid.py` **new** | `hybrid_retrieve(store, ast, k) -> list[(id, score)]` — the RRF fusion. |
| `aggregator/core/rerank.py` **new** | `Reranker` class: `score(query, docs) -> np.ndarray[N]`. Qwen3-Reranker-0.6B. Lazy load. |
| `aggregator/mcp.py` | `aggregator_search_memory` gets `rerank: bool = False` param. `_query_records_path` and `_query_sessions_path` route text-bearing queries through hybrid retriever; empty vec tables → passthrough to FTS. Result shape unchanged. |
| `aggregator/cli.py` | `aggregator embed [--catchup|--once] [--source records|observations|both]` subcommand. |
| `nix/aggregator.nix` | New `services.aggregator.embed` timer, oneshot, nice/idle-scheduled. Package `sqlite-vec`, `sentence-transformers` OR `llama-cpp-python`, model download step. |
| `flake.nix` | Add sqlite-vec + embedder deps to devShell + package output. |
| `pyproject.toml` | Add `sqlite-vec>=0.1.6`, `sentence-transformers>=3.0`, `numpy>=2.0`. Optional-dep `[embed-gguf] = ["llama-cpp-python>=0.3"]`. |

## Concurrency & durability

- **Ingest vs embed vs query** all touch cache.db. Existing WAL +
  busy_timeout=30s covers ingest ↔ embed. Query is read-only. Embed
  worker also takes an OS-level `flock(cache.db + '.embed.lock')` so
  only one embedder runs at a time (avoids double-work in the
  catchup+incremental overlap window).
- **Crash recovery**: embed batches are atomic per-transaction. If the
  process dies mid-batch, `embedding_state` stays NULL on the affected
  rows and the next tick retries.
- **Retry backoff**: `embedding_state='error'` rows retried once per
  hour, max 3 attempts. After that: state='skip' with `extra` JSON
  logging the last error class. Prevents runaway retries on a
  systematically bad row (e.g., encoding issue).
- **Read-only MCP invariant preserved**: the MCP surface never calls
  embed writes. The embed worker runs from the CLI path exclusively.

## Security

Unchanged. Vectors are 768 float32 values; they don't leak content
beyond what's already in cache.db. Scrub-on-write still happens on the
`observations.body` column (embedder reads from that scrubbed body, so
embeddings never encode PII the scrubber would have removed). No new
network egress (all local). No new secrets.

Content-scan on the embedder/reranker model artifacts at first load
(SHA256 pin in the Nix package derivation → tamper-evident).

## Testing strategy

- **Unit** — `tests/core/test_embed.py`: prefix template applied on
  query only; MRL truncation preserves L2 norm to 1e-3;
  `embed_documents` deterministic across calls with same input.
- **Unit** — `tests/core/test_hybrid.py`: RRF k=60 arithmetic on
  synthetic top-K lists; empty vector arm falls through to FTS-only;
  empty FTS arm falls through to vector-only; both empty → empty
  result.
- **Unit** — `tests/core/test_rerank.py`: `rerank=True` reorders vs
  `rerank=False`; identical query returns identical order across runs.
- **Integration** — `tests/test_mcp_hybrid.py`: seed cache with 50
  synthetic obs rows, run backfill in-process, assert
  `aggregator_search_memory` returns the semantically-relevant rows
  ahead of purely-lexical matches on a semantic query.
- **Migration** — `tests/core/test_store_v4_migration.py`: v3 → v4
  ALTER TABLE + vec table creation idempotent; existing rows get
  `embedding_state=NULL`; capabilities response reports schema v4.
- **Non-blocking guarantee** — `tests/test_mcp_hybrid.py::test_query_before_backfill`:
  spin up store with 100 unembedded obs, run text query, assert results
  return via FTS-only path (mode='sessions', total>0) without waiting
  for embed.
- **Ingest→embed→query e2e** — `tests/test_e2e_rag.py`: ingest a small
  fixture, run `aggregator embed --catchup`, run query, assert vector
  hits appear.

Gate criteria: `pytest` all green + `ruff check .` clean +
`ruff format --check .` clean.

## Acceptance criteria

1. `SCHEMA_VERSION = 5` and v4→v5 migration is idempotent and additive
   (no rebuild for existing users), and leaves `ingest_state`, the
   dead-letter table and `poison_faults` untouched.
2. `aggregator embed --catchup` embeds 100% of unembedded rows on a
   test fixture; second invocation is a no-op.
3. `aggregator_search_memory("<semantic query>")` returns fused results
   with at least one hit that FTS-only would rank outside top-10 (unit
   test with fixture proving the fusion adds recall).
4. `aggregator_search_memory` with an unembedded cache returns FTS-only
   results without error and without blocking.
5. `rerank=True` reorders results deterministically vs `rerank=False`
   on the same query.
6. Query latency p95 on the target laptop stays under 500ms for
   `fields=summary` on a hydrated cache (measured: `time
   aggregator query "test query"` × 20 runs).
7. Backfill wall-time on the actual dellan cache (~250K vectors)
   completes in under 6 hours (upper bound; expected ~3.5h).
8. `nix/aggregator.nix` embed timer is declaratively packaged; CI
   `flake check` passes.
9. All existing tests still pass (backwards compatibility).
10. `aggregator_capabilities` reports `"schema_version": 4` and a new
    `"vector_index": {"observations_embedded": N, "records_embedded": M}`
    key for observability.

## Rollout

1. Land schema migration + embed worker (non-blocking, no query-path
   changes yet). Ship, run backfill overnight.
2. Land hybrid retriever behind a feature flag env var
   `AGGREGATOR_HYBRID=1` (default off). Verify semantic queries work
   on a hydrated cache.
3. Flip default to on. Ship.
4. Land reranker (opt-in via `rerank=True`). Ship.

Steps 1-3 are one PR each. Step 4 can bundle with step 3 if the code
review comes together cleanly.

## Open questions

None load-bearing. Chunking parameters (8000 threshold, 4000 window,
400 overlap) are reasonable defaults from the research report; tune
after v1 lands if a specific corpus subset shows relevance loss.

## Out of scope for v1 (revisit later)

- Query-log-driven convex combination weight tuning.
- HNSW ANN when sqlite-vec ships it.
- Multilingual embedder swap (bge-m3, arctic-l-v2).
- Web UI for query-latency observability.
- Per-source embedding-model choice (e.g. code-specific embedder for
  github source).
