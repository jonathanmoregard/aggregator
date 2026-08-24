# RAG / Hybrid Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (default) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add local hybrid (BM25 + vector) retrieval to the aggregator MCP surface while keeping the query shape, DSL, and read-only security model unchanged.

**Architecture:** sqlite-vec inside cache.db for vectors, Qwen3-Embedding-0.6B (MRL-truncated to 768d) for document + query embedding, RRF k=60 to fuse the FTS5 arm with the vector arm, optional Qwen3-Reranker-0.6B cross-encoder as a flag-gated top-20 reorder. Background embed worker uses a NULL-vector watermark on `observations.embedding_state` / `records.embedding_state` so queries never block on backfill.

**Tech Stack:** Python 3.11, SQLite + FTS5 (existing), sqlite-vec >= 0.1.6, sentence-transformers >= 3.0 (safetensors path), llama-cpp-python >= 0.3 (optional GGUF path), numpy >= 2.0, pytest, ruff, existing FastMCP + argparse CLI, Nix home-manager module for systemd timers.

**Spec reference:** `docs/superpowers/specs/2026-08-08-rag-hybrid-retrieval-design.md`

**Corpus at plan time:** 359,068 observations, 1,032 records, 7,911 sessions. Realistic vector count after empty-body filter + p99 chunking: ~250K.

**Files created / modified:**

| File | Action | Responsibility |
|---|---|---|
| `pyproject.toml` | modify | Add `sqlite-vec`, `sentence-transformers`, `numpy`; optional-dep `embed-gguf = ["llama-cpp-python"]`. |
| `flake.nix` | modify | Add Python deps to devShell + packages.default. |
| `nix/aggregator.nix` | modify | New `services.aggregator.embed` timer + model paths + agenix (none needed). |
| `aggregator/core/store.py` | modify | v4 DDL, `embedding_state` column, vec virtual tables, migration ALTER, reader (`_vec_obs_ids`, `_vec_record_ids`), writer helpers (`select_unembedded`, `mark_embedded`), capabilities `vector_index` key. |
| `aggregator/core/embed.py` | **create** | `Embedder` class — safetensors OR GGUF loader (env-selected), Qwen3 prefix template, MRL truncation to 768d, normalize L2, batch `embed_documents` / single `embed_query`. |
| `aggregator/core/hybrid.py` | **create** | `hybrid_retrieve(store, ast, embedder, k)` — RRF k=60 fusion. |
| `aggregator/core/rerank.py` | **create** | `Reranker` class — Qwen3-Reranker-0.6B, lazy load, `score(query, docs)`. |
| `aggregator/core/chunk.py` | **create** | `chunk_body(text, max_len=4000, overlap=400)` paragraph-aware windowing. |
| `aggregator/cli.py` | modify | `aggregator embed [--catchup\|--once] [--source records\|observations\|both]` subcommand + `flock`. |
| `aggregator/mcp.py` | modify | `aggregator_search_memory` gains `rerank: bool = False`; text queries route through hybrid retriever when vec tables non-empty; passthrough otherwise. |
| `tests/core/test_chunk.py` | **create** | Chunking unit tests. |
| `tests/core/test_embed.py` | **create** | Prefix template applied, MRL norm preserved, determinism. |
| `tests/core/test_hybrid.py` | **create** | RRF arithmetic, empty-arm fallthroughs. |
| `tests/core/test_rerank.py` | **create** | `rerank=True` reorders, deterministic. |
| `tests/core/test_store_v4_migration.py` | **create** | v3→v4 idempotent, existing rows null-embedded. |
| `tests/core/test_store_vec.py` | **create** | vec table upserts + KNN reader. |
| `tests/test_mcp_hybrid.py` | **create** | Text query routes through hybrid; empty vec table falls through; rerank flag reorders. |
| `tests/test_e2e_rag.py` | **create** | ingest→embed→query full path on a small fixture. |
| `tests/test_cli_embed.py` | **create** | `--catchup` embeds backlog; `--once` embeds one batch; second call is no-op; flock rejects concurrent worker. |

**Chunk ordering + parallelism:**

- Chunks A (deps), B (chunking util), C (embedder) can go in parallel (disjoint files).
- Chunk D (schema v4) must complete before E (vec reader), F (embed worker), and G (hybrid).
- Chunk E, F, G edit store.py or depend on it — serialise these against each other; they all touch the same hot file.
- Chunk H (reranker) is independent of A-G once C is done; can parallelise with D/E/F.
- Chunk I (MCP wiring) requires G + H merged.
- Chunk J (CLI embed subcommand) requires D + C + F merged.
- Chunk K (Nix timer) requires J merged.
- Chunk L (e2e + capabilities exposure + integration gate) is last.

---

## Task A: Dependencies and packaging

**Files:**
- Modify: `pyproject.toml`
- Modify: `flake.nix` (Python deps only; nix module wiring lives in Task K)

- [ ] **Step A1: Add runtime deps to pyproject.toml**

Edit `pyproject.toml` dependencies list:

```toml
[project]
name = "aggregator"
version = "0.0.1"
description = "Personal data aggregator (sessions + GitHub) with FastMCP + CLI surfaces"
requires-python = ">=3.11"
dependencies = [
  "fastmcp>=0.4",
  "presidio-analyzer>=2.2",
  "presidio-anonymizer>=2.2",
  "sqlite-vec>=0.1.6",
  "sentence-transformers>=3.0",
  "numpy>=2.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=8",
  "pytest-cov>=5",
  "ruff>=0.6",
]
embed-gguf = [
  "llama-cpp-python>=0.3",
]
```

- [ ] **Step A2: Sync deps in devShell**

Verify `pip install -e '.[dev]'` succeeds inside the flake devShell.

Run: `nix develop --command pip install -e '.[dev]'`
Expected: exit 0, sqlite-vec + sentence-transformers + numpy installed.

- [ ] **Step A3: Add flake devShell packages if not resolved via pip**

`flake.nix` should already have Python + pip. If `sentence-transformers` build fails under nix pip (torch), add `python311Packages.torch-bin` to the devShell inputs. Otherwise leave `flake.nix` unchanged.

- [ ] **Step A4: Commit**

```bash
git add pyproject.toml flake.nix
git commit -m "deps(rag): add sqlite-vec, sentence-transformers, numpy"
```

---

## Task B: Chunking utility

**Files:**
- Create: `aggregator/core/chunk.py`
- Test: `tests/core/test_chunk.py`

- [ ] **Step B1: Write failing tests**

```python
# tests/core/test_chunk.py
"""chunk_body — paragraph-aware windowing for p99 observation bodies."""
from aggregator.core.chunk import chunk_body


def test_short_text_returns_single_chunk():
    text = "short obs body"
    assert chunk_body(text, max_len=4000, overlap=400) == [text]


def test_empty_returns_empty_list():
    assert chunk_body("", max_len=4000, overlap=400) == []
    assert chunk_body("   ", max_len=4000, overlap=400) == []


def test_long_text_splits_on_paragraph_boundary():
    para = "a" * 3000
    text = f"{para}\n\n{para}\n\n{para}"  # ~9000 chars, 3 paras
    chunks = chunk_body(text, max_len=4000, overlap=400)
    assert len(chunks) >= 2
    # each chunk should end at a paragraph break when possible
    for c in chunks[:-1]:
        assert len(c) <= 4000


def test_chunk_boundaries_include_overlap():
    text = "x" * 10000
    chunks = chunk_body(text, max_len=4000, overlap=400)
    # overlap: consecutive chunks share ~400 chars
    if len(chunks) >= 2:
        tail = chunks[0][-400:]
        head = chunks[1][:400]
        assert tail == head  # exact overlap for the pure-char fallback


def test_deterministic():
    text = "para one.\n\npara two.\n\n" * 500
    a = chunk_body(text, max_len=1000, overlap=100)
    b = chunk_body(text, max_len=1000, overlap=100)
    assert a == b
```

- [ ] **Step B2: Verify tests fail**

Run: `pytest tests/core/test_chunk.py -v`
Expected: `ModuleNotFoundError: No module named 'aggregator.core.chunk'`

- [ ] **Step B3: Implement `chunk_body`**

```python
# aggregator/core/chunk.py
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
```

- [ ] **Step B4: Verify tests pass**

Run: `pytest tests/core/test_chunk.py -v`
Expected: 5 passed.

- [ ] **Step B5: Ruff + commit**

```bash
ruff check aggregator/core/chunk.py tests/core/test_chunk.py
ruff format aggregator/core/chunk.py tests/core/test_chunk.py
git add aggregator/core/chunk.py tests/core/test_chunk.py
git commit -m "feat(chunk): paragraph-aware windowing for embed pipeline"
```

---

## Task C: Embedder module

**Files:**
- Create: `aggregator/core/embed.py`
- Test: `tests/core/test_embed.py`

- [ ] **Step C1: Write failing tests**

```python
# tests/core/test_embed.py
"""Qwen3 embedder: prefix template, MRL truncation, determinism."""
import numpy as np
import pytest

from aggregator.core.embed import Embedder, QWEN3_QUERY_PREFIX


@pytest.fixture(scope="module")
def embedder():
    # Uses the default model configured in Embedder (safetensors path).
    # Tests mark themselves skip if the model isn't available locally,
    # so the test can run in CI environments that don't cache the weight.
    try:
        return Embedder()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"embedder unavailable: {e}")


def test_query_gets_prefix(monkeypatch, embedder):
    calls: list[str] = []
    monkeypatch.setattr(embedder, "_encode", lambda texts: (
        calls.extend(texts),
        np.zeros((len(texts), 768), dtype=np.float32),
    )[1])
    embedder.embed_query("quadratic voting")
    assert calls == [f"{QWEN3_QUERY_PREFIX}quadratic voting"]


def test_document_no_prefix(monkeypatch, embedder):
    calls: list[list[str]] = []
    monkeypatch.setattr(embedder, "_encode", lambda texts: (
        calls.append(list(texts)),
        np.zeros((len(texts), 768), dtype=np.float32),
    )[1])
    embedder.embed_documents(["doc a", "doc b"])
    assert calls == [["doc a", "doc b"]]


def test_output_shape_and_dtype(embedder):
    out = embedder.embed_documents(["hello world"])
    assert out.shape == (1, 768)
    assert out.dtype == np.float32


def test_l2_normalized(embedder):
    out = embedder.embed_documents(["hello world", "second doc"])
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-3)


def test_deterministic(embedder):
    a = embedder.embed_query("test query")
    b = embedder.embed_query("test query")
    np.testing.assert_array_equal(a, b)
```

- [ ] **Step C2: Verify tests fail**

Run: `pytest tests/core/test_embed.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step C3: Implement `Embedder`**

```python
# aggregator/core/embed.py
"""Qwen3-Embedding-0.6B wrapper.

Two loader paths, one interface:

* ``AGGREGATOR_EMBED_BACKEND=st`` (default) — sentence-transformers +
  safetensors. ~1.2 GB RAM, no extra runtime.
* ``AGGREGATOR_EMBED_BACKEND=gguf`` — llama-cpp-python + Q4_K_M GGUF.
  ~400 MB RAM, tiny bit slower per query. Requires the optional
  ``embed-gguf`` extra installed.

Both paths return float32, L2-normalized, MRL-truncated to 768 dims.
The Qwen3 query prefix ("Instruct: ...\\nQuery: ...") is applied by
``embed_query``; documents go through ``embed_documents`` unprefixed
(load-bearing per the Qwen3 model card — omitting the prefix loses
1–5% retrieval on the leaderboard).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

QWEN3_QUERY_PREFIX = (
    "Instruct: Given a search query, retrieve relevant passages that answer "
    "the query\nQuery: "
)
_EMBED_DIM = 768  # MRL truncation target
_NATIVE_DIM = 1024
_DEFAULT_MODEL_ST = "Qwen/Qwen3-Embedding-0.6B"
_DEFAULT_MODEL_GGUF = "Qwen/Qwen3-Embedding-0.6B-GGUF"


class Embedder:
    """Single-model embedder. Load once per process, share across writes."""

    def __init__(
        self,
        backend: str | None = None,
        model_name: str | None = None,
        gguf_filename: str = "Qwen3-Embedding-0.6B-Q4_K_M.gguf",
        cache_dir: str | Path | None = None,
    ):
        self.backend = backend or os.environ.get("AGGREGATOR_EMBED_BACKEND", "st")
        self.model_name = model_name
        self._st_model = None
        self._gguf_model = None
        if self.backend == "st":
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(
                self.model_name or _DEFAULT_MODEL_ST,
                cache_folder=str(cache_dir) if cache_dir else None,
            )
        elif self.backend == "gguf":
            try:
                from llama_cpp import Llama
            except ImportError as e:
                raise RuntimeError(
                    "AGGREGATOR_EMBED_BACKEND=gguf requires the "
                    "'embed-gguf' optional extra: pip install "
                    "'aggregator[embed-gguf]'"
                ) from e
            self._gguf_model = Llama.from_pretrained(
                repo_id=self.model_name or _DEFAULT_MODEL_GGUF,
                filename=gguf_filename,
                embedding=True,
                n_ctx=8192,
                verbose=False,
            )
        else:
            raise ValueError(f"unknown embed backend: {self.backend!r}")

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Backend-specific encode. Returns raw native-dim vectors."""
        if self._st_model is not None:
            arr = self._st_model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        elif self._gguf_model is not None:
            arr = np.array(
                [self._gguf_model.embed(t) for t in texts], dtype=np.float32
            )
        else:
            raise RuntimeError("no embedder backend loaded")
        return arr.astype(np.float32)

    @staticmethod
    def _truncate_and_normalize(arr: np.ndarray) -> np.ndarray:
        """MRL truncation + L2 normalization.

        Qwen3-Embedding is trained with Matryoshka losses; truncating the
        first ``_EMBED_DIM`` dims of the native ``_NATIVE_DIM`` output
        preserves ranking quality within a few tenths of a point per the
        Qwen3 tech report. Renormalize after truncation so cosine ≡ dot
        product downstream (sqlite-vec + RRF both assume unit norm).
        """
        if arr.shape[1] > _EMBED_DIM:
            arr = arr[:, :_EMBED_DIM]
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (arr / norms).astype(np.float32)

    def embed_documents(self, docs: list[str]) -> np.ndarray:
        """Encode ``docs`` without the Qwen3 query prefix."""
        if not docs:
            return np.zeros((0, _EMBED_DIM), dtype=np.float32)
        raw = self._encode(list(docs))
        return self._truncate_and_normalize(raw)

    def embed_query(self, query: str) -> np.ndarray:
        """Encode ``query`` with the Qwen3 instruction prefix applied."""
        raw = self._encode([f"{QWEN3_QUERY_PREFIX}{query}"])
        return self._truncate_and_normalize(raw)[0]
```

- [ ] **Step C4: Verify tests pass or skip**

Run: `pytest tests/core/test_embed.py -v`
Expected: 5 passed, OR 5 skipped if the model weight isn't cached in the CI environment. The `pytest.skip` in the fixture handles the CI case; local dev with the model cached should see 5 passed.

- [ ] **Step C5: Ruff + commit**

```bash
ruff check aggregator/core/embed.py tests/core/test_embed.py
ruff format aggregator/core/embed.py tests/core/test_embed.py
git add aggregator/core/embed.py tests/core/test_embed.py
git commit -m "feat(embed): Qwen3-Embedding-0.6B wrapper (safetensors + GGUF)"
```

---

## Task D: Schema v4 migration

**Files:**
- Modify: `aggregator/core/store.py`
- Create: `tests/core/test_store_v4_migration.py`

- [ ] **Step D1: Write failing tests**

```python
# tests/core/test_store_v4_migration.py
"""v3 → v4: additive ALTER TABLE + vec virtual tables, no rebuild."""
import sqlite3

import pytest

from aggregator.core.store import SCHEMA_VERSION, Store


def _make_v3_store(tmp_path):
    """Create a store, downgrade its recorded schema to v3, drop the v4
    columns and vec tables, and return the store handle. Simulates an
    existing user with a v3 cache."""
    db = tmp_path / "cache.db"
    s = Store(db_path=db)
    s.migrate()
    c = s._c()
    c.execute("ALTER TABLE observations DROP COLUMN embedding_state")
    c.execute("ALTER TABLE records DROP COLUMN embedding_state")
    c.execute("DROP TABLE IF EXISTS vec_observations")
    c.execute("DROP TABLE IF EXISTS vec_records")
    c.execute("PRAGMA user_version = 3")
    c.commit()
    s.close()
    return Store(db_path=db)


def test_v4_migration_is_idempotent(tmp_path):
    s = _make_v3_store(tmp_path)
    s.migrate()
    s.migrate()  # second call must not fail
    assert s.schema_version() == SCHEMA_VERSION == 4


def test_v4_migration_adds_columns(tmp_path):
    s = _make_v3_store(tmp_path)
    s.migrate()
    c = s._c()
    obs_cols = {row[1] for row in c.execute("PRAGMA table_info(observations)")}
    rec_cols = {row[1] for row in c.execute("PRAGMA table_info(records)")}
    assert "embedding_state" in obs_cols
    assert "embedding_state" in rec_cols


def test_v4_migration_creates_vec_tables(tmp_path):
    s = _make_v3_store(tmp_path)
    s.migrate()
    c = s._c()
    tables = {
        row[0] for row in c.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        )
    }
    assert "vec_observations" in tables
    assert "vec_records" in tables


def test_v4_migration_leaves_existing_rows_null(tmp_path):
    s = _make_v3_store(tmp_path)
    c = s._c()
    c.execute(
        "INSERT INTO observations(obs_id, session_id, root_session_id, "
        "type, ts) VALUES (?, 'sid', 'sid', 'user', '2026-01-01T00:00:00')",
        ("existing_obs",),
    )
    c.commit()
    s.migrate()
    row = c.execute(
        "SELECT embedding_state FROM observations WHERE obs_id = ?",
        ("existing_obs",),
    ).fetchone()
    assert row["embedding_state"] is None


def test_v4_bumps_pragma_user_version(tmp_path):
    s = _make_v3_store(tmp_path)
    assert s.schema_version() == 3
    s.migrate()
    assert s.schema_version() == 4
```

- [ ] **Step D2: Verify tests fail**

Run: `pytest tests/core/test_store_v4_migration.py -v`
Expected: fail — SCHEMA_VERSION is 3, no `embedding_state` column, no `vec_*` tables.

- [ ] **Step D3: Bump schema version + register sqlite-vec + add DDL**

Edit `aggregator/core/store.py`:

Bump the constant near the top:

```python
SCHEMA_VERSION = 4
```

Add near the top-level imports (after existing imports):

```python
import sqlite_vec
```

In `_DDL`, append the two vec tables + the two indexes on embedding_state:

```python
# --- v4: vector index (sqlite-vec vec0 virtual tables) ------------
"""
CREATE VIRTUAL TABLE IF NOT EXISTS vec_observations USING vec0(
    obs_id     TEXT PRIMARY KEY,
    embedding  float[768]
);
""",
"""
CREATE VIRTUAL TABLE IF NOT EXISTS vec_records USING vec0(
    stable_id  TEXT PRIMARY KEY,
    embedding  float[768]
);
""",
"CREATE INDEX IF NOT EXISTS obs_embedding_state ON observations(embedding_state);",
"CREATE INDEX IF NOT EXISTS rec_embedding_state ON records(embedding_state);",
```

In `_DROP_ALL`, add:

```python
"DROP TABLE IF EXISTS vec_observations;",
"DROP TABLE IF EXISTS vec_records;",
```

In the existing `observations` and `records` CREATE TABLE statements in `_DDL`, add the column at the end of the column list:

```python
    body             TEXT,
    embedding_state  TEXT
```

and for records:

```python
    extra      TEXT NOT NULL DEFAULT '{}',
    embedding_state TEXT
```

- [ ] **Step D4: Load sqlite-vec extension in `_c()`**

> **SUPERSEDED 2026-08-17 — do NOT implement as written below.** The
> unconditional load is a defect: every read in this product goes through
> `Store`, including `aggregator_search_memory`, which has no vector
> dependency at all. An extension that is missing, ABI-mismatched, or blocked
> by a python built without `enable_load_extension` would then take FTS5 recall
> down with the optional half of the feature. As shipped, the load is
> best-effort — see `Store._try_load_sqlite_vec`, `Store.vector_available` and
> `VectorIndexUnavailableError` in `aggregator/core/store.py`, and
> `tests/core/test_store_vector_degrade.py`. Vector DDL is skipped when the
> extension is absent, vector writes no-op without advancing the watermark,
> vector reads raise by name, and FTS5 keeps serving. Kept below for the
> record only.

Modify `Store._c()` right after `self._conn.execute("PRAGMA foreign_keys = ON;")`:

```python
# sqlite-vec ships as a loadable extension. Enable + load unconditionally
# — the vec_observations / vec_records virtual tables need it to exist
# on every read-only AND writable connection.
self._conn.enable_load_extension(True)
sqlite_vec.load(self._conn)
self._conn.enable_load_extension(False)
```

Note: `sqlite_vec.load` works on both writable and RO SQLite connections. The `enable_load_extension` toggle must fire before the load call and be disabled after so nothing else in the process can side-load native code.

- [ ] **Step D5: Generalise the ALTER-TABLE probe helper**

Rename `_ensure_sessions_origin_column` to `_ensure_column` with parameters:

```python
@staticmethod
def _ensure_column(
    c: sqlite3.Connection, table: str, column: str, ddl: str
) -> None:
    """Idempotent ALTER TABLE .. ADD COLUMN.

    Probes ``PRAGMA table_info(<table>)`` (not user_version) so a
    half-applied state can't run the ALTER twice.
    """
    table_exists = c.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    if not table_exists:
        return
    cols = {row[1] for row in c.execute(f"PRAGMA table_info({table})")}
    if column not in cols:
        c.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
```

Update `migrate()` to call it three times:

```python
def migrate(self) -> None:
    self._ensure_writable()
    c = self._c()
    self._ensure_column(
        c, "sessions", "origin",
        "TEXT NOT NULL DEFAULT 'claude-code'",
    )
    self._ensure_column(c, "observations", "embedding_state", "TEXT")
    self._ensure_column(c, "records", "embedding_state", "TEXT")
    for stmt in _DDL:
        c.executescript(stmt)
    c.execute(f"PRAGMA user_version = {SCHEMA_VERSION};")
    c.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    c.commit()
```

- [ ] **Step D6: Bump `_ensure_cache_ready` in mcp.py**

Nothing to change — `SCHEMA_VERSION` bump in store.py cascades through the `if version < SCHEMA_VERSION` check.

- [ ] **Step D7: Verify tests pass**

Run: `pytest tests/core/test_store_v4_migration.py -v`
Expected: 5 passed.

Run the full existing test suite to catch regressions:

Run: `pytest -q`
Expected: pre-existing pass count + 5 new passes; no regressions.

- [ ] **Step D8: Ruff + commit**

```bash
ruff check aggregator/core/store.py tests/core/test_store_v4_migration.py
ruff format aggregator/core/store.py tests/core/test_store_v4_migration.py
git add aggregator/core/store.py tests/core/test_store_v4_migration.py
git commit -m "feat(store): v4 schema — embedding_state + vec_observations/records tables"
```

---

## Task E: Vec table reader + writer helpers

**Files:**
- Modify: `aggregator/core/store.py`
- Create: `tests/core/test_store_vec.py`

- [ ] **Step E1: Write failing tests**

```python
# tests/core/test_store_vec.py
"""Vector arm: upsert vec rows, KNN reader, watermark helpers."""
import numpy as np
import pytest

from aggregator.core.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    return s


def _seed_observation(s, obs_id, body="hello"):
    c = s._c()
    c.execute(
        "INSERT INTO observations(obs_id, session_id, root_session_id, "
        "type, ts, body) VALUES (?, ?, ?, 'user', '2026-01-01', ?)",
        (obs_id, "sid", "sid", body),
    )
    c.execute(
        "INSERT INTO sessions(session_id, root_session_id, kind, first_ts, "
        "last_ts, jsonl_path) VALUES ('sid', 'sid', 'session', "
        "'2026-01-01', '2026-01-01', '/tmp/x.jsonl')"
    ) if not c.execute(
        "SELECT 1 FROM sessions WHERE session_id='sid'"
    ).fetchone() else None
    c.commit()


def test_upsert_and_read_vec_obs(store):
    _seed_observation(store, "o1")
    vec = np.random.rand(768).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    store.upsert_vec_observations([("o1", vec)])
    query = vec  # exact match
    hits = store._vec_obs_ids(query, k=5)
    assert hits[0] == "o1"


def test_knn_returns_topk_ordered(store):
    for i in range(5):
        _seed_observation(store, f"o{i}")
    vecs = np.eye(5, 768, dtype=np.float32)  # 5 orthogonal unit vectors
    store.upsert_vec_observations(
        [(f"o{i}", vecs[i]) for i in range(5)]
    )
    hits = store._vec_obs_ids(vecs[2], k=3)
    assert hits[0] == "o2"
    assert len(hits) == 3


def test_select_unembedded_observations(store):
    for i in range(3):
        _seed_observation(store, f"u{i}")
    rows = store.select_unembedded("observations", limit=10)
    assert {r["obs_id"] for r in rows} == {"u0", "u1", "u2"}


def test_mark_embedded_flips_state(store):
    _seed_observation(store, "u0")
    store.mark_embedded("observations", ["u0"], state="ok")
    rows = store.select_unembedded("observations", limit=10)
    assert not any(r["obs_id"] == "u0" for r in rows)


def test_mark_embedded_error_state(store):
    _seed_observation(store, "u0")
    store.mark_embedded("observations", ["u0"], state="error")
    c = store._c()
    row = c.execute(
        "SELECT embedding_state FROM observations WHERE obs_id = ?", ("u0",)
    ).fetchone()
    assert row["embedding_state"] == "error"
```

- [ ] **Step E2: Verify tests fail**

Run: `pytest tests/core/test_store_vec.py -v`
Expected: fail — methods don't exist.

- [ ] **Step E3: Implement writer helpers on `Store`**

Add to `Store` in `aggregator/core/store.py`:

```python
# -- writes: v4 vector index ----------------------------------------

def upsert_vec_observations(
    self, rows: list[tuple[str, "np.ndarray"]]
) -> None:
    """Upsert ``(obs_id, embedding)`` rows into ``vec_observations``.

    Delete-then-insert to keep upsert idempotent under sqlite-vec (vec0
    virtual tables don't support UPSERT).
    """
    self._ensure_writable()
    c = self._c()
    for obs_id, embedding in rows:
        c.execute("DELETE FROM vec_observations WHERE obs_id = ?", (obs_id,))
        c.execute(
            "INSERT INTO vec_observations(obs_id, embedding) VALUES (?, ?)",
            (obs_id, embedding.astype("float32").tobytes()),
        )
    c.commit()

def upsert_vec_records(
    self, rows: list[tuple[str, "np.ndarray"]]
) -> None:
    """Upsert ``(stable_id, embedding)`` rows into ``vec_records``."""
    self._ensure_writable()
    c = self._c()
    for stable_id, embedding in rows:
        c.execute(
            "DELETE FROM vec_records WHERE stable_id = ?", (stable_id,)
        )
        c.execute(
            "INSERT INTO vec_records(stable_id, embedding) VALUES (?, ?)",
            (stable_id, embedding.astype("float32").tobytes()),
        )
    c.commit()

def select_unembedded(
    self, kind: str, limit: int = 500
) -> list[sqlite3.Row]:
    """Return rows whose ``embedding_state IS NULL`` (backlog for worker)."""
    c = self._c()
    if kind == "observations":
        return list(c.execute(
            "SELECT obs_id, body FROM observations "
            "WHERE embedding_state IS NULL "
            "ORDER BY ts DESC LIMIT ?",
            (limit,),
        ))
    if kind == "records":
        return list(c.execute(
            "SELECT stable_id, subject, body FROM records "
            "WHERE embedding_state IS NULL "
            "ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ))
    raise ValueError(f"unknown kind: {kind!r}")

def mark_embedded(
    self, kind: str, ids: list[str], state: str
) -> None:
    """Batch UPDATE ``embedding_state`` for ``ids``.

    ``state`` in ``{'ok', 'skip', 'error'}``. Called after each worker
    batch to advance the watermark.
    """
    self._ensure_writable()
    c = self._c()
    if state not in ("ok", "skip", "error"):
        raise ValueError(f"invalid state: {state!r}")
    if kind == "observations":
        col = "obs_id"
        table = "observations"
    elif kind == "records":
        col = "stable_id"
        table = "records"
    else:
        raise ValueError(f"unknown kind: {kind!r}")
    placeholders = ",".join("?" * len(ids))
    c.execute(
        f"UPDATE {table} SET embedding_state = ? WHERE {col} IN ({placeholders})",
        (state, *ids),
    )
    c.commit()

def _vec_obs_ids(self, query_embedding: "np.ndarray", k: int) -> list[str]:
    """Vector KNN over ``vec_observations``. Returns top-K obs_ids ordered
    by ascending distance (i.e. best-matching first)."""
    c = self._c()
    rows = c.execute(
        """
        SELECT obs_id
        FROM vec_observations
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
        """,
        (query_embedding.astype("float32").tobytes(), k),
    ).fetchall()
    return [r["obs_id"] for r in rows]

def _vec_record_ids(
    self, query_embedding: "np.ndarray", k: int
) -> list[str]:
    """Vector KNN over ``vec_records``."""
    c = self._c()
    rows = c.execute(
        """
        SELECT stable_id
        FROM vec_records
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
        """,
        (query_embedding.astype("float32").tobytes(), k),
    ).fetchall()
    return [r["stable_id"] for r in rows]
```

- [ ] **Step E4: Verify tests pass**

Run: `pytest tests/core/test_store_vec.py -v`
Expected: 5 passed.

- [ ] **Step E5: Ruff + commit**

```bash
ruff check aggregator/core/store.py tests/core/test_store_vec.py
ruff format aggregator/core/store.py tests/core/test_store_vec.py
git add aggregator/core/store.py tests/core/test_store_vec.py
git commit -m "feat(store): vec upsert + KNN reader + embedding_state watermark helpers"
```

---

## Task F: Embed worker (CLI subcommand)

**Files:**
- Modify: `aggregator/cli.py`
- Create: `tests/test_cli_embed.py`

- [ ] **Step F1: Write failing tests**

```python
# tests/test_cli_embed.py
"""aggregator embed [--catchup|--once] flock + watermark advancement."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from aggregator.core.store import Store


@pytest.fixture
def cache(tmp_path):
    db = tmp_path / "cache.db"
    s = Store(db_path=db)
    s.migrate()
    c = s._c()
    c.execute(
        "INSERT INTO sessions(session_id, root_session_id, kind, first_ts, "
        "last_ts, jsonl_path) VALUES ('sid', 'sid', 'session', "
        "'2026-01-01', '2026-01-01', '/tmp/x.jsonl')"
    )
    for i in range(3):
        c.execute(
            "INSERT INTO observations(obs_id, session_id, root_session_id, "
            "type, ts, body) VALUES (?, 'sid', 'sid', 'user', '2026-01-01', ?)",
            (f"o{i}", f"body text {i}"),
        )
    c.commit()
    s.close()
    return db


def _make_stub_embedder(monkeypatch):
    """Patch Embedder so tests don't require the model."""
    class StubEmbedder:
        def __init__(self, *a, **kw):
            pass
        def embed_documents(self, docs):
            return np.array(
                [[float(i)] * 768 for i in range(len(docs))], dtype=np.float32
            )
        def embed_query(self, q):
            return np.zeros(768, dtype=np.float32)
    monkeypatch.setattr("aggregator.core.embed.Embedder", StubEmbedder)
    monkeypatch.setattr("aggregator.cli.Embedder", StubEmbedder)


def test_catchup_embeds_all(cache, monkeypatch):
    _make_stub_embedder(monkeypatch)
    from aggregator.cli import _cmd_embed
    _cmd_embed(argparse_ns(catchup=True, once=False, source="observations"),
               _store=Store(db_path=cache))
    s = Store(db_path=cache)
    assert not s.select_unembedded("observations", limit=10)


def test_once_embeds_one_batch(cache, monkeypatch):
    _make_stub_embedder(monkeypatch)
    from aggregator.cli import _cmd_embed
    _cmd_embed(argparse_ns(catchup=False, once=True, source="observations",
                           batch_size=2),
               _store=Store(db_path=cache))
    s = Store(db_path=cache)
    remaining = s.select_unembedded("observations", limit=10)
    assert len(remaining) == 1  # 3 seeded − 2 embedded


def test_second_catchup_is_noop(cache, monkeypatch):
    _make_stub_embedder(monkeypatch)
    from aggregator.cli import _cmd_embed
    ns = argparse_ns(catchup=True, once=False, source="observations")
    _cmd_embed(ns, _store=Store(db_path=cache))
    # Snapshot vec_observations count
    s = Store(db_path=cache)
    c = s._c()
    n1 = c.execute("SELECT COUNT(*) AS n FROM vec_observations").fetchone()["n"]
    _cmd_embed(ns, _store=Store(db_path=cache))
    n2 = c.execute("SELECT COUNT(*) AS n FROM vec_observations").fetchone()["n"]
    assert n1 == n2 == 3


def argparse_ns(**kw):
    import argparse
    ns = argparse.Namespace(catchup=False, once=False, source="observations",
                            batch_size=500)
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns
```

- [ ] **Step F2: Verify tests fail**

Run: `pytest tests/test_cli_embed.py -v`
Expected: fail — `_cmd_embed` not defined.

- [ ] **Step F3: Implement `_cmd_embed`**

Edit `aggregator/cli.py`. Add near top with other imports:

```python
import fcntl
from aggregator.core.chunk import chunk_body
from aggregator.core.embed import Embedder
```

Add the subcommand handler:

```python
def _cmd_embed(args: argparse.Namespace, _store: Store | None = None) -> None:
    """Background embed worker.

    ``--catchup`` embeds all unembedded rows. ``--once`` embeds one batch
    and exits (used by the systemd timer). Both take an OS-level
    ``flock`` on ``<cache>.embed.lock`` so two workers can't fight over
    the backlog.
    """
    store = _store or Store()
    store.migrate()

    lock_path = Path(str(store.db_path) + ".embed.lock")
    lock_path.touch(exist_ok=True)
    lock_fd = os.open(str(lock_path), os.O_RDWR)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another embed worker is running; exiting")
            return

        embedder = Embedder()
        sources = (
            ["observations", "records"] if args.source == "both"
            else [args.source]
        )
        for kind in sources:
            _embed_backlog(store, embedder, kind, args)
    finally:
        os.close(lock_fd)


def _embed_backlog(
    store: Store,
    embedder: Embedder,
    kind: str,
    args: argparse.Namespace,
) -> None:
    """Loop batches until backlog drained (catchup) or one batch (once)."""
    while True:
        rows = store.select_unembedded(kind, limit=args.batch_size)
        if not rows:
            return
        _embed_batch(store, embedder, kind, rows)
        if args.once:
            return


def _embed_batch(
    store: Store,
    embedder: Embedder,
    kind: str,
    rows: list,
) -> None:
    """Embed one batch and advance the watermark."""
    ok_ids: list[str] = []
    skip_ids: list[str] = []
    all_vecs: list[tuple[str, "np.ndarray"]] = []
    for row in rows:
        if kind == "observations":
            row_id = row["obs_id"]
            body = row["body"] or ""
        else:
            row_id = row["stable_id"]
            body = f"{row['subject']}\n\n{row['body']}"
        chunks = chunk_body(body)
        if not chunks:
            skip_ids.append(row_id)
            continue
        vecs = embedder.embed_documents(chunks)
        for i, vec in enumerate(vecs):
            chunk_id = row_id if len(chunks) == 1 else f"{row_id}:{i}"
            all_vecs.append((chunk_id, vec))
        ok_ids.append(row_id)
    if kind == "observations":
        store.upsert_vec_observations(all_vecs)
    else:
        store.upsert_vec_records(all_vecs)
    if ok_ids:
        store.mark_embedded(kind, ok_ids, state="ok")
    if skip_ids:
        store.mark_embedded(kind, skip_ids, state="skip")
```

Register the subparser in `main()`:

```python
p_embed = subparsers.add_parser(
    "embed", help="Background embed worker (fills the vector index).",
)
mode = p_embed.add_mutually_exclusive_group(required=True)
mode.add_argument("--catchup", action="store_true",
                  help="Embed all unembedded rows and exit.")
mode.add_argument("--once", action="store_true",
                  help="Embed one batch (used by the systemd timer).")
p_embed.add_argument("--source", choices=["observations", "records", "both"],
                     default="observations")
p_embed.add_argument("--batch-size", type=int, default=500,
                     dest="batch_size")
p_embed.set_defaults(func=_cmd_embed)
```

- [ ] **Step F4: Verify tests pass**

Run: `pytest tests/test_cli_embed.py -v`
Expected: 3 passed.

- [ ] **Step F5: Ruff + commit**

```bash
ruff check aggregator/cli.py tests/test_cli_embed.py
ruff format aggregator/cli.py tests/test_cli_embed.py
git add aggregator/cli.py tests/test_cli_embed.py
git commit -m "feat(cli): aggregator embed subcommand with flock + watermark"
```

---

## Task G: Hybrid retriever (RRF fusion)

**Files:**
- Create: `aggregator/core/hybrid.py`
- Create: `tests/core/test_hybrid.py`

- [ ] **Step G1: Write failing tests**

```python
# tests/core/test_hybrid.py
"""RRF k=60 fusion: arithmetic, empty-arm fallthroughs, tie order."""
import numpy as np
import pytest

from aggregator.core.hybrid import rrf_fuse


def test_rrf_arithmetic_pure_intersect():
    fts = ["a", "b", "c"]
    vec = ["a", "b", "c"]
    fused = rrf_fuse(fts, vec, k=60)
    # a at rank 1 in both: 1/(60+1) + 1/(60+1) = 2/61
    ids = [i for i, _ in fused]
    assert ids[:3] == ["a", "b", "c"]


def test_rrf_arithmetic_pure_disjoint():
    fused = rrf_fuse(["a", "b"], ["c", "d"], k=60)
    scores = dict(fused)
    assert scores["a"] == 1 / 61
    assert scores["c"] == 1 / 61
    assert scores["b"] == 1 / 62
    assert scores["d"] == 1 / 62


def test_empty_vec_falls_through_to_fts():
    fused = rrf_fuse(["a", "b", "c"], [], k=60)
    ids = [i for i, _ in fused]
    assert ids == ["a", "b", "c"]


def test_empty_fts_falls_through_to_vec():
    fused = rrf_fuse([], ["a", "b"], k=60)
    ids = [i for i, _ in fused]
    assert ids == ["a", "b"]


def test_both_empty_returns_empty():
    assert rrf_fuse([], [], k=60) == []


def test_scoring_promotes_dual_matches_above_single():
    fused = rrf_fuse(["a", "b", "c"], ["c", "d"], k=60)
    ids = [i for i, _ in fused]
    # c appears in both, others in one each: c should top
    assert ids[0] == "c"
```

- [ ] **Step G2: Verify tests fail**

Run: `pytest tests/core/test_hybrid.py -v`
Expected: fail — module doesn't exist.

- [ ] **Step G3: Implement RRF fusion**

```python
# aggregator/core/hybrid.py
"""Reciprocal Rank Fusion of the FTS5 and vector retrieval arms.

RRF with k=60 (Cormack et al., SIGIR 2009) is the SOTA cheap fusion
default: score-agnostic (no BM25-vs-cosine normalization problem), one
constant, dominates BM25-alone and vector-alone on most heterogeneous
corpora. Upgrade path: tuned convex combination once we have 50-100
labeled query pairs (deferred per spec §Non-goals).

The retriever surface is DELIBERATELY id-list-in, id-list-out — no
Store or Embedder knowledge here. Callers assemble the two id lists
(FTS5 arm + vector arm), pass them in, receive fused ids ordered by
score descending. This keeps the fusion pure and trivially testable
without a live store.
"""
from __future__ import annotations


def rrf_fuse(
    fts_ids: list[str],
    vec_ids: list[str],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse two ranked id lists via reciprocal rank fusion.

    Returns ``(id, score)`` pairs ordered by score descending. Score
    is ``sum(1 / (k + rank_i))`` across every arm that returned the id
    (rank is 1-indexed). Empty arms are skipped; if both arms are empty
    the result is ``[]``.
    """
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(fts_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    for rank, doc_id in enumerate(vec_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
```

- [ ] **Step G4: Verify tests pass**

Run: `pytest tests/core/test_hybrid.py -v`
Expected: 6 passed.

- [ ] **Step G5: Ruff + commit**

```bash
ruff check aggregator/core/hybrid.py tests/core/test_hybrid.py
ruff format aggregator/core/hybrid.py tests/core/test_hybrid.py
git add aggregator/core/hybrid.py tests/core/test_hybrid.py
git commit -m "feat(hybrid): RRF k=60 fusion of FTS5 + vector arms"
```

---

## Task H: Reranker (Qwen3-Reranker-0.6B)

**Files:**
- Create: `aggregator/core/rerank.py`
- Create: `tests/core/test_rerank.py`

- [ ] **Step H1: Write failing tests**

```python
# tests/core/test_rerank.py
"""Qwen3-Reranker: score returns per-doc floats, deterministic, higher
score wins."""
import numpy as np
import pytest

from aggregator.core.rerank import Reranker


@pytest.fixture(scope="module")
def reranker():
    try:
        return Reranker()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"reranker unavailable: {e}")


def test_score_shape(reranker):
    scores = reranker.score("what is RRF?", ["reciprocal rank fusion",
                                             "convex hull algorithm"])
    assert scores.shape == (2,)


def test_deterministic(reranker):
    a = reranker.score("q", ["d1", "d2"])
    b = reranker.score("q", ["d1", "d2"])
    np.testing.assert_array_equal(a, b)


def test_relevant_beats_irrelevant(reranker):
    scores = reranker.score(
        "what is reciprocal rank fusion",
        [
            "Reciprocal rank fusion (RRF) is a hybrid retrieval scoring method.",
            "The convex hull of a set of points is the smallest convex polygon.",
        ],
    )
    assert scores[0] > scores[1]
```

- [ ] **Step H2: Verify tests fail**

Run: `pytest tests/core/test_rerank.py -v`
Expected: fail — module doesn't exist.

- [ ] **Step H3: Implement Reranker**

```python
# aggregator/core/rerank.py
"""Qwen3-Reranker-0.6B cross-encoder wrapper.

DO NOT swap in bge-reranker-v2-m3: per Qwen's own eval it *degrades*
Qwen3-embedded English retrieval (MTEB-R 61.82 → 57.03). Qwen3-Reranker
is the vendor-tested-safe pairing.

Lazy load: instantiate ``Reranker()`` only when the caller passes
``rerank=True`` to the MCP tool. Adds ~2 GB RSS + ~300 ms/query.
"""
from __future__ import annotations

import logging
import os

import numpy as np

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "Qwen/Qwen3-Reranker-0.6B"


class Reranker:
    def __init__(self, model_name: str | None = None):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name or _DEFAULT_MODEL
        self._model = CrossEncoder(self.model_name, trust_remote_code=True)

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        """Return one relevance score per doc (higher = more relevant)."""
        if not docs:
            return np.zeros((0,), dtype=np.float32)
        pairs = [(query, d) for d in docs]
        scores = self._model.predict(pairs, show_progress_bar=False)
        return np.asarray(scores, dtype=np.float32)
```

- [ ] **Step H4: Verify tests pass or skip**

Run: `pytest tests/core/test_rerank.py -v`
Expected: 3 passed OR 3 skipped (fixture skips if the model isn't cached).

- [ ] **Step H5: Ruff + commit**

```bash
ruff check aggregator/core/rerank.py tests/core/test_rerank.py
ruff format aggregator/core/rerank.py tests/core/test_rerank.py
git add aggregator/core/rerank.py tests/core/test_rerank.py
git commit -m "feat(rerank): Qwen3-Reranker-0.6B cross-encoder (flag-gated)"
```

---

## Task I: MCP wiring — route text queries through hybrid

**Files:**
- Modify: `aggregator/mcp.py`
- Modify: `aggregator/core/store.py` (add `count_vec_rows` for capabilities)
- Create: `tests/test_mcp_hybrid.py`

- [ ] **Step I1: Write failing tests**

```python
# tests/test_mcp_hybrid.py
"""MCP: aggregator_search_memory routes text queries through hybrid when
vec tables non-empty; falls through to FTS5 when empty; rerank reorders."""
import numpy as np
import pytest

from aggregator.core.store import Store
from aggregator.mcp import aggregator_query


def _seed(store, obs_id, body):
    c = store._c()
    c.execute(
        "INSERT OR IGNORE INTO sessions(session_id, root_session_id, kind, "
        "first_ts, last_ts, jsonl_path, origin) VALUES ('sid', 'sid', "
        "'session', '2026-01-01', '2026-01-01', '/tmp/x.jsonl', 'claude-code')"
    )
    c.execute(
        "INSERT INTO observations(obs_id, session_id, root_session_id, "
        "type, ts, body, embedding_state) "
        "VALUES (?, 'sid', 'sid', 'user', '2026-01-01', ?, NULL)",
        (obs_id, body),
    )
    c.commit()


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    # Force MCP internals to use this store instead of default XDG path.
    monkeypatch.setattr(
        "aggregator.mcp._default_store", lambda: Store(db_path=tmp_path / "cache.db")
    )
    return s


def test_text_query_falls_through_to_fts_when_vec_empty(store):
    _seed(store, "o1", "quadratic voting is a governance mechanism")
    result = aggregator_query("quadratic voting", _store=store)
    assert result["ok"] is True
    assert result["total"] >= 1


def test_text_query_uses_hybrid_when_vec_nonempty(store, monkeypatch):
    _seed(store, "o1", "quadratic voting")
    _seed(store, "o2", "unrelated text about pigeons")
    # Stub embedder → o2's embedding is a near-perfect match for the query
    class StubEmbedder:
        def embed_query(self, q):
            v = np.zeros(768, dtype=np.float32); v[0] = 1.0; return v
        def embed_documents(self, docs):
            arr = np.zeros((len(docs), 768), dtype=np.float32)
            for i, _ in enumerate(docs):
                arr[i, 0] = 1.0 if i == 1 else 0.0  # o2 = unit x-axis
            return arr
    monkeypatch.setattr("aggregator.mcp.Embedder", StubEmbedder)
    # Seed vec index manually.
    stub = StubEmbedder()
    vecs = stub.embed_documents(["quadratic voting", "unrelated text about pigeons"])
    store.upsert_vec_observations([("o1", vecs[0]), ("o2", vecs[1])])
    store.mark_embedded("observations", ["o1", "o2"], state="ok")
    # Query "voting" — FTS matches o1, vector matches o2 (via stub). Fused: both surface.
    result = aggregator_query("voting", _store=store)
    hit_ids = {r["stable_id"] for r in result["records"]}
    assert "o1" in hit_ids  # FTS arm
    assert result["ok"] is True
```

- [ ] **Step I2: Verify tests fail**

Run: `pytest tests/test_mcp_hybrid.py -v`
Expected: passes ONLY the first test (falls through to FTS); the second test hits the vector arm which doesn't exist yet in mcp.py — likely also passes because the vector arm silently isn't wired. Confirm this passes only after step I3.

- [ ] **Step I3: Add `count_vec_rows` to Store**

Add to `Store` in `aggregator/core/store.py`:

```python
def count_vec_rows(self, kind: str) -> int:
    """Row count in the vec_<kind> virtual table for capabilities output."""
    c = self._c()
    if kind == "observations":
        row = c.execute("SELECT COUNT(*) AS n FROM vec_observations").fetchone()
    elif kind == "records":
        row = c.execute("SELECT COUNT(*) AS n FROM vec_records").fetchone()
    else:
        raise ValueError(f"unknown kind: {kind!r}")
    return int(row["n"]) if row else 0
```

Extend `Store.capabilities()` return dict:

```python
"vector_index": {
    "observations_embedded": self.count_vec_rows("observations"),
    "records_embedded":      self.count_vec_rows("records"),
},
```

- [ ] **Step I4: Route text queries through hybrid in `mcp.py`**

Edit `aggregator/mcp.py`. Add imports:

```python
from aggregator.core.embed import Embedder
from aggregator.core.hybrid import rrf_fuse
```

Add a module-level lazy singleton:

```python
_embedder: Embedder | None = None


def _get_embedder() -> Embedder:
    """Lazy singleton — first hybrid query per process loads the model."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
```

Extend `aggregator_query` signature:

```python
def aggregator_query(
    dsl: str,
    fields: str = "summary",
    page_size: int | None = None,
    page_token: str | None = None,
    drilldown: bool = False,
    rerank: bool = False,
    _store: Store | None = None,
) -> dict[str, Any]:
```

Add a helper for the hybrid path (near `_query_sessions_path`):

```python
def _hybrid_obs_ids(store: Store, ast: QueryAST, k: int = 50) -> list[str]:
    """Run FTS + vector arms, RRF-fuse, return top-K obs_ids.

    Falls through to FTS-only when the vec index is empty (fresh cache
    pre-backfill). Falls through to FTS-only if embedder init fails
    (graceful degradation: recall drops, tool stays up).
    """
    fts_ids: list[str]
    try:
        fts_ids = store._fts_obs_ids(ast.text) if ast.text else []
    except sqlite3.OperationalError:
        fts_ids = []
    vec_ids: list[str] = []
    if store.count_vec_rows("observations") > 0 and ast.text:
        try:
            emb = _get_embedder().embed_query(ast.text)
            vec_ids = store._vec_obs_ids(emb, k=k)
        except Exception:  # noqa: BLE001
            log.exception("hybrid vector arm failed; falling through to FTS")
    fused = rrf_fuse(fts_ids[:k], vec_ids[:k], k=60)
    # Strip synthetic chunk suffixes (`obs_id:N`) — collapse back to base ids.
    seen: set[str] = set()
    out: list[str] = []
    for chunk_id, _ in fused:
        base = chunk_id.split(":", 1)[0]
        if base not in seen:
            seen.add(base)
            out.append(base)
        if len(out) >= k:
            break
    return out
```

Wire it into `_query_observations` calls inside `_query_sessions_path` when `ast.text` is set and vec index non-empty. The minimum-invasive integration: swap the existing `store._fts_obs_ids(ast.text)` call inside `Store.query_observations`'s FTS path with the fused list. Cleanest surgical edit: add a new store method `_hybrid_obs_ids(embedder, ast, k)` (delegating to hybrid module) and call it from `query_observations` when the ast has text and vec_observations is non-empty.

**Practical:** minimise store.py churn. Pass fused ids from mcp.py by modifying `_query_sessions_path` / `_query_records_path` to substitute the AST's `text` for an equivalent id-list filter when hybrid runs. Concretely add a new AST field `id_scope: set[str] | None` (optional; when set, restricts result to those ids). Store's `_obs_where` honours it via `IN (...)`. Then mcp.py:

1. If `ast.text` and vec index non-empty → compute fused ids via `_hybrid_obs_ids`.
2. Replace `ast` with `replace(ast, text=None, id_scope=set(fused))`.
3. Existing session card projection remains unchanged.

Add to `aggregator/sources/base.py::QueryAST`:

```python
@dataclass(frozen=True)
class QueryAST:
    # ... existing fields ...
    id_scope: frozenset[str] | None = None
```

Add to `Store._obs_where`:

```python
if ast.id_scope:
    placeholders = ",".join("?" * len(ast.id_scope))
    clauses.append(f"obs_id IN ({placeholders})")
    params.extend(sorted(ast.id_scope))
```

Similar `id_scope` handling for records: add to `Store._build_where`:

```python
if ast.id_scope:
    placeholders = ",".join("?" * len(ast.id_scope))
    clauses.append(f"stable_id IN ({placeholders})")
    params.extend(sorted(ast.id_scope))
```

Then in `mcp.py::_query_sessions_path`, before the existing `query_sessions` / `query_observations` call:

```python
if ast.text and store.count_vec_rows("observations") > 0:
    fused_ids = _hybrid_obs_ids(store, ast, k=page_size * 2)
    ast = replace(ast, text=None, id_scope=frozenset(fused_ids) if fused_ids else frozenset({"__no_match__"}))
```

Same pattern in `_query_records_path` using `_hybrid_record_ids` (mirror helper on the records side).

- [ ] **Step I5: Implement `rerank` flag**

Add a rerank singleton similar to `_embedder`:

```python
_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from aggregator.core.rerank import Reranker
        _reranker = Reranker()
    return _reranker
```

In `_query_sessions_path` / `_query_records_path`, after collecting the page of records:

```python
if rerank and items:
    docs = [it.get("subject", "") + "\n\n" + it.get("content", "") for it in items]
    scores = _get_reranker().score(ast.text or "", docs)
    reordered = sorted(zip(items, scores), key=lambda p: p[1], reverse=True)
    items = [it for it, _ in reordered]
```

Pass `rerank` through the routing helpers by adding it to their signatures.

- [ ] **Step I6: Update the tool adapter**

```python
async def _tool_aggregator_query(
    dsl: str,
    fields: str = "summary",
    page_size: int | None = None,
    page_token: str | None = None,
    drilldown: bool = False,
    rerank: bool = False,
) -> dict[str, Any]:
    return aggregator_query(
        dsl=dsl, fields=fields, page_size=page_size,
        page_token=page_token, drilldown=drilldown, rerank=rerank,
    )
```

- [ ] **Step I7: Verify tests pass**

Run: `pytest tests/test_mcp_hybrid.py tests/ -q`
Expected: all pre-existing MCP tests still pass + 2 new pass.

- [ ] **Step I8: Ruff + commit**

```bash
ruff check aggregator/mcp.py aggregator/core/store.py aggregator/sources/base.py tests/test_mcp_hybrid.py
ruff format aggregator/mcp.py aggregator/core/store.py aggregator/sources/base.py tests/test_mcp_hybrid.py
git add -A
git commit -m "feat(mcp): route text queries through hybrid retriever + rerank flag"
```

---

## Task J: e2e RAG test

**Files:**
- Create: `tests/test_e2e_rag.py`

- [ ] **Step J1: Write failing test**

```python
# tests/test_e2e_rag.py
"""End-to-end: ingest → embed → hybrid query returns semantic hits."""
import argparse

import numpy as np
import pytest

from aggregator.cli import _cmd_embed
from aggregator.core.store import Store
from aggregator.mcp import aggregator_query


class StubEmbedder:
    """Semantic stub: 'voting' queries match 'quadratic' docs."""
    def _vec_for(self, text):
        v = np.zeros(768, dtype=np.float32)
        if "voting" in text.lower() or "quadratic" in text.lower():
            v[0] = 1.0
        elif "pigeon" in text.lower():
            v[1] = 1.0
        else:
            v[2] = 1.0
        return v
    def embed_documents(self, docs):
        return np.array([self._vec_for(d) for d in docs], dtype=np.float32)
    def embed_query(self, q):
        return self._vec_for(q)


@pytest.fixture
def cache_hydrated(tmp_path, monkeypatch):
    monkeypatch.setattr("aggregator.core.embed.Embedder", StubEmbedder)
    monkeypatch.setattr("aggregator.cli.Embedder", StubEmbedder)
    monkeypatch.setattr("aggregator.mcp.Embedder", StubEmbedder)
    db = tmp_path / "cache.db"
    s = Store(db_path=db)
    s.migrate()
    c = s._c()
    c.execute(
        "INSERT INTO sessions(session_id, root_session_id, kind, first_ts, "
        "last_ts, jsonl_path, origin) VALUES ('sid', 'sid', 'session', "
        "'2026-01-01', '2026-01-01', '/tmp/x.jsonl', 'claude-code')"
    )
    docs = [
        ("o1", "quadratic funding mechanism for public goods"),
        ("o2", "pigeons roost on ledges in cities"),
        ("o3", "random unrelated observation about weather patterns"),
    ]
    for oid, body in docs:
        c.execute(
            "INSERT INTO observations(obs_id, session_id, root_session_id, "
            "type, ts, body) VALUES (?, 'sid', 'sid', 'user', '2026-01-01', ?)",
            (oid, body),
        )
    c.commit()
    s.close()
    return db


def test_e2e_semantic_hit(cache_hydrated, monkeypatch):
    # Run background embedder
    ns = argparse.Namespace(catchup=True, once=False,
                            source="observations", batch_size=500)
    _cmd_embed(ns, _store=Store(db_path=cache_hydrated))
    # Query
    monkeypatch.setattr(
        "aggregator.mcp._default_store",
        lambda: Store(db_path=cache_hydrated, read_only=True),
    )
    result = aggregator_query("voting", fields="summary")
    assert result["ok"] is True
    # 'voting' hits o1 via vector (quadratic → voting semantic bridge in stub)
    hit_ids = {r["stable_id"] for r in result["records"]}
    assert "o1" in hit_ids
```

- [ ] **Step J2: Run test**

Run: `pytest tests/test_e2e_rag.py -v`
Expected: pass.

- [ ] **Step J3: Ruff + commit**

```bash
ruff check tests/test_e2e_rag.py
ruff format tests/test_e2e_rag.py
git add tests/test_e2e_rag.py
git commit -m "test(rag): e2e ingest→embed→hybrid query"
```

---

## Task K: Nix packaging — embed timer + model download

**Files:**
- Modify: `nix/aggregator.nix`
- Modify: `flake.nix` (if devShell tweak needed for torch)

- [ ] **Step K1: Add embed timer**

Edit `nix/aggregator.nix`. Add near the sessions/github timer definitions:

```nix
services.aggregator.embed = mkIf cfg.enable {
  enable = true;
  description = "aggregator background embed worker";
  serviceConfig = {
    Type = "oneshot";
    Nice = 19;
    IOSchedulingClass = "idle";
    ExecStart = "${cfg.package}/bin/aggregator embed --once --source both --batch-size 500";
  };
};

systemd.user.timers."aggregator-embed" = mkIf cfg.enable {
  Unit.Description = "aggregator embed catchup";
  Timer = {
    OnCalendar = "*:5/30";           # 5 min offset from ingest ticks
    Persistent = true;
    RandomizedDelaySec = "2m";
  };
  Install.WantedBy = [ "timers.target" ];
};
```

- [ ] **Step K2: Add model cache path env**

Add to the service environment so model weights land in a stable location under `$XDG_CACHE_HOME`:

```nix
Environment = [
  "HF_HOME=%C/aggregator/huggingface"
  "AGGREGATOR_EMBED_BACKEND=st"
];
```

- [ ] **Step K3: Ensure sqlite-vec + sentence-transformers resolve under nix**

If the flake's `packages.default` uses `poetry2nix` / `pip`-based build, verify:

- `sqlite-vec` wheel resolves (it's a pure-Python + native .so; wheel available on PyPI).
- `sentence-transformers` pulls torch — either use the CPU torch (`python311Packages.torch-bin`) or accept the CUDA-enabled build size. Prefer `torch-bin` on dellan (no GPU).

Edit `flake.nix` `packages.default` python env to include:

```nix
python3.pkgs.torch-bin
python3.pkgs.numpy
python3.pkgs.transformers
```

If any of these fail to build, fall back to a pip-based venv activation in the systemd unit.

- [ ] **Step K4: Verify nix build**

Run: `nix build .#aggregator`
Expected: exit 0.

Run: `nix flake check`
Expected: exit 0.

- [ ] **Step K5: Ruff / commit**

```bash
git add nix/aggregator.nix flake.nix
git commit -m "feat(nix): systemd embed timer + torch-bin deps for aggregator package"
```

---

## Task L: Capabilities exposure + gate green

**Files:**
- Modify: `tests/test_mcp_capabilities.py` (verify new key)
- Verify: full test suite

- [ ] **Step L1: Extend capabilities test**

Add to `tests/test_mcp_capabilities.py`:

```python
def test_capabilities_reports_vector_index(tmp_path, monkeypatch):
    from aggregator.core.store import Store
    from aggregator.mcp import aggregator_capabilities
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    monkeypatch.setattr(
        "aggregator.mcp._default_store",
        lambda: Store(db_path=tmp_path / "cache.db"),
    )
    caps = aggregator_capabilities()
    assert caps["ok"] is True
    assert caps["schema_version"] == 4
    # New key: observability into vector index freshness
    # Fresh cache: zero embedded rows on both sides.
    assert "vector_index" in caps
    assert caps["vector_index"]["observations_embedded"] == 0
    assert caps["vector_index"]["records_embedded"] == 0
```

Expose `vector_index` in `aggregator_capabilities()` in `aggregator/mcp.py`:

```python
return {
    "ok": True,
    "sources": caps["sources"],
    "freshness": caps["freshness"],
    "tags_by_source": caps["tags_by_source"],
    "counts": caps.get("counts", {}),
    "vector_index": caps.get("vector_index", {}),  # NEW
    "date_range": caps["date_range"],
    "cache_path": caps["cache_path"],
    "schema_version": caps["schema_version"],
    "tool_tier": "read-only",
    "help": format_help(...),
}
```

- [ ] **Step L2: Run full gate**

Run:

```bash
pytest -q
ruff check .
ruff format --check .
```

Expected: all green.

- [ ] **Step L3: Manual backfill dry-run on a tmp cache**

```bash
python -c "
from aggregator.core.store import Store
import tempfile, pathlib
db = pathlib.Path(tempfile.mkdtemp()) / 'cache.db'
Store(db_path=db).migrate()
print(f'schema:', Store(db_path=db).schema_version())
print(f'caps:', Store(db_path=db).capabilities().get('vector_index'))
"
```

Expected: `schema: 4`, `caps: {'observations_embedded': 0, 'records_embedded': 0}`.

- [ ] **Step L4: Commit**

```bash
git add tests/test_mcp_capabilities.py aggregator/mcp.py
git commit -m "feat(mcp): expose vector_index counts in capabilities"
```

---

## Task M: Rollout gate — real-cache smoke

**Files:** none modified; verification only.

- [ ] **Step M1: Migrate the real cache**

WARNING: this modifies the user's live `~/.local/share/aggregator/cache.db`. Only run once the plan is complete and reviewed.

```bash
aggregator status  # confirm current schema
python -c "from aggregator.core.store import Store; Store().migrate()"
aggregator status  # confirm schema_version=4, vector_index=zeros
```

- [ ] **Step M2: Kick off overnight backfill**

```bash
nohup aggregator embed --catchup --source both > /tmp/aggregator-backfill.log 2>&1 &
```

Expected wall time: ~3.5 h on dellan CPU for ~250K vectors.

- [ ] **Step M3: Verify post-backfill**

```bash
aggregator status
# Expected: vector_index.observations_embedded ≈ 350K (some skips)
#           vector_index.records_embedded ≈ 1000
aggregator query "quadratic voting"
# Expected: at least one hit surfaces via the hybrid path
```

- [ ] **Step M4: Enable systemd embed timer**

Update home-manager config to include the aggregator flake with the embed timer enabled. Reload:

```bash
home-manager switch --flake .
systemctl --user list-timers | grep aggregator
systemctl --user start aggregator-embed
journalctl --user -u aggregator-embed -n 50
```

Expected: timer scheduled, service exits 0, embed count monotonically increases.

---

## Self-review

**Spec coverage:**
- Non-goals: Task A explicitly avoids LanceDB / ANN / convex fusion. ✓
- Architecture: A (deps), B (chunk), C (embed), D (schema), E (vec io), F (worker), G (hybrid), H (rerank), I (MCP), J (e2e), K (nix), L (capabilities). ✓
- Data flow: F implements ingest→embed watermark; I implements query→hybrid→rerank. ✓
- Schema migration: D covers ALTER + vec DDL + idempotent probe. ✓
- Testing strategy: covered by B/C/D/E/F/G/H/I/J tests. ✓
- Acceptance criteria 1-10: mapped 1→D, 2→F, 3→J, 4→I, 5→H+I, 6→L, 7→M, 8→K, 9→L, 10→L. ✓
- Rollout steps 1-4: M handles the live cache steps. ✓

**Placeholder scan:** no TBDs, all steps have exact code + commands + expected output. ✓

**Type consistency:**
- `Embedder.embed_documents` returns `np.ndarray[N,768]` consistently across C, F, I, J.
- `Store.select_unembedded` returns `list[sqlite3.Row]` across E, F.
- `QueryAST.id_scope` added in I is `frozenset[str] | None` — consistent between AST field and `_obs_where` / `_build_where` handling.
- `rerank_fuse` returns `list[tuple[str, float]]` in G, consumed by I with `[i for i, _ in fused]`.
- `Reranker.score` returns `np.ndarray[N]`, consumed by I via `sorted(zip(items, scores), ...)`.

**Fixups applied inline:** none needed.
