# Aggregator — design spec (2026-08-01)

## Purpose
Personal "everything about me, currently, one prompt" aggregator. Cache personal data from
several sources into a local store, expose a **single query DSL** front-ended for both a
human (Raycast/CLI) and models (FastMCP server). Chughtai `everything2prompt` pattern,
cohort-validated stack. Sessions (Claude Code JSONLs) is a first-class source alongside
GitHub. More sources added later, one at a time, when their absence hurts.

Requirements (owner): professional, robust, low-maintenance, simple, secure, best-practices.

Reference research: `~/Repos/research-agent/reports/bed893d79ee54c9196659389c8eb2a88.md`.

## Non-goals (YAGNI, documented)
- Vector search / sqlite-vec. Add if FTS proves insufficient for a real query.
- Graphiti / any memory framework (Mem0, Letta, Zep, Cognee). Cohort verdict: overweight
  for single-user aggregation; they solve a different problem.
- Publish frontier (AGENTS.md / llms.txt / JSON-LD Person). Separate future project.
- Web UI. Raycast + CLI suffice.
- Sources beyond sessions + GitHub in v1.
- Write tools on the MCP surface. Adding any later requires human-approve gate + a
  separate write credential.

## Architecture
- Private repo `~/Repos/aggregator`, Python 3.11+, `uv`-managed.
- `flake.nix` exposes a `devShell`, a `packages.default` (the CLI), and a `homeManagerModules.default`.
- Single **FastMCP** server registered as `aggregator`, stdio transport.
- **SQLite + FTS5** at `$XDG_DATA_HOME/aggregator/cache.db` (i.e. `~/.local/share/aggregator/cache.db`).
- Raw source data (session JSONLs referenced by path; GitHub JSON blobs stored) kept immutable
  alongside for audit + rebuild.
- **Per-source ingesters as systemd user timers**, wired by the Nix module.
- CLI entrypoint `aggregator` doubles as the Raycast target.

## Components
- `aggregator/core/store.py` — SQLite schema, FTS5 tables, per-source tables, migration
  runner. **Stable-ID discipline** (Gooen): every entity gets a stable local ID on first
  cache; never trust re-caching without the ID persisted.
- `aggregator/core/dsl.py` — DSL parser: `source:X tag:a,b from:D to:D` + per-source keys
  (`project:`, `status:`, `author:`, etc.). Dynamically-generated help from the cache
  (so the model always sees valid options).
- `aggregator/core/scrub.py` — Presidio + gitleaks middleware. Applied pre-write AND
  pre-return. Wraps returned content in `<ExternalContent source="…">` delimiters.
- `aggregator/sources/base.py` — abstract `Source`:
  - `ingest(since: datetime | None) -> IngestResult`
  - `search(ast: QueryAST) -> list[Record]`
  - `record_shape() -> dict` (used by DSL help generator)
- `aggregator/sources/sessions.py` — walks `~/.claude/projects/*/*.jsonl`. Per-session
  extract: `session_id, project (from encoded cwd), started, ended, model, cost,
  first_user_prompt, files_touched, git_commits_during, top_tool_calls (name+count),
  tail_summary (last user+assistant turn text)`. FTS body = concatenated user turns +
  assistant top-level claims (skip tool schemas, system reminders). Skip files modified in
  the last 5 minutes (live session).
- `aggregator/sources/github.py` — uses `gh api` under the hood (reuse existing `gh auth`
  credential, no new OAuth flow). Cache: my PRs (authored), my issues (authored + assigned),
  review-requested PRs. Per-record: repo, number, title, state, mergeable, checks summary,
  updated_at, url, body_excerpt. Filters: `state:open|closed`, `check:pass|fail|pending`,
  `mergeable:conflict`, `author:@me`.
- `aggregator/mcp.py` — FastMCP server. Tools:
  - `aggregator_query(dsl: str, fields: "summary"|"full" = "summary", page_size: int?, page_token: str?)`
    — the single work tool. Returns records + `notice` when fields omitted + total-count.
  - `aggregator_capabilities()` — read-only inventory: sources present, freshness per source,
    tool tier, cache path.
  - `aggregator_ingest(source: str)` — human-approve gate (never auto-run for the model).
- `aggregator/cli.py` — `aggregator query "source:sessions from:2026-07-25"` (Raycast target),
  plus `aggregator ingest <source>` and `aggregator status`.
- `nix/aggregator.nix` — home-manager module:
  - user systemd timer for each source (sessions hourly; GitHub every 30 min)
  - MCP server registration (writes to `~/.claude.json` via a small activation script or
    documents the manual `claude mcp add` line — decide during Nix chunk based on what
    stays declarative)
  - agenix for any tokens (v1 uses `gh auth token` — no new secret today, but the module
    exposes the seam for later sources)

## Data flow
1. Timer fires → `Source.ingest(since=last_seen)` → new records.
2. Scrub pre-write (Presidio + gitleaks). Redactions logged (not the content, the counts
   and shapes).
3. Extract lightweight fields to SQLite FTS5. Raw kept: sessions referenced by absolute
   path (immutable JSONL already), GitHub raw JSON written under `raw/github/<repo>/<num>.json`.
4. Query flow: DSL parse → per-source `search()` dispatch → results → scrub pre-return →
   wrap in `<ExternalContent source="source:id">` → return with pagination + notice.
5. Rebuild: `aggregator ingest --rebuild <source>` drops that source's tables and re-scans raw.

## Error handling
- Ingest failures logged JSONL to `~/.local/state/aggregator/ingest.log`. Timer retries next tick.
- `claude-runner` (De Leo, PyPI) wraps every LLM call (v1 has none, but the wrapper is
  imported and used for any future call — sets the pattern).
- MCP tools return `{ok: bool, reason?, remediation?}`. Never leak a stack trace to the model.
- Corrupt JSONL: skip file, record failure with path + line, don't abort ingest.
- FTS5 syntax errors from bad DSL: return `ok: false, reason: "…", remediation: "…"` with
  the parsed AST for the model to correct.

## Security
- **Read-only** credentials only. `gh auth token` scope reviewed at ingest time; ingester
  refuses to start with a write-capable token unless `AGGREGATOR_ALLOW_WRITE_TOKEN=1`.
- MCP has **no write tools** in v1. Not now, not "just in case." Adding any later requires
  a documented human-approve gate + a separate credential.
- Scrub applies pre-store AND pre-return (defense in depth). Presidio for PII shapes;
  gitleaks patterns for secrets.
- Content wrapped in `<ExternalContent source="…">` delimiters at aggregator boundary;
  MCP tool docstring tells the model to treat wrapped content as data.
- Return-shape discipline: subject + labels, not body, unless the task demands it. `full`
  is opt-in.
- Repo gitignore: `credentials/`, `.env`, `*.db`, `raw/`, `ingest.log`, `.venv/`,
  `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.ruff_cache/`.
- SQLite plaintext, rely on FDE (documented tradeoff). SQLCipher = follow-up if threat
  model changes.

## Testing
- **Unit** (pytest): DSL parser (property tests for round-trip),
  scrubber (positive: known-shape secrets removed; negative: benign text untouched),
  per-source record extractors (fixtures under `tests/fixtures/`).
- **Integration**: real SQLite temp file, synthetic source data, e2e query round-trip.
  Assert stable-ID discipline across a rebuild (IDs persist).
- **Real-data smoke** (opt-in via env var): read-only fixture snapshot of a few real
  session JSONLs (committed under `tests/fixtures/real-sessions/`). Assert a known query
  returns a known session.
- **Security tests**: assert scrubber removes known-shape secrets, assert MCP server
  exposes no write tool (list-tools contract test), assert scope check refuses a
  write-capable `gh` token.
- **No LLM in tests** unless mocked.

## Milestones (proposed for the plan step)
- **M0** — repo scaffold, uv/pyproject, ruff config, tests skeleton, flake.nix devShell.
- **M1a** — `sessions` source (parse JSONL → SQLite FTS5). No auth.
- **M1b** — `github` source (`gh api` → SQLite). Parallel with M1a (disjoint files).
- **M2** — DSL parser + `core/store.py` join layer + `core/scrub.py`.
- **M3** — FastMCP surface (`aggregator_query`, `aggregator_capabilities`,
  `aggregator_ingest` behind human-approve).
- **M4** — `nix/aggregator.nix` home-manager module (timers + MCP registration).
- **M5** — CLI + Raycast wrapper script.
- **M6** — advice-refine-test-loop close-out.

## Open items to resolve during build (not blockers)
- Exact home-manager module shape for MCP registration (declarative merge into
  `~/.claude.json` vs generate a `.mcp-registration.sh` invoked by the module). Decide in M4.
- FTS5 tokenizer choice (`unicode61 remove_diacritics 2` default; test against real
  content before locking).
- Session `top_tool_calls` shape: `[{name, count}]` sufficient, or need arg patterns?
  Start simple; extend if a real query needs more.
