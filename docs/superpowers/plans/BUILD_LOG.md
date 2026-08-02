# Aggregator build log

One line per milestone landing green. Owner-facing status.

## 2026-08-01

- `cc1ebcb` — **M0 scaffold** — 17 files (304 LOC), 12 unit tests pass, ruff clean, gitleaks clean. Reference `Source` Protocol, `wrap_record`/`wrap_records`, `stable_id_for`, TDD-verified.
- `48f25ff` — **M1a sessions** — sessions.py (203 LOC), 35 pass / 1 skip (real-e2e gated on AGGREGATOR_REAL_E2E=1). Record shape: extra{session_id,project,model,cost_usd,first_user_prompt,top_tool_calls,tail_summary,source_path}; tags=[project,model]; body=USER/ASSISTANT concat.
- `e4918e9` — **M1b github** — github.py (275 LOC), 19 new tests (write-scope refusal, env override, subprocess mocks). Stable ID: github:{owner/repo}:{number} (2 colons — DSL split on first). tags=[repo,kind,state,...] (repo always tags[0]). Total suite 54 pass / 1 skip.
- `5b4db7d..c25d14b` — **M2 dsl+scrub+store** — 3 commits (dsl 116/scrub 181/store 305 LOC). 30 new tests. Presidio+regex fallback. Store.query() returns [] on FTS5 syntax err (logged). Full suite 65 pass / 1 skip.
- `ed0c8c2` — **M3 FastMCP surface** — mcp.py (293 LOC), 29 new tests (4 no-write contract + 13 query + 8 capabilities + 4 ingest gate). Three read-only tools: aggregator_query (wrap+scrub+pagination), aggregator_capabilities (side-effect-free inventory), aggregator_ingest (human-approve gate — returns CLI instructions, never triggers ingest). Contract test = exact allow-list `{query,capabilities,ingest}` AND regex `^(write_|send_|post_|delete_|create_|update_|patch_|put_|manage_|resolve_|insert_|modify_)|(mutate|sudo)/i`. FTS5 syntax err detected via lightweight probe MATCH (store swallows to []). Adapter split: `_tool_*` clean-signature wrappers registered with FastMCP; module-level `_store`-taking fns used by tests (pydantic can't schema Store). Full suite 94 pass / 1 skip, ruff clean.
- `b273528` — **M4 nix** — home-manager module + timers + manual MCP registration doc. flake.nix + nix/aggregator.nix (106 LOC) + README + flake.lock.
- `e85ceb3` — **M5 cli+raycast** — cli.py (172 LOC), 6 tests, raycast wrapper. 102 pass / 1 skip.
- `5b4db7d..c25d14b` — **M2 dsl+scrub+store** — earlier logged.
- `b273528` **M4 nix**, `e85ceb3` **M5 cli+raycast** — earlier logged.
- **M6 close-out (advice-refine-test-loop)** — Phase 1 Opus (4 rounds, opus-4-7 fallback for fable-429), Phase 2 Codex/GPT-5.5 (2 rounds). Fixed: 3 BLOCKERs (ingest not persisted; claude_runner import still present; /search/issues shape mis-parsed as Pulls shape → repo collisions), 6 HIGHs (delimiter injection, scope fail-open, 500-row silent truncation, iter_records ignored since, rebuild non-atomic, transient-wipe with --rebuild), 8+ MEDIUMs (scrub regex fallbacks, ipv6 tighten, WAL+busy_timeout, GhApiError, UTC normalisation, upsert-no-commit closure, tz-safe compare, etc). Final: 148 pass / 1 skip, ruff clean, gitleaks clean. Both advisor phases returned SAFE TO SHIP.

## 2026-08-02 (v2 real-data close-out)
- `6a44155..6de2caa` — v2 Schema B migration (5 commits: schema, sessions v2 parser, dsl, mcp routing, one-shot reingest script)
- `f34b1c7` — real-data smoke transcript: 3 BLOCKERs found by first live run (top: returns 0, FTS matches=0, spawn recovery 0/1170)
- `c2e45b8` **B2 top:** synthesise orphan-root for sessions where only subagents ingested
- `63a42c3` **B3 FTS** populate tool_use body + count matches per exact session
- `972b214` **M1 empty wrap** drop <ExternalContent> in summary mode; also renamed BATCH→batch_size in reingest script
- `f2aa2ba` **B1 spawn recovery** Task→Agent tool rename (2026 Claude Code change); prefer toolUseResult.agentId structured field, regex fallback. **0.0% → 90.3%** on live cache (remaining 10% = parent JSONL not on disk, honest limit).
- `52d45f5` **M2 other-type** widened _KNOWN_TYPES to include attachment (99.8% of what was other), progress, plus 6 future-emitted types. Reclassified live cache. **21.4% → 0.00%**.
- Ingest verify: 5678 sessions, 1170 subagents, 348168 observations, 3.1h wall.
- Suite 191 pass, ruff clean.
- Known remaining: matches=0 counter on FTS+type: hits (hits listed correctly, just count is wrong — advisor round should catch).
