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
