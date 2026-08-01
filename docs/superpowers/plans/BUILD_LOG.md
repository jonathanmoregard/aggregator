# Aggregator build log

One line per milestone landing green. Owner-facing status.

## 2026-08-01

- `cc1ebcb` — **M0 scaffold** — 17 files (304 LOC), 12 unit tests pass, ruff clean, gitleaks clean. Reference `Source` Protocol, `wrap_record`/`wrap_records`, `stable_id_for`, TDD-verified.
- `48f25ff` — **M1a sessions** — sessions.py (203 LOC), 35 pass / 1 skip (real-e2e gated on AGGREGATOR_REAL_E2E=1). Record shape: extra{session_id,project,model,cost_usd,first_user_prompt,top_tool_calls,tail_summary,source_path}; tags=[project,model]; body=USER/ASSISTANT concat.
- `e4918e9` — **M1b github** — github.py (275 LOC), 19 new tests (write-scope refusal, env override, subprocess mocks). Stable ID: github:{owner/repo}:{number} (2 colons — DSL split on first). tags=[repo,kind,state,...] (repo always tags[0]). Total suite 54 pass / 1 skip.
