# v2 real-data smoke transcript (2026-08-02 03:15)

Ingest complete at 02:41, elapsed 3.1h. Cache 616MB.

## Counts (verified integrity)
- Top-level sessions: **5678**
- Subagents: **1170**
- Total observations: **348168**
- Date range: 2026-03-10 → 2026-08-01

## Observation type distribution
```
tool_use     83798
tool_result  83768
assistant    80761
other        74404   ← 21% unrecognized types (likely queue-operation, permission-mode, etc)
user         15205
system       10232
```

## DSL smoke — 5 queries against real data

### 1. `session:6d0cd848-…` (root + subagents)
Returned 2 subagent hit summaries, **`matches=0` for each**.
`<ExternalContent>` bodies **empty**. Summary mode omits body — but wrapping empty content is misleading UX.

### 2. `top:6d0cd848-…` (top-level only)
**`# total: 0`** — session exists (has subagents) yet returns nothing. **BLOCKER: `top:` DSL routing broken.**

### 3. `agent:a782813…`
Returned 1 subagent hit with `matches=0`. Wrap body empty. Routing OK.

### 4. `active:2026-07-25..2026-08-01`
Returned real sessions with `matches=1217`. Activity-window query working.

### 5. `type:tool_use suggest_doc_edit` (FTS + type)
Returned subagent hits with `matches=0`. **BLOCKER: FTS matches not being counted or joined correctly.**

## Findings (blockers)

- **B1 `spawned_by_tool_use_id` recovery = 0/1170 = 0.0%** on real data. Brief threshold was ≥80% or redesign. The ts-window heuristic is fundamentally not matching real Task tool_use windows.
- **B2 `top:<sid>` returns 0** on session with confirmed subagents (its subagents matched `agent:`).
- **B3 FTS `matches` count = 0 for every result across queries with body text hits.**

## Findings (mediums)

- **M1 empty wrap** — summary mode returns records with no body; `<ExternalContent source="X">\n\n</ExternalContent>` is cosmetically wrong.
- **M2 74404 observations classified `other`** — 21% of all obs. Parser dropping type info on real message shapes.

Root cause candidates:
- B2: `top:` filter may be checking session_id equality against observations.root_session_id or session_id inconsistently.
- B3: FTS join returning obs rows but matches_count aggregation missing / joining wrong.
- B1: real Task tool_use timing differs from expected window; parent-side Task input may not carry agentId (per research report §5, this is version-dependent).

None of these would have been catchable without real ingest. Advisor loop reviewed the DSL/store impl but had no data to exercise the queries against.
