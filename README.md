# aggregator

Personal "everything about me, currently, one prompt" aggregator. Caches Claude Code sessions and GitHub records into SQLite+FTS5, exposes a single query DSL via FastMCP (for models) and CLI/Raycast (for humans).

## Status

v1: two sources (Claude Code sessions, GitHub PRs + issues via `gh api /search/issues`), SQLite+FTS5 store with WAL-mode concurrent-writer safety, four surfaces:

- **FastMCP** (`aggregator-mcp`) — three read-only tools: `aggregator_query`, `aggregator_capabilities`, `aggregator_ingest` (a human-approve gate that only prints the CLI command).
- **CLI** (`aggregator`) — `query`, `ingest SOURCE [--since ISO] [--rebuild]`, `status`.
- **Raycast** — scripts in `scripts/raycast/` wrap the CLI for one-shot triage.
- **Nix module** (`nix/aggregator.nix`) — home-manager module with systemd user timers on `*:0/30` for both sources.

Design docs: `docs/superpowers/plans/2026-08-01-aggregator-plan.md` (chunked build plan) and `docs/superpowers/specs/2026-08-01-aggregator-design.md` (design spec).

## Non-negotiables (from spec)

- Read-only credentials only. GitHub ingester refuses to run against a write-capable token unless `AGGREGATOR_ALLOW_WRITE_TOKEN=1`.
- MCP has NO write tools in v1.
- Scrub (Presidio + gitleaks) pre-store AND pre-return.
- All returned content wrapped in `<ExternalContent source="…">` delimiters.
- Stable local IDs persist across `--rebuild`.

## Dev setup

```
nix develop
uv sync --extra dev
uv run pytest -q
uv run ruff check .
```
