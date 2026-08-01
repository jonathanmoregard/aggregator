# aggregator

Personal "everything about me, currently, one prompt" aggregator. Caches Claude Code sessions and GitHub records into SQLite+FTS5, exposes a single query DSL via FastMCP (for models) and CLI/Raycast (for humans).

## Status

M0 scaffold. See `docs/superpowers/plans/2026-08-01-aggregator-plan.md` for the chunked build plan and `docs/superpowers/specs/2026-08-01-aggregator-design.md` for the design spec.

## Non-negotiables (from spec)

- Read-only credentials only. GitHub ingester refuses to run against a write-capable token unless `AGGREGATOR_ALLOW_WRITE_TOKEN=1`.
- MCP has NO write tools in v1.
- Scrub (Presidio + gitleaks) pre-store AND pre-return.
- All returned content wrapped in `<ExternalContent source="…">` delimiters.
- Stable local IDs persist across `--rebuild`.

## Dev setup (M0)

```
nix develop
uv sync --extra dev
uv run pytest -q
uv run ruff check .
```
