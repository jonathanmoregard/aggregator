# Shared MCP Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (default) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move aggregator model and Presidio state from every MCP pane into one shared loopback backend while preserving existing stdio client configuration.

**Architecture:** `aggregator.mcp.main()` selects a FastMCP stdio proxy only when `AGGREGATOR_MCP_BACKEND_URL` is non-empty. Otherwise it builds, warms, and serves the existing aggregator server. NixOS later deploys the same artifact as one HTTP backend plus many lightweight stdio proxies.

**Tech Stack:** Python 3.11, FastMCP, pytest, ruff, uv.

---

### Task 1: Pin proxy selection with failing tests

**Files:**
- Modify: `tests/test_mcp_startup_warmup.py`

- [ ] Add `test_main_proxies_without_building_or_warming` that sets `AGGREGATOR_MCP_BACKEND_URL=http://127.0.0.1:8765/mcp`, replaces `fastmcp.server.create_proxy`, makes `build_server` and `start_background_init` fail if called, invokes `main()`, and asserts the proxy received the URL then ran with `show_banner=False`.
- [ ] Parameterize the existing direct startup test with missing and empty backend values, proving both still order calls as `build`, `warm`, `run`.
- [ ] Run `uv run pytest -q tests/test_mcp_startup_warmup.py`; expect the proxy test to fail because current `main()` calls `build_server()`.

### Task 2: Implement minimal proxy mode

**Files:**
- Modify: `aggregator/mcp.py`
- Modify: `tests/test_mcp_startup_warmup.py`

- [ ] In `main()`, read and strip `AGGREGATOR_MCP_BACKEND_URL`.
- [ ] For a non-empty value, lazily import `create_proxy` from `fastmcp.server`, construct the proxy from the URL, run it with `show_banner=False`, and return before server construction or warm-up.
- [ ] Keep the existing direct build/warm/run block unchanged for absent or whitespace-only configuration.
- [ ] Update the module security/startup commentary to describe the deployed backend boundary without weakening the no-model-import startup contract.
- [ ] Run `uv run pytest -q tests/test_mcp_startup_warmup.py`; expect all tests green.
- [ ] Run `uv run ruff check aggregator/mcp.py tests/test_mcp_startup_warmup.py` and `uv run python -m py_compile aggregator/mcp.py`; expect exit 0.

### Task 3: Prove compatibility and publish reviewed commit

**Files:**
- No source changes unless a gate finds a defect.

- [ ] Run `uv run pytest -q tests/test_mcp_cold_start.py tests/test_mcp_no_write_tools.py tests/test_mcp_discoverability.py tests/test_schema_health_probe.py`.
- [ ] Run `uv run pytest -q` and `uv run ruff check .`.
- [ ] Run `git diff --check`, inspect the complete diff, and commit implementation separately from the design commit.
- [ ] Push `feat/shared-mcp-backend`, open a PR against `main`, and record the pushed implementation SHA for the dependent NixOS pin.
- [ ] Run `python3 /home/jonathan/.claude/scripts/dod-check.py --repo /home/jonathan/Repos/aggregator`; expect only remote checks or the deliberate merge click to remain.
