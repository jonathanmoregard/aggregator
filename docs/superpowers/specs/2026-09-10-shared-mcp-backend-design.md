# Shared MCP backend

Date: 2026-09-10
Status: approved

## Problem

Every Claude and Codex pane starts its own `aggregator-mcp` process. Each
process imports the query stack, initializes Presidio, and loads its own
embedding/reranking models. On the live host, 17 processes across nine Codex
parents retained about 2.9 GiB of resident anonymous memory and 14 GiB of
swap. This duplicates read-mostly model state per pane and contributed to a
period where RAM and the full 31.3 GiB swap pool were exhausted.

## Decision

Run one long-lived aggregator MCP server over FastMCP Streamable HTTP on
`127.0.0.1`. Keep each client on its existing stdio command, but make that
command a small FastMCP proxy when `AGGREGATOR_MCP_BACKEND_URL` is set.

This keeps Claude and Codex manifests, lifecycle, and schema-health probes
unchanged. Per-pane children remain, but no longer construct an aggregator
server or initialize Presidio/model state. The shared backend owns that state
once.

`aggregator-mcp` keeps its current direct stdio behavior when the environment
variable is absent. The same artifact can therefore serve both roles:

- deployed client wrapper sets `AGGREGATOR_MCP_BACKEND_URL` and runs a stdio
  proxy;
- backend wrapper leaves it unset and selects FastMCP HTTP transport, loopback
  host, fixed port, and `/mcp` endpoint.

The proxy must not call `build_server()` or `start_background_init()`. Backend
startup retains both calls and all existing tool/resource behavior.

## Failure behavior

If the backend is unavailable, proxy startup or its first request fails loudly
through the existing MCP client boundary. No proxy falls back to an in-process
server: fallback would silently recreate the memory problem during an outage.
The systemd user service restarts after failure.

The backend binds only loopback. It gets explicit memory pressure and maximum
limits, read-only home access except for aggregator state, loopback-only network
access, and standard process/filesystem hardening. It remains outside the
disposable heavy-job slice because killing a query backend is not useful build
pressure relief.

## Alternatives rejected

1. Point every MCP manifest directly at HTTP. This removes per-pane proxies,
   but breaks current command-based schema-health resolution and requires a
   coordinated Claude/Codex configuration migration.
2. Keep independent servers and cap each process. Caps reduce blast radius but
   preserve multiplied model state and make normal concurrent use unreliable.
3. Use a Unix socket. FastMCP's supported deployment transport is HTTP; a
   loopback listener plus systemd address restrictions provides a small,
   testable boundary without a custom transport.

## Verification

Focused tests will prove that a configured backend URL selects `create_proxy`,
never builds local model state, and runs the proxy through stdio. A second test
will prove unset or empty configuration retains current in-process behavior.
Existing startup-contract, schema, no-write, and full-suite tests remain green.

The dependent NixOS change will pin the reviewed aggregator commit, package
separate proxy/backend launchers, start one hardened user backend, and prove in
the VM that multiple proxy processes share one backend without creating local
servers.
