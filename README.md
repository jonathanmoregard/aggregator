# aggregator

Personal "everything about me, currently, one prompt" aggregator. Caches Claude Code sessions and GitHub records into SQLite+FTS5, exposes a single query DSL via FastMCP (for models) and CLI/Raycast (for humans).

## Status

v1: nine registered sources (see [Sources](#sources) below), SQLite+FTS5 store with WAL-mode concurrent-writer safety, four surfaces:

- **FastMCP** (`aggregator-mcp`) — three read-only tools: `aggregator_query`, `aggregator_capabilities`, `aggregator_ingest` (a human-approve gate that only prints the CLI command).
- **CLI** (`aggregator`) — `query`, `ingest SOURCE [--since ISO] [--rebuild]`, `status`, `embed`, `provenance`.
- **Raycast** — scripts in `scripts/raycast/` wrap the CLI for one-shot triage.
- **Nix module** (`nix/aggregator.nix`) — home-manager module with systemd user timers on `*:0/30`, currently for `sessions` and `github` only. Note that this module is not what runs in production: `nixos-config` ships a standalone duplicate (`modules/nixos/aggregator-github-timer.nix`) because this repo is local-only and cannot be a flake input on a CI runner. Every other source is hand-run today.

Design docs: `docs/superpowers/plans/2026-08-01-aggregator-plan.md` (chunked build plan) and `docs/superpowers/specs/2026-08-01-aggregator-design.md` (design spec).

## Sources

`aggregator ingest SOURCE` registers these in `aggregator/cli.py::_default_sources()`:

| source | reads | credential | refreshes unattended? |
|---|---|---|---|
| `sessions` | `~/.claude/projects/**/*.jsonl` — Claude Code session + subagent transcripts | none | yes — local scan, already timer-driven |
| `github` | PRs + issues via `gh api /search/issues` | the `gh` CLI's own auth token; refuses a write-capable token unless `AGGREGATOR_ALLOW_WRITE_TOKEN=1` | yes — already timer-driven |
| `dropbox` | local `~/Dropbox` tree (override `AGGREGATOR_DROPBOX_ROOT`); prose/docs only, see below | none — Dropbox's own client keeps the tree synced | yes — local scan |
| `research` | `~/Repos/research-agent/reports/*.md` (top level only; `_quarantine/` is never read) | none | yes — local scan |
| `sota-watch` | `~/Repos/sota-watch/proposals/*.md` | none | yes — local scan |
| `chatgpt` | a manually downloaded ChatGPT export (`conversations.json` / `conversations-*.json`, bare or inside a zip) from `~/.local/share/aggregator/drops/` (override `AGGREGATOR_DROPS_DIR`) and `~/Downloads` (override `AGGREGATOR_DOWNLOADS_DIR`) | none | **no** — see "Manually downloaded exports" below |
| `claude-web` | a manually downloaded Claude.ai export (`conversations.json`, bare or inside a zip), same two dirs as `chatgpt` | none | **no** |
| `substack` | a manually downloaded Substack export zip (matched on a `posts/*.html` member), same two dirs | none | **no** |
| `ticktick` | TickTick Open API (all projects except Inbox) **and** a manually downloaded backup CSV, `~/Downloads` by default (override `AGGREGATOR_TICKTICK_DIR`) | bearer token, optional — see TickTick below | API leg: yes. CSV leg: **no** |

### Manually downloaded exports

`chatgpt`, `claude-web`, `substack`, and TickTick's CSV backup leg all read a **manually downloaded export archive** — nothing on this machine goes and fetches it. A timer running `aggregator ingest` on one of these sources just re-parses whatever file is already sitting in the drop directory; if that file is three months old, the run still reports success while the index quietly stops gaining new history. The human has to periodically generate a fresh export and drop it into the source's directory (`~/Downloads` for all four, or `~/.local/share/aggregator/drops/` for the three chat/post exports) for these sources to stay current.

Where each export comes from:

- **chatgpt** — ChatGPT's own account data export. Per the 2026-08-02 export-format research in the index, this is request-and-wait (up to 7 days) with a 24h link expiry, delivered by email.
- **claude-web** — Claude.ai's own account data export, delivered similarly by email.
- **substack** — Settings → Exports (documented directly in `aggregator/sources/substack.py`).
- **ticktick (CSV leg)** — TickTick app → Settings → Account → Backup & Import → Generate Backup.

### Dropbox

`AGGREGATOR_DROPBOX_EXCLUDE` is a colon-separated list of glob patterns matched against each file's path relative to the Dropbox root. A pattern excludes both the path itself and everything beneath it — `AGGREGATOR_DROPBOX_EXCLUDE="Private:Work/ClientX"` excludes `Private/anything` and `Work/ClientX/anything` without needing a trailing `/**`.

Only prose/document extensions are indexed (`.md`, `.markdown`, `.txt`, `.docx`, `.pdf`); everything else in the ~25k-file, 4 GB tree (source code, media, `node_modules`) is skipped. No OCR: a PDF with no extractable text layer is skipped as an expected outcome (not an error) once its extracted text falls under 50 characters. Size caps: 2 MB for text/docx files, 20 MB for PDFs; the extracted body itself is truncated at 200,000 characters (`extra.truncated=True` marks a cut record).

A **root that cannot be listed at all** — Dropbox not mounted, not running, `AGGREGATOR_DROPBOX_ROOT` pointing at nothing — is a hard failure, not an empty scan: the run fails loudly rather than reporting `added=0 errors=0`, because nothing was walked and this source has no staleness warning to catch it later. A single **subdirectory** that cannot be listed (permissions) is a per-item error instead: its subtree is missing from the index, one line says so, and the rest of the tree still ingests.

### TickTick

Task history is merged from two legs, by task id, newest observation wins:

- **Open API leg** (`aggregator/sources/ticktick_api.py`) — `GET /open/v1/project` and `/project/{id}/data`. The Open API returns **only open tasks**: every read endpoint filters completed ones out, so this leg alone can never carry completed/abandoned history.
- **CSV backup leg** (`aggregator/sources/ticktick_csv.py`) — parses a manually generated backup CSV. This is the only place completed/abandoned task history exists in this pipeline; a copy is also archived under `$XDG_DATA_HOME/aggregator/ticktick/backups` so a `--rebuild` still sees history after `~/Downloads` has been cleared out.

The merge is what makes the CSV leg authoritative for completed/abandoned history (a finished task is never in an API poll at all, so its backup row is unopposed) and the API leg authoritative for what's open right now (a task the last backup shows completed, but the live poll still serves, correctly reads as open again).

**Credential.** The API leg needs a bearer token. By default it comes from the shared store `~/.config/todo/env` (key `TICKTICK_ACCESS_TOKEN`), which `~/.claude/todo/backends/ticktick.py` rewrites on every OAuth refresh — that's where the live token actually lives. It's overridable with `$TICKTICK_ACCESS_TOKEN`, or pointed at a specific file with `AGGREGATOR_TICKTICK_TOKEN_FILE` (or supplied directly via `AGGREGATOR_TICKTICK_TOKEN`) for a unit file that shouldn't read the todo backend's store. Resolution order: `AGGREGATOR_TICKTICK_TOKEN` → `AGGREGATOR_TICKTICK_TOKEN_FILE` → `$TICKTICK_ACCESS_TOKEN` → the shared store. An **expired** token fails loudly and names the fix: `~/.claude/todo-add --login`. Any other missing/broken token degrades the run to **CSV-only** — recorded as an error, but it never kills the ingest.

**Known coverage gap.** `GET /open/v1/project` does not list the Inbox, so Inbox tasks are invisible to the API leg (both for the live poll and for its completion inference). Measured on the reference export: 59 of 1302 tasks, 5 of the 238 currently-open tasks, live only in the Inbox. The CSV leg still covers them — it reads the whole account, not a project listing.

**Vocabulary.** Status: `0` open, `2` completed, `-1` abandoned — there is no status `1`. Priority is stored as a name, not a number: `none | low | medium | high`.

## How ingestion is structured

Every registered source is a plain object with `iter_records`/`iter_entities` (or the older `ingest`) that `aggregator ingest SOURCE` calls directly. Alongside that, `aggregator/imports/` defines a ports-and-adapters seam for a unified runner: the `ImportAdapter` protocol (`aggregator/imports/port.py`) asks only for a `name` and a single async `get_data()` that yields `Record`/`SessionRow`/`ObservationRow` items, and `aggregator/imports/runner.py` drives every configured adapter concurrently, isolating one source's failure from the rest and folding per-adapter errors and input-freshness into one report. Existing synchronous sources are wrapped onto that port through `SyncSourceAdapter` (`aggregator/imports/sync_bridge.py`), which runs the sync iterator in a worker thread rather than rewriting it — `aggregator/imports/ticktick.py` is the TickTick example. See `docs/superpowers/specs/2026-08-08-dropbox-ticktick-sources-design.md` for the design rationale.

## Provenance — `type:user` is a transport role, not an authorship claim

`type:` records the channel a JSONL line arrived on, and that is **not** who
wrote it. Measured against the vendor's own structural fields over a fixed
3,000-file sample of `~/.claude/projects`, **59% of `type='user'` observations
were composed by a machine**: hook-injected classifier prompts, headless SDK
briefs, subagent task briefs, slash-command output, and client notices. Until
the backfill below has run over a cache, treat every `type:user` result as
"arrived on the user channel", never as "the user said this".

The `observations.provenance` column (schema v6) records the author as one of
five closed values — `human` / `agent` / `hook` / `command` / `system` — and
`NULL` for "not classified yet". `NULL` is the backfill's cursor, exactly as
`embedding_state IS NULL` is the embed worker's; `'unknown'` is never stored,
because "we looked and could not tell" and "nobody has looked" are different
facts.

**`human` is a residual, never a positive claim.** The vendor exposes no
authorship field to make one from: `promptSource='typed'` and
`origin.kind='human'` are transport labels with the same bug one layer down —
the self-compact resume banner is 43 of 43 both, because the harness injects it
through the interactive input channel and the client honestly records how it
arrived. `entrypoint` looks like the signal and is not (2.1.222+ reports
`sdk-cli` for ordinary interactive sessions). So structure and body markers may
only ever produce a positive *machine* claim; if the vendor drops a field the
classifier loses recall rather than mislabelling. The rules live in one place,
`aggregator/core/provenance.py`, and both ingest and the backfill call it.

Query it with `by:` — `by:human`, `by:agent`, `by:hook`, `by:command`,
`by:system`, or `by:machine` for any of the four non-human ones. **Absent, it
filters nothing:** every row comes back carrying a `provenance` field, and when
a page holds machine-authored rows the response's `notice` says how many and
how to exclude them. Nothing defaults to human-only, in the store or in the
tool — that would silently narrow the session-card labels, the frozen eval
baseline and every `matching_observations` count at once. Rows whose
`provenance` is still `NULL` match **no** `by:` value.

Fill the column with:

```
aggregator provenance --backfill      # resumable, chunked, pure UPDATE
aggregator provenance --reclassify    # after CHANGING the classifier
```

It classifies from the JSONL archive where that exists (the structural fields
only live there) and from type + body + the owning session's `kind` for
everything else — the `claude-web` rows whose export is not on disk, live files
the walk skips, sessions whose archive is gone. It writes one column: no
re-ingest, no re-scrub, no `embedding_state` reset. `provenance` is deliberately
**not** part of `_src_hash`, so a classifier revision costs nothing beyond the
re-run; putting it in the digest would re-run Presidio over the whole corpus
(~11 hours at the measured 827 rows/min) and discard the observation vector arm.

## Ingest exit codes

`aggregator ingest SOURCE` exits with one of four codes (defined in `aggregator/cli.py`):

- `0` — clean: the run completed with an empty errors list.
- `1` — hard failure: the source raised, or a `--rebuild` was refused (row-drop guard, empty-rebuild guard, or a declined `--force` confirmation).
- `2` — usage error: unknown source name, an unparseable `--since`, or an unrecognised subcommand.
- `3` — completed with errors (`EXIT_COMPLETED_WITH_ERRORS`): the run finished and wrote what it could, but its errors list is non-empty. A partially-successful run still exits 3, not 0 — distinct from `2` so a systemd wrapper can tell "you typed a bad source name" apart from "the run dropped three PDFs", which need different notifications.

**There is deliberately no fourth code for "finished, known poison present".** Input that can never be parsed — two malformed lines in a JSONL file — is reported loudly the first time its exact identity is seen (exit 3, notification) and is a *note* on every run after that, so a run whose only faults are already-known ones exits `0`. A distinct code would still be non-zero, and `aggregator-ingest.service` treats every non-zero exit as a failure and fires `OnFailure=`, so introducing one would keep notifying every 30 minutes about a file that has been broken since March — the exact alarm fatigue the ledger exists to end. What a non-zero code would have signalled is signalled instead by things a stale unit file cannot suppress: `poison=N` on the run summary, a `note:` line naming each fault under its source, and the full listing in `aggregator status` (file, record count, first-seen date). The ledger itself is `PoisonLedger` in `aggregator/imports/ingest_state.py`.

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
