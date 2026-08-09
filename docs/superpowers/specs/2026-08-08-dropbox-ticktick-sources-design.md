# Design — Dropbox and TickTick sources

Date: 2026-08-08
Status: approved for planning

## Goal

Add two sources to the aggregator index:

1. `dropbox` — prose and documents from the locally-synced `~/Dropbox`.
2. `ticktick` — task history from TickTick, including completed tasks.

Both are records-shaped: they carry a `name`, implement `record_shape()` for
the DSL help generator, and implement `iter_records(since, errors)` yielding
`Record` (`stable_id`, `source`, `subject`, `body`, `tags`, `created_at`,
`updated_at`, `extra` — see `aggregator/sources/base.py`). Both register in
`_default_sources()` in `aggregator/cli.py`. Neither does chunking or
embedding — the RAG hybrid-retrieval plan operates generically over the
`records` table, so these inherit vectors when that work lands.

## Source 1 — `dropbox`

### Discovery

Root from `AGGREGATOR_DROPBOX_ROOT`, default `~/Dropbox`. Recursive walk.

Included extensions: `.md`, `.markdown`, `.txt`, `.docx`, `.pdf`.

Excluded unconditionally: any path segment matching `node_modules`, `.git`,
`.dropbox.cache`, or a dot-directory. Additional user exclusions come from
`AGGREGATOR_DROPBOX_EXCLUDE`, a colon-separated list of glob patterns matched
against the path relative to the root. Default empty.

The exclude knob ships from day one rather than being deferred: this index is
exposed to Claude over MCP and the Dropbox tree contains contracts, health
records, and coaching material. Users need a way to keep folders out without
patching code.

### Extraction

| Type | Method |
| --- | --- |
| `.md`, `.markdown`, `.txt` | direct read, UTF-8 with `errors="replace"` |
| `.docx` | `python-docx`, paragraph text joined by newline |
| `.pdf` | `pypdf` text layer only |

No OCR. A PDF whose extracted text is under 50 characters is treated as
image-only: skipped, counted in a summary, and **not** appended to `errors`
(it is an expected outcome, not a failure).

Size limits, applied before extraction:

- text files over 2 MB are skipped (this excludes the two 5.4 MB Swedish
  wordlists, which are the only files over the limit today)
- PDFs over 20 MB are skipped
- extracted body is truncated to 200,000 characters, with
  `extra.truncated = true` set

Extraction failures (corrupt PDF, malformed docx) append one line per file to
`errors` and skip that file. One bad file never aborts the ingest.

### Record mapping

- `stable_id`: `dropbox:<relpath>` via `stable_id_for()`
- `subject`: first markdown ATX heading if the file has one, else the filename
  stem
- `body`: extracted text
- `tags`: top-level folder name, plus the extension without the dot
- `created_at`: unset — filesystem birth time is unreliable across the Dropbox
  sync boundary
- `updated_at`: file mtime, ISO 8601
- `extra`: `relpath`, `ext`, `size_bytes`, and `truncated` when applicable

### Incremental behaviour

`--since` filters on mtime: files with `mtime <= since` are not read at all,
so an incremental run costs a `stat` per file rather than a parse.

A renamed file produces a new `stable_id` and leaves the old row orphaned.
This is accepted; `--rebuild` clears it. Content-hash identity was considered
and rejected as more machinery than the problem warrants.

## Source 2 — `ticktick`

TickTick's official Open API cannot return completed tasks — confirmed against
two independent sources, and the reason every TickTick MCP server (including
the first-party hosted one) is unsuitable as the sole feed for a history
index. Completed history is only available from the manual CSV backup export.

The source therefore has two legs, merged before yielding.

### Leg A — CSV backup (archive, authoritative)

Directory from `AGGREGATOR_TICKTICK_DIR`, default `~/Downloads`. Every `*.csv`
is opened and validated by structure, not filename: TickTick backups carry six
metadata lines followed by a header on line 7. Files that do not match are
ignored silently.

Columns consumed: `Folder Name`, `List Name`, `Title`, `Tags`, `Content`,
`Start Date`, `Due Date`, `Repeat`, `Priority`, `Status`, `Created Time`,
`Completed Time`, `taskId`, `parentId`. `Status` is `0` normal, `1` completed,
`2` archived. `Content` is multiline-quoted and must be parsed with the `csv`
module, not split.

On successful parse, the CSV is copied into
`$XDG_DATA_HOME/aggregator/ticktick/backups/`. This makes `--rebuild`
reproducible after the file is cleared out of `~/Downloads`, which would
otherwise silently destroy the deep archive.

### Leg B — Open API delta (freshness)

Bearer token read from `AGGREGATOR_TICKTICK_TOKEN`, or from a file named by
`AGGREGATOR_TICKTICK_TOKEN_FILE` (the agenix path). If neither is set, leg B is
skipped entirely and the source runs CSV-only — no error.

Calls `GET /open/v1/project` then `GET /open/v1/project/{id}/data` per project,
yielding currently-open tasks. Undocumented endpoints reported for completed
tasks are deliberately not used; they could not be verified.

The token grants write scope. Following the precedent set by the GitHub source,
the source refuses to issue any non-GET request, and this is asserted in tests.

### Inferred completions

Leg B needs prior state to notice a task leaving the open set. State lives at
`$XDG_STATE_HOME/aggregator/ticktick/open_tasks.json`, mapping task id to the
last-seen task payload plus a `last_seen` timestamp.

On each API poll: task ids present last time but absent now are emitted as
records built from their stored payload, with `Status` treated as completed and
completion time set to the poll timestamp. These carry
`extra.provenance = "api-inferred-complete"` and
`extra.completed_time_approx = true`, so an approximate timestamp is never
mistaken for a real one.

A missing or corrupt state file means no inferences that run — the file is
rewritten and inference resumes next poll. It is a cache, not a database.

### Merge and precedence

Both legs produce candidate records keyed by task id. Before yielding, the
source resolves each id to one record by observation recency:

- a CSV row wins if its file's mtime is newer than the API observation
- otherwise the API record wins

This handles the un-complete case (a task completed in an old CSV but open in
the live API correctly reads as open) without special-casing it. Every record
carries `extra.provenance` of `csv`, `api`, or `api-inferred-complete`.

### Record mapping

- `stable_id`: `ticktick:<taskId>`
- `subject`: `Title`
- `body`: `Content`
- `tags`: task tags, plus `List Name`, `Folder Name`, and a status word
  (`open` / `completed` / `archived`)
- `created_at`: `Created Time`
- `updated_at`: `Completed Time` when present, else `Created Time`
- `extra`: `priority`, `due_date`, `start_date`, `repeat`, `parent_id`,
  `status`, `provenance`, `source_file`, and `completed_time_approx` when
  applicable

### Incremental behaviour

`--since` skips CSV files with `mtime <= since`. Leg B always polls in full —
the open-task set is small and the API exposes no cursor.

## Dependencies

`pypdf` and `python-docx` are added to the main `dependencies` list in
`pyproject.toml`. The project runs under `uv run`, so no Nix packaging is
involved.

The repo keeps its dependency list deliberately short, and an optional extras
group was considered. It was rejected: with extras, a missing install turns 590
Dropbox files into a silent gap in search results. Two small pure-Python
libraries are the cheaper failure mode.

No HTTP client dependency is added. Leg B is the first direct HTTP call in the
repo — the GitHub source shells out to `gh` instead — so it uses
`urllib.request` from the standard library rather than introducing `requests`
or `httpx` for a handful of GETs.

## Testing

Tests live in `tests/sources/test_dropbox.py` and `tests/sources/test_ticktick.py`,
following the fixture-builder pattern of `tests/sources/test_substack.py`.

Dropbox coverage: extension filtering; exclusion of `node_modules` and dot-dirs;
`AGGREGATOR_DROPBOX_EXCLUDE` globs; size caps and truncation flag; heading vs
filename subject; mtime `--since` filtering; a corrupt PDF appending to `errors`
without aborting; an image-only PDF being skipped without an error.

TickTick coverage: CSV preamble parsing and header detection; a non-TickTick CSV
in the same directory being ignored; multiline quoted `Content`; status mapping;
backup copying; leg B absent when no token is configured; inferred completion
across two polls using a temp state file; corrupt state file recovery;
CSV-versus-API precedence in both directions; refusal to issue non-GET requests.

Gate: `uv run pytest -q` and `uv run ruff check .`, both exit 0.

## Out of scope

- Systemd timers and the agenix secret for the TickTick token. These belong to
  the nixos-config repo and ship through its worktree/PR pipeline;
  `modules/nixos/aggregator-github-timer.nix` is the template.
- The unofficial TickTick v2 API (cookie auth). Rejected: undocumented, unknown
  cookie lifetime, and reports of rate-limiting and account lockout.
- OCR for image-only PDFs.
- Indexing source code, media, or JSON from Dropbox.
