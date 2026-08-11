"""TickTick source: task history from CSV backups plus a live open-task poll.

Two legs, because neither is sufficient alone:

* CSV backup — the only place completed tasks exist, but manual and stale.
* Open API — always current, but structurally blind to completed tasks.

They are merged by observation recency (newest wins per task), which is what
makes the CSV leg authoritative for completed/abandoned history (a finished
task is never in an API poll at all, so its backup row is unopposed) and the
API leg authoritative for what is currently open (a task marked completed in
last month's backup but served by today's poll correctly reads as open again).

Every emitted record carries ``extra.provenance`` so a search result never
hides which leg it came from.
"""
from __future__ import annotations

import csv
import logging
import os
import shutil
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from aggregator.sources import ticktick_api
from aggregator.sources.base import IngestResult, Record
from aggregator.sources.ticktick_csv import is_ticktick_backup, parse_backup, row_to_record

log = logging.getLogger(__name__)

DEFAULT_BACKUP_DIR = "~/Downloads"

# Aggregator-side overrides, so a unit file can point this leg at a credential
# without a code change. Neither declares a second at-rest COPY of the token
# (2026-08-11 constraint): with both unset, ``resolve_token`` falls through to
# ``$TICKTICK_ACCESS_TOKEN`` and then to the shared ``~/.config/todo/env``
# store that the todo backend rewrites on every OAuth refresh — the one place
# the live token exists.
TOKEN_ENV_VAR = "AGGREGATOR_TICKTICK_TOKEN"
TOKEN_FILE_ENV_VAR = "AGGREGATOR_TICKTICK_TOKEN_FILE"


def _default_archive_dir() -> Path:
    """Where parsed backups are kept so ``--rebuild`` can still see them.

    Data, not state: this is the only surviving copy of a manual export the
    user deletes from ~/Downloads, and nothing can regenerate it. An unset *or
    empty* ``XDG_DATA_HOME`` takes the spec default — reading an empty one
    literally yields a relative path, so the archive would land wherever the
    timer happened to start.
    """
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return Path(base) / "aggregator" / "ticktick" / "backups"


def _note(errors: list[str] | None, message: str) -> None:
    """Log a fault and, when the caller is collecting them, surface it too.

    Both, always. The log line is what an interactive run shows; the ``errors``
    entry is what makes the run report non-empty, which is what a timer turns
    into a failure notification.
    """
    log.warning("%s", message)
    if errors is not None:
        errors.append(message)


def _merge_key(record: Record) -> str:
    """The task id as the record itself minted it.

    Both legs are keyed through this and nothing else. The API leg's
    ``_task_id`` STRIPS before minting a stable_id, so keying the merge on a
    raw payload ``id`` puts a padded value under a key the CSV leg's clean
    ``taskId`` can never match — the same task then survives the merge twice,
    as two rows sharing one stable_id, and which of them the store keeps is
    decided by write order. Reading the key back off the record makes the
    merge key equal the store's own identity by construction.
    """
    return record.stable_id.split(":", 1)[1]


class TickTickSource:
    """Source implementation for TickTick tasks."""

    name = "ticktick"

    def __init__(
        self,
        backup_dir: str | os.PathLike[str] | None = None,
        token: str | None = None,
        token_file: str | None = None,
        state_file: str | os.PathLike[str] | None = None,
        archive_dir: str | os.PathLike[str] | None = None,
    ):
        self.backup_dir = Path(
            backup_dir
            or os.environ.get("AGGREGATOR_TICKTICK_DIR")
            or os.path.expanduser(DEFAULT_BACKUP_DIR)
        )
        self.archive_dir = Path(archive_dir) if archive_dir else _default_archive_dir()
        self.state_file = Path(state_file) if state_file else ticktick_api.default_state_path()
        self._token_arg = token if token is not None else os.environ.get(TOKEN_ENV_VAR)
        self._token_file = (
            token_file if token_file is not None else os.environ.get(TOKEN_FILE_ENV_VAR)
        )
        # Overridable in tests to exercise both sides of the precedence rule.
        self._api_observed_at = datetime.now(UTC)

    def manual_export_input(self) -> str:
        """``sources.base.ReadsManualExport`` — what ``--rebuild`` is refused on.

        The CSV leg. A human exports it out of the TickTick app into
        ~/Downloads and nothing on this machine refreshes it; ``_archive`` keeps
        the only surviving copy once ~/Downloads is cleared. (The API leg is
        unrebuildable for a second, independent reason — see
        ``cli.REBUILD_UNSUPPORTED_SOURCES``.)
        """
        return (
            "a TickTick backup CSV a human exports from the app; nothing on "
            "this machine refreshes it, and the local archive is its only "
            "surviving copy once ~/Downloads is cleared"
        )

    def record_shape(self) -> dict[str, str]:
        """DSL-facing field surface (M2 help generator).

        The union of what both legs write, because the merge means any task can
        arrive from either. ``extra`` is indexed as text and every value in it
        is a ``str`` — including ``completed_time_approx``, which is the string
        ``"true"`` and not a bool, so an exact-match filter has text to match.
        """
        return {
            "subject": "str (task title)",
            "body": "str (task content/notes, plus checklist items on the API leg)",
            "provenance": "str (csv | api | api-inferred-complete)",
            "status": "str (0 open, 2 completed, -1 abandoned; there is no 1)",
            "priority": "str (none | low | medium | high)",
            "due_date": "str (ISO 8601 +0000, may be empty)",
            "start_date": "str (ISO 8601 +0000, may be empty)",
            "repeat": "str (recurrence rule, may be empty)",
            "parent_id": "str (parent taskId, may be empty)",
            "project_id": "str (API leg only)",
            "source_file": "str (backup filename; CSV leg only)",
            "completed_time_approx": 'str ("true" when the completion was inferred)',
        }

    def _backup_files(self, since: datetime | None) -> list[tuple[Path, datetime]]:
        """Return (path, mtime) for every TickTick backup CSV, newest last.

        Both the download dir and the archive are scanned, so a ``--rebuild``
        still sees the deep history after ~/Downloads has been cleared. The
        download dir is scanned second and keyed by filename, so the live copy
        of a file present in both wins.
        """
        found: dict[str, tuple[Path, datetime]] = {}
        for directory in (self.archive_dir, self.backup_dir):
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.csv")):
                try:
                    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
                except OSError:
                    continue
                if since is not None and mtime <= since:
                    continue
                if not is_ticktick_backup(path):
                    continue
                found[path.name] = (path, mtime)
        return sorted(found.values(), key=lambda pair: pair[1])

    def _archive(self, path: Path) -> None:
        """Copy a freshly-seen backup into the archive, best effort.

        A failure here is logged, not raised: the rows have already been read,
        so losing the copy costs a future ``--rebuild`` its deep history but
        must not cost this run its 1302 records.
        """
        if path.parent == self.archive_dir:
            return
        try:
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, self.archive_dir / path.name)
        except OSError as e:
            log.warning("could not archive backup %s: %s", path, e)

    def _csv_candidates(
        self, since: datetime | None, errors: list[str] | None = None
    ) -> dict[str, tuple[datetime, Record]]:
        candidates: dict[str, tuple[datetime, Record]] = {}
        for path, mtime in self._backup_files(since):
            self._archive(path)
            try:
                rows = parse_backup(path)
            except (OSError, UnicodeDecodeError, csv.Error) as e:
                # Detection and parse are two separate reads, and ~/Downloads
                # is live — a browser can replace or truncate a file between
                # them. Per-file failures go to ``errors`` and the scan carries
                # on (partial ingest beats total loss), which is the same policy
                # every other source follows.
                _note(errors, f"ticktick backup {path.name} could not be parsed: {e}")
                continue
            for row in rows:
                record = row_to_record(row, source_file=path.name)
                task_id = _merge_key(record)
                if task_id not in candidates or candidates[task_id][0] <= mtime:
                    candidates[task_id] = (mtime, record)
        return candidates

    def _api_candidates(self, errors: list[str] | None) -> dict[str, tuple[datetime, Record]]:
        # INSIDE a try, deliberately. ``resolve_token`` raises
        # TokenUnavailableError (an OSError subclass) when the API leg is
        # configured but its secret cannot be read — an unreadable token file,
        # or an expired token in the shared ``~/.config/todo/env`` store, which
        # is the failure that actually happens: nothing re-authorizes that
        # token unattended. Resolved outside the try, that exception propagates
        # out of iter_records and takes the CSV leg's 1302 backup rows with it
        # — over a leg that needs no credential at all. The error is still
        # recorded, so the run is loud (2026-08-08 constraint) rather than
        # silently CSV-only.
        try:
            token = ticktick_api.resolve_token(self._token_arg, self._token_file)
        except ticktick_api.TokenUnavailableError as e:
            _note(errors, f"ticktick token unavailable: {e}")
            return {}
        if not token:
            # WARNING, not INFO, and that is load-bearing for the same reason
            # it is in ``ticktick_api._note_inbox_gap``: nothing under
            # ``aggregator/`` configures logging, so ``logging.lastResort``
            # (level WARNING) is the only thing that prints and anything
            # quieter reaches nobody. At INFO this line produced literally no
            # output, so an operator who believed the API leg was configured
            # got a CSV-only run and no way at all to notice.
            #
            # Not an ``errors`` entry: running without the API leg is a
            # supported configuration (the CSV backup carries all the history)
            # and an error here would fire a CRITICAL notification on every
            # timer tick for a user who simply never wanted it. A credential
            # that is CONFIGURED and broken is the loud case, and
            # ``resolve_token`` raises for it above.
            log.warning(
                "no ticktick token configured (checked the explicit token, "
                "$%s / $%s, $%s, and %s); running CSV-only, so completed-task "
                "history will be as old as the newest backup export",
                TOKEN_ENV_VAR,
                TOKEN_FILE_ENV_VAR,
                ticktick_api.TOKEN_ENV_VAR,
                ticktick_api.DEFAULT_ENV_FILE,
            )
            return {}

        observed = self._api_observed_at
        try:
            tasks = ticktick_api.fetch_open_tasks(token, errors=errors)
        except Exception as e:  # noqa: BLE001 -- network, auth, malformed payload
            # Recorded, never swallowed: the run still reports a non-empty
            # errors list (2026-08-08 "fail loudly"), it just does not lose
            # the CSV archive over an outage in the optional leg.
            _note(errors, f"ticktick api poll failed: {type(e).__name__}: {e}")
            return {}

        candidates: dict[str, tuple[datetime, Record]] = {}
        for task in tasks:
            try:
                record = ticktick_api.task_to_record(task)
            except (ValueError, AttributeError) as e:
                # A payload with no usable id, or one that is not an object at
                # all. Skipped rather than allowed to abort the loop: one
                # surprising task must not cost the other 237 — nor, since this
                # runs mid-merge, the CSV archive.
                _note(errors, f"ticktick api: unusable task payload: {e}")
                continue
            candidates[_merge_key(record)] = (observed, record)

        # ``reconcile_open_tasks`` is the whole state protocol in one call:
        # load the previous poll, diff, and only THEN make this poll the new
        # baseline. Hand-rolling load/diff/save is a trap — saving before
        # loading means nothing ever looks disappeared and inference is
        # silently dead, with no error and no warning. It persists internally,
        # so there is no save_state call here.
        state = ticktick_api.JsonFileState(self.state_file)
        try:
            inferred = ticktick_api.reconcile_open_tasks(state, tasks, observed)
        except OSError as e:
            # Persisting the baseline is the one part of the poll that writes
            # to disk. An unwritable state path costs the NEXT run its
            # completion inference; it must not cost this one its archive, and
            # it must not be silent, because a baseline that never updates
            # loses every completion from here on.
            _note(errors, f"ticktick state could not be updated at {self.state_file}: {e}")
            return candidates
        for record in inferred:
            candidates[_merge_key(record)] = (observed, record)
        return candidates

    def iter_records(
        self,
        since: datetime | None,
        errors: list[str] | None = None,
    ) -> Iterator[Record]:
        """Yield one Record per task, newest observation winning per task id."""
        # ``--since`` parses to a naive datetime; backup mtimes are aware, and
        # comparing the two raises TypeError. Normalised once, here, so the
        # scan below can compare unconditionally.
        since_utc: datetime | None = None
        if since is not None:
            since_utc = since if since.tzinfo is not None else since.replace(tzinfo=UTC)

        merged = self._csv_candidates(since_utc, errors)
        for task_id, (observed, record) in self._api_candidates(errors).items():
            if task_id not in merged or merged[task_id][0] < observed:
                merged[task_id] = (observed, record)

        for task_id in sorted(merged):
            yield merged[task_id][1]

    def newest_backup_mtime(self) -> datetime | None:
        """When the newest TickTick backup CSV was last written, or None.

        The backup is a MANUAL export: a human clicks it out of the TickTick
        app and drops it in ~/Downloads, and nothing on this machine refreshes
        it. Without this, a timer re-imports the same months-old file forever
        and reports success every time — the "empty result looks like success"
        shape the fail-loudly constraint exists to stop. Scanned WITHOUT
        ``since``, because the run that skips a stale backup for being outside
        the window is exactly the run that has to be able to report its age.
        """
        files = self._backup_files(None)
        return files[-1][1] if files else None

    def ingest(self, since: datetime | None) -> IngestResult:
        """Count-only path for protocol compat; persistence is the CLI's job."""
        errors: list[str] = []
        added = sum(1 for _ in self.iter_records(since, errors=errors))
        return IngestResult(added=added, updated=0, skipped=0, errors=errors)
