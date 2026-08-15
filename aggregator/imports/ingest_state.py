"""State that belongs to an ingest RUN rather than to any one source.

ONE FILE, SECTIONED, not one file per source. The first tenant is the
staleness episode marker: a source whose hand-downloaded export has gone stale
must be warned about ONCE and then go quiet, and something on disk has to
remember that it was. Four of the nine sources can go stale (``chatgpt``,
``claude-web``, ``substack``, and ticktick's CSV leg) and only TickTick has a
state file of its own — giving the other three one apiece would put four
files, four load/save pairs and four sets of permission and atomicity rules
where one will do, and the next ingest-level marker would make it five.

WHERE, AND WHY IT IS STATE. ``$XDG_STATE_HOME/aggregator/ingest/markers.json``,
beside ``$XDG_STATE_HOME/aggregator/ticktick/open_tasks.json`` and for the same
reason: it is regenerable — losing it costs one extra warning — but nothing
else can reconstruct it, so it is not a cache. An unset *or empty* variable
takes the spec default; reading an empty one literally yields a RELATIVE path,
so the markers would be written wherever the timer happened to start and the
next run, started elsewhere, would find none and warn all over again.

0600 AND A DURABLE RENAME, both copied from ``ticktick_api._write_state``. The
mode is set on the scratch fd BEFORE any bytes are written, so there is no
window at 0644, and applied explicitly rather than through ``O_CREAT``'s mode
argument, which does nothing when a scratch file from an earlier run already
exists. The content is a list of the user's source names and export dates —
not a credential, but not something a state file should publish either.

"Durable" rather than merely "atomic" because the rename alone only guarantees
that no READER sees a half-written document; the bytes behind it can still be
in the page cache when the power goes, leaving a zero-length markers file. Here
that is the mildest of the three write paths that share this recipe — reading
fails safe, so a lost file costs one repeated warning — but it is the same
recipe, and a copy that quietly drops a step is how the recipe rots. See
``core/durable.py``.

READING FAILS SAFE, WHICH IS THE OPPOSITE OF THE TICKTICK BASELINE. There the
absent file and the broken file are opposite answers and a broken one RAISES,
because the caller's next act would overwrite unrecoverable completions. Here
the only thing a marker buys is silence, so an unreadable file must resolve to
"nothing is suppressed" — one toast too many, never one too few — and may be
replaced by the next write. A broken file is logged at WARNING (nothing under
``aggregator/`` configures logging, so ``logging.lastResort`` is what prints
and anything quieter reaches nobody) rather than raised: the dedup being dead
is worth saying, and it must not cost the run its ingest.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from aggregator.core.durable import flush_to_disk, replace_durably

log = logging.getLogger(__name__)

# The section holding one marker per source whose input has gone stale. Keyed
# by adapter name; see ``runner.plan_staleness_report`` for what a value means.
STALE_INPUTS = "stale_inputs"


def default_marker_path() -> Path:
    """Where ingest-level markers live when the caller names no path.

    Same rule as ``ticktick_api.default_state_path``, including the empty-value
    trap: ``XDG_STATE_HOME=`` resolves to a relative path, which would scatter
    the markers across whatever directory each run started in.
    """
    base = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
    return Path(base) / "aggregator" / "ingest" / "markers.json"


@contextmanager
def _marker_lock(path: Path) -> Iterator[None]:
    """Serialise marker writes across processes, via a sidecar lock file.

    A sidecar rather than the file itself, because the file is replaced by
    ``rename`` on every save: a lock held on the old inode says nothing about
    the new one. Ingests DO overlap (a timer firing over a manual run), and the
    two would otherwise share one ``.tmp`` scratch path and interleave their
    bytes in it.

    No compare-and-swap, unlike the open-task baseline. There the loser of a
    race destroys completions the Open API can never re-serve; here it costs
    one duplicated warning, which the next run corrects by itself.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


@dataclass(frozen=True)
class IngestMarkers:
    """The marker document as one JSON file. Sectioned, so it stays one file.

    Thin on purpose: it knows about bytes and permissions and nothing about
    what a marker means. The episode rules live in ``imports/runner.py``, with
    the code that computes the warning a marker suppresses, so "stale" cannot
    come to mean one thing where the warning is raised and another where it is
    silenced.
    """

    path: Path = field(default_factory=default_marker_path)

    def _read(self) -> dict[str, object]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
            log.warning(
                "aggregator ingest markers at %s could not be read (%s: %s); "
                "treating every marker as absent, so anything they were "
                "suppressing is reported again. The next run that has "
                "something to record replaces the file",
                self.path,
                type(e).__name__,
                e,
            )
            return {}
        if not isinstance(data, dict):
            log.warning(
                "aggregator ingest markers at %s hold a %s, not a section map; "
                "treating every marker as absent",
                self.path,
                type(data).__name__,
            )
            return {}
        return data

    def load(self, section: str) -> dict[str, dict]:
        """One section's markers, keyed by source. ``{}`` when there are none.

        NEVER RAISES, and every failure answers ``{}`` — which means "nothing is
        suppressed", i.e. warn. See the module docstring for why that direction
        is not negotiable.
        """
        value = self._read().get(section)
        if not isinstance(value, dict):
            return {}
        return {name: mark for name, mark in value.items() if isinstance(name, str)}

    def save(self, section: str, markers: Mapping[str, dict]) -> None:
        """Replace one section, leaving every other section alone.

        Read-modify-write INSIDE the lock: a second section (a future
        ingest-level marker) must not be lost to a staleness write that read the
        document before it was added.

        Raises ``OSError`` when the file cannot be written. The caller reports
        it rather than failing the run — the ingest itself succeeded and the
        cost is one repeated warning — but a marker that silently never lands
        turns "reported once" back into "reported every 30 minutes".
        """
        with _marker_lock(self.path):
            document = self._read()
            document[section] = dict(markers)
            scratch = self.path.with_name(self.path.name + ".tmp")
            fd = os.open(scratch, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    os.fchmod(handle.fileno(), 0o600)
                    handle.write(json.dumps(document))
                    flush_to_disk(handle)
            except BaseException:
                scratch.unlink(missing_ok=True)
                raise
            replace_durably(scratch, self.path)


def stale_input_markers(path: Path | None = None) -> dict[str, dict]:
    """Every source whose staleness warning is currently suppressed.

    THE VISIBLE HALF of the report-once rule, exactly as
    ``ticktick_api.uncovered_projects`` is for a vanished project: suppressing a
    repeat is only defensible while the suppressed state is somewhere a human
    can go and look, or "quiet" has become "forgotten". ``aggregator status``
    prints this.

    Never raises, for the same reason that one does not — a read-only report
    that cannot show anything at all is worse than one showing nothing.
    """
    return IngestMarkers(path or default_marker_path()).load(STALE_INPUTS)


__all__ = [
    "STALE_INPUTS",
    "IngestMarkers",
    "default_marker_path",
    "stale_input_markers",
]
