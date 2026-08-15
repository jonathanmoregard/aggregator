"""Round-2 HIGH-2: ``except TypeError`` cannot tell a signature from a bug.

Both ingest paths used to call the source with ``errors=`` inside a
``try``/``except TypeError`` and, on TypeError, RE-ITERATE it without the
kwarg. Argument binding raises TypeError at call time, so that caught the
old-signature case it was written for — and equally caught a genuine TypeError
raised from arbitrary depth INSIDE the iteration, at which point the retry
silently ran the source a second time.

Iterating a source is not a read-only act. The TickTick poll's
``reconcile_open_tasks`` diffs the live open-task set against the previous
poll and then makes this poll the new baseline, on disk, during iteration. A
second pass therefore sees no disappearances: the API only ever serves OPEN
tasks, so an inferred completion is reported exactly once and the retry loses
it for good — while the run exits 0, because the retry succeeded.

The fix probes the signature (``accepts_errors_kwarg``) instead, so the
fallback only ever fires for a source that genuinely lacks the parameter.
"""
from __future__ import annotations

from datetime import UTC, datetime

from aggregator.cli import main
from aggregator.core.store import Store
from aggregator.sources.base import Record, SessionRow


def _record(name: str) -> Record:
    return Record(stable_id=f"t:{name}", source="t", subject=name, body="b")


class _TickTickLike:
    """A source whose iteration CONSUMES state, then raises a real TypeError.

    Poll 1 reports a completion it inferred from the baseline diff and then
    advances the baseline. Poll 2 cannot see that disappearance again — which
    is exactly why the retry is not a harmless repeat.
    """

    name = "t"

    def __init__(self) -> None:
        self.polls = 0

    def iter_records(self, since, errors=None):
        self.polls += 1
        if self.polls == 1:
            yield _record("inferred-complete")
            raise TypeError("unsupported operand type(s) for -: 'str' and 'int'")
        yield _record("still-open")


class _EntityTickTickLike:
    """The same shape on the entity path (the second copy of the pattern)."""

    name = "t"

    def __init__(self) -> None:
        self.polls = 0

    def iter_entities(self, since, errors=None):
        self.polls += 1
        if self.polls == 1:
            yield _session("inferred-complete")
            raise TypeError("unsupported operand type(s) for -: 'str' and 'int'")
        yield _session("still-open")


class _OldStyleRecords:
    """A source predating the ``errors`` sink. The fallback's real customer."""

    name = "t"

    def __init__(self) -> None:
        self.calls = 0

    def iter_records(self, since):
        self.calls += 1
        yield _record("legacy")


class _OldStyleEntities:
    name = "t"

    def __init__(self) -> None:
        self.calls = 0

    def iter_entities(self, since):
        self.calls += 1
        yield _session("legacy")


def _session(sid: str) -> SessionRow:
    now = datetime(2026, 8, 11, tzinfo=UTC)
    return SessionRow(
        session_id=sid,
        root_session_id=sid,
        parent_session_id=None,
        kind="session",
        agent_id=None,
        agent_type=None,
        spawned_by_tool_use_id=None,
        cwd="/tmp",
        git_branch=None,
        first_ts=now,
        last_ts=now,
        jsonl_path=f"/tmp/{sid}.jsonl",
    )


def _store(tmp_path) -> Store:
    store = Store(db_path=tmp_path / "cache.db")
    store.migrate()
    return store


def test_a_genuine_typeerror_does_not_re_poll_the_source(tmp_path, capsys):
    """THE finding. The retry used to hide the bug AND consume the baseline:
    exit 0, one poll's inferred completions gone, nothing in ``errors``."""
    src = _TickTickLike()

    rc = main(["ingest", "t"], _store=_store(tmp_path), _sources={"t": src})

    assert src.polls == 1, "the source must not be iterated a second time"
    assert rc == 1
    assert "unsupported operand type(s)" in capsys.readouterr().err


def test_a_genuine_typeerror_does_not_re_poll_on_the_entity_path(
    tmp_path, capsys
):
    src = _EntityTickTickLike()

    rc = main(["ingest", "t"], _store=_store(tmp_path), _sources={"t": src})

    assert src.polls == 1
    assert rc == 1
    assert "unsupported operand type(s)" in capsys.readouterr().err


def test_an_old_signature_source_is_still_driven_once(tmp_path):
    """The fallback the ``except TypeError`` existed for has to keep working —
    and has to run the source exactly once, not twice.

    Exit 3, not 0, since round 5: the source runs and its records land (that is
    the point of the fallback), but it was driven with no errors sink, so the
    run cannot vouch for its error count and says so. Ingestion working and the
    run being clean are different claims.
    """
    src = _OldStyleRecords()
    store = _store(tmp_path)

    rc = main(["ingest", "t"], _store=store, _sources={"t": src})

    assert rc == 3
    assert src.calls == 1
    assert store.count_by_source("t") == 1, "the fallback must still ingest"


def test_an_old_signature_entity_source_is_still_driven_once(tmp_path):
    src = _OldStyleEntities()

    rc = main(["ingest", "t"], _store=_store(tmp_path), _sources={"t": src})

    assert rc == 3
    assert src.calls == 1


def test_falling_back_to_an_old_signature_is_itself_reported(tmp_path, capsys):
    """Round-5 inventory. The fallback silently un-wires the run's errors sink.

    A source whose iterator takes no ``errors`` is driven without one, so every
    per-item failure it hits internally — an unreadable file, a dropped row —
    has nowhere to land. The run then reports ``errors=0`` and exits 0 while
    the source is quietly shedding data, which is the exact shape the
    fail-loudly constraint exists to stop, one level up from the sources.

    Latent today (all nine registered sources declare ``errors``), so this is
    a regression guard: the day someone adds a source without the parameter,
    or renames it, the run says so instead of going quiet.
    """
    src = _OldStyleRecords()

    rc = main(["ingest", "t"], _store=_store(tmp_path), _sources={"t": src})

    assert src.calls == 1, "the source must still be driven, and exactly once"
    assert rc == 3, "an un-wired errors sink was not reported at all"
    assert "errors" in capsys.readouterr().err


def test_the_errors_sink_still_reaches_a_modern_source(tmp_path, capsys):
    """And the probe must not cost a current source its error reporting."""

    class _Noisy:
        name = "t"

        def iter_records(self, since, errors=None):
            errors.append("a/file.pdf: parse failed")
            yield _record("ok")

    rc = main(["ingest", "t"], _store=_store(tmp_path), _sources={"t": _Noisy()})

    assert rc == 3
    assert "a/file.pdf" in capsys.readouterr().err
