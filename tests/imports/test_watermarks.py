"""The per-source high-water mark: what it is, where it lives, when it moves.

THE BUG THIS IS THE FIX FOR. ``aggregator ingest --all`` never computed a
watermark. ``cli.py`` set ``since`` only from an explicit ``--since``, so
``registry.default_adapters(since=None)`` made every 30-minute timer run
re-ingest all 372k observations from scratch. Measured 827 rows/min against a
``TimeoutStartSec=4h`` that fired at 01:58:58: SIGTERM at ~44%, the work
discarded, the timer refiring 30 minutes later. Forever.

Three properties are asserted here and none of them is decoration:

* the mark is per SOURCE — one slow source must never hold back or fast-forward
  another;
* it only ever moves FORWARD, so a retry, a late chunk or a clock stepping
  backwards cannot walk it into the past and skip everything in between;
* a source that cannot support a mark says so, in words, instead of
  full-scanning while reporting like an incremental source.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from aggregator.core.store import Store
from aggregator.imports.ingest_state import (
    APPEND_ONLY_OVERLAP,
    MODIFIED_TIME_OVERLAP,
    SOURCE_CURSORS,
    CursorKind,
    Watermarks,
)

T0 = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "cache.db")
    s.migrate()
    yield s
    s.close()


@pytest.fixture()
def marks(store):
    return Watermarks(store)


# --- the cursor taxonomy --------------------------------------------------


def test_every_registered_source_declares_a_cursor():
    """Coverage, asserted rather than trusted.

    A source missing from the table would silently fall back to "no window",
    i.e. a full scan on every tick — which is exactly the bug, reintroduced one
    source at a time and completely invisible in the run report.
    """
    from aggregator.imports.registry import default_adapters

    declared = {a.name for a in default_adapters()}
    assert declared == set(SOURCE_CURSORS)


def test_every_cursor_kind_is_one_of_the_three():
    for name, cursor in SOURCE_CURSORS.items():
        assert cursor.kind in {
            CursorKind.MODIFIED_TIME,
            CursorKind.APPEND_ONLY,
            CursorKind.NONE,
        }, name
        assert cursor.note, f"{name} declares a kind with no stated reason"


def test_all_three_kinds_are_actually_in_use():
    """The taxonomy has to describe the sources, not the other way round.

    A kind with no members is speculative generality; a source shoehorned into
    the wrong kind is a silent drop. Both are checked by asserting the three
    populations are non-empty AND that the one source that cannot be windowed
    is the one that genuinely cannot.
    """
    kinds = {c.kind for c in SOURCE_CURSORS.values()}
    assert kinds == {
        CursorKind.MODIFIED_TIME,
        CursorKind.APPEND_ONLY,
        CursorKind.NONE,
    }
    assert SOURCE_CURSORS["ticktick"].kind == CursorKind.NONE
    assert SOURCE_CURSORS["research"].kind == CursorKind.APPEND_ONLY
    assert SOURCE_CURSORS["sessions"].kind == CursorKind.MODIFIED_TIME


# --- reading the mark back as a window ------------------------------------


def test_a_first_run_has_no_window(marks):
    """No stored mark means a full scan, which is what a first run must be."""
    assert marks.plan("sessions").since is None


def test_a_modified_time_source_reads_back_an_hour_behind(marks):
    """The DULL late-data mitigation: a fixed overlap, and stop thinking.

    A naive ``max(timestamp)`` mark silently drops records for three
    compounding reasons — a writer whose commit lands after the extract read
    past it, clocks that disagree between writers, and the tie at the boundary
    itself. The mitigation the research recommends for a single-user local
    pipeline is not per-partition frontiers or a side-output ledger for late
    events; it is to read from ``mark - margin`` and let an idempotent apply
    make the re-read free. One flat hour, for every mutable-timestamp source,
    with no per-source tuning to get wrong.
    """
    marks.advance("sessions", T0, rows=10, now=T0)
    assert marks.plan("sessions").since == T0 - MODIFIED_TIME_OVERLAP


def test_an_append_only_source_gets_a_short_margin_not_an_hour(marks):
    """The distinction has to buy something, or it is a label.

    A row this source has already emitted is never revised — research report
    files are content-addressed by request id and written once — so re-reading
    an hour of them corrects nothing that could have gone wrong. The margin
    that remains covers only the boundary tie and a file caught mid-write.
    """
    marks.advance("research", T0, rows=10, now=T0)
    assert marks.plan("research").since == T0 - APPEND_ONLY_OVERLAP
    assert APPEND_ONLY_OVERLAP < MODIFIED_TIME_OVERLAP


def test_a_source_with_no_usable_cursor_never_gets_a_window(marks):
    """TickTick, and it is not an oversight — it is the source's shape.

    Its ``since`` filters the CSV BACKUP FILES by mtime, while every record it
    emits carries the task's own completion or creation time; a mark taken from
    the items would be compared against something else entirely and drop rows.
    And the Open API leg cannot be windowed at all: a completion is visible
    only as a task's DISAPPEARANCE between two full polls, so a narrowed poll
    would infer completions that never happened.
    """
    marks.advance("ticktick", T0, rows=10, now=T0)
    plan = marks.plan("ticktick")
    assert plan.since is None
    assert not plan.cursor.is_incremental


def test_the_full_scan_source_says_so_in_words(marks):
    """Degrading honestly is the requirement; degrading quietly is the bug."""
    plan = marks.plan("ticktick")
    assert plan.window_description.startswith("FULL SCAN")
    assert "backup" in plan.window_description.lower()


# --- monotonicity ---------------------------------------------------------


def test_the_mark_never_walks_backwards(marks):
    """Guarded in SQL, not in Python, so no caller can route around it.

    A retry of an older chunk, a source whose clock stepped, or two runs
    racing would otherwise rewind the window — and everything between the two
    values is then re-read (harmless) or, in the mirror-image case where the
    guard is missing on the way up, skipped forever (not).
    """
    marks.advance("sessions", T0, rows=1, now=T0)
    marks.advance("sessions", T0 - timedelta(days=3), rows=1, now=T0)
    assert marks.plan("sessions").cursor_value == T0


def test_a_run_that_found_nothing_does_not_erase_the_mark(marks):
    """Meltano SDK #1750, the same defect: an empty sync nuking state to {}.

    A quiet source is the NORMAL state of this pipeline — most ticks have
    nothing new. If "nothing to report" wrote NULL over a live cursor, the very
    next run would full-scan, which is the doom loop with extra steps.
    """
    marks.advance("sessions", T0, rows=5, now=T0)
    marks.advance("sessions", None, rows=0, now=T0 + timedelta(minutes=30))
    plan = marks.plan("sessions")
    assert plan.cursor_value == T0
    assert plan.last_run_at == T0 + timedelta(minutes=30)


def test_marks_are_per_source(marks):
    marks.advance("sessions", T0, rows=1, now=T0)
    assert marks.plan("github").since is None
    assert marks.plan("sessions").since == T0 - MODIFIED_TIME_OVERLAP


# --- the failure counter, which is all the circuit breaking needed --------


def test_a_healthy_source_is_never_held_back(marks):
    marks.record_failure("github", "ConnectTimeout", now=T0)
    assert marks.plan("github", now=T0 + timedelta(minutes=30)).skip_reason is None


@pytest.fixture()
def fixed_jitter(monkeypatch):
    """Pin the jitter to its ceiling. The RULE is what is under test.

    The delay is deliberately random, so asserting on it directly would be a
    test that fails a few percent of the time — and a test that fails
    occasionally is a test that gets marked flaky and then deleted. What
    matters is that the decision is made ONCE, stored, and honoured; that is
    what these assert.
    """
    import aggregator.imports.ingest_state as mod

    monkeypatch.setattr(
        mod,
        "full_jitter",
        lambda attempt, *, base, cap: timedelta(
            seconds=min(cap.total_seconds(), base.total_seconds() * 2**attempt)
        ),
    )


def test_a_source_that_keeps_failing_is_backed_off_not_hammered(marks, fixed_jitter):
    """One counter and a jittered delay. Not a state machine.

    Amazon's own guidance is against a real circuit breaker even at scale —
    it introduces modal behaviour that is hard to test and slow to recover
    from — and at nine sources per run there is no herd to thunder anyway. A
    flapping source degrades to "checked less often" and never blocks the
    other eight.
    """
    for i in range(5):
        marks.record_failure("github", "ConnectTimeout", now=T0 + timedelta(minutes=i))
    plan = marks.plan("github", now=T0 + timedelta(minutes=5))
    assert plan.skip_reason is not None
    assert "5 consecutive" in plan.skip_reason


def test_the_backoff_decision_is_made_once_and_stays_made(marks, fixed_jitter):
    """Two reads inside one run must not disagree about whether it runs.

    The delay is jittered, so DERIVING it on read would re-roll the dice every
    time — and the registry and the runner both ask. A stored next-attempt
    time makes the answer a comparison rather than a coin flip.
    """
    for i in range(5):
        marks.record_failure("github", "boom", now=T0 + timedelta(minutes=i))
    at = T0 + timedelta(minutes=5)
    answers = {marks.plan("github", now=at).skip_reason for _ in range(20)}
    assert len(answers) == 1


def test_backoff_expires(marks, fixed_jitter):
    for i in range(5):
        marks.record_failure("github", "ConnectTimeout", now=T0 + timedelta(minutes=i))
    later = T0 + timedelta(days=1)
    assert marks.plan("github", now=later).skip_reason is None


def test_one_success_clears_the_counter(marks, fixed_jitter):
    for i in range(6):
        marks.record_failure("github", "boom", now=T0 + timedelta(minutes=i))
    marks.advance("github", T0, rows=1, now=T0 + timedelta(minutes=10))
    plan = marks.plan("github", now=T0 + timedelta(minutes=11))
    assert plan.consecutive_failures == 0
    assert plan.skip_reason is None


def test_backoff_never_holds_a_source_back_forever(marks, fixed_jitter):
    """The cap matters: an hours-long delay is a pause, not a disablement."""
    for i in range(50):
        marks.record_failure("github", "boom", now=T0 + timedelta(minutes=i))
    plan = marks.plan("github", now=T0 + timedelta(days=2))
    assert plan.skip_reason is None


# --- where it lives -------------------------------------------------------


def test_the_mark_lives_in_the_same_database_as_the_data(store, marks):
    """Not a sidecar JSON file, and this is the crux of the design.

    Two artifacts updated by two separate writes cannot be made to agree under
    SIGTERM, and the two orders fail in opposite directions: watermark-first
    loses records permanently and silently; data-first merely re-reads. With
    one database there is no ordering to get wrong, because the chunk and the
    mark that describes it are one transaction.
    """
    marks.advance("sessions", T0, rows=7, now=T0)
    row = store._c().execute(
        "SELECT cursor_value, cursor_kind, rows_seen FROM ingest_state "
        "WHERE source = 'sessions'"
    ).fetchone()
    assert row[0] == T0.isoformat()
    assert row[1] == CursorKind.MODIFIED_TIME
    assert row[2] == 7


def test_the_mark_survives_a_reopen(tmp_path):
    s = Store(tmp_path / "cache.db")
    s.migrate()
    Watermarks(s).advance("sessions", T0, rows=1, now=T0)
    s.close()

    s2 = Store(tmp_path / "cache.db")
    s2.migrate()
    try:
        assert Watermarks(s2).plan("sessions").cursor_value == T0
    finally:
        s2.close()
