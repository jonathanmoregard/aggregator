"""``aggregator provenance --backfill``: classify the corpus, resumably.

WHY IT IS A SUBCOMMAND AND NOT PART OF ``migrate()``. ``migrate()`` runs on
EVERY subcommand, read-only queries included. A classification pass over
549,952 rows behind an ``aggregator query`` is exactly the hours-long
unattended surprise the incremental-ingest rules exist to abolish.

THE SHAPE IS THE 2026-08-16 INGEST CONTRACT, in full:

* **Streaming** — the JSONL walk is a generator per file; nothing holds the
  corpus. The only thing materialised is the work list of files that still owe
  rows, which is ~11k paths.
* **History-aware** — ``provenance IS NULL`` IS the watermark, and it lives in
  the same rows it describes. There is no sidecar to fall out of step, and a
  file whose rows are all classified is skipped without being read.
* **Chunked** — bounded batches, committed per chunk, so a kill costs at most
  one chunk and the next run starts where this one stopped.
* **SOTA errors** — a file that cannot be read is recorded and skipped, the
  rest of the run continues, and the run exits non-zero so nobody reads a
  partial pass as a complete one.

IT IS A PURE UPDATE. No re-ingest, no re-scrub, no ``body`` write, no
``src_hash`` write, no ``embedding_state`` reset — the classifier's output must
never be able to cost the corpus a Presidio pass or the vector arm its work.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from aggregator.cli import main
from aggregator.core.provenance import (
    AGENT,
    COMMAND,
    HOOK,
    HUMAN,
    PROVENANCE_VALUES,
    SYSTEM,
)
from aggregator.core.store import Store
from aggregator.sources.base import ObservationRow, SessionRow
from aggregator.sources.sessions import SessionsSource

_TS = datetime(2026, 7, 25, 10, 0, tzinfo=UTC)


def _line(uuid: str, body: str, line_type: str = "user", **extra) -> dict:
    return {
        "sessionId": "sess-p",
        "uuid": uuid,
        "parentUuid": None,
        "timestamp": "2026-07-25T10:00:00.000Z",
        "type": line_type,
        "cwd": "/home/u/proj",
        "gitBranch": "main",
        "message": {"role": line_type, "content": [{"type": "text", "text": body}]},
        **extra,
    }


_LINES = [
    _line("u-human", "please make the PR links clickable"),
    _line("u-sdk", "You are a NixOS config drift analyzer.", promptSource="sdk"),
    _line("u-cmd", "<command-name>/loop</command-name>"),
    _line("u-meta", "context injected", isMeta=True),
    _line("a-1", "here is the answer", line_type="assistant"),
]


def _projects_root(tmp_path: Path) -> Path:
    root = tmp_path / "projects"
    p = root / "proj" / "sess-p.jsonl"
    p.parent.mkdir(parents=True)
    p.write_text("\n".join(json.dumps(o) for o in _LINES), encoding="utf-8")
    old = time.time() - 24 * 60 * 60
    os.utime(p, (old, old))
    return root


def _unclassified_store(tmp_path: Path, root: Path) -> Store:
    """A cache holding the archive's rows with ``provenance`` wiped to NULL.

    That is what a database looks like the moment the v6 ALTER lands: every
    row present, every row unclassified.
    """
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    src = SessionsSource(projects_root=str(root))
    errors: list[str] = []
    s.upsert_entities(list(src.iter_entities(errors=errors)))
    assert errors == []
    c = s._c()
    c.execute("UPDATE observations SET provenance = NULL")
    c.commit()
    return s


def _provenance(store: Store) -> dict[str, str | None]:
    return {
        row["obs_id"]: row["provenance"]
        for row in store._c().execute("SELECT obs_id, provenance FROM observations")
    }


def _run(store: Store, root: Path, *args: str) -> int:
    return main(
        ["provenance", *args],
        _store=store,
        _sources={"sessions": SessionsSource(projects_root=str(root))},
    )


# --- the happy path ---------------------------------------------------------


def test_backfill_classifies_every_row_from_the_archive(tmp_path, capsys):
    root = _projects_root(tmp_path)
    store = _unclassified_store(tmp_path, root)
    assert set(_provenance(store).values()) == {None}

    assert _run(store, root, "--backfill") == 0
    assert _provenance(store) == {
        "u-human": HUMAN,
        "u-sdk": HOOK,
        "u-cmd": COMMAND,
        "u-meta": SYSTEM,
        "a-1": AGENT,
    }


def test_no_row_is_left_unclassified_or_stamped_unknown(tmp_path):
    root = _projects_root(tmp_path)
    store = _unclassified_store(tmp_path, root)
    _run(store, root, "--backfill")
    values = set(_provenance(store).values())
    assert None not in values
    assert "unknown" not in values
    assert values <= set(PROVENANCE_VALUES)


def test_the_db_only_route_covers_rows_with_no_archive_on_disk(tmp_path):
    """3,621 live ``claude-web`` rows have no JSONL anywhere; they must still
    be classified, from type and body, rather than left as a permanent NULL."""
    root = _projects_root(tmp_path)
    store = _unclassified_store(tmp_path, root)
    store.upsert_entities(
        [
            SessionRow(
                session_id="claude-web:c1", root_session_id="claude-web:c1",
                parent_session_id=None, kind="session", agent_id=None,
                agent_type=None, spawned_by_tool_use_id=None, cwd=None,
                git_branch=None, first_ts=_TS, last_ts=_TS,
                jsonl_path="conversations.json", origin="claude-web",
            ),
            ObservationRow(
                obs_id="claude-web:m1", session_id="claude-web:c1",
                root_session_id="claude-web:c1", parent_obs_id=None,
                type="user", ts=_TS, model=None, input_tokens=None,
                output_tokens=None, tool_name=None, tool_use_id=None,
                body="what did we decide about quadratic voting",
            ),
        ]
    )
    store._c().execute("UPDATE observations SET provenance = NULL")
    store._c().commit()

    assert _run(store, root, "--backfill") == 0
    assert _provenance(store)["claude-web:m1"] == HUMAN


def test_a_subagent_stream_is_agent_authored_on_the_db_only_route(tmp_path):
    """``sessions.kind`` is how the DB-only route recovers what
    ``isSidechain`` says at ingest — the class the census missed entirely."""
    root = _projects_root(tmp_path)
    store = _unclassified_store(tmp_path, root)
    store.upsert_entities(
        [
            SessionRow(
                session_id="sess-p:a1", root_session_id="sess-p",
                parent_session_id="sess-p", kind="subagent", agent_id="a1",
                agent_type=None, spawned_by_tool_use_id=None, cwd=None,
                git_branch=None, first_ts=_TS, last_ts=_TS,
                jsonl_path="/nowhere/agent-a1.jsonl",
            ),
            ObservationRow(
                obs_id="sub-1", session_id="sess-p:a1",
                root_session_id="sess-p", parent_obs_id=None, type="user",
                ts=_TS, model=None, input_tokens=None, output_tokens=None,
                tool_name=None, tool_use_id=None,
                body="Research the thing and report back",
            ),
        ]
    )
    store._c().execute("UPDATE observations SET provenance = NULL")
    store._c().commit()
    assert _run(store, root, "--backfill") == 0
    assert _provenance(store)["sub-1"] == AGENT


# --- it is a PURE UPDATE ----------------------------------------------------


def test_the_backfill_rewrites_nothing_but_provenance(tmp_path):
    """No re-scrub, no body write, no ``embedding_state`` reset.

    Those are what a re-ingest would cost — ~11 hours of Presidio over this
    corpus, and the observation vector arm discarded. A classification pass
    must never be able to buy either.
    """
    root = _projects_root(tmp_path)
    store = _unclassified_store(tmp_path, root)
    c = store._c()
    c.execute("UPDATE observations SET embedding_state = 'ok'")
    c.commit()
    before = {
        r["obs_id"]: (r["body"], r["src_hash"], r["embedding_state"])
        for r in c.execute("SELECT obs_id, body, src_hash, embedding_state FROM observations")
    }
    _run(store, root, "--backfill")
    after = {
        r["obs_id"]: (r["body"], r["src_hash"], r["embedding_state"])
        for r in c.execute("SELECT obs_id, body, src_hash, embedding_state FROM observations")
    }
    assert after == before


def test_the_backfill_does_not_touch_the_fts_index(tmp_path):
    """The narrowed ``observations_au`` is what makes this true, and it is the
    difference between 37 s and 7 min plus several hundred MB of bloat."""
    root = _projects_root(tmp_path)
    store = _unclassified_store(tmp_path, root)
    c = store._c()
    before = c.execute("SELECT count(*) FROM obs_fts_data").fetchone()[0]
    _run(store, root, "--backfill")
    assert c.execute("SELECT count(*) FROM obs_fts_data").fetchone()[0] == before


# --- resumable, chunked, committed -----------------------------------------


def test_a_second_run_over_a_classified_corpus_does_nothing(tmp_path, capsys):
    """``provenance IS NULL`` is the watermark, so a re-run is a no-op.

    "A run that re-does work it already did is a bug, even when it produces the
    right answer."
    """
    root = _projects_root(tmp_path)
    store = _unclassified_store(tmp_path, root)
    _run(store, root, "--backfill")
    capsys.readouterr()
    assert _run(store, root, "--backfill") == 0
    out = capsys.readouterr().out
    assert "classified=0" in out, out


def test_a_stop_is_reachable_inside_a_file_not_only_between_files(
    tmp_path, monkeypatch
):
    """A kill costs at most one chunk, and the next run finishes the job.

    The stop has to be checked at every CHUNK. Files in this archive are not
    uniform — the largest hold tens of thousands of lines — so a stop honoured
    only between files is one a SIGTERM can wait a long time for. This fixture
    is a single file, so a between-files check would classify all five rows and
    report itself interrupted, which is the bug.
    """
    root = _projects_root(tmp_path)
    store = _unclassified_store(tmp_path, root)

    import aggregator.cli as cli

    calls = {"n": 0}

    def stop_after_one_chunk():
        calls["n"] += 1
        return calls["n"] > 1

    monkeypatch.setattr(cli, "graceful_shutdown", _fake_shutdown(stop_after_one_chunk))
    rc = _run(store, root, "--backfill", "--chunk-size", "1")
    assert rc == 0
    got = _provenance(store)
    classified = [v for v in got.values() if v is not None]
    assert 0 < len(classified) < len(got), got

    # A fresh process picks up exactly where that one stopped.
    monkeypatch.undo()
    assert _run(store, root, "--backfill") == 0
    assert None not in _provenance(store).values()


def _fake_shutdown(predicate):
    import contextlib

    @contextlib.contextmanager
    def fake(*_a, **_kw):
        yield predicate

    return fake


# --- errors are loud --------------------------------------------------------


def test_an_unreadable_archive_file_is_recorded_and_the_run_still_exits_nonzero(
    tmp_path, capsys, monkeypatch
):
    root = _projects_root(tmp_path)
    store = _unclassified_store(tmp_path, root)

    real_open = Path.open

    def boom(self, *a, **kw):
        if self.suffix == ".jsonl":
            raise OSError("simulated unreadable archive")
        return real_open(self, *a, **kw)

    monkeypatch.setattr(Path, "open", boom)
    rc = _run(store, root, "--backfill")
    monkeypatch.undo()
    err = capsys.readouterr().err
    assert rc != 0, "a run with errors must not look like a clean one"
    assert "simulated unreadable archive" in err
    # And the rest of the corpus still got classified, from the DB-only route.
    assert None not in _provenance(store).values()


def test_a_row_the_classifier_cannot_place_stops_the_run_instead_of_looping(
    tmp_path, capsys, monkeypatch
):
    """A chunk that selects rows and classifies none would spin forever.

    "Poison records must not abort a run, and must not be silently retried
    forever" — so the loop refuses to ask the same question a second time.

    Scoped to the DB-only route, which is the one that re-selects: an EMPTY
    archive root means route 1 reaches nothing and every row falls through.
    """
    empty_root = tmp_path / "empty-projects"
    empty_root.mkdir()
    store = Store(db_path=tmp_path / "cache.db")
    store.migrate()
    store.upsert_entities(
        [
            SessionRow(
                session_id="s-db", root_session_id="s-db",
                parent_session_id=None, kind="session", agent_id=None,
                agent_type=None, spawned_by_tool_use_id=None, cwd=None,
                git_branch=None, first_ts=_TS, last_ts=_TS,
                jsonl_path="/gone/s-db.jsonl",
            ),
            ObservationRow(
                obs_id="o-db", session_id="s-db", root_session_id="s-db",
                parent_obs_id=None, type="user", ts=_TS, model=None,
                input_tokens=None, output_tokens=None, tool_name=None,
                tool_use_id=None, body="a turn with no archive",
            ),
        ]
    )
    store._c().execute("UPDATE observations SET provenance = NULL")
    store._c().commit()

    import aggregator.cli as cli

    monkeypatch.setattr(cli, "classify", lambda *a, **kw: None)
    rc = _run(store, empty_root, "--backfill")
    assert rc != 0
    assert "stalled" in capsys.readouterr().err.lower()
    assert _provenance(store) == {"o-db": None}


# --- reclassify -------------------------------------------------------------


def test_reclassify_resets_and_runs_again(tmp_path):
    """The documented path after a classifier revision. Provenance is NOT in
    ``_src_hash``, so nothing else in the corpus notices."""
    root = _projects_root(tmp_path)
    store = _unclassified_store(tmp_path, root)
    _run(store, root, "--backfill")
    c = store._c()
    c.execute("UPDATE observations SET provenance = 'human'")
    c.commit()
    assert _run(store, root, "--reclassify") == 0
    assert _provenance(store)["u-sdk"] == HOOK


# --- what must NOT happen ---------------------------------------------------


def test_migrate_does_not_classify_anything(tmp_path):
    """``migrate()`` runs on every subcommand, read-only queries included."""
    root = _projects_root(tmp_path)
    store = _unclassified_store(tmp_path, root)
    store.migrate()
    assert set(_provenance(store).values()) == {None}


def test_ingest_and_backfill_agree_on_the_same_rows(tmp_path):
    """Two routes into one column. If they disagree, nothing says which is
    right and the disagreement is invisible: both rows look classified."""
    root = _projects_root(tmp_path)
    ingested = {}
    src = SessionsSource(projects_root=str(root))
    for e in src.iter_entities(errors=[]):
        if isinstance(e, ObservationRow):
            ingested[e.obs_id] = e.provenance

    backfilled_store = _unclassified_store(tmp_path, root)
    _run(backfilled_store, root, "--backfill")
    assert _provenance(backfilled_store) == ingested
    backfilled_store.close()


@pytest.fixture(autouse=True)
def _never_the_live_cache(monkeypatch):
    """Every test here builds its own database; make that structural."""
    live = Path.home() / ".local" / "share" / "aggregator" / "cache.db"
    real_connect = sqlite3.connect

    def guarded(target, *a, **kw):
        assert str(live) not in str(target), f"test touched the live cache: {target}"
        return real_connect(target, *a, **kw)

    monkeypatch.setattr(sqlite3, "connect", guarded)


def test_a_corrupt_archive_line_does_not_make_the_run_red_for_ever(
    tmp_path, capsys
):
    """It is damage to one line, and the parser has already DECLARED it.

    A line the JSON parser rejects will never parse, no run will ever classify
    it, and ingest never stored a row for it. Counting it as a backfill failure
    would make this command exit non-zero for ever over something that has
    nothing to do with provenance — the permanently-red alarm an operator
    learns to dismiss unread.
    """
    root = _projects_root(tmp_path)
    store = _unclassified_store(tmp_path, root)
    # Damaged AFTER ingest, which is how it happens: a truncated-and-reappended
    # file leaves rows in the index and a line nothing can parse on disk.
    p = root / "proj" / "sess-p.jsonl"
    p.write_text(p.read_text(encoding="utf-8") + "\n{not json at all", encoding="utf-8")
    old = time.time() - 24 * 60 * 60
    os.utime(p, (old, old))

    assert _run(store, root, "--backfill") == 0
    assert None not in _provenance(store).values()


def test_an_undeclared_read_failure_stays_loud(tmp_path, capsys, monkeypatch):
    """A file that cannot be STATTED is transient, not permanent — the next run
    might succeed, so it is reported on every run until somebody fixes it."""
    root = _projects_root(tmp_path)
    store = _unclassified_store(tmp_path, root)

    real_stat = Path.stat

    def boom(self, *a, **kw):
        if self.suffix == ".jsonl":
            raise OSError("simulated stat failure")
        return real_stat(self, *a, **kw)

    monkeypatch.setattr(Path, "stat", boom)
    rc = _run(store, root, "--backfill")
    monkeypatch.undo()
    assert rc != 0
    assert "simulated stat failure" in capsys.readouterr().err


# --- the human surface ------------------------------------------------------


def test_the_cli_prints_who_wrote_each_observation(tmp_path, capsys):
    """``by=`` sits next to ``type=`` because the two are constantly confused
    and only one of them is about authorship."""
    root = _projects_root(tmp_path)
    store = _unclassified_store(tmp_path, root)
    _run(store, root, "--backfill")
    capsys.readouterr()

    rc = main(
        ["query", "source:sessions type:user NixOS", "--drilldown"],
        _store=store,
        _sources={"sessions": SessionsSource(projects_root=str(root))},
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "by=hook" in out, out


def test_the_cli_calls_an_unclassified_row_unclassified(tmp_path, capsys):
    """Not a blank. An un-backfilled cache must read as "nobody has looked",
    never as a missing field."""
    root = _projects_root(tmp_path)
    store = _unclassified_store(tmp_path, root)
    main(
        ["query", "source:sessions type:user NixOS", "--drilldown"],
        _store=store,
        _sources={"sessions": SessionsSource(projects_root=str(root))},
    )
    assert "by=unclassified" in capsys.readouterr().out
