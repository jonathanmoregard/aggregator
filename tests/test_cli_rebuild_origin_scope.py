"""Round-1 HIGH: ``ingest sessions --rebuild`` wiped the chat-export origins.

The ``sessions`` and ``observations`` tables hold three populations under
``sessions.origin``:

* ``claude-code`` — the local ~/.claude/projects JSONLs. Rebuildable: the
  source of truth is still on disk.
* ``chatgpt`` / ``claude-web`` — vendor data-export archives a human downloads
  by hand. NOT rebuildable. Once the drop is gone these rows are the last
  copy, and reacquiring them means a person clicking through a vendor export
  flow and waiting up to seven days.

``rebuild_and_upsert_entities`` ran ``DELETE FROM sessions`` unqualified, so a
rebuild driven by the sessions source destroyed all three. The >20% shrink
guard could not catch it either: it compared the incoming claude-code count
against the WHOLE table, so 840 claude-code rows replacing a store of 840
claude-code + 160 claude-web read as a 16% shrink — inside the slack, no
prompt, exit 0, 160 unrecoverable rows gone.

Fix: the DELETE is scoped to the origins the source can regenerate, and the
guard counts the same population the DELETE will reach.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from aggregator import cli
from aggregator.core.store import EmptyRebuildRefusedError, Store
from aggregator.sources.base import ObservationRow, SessionRow


def _session(sid: str, origin: str) -> SessionRow:
    return SessionRow(
        session_id=sid,
        root_session_id=sid,
        parent_session_id=None,
        kind="session",
        agent_id=None,
        agent_type=None,
        spawned_by_tool_use_id=None,
        cwd=None,
        git_branch=None,
        first_ts=datetime(2026, 7, 1, tzinfo=UTC),
        last_ts=datetime(2026, 7, 1, tzinfo=UTC),
        jsonl_path=f"/x/{sid}.jsonl",
        origin=origin,
    )


def _obs(oid: str, sid: str) -> ObservationRow:
    return ObservationRow(
        obs_id=oid,
        session_id=sid,
        root_session_id=sid,
        parent_obs_id=None,
        type="user",
        ts=datetime(2026, 7, 1, tzinfo=UTC),
        model=None,
        input_tokens=None,
        output_tokens=None,
        tool_name=None,
        tool_use_id=None,
        body="hello",
    )


class _SessionsStub:
    """Stands in for ``SessionsSource``: yields claude-code rows only."""

    name = "sessions"

    def __init__(self, rows):
        self._rows = list(rows)

    def iter_entities(self, since, errors=None):
        yield from self._rows


def _store(tmp_path) -> Store:
    store = Store(db_path=tmp_path / "cache.db")
    store.migrate()
    return store


def _origin_counts(store: Store) -> dict[str, int]:
    rows = store._c().execute(
        "SELECT origin, COUNT(*) AS n FROM sessions GROUP BY origin"
    ).fetchall()
    return {r["origin"]: r["n"] for r in rows}


# --- CLI layer -------------------------------------------------------------


def test_rebuilding_sessions_leaves_the_chat_export_origins_alone(
    tmp_path, capsys
):
    """THE finding. 16% of the table is unrecoverable vendor-export data and
    the shrink guard's slack hid its destruction completely."""
    store = _store(tmp_path)
    store.upsert_entities(
        [_session(f"cc-{i}", "claude-code") for i in range(840)]
        + [_session(f"cw-{i}", "claude-web") for i in range(160)]
    )

    src = _SessionsStub(_session(f"cc-{i}", "claude-code") for i in range(840))
    rc = cli.main(
        ["ingest", "sessions", "--rebuild"],
        _store=store,
        _sources={"sessions": src},
    )

    assert rc == 0
    assert _origin_counts(store) == {"claude-code": 840, "claude-web": 160}, (
        "a sessions rebuild must not reach an origin it cannot regenerate"
    )
    _ = capsys


def test_a_chat_export_sessions_observations_survive_too(tmp_path):
    """The observations are the searchable content; losing them loses the
    conversation even if the session row is still there."""
    store = _store(tmp_path)
    store.upsert_entities(
        [
            _session("cc-1", "claude-code"),
            _obs("cc-1-o1", "cc-1"),
            _session("gpt-1", "chatgpt"),
            _obs("gpt-1-o1", "gpt-1"),
            _session("cw-1", "claude-web"),
            _obs("cw-1-o1", "cw-1"),
        ]
    )

    src = _SessionsStub([_session("cc-1", "claude-code"), _obs("cc-1-o1", "cc-1")])
    rc = cli.main(
        ["ingest", "sessions", "--rebuild"],
        _store=store,
        _sources={"sessions": src},
    )

    assert rc == 0
    surviving = {
        r["obs_id"]
        for r in store._c().execute("SELECT obs_id FROM observations").fetchall()
    }
    assert surviving == {"cc-1-o1", "gpt-1-o1", "cw-1-o1"}


def test_the_rebuild_still_replaces_the_origin_it_owns(tmp_path):
    """Scoping must not turn --rebuild into a plain upsert: a claude-code row
    that no longer exists on disk still has to disappear."""
    store = _store(tmp_path)
    store.upsert_entities(
        [
            _session("cc-gone", "claude-code"),
            _session("cc-kept", "claude-code"),
            _session("cw-1", "claude-web"),
        ]
    )

    src = _SessionsStub([_session("cc-kept", "claude-code")])
    rc = cli.main(
        ["ingest", "sessions", "--rebuild"],
        _store=store,
        _sources={"sessions": src},
    )

    assert rc == 0
    ids = {
        r["session_id"]
        for r in store._c().execute("SELECT session_id FROM sessions").fetchall()
    }
    assert ids == {"cc-kept", "cw-1"}


def test_the_shrink_guard_counts_only_the_origins_it_will_delete(
    tmp_path, capsys
):
    """The guard's denominator has to be the population at risk. Padded with
    chat-export rows it answers a question nobody asked."""
    store = _store(tmp_path)
    store.upsert_entities(
        [_session(f"cc-{i}", "claude-code") for i in range(200)]
        + [_session(f"cw-{i}", "claude-web") for i in range(400)]
    )

    # 100 new vs 200 existing claude-code = 50% drop -> must refuse, even
    # though 100 vs the 600-row whole table would too. What matters is that
    # the number quoted to the operator is the one they can act on.
    src = _SessionsStub(_session(f"cc-{i}", "claude-code") for i in range(100))
    rc = cli.main(
        ["ingest", "sessions", "--rebuild"],
        _store=store,
        _sources={"sessions": src},
    )

    assert rc == 1
    err = capsys.readouterr().err
    assert "200" in err, f"guard must quote the claude-code count, got {err!r}"
    assert _origin_counts(store) == {"claude-code": 200, "claude-web": 400}


# --- store layer -----------------------------------------------------------


def test_scoped_rebuild_deletes_only_the_named_origins(tmp_path):
    store = _store(tmp_path)
    store.upsert_entities(
        [
            _session("cc-1", "claude-code"),
            _session("gpt-1", "chatgpt"),
            _session("cw-1", "claude-web"),
        ]
    )

    store.rebuild_and_upsert_entities(
        [_session("cc-2", "claude-code")], origins=("claude-code",)
    )

    assert _origin_counts(store) == {
        "claude-code": 1,
        "chatgpt": 1,
        "claude-web": 1,
    }


def test_an_unscoped_rebuild_still_clears_the_whole_table(tmp_path):
    """``origins=None`` keeps the historical behaviour for a caller that
    genuinely means every origin — the sessions CLI path no longer does."""
    store = _store(tmp_path)
    store.upsert_entities(
        [_session("cc-1", "claude-code"), _session("cw-1", "claude-web")]
    )

    store.rebuild_and_upsert_entities([_session("cc-2", "claude-code")])

    assert _origin_counts(store) == {"claude-code": 1}


def test_a_stream_carrying_an_origin_outside_the_scope_is_refused(tmp_path):
    """Those rows would be INSERTed while the DELETE never reached their
    existing counterparts — a rebuild that is not a rebuild. Refuse rather
    than half-apply."""
    store = _store(tmp_path)
    store.upsert_entities([_session("cc-1", "claude-code")])

    with pytest.raises(EmptyRebuildRefusedError, match="chatgpt"):
        store.rebuild_and_upsert_entities(
            [_session("cc-2", "claude-code"), _session("gpt-1", "chatgpt")],
            origins=("claude-code",),
        )

    assert _origin_counts(store) == {"claude-code": 1}, "refusal must not write"


def test_count_sessions_by_origin_filters(tmp_path):
    store = _store(tmp_path)
    store.upsert_entities(
        [
            _session("cc-1", "claude-code"),
            _session("cc-2", "claude-code"),
            _session("cw-1", "claude-web"),
        ]
    )

    assert store.count_sessions_by_origin() == 3
    assert store.count_sessions_by_origin(("claude-code",)) == 2
    assert store.count_sessions_by_origin(("claude-web",)) == 1
    assert store.count_sessions_by_origin(("claude-code", "claude-web")) == 3
    assert store.count_sessions_by_origin(()) == 0
