"""The vector arm fills in the order the user asked for, and says how far.

USER DIRECTIVE, 2026-08-21, verbatim: ``dropbox -> blog -> llm -> claude code``.

The full-corpus backfill is a measured 25-30 days of continuous CPU on this
hardware, and the FTS5 arm serves throughout, so the ORDER is not a tuning
detail — it is the entire difference between the vector arm being useful in
week one and in week five. The user was offered four ways to deal with the 25-30
days (incremental, quantize, a smaller model, filter the corpus), chose
incremental, and then OVERRODE the proposed priority: the proposal put
claude-code sessions first on the theory that the agent's own history is what
gets searched most, and the user put it last.

Category names mapped to cache source names, recorded as an assumption in
session-constraints.md because the user named categories:

    dropbox      -> dropbox              (records)
    blog         -> substack             (records)
    llm          -> claude-web, chatgpt  (observations, by session origin)
    claude code  -> sessions, subagents  (observations, by session kind)
    everything else                      after the named four, unranked

THE ORDER SPANS BOTH ONTOLOGIES, which is why it cannot be expressed as "do
observations, then records". Two records sources come before every observation,
and four more come after.

WHAT THIS FILE EXISTS TO PREVENT, beyond the ordering itself:

* a row belonging to no group. It would never be selected, never be embedded,
  and never be reported missing — the index rotting silently, which is the one
  failure mode this project bans by name. Asserted as a partition.
* a source reporting "complete" when it holds nothing at all. "Fully embedded"
  and "empty" produce identical hit counts, and telling them apart is the
  question a user actually asks of a half-built index.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import numpy as np
import pytest

from aggregator.core.store import (
    EMBED_BACKLOG_ORDER,
    EMBED_REST,
    Store,
)
from aggregator.sources.base import ObservationRow, Record, SessionRow

_DIM = 768


def _session(session_id, kind="session", origin="claude-code"):
    ts = datetime(2026, 7, 1, 8, 0, tzinfo=UTC)
    return SessionRow(
        session_id=session_id,
        root_session_id=session_id,
        parent_session_id=None,
        kind=kind,
        agent_id=None,
        agent_type=None,
        spawned_by_tool_use_id=None,
        cwd="/x",
        git_branch="main",
        first_ts=ts,
        last_ts=ts,
        jsonl_path=f"/tmp/{session_id}.jsonl",
        origin=origin,
    )


def _obs(obs_id, session_id, body="a body worth embedding", day=1):
    return ObservationRow(
        obs_id=obs_id,
        session_id=session_id,
        root_session_id=session_id,
        parent_obs_id=None,
        type="user",
        ts=datetime(2026, 7, day, 8, 0, tzinfo=UTC),
        model=None,
        input_tokens=None,
        output_tokens=None,
        tool_name=None,
        tool_use_id=None,
        body=body,
    )


def _record(stable_id, source):
    ts = datetime(2026, 7, 1, tzinfo=UTC)
    return Record(
        stable_id=stable_id,
        source=source,
        subject=f"subject of {stable_id}",
        body=f"body of {stable_id}",
        tags=[],
        created_at=ts,
        updated_at=ts,
    )


@pytest.fixture
def store(tmp_path):
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert_entities(
        [
            _session("s-code"),
            _session("s-sub", kind="subagent"),
            _session("s-web", origin="claude-web"),
            _session("s-gpt", origin="chatgpt"),
            _obs("o-code", "s-code"),
            _obs("o-sub", "s-sub"),
            _obs("o-web", "s-web"),
            _obs("o-gpt", "s-gpt"),
        ]
    )
    s.upsert(
        [
            _record("dropbox:a", "dropbox"),
            _record("substack:a", "substack"),
            _record("github:a", "github"),
            _record("research:a", "research"),
            _record("ticktick:a", "ticktick"),
            _record("sota-watch:a", "sota-watch"),
        ]
    )
    return s


def _ids(rows) -> set[str]:
    out = set()
    for row in rows:
        keys = row.keys()
        out.add(row["obs_id"] if "obs_id" in keys else row["stable_id"])
    return out


# --- the order itself -------------------------------------------------------


def test_the_backlog_order_is_the_one_the_user_specified():
    """THE DIRECTIVE, as an assertion. Not a comment somebody can drift from."""
    assert EMBED_BACKLOG_ORDER[:6] == (
        ("records", "dropbox"),
        ("records", "substack"),
        ("observations", "claude-web"),
        ("observations", "chatgpt"),
        ("observations", "sessions"),
        ("observations", "subagents"),
    )
    assert set(EMBED_BACKLOG_ORDER[6:]) == {
        ("records", EMBED_REST),
        ("observations", EMBED_REST),
    }


def test_the_named_four_come_before_everything_unranked():
    """The user ranked four categories and left the rest unranked. Unranked
    means LATER, and it has to be later than all four rather than interleaved
    with them — otherwise the ordering the whole directive is about is gone."""
    ranks = {pair: i for i, pair in enumerate(EMBED_BACKLOG_ORDER)}
    last_named = max(
        i for pair, i in ranks.items() if pair[1] != EMBED_REST
    )
    first_rest = min(i for pair, i in ranks.items() if pair[1] == EMBED_REST)
    assert first_rest > last_named


# --- selecting one source ---------------------------------------------------


def test_the_backlog_can_be_scoped_to_one_records_source(store):
    assert _ids(store.select_unembedded("records", source="dropbox")) == {
        "dropbox:a"
    }


def test_observation_sources_come_from_the_session_that_owns_the_row(store):
    """An observation has no ``source`` column. Its bucket is a property of the
    session it belongs to: origin says which product, kind says session vs
    subagent, and 'claude code' means the two claude-code kinds only."""
    assert _ids(store.select_unembedded("observations", source="sessions")) == {
        "o-code"
    }
    assert _ids(store.select_unembedded("observations", source="subagents")) == {
        "o-sub"
    }
    assert _ids(store.select_unembedded("observations", source="claude-web")) == {
        "o-web"
    }
    assert _ids(store.select_unembedded("observations", source="chatgpt")) == {
        "o-gpt"
    }


def test_the_catch_all_takes_every_source_the_user_did_not_rank(store):
    assert _ids(store.select_unembedded("records", source=EMBED_REST)) == {
        "github:a",
        "research:a",
        "ticktick:a",
        "sota-watch:a",
    }


def test_an_origin_nobody_has_thought_of_still_lands_in_the_catch_all(store):
    """THE ROW THAT MUST NOT VANISH. A new chat export, a new session origin —
    anything the CASE does not name — has to fall into the unranked group. A
    row belonging to no group is never selected, never embedded, and never
    reported missing."""
    store.upsert_entities(
        [_session("s-new", origin="some-future-product"), _obs("o-new", "s-new")]
    )
    assert "o-new" in _ids(store.select_unembedded("observations", source=EMBED_REST))


def test_the_groups_partition_the_backlog_exactly(store):
    """Every row in exactly one group, no row in none. The two halves are
    different bugs: a row in no group is silently never embedded, and a row in
    two is embedded twice and inflates every per-source percentage."""
    for kind in ("observations", "records"):
        whole = _ids(store.select_unembedded(kind))
        seen: list[str] = []
        for group_kind, source in EMBED_BACKLOG_ORDER:
            if group_kind != kind:
                continue
            seen += sorted(_ids(store.select_unembedded(kind, source=source)))
        assert sorted(seen) == sorted(whole), f"{kind}: groups do not cover the backlog"
        assert len(seen) == len(set(seen)), f"{kind}: a row is in two groups"


def test_the_schema_is_what_stops_an_orphan_observation_being_written(store):
    """FIRST DEFENCE, AND IT HOLDS. ``observations.session_id`` is declared
    ``NOT NULL REFERENCES sessions(session_id)`` and the store opens every
    connection with ``PRAGMA foreign_keys = ON``, so no path through this API
    can write an observation whose session row is missing. Pinned here because
    the two tests below deliberately go around it, and if the constraint were
    ever dropped they would silently become the only thing left."""
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        store.upsert_entities([_obs("o-orphan", "s-that-was-never-written")])


def _write_orphan_around_the_foreign_key(store, obs_id="o-orphan"):
    """An orphan the way a real cache would acquire one: from OUTSIDE this API.

    ``PRAGMA foreign_keys`` is per-connection and defaults to OFF — the sqlite3
    CLI, a DB browser, and a hand-written repair script all arrive with it off,
    and SQLite never re-validates existing rows when a constraint is declared.
    So a cache someone has poked at by hand can hold this row even though the
    store itself cannot create it. That is the scenario, and it is why the
    grouping must not depend on the constraint holding.
    """
    c = store._c()
    c.execute("PRAGMA foreign_keys = OFF")
    try:
        c.execute(
            "INSERT INTO observations "
            "(obs_id, session_id, root_session_id, type, ts, body, src_hash) "
            "VALUES (?, ?, ?, 'user', '2026-07-01T08:00:00+00:00', ?, 'deadbeef')",
            (obs_id, "s-that-was-never-written", "s-that-was-never-written",
             "a body worth embedding"),
        )
        c.commit()
    finally:
        c.execute("PRAGMA foreign_keys = ON")


def test_an_observation_whose_session_row_is_missing_still_belongs_to_a_group(store):
    """THE OTHER WAY A ROW BELONGS TO NO GROUP, and the one the CASE cannot fix.

    ``_OBS_SOURCE_CASE`` ends in ``ELSE '(other)'``, so an origin nobody has
    thought of lands in the catch-all — that is the test above. But the CASE
    reads ``sessions.origin``, and reaching it at all takes a join. Under an
    INNER JOIN an observation whose session row is missing matches no branch,
    including the catch-all, because it never reaches the CASE: it is absent
    from every group, so the worker never selects it and
    ``mark_embedding_version_complete`` waits forever on a row nobody can name.

    The foreign key above is what makes this unreachable from inside, and the
    LEFT JOIN is what makes it survivable from outside. The cost of getting it
    wrong is not one unembedded row — it is that the embedding version never
    completes, on a backfill measured in weeks, with nothing naming the cause.
    """
    _write_orphan_around_the_foreign_key(store)

    whole = _ids(store.select_unembedded("observations"))
    assert "o-orphan" in whole, (
        "the unscoped backlog does not even hold the orphan; the join that "
        "drops it is upstream of the grouping"
    )
    groups = [
        source
        for kind, source in EMBED_BACKLOG_ORDER
        if kind == "observations"
        and "o-orphan" in _ids(store.select_unembedded("observations", source=source))
    ]
    assert groups == [EMBED_REST], (
        f"an observation with no session row belongs to {groups or 'NO group'}; "
        f"the groups are documented as a partition of the backlog"
    )


def test_the_progress_tally_counts_a_row_whose_session_row_is_missing(store):
    """The backlog and the tally must answer the same question. If the tally
    drops the orphan too, per-source progress reads 100% while the backlog is
    not empty — a source that says complete and a worker that disagrees."""
    _write_orphan_around_the_foreign_key(store)
    obs_rows = store._c().execute("SELECT COUNT(*) AS n FROM observations").fetchone()

    tallied = sum(
        row["total"]
        for row in store.embed_progress_by_source()
        if row["kind"] == "observations"
    )
    assert tallied == obs_rows["n"], (
        f"the tally sees {tallied} observations, the table holds "
        f"{obs_rows['n']}; the difference is invisible in every progress display"
    )


def test_an_unknown_source_is_refused_rather_than_selecting_nothing(store):
    """A typo must not read as 'that source is fully embedded'."""
    with pytest.raises(ValueError, match="dropbcx"):
        store.select_unembedded("records", source="dropbcx")


def test_scoping_still_hides_rows_the_worker_set_aside(store):
    """The source filter narrows the backlog; it does not reopen it. 'skip' and
    'error' are facts no chunk_embeddings row could record, and they hold a row
    out of every group equally."""
    store.mark_embedded("records", ["dropbox:a"], "skip")
    assert store.select_unembedded("records", source="dropbox") == []


# --- per-source progress ----------------------------------------------------


def test_progress_is_reported_per_source_in_priority_order(store):
    """'Which sources are fully embedded' is the question a user actually asks,
    and a global percentage cannot answer it."""
    rows = store.embed_progress_by_source()

    assert [(r["kind"], r["source"]) for r in rows] == list(EMBED_BACKLOG_ORDER)
    # Keyed on the PAIR: ``EMBED_REST`` exists once per ontology, and collapsing
    # the two would hide whichever came second.
    by_group = {(r["kind"], r["source"]): r for r in rows}
    assert by_group[("records", "dropbox")]["total"] == 1
    assert by_group[("observations", "sessions")]["total"] == 1
    assert by_group[("records", EMBED_REST)]["total"] == 4
    assert by_group[("observations", EMBED_REST)]["total"] == 0


def test_a_source_with_no_rows_says_empty_and_never_complete(store, tmp_path):
    """THE LOOKALIKE THIS EXISTS TO BREAK. A source holding nothing and a
    source fully embedded both answer every query with zero vector hits."""
    empty = Store(db_path=tmp_path / "empty.db")
    empty.migrate()
    states = {r["source"]: r["state"] for r in empty.embed_progress_by_source()}
    assert set(states.values()) == {"empty"}


def test_a_half_embedded_source_is_not_reported_as_complete(store):
    """Half of dropbox embedded is not dropbox."""
    store.upsert([_record("dropbox:b", "dropbox")])
    store.upsert_vec_records([("dropbox:a", np.zeros(_DIM, dtype=np.float32))])
    store.mark_embedded(
        "records", ["dropbox:a"], "ok", expected={"dropbox:a": None}
    )

    row = next(
        r for r in store.embed_progress_by_source() if r["source"] == "dropbox"
    )
    assert row["embedded"] == 1
    assert row["pending"] == 1
    assert row["state"] == "in_progress"


def test_a_fully_embedded_source_says_so(store):
    """The other half of the same claim: 'complete' has to be reachable, or
    the answer is useless."""
    store.upsert_vec_records([("dropbox:a", np.zeros(_DIM, dtype=np.float32))])
    store.mark_embedded(
        "records", ["dropbox:a"], "ok", expected={"dropbox:a": None}
    )

    row = next(
        r for r in store.embed_progress_by_source() if r["source"] == "dropbox"
    )
    assert row["state"] == "complete"
    assert row["pending"] == 0


def test_rows_the_worker_gave_up_on_keep_a_source_out_of_complete(store):
    """Drained is not whole. A source whose only unembedded rows are ones the
    worker set aside has stopped making progress, and waiting cannot fix it —
    reporting that as 'complete' is the index rotting while the counts say
    fine."""
    store.mark_embedded("records", ["dropbox:a"], "error")

    row = next(
        r for r in store.embed_progress_by_source() if r["source"] == "dropbox"
    )
    assert row["errors"] == 1
    assert row["state"] == "degraded"


def test_progress_counts_the_model_that_is_configured_now(store):
    """Vectors are keyed (chunk_id, model). A tally that ignored the model
    would report a complete index for an embedding space holding nothing —
    the same bug the backlog query itself was fixed for."""
    store.upsert_vec_records([("dropbox:a", np.zeros(_DIM, dtype=np.float32))])
    store.mark_embedded(
        "records", ["dropbox:a"], "ok", expected={"dropbox:a": None}
    )

    rows = store.embed_progress_by_source(model="some/other-model@768/x/norm-l2")
    row = next(r for r in rows if r["source"] == "dropbox")
    assert row["embedded"] == 0
    assert row["state"] == "not_started"
