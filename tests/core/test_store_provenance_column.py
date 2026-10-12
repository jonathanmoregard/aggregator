"""v5 → v6: ``observations.provenance``, and the digest it must stay out of.

The column is metadata-only (``ADD COLUMN`` with no default touches no page:
measured at 0.006 s on a throwaway database at 1/5 of live scale), and NULL for
every pre-existing row — which is the backfill's cursor, exactly as
``embedding_state IS NULL`` is the embed worker's.

THE TEST THAT MATTERS MOST HERE IS
:func:`test_provenance_is_not_in_the_source_digest`. If the classifier's output
enters ``_src_hash``, every classifier revision changes all 549,952 digests,
the next ingest takes the ``DO UPDATE`` branch for every row, re-runs Presidio
on each (~11 hours at this repo's measured 827 rows/min) and sets
``embedding_state = NULL`` — discarding the observation vector arm. The
``SCRUB_FINGERPRINT`` comment right above the digest actively invites exactly
that pattern, so the refusal is pinned rather than commented.
"""

import sqlite3
from datetime import UTC, datetime

from aggregator.core import store as store_module
from aggregator.core.provenance import AGENT, HOOK, HUMAN
from aggregator.core.store import SCHEMA_VERSION, Store
from aggregator.sources.base import ObservationRow, QueryAST, SessionRow


def _session(session_id: str = "s1", kind: str = "session") -> SessionRow:
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
        first_ts=datetime(2026, 7, 25, 10, 0, tzinfo=UTC),
        last_ts=datetime(2026, 7, 25, 10, 5, tzinfo=UTC),
        jsonl_path="/tmp/x.jsonl",
    )


def _obs(
    obs_id: str,
    body: str,
    *,
    provenance: str | None = None,
    session_id: str = "s1",
) -> ObservationRow:
    return ObservationRow(
        obs_id=obs_id,
        session_id=session_id,
        root_session_id=session_id,
        parent_obs_id=None,
        type="user",
        ts=datetime(2026, 7, 25, 10, 0, tzinfo=UTC),
        model=None,
        input_tokens=None,
        output_tokens=None,
        tool_name=None,
        tool_use_id=None,
        body=body,
        provenance=provenance,
    )


def _columns(c: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in c.execute(f"PRAGMA table_info({table})")}


def test_schema_version_is_seven():
    """v6 added ``provenance``; v7 (porter stemming, ``ededbf2``) is current.

    Pinned to the literal so a migration bump is a conscious edit here rather
    than silent drift — same precedent as ``test_store.py``'s
    ``test_schema_version_is_7``. The provenance column this file guards
    shipped in v6 and survives unchanged through v7.
    """
    assert SCHEMA_VERSION == 7


def test_a_fresh_database_has_the_column_and_its_index(tmp_path):
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    c = s._c()
    assert "provenance" in _columns(c, "observations")
    indexes = {
        row[0]
        for row in c.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='observations'"
        )
    }
    assert "obs_provenance" in indexes
    s.close()


def test_a_v5_database_upgrades_in_place_with_every_row_unclassified(tmp_path):
    """NULL for every existing row: the backfill's cursor, and no rewrite.

    A non-NULL default would be the same bug as advancing a watermark past
    data nothing processed — the backfill would find nothing to do and every
    row would claim an authorship no classifier ever assigned it.
    """
    db = tmp_path / "cache.db"
    s = Store(db_path=db)
    s.migrate()
    s.upsert_entities([_session(), _obs("o1", "an ordinary turn")])
    c = s._c()
    c.execute("DROP INDEX IF EXISTS obs_provenance")
    c.execute("ALTER TABLE observations DROP COLUMN provenance")
    c.execute("PRAGMA user_version = 5")
    c.commit()
    before = c.execute(
        "SELECT body, src_hash, embedding_state FROM observations WHERE obs_id='o1'"
    ).fetchone()
    s.close()

    s2 = Store(db_path=db)
    s2.migrate()
    c2 = s2._c()
    assert "provenance" in _columns(c2, "observations")
    row = c2.execute(
        "SELECT body, src_hash, embedding_state, provenance FROM observations "
        "WHERE obs_id='o1'"
    ).fetchone()
    assert row["provenance"] is None
    assert (row["body"], row["src_hash"], row["embedding_state"]) == tuple(before)
    assert s2.schema_version() == SCHEMA_VERSION
    s2.close()


def test_provenance_is_not_in_the_source_digest(tmp_path):
    """A classifier revision must not re-scrub the corpus or void its vectors.

    Same row, same body, different provenance: the digest is over what the
    SOURCE produced, so the second write is recognised as unchanged and never
    reaches the ``DO UPDATE`` branch that would reset ``embedding_state``.
    """
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert_entities([_session(), _obs("o1", "a turn", provenance=HUMAN)])
    c = s._c()
    c.execute("UPDATE observations SET embedding_state = 'ok' WHERE obs_id = 'o1'")
    c.commit()

    unchanged = s.upsert_entities([_session(), _obs("o1", "a turn", provenance=HOOK)])
    assert unchanged == 2
    row = c.execute(
        "SELECT provenance, embedding_state FROM observations WHERE obs_id='o1'"
    ).fetchone()
    # The row was not rewritten, so the vector arm keeps its state AND the
    # stored provenance is the one the backfill owns — not the new guess.
    assert row["embedding_state"] == "ok"
    assert row["provenance"] == HUMAN
    s.close()


def test_there_is_no_provenance_fingerprint_constant():
    """The trap ``SCRUB_FINGERPRINT`` invites, named so it stays refused.

    A ``PROVENANCE_FINGERPRINT`` beside it would be the one-line way to put the
    classifier's version into every digest — ~11 hours of re-scrub now, and
    weeks of discarded embedding CPU once the observation vector arm is warm.
    """
    assert not hasattr(store_module, "PROVENANCE_FINGERPRINT")


def test_a_changed_body_still_carries_the_new_provenance(tmp_path):
    """The ``DO UPDATE`` branch is already rewriting the row, so it writes it."""
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert_entities([_session(), _obs("o1", "a turn", provenance=HUMAN)])
    s.upsert_entities([_session(), _obs("o1", "a different turn", provenance=AGENT)])
    row = s._c().execute(
        "SELECT body, provenance FROM observations WHERE obs_id='o1'"
    ).fetchone()
    assert row["provenance"] == AGENT
    assert "different" in row["body"]
    s.close()


def test_query_observations_hands_provenance_back(tmp_path):
    """The read-back hydrator must carry it or the MCP cannot surface it."""
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert_entities(
        [
            _session(),
            _obs("o1", "a human turn", provenance=HUMAN),
            _obs("o2", "a hook prompt", provenance=HOOK),
        ]
    )
    rows = s.query_observations(QueryAST())
    assert {r.obs_id: r.provenance for r in rows} == {"o1": HUMAN, "o2": HOOK}
    s.close()


def test_an_unclassified_row_reads_back_as_none(tmp_path):
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert_entities([_session(), _obs("o1", "a turn")])
    assert s.query_observations(QueryAST())[0].provenance is None
    s.close()


# --- the ``by:`` filter, at the SQL layer ----------------------------------


def _seeded(tmp_path) -> Store:
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    s.upsert_entities(
        [
            _session(),
            _obs("o-human", "clickable links please", provenance=HUMAN),
            _obs("o-hook", "You are watching a Claude Code session", provenance=HOOK),
            _obs("o-agent", "research the thing", provenance=AGENT),
            _obs("o-null", "not classified yet"),
        ]
    )
    return s


def test_no_filter_returns_every_row(tmp_path):
    """DEFAULT ABSENT MEANS NO FILTER, exactly like ``type:`` today.

    A default human-only filter would silently drop the machine-authored
    majority from ``_first_user_prompt``, from the eval baseline and from
    ``matching_observations`` — a narrowing the caller never asked for and
    cannot see.
    """
    s = _seeded(tmp_path)
    assert len(s.query_observations(QueryAST())) == 4
    s.close()


def test_by_human_selects_only_the_human_rows(tmp_path):
    s = _seeded(tmp_path)
    rows = s.query_observations(QueryAST(provenance=HUMAN))
    assert [r.obs_id for r in rows] == ["o-human"]
    assert s.count_observations(QueryAST(provenance=HUMAN)) == 1
    s.close()


def test_by_machine_is_every_non_human_member(tmp_path):
    s = _seeded(tmp_path)
    rows = s.query_observations(QueryAST(provenance="machine"))
    assert sorted(r.obs_id for r in rows) == ["o-agent", "o-hook"]
    s.close()


def test_an_unclassified_row_is_neither_human_nor_machine(tmp_path):
    """NULL is not a claim. Folding it into either side would invent one."""
    s = _seeded(tmp_path)
    for value in (HUMAN, "machine", HOOK, AGENT):
        assert "o-null" not in {
            r.obs_id for r in s.query_observations(QueryAST(provenance=value))
        }
    s.close()


def test_the_store_can_say_the_corpus_is_not_fully_classified(tmp_path):
    """So an empty ``by:`` page can explain itself instead of looking empty."""
    s = _seeded(tmp_path)
    assert s.has_unclassified_observations() is True
    s._c().execute("UPDATE observations SET provenance = 'human' WHERE provenance IS NULL")
    s._c().commit()
    assert s.has_unclassified_observations() is False
    s.close()
