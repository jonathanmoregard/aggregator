"""``migrate()`` moves a cache FORWARD, and used to move it backward in silence.

THE INCIDENT THIS FILE PINS. Two builds of the same program ran as different
roles against one shared cache. The MCP reader ran out of the live checkout at
``SCHEMA_VERSION`` 6; the ``aggregator`` CLI and the ``aggregator-ingest.timer``
behind it were a pinned Nix build at 5. ``migrate()`` stamped
``PRAGMA user_version = SCHEMA_VERSION`` unconditionally, so every thirty
minutes the v5 writer opened the cache, re-stamped it down to 5, and exited 0.
``systemctl`` reported ``ExecMainStatus=0``, every source read fresh, and every
single recall was refused. Recall was 100% dead for weeks, and nothing noticed,
because A SUCCESSFUL DOWNGRADE LOOKS EXACTLY LIKE A SUCCESSFUL MIGRATION.

The unconditional stamp is what made it invisible. A cache stamped ABOVE this
build's ``SCHEMA_VERSION`` means an older build is running against a newer
cache — this process does not know what the newer schema contains, which
columns are now NOT NULL, which triggers now fire, or what the vector tables
are shaped like — so writing into it is how a cache that merely could not be
read becomes a cache that is wrong. There is exactly one safe move and it is to
stop.

WHAT THE TESTS ASSERT IS THE DATA, NOT THE EXCEPTION. An exception that is
raised after the stamp has already landed fixes nothing at all, and a test that
only catches ``pytest.raises`` cannot tell the two apart. So every refusal test
below re-opens the file afterwards and reads ``PRAGMA user_version`` back with
a raw ``sqlite3`` connection, out of band from the code under test.

FORWARD ONLY. The fix for skew is to bring the lagging side UP — deploy a
writer that understands schema N. Downgrading the cache to suit an old writer,
by ``rebuild_all()`` or by any other route, is not an available answer and the
refusal message must not offer it.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pytest

from aggregator.core.store import SCHEMA_VERSION, SchemaAheadError, Store
from aggregator.sources.base import ObservationRow, SessionRow

_TS = datetime(2026, 8, 30, tzinfo=UTC)


def _seed(store: Store, n: int = 3) -> None:
    rows: list = []
    for i in range(n):
        sid = f"s{i}"
        rows.append(
            SessionRow(
                session_id=sid,
                root_session_id=sid,
                parent_session_id=None,
                kind="session",
                agent_id=None,
                agent_type=None,
                spawned_by_tool_use_id=None,
                cwd="/x",
                git_branch="main",
                first_ts=_TS,
                last_ts=_TS,
                jsonl_path=f"/tmp/{sid}.jsonl",
            )
        )
        rows.append(
            ObservationRow(
                obs_id=f"o{i}",
                session_id=sid,
                root_session_id=sid,
                parent_obs_id=None,
                type="user",
                ts=_TS,
                model=None,
                input_tokens=None,
                output_tokens=None,
                tool_name=None,
                tool_use_id=None,
                body=f"quadratic voting note {i}",
            )
        )
    store.upsert_entities(rows)


def _raw_user_version(db) -> int:
    """Read the stamp OUT OF BAND, with no help from the code under test.

    Deliberately not ``Store.schema_version()``. The claim being checked is
    "the bytes on disk were not changed", and asking the same module that may
    have changed them is how a test ends up agreeing with the bug.
    """
    c = sqlite3.connect(db)
    try:
        return int(c.execute("PRAGMA user_version").fetchone()[0])
    finally:
        c.close()


def _raw_meta_schema_version(db) -> str | None:
    c = sqlite3.connect(db)
    try:
        row = c.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        return None if row is None else str(row[0])
    finally:
        c.close()


def _raw_table_names(db) -> set[str]:
    c = sqlite3.connect(db)
    try:
        return {
            r[0]
            for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    finally:
        c.close()


@pytest.fixture
def current_db(tmp_path):
    """A cache this build migrated itself. The control for everything here."""
    db = tmp_path / "cache.db"
    store = Store(db_path=db)
    store.migrate()
    _seed(store)
    store.close()
    return db


@pytest.fixture
def ahead_db(current_db):
    """The same cache, re-stamped one schema version FORWARD.

    Both stamps are moved, because a genuine cache written by a newer build
    carries both: ``migrate()`` writes ``PRAGMA user_version`` and the
    ``meta.schema_version`` row in the same commit. A fixture that moved only
    the pragma would leave the guard free to pass by consulting the row it
    happened not to touch.

    The tables are left in their v6 shape on purpose. The refusal under test
    reads the stamp and nothing else — reconstructing a plausible v7 table
    layout would make the fixture slower, more brittle, and no more
    discriminating, and there is no v7 to reconstruct.
    """
    c = sqlite3.connect(current_db)
    c.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1};")
    c.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION + 1),),
    )
    c.commit()
    c.close()
    # The fixture proves its own premise rather than assuming it. A stamp that
    # silently failed to take would make every assertion below vacuous.
    assert _raw_user_version(current_db) == SCHEMA_VERSION + 1
    return current_db


# --- the refusal, and what it must leave behind -----------------------------


def test_migrate_refuses_a_cache_stamped_ahead_of_this_build(ahead_db):
    """THE REPRO. Pre-fix this returned normally, having stamped the cache down."""
    store = Store(db_path=ahead_db)
    with pytest.raises(SchemaAheadError):
        store.migrate()
    store.close()


def test_the_refused_migration_leaves_user_version_exactly_where_it_was(ahead_db):
    """THE REPRO, AND THE POINT OF THE WHOLE CHANGE.

    Raising is not the fix; not writing is the fix. A guard that raised after
    the ``PRAGMA user_version`` statement had already run would satisfy the
    test above and reproduce the incident unchanged, which is why the stamp is
    read back here from a connection the store never touched.
    """
    store = Store(db_path=ahead_db)
    with pytest.raises(SchemaAheadError):
        store.migrate()
    store.close()
    assert _raw_user_version(ahead_db) == SCHEMA_VERSION + 1


def test_the_refused_migration_leaves_the_meta_row_alone_too(ahead_db):
    """The second stamp, written in the same commit as the first.

    Two independent records of the same fact are worth having only while they
    agree; a refusal that reset one and not the other would leave the cache
    self-contradictory and a later diagnosis reading whichever one it happened
    to pick.
    """
    store = Store(db_path=ahead_db)
    with pytest.raises(SchemaAheadError):
        store.migrate()
    store.close()
    assert _raw_meta_schema_version(ahead_db) == str(SCHEMA_VERSION + 1)


def test_the_refusal_lands_before_any_ddl_runs(tmp_path):
    """Nothing at all is written — not the stamp, not the tables, not a row.

    An empty file carrying a forward stamp is the sharpest available probe:
    every table this build would have created is observable by its absence, so
    a guard placed halfway down ``migrate()`` (after the ``CREATE TABLE`` pass,
    before the stamp) fails here and passes everywhere else.
    """
    db = tmp_path / "empty-but-ahead.db"
    c = sqlite3.connect(db)
    c.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1};")
    c.commit()
    c.close()

    store = Store(db_path=db)
    with pytest.raises(SchemaAheadError):
        store.migrate()
    store.close()

    assert _raw_table_names(db) == set()
    assert _raw_user_version(db) == SCHEMA_VERSION + 1


def test_a_cache_far_ahead_is_refused_and_not_merely_the_next_version(ahead_db):
    """The comparison is ``>``, not ``== SCHEMA_VERSION + 1``.

    Skew is not limited to one version of drift — the writer here was three
    months behind its reader — and a guard that only recognised the adjacent
    version would wave through exactly the worst cases.
    """
    c = sqlite3.connect(ahead_db)
    c.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 9};")
    c.commit()
    c.close()

    store = Store(db_path=ahead_db)
    with pytest.raises(SchemaAheadError):
        store.migrate()
    store.close()
    assert _raw_user_version(ahead_db) == SCHEMA_VERSION + 9


# --- what the refusal has to say --------------------------------------------


def _refusal_text(db) -> str:
    store = Store(db_path=db)
    with pytest.raises(SchemaAheadError) as excinfo:
        store.migrate()
    store.close()
    return str(excinfo.value)


def test_the_refusal_names_both_versions_it_compared(ahead_db):
    """A reader who cannot see which two quantities disagreed cannot act.

    Both numbers, because either one alone leaves the reader unable to tell
    whether their cache is exotic or their build is old.
    """
    text = _refusal_text(ahead_db)
    assert str(SCHEMA_VERSION + 1) in text, text
    assert str(SCHEMA_VERSION) in text, text


def test_the_refusal_says_nothing_was_written(ahead_db):
    """The single most valuable sentence in the message.

    The operator's first fear on seeing a schema error is that the cache has
    been half-migrated into an unreadable state. It has not, and saying so is
    what keeps them from reaching for a rebuild that would destroy weeks of
    vectors to fix a problem that does not exist.
    """
    text = _refusal_text(ahead_db).lower()
    assert "nothing was written" in text, text


def test_the_refusal_says_to_bring_this_build_up(ahead_db):
    """Forward, and named as the action to take.

    The lagging side here is the process raising the error, so the remedy is
    to deploy a newer one — not to do anything whatsoever to the cache.
    """
    text = _refusal_text(ahead_db).lower()
    assert "upgrade" in text or "newer build" in text, text


def test_the_refusal_never_offers_to_downgrade_the_cache(ahead_db):
    """Skew is resolved by raising the lagging side, never by lowering the
    leading one. Offering a rollback as one arm of a choice is not a neutral
    listing of options: it costs the reader a round trip to reject it, and one
    reader in ten will take it and lose their cache."""
    text = _refusal_text(ahead_db).lower()
    for banned in ("downgrade", "roll back", "rollback", "revert", "rebuild_all"):
        assert banned not in text, (banned, text)


# --- the destructive neighbour ----------------------------------------------


def test_rebuild_all_refuses_before_it_drops_anything(ahead_db):
    """``rebuild_all()`` drops every table and then calls ``migrate()``.

    Guarding only ``migrate()`` would therefore convert this method into
    "destroy the cache, then refuse to rebuild it" — strictly worse than the
    bug being fixed, because the incident cost recall while this would cost
    the data. The guard has to run before the first ``DROP``.

    Asserted on the surviving rows rather than on the exception, for the same
    reason as every other test in this file.
    """
    store = Store(db_path=ahead_db)
    with pytest.raises(SchemaAheadError):
        store.rebuild_all()
    store.close()

    c = sqlite3.connect(ahead_db)
    try:
        assert c.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 3
    finally:
        c.close()
    assert _raw_user_version(ahead_db) == SCHEMA_VERSION + 1


# --- and the two paths that must not have changed at all --------------------


def test_an_equal_version_cache_still_migrates(current_db):
    """The overwhelmingly common path: every subcommand runs one of these.

    Re-migrating a current cache is a no-op that must stay a no-op — silent,
    idempotent, and no slower for the guard being there.
    """
    store = Store(db_path=current_db)
    store.migrate()
    store.close()
    assert _raw_user_version(current_db) == SCHEMA_VERSION


def test_a_fresh_database_still_migrates(tmp_path):
    """``PRAGMA user_version`` is 0 on a file nothing has migrated.

    Zero is below ``SCHEMA_VERSION``, not above it, so a fresh machine's very
    first ingest must be entirely ordinary. A guard that read the stamp with
    the comparison inverted would brick exactly this case, and it is the one
    case with no cache to inspect afterwards.
    """
    db = tmp_path / "fresh.db"
    store = Store(db_path=db)
    store.migrate()
    store.close()
    assert _raw_user_version(db) == SCHEMA_VERSION
    assert "observations" in _raw_table_names(db)


def test_a_cache_behind_this_build_still_migrates_forward(current_db):
    """The direction that WAS always allowed and must remain allowed.

    A v5 cache under a v6 writer is the ordinary upgrade, and it is the thing
    the whole schema mechanism exists to do. Refusing in both directions would
    turn one incident into two.
    """
    c = sqlite3.connect(current_db)
    c.execute(f"PRAGMA user_version = {SCHEMA_VERSION - 1};")
    c.commit()
    c.close()

    store = Store(db_path=current_db)
    store.migrate()
    store.close()
    assert _raw_user_version(current_db) == SCHEMA_VERSION
