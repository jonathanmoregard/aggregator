"""The FTS5 arm must survive a hit list bigger than SQLite's parameter cap.

``SQLITE_MAX_VARIABLE_NUMBER`` is 32,766 on this build, and the live corpus is
483,193 observations, so a common word clears the cap easily. Every one of the
three binding sites below used to emit one ``?`` per hit, and they failed two
different ways — which is worse than failing one way, because the two disagree
about the same query:

* ``query_observations`` and ``query_sessions`` wrap their execute in
  ``except sqlite3.OperationalError -> []``, so they returned an EMPTY PAGE. A
  recall tool answering "nothing found" when the truth is "tens of thousands of
  hits" is the quietest possible wrong answer.
* ``count_observations`` and ``count_sessions`` do not wrap theirs, so they
  raised ``too many SQL variables`` at the caller.

Reproduced at real scale rather than against a lowered cap: the point is the
corpus size, and a mocked limit would not have shown that the two surfaces
disagree.

Round 1 fixed exactly this class in ``_apply_id_scope`` (the hybrid arm) with
``json_each``. These tests pin the same technique across the FTS5 arm — one
bound parameter per id set, however many ids — because two techniques for one
defect is how the next site gets missed.
"""

import pytest

from aggregator.core.store import Store
from aggregator.sources.base import QueryAST

#: Comfortably over the 32,766 cap. Enough to fail the old binding, small
#: enough that seeding costs about a second.
N = 33_000


@pytest.fixture(scope="module")
def big_corpus(tmp_path_factory):
    """One session and one matching observation per id, plus one subagent."""
    db = tmp_path_factory.mktemp("scale") / "cache.db"
    store = Store(db_path=db)
    store.migrate()
    c = store._c()
    c.executemany(
        "INSERT INTO sessions(session_id, root_session_id, kind, first_ts, "
        "last_ts, jsonl_path, origin) VALUES (?,?,'session','2026-01-01',"
        "'2026-01-01','/tmp/x.jsonl','claude-code')",
        [(f"s{i}", f"s{i}") for i in range(N)],
    )
    c.executemany(
        "INSERT INTO observations(obs_id, session_id, root_session_id, type, "
        "ts, body) VALUES (?,?,?,'user','2026-01-01',?)",
        [(f"o{i}", f"s{i}", f"s{i}", f"aggregator note {i}") for i in range(N)],
    )
    # One subagent so the second half of the scope clause carries a real id
    # rather than only an empty set.
    c.execute(
        "INSERT INTO sessions(session_id, root_session_id, kind, first_ts, "
        "last_ts, jsonl_path, origin) VALUES ('s0/sub','s0','subagent',"
        "'2026-01-01','2026-01-01','/tmp/x.jsonl','claude-code')"
    )
    c.execute(
        "INSERT INTO observations(obs_id, session_id, root_session_id, type, "
        "ts, body) VALUES ('osub','s0/sub','s0','user','2026-01-01',"
        "'aggregator note from a subagent')"
    )
    c.commit()
    yield store
    store.close()


def test_the_fixture_really_is_over_the_parameter_cap(big_corpus):
    """Otherwise every assertion below passes for the wrong reason."""
    assert len(big_corpus._fts_obs_ids("aggregator")) > 32_766


def test_query_observations_does_not_answer_empty_at_scale(big_corpus):
    rows = big_corpus.query_observations(QueryAST(text="aggregator"), limit=25)
    assert len(rows) == 25


def test_count_observations_does_not_raise_at_scale(big_corpus):
    assert big_corpus.count_observations(QueryAST(text="aggregator")) == N + 1


def test_query_sessions_does_not_answer_empty_at_scale(big_corpus):
    rows = big_corpus.query_sessions(QueryAST(text="aggregator"), limit=25)
    assert len(rows) == 25


def test_count_sessions_does_not_raise_at_scale(big_corpus):
    # Every top-level session, plus the one subagent card.
    assert big_corpus.count_sessions(QueryAST(text="aggregator")) == N + 1


def test_the_two_surfaces_agree_at_scale(big_corpus):
    """The old failure was asymmetric, and that is its own bug: paging asked
    ``count_*`` for a total and ``query_*`` for the page."""
    ast = QueryAST(text="aggregator")
    assert big_corpus.count_observations(ast) == len(
        big_corpus.query_observations(ast)
    )
    assert big_corpus.count_sessions(ast) == len(big_corpus.query_sessions(ast))


def test_the_scope_still_narrows_rather_than_matching_everything(big_corpus):
    """A clause that bound nothing would also 'pass' every test above."""
    ast = QueryAST(text="note 4242")
    rows = big_corpus.query_observations(ast)
    assert [r.obs_id for r in rows] == ["o4242"]
    assert big_corpus.count_observations(ast) == 1
    assert [s.session_id for s in big_corpus.query_sessions(ast)] == ["s4242"]


def test_the_subagent_half_of_the_scope_still_binds(big_corpus):
    """Subagent cards surface only on a hit in their OWN stream."""
    ast = QueryAST(text="subagent")
    kinds = {s.session_id for s in big_corpus.query_sessions(ast)}
    assert kinds == {"s0", "s0/sub"}
