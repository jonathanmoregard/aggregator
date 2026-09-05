"""Progressive lexical relaxation: strict AND → OR → prefix, one tier at a time.

THE FAILURE THIS REMOVES, the other half of the porter change: free text is
an implicit AND in one observation, so a remembered-gist query of several
words reliably returned nothing at all. The ladder keeps the strict
semantics as tier 1 — a query that matches exactly still answers exactly,
with identical rows — and only when a tier matches NOTHING does the next one
run: the same sanitized conjuncts joined with OR, then OR with a prefix
``*`` on the final conjunct.

TIERS NEVER INTERLEAVE. Exactly one tier's rows are returned — tier N exists
only because every earlier tier was empty — so a relaxed row can never sit
next to an exact row, and the per-tier ordering contract is exactly the
strict path's (the id set feeds the same recency-ordered SQL as before).

TRUTHFULNESS IS STATE, NOT VIBES. The store records which tier answered in
``lexical_relaxation`` (``None`` for exact, ``"or"``, ``"prefix"``), reset
per request at the tool boundary, and the MCP/CLI layers surface it — a
relaxed match must never masquerade as an exact one.

STRICT HELPERS STAY STRICT. ``scope:session`` (per-conjunct intersection),
the health probe, and the internal scope widening keep exact semantics: the
scope:session probe is what diagnoses a dead conjunct, and an OR inside an
intersection-of-conjuncts is a different question nobody asked.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from aggregator.core.store import (
    LEXICAL_RELAX_OR,
    LEXICAL_RELAX_PREFIX,
    Store,
    fts5_match_conjuncts,
    fts5_relaxation_tiers,
)
from aggregator.sources.base import ObservationRow, QueryAST, Record, SessionRow


def _session(session_id: str = "s1") -> SessionRow:
    return SessionRow(
        session_id=session_id,
        root_session_id=session_id,
        parent_session_id=None,
        kind="session",
        agent_id=None,
        agent_type=None,
        spawned_by_tool_use_id=None,
        cwd="/x",
        git_branch="main",
        first_ts=datetime(2026, 7, 25, 10, 0, tzinfo=UTC),
        last_ts=datetime(2026, 7, 25, 10, 5, tzinfo=UTC),
        jsonl_path="/tmp/x.jsonl",
    )


def _obs(obs_id: str, body: str, session_id: str = "s1") -> ObservationRow:
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
    )


def _rec(sid: str, subject: str, body: str) -> Record:
    return Record(
        stable_id=sid,
        source="github",
        subject=subject,
        body=body,
        tags=[],
        created_at=datetime(2026, 7, 25, tzinfo=UTC),
        updated_at=datetime(2026, 7, 25, tzinfo=UTC),
    )


@pytest.fixture
def store(tmp_path):
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    yield s
    s.close()


# --- the tier builder, as a pure function ------------------------------------


def test_tiers_strict_then_or_then_prefix():
    tiers = fts5_relaxation_tiers(fts5_match_conjuncts("alpha beta"))
    assert tiers == [
        (None, '"alpha" "beta"'),
        (LEXICAL_RELAX_OR, '"alpha" OR "beta"'),
        (LEXICAL_RELAX_PREFIX, '"alpha" OR "beta"*'),
    ]


def test_single_token_skips_the_redundant_or_tier():
    """OR over one conjunct IS the strict query; running it again would burn
    an FTS scan to learn nothing."""
    tiers = fts5_relaxation_tiers(fts5_match_conjuncts("alpha"))
    assert tiers == [
        (None, '"alpha"'),
        (LEXICAL_RELAX_PREFIX, '"alpha"*'),
    ]


def test_phrases_stay_phrases_in_every_tier():
    tiers = fts5_relaxation_tiers(fts5_match_conjuncts('"red green" purple'))
    assert tiers == [
        (None, '"red green" "purple"'),
        (LEXICAL_RELAX_OR, '"red green" OR "purple"'),
        (LEXICAL_RELAX_PREFIX, '"red green" OR "purple"*'),
    ]


def test_no_conjuncts_means_no_tiers():
    assert fts5_relaxation_tiers([]) == []


# --- observations arm --------------------------------------------------------


def test_strict_match_returns_exact_rows_and_no_marker(store):
    store.upsert_entities(
        [
            _session(),
            _obs("o-both", "alpha and beta together"),
            _obs("o-alpha", "alpha alone"),
        ]
    )
    assert store._fts_obs_ids("alpha beta") == ["o-both"]
    assert store.lexical_relaxation is None


def test_and_miss_is_rescued_by_or_and_marked(store):
    store.upsert_entities(
        [
            _session(),
            _obs("o-alpha", "alpha alone"),
            _obs("o-beta", "beta alone"),
        ]
    )
    assert set(store._fts_obs_ids("alpha beta")) == {"o-alpha", "o-beta"}
    assert store.lexical_relaxation == LEXICAL_RELAX_OR


def test_or_miss_is_rescued_by_prefix_on_the_final_token(store):
    store.upsert_entities([_session(), _obs("o1", "alphabet soup for lunch")])
    assert store._fts_obs_ids("zzzmissing alphab") == ["o1"]
    assert store.lexical_relaxation == LEXICAL_RELAX_PREFIX


def test_tiers_do_not_interleave(store):
    """When OR matches, prefix-only rows must NOT ride along."""
    store.upsert_entities(
        [
            _session(),
            _obs("o-or", "beta alone"),
            # Matches only via prefix: "betamax" is not the term "beta".
            _obs("o-prefix", "alphabet soup"),
        ]
    )
    assert store._fts_obs_ids("beta alphab") == ["o-or"]
    assert store.lexical_relaxation == LEXICAL_RELAX_OR


def test_all_tiers_empty_returns_empty_and_no_marker(store):
    store.upsert_entities([_session(), _obs("o1", "alpha")])
    assert store._fts_obs_ids("zzz qqq") == []
    assert store.lexical_relaxation is None


def test_reset_clears_the_marker(store):
    store.upsert_entities(
        [_session(), _obs("o-alpha", "alpha alone"), _obs("o-beta", "beta alone")]
    )
    store._fts_obs_ids("alpha beta")
    assert store.lexical_relaxation == LEXICAL_RELAX_OR
    store.reset_lexical_relaxation()
    assert store.lexical_relaxation is None


# --- records arm -------------------------------------------------------------


def test_records_and_miss_is_rescued_by_or(store):
    store.upsert(
        [_rec("r-alpha", "n", "alpha alone"), _rec("r-beta", "n", "beta alone")]
    )
    assert store._fts_ids("alpha beta") == {"r-alpha", "r-beta"}
    assert store.lexical_relaxation == LEXICAL_RELAX_OR


def test_records_exact_match_keeps_no_marker(store):
    store.upsert([_rec("r1", "n", "alpha and beta")])
    assert store._fts_ids("alpha beta") == {"r1"}
    assert store.lexical_relaxation is None


# --- session-card arm --------------------------------------------------------


def test_session_hit_scope_relaxes_like_the_other_arms(store):
    store.upsert_entities(
        [
            _session("s1"),
            _obs("o-alpha", "alpha alone", "s1"),
            _obs("o-beta", "beta alone", "s1"),
        ]
    )
    roots, exacts = store._fts_hit_scope("alpha beta")
    assert roots == {"s1"}
    assert store.lexical_relaxation == LEXICAL_RELAX_OR


# --- strict helpers stay strict ----------------------------------------------


def test_scope_session_keeps_exact_conjunct_semantics(store):
    store.upsert_entities(
        [
            _session("s1"),
            _obs("o-alpha", "alpha alone", "s1"),
            _obs("o-beta", "beta alone", "s1"),
        ]
    )
    # Both conjuncts satisfied somewhere under s1: matches WITHOUT relaxation.
    roots, _ = store._session_hit_scope("alpha beta")
    assert roots == {"s1"}
    assert store.lexical_relaxation is None
    # A conjunct nothing satisfies stays a miss — no OR rescue here: this
    # helper also powers the conjunction-notice probe, whose entire job is
    # reporting the strict answer.
    roots, _ = store._session_hit_scope("alpha zzzmissing")
    assert roots == set()
    assert store.lexical_relaxation is None


def test_phrase_adjacency_survives_relaxation(store):
    store.upsert_entities(
        [
            _session(),
            _obs("o-adjacent", "red green wall"),
            _obs("o-reversed", "green red apple"),
        ]
    )
    # Strict: phrase + missing word -> nothing. OR tier: the PHRASE matches
    # only the adjacent row; the reversed row must not sneak in.
    assert store._fts_obs_ids('"red green" zzzmissing') == ["o-adjacent"]
    assert store.lexical_relaxation == LEXICAL_RELAX_OR


def test_relaxed_query_ast_reaches_query_observations(store):
    """End-to-end through the store's own query path: the OR rescue rows come
    back from ``query_observations`` exactly as strict rows would."""
    store.upsert_entities(
        [
            _session("s1"),
            _obs("o-alpha", "alpha alone", "s1"),
            _obs("o-beta", "beta alone", "s1"),
        ]
    )
    rows = store.query_observations(QueryAST(text="alpha beta"))
    assert {r.obs_id for r in rows} == {"o-alpha", "o-beta"}
    assert store.lexical_relaxation == LEXICAL_RELAX_OR
    assert store.count_observations(QueryAST(text="alpha beta")) == 2


# --- what the ladder RECORDS, and when it records nothing at all -------------
#
# ``lexical_matches`` is three-valued on purpose (see ``LexicalProbe``): None =
# the ladder never ran, 0 = it ran and matched nothing, > 0 = the words are in
# the corpus. Every site that can collapse None into 0 is a site where an
# empty-page notice states a fact about the corpus out of a query that never
# asked it.


def test_a_query_with_no_word_chars_never_records_a_probe(store):
    """NO TIER RAN, SO NOTHING WAS MEASURED. ``fts5_relaxation_tiers`` returns
    an empty ladder for text with no word characters, and the arms used to
    record a zero anyway on the way past — turning "nobody looked" into "we
    looked and the corpus has none of it"."""
    store.upsert_entities([_session(), _obs("o1", "alpha")])
    store.upsert([_rec("r1", "n", "alpha")])
    for probe in (
        lambda: store._fts_obs_ids("---"),
        lambda: store._fts_ids("---"),
        lambda: store._fts_hit_scope("---"),
        lambda: store._session_hit_scope("---"),
    ):
        store.reset_lexical_relaxation()
        probe()
        assert store.lexical_matches is None, probe


def test_a_ladder_that_ran_and_missed_records_a_measured_zero(store):
    """The other half of the same discipline: a real miss IS a measurement."""
    store.upsert_entities([_session(), _obs("o1", "alpha")])
    assert store._fts_obs_ids("zzzmissing qqqmissing") == []
    assert store.lexical_matches == 0


def test_scope_session_records_what_the_intersection_found(store):
    """``scope:session`` used to record NOTHING, so an empty page under it was
    diagnosed from a probe nobody ran — and the notice said the terms "do not
    co-occur in any one session" about terms that do."""
    store.upsert_entities(
        [
            _session("s1"),
            _obs("o-alpha", "alpha alone", "s1"),
            _obs("o-beta", "beta alone", "s1"),
        ]
    )
    roots, _ = store._session_hit_scope("alpha beta")
    assert roots == {"s1"}
    assert store.lexical_matches == 1


def test_scope_session_records_a_measured_zero_for_a_real_miss(store):
    """An intersection that really is empty must still be a MEASURED zero, or
    the honest "they do not co-occur" wording loses its licence."""
    store.upsert_entities(
        [
            _session("s1"),
            _obs("o-alpha", "alpha alone", "s1"),
            _session("s2"),
            _obs("o-beta", "beta alone", "s2"),
        ]
    )
    roots, _ = store._session_hit_scope("alpha beta")
    assert roots == set()
    assert store.lexical_matches == 0


def test_scope_session_obs_ids_record_the_rows_they_saw(store):
    """The widened row set is what a ``scope:session`` drilldown page is built
    from, so it is the count the empty-page notice means by "row(s)"."""
    store.upsert_entities(
        [
            _session("s1"),
            _obs("o-alpha", "alpha alone", "s1"),
            _obs("o-beta", "beta alone", "s1"),
        ]
    )
    ast = QueryAST(text="alpha beta", scope="session")
    assert store._scoped_obs_ids(ast) == ["o-alpha", "o-beta"]
    assert store.lexical_matches == 2
