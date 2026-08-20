"""The frozen golden query set, and the loader that refuses a broken one.

The set is a DATA FILE IN THE REPO, never a table in the user's live cache.
A golden set that lives in the database it measures moves whenever the
database moves, which is the one thing a baseline may not do.

The tests below pin the composition the research report asks for — 50-200
queries, identifier-shaped and natural-language, and negatives so abstention
is testable at all — plus every row of the reported FTS5 failure table, so
criterion B has something to regress against on day one.
"""

import json

import pytest

from aggregator.evals.golden import (
    KINDS,
    GoldenQuery,
    GoldenSetError,
    golden_set_path,
    load_golden_queries,
    load_labels,
    suggest_from_misses,
)


@pytest.fixture
def golden():
    return load_golden_queries()


# --- the shipped set --------------------------------------------------------


def test_the_shipped_golden_set_loads():
    assert golden_set_path().exists()
    assert load_golden_queries()


def test_the_set_is_inside_the_cited_50_to_200_band(golden):
    assert 50 <= len(golden) <= 200


def test_query_ids_are_unique(golden):
    ids = [q.id for q in golden]
    assert len(ids) == len(set(ids))


def test_every_kind_the_report_asks_for_is_represented(golden):
    kinds = {q.kind for q in golden}
    assert {"identifier", "natural", "negative"} <= kinds
    assert kinds <= KINDS


def test_there_are_enough_negatives_to_measure_abstention(golden):
    negatives = [q for q in golden if q.is_negative]
    assert len(negatives) >= 10


def test_negative_queries_are_the_only_ones_expected_to_abstain(golden):
    for q in golden:
        assert q.is_negative == (q.kind == "negative")


def test_the_reported_fts5_failure_table_is_frozen_into_the_set(golden):
    """Every row of the report's failure table, verbatim. Criterion B's target."""
    texts = {q.query for q in golden}
    for failing in ("power-on", "on/off toggle", "C++ 17", "wifi-6E", '"unbalanced quote'):
        assert failing in texts, f"{failing!r} missing from the golden set"


def test_the_fts5_failure_rows_are_tagged_so_they_can_be_selected(golden):
    tagged = {q.query for q in golden if "fts5-failure" in q.tags}
    assert "on/off toggle" in tagged
    assert len(tagged) >= 5


def test_identifier_queries_cover_the_shapes_the_corpus_is_saturated_with(golden):
    texts = {q.query for q in golden}
    assert "ERR_TLS_CERT" in texts
    assert "wifi-6E" in texts
    assert any(q.startswith("#") and q[1:].isdigit() for q in texts), "no PR-number query"


def test_queries_seeded_from_the_repo_are_marked_as_such(golden):
    origins = {q.origin for q in golden}
    assert "repo" in origins
    assert "authored" in origins


def test_no_query_is_blank(golden):
    for q in golden:
        assert q.query.strip(), f"{q.id} has an empty query"


# --- the loader refuses bad data -------------------------------------------


def _write(tmp_path, payload):
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loader_rejects_duplicate_ids(tmp_path):
    path = _write(
        tmp_path,
        {
            "version": 1,
            "queries": [
                {"id": "dup", "query": "a", "kind": "natural", "origin": "authored"},
                {"id": "dup", "query": "b", "kind": "natural", "origin": "authored"},
            ],
        },
    )
    with pytest.raises(GoldenSetError, match="duplicate"):
        load_golden_queries(path)


def test_loader_rejects_an_unknown_kind(tmp_path):
    path = _write(
        tmp_path,
        {
            "version": 1,
            "queries": [
                {"id": "q1", "query": "a", "kind": "vibes", "origin": "authored"}
            ],
        },
    )
    with pytest.raises(GoldenSetError, match="kind"):
        load_golden_queries(path)


def test_loader_rejects_a_blank_query(tmp_path):
    path = _write(
        tmp_path,
        {
            "version": 1,
            "queries": [
                {"id": "q1", "query": "   ", "kind": "natural", "origin": "authored"}
            ],
        },
    )
    with pytest.raises(GoldenSetError, match="empty"):
        load_golden_queries(path)


def test_loader_rejects_an_empty_set_rather_than_returning_nothing(tmp_path):
    """An empty golden set would make every later run report a clean bill."""
    path = _write(tmp_path, {"version": 1, "queries": []})
    with pytest.raises(GoldenSetError, match="empty"):
        load_golden_queries(path)


def test_loader_rejects_an_unreadable_file(tmp_path):
    path = tmp_path / "nope.json"
    with pytest.raises(GoldenSetError):
        load_golden_queries(path)


# --- labels are optional, and their absence is explicit ---------------------


def test_labels_are_empty_when_nobody_has_labelled_anything(tmp_path):
    assert load_labels(tmp_path / "absent.json") == {}


def test_labels_load_as_query_id_to_result_grade(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"labels": {"q1": {"o1": 3, "o2": 1}}}), encoding="utf-8")
    assert load_labels(path) == {"q1": {"o1": 3, "o2": 1}}


def test_labels_for_an_unknown_shape_fail_loudly(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"labels": {"q1": ["o1", "o2"]}}), encoding="utf-8")
    with pytest.raises(GoldenSetError):
        load_labels(path)


# --- growing the set from the zero-result log -------------------------------


def test_zero_result_queries_are_suggested_as_golden_candidates():
    existing = [
        GoldenQuery(id="a", query="already in", kind="natural", origin="repo")
    ]
    misses = [{"query_text": "already in"}, {"query_text": "brand new miss"}]
    assert suggest_from_misses(misses, existing) == ["brand new miss"]


def test_repeated_misses_are_suggested_once():
    misses = [{"query_text": "same"}, {"query_text": "same"}]
    assert suggest_from_misses(misses, []) == ["same"]
