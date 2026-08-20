"""The eval database: baselines, run history, and the zero-result log.

Schema shape lifted from ``shawnhack/exocortex`` as the research report §8
describes it: ``retrieval_regression_baselines`` (golden query -> baseline
result ids), ``retrieval_regression_runs`` (run history + drift metrics),
``search_misses`` (zero-result query log).

IT IS A SEPARATE FILE FROM ``cache.db`` ON PURPOSE. The live cache is 1.2 GB,
WAL-hot, and has an ingest timer writing to it every 30 minutes; an eval
harness that migrates it to store its own bookkeeping would be a schema change
to the artifact under measurement. The constructor refuses that path outright.
"""

from datetime import UTC, datetime

import pytest

from aggregator.evals.db import EvalStore, EvalStoreError, default_eval_db_path

FROZEN = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)


@pytest.fixture
def eval_store(tmp_path):
    store = EvalStore(tmp_path / "retrieval_eval.db")
    yield store
    store.close()


# --- it is never the live cache --------------------------------------------


def test_the_eval_db_is_not_the_live_cache(tmp_data_home):
    assert default_eval_db_path().name != "cache.db"
    assert default_eval_db_path().name == "retrieval_eval.db"


def test_opening_cache_db_as_an_eval_store_is_refused(tmp_path):
    with pytest.raises(EvalStoreError, match="cache.db"):
        EvalStore(tmp_path / "cache.db")


def test_the_default_eval_db_lives_under_xdg_data_home(tmp_data_home):
    assert str(default_eval_db_path()).startswith(str(tmp_data_home))


# --- baselines --------------------------------------------------------------


def test_a_frozen_baseline_reads_back_in_rank_order(eval_store):
    eval_store.freeze_baseline(
        {"q1": ["o3", "o1", "o2"]},
        mode="lexical",
        query_texts={"q1": "power-on"},
        frozen_at=FROZEN,
    )
    assert eval_store.baseline("lexical") == {"q1": ["o3", "o1", "o2"]}


def test_an_abstaining_query_freezes_an_empty_baseline_not_a_missing_one(eval_store):
    """A negative query's baseline is "nothing", and that has to be recorded.

    Absent rows would be indistinguishable from "never frozen", and the run
    would skip exactly the queries that test abstention.
    """
    eval_store.freeze_baseline(
        {"neg1": []},
        mode="lexical",
        query_texts={"neg1": "recipe for beef wellington"},
        frozen_at=FROZEN,
    )
    assert eval_store.baseline("lexical") == {"neg1": []}
    assert eval_store.has_baseline("lexical") is True


def test_baselines_are_kept_per_mode(eval_store):
    eval_store.freeze_baseline(
        {"q1": ["a"]}, mode="lexical", query_texts={"q1": "x"}, frozen_at=FROZEN
    )
    eval_store.freeze_baseline(
        {"q1": ["b"]}, mode="hybrid", query_texts={"q1": "x"}, frozen_at=FROZEN
    )
    assert eval_store.baseline("lexical") == {"q1": ["a"]}
    assert eval_store.baseline("hybrid") == {"q1": ["b"]}


def test_refreezing_replaces_the_previous_baseline_for_that_mode(eval_store):
    eval_store.freeze_baseline(
        {"q1": ["a", "b", "c"]},
        mode="lexical",
        query_texts={"q1": "x"},
        frozen_at=FROZEN,
    )
    eval_store.freeze_baseline(
        {"q1": ["z"]}, mode="lexical", query_texts={"q1": "x"}, frozen_at=FROZEN
    )
    assert eval_store.baseline("lexical") == {"q1": ["z"]}


def test_no_baseline_reads_as_no_baseline(eval_store):
    assert eval_store.has_baseline("lexical") is False
    assert eval_store.baseline("lexical") == {}


def test_the_frozen_query_text_is_kept_so_a_stale_baseline_is_visible(eval_store):
    eval_store.freeze_baseline(
        {"q1": ["a"]},
        mode="lexical",
        query_texts={"q1": "power-on"},
        frozen_at=FROZEN,
    )
    assert eval_store.baseline_query_texts("lexical") == {"q1": "power-on"}


# --- run history ------------------------------------------------------------


def test_a_run_is_persisted_and_reads_back(eval_store):
    eval_store.record_run(
        run_id="r1",
        mode="lexical",
        started_at=FROZEN,
        label="feat/rag-hybrid-v2",
        queries=3,
        mean_drift=0.25,
        max_drift=0.9,
        drifted_queries=1,
        top1_changes=1,
        abstention_violations=0,
        labelled_queries=0,
        ndcg_at_10=None,
        recall_at_50=None,
        mrr_at_10=None,
        detail={"q1": 0.9},
    )
    runs = eval_store.runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == "r1"
    assert runs[0]["mean_drift"] == pytest.approx(0.25)
    assert runs[0]["detail"] == {"q1": 0.9}


def test_an_unlabelled_run_stores_null_metrics_never_zero(eval_store):
    """SQL NULL, so "nobody labelled this" cannot be read as "scored 0.0"."""
    eval_store.record_run(
        run_id="r1",
        mode="lexical",
        started_at=FROZEN,
        label=None,
        queries=1,
        mean_drift=0.0,
        max_drift=0.0,
        drifted_queries=0,
        top1_changes=0,
        abstention_violations=0,
        labelled_queries=0,
        ndcg_at_10=None,
        recall_at_50=None,
        mrr_at_10=None,
        detail={},
    )
    assert eval_store.runs()[0]["ndcg_at_10"] is None


def test_runs_come_back_newest_first(eval_store):
    for n, day in enumerate((18, 19, 20), start=1):
        eval_store.record_run(
            run_id=f"r{n}",
            mode="lexical",
            started_at=datetime(2026, 8, day, tzinfo=UTC),
            label=None,
            queries=1,
            mean_drift=0.0,
            max_drift=0.0,
            drifted_queries=0,
            top1_changes=0,
            abstention_violations=0,
            labelled_queries=0,
            ndcg_at_10=None,
            recall_at_50=None,
            mrr_at_10=None,
            detail={},
        )
    assert [r["run_id"] for r in eval_store.runs()] == ["r3", "r2", "r1"]


# --- the zero-result log ----------------------------------------------------


def test_a_zero_result_query_is_logged_and_reads_back(eval_store):
    eval_store.record_search_miss("wifi-6E", mode="hybrid", observed_at=FROZEN)
    misses = eval_store.search_misses()
    assert [m["query_text"] for m in misses] == ["wifi-6E"]
    assert misses[0]["mode"] == "hybrid"
    assert misses[0]["result_count"] == 0


def test_the_same_miss_seen_twice_is_counted_not_deduplicated(eval_store):
    """Frequency is the ranking signal for which misses become golden queries."""
    eval_store.record_search_miss("wifi-6E", observed_at=FROZEN)
    eval_store.record_search_miss("wifi-6E", observed_at=FROZEN)
    assert len(eval_store.search_misses()) == 2


def test_a_miss_that_cannot_be_logged_fails_loudly(eval_store):
    """No swallowed exceptions: a dead miss log must not look like no misses.

    Silently dropping writes here would starve the golden set of exactly the
    queries worth freezing, and nothing would ever say so.
    """
    eval_store.close()
    with pytest.raises(EvalStoreError, match="closed"):
        eval_store.record_search_miss("after close", observed_at=FROZEN)
