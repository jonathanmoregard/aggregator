"""The regression run itself: freeze, re-run, report drift — with no labels.

THE LABEL-FREE PATH IS THE ONE THAT HAS TO WORK TODAY. Nobody has hand-labelled
anything, and the report is explicit that you do not need labels to catch a
regression — you need a frozen baseline and a drift metric. So every test here
that exercises drift passes ``labels=None``, and the labelled metrics are
tested for two things only: that they are computable when labels exist, and
that their absence is reported as absence rather than as a score.
"""

import pytest

from aggregator.evals.db import EvalStore
from aggregator.evals.golden import GoldenQuery
from aggregator.evals.harness import (
    BASELINE_DEPTH,
    RUN_DEPTH,
    MissingBaselineError,
    freeze_baseline,
    retrieval_regression_command,
    run_regression,
)


@pytest.fixture
def eval_store(tmp_path):
    store = EvalStore(tmp_path / "retrieval_eval.db")
    yield store
    store.close()


def _q(qid, text, kind="natural"):
    return GoldenQuery(id=qid, query=text, kind=kind, origin="authored")


QUERIES = [
    _q("pos1", "rrf fusion", "natural"),
    _q("pos2", "ERR_TLS_CERT", "identifier"),
    _q("neg1", "recipe for beef wellington", "negative"),
]


def fake_search(table):
    """Build a search fn from ``{query_text: [ids]}``. Missing key -> no hits."""

    def search(text: str, limit: int) -> list[str]:
        return list(table.get(text, []))[:limit]

    return search


STABLE = {
    "rrf fusion": [f"a{i}" for i in range(20)],
    "ERR_TLS_CERT": [f"b{i}" for i in range(20)],
    "recipe for beef wellington": [],
}


# --- freezing ---------------------------------------------------------------


def test_freezing_records_the_top_10_ids_for_every_golden_query(eval_store):
    frozen = freeze_baseline(eval_store, QUERIES, fake_search(STABLE), mode="lexical")
    assert frozen == 3
    baseline = eval_store.baseline("lexical")
    assert set(baseline) == {"pos1", "pos2", "neg1"}
    assert baseline["pos1"] == [f"a{i}" for i in range(BASELINE_DEPTH)]


def test_freezing_records_an_abstention_as_an_empty_baseline(eval_store):
    freeze_baseline(eval_store, QUERIES, fake_search(STABLE), mode="lexical")
    assert eval_store.baseline("lexical")["neg1"] == []


def test_the_baseline_depth_is_ten_and_the_run_depth_reaches_recall_at_50():
    assert BASELINE_DEPTH == 10
    assert RUN_DEPTH >= 50


# --- the label-free regression run -----------------------------------------


def test_running_without_a_baseline_fails_loudly(eval_store):
    """Not "zero drift, all clear" — that is the failure mode being prevented."""
    with pytest.raises(MissingBaselineError, match="lexical"):
        run_regression(eval_store, QUERIES, fake_search(STABLE), mode="lexical")


def test_an_unchanged_retriever_reports_zero_drift(eval_store):
    freeze_baseline(eval_store, QUERIES, fake_search(STABLE), mode="lexical")
    report = run_regression(eval_store, QUERIES, fake_search(STABLE), mode="lexical")
    assert report.mean_drift == pytest.approx(0.0)
    assert report.max_drift == pytest.approx(0.0)
    assert report.drifted_queries == 0
    assert report.top1_changes == 0


def test_a_reordered_result_is_detected_as_drift_with_zero_hand_labels(eval_store):
    freeze_baseline(eval_store, QUERIES, fake_search(STABLE), mode="lexical")
    changed = dict(STABLE)
    changed["rrf fusion"] = ["a5", *[f"a{i}" for i in range(20) if i != 5]]
    report = run_regression(
        eval_store, QUERIES, fake_search(changed), mode="lexical", labels=None
    )
    assert report.drifted_queries == 1
    assert report.top1_changes == 1
    assert report.max_drift > 0.0
    drifted = {o.query_id for o in report.outcomes if o.drift > 0}
    assert drifted == {"pos1"}


def test_a_query_that_stops_returning_anything_is_maximum_drift(eval_store):
    freeze_baseline(eval_store, QUERIES, fake_search(STABLE), mode="lexical")
    broken = dict(STABLE, **{"ERR_TLS_CERT": []})
    report = run_regression(eval_store, QUERIES, fake_search(broken), mode="lexical")
    outcome = next(o for o in report.outcomes if o.query_id == "pos2")
    assert outcome.drift == pytest.approx(1.0)


def test_a_golden_query_with_no_frozen_baseline_fails_loudly(eval_store):
    freeze_baseline(eval_store, QUERIES[:2], fake_search(STABLE), mode="lexical")
    with pytest.raises(MissingBaselineError, match="neg1"):
        run_regression(eval_store, QUERIES, fake_search(STABLE), mode="lexical")


# --- abstention -------------------------------------------------------------


def test_a_negative_query_that_starts_returning_hits_is_an_abstention_violation(
    eval_store,
):
    freeze_baseline(eval_store, QUERIES, fake_search(STABLE), mode="lexical")
    leaky = dict(STABLE, **{"recipe for beef wellington": ["a1", "a2"]})
    report = run_regression(eval_store, QUERIES, fake_search(leaky), mode="lexical")
    assert report.abstention_violations == 1
    outcome = next(o for o in report.outcomes if o.query_id == "neg1")
    assert outcome.abstention_violation is True


def test_a_negative_query_that_keeps_abstaining_is_not_a_violation(eval_store):
    freeze_baseline(eval_store, QUERIES, fake_search(STABLE), mode="lexical")
    report = run_regression(eval_store, QUERIES, fake_search(STABLE), mode="lexical")
    assert report.abstention_violations == 0


def test_a_positive_query_returning_nothing_is_not_an_abstention_violation(eval_store):
    """It is a recall regression, and drift already says so. Keep them distinct."""
    freeze_baseline(eval_store, QUERIES, fake_search(STABLE), mode="lexical")
    broken = dict(STABLE, **{"ERR_TLS_CERT": []})
    report = run_regression(eval_store, QUERIES, fake_search(broken), mode="lexical")
    assert report.abstention_violations == 0
    assert report.max_drift == pytest.approx(1.0)


# --- labels: computable when present, absent when not -----------------------


def test_with_no_labels_the_metrics_are_absent_not_zero(eval_store):
    freeze_baseline(eval_store, QUERIES, fake_search(STABLE), mode="lexical")
    report = run_regression(eval_store, QUERIES, fake_search(STABLE), mode="lexical")
    assert report.labelled_queries == 0
    assert report.ndcg_at_10 is None
    assert report.recall_at_50 is None
    assert report.mrr_at_10 is None
    assert "no labels" in report.labels_note.lower()


def test_the_rendered_report_says_no_labels_rather_than_printing_a_zero(eval_store):
    freeze_baseline(eval_store, QUERIES, fake_search(STABLE), mode="lexical")
    report = run_regression(eval_store, QUERIES, fake_search(STABLE), mode="lexical")
    text = report.to_text()
    assert "nDCG@10" in text
    assert "0.000" not in text.split("nDCG@10")[1].split("\n")[0]
    assert "no labels" in text.lower()


def test_labelled_metrics_are_computed_when_labels_exist(eval_store):
    freeze_baseline(eval_store, QUERIES, fake_search(STABLE), mode="lexical")
    labels = {"pos1": {"a0": 3, "a1": 1}}
    report = run_regression(
        eval_store, QUERIES, fake_search(STABLE), mode="lexical", labels=labels
    )
    assert report.labelled_queries == 1
    assert report.ndcg_at_10 == pytest.approx(1.0)
    assert report.recall_at_50 == pytest.approx(1.0)
    assert report.mrr_at_10 == pytest.approx(1.0)


def test_a_partly_labelled_set_averages_only_the_labelled_queries(eval_store):
    freeze_baseline(eval_store, QUERIES, fake_search(STABLE), mode="lexical")
    labels = {"pos1": {"a0": 3}, "pos2": {"unreachable": 3}}
    report = run_regression(
        eval_store, QUERIES, fake_search(STABLE), mode="lexical", labels=labels
    )
    assert report.labelled_queries == 2
    assert report.mrr_at_10 == pytest.approx(0.5)


def test_labels_for_a_query_that_is_not_in_the_golden_set_fail_loudly(eval_store):
    freeze_baseline(eval_store, QUERIES, fake_search(STABLE), mode="lexical")
    with pytest.raises(ValueError, match="ghost"):
        run_regression(
            eval_store,
            QUERIES,
            fake_search(STABLE),
            mode="lexical",
            labels={"ghost": {"a0": 3}},
        )


# --- what the number cannot see ---------------------------------------------
#
# A metric that cannot reach the layer under test must not print a confident
# zero. Criteria D, G and H all changed ``aggregator.mcp`` and all three were
# reported at 0.000 drift by a ``lexical`` run that talks to the Store — so the
# zero was structural, not evidence, and nothing in the output said so.


def test_the_report_names_the_layers_its_mode_cannot_see(eval_store):
    freeze_baseline(eval_store, QUERIES, fake_search(STABLE), mode="lexical")
    report = run_regression(eval_store, QUERIES, fake_search(STABLE), mode="lexical")
    text = report.to_text()
    assert "scope" in text.lower()
    assert "aggregator.mcp" in text, (
        "a lexical run must name the module it cannot reach; without that, "
        "0.000 reads as evidence about the whole pipeline"
    )


def test_a_zero_drift_run_says_a_zero_is_not_a_clean_bill_of_health(eval_store):
    freeze_baseline(eval_store, QUERIES, fake_search(STABLE), mode="lexical")
    report = run_regression(eval_store, QUERIES, fake_search(STABLE), mode="lexical")
    assert report.mean_drift == pytest.approx(0.0)
    text = report.to_text()
    assert "not evidence" in text.lower(), text


def test_a_run_that_moved_does_not_carry_the_zero_caveat(eval_store):
    """The caveat is about a zero specifically. Printed on every run it would
    become furniture and stop being read."""
    freeze_baseline(eval_store, QUERIES, fake_search(STABLE), mode="lexical")
    moved = dict(STABLE)
    moved["rrf fusion"] = list(reversed(STABLE["rrf fusion"]))
    report = run_regression(eval_store, QUERIES, fake_search(moved), mode="lexical")
    assert report.mean_drift > 0.0
    assert "not evidence" not in report.to_text().lower()


def test_every_mode_has_a_scope_note(eval_store):
    """A mode added without one would print an empty caveat, which reads as
    'nothing is out of scope' — the opposite of the truth."""
    from aggregator.evals.harness import MODE_SCOPE
    from aggregator.evals.search import SEARCH_MODES

    assert set(MODE_SCOPE) == set(SEARCH_MODES)
    assert all(MODE_SCOPE[m].strip() for m in SEARCH_MODES)


# --- persistence ------------------------------------------------------------


def test_the_run_is_written_to_the_run_history(eval_store):
    freeze_baseline(eval_store, QUERIES, fake_search(STABLE), mode="lexical")
    report = run_regression(
        eval_store, QUERIES, fake_search(STABLE), mode="lexical", label="HEAD"
    )
    runs = eval_store.runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == report.run_id
    assert runs[0]["label"] == "HEAD"
    assert runs[0]["queries"] == 3


def test_two_runs_accumulate_so_drift_is_comparable_over_time(eval_store):
    freeze_baseline(eval_store, QUERIES, fake_search(STABLE), mode="lexical")
    run_regression(eval_store, QUERIES, fake_search(STABLE), mode="lexical")
    run_regression(eval_store, QUERIES, fake_search(STABLE), mode="lexical")
    assert len(eval_store.runs()) == 2


# --- the CLI entry point ----------------------------------------------------


def test_freeze_then_run_is_green(eval_store, capsys):
    assert (
        retrieval_regression_command(
            "freeze",
            mode="lexical",
            queries=QUERIES,
            search=fake_search(STABLE),
            eval_store=eval_store,
        )
        == 0
    )
    assert (
        retrieval_regression_command(
            "run",
            mode="lexical",
            queries=QUERIES,
            search=fake_search(STABLE),
            eval_store=eval_store,
        )
        == 0
    )
    assert "drift" in capsys.readouterr().out.lower()


def test_an_abstention_violation_exits_non_zero(eval_store):
    retrieval_regression_command(
        "freeze",
        mode="lexical",
        queries=QUERIES,
        search=fake_search(STABLE),
        eval_store=eval_store,
    )
    leaky = dict(STABLE, **{"recipe for beef wellington": ["a1"]})
    assert (
        retrieval_regression_command(
            "run",
            mode="lexical",
            queries=QUERIES,
            search=fake_search(leaky),
            eval_store=eval_store,
        )
        == 1
    )


def test_drift_alone_does_not_fail_the_command_unless_a_threshold_is_set(eval_store):
    """Drift is directionless. Failing on any drift would block every change."""
    retrieval_regression_command(
        "freeze",
        mode="lexical",
        queries=QUERIES,
        search=fake_search(STABLE),
        eval_store=eval_store,
    )
    churned = dict(STABLE, **{"rrf fusion": [f"z{i}" for i in range(20)]})
    assert (
        retrieval_regression_command(
            "run",
            mode="lexical",
            queries=QUERIES,
            search=fake_search(churned),
            eval_store=eval_store,
        )
        == 0
    )
    assert (
        retrieval_regression_command(
            "run",
            mode="lexical",
            queries=QUERIES,
            search=fake_search(churned),
            eval_store=eval_store,
            drift_threshold=0.1,
        )
        == 1
    )


def test_running_without_a_baseline_exits_non_zero_with_a_message(eval_store, capsys):
    code = retrieval_regression_command(
        "run",
        mode="lexical",
        queries=QUERIES,
        search=fake_search(STABLE),
        eval_store=eval_store,
    )
    assert code == 2
    assert "freeze" in capsys.readouterr().err.lower()


def test_an_unknown_action_exits_non_zero(eval_store, capsys):
    assert (
        retrieval_regression_command(
            "demolish",
            mode="lexical",
            queries=QUERIES,
            search=fake_search(STABLE),
            eval_store=eval_store,
        )
        == 2
    )
    assert "demolish" in capsys.readouterr().err


def test_unfrozen_zero_result_queries_are_surfaced_for_the_golden_set(
    eval_store, capsys
):
    eval_store.record_search_miss("some query nobody froze")
    retrieval_regression_command(
        "freeze",
        mode="lexical",
        queries=QUERIES,
        search=fake_search(STABLE),
        eval_store=eval_store,
    )
    retrieval_regression_command(
        "run",
        mode="lexical",
        queries=QUERIES,
        search=fake_search(STABLE),
        eval_store=eval_store,
    )
    assert "some query nobody froze" in capsys.readouterr().out
