"""Freeze a baseline, re-run it later, report drift. The CLI entry point.

TODO(cli-wiring): ``aggregator/cli.py`` is owned by another change this wave.
Wire ``aggregator.evals.harness.retrieval_regression_command`` in as
``aggregator retrieval-regression [freeze|run] [--mode] [--drift-threshold]``
once that lands. Nothing in this package imports ``cli``.

THE ORDER OF OPERATIONS IS THE WHOLE VALUE. Freeze BEFORE changing retrieval,
re-run AFTER, and the diff between the two runs is evidence. Freeze after the
change and you have baselined the bug.

WHAT THE DRIFT NUMBER MEANS, precisely, because a metric nobody can interpret
gets ignored:

* It is ``1 - normalized RBO`` between the frozen top-10 and this run's top-10,
  per query, averaged. 0.0 is "byte-identical ranking"; 1.0 is "shares nothing,
  at any prefix", which includes a query that used to return results and now
  returns none.
* It is TOP-WEIGHTED. A different first hit costs far more than a reshuffle at
  rank 9, because the first hit is what an agent actually reads.
* IT IS DIRECTIONLESS, AND THAT IS DELIBERATE. Fixing the FTS5 escaping so
  ``power-on`` finally returns results scores exactly the same drift as a bug
  that makes it return noise. Drift says WHERE the ranking moved, never whether
  the move was good. Only labels — or a human looking at the diff — answer
  that. So drift never fails the command on its own unless a caller passes an
  explicit ``drift_threshold``; failing on any drift would block every
  intentional improvement.
* It says nothing about the QUALITY of the baseline. A frozen-in bad ranking
  reproduces at zero drift forever. That is what the labelled metrics and the
  negative queries are for.

THE ONE THING THAT IS UNAMBIGUOUS WITHOUT LABELS is abstention: a negative
query that returned nothing and now returns something is a regression by
construction, and it is the only condition that fails the command by default.
"""

from __future__ import annotations

import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from aggregator.evals.db import EvalStore, EvalStoreError
from aggregator.evals.golden import (
    GoldenQuery,
    GoldenSetError,
    load_golden_queries,
    load_labels,
    suggest_from_misses,
)
from aggregator.evals.metrics import (
    NoLabelsError,
    drift,
    mrr_at_k,
    ndcg_at_k,
    overlap_at_k,
    recall_at_k,
    top1_changed,
)
from aggregator.evals.search import SearchFn, resolve_search_fn

#: How many ids per query the baseline freezes. Ten, per the report.
BASELINE_DEPTH = 10

#: How deep each run retrieves. Fifty, because Recall@50 is the rerank ceiling
#: and it cannot be computed from a shallower list.
RUN_DEPTH = 50

#: Below this, a query counts as unchanged. Not zero: floating-point noise in
#: the fused score can reorder tied results without anything having changed.
DRIFT_EPSILON = 1e-9


class MissingBaselineError(RuntimeError):
    """Raised when a run has no baseline to compare against.

    Its own type, and it stops the run dead, because the alternative is a
    report of "0.000 drift, all clear" produced by comparing against nothing.
    """


@dataclass(frozen=True)
class QueryOutcome:
    query_id: str
    query: str
    kind: str
    result_ids: list[str]
    baseline_ids: list[str]
    drift: float
    overlap_at_10: float
    top1_changed: bool
    abstention_violation: bool
    ndcg_at_10: float | None = None
    recall_at_50: float | None = None
    mrr_at_10: float | None = None


@dataclass(frozen=True)
class RegressionReport:
    run_id: str
    mode: str
    started_at: datetime
    label: str | None
    outcomes: list[QueryOutcome]
    mean_drift: float
    max_drift: float
    drifted_queries: int
    top1_changes: int
    abstention_violations: int
    negative_queries: int
    labelled_queries: int
    labels_note: str
    ndcg_at_10: float | None = None
    recall_at_50: float | None = None
    mrr_at_10: float | None = None
    suggested_queries: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        """Human-readable summary. Unlabelled metrics render as '—', never 0."""

        def metric(name: str, value: float | None) -> str:
            if value is None:
                return f"  {name:<12} —  ({self.labels_note})"
            return f"  {name:<12} {value:.3f}  ({self.labels_note})"

        total = len(self.outcomes)
        lines = [
            f"retrieval regression {self.run_id}  mode={self.mode}  "
            f"queries={total}"
            + (f"  label={self.label}" if self.label else ""),
            f"  drift        mean {self.mean_drift:.3f}  max {self.max_drift:.3f}  "
            f"moved {self.drifted_queries}/{total}  top-1 changes "
            f"{self.top1_changes}",
            f"  abstention   {self.abstention_violations} violation(s) across "
            f"{self.negative_queries} negative queries",
            metric("nDCG@10", self.ndcg_at_10),
            metric("Recall@50", self.recall_at_50),
            metric("MRR@10", self.mrr_at_10),
        ]
        moved = sorted(
            (o for o in self.outcomes if o.drift > DRIFT_EPSILON),
            key=lambda o: o.drift,
            reverse=True,
        )
        if moved:
            lines.append("  queries that moved (largest first):")
            lines.extend(
                f"    {o.drift:.3f}  {o.query_id:<24} {o.query!r}" for o in moved[:20]
            )
        violations = [o for o in self.outcomes if o.abstention_violation]
        if violations:
            lines.append("  ABSTENTION VIOLATIONS — these must return nothing:")
            lines.extend(
                f"    {o.query_id:<24} {o.query!r} -> {len(o.result_ids)} result(s)"
                for o in violations
            )
        if self.suggested_queries:
            lines.append(
                "  zero-result queries seen in the wild and not yet frozen "
                "(add deliberately to golden_queries.json):"
            )
            lines.extend(f"    {q!r}" for q in self.suggested_queries[:20])
        return "\n".join(lines)


def freeze_baseline(
    eval_store: EvalStore,
    queries: Sequence[GoldenQuery],
    search: SearchFn,
    *,
    mode: str = "lexical",
    depth: int = BASELINE_DEPTH,
    now: datetime | None = None,
) -> int:
    """Record the current top-``depth`` result ids for every golden query.

    An abstaining query freezes an EMPTY list, and that empty list is recorded
    as a fact rather than as an absent row — otherwise the run would skip
    exactly the negative queries that make abstention testable.
    """
    results = {q.id: list(search(q.query, depth))[:depth] for q in queries}
    texts = {q.id: q.query for q in queries}
    return eval_store.freeze_baseline(
        results, mode=mode, query_texts=texts, frozen_at=now or datetime.now(UTC)
    )


def run_regression(
    eval_store: EvalStore,
    queries: Sequence[GoldenQuery],
    search: SearchFn,
    *,
    mode: str = "lexical",
    labels: Mapping[str, Mapping[str, int]] | None = None,
    run_id: str | None = None,
    label: str | None = None,
    now: datetime | None = None,
    persist: bool = True,
) -> RegressionReport:
    """Re-run the golden set and report drift against the frozen baseline."""
    baseline = eval_store.baseline(mode)
    if not baseline:
        raise MissingBaselineError(
            f"no frozen baseline for mode {mode!r}; run the freeze action first. "
            "Reporting zero drift against nothing would be worse than failing."
        )
    known = {q.id for q in queries}
    for query in queries:
        if query.id not in baseline:
            raise MissingBaselineError(
                f"golden query {query.id!r} has no frozen baseline for mode "
                f"{mode!r}; re-run the freeze action so every query is covered"
            )
    labels = dict(labels or {})
    for qid in labels:
        if qid not in known:
            raise ValueError(
                f"labels reference query id {qid!r}, which is not in the golden "
                "set; a typo here would silently score the wrong query"
            )

    started = now or datetime.now(UTC)
    outcomes: list[QueryOutcome] = []
    ndcgs: list[float] = []
    recalls: list[float] = []
    mrrs: list[float] = []
    labelled = 0

    for query in queries:
        found = list(search(query.query, RUN_DEPTH))
        head = found[:BASELINE_DEPTH]
        frozen = baseline[query.id]
        grades = labels.get(query.id, {})
        q_ndcg = q_recall = q_mrr = None
        try:
            q_ndcg = ndcg_at_k(found, grades, k=10)
            q_recall = recall_at_k(found, grades, k=50)
            q_mrr = mrr_at_k(found, grades, k=10)
        except NoLabelsError:
            pass
        else:
            labelled += 1
            ndcgs.append(q_ndcg)
            recalls.append(q_recall)
            mrrs.append(q_mrr)
        outcomes.append(
            QueryOutcome(
                query_id=query.id,
                query=query.query,
                kind=query.kind,
                result_ids=found,
                baseline_ids=frozen,
                drift=drift(frozen, head),
                overlap_at_10=overlap_at_k(frozen, head),
                top1_changed=top1_changed(frozen, head),
                abstention_violation=query.is_negative and bool(found),
                ndcg_at_10=q_ndcg,
                recall_at_50=q_recall,
                mrr_at_10=q_mrr,
            )
        )

    drifts = [o.drift for o in outcomes]
    mean_drift = sum(drifts) / len(drifts) if drifts else 0.0
    negatives = sum(1 for o in outcomes if o.kind == "negative")
    note = (
        f"{labelled}/{len(outcomes)} queries labelled"
        if labelled
        else f"no labels: 0/{len(outcomes)} queries labelled, so nDCG@10, "
        "Recall@50 and MRR@10 are not computable"
    )
    report = RegressionReport(
        run_id=run_id or uuid.uuid4().hex[:12],
        mode=mode,
        started_at=started,
        label=label,
        outcomes=outcomes,
        mean_drift=mean_drift,
        max_drift=max(drifts, default=0.0),
        drifted_queries=sum(1 for d in drifts if d > DRIFT_EPSILON),
        top1_changes=sum(1 for o in outcomes if o.top1_changed),
        abstention_violations=sum(1 for o in outcomes if o.abstention_violation),
        negative_queries=negatives,
        labelled_queries=labelled,
        labels_note=note,
        ndcg_at_10=(sum(ndcgs) / len(ndcgs)) if ndcgs else None,
        recall_at_50=(sum(recalls) / len(recalls)) if recalls else None,
        mrr_at_10=(sum(mrrs) / len(mrrs)) if mrrs else None,
        suggested_queries=suggest_from_misses(eval_store.search_misses(), queries),
    )
    if persist:
        eval_store.record_run(
            run_id=report.run_id,
            mode=mode,
            started_at=started,
            label=label,
            queries=len(outcomes),
            mean_drift=report.mean_drift,
            max_drift=report.max_drift,
            drifted_queries=report.drifted_queries,
            top1_changes=report.top1_changes,
            abstention_violations=report.abstention_violations,
            labelled_queries=labelled,
            ndcg_at_10=report.ndcg_at_10,
            recall_at_50=report.recall_at_50,
            mrr_at_10=report.mrr_at_10,
            detail={o.query_id: round(o.drift, 6) for o in outcomes},
        )
    return report


def retrieval_regression_command(
    action: str = "run",
    *,
    mode: str = "lexical",
    queries: Sequence[GoldenQuery] | None = None,
    search: SearchFn | None = None,
    eval_store: EvalStore | None = None,
    labels: Mapping[str, Mapping[str, int]] | None = None,
    golden_path=None,
    labels_file=None,
    db_path=None,
    eval_db_path=None,
    drift_threshold: float | None = None,
    label: str | None = None,
    out=None,
    err=None,
) -> int:
    """CLI entry point. Returns a process exit code.

    ``0`` clean, ``1`` a regression that needs no labels to be sure of
    (abstention violation, or mean drift above an explicit threshold), ``2``
    the harness could not run — no baseline, unknown mode, broken golden set.

    ``queries``/``search``/``eval_store`` are ordinary dependency-injection
    parameters, not test hooks: omit them and the production trio is built.
    """
    out = out or sys.stdout
    err = err or sys.stderr
    owns_store = eval_store is None
    owns_search_store = None
    if action not in ("freeze", "run"):
        print(
            f"unknown action {action!r}: expected 'freeze' or 'run'",
            file=err,
        )
        return 2
    try:
        if queries is None:
            queries = load_golden_queries(golden_path)
        if labels is None:
            labels = load_labels(labels_file)
        if eval_store is None:
            eval_store = EvalStore(eval_db_path)
        if search is None:
            from aggregator.core.store import Store

            owns_search_store = Store(db_path=db_path, read_only=True)
            embedder = None
            if mode == "hybrid":
                from aggregator.core.embed import Embedder

                embedder = Embedder()
            search = resolve_search_fn(mode, owns_search_store, embedder)

        if action == "freeze":
            frozen = freeze_baseline(eval_store, queries, search, mode=mode)
            print(
                f"froze {frozen} golden queries as the {mode} baseline "
                f"(top {BASELINE_DEPTH} ids each)",
                file=out,
            )
            return 0

        report = run_regression(
            eval_store, queries, search, mode=mode, labels=labels, label=label
        )
        print(report.to_text(), file=out)
        if report.abstention_violations:
            print(
                f"FAIL: {report.abstention_violations} negative query(ies) stopped "
                "abstaining — this is a regression, no labels needed",
                file=err,
            )
            return 1
        if drift_threshold is not None and report.mean_drift > drift_threshold:
            print(
                f"FAIL: mean drift {report.mean_drift:.3f} exceeds the threshold "
                f"{drift_threshold:.3f}",
                file=err,
            )
            return 1
        return 0
    except (GoldenSetError, EvalStoreError, MissingBaselineError, ValueError) as e:
        print(f"retrieval-regression: {e}", file=err)
        return 2
    finally:
        if owns_store and eval_store is not None:
            eval_store.close()
        if owns_search_store is not None:
            owns_search_store.close()
