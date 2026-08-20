"""Retrieval evaluation: a frozen golden query set, baselines, and drift.

WHY THIS PACKAGE EXISTS AND WHY IT LANDED BEFORE ANY RETRIEVAL CHANGE. Every
remaining tuning decision on this branch — rerank threshold, Matryoshka
truncation, quantization level, fusion depth, FTS5 tokenizer — is a claim that
retrieval got better. Without a frozen query set and a recorded baseline those
claims are unfalsifiable, and a change that quietly halves recall ships green
because no test in the suite can see ranking quality.

THE HARD PART IS THAT THERE ARE NO RELEVANCE LABELS, and there will not be any
until a human spends an afternoon making them. So the design splits in two:

* **Label-free, works today.** Freeze the top-10 result ids per golden query,
  re-run later, and report DRIFT — how far the ranking moved. A regression is
  visible as drift with nobody labelling anything. Negative queries make
  abstention testable the same way: a query that returned nothing and now
  returns something is an unambiguous regression, no labels required.
* **Labelled, works when labels exist.** nDCG@10 (primary — graded relevance,
  rank-aware), Recall@50 (the rerank ceiling) and MRR@10 (the rerank need).
  Absent labels these report ABSENCE, never 0.0 and never 1.0.

Layout mirrors ``aggregator/core`` and ``aggregator/imports``: small modules,
one concern each.

* ``golden``  — the frozen query set (a data file in the repo, not the cache).
* ``metrics`` — nDCG@10 / Recall@50 / MRR@10, and the drift metric.
* ``db``      — baselines, run history, and the zero-result log.
* ``search``  — the ``(query, limit) -> [ids]`` callables under measurement.
* ``harness`` — freeze, run, report; and the CLI entry point.
"""
