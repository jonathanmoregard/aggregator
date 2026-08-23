#!/usr/bin/env python3
"""Derive and re-check ``hybrid.VECTOR_FLOOR_MAX_DISTANCE``.

THE NUMBER IN THE DOCSTRING HAS TO BE RE-RUNNABLE. The floor this replaced
carried a derivation ("the minimum lands about 1.7 sd below the window mean")
that no harness produced and nobody could reproduce, and it was backwards: it
modelled the retrieval window as a whole distribution when the window is by
construction the extreme left tail of one. This script is the missing half.

Two subcommands, and the shipped constant needs both:

* ``simulate`` — pure numpy, seeded, instant, no data required. Models the
  window the way the arm actually produces it: the ``k`` smallest distances out
  of a ``N``-document corpus, with ``m`` of the documents relevant. Reports
  ``P(the floor empties the arm)`` as a function of ``m`` — the MONOTONICITY
  property — for the shipped rule and for the z-score rule it replaced.
* ``spot-check`` — real Qwen3-Embedding-0.6B vectors over real cache text.
  Answers the one question simulation cannot: what a distance MEANS in this
  embedding space, i.e. where relevant documents sit and where the background
  sits. Needs a cache with bodies in it (a read-only snapshot is fine) and the
  model already in the local HF cache; it never downloads weights.

WHY BOTH. The floor is an absolute distance, so it needs a scale (spot-check)
AND a decision about how far into a 400k-document corpus's own coincidence tail
that scale has to reach (simulate). Neither alone justifies a constant.

Usage::

    uv run python scripts/vector_floor_calibration.py simulate
    uv run python scripts/vector_floor_calibration.py spot-check --db PATH
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections.abc import Callable

import numpy as np

# The live cache's scale, and the depth the arm is retrieved to. Both are
# load-bearing for the extreme-value question: the closest of N background
# documents gets closer as N grows, and it is the closest ones that the window
# is made of.
CORPUS_N = 400_000
WINDOW_K = 150


# --- corpus shapes ----------------------------------------------------------
#
# Three background shapes with the SAME mean and standard deviation, differing
# only in how heavy the left tail is — which is the only thing that matters
# here, because the window is drawn from that tail. Ordered by how hard they
# are on an absolute floor: uniform (bounded, easiest), gaussian, skewed-tail
# (exponential left tail, hardest).


def _inv_gaussian(u: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    from statistics import NormalDist

    nd = NormalDist()
    return np.array([mu + sigma * nd.inv_cdf(float(x)) for x in u])


def _inv_uniform(u: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    half = sigma * math.sqrt(3.0)
    return mu - half + 2.0 * half * u


def _inv_skewed_tail(u: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Gumbel-min: an exponential left tail, so coincidentally-close background
    documents are far more common than under a normal. The pessimistic case."""
    beta = sigma * math.sqrt(6.0) / math.pi
    loc = mu + beta * 0.5772156649015329
    return loc + beta * np.log(-np.log1p(-u))


SHAPES: dict[str, Callable[[np.ndarray, float, float], np.ndarray]] = {
    "gaussian": _inv_gaussian,
    "uniform": _inv_uniform,
    "skewed-tail": _inv_skewed_tail,
}


def k_smallest_uniforms(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    """The ``k`` smallest of ``n`` iid uniforms, drawn exactly, in O(k).

    Renyi's representation: consecutive uniform order statistics have
    exponential spacings, so the first ``k`` are ``cumsum(Exp(1))`` divided by
    a Gamma-distributed total. Drawing 400k variates per trial and sorting them
    would give the same answer 2000x slower.
    """
    spacings = rng.exponential(size=k)
    partial = np.cumsum(spacings)
    rest = rng.gamma(shape=n + 1 - k)
    return partial / (partial[-1] + rest)


def draw_window(
    rng: np.random.Generator,
    *,
    shape: str,
    bg_mu: float,
    bg_sigma: float,
    n_relevant: int,
    rel_mu: float,
    rel_sigma: float,
    n_corpus: int = CORPUS_N,
    k: int = WINDOW_K,
) -> tuple[np.ndarray, np.ndarray]:
    """One retrieval window: ``(distances, is_relevant)``, ascending distance.

    THE MODEL THE OLD DERIVATION GOT WRONG. The arm returns the ``k`` NEAREST
    of the whole corpus, so the window is an extreme order statistic of
    ``n_corpus`` draws — not a sample of the corpus distribution. Relevant
    documents are drawn from their own, closer distribution and compete for the
    same ``k`` slots.
    """
    u = k_smallest_uniforms(rng, n_corpus - n_relevant, k)
    bg = SHAPES[shape](u, bg_mu, bg_sigma)
    rel = rng.normal(rel_mu, rel_sigma, size=n_relevant)
    dists = np.concatenate([bg, rel])
    labels = np.concatenate(
        [np.zeros(len(bg), dtype=bool), np.ones(len(rel), dtype=bool)]
    )
    order = np.argsort(dists, kind="stable")[:k]
    return dists[order], labels[order]


# --- the two rules ----------------------------------------------------------


def keep_absolute(dists: np.ndarray, max_distance: float) -> np.ndarray:
    """The shipped rule. Imported rather than reimplemented where possible —
    see ``_assert_matches_shipped``."""
    return dists <= max_distance


def keep_zscore(dists: np.ndarray, z_threshold: float) -> np.ndarray:
    """The rule this replaced, kept so the evidence stays reproducible after
    the replacement: keep a candidate whose distance is ``z_threshold``
    standard deviations BELOW the mean of its own window."""
    if len(dists) < 20:
        return np.ones(len(dists), dtype=bool)
    spread = float(np.std(dists))
    if spread == 0.0:
        return np.ones(len(dists), dtype=bool)
    z = -(dists - float(np.mean(dists))) / spread
    return z >= z_threshold


def _assert_matches_shipped(max_distance: float) -> None:
    """The harness must exercise the SHIPPED function, not a lookalike."""
    from aggregator.core.hybrid import vector_floor

    probe = [("a", max_distance - 1e-6), ("b", max_distance + 1e-6)]
    kept = vector_floor(probe, max_distance=max_distance)
    assert kept == ["a"], (
        f"scripts/vector_floor_calibration.py models a rule the shipped "
        f"hybrid.vector_floor no longer implements (kept {kept!r})"
    )


# --- the experiments --------------------------------------------------------


def p_empty_curve(
    rule: Callable[[np.ndarray], np.ndarray],
    *,
    shape: str,
    bg_mu: float,
    bg_sigma: float,
    rel_mu: float,
    rel_sigma: float,
    relevant_counts: tuple[int, ...],
    trials: int,
    seed: int,
) -> dict[int, float]:
    """``P(the arm comes back empty)`` for each number of relevant documents.

    THE PROPERTY THE OLD RULE VIOLATED. More evidence in the corpus must not
    make the arm abstain more often. A rule that reads the window's own spread
    fails this because the relevant documents ARE the spread.
    """
    out: dict[int, float] = {}
    for m in relevant_counts:
        rng = np.random.default_rng(seed + m)
        empties = 0
        for _ in range(trials):
            dists, _labels = draw_window(
                rng,
                shape=shape,
                bg_mu=bg_mu,
                bg_sigma=bg_sigma,
                n_relevant=m,
                rel_mu=rel_mu,
                rel_sigma=rel_sigma,
            )
            if not rule(dists).any():
                empties += 1
        out[m] = empties / trials
    return out


def recall_curve(
    rule: Callable[[np.ndarray], np.ndarray],
    *,
    shape: str,
    bg_mu: float,
    bg_sigma: float,
    rel_mu: float,
    rel_sigma: float,
    relevant_counts: tuple[int, ...],
    trials: int,
    seed: int,
) -> dict[int, float]:
    """Fraction of the relevant documents IN THE WINDOW that survive the rule."""
    out: dict[int, float] = {}
    for m in relevant_counts:
        rng = np.random.default_rng(seed + 1000 + m)
        kept_rel = total_rel = 0
        for _ in range(trials):
            dists, labels = draw_window(
                rng,
                shape=shape,
                bg_mu=bg_mu,
                bg_sigma=bg_sigma,
                n_relevant=m,
                rel_mu=rel_mu,
                rel_sigma=rel_sigma,
            )
            keep = rule(dists)
            kept_rel += int((keep & labels).sum())
            total_rel += int(labels.sum())
        out[m] = (kept_rel / total_rel) if total_rel else float("nan")
    return out


def sweep_table(
    *,
    bg_mu: float,
    bg_sigma: float,
    rel_mu: float,
    rel_sigma: float,
    distances: tuple[float, ...],
    relevant_counts: tuple[int, ...],
    trials: int,
    seed: int,
) -> None:
    """P(a NO-ANSWER arm empties) and relevant recall, against the floor.

    THE TRADE, PRINTED. A smaller floor abstains more often on a query with no
    answer and deletes more of the answer when there is one; the two columns
    move in opposite directions and the constant is a choice between them. The
    three shapes are the same background mean and spread with different left
    tails, because the left tail is the only part of the distribution a floor
    at this depth ever touches — and it is the part 400k documents of sample
    make unmeasurable.
    """
    print("\n=== the trade: P(no-answer arm emptied) vs relevant recall ===")
    counts = "  ".join(f"m={m}" for m in relevant_counts)
    print(f"  {'d<=':>6s}  {'gaussian':>8s} {'uniform':>8s} {'skewed':>8s}   rel kept: {counts}")
    for d in distances:

        def rule(x: np.ndarray, _d: float = d) -> np.ndarray:
            return keep_absolute(x, _d)

        empties = []
        for shape in ("gaussian", "uniform", "skewed-tail"):
            curve = p_empty_curve(
                rule,
                shape=shape,
                bg_mu=bg_mu,
                bg_sigma=bg_sigma,
                rel_mu=rel_mu,
                rel_sigma=rel_sigma,
                relevant_counts=(0,),
                trials=trials,
                seed=seed,
            )
            empties.append(curve[0])
        recall = recall_curve(
            rule,
            shape="gaussian",
            bg_mu=bg_mu,
            bg_sigma=bg_sigma,
            rel_mu=rel_mu,
            rel_sigma=rel_sigma,
            relevant_counts=relevant_counts,
            trials=trials,
            seed=seed,
        )
        cells = "  ".join(f"{recall[m]:.2f}" for m in relevant_counts)
        print(
            f"  {d:6.2f}  {empties[0]:8.2f} {empties[1]:8.2f} {empties[2]:8.2f}"
            f"        {cells}"
        )


def closest_by_chance(
    *, shape: str, bg_mu: float, bg_sigma: float, trials: int, seed: int
) -> tuple[float, float, float]:
    """Where the NEAREST of ``CORPUS_N`` irrelevant documents lands.

    This is the number an absolute floor has to clear to abstain on a corpus
    that holds no answer, and it is the whole reason abstention gets harder as
    a corpus grows. p05/p50/p95 over ``trials`` corpora.
    """
    rng = np.random.default_rng(seed)
    mins = []
    for _ in range(trials):
        u = k_smallest_uniforms(rng, CORPUS_N, 1)
        mins.append(float(SHAPES[shape](u, bg_mu, bg_sigma)[0]))
    a = np.array(mins)
    return float(np.percentile(a, 5)), float(np.median(a)), float(np.percentile(a, 95))


# --- subcommands ------------------------------------------------------------


def cmd_simulate(args: argparse.Namespace) -> int:
    ms = tuple(int(x) for x in args.relevant_counts.split(","))
    print(
        f"corpus N={CORPUS_N}  window k={WINDOW_K}  trials={args.trials}  "
        f"seed={args.seed}"
    )
    print(
        f"background mu={args.bg_mu} sigma={args.bg_sigma}   "
        f"relevant mu={args.rel_mu} sigma={args.rel_sigma}   "
        f"(L2 on unit-normalized vectors; see spot-check)"
    )

    print("\n=== where the closest IRRELEVANT document lands (no answer) ===")
    for shape in SHAPES:
        lo, mid, hi = closest_by_chance(
            shape=shape,
            bg_mu=args.bg_mu,
            bg_sigma=args.bg_sigma,
            trials=args.trials,
            seed=args.seed,
        )
        print(f"  {shape:12s} p05={lo:.3f}  p50={mid:.3f}  p95={hi:.3f}")

    print("\n=== P(arm emptied) vs number of relevant documents ===")
    print("    the SHIPPED absolute floor must be non-increasing across a row")
    header = "  ".join(f"m={m:<4d}" for m in ms)
    print(f"  {'rule / shape':28s} {header}")
    for shape in SHAPES:
        for label, rule in (
            (f"absolute d<={args.max_distance}", lambda d: keep_absolute(d, args.max_distance)),
            (f"z-score z>={args.z}", lambda d: keep_zscore(d, args.z)),
        ):
            curve = p_empty_curve(
                rule,
                shape=shape,
                bg_mu=args.bg_mu,
                bg_sigma=args.bg_sigma,
                rel_mu=args.rel_mu,
                rel_sigma=args.rel_sigma,
                relevant_counts=ms,
                trials=args.trials,
                seed=args.seed,
            )
            cells = "  ".join(f"{curve[m]:<6.2f}" for m in ms)
            monotone = all(
                curve[a] >= curve[b] - 1e-12 for a, b in zip(ms, ms[1:], strict=False)
            )
            flag = "" if monotone else "   <-- RISES WITH EVIDENCE"
            print(f"  {label + ' / ' + shape:28s} {cells}{flag}")

    print("\n=== relevant documents kept (of those in the window) ===")
    for shape in SHAPES:
        for label, rule in (
            (f"absolute d<={args.max_distance}", lambda d: keep_absolute(d, args.max_distance)),
            (f"z-score z>={args.z}", lambda d: keep_zscore(d, args.z)),
        ):
            curve = recall_curve(
                rule,
                shape=shape,
                bg_mu=args.bg_mu,
                bg_sigma=args.bg_sigma,
                rel_mu=args.rel_mu,
                rel_sigma=args.rel_sigma,
                relevant_counts=ms,
                trials=args.trials,
                seed=args.seed,
            )
            cells = "  ".join(f"{curve[m]:<6.2f}" for m in ms)
            print(f"  {label + ' / ' + shape:28s} {cells}")

    sweep_table(
        bg_mu=args.bg_mu,
        bg_sigma=args.bg_sigma,
        rel_mu=args.rel_mu,
        rel_sigma=args.rel_sigma,
        distances=tuple(float(x) for x in args.sweep.split(",")),
        relevant_counts=tuple(m for m in ms if m),
        trials=args.trials,
        seed=args.seed,
    )

    if not args.skip_shipped_check:
        _assert_matches_shipped(args.max_distance)
        print("\nshipped hybrid.vector_floor agrees with the rule modelled here.")
    return 0


# --- spot-check: what a distance means in the real embedding space ----------

_PRESENT = [
    ("agenix", "how do I add a new agenix secret"),
    ("sqlite-vec", "sqlite-vec extension loading and the vec0 virtual table"),
    ("notify-send", "desktop notification on a failed systemd timer"),
]
_ABSENT = [
    "beef wellington braising temperature and resting time",
    "medieval portuguese maritime law and the treaty of tordesillas",
    "cable knit sweater pattern for a raglan sleeve",
]


def _sample_docs(
    con: sqlite3.Connection,
    n_random: int,
    per_term: int,
    seed: int,
    min_len: int = 40,
) -> list[tuple[str, str]]:
    con.execute("SELECT 1")
    docs: list[tuple[str, str]] = []
    seen: set[str] = set()
    rows = con.execute(
        "SELECT obs_id, body FROM observations "
        "WHERE body IS NOT NULL AND LENGTH(body) BETWEEN ? AND 8000 "
        "ORDER BY substr(src_hash || obs_id, ?) LIMIT ?",
        (min_len, 1 + (seed % 8), n_random),
    ).fetchall()
    for r in rows:
        if r["obs_id"] not in seen:
            seen.add(r["obs_id"])
            docs.append((r["obs_id"], r["body"]))
    for term, _q in _PRESENT:
        hits = con.execute(
            "SELECT obs_id, body FROM observations "
            "WHERE body LIKE ? AND LENGTH(body) BETWEEN ? AND 8000 "
            "ORDER BY substr(src_hash || obs_id, ?) LIMIT ?",
            (f"%{term}%", min_len, 1 + (seed % 8), per_term),
        ).fetchall()
        for r in hits:
            if r["obs_id"] not in seen:
                seen.add(r["obs_id"])
                docs.append((r["obs_id"], r["body"]))
    return docs


def cmd_spot_check(args: argparse.Namespace) -> int:
    from aggregator.core.chunk import chunk_body
    from aggregator.core.embed import Embedder

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    docs = _sample_docs(
        con, args.random_docs, args.per_term, args.seed, min_len=args.min_len
    )
    texts = [chunk_body(body)[0][: args.max_chars] for _oid, body in docs]
    print(
        f"{len(texts)} documents, {sum(len(t) for t in texts)} chars total",
        file=sys.stderr,
        flush=True,
    )

    emb = Embedder()
    doc_vecs = np.zeros((len(texts), 768), dtype=np.float32)
    for i, t in enumerate(texts):
        doc_vecs[i] = emb.embed_documents([t])[0]
        print(f"  doc {i + 1}/{len(texts)} ({len(t)} chars)", file=sys.stderr, flush=True)

    out: dict[str, object] = {"n_docs": len(texts), "queries": {}}
    for term, query in _PRESENT:
        qv = emb.embed_query(query)
        d = np.linalg.norm(doc_vecs - qv, axis=1)
        hit = np.array([term.lower() in body.lower() for _oid, body in docs])
        out["queries"][query] = {
            "term": term,
            "relevant": sorted(float(x) for x in d[hit]),
            "background": sorted(float(x) for x in d[~hit]),
        }
    for query in _ABSENT:
        qv = emb.embed_query(query)
        d = np.linalg.norm(doc_vecs - qv, axis=1)
        out["queries"][query] = {
            "term": None,
            "relevant": [],
            "background": sorted(float(x) for x in d),
        }

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1)

    print(f"\nreal Qwen3-Embedding-0.6B L2 distances over {len(texts)} cache documents")
    print(
        f"{'query':44s} {'class':11s} {'n':>4s} {'min':>6s} {'p10':>6s} "
        f"{'p50':>6s} {'p90':>6s} {'mean':>6s} {'sd':>6s}"
    )
    for query, rec in out["queries"].items():
        for cls in ("relevant", "background"):
            vals = rec[cls]
            if not vals:
                continue
            a = np.array(vals)
            print(
                f"{query[:44]:44s} {cls:11s} {len(a):4d} "
                f"{a.min():6.3f} {np.percentile(a, 10):6.3f} {np.median(a):6.3f} "
                f"{np.percentile(a, 90):6.3f} {a.mean():6.3f} {a.std():6.3f}"
            )
    allbg = np.array(
        [v for rec in out["queries"].values() for v in rec["background"]]
    )
    print(
        f"\nbackground pooled: mu={allbg.mean():.4f} sigma={allbg.std():.4f} "
        f"min={allbg.min():.4f}"
    )
    allrel = np.array(
        [v for rec in out["queries"].values() for v in rec["relevant"]]
    )
    if len(allrel):
        print(
            f"term-matching pooled: mu={allrel.mean():.4f} "
            f"sigma={allrel.std():.4f} p90={np.percentile(allrel, 90):.4f}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate", help="seeded Monte Carlo, no data required")
    s.add_argument("--trials", type=int, default=400)
    s.add_argument("--seed", type=int, default=20260821)
    # DEFAULTS ARE THE MEASURED ONES, not placeholders — ``spot-check`` over 228
    # real cache documents, 2026-08-23. Background is the pooled off-domain
    # (unanswerable) case, which is the one abstention has to catch; relevant is
    # the pooled lexical-hit case. Both are printed by that subcommand.
    s.add_argument("--bg-mu", type=float, default=1.33)
    s.add_argument("--bg-sigma", type=float, default=0.034)
    s.add_argument("--rel-mu", type=float, default=1.09)
    s.add_argument("--rel-sigma", type=float, default=0.09)
    s.add_argument("--max-distance", type=float, default=1.00)
    s.add_argument("--z", type=float, default=3.0)
    s.add_argument("--relevant-counts", default="0,1,3,10,30,60")
    s.add_argument(
        "--sweep",
        default="0.90,0.95,1.00,1.05,1.10,1.15,1.20,1.25",
        help="floors to tabulate the abstention/recall trade over",
    )
    s.add_argument("--skip-shipped-check", action="store_true")
    s.set_defaults(func=cmd_simulate)

    c = sub.add_parser("spot-check", help="real embeddings over real cache text")
    c.add_argument("--db", required=True, help="cache.db (a read-only snapshot is fine)")
    c.add_argument("--random-docs", type=int, default=60)
    c.add_argument("--per-term", type=int, default=5)
    c.add_argument(
        "--max-chars",
        type=int,
        default=4000,
        help=(
            "truncate each chunk before embedding. 4000 is the production chunk "
            "size; the embedder runs at ~40 tok/s on this box (~20 s per 4000 "
            "chars), so a smaller value is the only way to sample more documents "
            "in a tractable wall time. Record whatever you used."
        ),
    )
    c.add_argument("--seed", type=int, default=3)
    c.add_argument(
        "--min-len",
        type=int,
        default=40,
        help=(
            "minimum body length to sample. Raise it to check that the distance "
            "scale does not move when the chunks are production-sized; most cache "
            "bodies are only a few hundred characters."
        ),
    )
    c.add_argument("--out", default=None)
    c.set_defaults(func=cmd_spot_check)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
