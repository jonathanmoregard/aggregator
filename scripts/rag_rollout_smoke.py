"""Task M rollout gate — real-cache smoke for the v5 hybrid retrieval path.

INERT BY DEFAULT. Every subcommand requires an explicit ``--scratch-root``
and every one of them runs :func:`assert_not_live` first, which refuses any
path that resolves inside the live aggregator data directory. There is no
flag that turns that refusal off. This file is NOT a pytest test and must
never become one: it needs a multi-GB copy of a real cache to say anything,
and a gate that cannot run on a fresh checkout is not a gate.

Why a script in the repo rather than a transcript of ad-hoc commands: the
numbers it produces (migration duration, embed throughput, the cosine
distance distribution behind the KNN floor) are rollout facts with a shelf
life. When the corpus doubles or the embedding model changes, the decision
has to be re-derived, not remembered.

Usage, in order::

    R=/some/scratch/root
    python scripts/rag_rollout_smoke.py snapshot  --scratch-root $R
    python scripts/rag_rollout_smoke.py migrate   --scratch-root $R
    python scripts/rag_rollout_smoke.py sample    --scratch-root $R --obs 8000
    python scripts/rag_rollout_smoke.py embed     --scratch-root $R
    python scripts/rag_rollout_smoke.py measure   --scratch-root $R
    python scripts/rag_rollout_smoke.py distances --scratch-root $R

``snapshot`` is the only step that touches the live file, and it opens it
``mode=ro`` and copies via the SQLite backup API — a read-only source handle
and a WAL-consistent snapshot out, which is exactly what ``sqlite3 -readonly
'.backup'`` does (same C entry points, ``sqlite3_open_v2(SQLITE_OPEN_READONLY)``
plus ``sqlite3_backup_step``).
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# The one path this script exists to protect. Resolved, not compared as text,
# so a symlink or a ``..`` cannot walk around it.
LIVE_DATA_DIR = Path.home() / ".local" / "share" / "aggregator"
LIVE_DB = LIVE_DATA_DIR / "cache.db"


class LiveDatabaseRefusalError(RuntimeError):
    """Raised when an operation would have touched the live cache."""


def assert_not_live(path: Path, what: str) -> Path:
    """Refuse ``path`` if it is, or is inside, the live aggregator data dir.

    Checks the RESOLVED path against the RESOLVED live directory, so a
    symlinked scratch root pointing back at production is caught. Also
    refuses the live file by inode when it exists, which catches a hard
    link — a case string comparison silently passes.
    """
    resolved = path.expanduser().resolve()
    live_dir = LIVE_DATA_DIR.expanduser().resolve()
    if resolved == live_dir or live_dir in resolved.parents:
        raise LiveDatabaseRefusalError(
            f"refusing to use {resolved} as {what}: it is inside the live "
            f"aggregator data directory {live_dir}. This script only ever "
            f"operates on a copy."
        )
    if LIVE_DB.exists() and resolved.exists():
        live_stat = LIVE_DB.stat()
        here = resolved.stat()
        if (live_stat.st_dev, live_stat.st_ino) == (here.st_dev, here.st_ino):
            raise LiveDatabaseRefusalError(
                f"refusing to use {resolved} as {what}: it is the same inode "
                f"as the live cache {LIVE_DB} (hard link)."
            )
    return resolved


def scratch_db(scratch_root: Path) -> Path:
    """``$scratch_root/aggregator/cache.db`` — the XDG layout ``Store`` expects.

    Returned so that pointing ``XDG_DATA_HOME`` at ``scratch_root`` and
    constructing a bare ``Store()`` resolve to the same file. Both are used:
    the explicit path for our own SQL, the env var for anything that builds
    its own store.
    """
    assert_not_live(scratch_root, "scratch root")
    return scratch_root / "aggregator" / "cache.db"


def bind_scratch_env(scratch_root: Path) -> Path:
    """Point ``XDG_DATA_HOME`` at the scratch root for this process.

    Set unconditionally on every subcommand rather than trusted from the
    caller's environment: a bare ``Store()`` with an unset ``XDG_DATA_HOME``
    resolves to production, and "I exported it in the other shell" is exactly
    how that happens.
    """
    db = scratch_db(scratch_root)
    os.environ["XDG_DATA_HOME"] = str(scratch_root.expanduser().resolve())
    return db


# --------------------------------------------------------------------------
# snapshot
# --------------------------------------------------------------------------


def cmd_snapshot(args: argparse.Namespace) -> int:
    """Read-only, WAL-consistent copy of the live cache into the scratch root."""
    dest = bind_scratch_env(Path(args.scratch_root))
    assert_not_live(dest, "snapshot destination")
    src = Path(args.source).expanduser().resolve()
    if dest.exists() and not args.force:
        print(f"snapshot already present at {dest}; pass --force to overwrite")
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)

    free = os.statvfs(dest.parent).f_bavail * os.statvfs(dest.parent).f_frsize
    need = src.stat().st_size
    if free < need * 2:
        print(
            f"ERROR: {free / 1e9:.1f} GB free at {dest.parent}, need at least "
            f"{need * 2 / 1e9:.1f} GB (snapshot plus room for the vector "
            f"index it will grow). Refusing.",
            file=sys.stderr,
        )
        return 1

    t0 = time.monotonic()
    # mode=ro, NOT immutable=1: the ingest timer writes every 30 minutes, and
    # immutable=1 tells SQLite the file never changes, which would licence it
    # to ignore the WAL and hand back a torn read.
    source = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    try:
        target = sqlite3.connect(str(dest))
        try:
            source.backup(target)
        finally:
            target.close()
    finally:
        source.close()
    dt = time.monotonic() - t0
    print(
        f"snapshot: {src} -> {dest} "
        f"({dest.stat().st_size / 1e9:.2f} GB in {dt:.1f}s)"
    )
    return 0


# --------------------------------------------------------------------------
# migrate
# --------------------------------------------------------------------------

# The v4 surface that must survive the upgrade untouched. Tables the
# incremental-ingest work landed, plus the columns it added.
V4_TABLES = ("ingest_state", "poison_faults", "quarantine", "meta", "sessions")
V4_COLUMNS = {"observations": "src_hash", "records": "src_hash"}


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def _v4_surface(conn: sqlite3.Connection) -> dict[str, Any]:
    """Snapshot of everything the v5 migration must not disturb."""
    out: dict[str, Any] = {"tables": {}, "columns": {}, "watermarks": {}}
    for t in V4_TABLES:
        try:
            out["tables"][t] = conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        except sqlite3.OperationalError:
            out["tables"][t] = None
    for table, col in V4_COLUMNS.items():
        out["columns"][f"{table}.{col}"] = col in _columns(conn, table)
    try:
        out["watermarks"] = {
            r[0]: (r[1], r[2])
            for r in conn.execute(
                "SELECT source, cursor_value, rows_seen FROM ingest_state"
            )
        }
    except sqlite3.OperationalError:
        out["watermarks"] = {}
    return out


def cmd_devolve(args: argparse.Namespace) -> int:
    """Strip the v5 shape off the copy, producing a TRUE v4 cache.

    Needed because this host's live cache is not a clean v4: the abandoned
    2026-08-08 branch numbered the RAG schema 4 as well, so it already carries
    ``embedding_state`` and the ``vec_*`` tables while ``user_version`` says 4.
    Migrating that measures an almost-empty code path. Any OTHER machine —
    and this one, had the collision not happened — pays a real
    ``ALTER TABLE ADD COLUMN`` on a 483k-row table, so that is timed too.

    ``DROP COLUMN`` rewrites the table, which is the expensive direction; the
    cost of getting back to v4 is not itself a rollout number.
    """
    db = bind_scratch_env(Path(args.scratch_root))
    assert_not_live(db, "devolve target")
    conn = sqlite3.connect(str(db))
    # A vec0 virtual table cannot be dropped on a connection where the
    # extension never loaded — the drop needs the module to tear down its
    # shadow tables.
    import sqlite_vec

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    t0 = time.monotonic()
    for table in ("vec_observations", "vec_records"):
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    # The v5 indexes name the column, so they have to go first; SQLite
    # refuses DROP COLUMN while an index references it.
    for idx in ("obs_embedding_state", "rec_embedding_state"):
        conn.execute(f"DROP INDEX IF EXISTS {idx}")
    for table in ("observations", "records"):
        if "embedding_state" in _columns(conn, table):
            conn.execute(f"ALTER TABLE {table} DROP COLUMN embedding_state")
    conn.execute("PRAGMA user_version = 4")
    conn.execute("UPDATE meta SET value = '4' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()
    print(f"devolved to a true v4 cache in {time.monotonic() - t0:.1f}s")
    return 0


def cmd_migrate(args: argparse.Namespace) -> int:
    """Time the v4 -> v5 migration on the copy and prove the v4 surface survived."""
    db = bind_scratch_env(Path(args.scratch_root))
    assert_not_live(db, "migration target")
    if not db.exists():
        print(f"ERROR: no snapshot at {db}; run `snapshot` first", file=sys.stderr)
        return 1

    from aggregator.core.store import SCHEMA_VERSION, Store

    conn = sqlite3.connect(str(db))
    before_version = conn.execute("PRAGMA user_version").fetchone()[0]
    before = _v4_surface(conn)
    before_shape = {
        "observations.embedding_state": "embedding_state"
        in _columns(conn, "observations"),
        "records.embedding_state": "embedding_state" in _columns(conn, "records"),
        "vec_observations": bool(
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name='vec_observations'"
            ).fetchone()
        ),
    }
    conn.close()
    size_before = db.stat().st_size

    t0 = time.monotonic()
    store = Store(db_path=db)
    store.migrate()
    dt = time.monotonic() - t0

    conn = sqlite3.connect(str(db))
    after_version = conn.execute("PRAGMA user_version").fetchone()[0]
    after = _v4_surface(conn)
    null_obs = conn.execute(
        "SELECT count(*) FROM observations WHERE embedding_state IS NOT NULL"
    ).fetchone()[0]
    null_rec = conn.execute(
        "SELECT count(*) FROM records WHERE embedding_state IS NOT NULL"
    ).fetchone()[0]
    conn.close()

    report = {
        "seconds": round(dt, 3),
        "schema_version": [before_version, after_version],
        "expected_version": SCHEMA_VERSION,
        "vector_available": store.vector_available,
        "pre_existing_v5_shape": before_shape,
        "v4_surface_identical": before == after,
        "v4_surface_before": before,
        "v4_surface_after": after,
        "rows_with_non_null_embedding_state": {
            "observations": null_obs,
            "records": null_rec,
        },
        "size_bytes": [size_before, db.stat().st_size],
    }
    print(json.dumps(report, indent=2, default=str))
    ok = (
        after_version == SCHEMA_VERSION
        and before == after
        and null_obs == 0
        and null_rec == 0
    )
    print("MIGRATION OK" if ok else "MIGRATION PROBLEM — see report above")
    return 0 if ok else 1


# --------------------------------------------------------------------------
# sample
# --------------------------------------------------------------------------


def _time_bucket(ts: str | None) -> str:
    """Year-quarter bucket for stratification. ``ts`` is ISO-8601 or NULL."""
    if not ts or len(ts) < 7:
        return "unknown"
    year, month = ts[:4], ts[5:7]
    try:
        q = (int(month) - 1) // 3 + 1
    except ValueError:
        return "unknown"
    return f"{year}Q{q}"


def cmd_sample(args: argparse.Namespace) -> int:
    """Draw a reproducible proportional stratified sample of observations.

    Strata are ``type`` x year-quarter. Proportional allocation with a floor
    of one per stratum, seeded RNG, so the sample mirrors the corpus on both
    the axis that drives CONTENT (an ``attachment`` body and a ``user`` turn
    are different kinds of text) and the axis that drives DRIFT (2024 sessions
    do not talk about the same things as 2026 ones).

    Systematic every-Nth sampling was rejected: session traffic is cyclic
    (user -> assistant -> tool_use -> tool_result), so a fixed stride can lock
    onto a phase of that cycle and silently over-sample one type.

    Records are not sampled at all — all 4.2k of them are embedded, so the
    record-side measurement runs against the COMPLETE corpus with no sampling
    error to defend.
    """
    import random

    db = bind_scratch_env(Path(args.scratch_root))
    assert_not_live(db, "sample target")
    rng = random.Random(args.seed)

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT obs_id, type, ts FROM observations").fetchall()
    strata: dict[tuple[str, str], list[str]] = {}
    for r in rows:
        strata.setdefault((r["type"] or "?", _time_bucket(r["ts"])), []).append(
            r["obs_id"]
        )
    total = len(rows)
    target = min(args.obs, total)

    chosen: list[str] = []
    for _key, ids in sorted(strata.items()):
        share = len(ids) / total
        take = max(1, round(share * target))
        take = min(take, len(ids))
        chosen.extend(rng.sample(ids, take))
    rng.shuffle(chosen)

    out = Path(args.scratch_root) / "sample_obs_ids.json"
    out.write_text(json.dumps(chosen))
    dist: dict[str, int] = {}
    for r in rows:
        dist[r["type"] or "?"] = dist.get(r["type"] or "?", 0) + 1
    picked = set(chosen)
    got: dict[str, int] = {}
    for r in rows:
        if r["obs_id"] in picked:
            got[r["type"] or "?"] = got.get(r["type"] or "?", 0) + 1
    conn.close()
    print(
        json.dumps(
            {
                "population": total,
                "strata": len(strata),
                "sampled": len(chosen),
                "fraction": round(len(chosen) / total, 5),
                "corpus_type_mix": {
                    k: round(v / total, 4) for k, v in sorted(dist.items())
                },
                "sample_type_mix": {
                    k: round(v / len(chosen), 4) for k, v in sorted(got.items())
                },
                "written": str(out),
            },
            indent=2,
        )
    )
    return 0


# --------------------------------------------------------------------------
# embed
# --------------------------------------------------------------------------


def _embed_batch_fast(store, embedder, kind: str, rows: list) -> None:
    """Same vectors, same watermarks as ``cli._embed_batch``, one encode call.

    WRITTEN ON A HYPOTHESIS THAT THE MEASUREMENT THEN KILLED. ``cli._embed_batch``
    calls ``embed_documents`` once per ROW, so the model usually runs at batch
    size 1; that looked like obvious waste. ``bench`` says otherwise on this
    hardware — over an identical 47-chunk set, batch 1 ran at 249.6 chars/s
    (28.7 chars per CPU-second) and batch 32 at 73.3 chars/s (9.0 per
    CPU-second). Batching is **3.4x slower**, and the CPU-second figures show
    it is more work done, not merely worse parallelism: padding a batch to its
    longest member costs more than the per-call overhead it saves, because
    chunk lengths here span two orders of magnitude (p50 98 chars, max 4000).

    Kept, still off by default, for exactly two reasons: it is the evidence
    for that claim, and it lets the comparison be re-run when the hardware or
    the chunker changes. THE SHIPPED PER-ROW LOOP IS THE FAST ONE — do not
    "optimise" it into a batched encode without re-running ``bench`` first.
    """
    from aggregator.core.chunk import chunk_body

    ok_ids: list[str] = []
    skip_ids: list[str] = []
    texts: list[str] = []
    owners: list[tuple[str, int, int]] = []  # row_id, chunk_index, n_chunks
    for row in rows:
        if kind == "observations":
            row_id, body = row["obs_id"], row["body"] or ""
        else:
            row_id = row["stable_id"]
            body = f"{row['subject']}\n\n{row['body']}"
        chunks = chunk_body(body)
        if not chunks:
            skip_ids.append(row_id)
            continue
        for i, ch in enumerate(chunks):
            texts.append(ch)
            owners.append((row_id, i, len(chunks)))
        ok_ids.append(row_id)

    all_vecs: list[tuple[str, Any]] = []
    if texts:
        vecs = embedder.embed_documents(texts)
        for (row_id, i, n), vec in zip(owners, vecs, strict=True):
            all_vecs.append((row_id if n == 1 else f"{row_id}:{i}", vec))
    if kind == "observations":
        store.upsert_vec_observations(all_vecs)
    else:
        store.upsert_vec_records(all_vecs)
    store.mark_embedded(kind, ok_ids, "ok")
    store.mark_embedded(kind, skip_ids, "skip")


def cmd_embed(args: argparse.Namespace) -> int:
    """Embed the sampled observations plus every record, on the real code path.

    Deliberately reuses ``aggregator.cli._embed_batch`` rather than
    reimplementing it, so the throughput number measures what the systemd
    timer will actually do — including the per-row ``embed_documents`` call,
    which is a real cost and would vanish from a hand-rolled batch loop.
    """
    db = bind_scratch_env(Path(args.scratch_root))
    assert_not_live(db, "embed target")

    from aggregator.cli import _embed_batch
    from aggregator.core.embed import Embedder
    from aggregator.core.store import Store

    store = Store(db_path=db)
    if not store.vector_available:
        print("ERROR: sqlite-vec did not load; cannot embed", file=sys.stderr)
        return 1

    sample_path = Path(args.scratch_root) / (args.obs_ids or "sample_obs_ids.json")
    obs_ids = json.loads(sample_path.read_text()) if sample_path.exists() else []
    rec_path = Path(args.scratch_root) / (args.rec_ids or "")
    rec_ids = (
        json.loads(rec_path.read_text())
        if args.rec_ids and rec_path.exists()
        else None
    )

    t_model = time.monotonic()
    embedder = Embedder()
    model_load = time.monotonic() - t_model

    stats: dict[str, Any] = {"model_load_seconds": round(model_load, 2)}
    conn = store._c()

    for kind, ids in (("records", rec_ids), ("observations", obs_ids)):
        if kind not in args.kinds:
            continue
        if kind == "observations" and not ids:
            continue
        t0 = time.monotonic()
        done = chars = 0
        pending = list(ids) if ids is not None else [
            r[0] for r in conn.execute("SELECT stable_id FROM records")
        ]
        for i in range(0, len(pending), args.batch_size):
            batch_ids = pending[i : i + args.batch_size]
            marks = ",".join("?" * len(batch_ids))
            if kind == "observations":
                rows = conn.execute(
                    f"SELECT obs_id, body FROM observations WHERE obs_id IN ({marks})",  # noqa: S608
                    batch_ids,
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT stable_id, subject, body FROM records WHERE stable_id IN ({marks})",  # noqa: S608
                    batch_ids,
                ).fetchall()
            # ``--max-body-chars`` trades fidelity for wall time, and only in
            # ONE direction: truncating a document can only make its best
            # chunk harder to find, so a positive pair measured here is an
            # UPPER bound on the distance production would see. A floor
            # calibrated against upper-bound positives is lenient, never
            # over-aggressive. Long documents are a small minority of rows and
            # a large majority of the embedding bill, so the cap buys most of
            # the wall time back for very little of the corpus.
            if args.max_body_chars:
                rows = [
                    {k: r[k] for k in r.keys()}  # noqa: SIM118 - sqlite3.Row
                    for r in rows
                ]
                for r in rows:
                    if r.get("body"):
                        r["body"] = r["body"][: args.max_body_chars]
            chars += sum(len(r["body"] or "") for r in rows)
            if args.fast_batch:
                _embed_batch_fast(store, embedder, kind, rows)
            else:
                _embed_batch(store, embedder, kind, rows)
            done += len(rows)
            if args.progress and (i // args.batch_size) % 5 == 0:
                el = time.monotonic() - t0
                print(
                    f"  {kind}: {done}/{len(pending)} rows, "
                    f"{done / el:.1f} rows/s, {el:.0f}s elapsed",
                    flush=True,
                )
        el = time.monotonic() - t0
        stats[kind] = {
            "rows": done,
            "seconds": round(el, 1),
            "rows_per_second": round(done / el, 2) if el else None,
            "body_chars": chars,
            "chars_per_second": round(chars / el, 0) if el else None,
            "mean_body_chars": round(chars / done, 1) if done else 0,
        }
        print(json.dumps({kind: stats[kind]}, indent=2), flush=True)

    stats["db_size_bytes"] = db.stat().st_size
    print(json.dumps(stats, indent=2))
    (Path(args.scratch_root) / "embed_stats.json").write_text(json.dumps(stats))
    return 0


# --------------------------------------------------------------------------
# the query set and the evaluation pool
# --------------------------------------------------------------------------

# Filter keys that make a DSL string something other than pure free text.
# A query carrying one of these does not reach the vector arm at all
# (``_vector_arm_engaged`` returns False when ``ast.text`` is empty), so it
# cannot participate in a hybrid-vs-FTS5 comparison.
_DSL_KEYS = (
    "source:",
    "session:",
    "agent:",
    "type:",
    "from:",
    "to:",
    "top:",
    "active:",
    "state:",
)

# Queries whose subject matter is verifiably absent from this corpus. The
# point of the whole distance-floor question: with no floor, a warm index
# answers every one of these with 50 neighbours, and "I have nothing on that"
# stops being an answer the tool can give. Each one is checked against FTS5
# at measure time and dropped from the analysis if the corpus turns out to
# contain it after all — an assumed-absent topic that is present would
# silently invert the result.
NO_ANSWER_QUERIES = [
    "hungarian algorithm assignment problem implementation",
    "scuba diving certification padi open water logbook",
    "sourdough starter hydration bulk fermentation schedule",
    "kubernetes ingress controller nginx annotations tls",
    "motorcycle chain lubrication torque spec service interval",
    "bassoon reed scraping profile gouge cane",
    "mortgage amortisation offset account overpayment",
    "coral reef bleaching symbiodinium thermal tolerance",
    "medieval manuscript illumination gold leaf gesso",
    "curling stone pebble ice sweeping strategy",
]


def _free_text_queries(scratch_root: Path) -> list[str]:
    qs = json.loads((scratch_root / "real_queries.json").read_text())
    return [q for q in qs if not any(k in q for k in _DSL_KEYS)]


def _fts_ranked(
    conn: sqlite3.Connection, table: str, id_col: str, join: str, text: str, k: int
) -> tuple[list[str], str | None]:
    """``(bm25-ranked hits, error)``. THE ERROR IS RETURNED, NOT SWALLOWED.

    Collapsing "the index is broken" into "this query has no matches" would put
    a lock or a corrupt table into the no-answer bucket and inflate exactly the
    number this whole exercise turns on. They are different facts and the
    caller has to be able to tell them apart.

    IT MEASURES THE SHIPPED PATH, WHICH IS THE ONLY THING WORTH MEASURING.
    This bound ``text`` to ``MATCH`` raw, and that was correct until b4eab9b:
    the whole point of the exercise was that ~34% of real queries came back
    ``ok: false`` from an unescaped MATCH. Now every MATCH binding site in
    ``aggregator.core.store`` rewrites through :func:`fts5_match_query` first,
    so a raw bind here would keep reporting a failure rate for a code path that
    no longer exists — and would report it in the same JSON field, under the
    same name, to somebody who had no reason to doubt it. A measurement script
    that has quietly stopped measuring the product is worse than no script.

    THE SHIPPED FUNCTION, NOT A COPY OF IT. Imported inside the call like every
    other ``aggregator`` import in this file — the module-level form would drag
    spaCy in through ``core.scrub`` at import time — and looked up per call, so
    it cannot drift from what the store does.

    An empty rewrite means "no word characters at all", and the store then runs
    NO MATCH. Reproduced here rather than approximated: ``MATCH ''`` is a
    different question, and an unconstrained MATCH returning the whole corpus
    would be the dangerous way to get this wrong in a script whose entire
    output is hit counts.

    ``Store._fts_obs_ids`` returns matches UNRANKED and uncapped, which is
    right for the union it feeds but useless for "the documents this query is
    actually about". Ranking here rather than changing the store keeps the
    production path exactly as shipped.
    """
    from aggregator.core.store import fts5_match_query

    match_expr = fts5_match_query(text)
    if not match_expr:
        return [], None
    try:
        rows = conn.execute(
            f"SELECT {id_col} AS i FROM {table} f {join} "  # noqa: S608
            f"WHERE {table} MATCH ? ORDER BY bm25({table}) LIMIT ?",
            (match_expr, k),
        ).fetchall()
    except sqlite3.OperationalError as e:
        return [], str(e)
    return [r["i"] for r in rows], None


def fts_obs_ranked(
    conn: sqlite3.Connection, text: str, k: int
) -> tuple[list[str], str | None]:
    return _fts_ranked(
        conn,
        "obs_fts",
        "o.obs_id",
        "JOIN observations o ON o.rowid = f.rowid",
        text,
        k,
    )


def fts_rec_ranked(
    conn: sqlite3.Connection, text: str, k: int
) -> tuple[list[str], str | None]:
    return _fts_ranked(conn, "records_fts", "f.stable_id", "", text, k)


def _fts_both(
    conn: sqlite3.Connection, text: str, k: int
) -> dict[str, Any]:
    obs, obs_err = fts_obs_ranked(conn, text, k)
    rec, rec_err = fts_rec_ranked(conn, text, k)
    return {"obs": obs, "rec": rec, "error": obs_err or rec_err}


def cmd_pool(args: argparse.Namespace) -> int:
    """Choose what to embed: query-relevant documents plus a random background.

    Two disjoint groups, because they answer two different questions and
    conflating them would bias both:

    * **relevant** — the BM25 top-K for each real query. These are the
      positives for the distance study: a document that lexically answers the
      query is one the embedder should place NEAR it, and if it does not,
      no floor is safe at any value.
    * **background** — a proportional stratified random draw. These are the
      negatives: a random document is, with overwhelming probability,
      irrelevant to any given query. 88 queries x N background documents
      yields tens of thousands of negative pairs from N embeddings, which is
      why the background can be small and the statistics still large.

    Embedding budget is the binding constraint on this machine (see
    ``bench``), so this deliberately does NOT embed a large uniform sample.
    The distance distribution it measures is a property of the model and the
    query-document semantics, not of how many other documents happen to be in
    the index, so it transfers to the full backfill unchanged.
    """
    import random

    db = bind_scratch_env(Path(args.scratch_root))
    assert_not_live(db, "pool target")
    root = Path(args.scratch_root)
    rng = random.Random(args.seed)
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row

    queries = _free_text_queries(root)
    per_query = {q: _fts_both(conn, q, args.top_k) for q in queries}
    no_answer = {q: _fts_both(conn, q, args.top_k) for q in NO_ANSWER_QUERIES}

    relevant_obs = {i for v in per_query.values() for i in v["obs"]}
    relevant_rec = {i for v in per_query.values() for i in v["rec"]}

    # Background: proportional stratified by type (observations) and by
    # source (records), same reasoning as ``cmd_sample``, and excluding the
    # relevant set so "random" really means random.
    obs_rows = conn.execute(
        "SELECT obs_id, type FROM observations WHERE body IS NOT NULL AND body != ''"
    ).fetchall()
    strata: dict[str, list[str]] = {}
    for r in obs_rows:
        if r["obs_id"] not in relevant_obs:
            strata.setdefault(r["type"] or "?", []).append(r["obs_id"])
    n_obs = sum(len(v) for v in strata.values())
    background_obs: list[str] = []
    for _t, ids in sorted(strata.items()):
        take = min(len(ids), max(1, round(len(ids) / n_obs * args.background_obs)))
        background_obs.extend(rng.sample(ids, take))

    rec_rows = conn.execute(
        "SELECT stable_id, source FROM records WHERE body IS NOT NULL AND body != ''"
    ).fetchall()
    rstrata: dict[str, list[str]] = {}
    for r in rec_rows:
        if r["stable_id"] not in relevant_rec:
            rstrata.setdefault(r["source"], []).append(r["stable_id"])
    n_rec = sum(len(v) for v in rstrata.values())
    background_rec: list[str] = []
    for _s, ids in sorted(rstrata.items()):
        take = min(len(ids), max(1, round(len(ids) / n_rec * args.background_rec)))
        background_rec.extend(rng.sample(ids, take))
    conn.close()

    pool = {
        "queries": queries,
        "per_query": per_query,
        "no_answer": no_answer,
        "background_obs": background_obs,
        "background_rec": background_rec,
    }
    (root / "pool.json").write_text(json.dumps(pool))
    obs_all = sorted(relevant_obs | set(background_obs))
    rec_all = sorted(relevant_rec | set(background_rec))
    (root / "pool_obs_ids.json").write_text(json.dumps(obs_all))
    (root / "pool_rec_ids.json").write_text(json.dumps(rec_all))
    print(
        json.dumps(
            {
                "queries": len(queries),
                "no_answer_queries": len(no_answer),
                "no_answer_with_fts_hits": sum(
                    1 for v in no_answer.values() if v["obs"] or v["rec"]
                ),
                # RENAMED WITH THE THING IT COUNTS. It was
                # ``queries_with_fts_syntax_error``, and after b4eab9b a syntax
                # error is not a thing a user query can cause: every string
                # reaching MATCH is whitelisted first. What survives is a lock,
                # a corrupt index or a missing table, so a non-zero value here
                # is now an infrastructure fault and not a tokenizer story. The
                # old key would have kept reading as "N% of queries are still
                # malformed" while meaning something else entirely.
                "queries_with_fts_index_error": sum(
                    1 for v in per_query.values() if v["error"]
                ),
                "queries_with_zero_fts_no_error": sum(
                    1
                    for v in per_query.values()
                    if not v["obs"] and not v["rec"] and not v["error"]
                ),
                "relevant_obs": len(relevant_obs),
                "relevant_rec": len(relevant_rec),
                "background_obs": len(background_obs),
                "background_rec": len(background_rec),
                "to_embed_obs": len(obs_all),
                "to_embed_rec": len(rec_all),
            },
            indent=2,
        )
    )
    return 0


# --------------------------------------------------------------------------
# census + bench — the two numbers behind the backfill ETA
# --------------------------------------------------------------------------


def cmd_census(args: argparse.Namespace) -> int:
    """Exact full-corpus chunk census. No model, no embedding, ~1 min.

    The point: index size and total embedding work are functions of the CHUNK
    count, and ``chunk_body`` is pure Python, so both can be computed exactly
    for the whole corpus instead of extrapolated from a sample. Only the
    per-chunk COST has to be measured (``bench``); the workload itself is
    counted.
    """
    db = bind_scratch_env(Path(args.scratch_root))
    assert_not_live(db, "census target")
    from aggregator.core.chunk import chunk_body

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    out: dict[str, Any] = {}
    for kind, sql in (
        ("observations", "SELECT body FROM observations"),
        ("records", "SELECT subject, body FROM records"),
    ):
        rows = chunks = chunk_chars = embedded_rows = 0
        for r in conn.execute(sql):
            rows += 1
            body = (
                (r["body"] or "")
                if kind == "observations"
                else f"{r['subject']}\n\n{r['body']}"
            )
            cs = chunk_body(body)
            if cs:
                embedded_rows += 1
                chunks += len(cs)
                chunk_chars += sum(len(c) for c in cs)
        out[kind] = {
            "rows": rows,
            "rows_with_chunks": embedded_rows,
            "rows_skipped_empty": rows - embedded_rows,
            "chunks": chunks,
            "chunk_chars": chunk_chars,
            "approx_tokens": round(chunk_chars / 4),
            "vector_bytes_raw": chunks * 768 * 4,
        }
    conn.close()
    total_chunks = sum(v["chunks"] for v in out.values())
    out["total"] = {
        "chunks": total_chunks,
        "chunk_chars": sum(v["chunk_chars"] for v in out.values()),
        "vector_bytes_raw": total_chunks * 768 * 4,
    }
    print(json.dumps(out, indent=2))
    (Path(args.scratch_root) / "census.json").write_text(json.dumps(out))
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    """Embed throughput at several batch sizes, wall AND cpu seconds.

    CPU seconds matter because this machine is shared: wall-clock throughput
    measured while something else saturates the cores understates the rate by
    however much of the machine we did not get. Reporting both lets the ETA be
    stated honestly for an otherwise-idle host, which is the condition the
    systemd timer actually runs under.

    Sweeping batch size is not tuning for its own sake: ``_embed_batch`` calls
    ``embed_documents`` once per ROW, and most rows are a single chunk, so the
    production path runs the model at batch size 1. Whether that costs
    anything is a rollout fact, not a refactor proposal.
    """
    db = bind_scratch_env(Path(args.scratch_root))
    assert_not_live(db, "bench target")
    from aggregator.core.chunk import chunk_body
    from aggregator.core.embed import Embedder

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    ids = json.loads((Path(args.scratch_root) / "sample_obs_ids.json").read_text())
    ids = ids[: args.rows]
    marks = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT obs_id, body FROM observations WHERE obs_id IN ({marks})",  # noqa: S608
        ids,
    ).fetchall()
    texts: list[str] = []
    for r in rows:
        texts.extend(chunk_body(r["body"] or ""))
    conn.close()
    if not texts:
        print("no non-empty bodies in the sample slice", file=sys.stderr)
        return 1

    embedder = Embedder()
    embedder.embed_documents(texts[:2])  # warm the graph, exclude from timing

    results = []
    for bs in args.batch_sizes:
        w0, c0 = time.monotonic(), time.process_time()
        n_chars = 0
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            embedder.embed_documents(batch)
            n_chars += sum(len(t) for t in batch)
        wall = time.monotonic() - w0
        cpu = time.process_time() - c0
        results.append(
            {
                "batch_size": bs,
                "chunks": len(texts),
                "chars": n_chars,
                "wall_seconds": round(wall, 1),
                "cpu_seconds": round(cpu, 1),
                "chunks_per_wall_second": round(len(texts) / wall, 3),
                "chars_per_wall_second": round(n_chars / wall, 1),
                "chars_per_cpu_second": round(n_chars / cpu, 1),
                "cpu_over_wall": round(cpu / wall, 2),
            }
        )
        print(json.dumps(results[-1]), flush=True)
    (Path(args.scratch_root) / "bench.json").write_text(json.dumps(results))
    return 0


# --------------------------------------------------------------------------
# distances — the evidence the KNN floor decision rests on
# --------------------------------------------------------------------------


def _pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(p / 100 * (len(s) - 1)))))
    return round(s[i], 4)


def _summary(xs: list[float]) -> dict[str, Any]:
    return {
        "n": len(xs),
        "min": round(min(xs), 4) if xs else None,
        "p1": _pct(xs, 1),
        "p5": _pct(xs, 5),
        "p25": _pct(xs, 25),
        "p50": _pct(xs, 50),
        "p75": _pct(xs, 75),
        "p95": _pct(xs, 95),
        "max": round(max(xs), 4) if xs else None,
        "mean": round(sum(xs) / len(xs), 4) if xs else None,
    }


def self_referential_obs(conn: sqlite3.Connection) -> set[str]:
    """Observations that are RECORDS OF AGGREGATOR SEARCHES, not content.

    The query set is mined from the user's own recorded ``search_memory``
    calls, which means every query in it appears verbatim inside an
    observation in the very corpus being searched — the tool_use row that
    logged the call. BM25 ranks that row first for its own query, every
    time, because it is an exact lexical copy.

    Leaving those in would flatter both arms with a match that answers
    nothing: FTS5 looks precise, the vector arm looks like it found the
    topic, and neither told the user anything they did not just type. They
    are excluded from the relevance labels. They are left in the INDEX,
    where they are ordinary background text — removing them there would be
    a different distortion, since production will certainly contain them.

    The invented no-answer queries are unaffected either way: they were
    never run, so no log entry of them exists.
    """
    return {
        r[0]
        for r in conn.execute(
            "SELECT obs_id FROM observations WHERE tool_name LIKE '%aggregator%'"
        )
    }


def store_fts_all(db: Path, text: str) -> list[str]:
    """Every FTS5 hit for ``text`` across both ontologies, uncapped.

    Uncapped on purpose: this is used to decide whether FTS5 can reach a
    document at all, and a top-K view would call a document unreachable
    merely because it ranked 11th.
    """
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        out: list[str] = []
        for sql in (
            "SELECT o.obs_id AS i FROM obs_fts f "
            "JOIN observations o ON o.rowid = f.rowid WHERE obs_fts MATCH ?",
            "SELECT stable_id AS i FROM records_fts WHERE records_fts MATCH ?",
        ):
            out.extend(r["i"] for r in conn.execute(sql, (text,)))
        return out
    finally:
        conn.close()


def _scale_extrapolation(negatives: list[float], full_chunks: int) -> dict[str, Any]:
    """What the NEAREST irrelevant chunk looks like once the index is full.

    THE SMOKE INDEX IS ~2000 CHUNKS AND PRODUCTION IS ~422000, AND THAT GAP
    IS NOT A DETAIL — it is the whole reason a floor measured naively here
    would be wrong. The closest of N random documents gets closer as N grows,
    so a threshold that cleanly rejects every irrelevant neighbour in a small
    index will start admitting them once the backfill finishes. Any floor has
    to be judged against the distance an irrelevant chunk reaches at FULL
    corpus size, not at smoke size.

    Two estimates, deliberately both:

    * **empirical** — negatives pooled across queries give tens of thousands
      of (query, irrelevant-chunk) draws, so the far-left quantiles can just
      be read off. This needs no distributional assumption but runs out of
      resolution around ``1/len(negatives)``.
    * **gaussian** — random pairs in a high-dimensional embedding space
      concentrate, and a normal fit lets the tail be pushed past the
      empirical resolution to ``1/N``. Reported alongside the empirical
      value so the two can be checked against each other where they overlap;
      if they disagree there, distrust the extrapolation.
    """
    import math

    if not negatives:
        return {}
    n = len(negatives)
    mean = sum(negatives) / n
    var = sum((x - mean) ** 2 for x in negatives) / max(1, n - 1)
    sd = math.sqrt(var)

    def gaussian_min(draws: int) -> float:
        # Inverse normal CDF at p = 1/(draws+1), Acklam-style approximation
        # is overkill here; a bisection on math.erf is exact enough.
        p = 1.0 / (draws + 1)
        lo, hi = -12.0, 0.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
                lo = mid
            else:
                hi = mid
        return mean + sd * (lo + hi) / 2

    return {
        "negative_pairs_measured": n,
        "mean": round(mean, 4),
        "sd": round(sd, 4),
        "empirical_left_tail": {
            "p1e-2": _pct(negatives, 1),
            "p1e-3": _pct(negatives, 0.1),
            "p1e-4": _pct(negatives, 0.01),
            "observed_min": round(min(negatives), 4),
        },
        "expected_nearest_irrelevant_at_scale": {
            str(k): round(gaussian_min(k), 4)
            for k in (n, 10_000, 100_000, full_chunks)
        },
        "full_corpus_chunks": full_chunks,
    }


def _load_vectors(db: Path) -> dict[str, dict[str, np.ndarray]]:
    """Every stored vector, keyed by chunk id, per ontology.

    Read wholesale into numpy rather than asked for one KNN at a time: the
    smoke index is a few thousand chunks, and having the raw matrix makes it
    possible to compute the distance from a query to a SPECIFIC document —
    which ``MATCH ... ORDER BY distance`` cannot do, since it only ever
    answers "what is nearest".
    """
    import sqlite_vec

    conn = sqlite3.connect(str(db))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    out: dict[str, dict[str, np.ndarray]] = {}
    for kind, table, idcol in (
        ("observations", "vec_observations", "obs_id"),
        ("records", "vec_records", "stable_id"),
    ):
        out[kind] = {
            r[0]: np.frombuffer(r[1], dtype=np.float32)
            for r in conn.execute(f"SELECT {idcol}, embedding FROM {table}")  # noqa: S608
        }
    conn.close()
    return out


def _base_id(chunk_id: str, known: set[str]) -> str:
    """Map a ``<id>:<n>`` chunk id back to its document id when unambiguous."""
    if chunk_id in known:
        return chunk_id
    base, sep, tail = chunk_id.rpartition(":")
    if sep and base and tail.isdigit() and base in known:
        return base
    return chunk_id


def cmd_distances(args: argparse.Namespace) -> int:
    """Cosine-distance distributions for relevant vs irrelevant pairs.

    THE LABELLING SCHEME, stated plainly because the whole floor decision
    inherits its biases:

    * **positives** = the BM25 top-K documents for a real query. A document
      that lexically answers the query is relevant by construction. This
      under-counts semantic-only matches, so the positive distribution
      measured here is if anything WIDER (worse) than the true one.
    * **negatives** = the background sample, crossed with every query. A
      randomly drawn document is irrelevant to a given query with
      probability ~1; the handful of accidental hits shift the negative
      distribution toward zero, which again is the conservative direction.
    * **no-answer** = ten queries on subjects verified absent from this
      corpus (0 FTS5 hits, checked, not assumed). Their NEAREST neighbour
      distance is the number that matters: it is exactly what a floorless
      KNN would serve as an answer to a question with no answer.

    Distance is cosine (``1 - cos_sim``) on the L2-normalised vectors the
    embedder emits. sqlite-vec's ``vec0`` default metric is L2, and on unit
    vectors ``l2^2 == 2 * cosine``, so the two orderings are identical and a
    threshold in either converts exactly. Both are reported.
    """
    db = bind_scratch_env(Path(args.scratch_root))
    assert_not_live(db, "distance target")
    root = Path(args.scratch_root)
    from aggregator.core.embed import Embedder

    pool = json.loads((root / "pool.json").read_text())
    vecs = _load_vectors(db)
    if not vecs["observations"] and not vecs["records"]:
        print("ERROR: vector index is empty; run `embed` first", file=sys.stderr)
        return 1

    doc_ids = {k: {v for v in vs} for k, vs in vecs.items()}
    by_doc: dict[str, dict[str, list[np.ndarray]]] = {"observations": {}, "records": {}}
    for kind, vs in vecs.items():
        known = doc_ids[kind]
        for cid, vec in vs.items():
            by_doc[kind].setdefault(_base_id(cid, known), []).append(vec)

    embedder = Embedder()
    queries = pool["queries"]
    no_answer = pool["no_answer"]
    qvecs = {q: embedder.embed_query(q) for q in queries}
    qvecs.update({q: embedder.embed_query(q) for q in no_answer})

    bg = {
        "observations": set(pool["background_obs"]),
        "records": set(pool["background_rec"]),
    }

    def dist_to(qv: np.ndarray, kind: str, doc: str) -> float | None:
        chunks = by_doc[kind].get(doc)
        if not chunks:
            return None
        return float(min(1.0 - float(qv @ c) for c in chunks))

    positives: list[float] = []
    negatives: list[float] = []
    per_query_best_neg: list[float] = []
    per_query: dict[str, Any] = {}

    conn = sqlite3.connect(str(db))
    selfref = self_referential_obs(conn)
    conn.close()

    for q in queries:
        qv = qvecs[q]
        pos = [
            d
            for kind in ("observations", "records")
            for doc in pool["per_query"][q]["obs" if kind == "observations" else "rec"][
                : args.top_k
            ]
            if doc not in selfref and (d := dist_to(qv, kind, doc)) is not None
        ]
        neg = [
            d
            for kind in ("observations", "records")
            for doc in bg[kind]
            if (d := dist_to(qv, kind, doc)) is not None
        ]
        positives.extend(pos)
        negatives.extend(neg)
        if neg:
            per_query_best_neg.append(min(neg))
        per_query[q] = {
            "n_pos": len(pos),
            "best_pos": round(min(pos), 4) if pos else None,
            "best_neg": round(min(neg), 4) if neg else None,
            "fts_error": bool(pool["per_query"][q]["error"]),
        }

    # No-answer queries: nearest neighbour over the WHOLE index, which is
    # exactly what a floorless vector arm hands back for a question the
    # corpus cannot answer.
    matrix = np.stack(
        [v for kind in ("observations", "records") for v in vecs[kind].values()]
    )

    def nearest(qv: np.ndarray) -> float:
        return float(1.0 - (matrix @ qv).max())

    no_answer_nearest = {q: round(nearest(qvecs[q]), 4) for q in no_answer}
    real_query_nearest = {q: round(nearest(qvecs[q]), 4) for q in queries}

    # THE UNCONTAMINATED NEGATIVE SAMPLE. The background-vs-real-query pairs
    # above are *mostly* irrelevant, but a random document occasionally does
    # answer a real query, and those accidental positives sit in the left
    # tail — exactly the region the floor decision reads. Pairing the
    # verified-absent queries against every chunk gives negatives that cannot
    # be contaminated that way, because there is nothing in the corpus for
    # them to accidentally match.
    clean_negatives = [
        float(d) for q in no_answer for d in (1.0 - (matrix @ qvecs[q]))
    ]

    # THE DISTRIBUTION THE FLOOR DECISION ACTUALLY TURNS ON.
    #
    # Everything above labels relevance with BM25, so every "positive" is a
    # document FTS5 already returns. Filtering those out of the vector arm
    # costs nothing — the FTS5 arm is uncapped and still carries them. The
    # vector arm's only real contribution is documents FTS5 CANNOT reach, and
    # a floor is only dangerous if it cuts into those.
    #
    # Those documents are found without inventing queries: the user reformulates.
    # The log contains pairs like "context reset" / "context handoff fresh
    # session compaction" — same intent, different words. A document FTS5
    # returns for one and not the other is, for the other, exactly a relevant
    # document with no lexical handle. Their distances are the population a
    # floor would be cutting into, and they are labelled by the user's own
    # behaviour rather than by our guess about what is similar.
    qmat = np.stack([qvecs[q] for q in queries])
    qq = 1.0 - (qmat @ qmat.T)
    vector_only: list[float] = []
    vector_only_pairs = 0
    for i, b in enumerate(queries):
        try:
            fts_b = set(store_fts_all(db, b))
        except sqlite3.OperationalError:
            fts_b = set()
        for j, a in enumerate(queries):
            if i == j or float(qq[i, j]) > args.paraphrase_max:
                continue
            vector_only_pairs += 1
            for kind, key in (("observations", "obs"), ("records", "rec")):
                for doc in pool["per_query"][a][key][: args.top_k]:
                    if doc in fts_b or doc in selfref:
                        continue
                    d = dist_to(qvecs[b], kind, doc)
                    if d is not None:
                        vector_only.append(d)

    report = {
        "index": {
            "obs_chunks": len(vecs["observations"]),
            "rec_chunks": len(vecs["records"]),
            "obs_docs": len(by_doc["observations"]),
            "rec_docs": len(by_doc["records"]),
        },
        "reformulation_pairs_used": vector_only_pairs,
        "self_referential_obs_excluded_from_labels": len(selfref),
        "scale_extrapolation_contaminated": _scale_extrapolation(
            negatives, args.full_corpus_chunks
        ),
        "scale_extrapolation_clean": _scale_extrapolation(
            clean_negatives, args.full_corpus_chunks
        ),
        "cosine_distance": {
            "positives_relevant": _summary(positives),
            "negatives_random": _summary(negatives),
            "per_query_nearest_negative": _summary(per_query_best_neg),
            "vector_only_relevant": _summary(vector_only),
            "no_answer_vs_all_chunks": _summary(clean_negatives),
            "no_answer_nearest_neighbour": _summary(list(no_answer_nearest.values())),
            "real_query_nearest_neighbour": _summary(
                list(real_query_nearest.values())
            ),
        },
        "no_answer_nearest_by_query": no_answer_nearest,
        "real_query_nearest_by_query": real_query_nearest,
        "per_query": per_query,
    }
    (root / "distances.json").write_text(json.dumps(report, indent=2))
    printable = dict(report)
    printable.pop("per_query")
    print(json.dumps(printable, indent=2))
    return 0


# --------------------------------------------------------------------------
# measure — hybrid vs FTS5-only on the real query log
# --------------------------------------------------------------------------


def cmd_measure(args: argparse.Namespace) -> int:
    """Compare the two arms on every real query, through the production code.

    Uses ``mcp._fused_id_scope`` and ``Store._fts_obs_ids`` rather than
    reimplementing fusion, so what is measured is what ships. Emits an
    aggregate report plus ``labelling.md`` — the top vector-only hits per
    query, with snippets, for a human to judge. Automated relevance proxies
    are not trusted here: the reranker is a Qwen3 sibling of the embedder and
    would mostly agree with it for the wrong reasons.
    """
    db = bind_scratch_env(Path(args.scratch_root))
    assert_not_live(db, "measure target")
    root = Path(args.scratch_root)

    from aggregator.core.embed import Embedder
    from aggregator.core.store import Store
    from aggregator.mcp import _fused_id_scope, _widen_chunk_ids

    pool = json.loads((root / "pool.json").read_text())
    store = Store(db_path=db, read_only=True)
    embedder = Embedder()
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    selfref = self_referential_obs(conn)

    def snippet(kind: str, doc: str) -> str:
        if kind == "observations":
            r = conn.execute(
                "SELECT type, body FROM observations WHERE obs_id = ?", (doc,)
            ).fetchone()
            return f"[{r['type']}] {(r['body'] or '')[:200]}" if r else "<missing>"
        r = conn.execute(
            "SELECT source, subject, body FROM records WHERE stable_id = ?", (doc,)
        ).fetchone()
        return (
            f"[{r['source']}] {r['subject']}: {(r['body'] or '')[:200]}"
            if r
            else "<missing>"
        )

    rows: list[dict[str, Any]] = []
    lab: list[str] = ["# Vector-only hits to judge\n"]
    for group, qs in (("real", pool["queries"]), ("no_answer", list(pool["no_answer"]))):
        for q in qs:
            qv = embedder.embed_query(q)
            fts_err = None
            try:
                fts = set(store._fts_obs_ids(q)) | set(store._fts_ids(q))
            except sqlite3.OperationalError as e:
                fts, fts_err = set(), str(e)
            vec_obs = store._vec_obs_ids(qv, 50)
            vec_rec = store._vec_record_ids(qv, 50)
            vec_raw = [("observations", i) for i in vec_obs] + [
                ("records", i) for i in vec_rec
            ]
            vec = set(_widen_chunk_ids(vec_obs + vec_rec))
            scope_o = _fused_id_scope(store, "observations", q, qv) or frozenset()
            scope_r = _fused_id_scope(store, "records", q, qv) or frozenset()
            hybrid = set(scope_o) | set(scope_r)
            extra = vec - fts
            rows.append(
                {
                    "group": group,
                    "query": q,
                    "fts_error": fts_err,
                    "n_fts": len(fts),
                    # FTS5 hit count with the query's own tool_use log removed
                    # — see ``self_referential_obs``. This is the number that
                    # says whether FTS5 answered the QUESTION, rather than
                    # echoing the text the user typed.
                    "n_fts_real": len(fts - selfref),
                    "n_vec": len(vec_raw),
                    "n_hybrid": len(hybrid),
                    "n_vector_only": len(extra),
                    "n_vector_only_real": len(extra - selfref),
                }
            )
            if group == "no_answer" or fts_err or args.label_all:
                lab.append(f"\n## [{group}] {q}")
                lab.append(
                    f"FTS5: {'ERROR ' + fts_err if fts_err else str(len(fts)) + ' hits'}"
                    f" | vector: {len(vec_raw)} hits\n"
                )
                for kind, cid in vec_raw[: args.label_top]:
                    base = cid
                    b, sep, tail = cid.rpartition(":")
                    if sep and tail.isdigit():
                        base = b
                    txt = snippet(kind, base)
                    if txt == "<missing>":
                        txt = snippet(kind, cid)
                    lab.append(f"- `{base}` — {txt}".replace("\n", " ")[:400])
    conn.close()

    def agg(pred) -> dict[str, Any]:
        sel = [r for r in rows if pred(r)]
        return {
            "queries": len(sel),
            "median_fts_hits": _pct([r["n_fts"] for r in sel], 50),
            "median_vector_hits": _pct([r["n_vec"] for r in sel], 50),
            "median_vector_only": _pct([r["n_vector_only"] for r in sel], 50),
            "median_fts_hits_excl_selfref": _pct([r["n_fts_real"] for r in sel], 50),
            "queries_with_zero_fts_excl_selfref": sum(
                1 for r in sel if r["n_fts_real"] == 0
            ),
            "queries_where_vector_added_nothing": sum(
                1 for r in sel if r["n_vector_only"] == 0
            ),
        }

    # ``*_syntax_error`` RENAMED TO ``*_index_error`` throughout, for the same
    # reason as in ``cmd_sample``: the lexical arm rewrites user text through
    # ``fts5_match_query`` before it reaches MATCH, so a syntax error is no
    # longer reachable from a query. A non-zero count here means the index
    # itself is unavailable. The old name is the number this whole script is
    # famous for producing ("~34% of real queries error"), which is exactly why
    # leaving it pointed at a different quantity would be believed.
    report = {
        "real_queries_total": sum(1 for r in rows if r["group"] == "real"),
        "fts5_index_errors": sum(
            1 for r in rows if r["group"] == "real" and r["fts_error"]
        ),
        "self_referential_obs_in_corpus": len(selfref),
        "fts5_zero_hits_no_error": sum(
            1
            for r in rows
            if r["group"] == "real" and not r["fts_error"] and r["n_fts"] == 0
        ),
        "by_bucket": {
            "real_fts_ok": agg(
                lambda r: r["group"] == "real" and not r["fts_error"]
            ),
            "real_fts_index_error": agg(
                lambda r: r["group"] == "real" and r["fts_error"]
            ),
            "no_answer": agg(lambda r: r["group"] == "no_answer"),
        },
        "per_query": rows,
    }
    (root / "measure.json").write_text(json.dumps(report, indent=2))
    (root / "labelling.md").write_text("\n".join(lab))
    printable = {k: v for k, v in report.items() if k != "per_query"}
    print(json.dumps(printable, indent=2))
    return 0


# --------------------------------------------------------------------------
# latency
# --------------------------------------------------------------------------

# vec0's ``distance`` column is plain L2, and the embedder emits unit
# vectors, so cosine distance and L2 are related by ``cos = l2**2 / 2``.
# Verified empirically against numpy on this corpus, not assumed from docs.
def cos_to_l2(cos_distance: float) -> float:
    return float((2.0 * cos_distance) ** 0.5)


def cmd_latency(args: argparse.Namespace) -> int:
    """End-to-end ``aggregator_search_memory`` latency, cold and warm.

    COLD is measured by this process being fresh: the first call pays the
    embedder construction and the first page-ins of the vec table. Run the
    subcommand again for a second cold sample; do not try to "reset" warmth
    in-process, because the thing that makes it warm (an imported torch, a
    loaded model, a hot page cache) cannot be honestly undone from inside.
    """
    db = bind_scratch_env(Path(args.scratch_root))
    assert_not_live(db, "latency target")
    root = Path(args.scratch_root)
    from aggregator import mcp as mcpmod

    pool = json.loads((root / "pool.json").read_text())
    qs = [q for q in pool["queries"] if not pool["per_query"][q]["error"]][
        : args.queries
    ]
    from aggregator.core.store import Store

    store = Store(db_path=db, read_only=True)

    def run(q: str, rerank: bool) -> float:
        t0 = time.monotonic()
        mcpmod.aggregator_query(q, rerank=rerank, _store=store)
        return time.monotonic() - t0

    out: dict[str, Any] = {}
    cold = run(qs[0], False)
    warm = [run(q, False) for q in qs]
    out["cold_first_query_seconds"] = round(cold, 3)
    out["warm_seconds"] = {
        "n": len(warm),
        "p50": _pct(warm, 50),
        "p95": _pct(warm, 95),
        "max": round(max(warm), 3),
    }
    if args.rerank:
        cold_r = run(qs[0], True)
        warm_r = [run(q, True) for q in qs[: args.rerank_queries]]
        out["rerank_cold_seconds"] = round(cold_r, 3)
        out["rerank_warm_seconds"] = {
            "n": len(warm_r),
            "p50": _pct(warm_r, 50),
            "p95": _pct(warm_r, 95),
            "max": round(max(warm_r), 3),
        }
    print(json.dumps(out, indent=2))
    return 0


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rag_rollout_smoke",
        description=(
            "Task M rollout gate. Operates ONLY on a copy of the cache; every "
            "subcommand refuses a path inside the live aggregator data dir."
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--scratch-root",
        required=True,
        help="scratch XDG_DATA_HOME; the copy lives at <root>/aggregator/cache.db",
    )

    s = sub.add_parser("snapshot", parents=[common], help="read-only copy of the live cache")
    s.add_argument("--source", default=str(LIVE_DB))
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_snapshot)

    d = sub.add_parser(
        "devolve", parents=[common], help="strip v5 shape off the copy (true v4)"
    )
    d.set_defaults(func=cmd_devolve)

    m = sub.add_parser("migrate", parents=[common], help="time and verify v4 -> v5")
    m.set_defaults(func=cmd_migrate)

    sa = sub.add_parser("sample", parents=[common], help="stratified observation sample")
    sa.add_argument("--obs", type=int, default=8000)
    sa.add_argument("--seed", type=int, default=20260818)
    sa.set_defaults(func=cmd_sample)

    e = sub.add_parser("embed", parents=[common], help="embed the sample + all records")
    e.add_argument("--batch-size", type=int, default=250)
    e.add_argument("--progress", action="store_true")
    e.add_argument(
        "--kinds",
        default="records,observations",
        type=lambda s: tuple(x.strip() for x in s.split(",")),
    )
    e.add_argument(
        "--fast-batch",
        action="store_true",
        help="one encode call per batch; identical index, measures batching headroom",
    )
    e.add_argument("--obs-ids", help="json id list under the scratch root")
    e.add_argument("--rec-ids", help="json id list under the scratch root")
    e.add_argument(
        "--max-body-chars",
        type=int,
        default=0,
        help="truncate bodies before chunking (0 = production-faithful)",
    )
    e.set_defaults(func=cmd_embed)

    pl = sub.add_parser("pool", parents=[common], help="pick the evaluation pool")
    pl.add_argument("--top-k", type=int, default=8)
    pl.add_argument("--background-obs", type=int, default=500)
    pl.add_argument("--background-rec", type=int, default=150)
    pl.add_argument("--seed", type=int, default=20260818)
    pl.set_defaults(func=cmd_pool)

    c = sub.add_parser("census", parents=[common], help="exact full-corpus chunk count")
    c.set_defaults(func=cmd_census)

    b = sub.add_parser("bench", parents=[common], help="embed throughput vs batch size")
    b.add_argument("--rows", type=int, default=250)
    b.add_argument(
        "--batch-sizes",
        default=(1, 8, 32),
        type=lambda s: tuple(int(x) for x in s.split(",")),
    )
    b.set_defaults(func=cmd_bench)

    di = sub.add_parser(
        "distances", parents=[common], help="cosine distance distributions"
    )
    di.add_argument("--top-k", type=int, default=5)
    di.add_argument("--full-corpus-chunks", type=int, default=422261)
    di.add_argument(
        "--paraphrase-max",
        type=float,
        default=0.35,
        help="max query-query cosine distance to count two queries as the same intent",
    )
    di.set_defaults(func=cmd_distances)

    me = sub.add_parser("measure", parents=[common], help="hybrid vs FTS5-only")
    me.add_argument("--label-top", type=int, default=5)
    me.add_argument("--label-all", action="store_true")
    me.set_defaults(func=cmd_measure)

    la = sub.add_parser("latency", parents=[common], help="query latency cold/warm")
    la.add_argument("--queries", type=int, default=20)
    la.add_argument("--rerank", action="store_true")
    la.add_argument("--rerank-queries", type=int, default=5)
    la.set_defaults(func=cmd_latency)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except LiveDatabaseRefusalError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
