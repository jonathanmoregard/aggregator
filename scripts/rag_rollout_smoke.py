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

    Collapsing "this is not valid FTS5 syntax" into "this query has no
    matches" would put every malformed query into the no-answer bucket and
    inflate exactly the number this whole exercise turns on. They are
    different facts and the caller has to be able to tell them apart.

    ``Store._fts_obs_ids`` returns matches UNRANKED and uncapped, which is
    right for the union it feeds but useless for "the documents this query is
    actually about". Ranking here rather than changing the store keeps the
    production path exactly as shipped.
    """
    try:
        rows = conn.execute(
            f"SELECT {id_col} AS i FROM {table} f {join} "  # noqa: S608
            f"WHERE {table} MATCH ? ORDER BY bm25({table}) LIMIT ?",
            (text, k),
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
                "queries_with_fts_syntax_error": sum(
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
