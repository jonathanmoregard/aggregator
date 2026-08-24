"""One-shot re-ingest for the v2 migration.

Runs ``store.rebuild_all()`` (drops every table, including the pre-v2
``records`` shape sessions used to occupy) then ingests both sources against
their real endpoints. Intended to be launched detached (systemd-run) —
sessions ingest walks ``~/.claude/projects`` and can take a while.

THE VECTOR INDEX IS NOT PART OF "RE-INGEST", and round 4 found this script was
the counter-example to a rule three rounds had spent their effort on. Rounds
1-3 narrowed who may destroy computed vectors down to a single per-call
argument behind a printed preview and a ``y`` on stdin — and line 32 here
called ``rebuild_all()``, which ran ``_VEC_DROP_ALL`` unconditionally. The rows
this script drops come back from their sources in minutes; the vectors are
recomputed, and on this hardware that is a multi-week operation (see
``docs/embedding-throughput.md``).

So it asks. ``--yes`` is for scripted use and mirrors ``ingest --yes``: it
skips the question, never the report, so a run in a journal still says what it
destroyed.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("aggregator.reingest")


def main(argv: list[str] | None = None) -> int:
    from aggregator.cli import _confirm_force_on_stdin
    from aggregator.core.store import Store
    from aggregator.sources.base import ObservationRow, QueryAST, SessionRow
    from aggregator.sources.sessions import SessionsSource

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the confirmation for deleting the vector index (never the report)",
    )
    args = parser.parse_args(argv)

    started = time.time()
    store = Store()
    log.info("db path: %s", store.db_path)
    log.info("current schema_version: %s", store.schema_version())

    # THE GATE, before anything is dropped. ``rebuild_all`` refuses on its own
    # too — this is the half that can explain the cost and take an answer.
    vectors, rows = store.vector_reindex_preview()
    if vectors:
        print(
            f"reingest_v2 will DELETE {vectors} computed vector(s) and return "
            f"{rows} row(s) to the embed backlog, on top of re-ingesting every "
            f"session.\nThe rows come back in minutes. The vectors do not: "
            f"refilling them is a multi-week operation on this hardware (see "
            f"docs/embedding-throughput.md).\nIf you only meant to re-ingest "
            f"the corpus, stop here — ordinary `aggregator ingest --all` keeps "
            f"the vector index.",
            file=sys.stderr,
        )
        if args.yes:
            print("reingest_v2: --yes given; proceeding without asking.")
        elif not _confirm_force_on_stdin(
            "Type 'y' to delete them and re-ingest: "
        ):
            print(
                "aborted: vector reindex not confirmed. NOTHING WAS DELETED.",
                file=sys.stderr,
            )
            return 1

    log.info("rebuild_all(): dropping every table + re-running v2 DDL")
    store.rebuild_all(allow_vector_reindex=True)
    log.info("post-rebuild schema_version: %s", store.schema_version())

    # --- sessions -----------------------------------------------------
    log.info("sessions: walking ~/.claude/projects ...")
    src = SessionsSource()
    errors: list[str] = []
    session_count = 0
    obs_count = 0
    batch: list = []
    batch_size = 5000
    for e in src.iter_entities(errors=errors):
        batch.append(e)
        if isinstance(e, SessionRow):
            session_count += 1
        elif isinstance(e, ObservationRow):
            obs_count += 1
        if len(batch) >= batch_size:
            store.upsert_entities(batch)
            log.info(
                "batch persisted: sessions_so_far=%d obs_so_far=%d",
                session_count, obs_count,
            )
            batch.clear()
    if batch:
        store.upsert_entities(batch)
    log.info(
        "sessions done: sessions=%d observations=%d errors=%d elapsed=%.1fs",
        session_count, obs_count, len(errors), time.time() - started,
    )
    if errors:
        for e in errors[:10]:
            log.warning("  parse error: %s", e)

    # Post-ingest verification: query some obvious things.
    sess_total = store.count_sessions(QueryAST())
    obs_total = store.count_observations(QueryAST())
    subs = store.count_sessions(QueryAST(agent_id=None))
    log.info(
        "verify: sess_total=%d obs_total=%d subs_via_countkind=%d",
        sess_total, obs_total, subs,
    )
    c = store._c()
    row = c.execute(
        "SELECT kind, COUNT(*) AS n FROM sessions GROUP BY kind"
    ).fetchall()
    for r in row:
        log.info("  kind=%s count=%d", r["kind"], r["n"])

    log.info("re-ingest total elapsed=%.1fs", time.time() - started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
