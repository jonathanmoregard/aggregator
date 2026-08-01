"""One-shot re-ingest for the v2 migration.

Runs ``store.rebuild_all()`` (drops every table, including the pre-v2
``records`` shape sessions used to occupy) then ingests both sources against
their real endpoints. Intended to be launched detached (systemd-run) —
sessions ingest walks ``~/.claude/projects`` and can take a while.
"""
from __future__ import annotations

import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("aggregator.reingest")


def main() -> int:
    from aggregator.core.store import Store
    from aggregator.sources.base import ObservationRow, QueryAST, SessionRow
    from aggregator.sources.sessions import SessionsSource

    started = time.time()
    store = Store()
    log.info("db path: %s", store.db_path)
    log.info("current schema_version: %s", store.schema_version())

    log.info("rebuild_all(): dropping every table + re-running v2 DDL")
    store.rebuild_all()
    log.info("post-rebuild schema_version: %s", store.schema_version())

    # --- sessions -----------------------------------------------------
    log.info("sessions: walking ~/.claude/projects ...")
    src = SessionsSource()
    errors: list[str] = []
    session_count = 0
    obs_count = 0
    batch: list = []
    BATCH = 5000
    for e in src.iter_entities(errors=errors):
        batch.append(e)
        if isinstance(e, SessionRow):
            session_count += 1
        elif isinstance(e, ObservationRow):
            obs_count += 1
        if len(batch) >= BATCH:
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
