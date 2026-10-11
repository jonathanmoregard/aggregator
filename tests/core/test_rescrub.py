"""``aggregator rescrub``: the secret layers applied to rows already stored.

Rows are written RAW (straight SQL, bypassing the ingest scrub) to stand in
for rows stored before the scanner knew their format. Tokens are generated at
test time; see test_scrub_gitleaks.py for why there are no literals.
"""
from __future__ import annotations

import secrets
import string

import pytest

from aggregator.cli import main
from aggregator.core.rescrub import rescrub
from aggregator.core.store import _RECORDS_FTS_REFRESH_SQL, Store

_ALNUM = string.ascii_letters + string.digits


def _r(n: int) -> str:
    return "".join(secrets.choice(_ALNUM) for _ in range(n))


@pytest.fixture()
def seeded(tmp_data_home):
    store = Store()
    store.migrate()
    obs_secret = _r(32)
    rec_secret = _r(32)
    token_obs = "tvly" + "-dev-" + obs_secret
    token_rec = "tvly" + "-dev-" + rec_secret
    c = store._c()
    c.execute(
        "INSERT INTO sessions(session_id, root_session_id, kind, first_ts, "
        "last_ts, jsonl_path) VALUES ('s1', 's1', 'session', "
        "'2026-10-01T00:00:00+00:00', '2026-10-01T00:00:00+00:00', '/x.jsonl')"
    )
    for obs_id, body in (("o-leak", f"my key is {token_obs} ok"), ("o-clean", "plain words")):
        c.execute(
            "INSERT INTO observations(obs_id, session_id, root_session_id, type, "
            "ts, body, src_hash, embedding_state) VALUES (?, 's1', 's1', 'user', "
            "'2026-10-01T00:00:00+00:00', ?, 'h-' || ?, 'done')",
            (obs_id, body, obs_id),
        )
    c.execute(
        "INSERT INTO records(stable_id, source, subject, body, tags, src_hash, "
        "embedding_state) VALUES ('r1', 'dropbox', ?, ?, '[]', 'h-r1', 'done')",
        (f"subject {token_rec}", f"body {token_rec}"),
    )
    c.execute(_RECORDS_FTS_REFRESH_SQL, ("r1",))
    c.commit()
    yield store, obs_secret, rec_secret
    store.close()


def _row(store, sql, *args):
    return store._c().execute(sql, args).fetchone()


def test_dry_run_counts_and_writes_nothing(seeded):
    store, obs_secret, _ = seeded
    report = rescrub(store, dry_run=True, chunk_size=1, jobs=2)
    assert report.changed["observations"] == 1
    assert report.changed["records"] == 1
    assert report.by_rule["tavily_key"] == 3
    body = _row(store, "SELECT body FROM observations WHERE obs_id='o-leak'")[0]
    assert obs_secret in body


def test_write_run_redacts_and_keeps_derived_state_consistent(seeded):
    store, obs_secret, rec_secret = seeded
    report = rescrub(store, dry_run=False, chunk_size=1, jobs=2)
    assert report.changed["observations"] == 1

    body, state, src = _row(
        store,
        "SELECT body, embedding_state, src_hash FROM observations WHERE obs_id='o-leak'",
    )
    assert obs_secret not in body and "[REDACTED:tavily_key]" in body
    assert state is None, "a rewritten row must go back into the embed backlog"
    assert src == "h-o-leak", "src_hash describes the source and must not move"
    # Untouched row keeps its vector state.
    assert _row(store, "SELECT embedding_state FROM observations WHERE obs_id='o-clean'")[0] == "done"
    # Keyword index follows the body (observations_au trigger).
    hits = store._c().execute(
        "SELECT COUNT(*) FROM obs_fts WHERE obs_fts MATCH ?", (f'"{obs_secret}"',)
    ).fetchone()[0]
    assert hits == 0

    subject, rbody, rstate = _row(
        store, "SELECT subject, body, embedding_state FROM records WHERE stable_id='r1'"
    )
    assert rec_secret not in subject and rec_secret not in rbody
    assert rstate is None
    fts = store._c().execute(
        "SELECT COUNT(*) FROM records_fts WHERE records_fts MATCH ?", (f'"{rec_secret}"',)
    ).fetchone()[0]
    assert fts == 0
    assert store._c().execute(
        "SELECT COUNT(*) FROM records_fts WHERE stable_id='r1'"
    ).fetchone()[0] == 1

    again = rescrub(store, dry_run=False)
    assert again.changed["observations"] == 0 and again.changed["records"] == 0


def test_cli_prints_counts_never_values(seeded, capsys):
    _store, obs_secret, rec_secret = seeded
    assert main(["rescrub", "--dry-run"]) == 0
    out = capsys.readouterr()
    text = out.out + out.err
    assert "observations: scanned=2 would_change=1" in text
    assert "rule tavily_key: 3" in text
    assert obs_secret not in text and rec_secret not in text


def test_cli_fails_closed_without_the_scanner(seeded, monkeypatch, tmp_path, capsys):
    store, obs_secret, _ = seeded
    monkeypatch.setenv("AGGREGATOR_GITLEAKS", str(tmp_path / "absent"))
    assert main(["rescrub"]) == 3
    body = _row(store, "SELECT body FROM observations WHERE obs_id='o-leak'")[0]
    assert obs_secret in body
