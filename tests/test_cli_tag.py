"""``aggregator tag``: LLM topic tags for non-chat records, resumably.

THE SHAPE IS THE 2026-08-16 INGEST CONTRACT, like the provenance backfill it
is modeled on:

* **Streaming** — the run snapshots the ids it owes (a few thousand strings)
  and pulls full rows one batch at a time; bodies are never all in memory.
* **History-aware** — ``llm_tags_src_hash`` disagreeing with ``src_hash`` IS
  the watermark, and it lives in the rows it describes. A second run over a
  tagged corpus makes ZERO LLM calls.
* **Chunked** — one committed checkpoint per prompt batch, so a kill costs at
  most one batch and the next run starts where this one stopped.
* **SOTA errors** — a record the model mis-answers (bad JSON, missing id, no
  valid tag) is recorded and skipped, the run continues, and it still exits
  non-zero so a partial pass is never read as a complete one. A transient CLI
  failure (nonzero exit, timeout) retries with bounded backoff, then isolates
  the whole batch as failed.

NO TEST HERE INVOKES THE REAL ``claude`` CLI OR THE NETWORK: the shell-out is
a monkeypatched ``subprocess.run``. Record bodies are UNTRUSTED — the prompt
must say so, and the parser whitelists tag shape rather than trusting output.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

import aggregator.cli as cli
from aggregator.cli import EXIT_COMPLETED_WITH_ERRORS, main
from aggregator.core.dsl import parse
from aggregator.core.store import Store
from aggregator.sources.base import Record

_TS = datetime(2026, 7, 25, tzinfo=UTC)


def _rec(sid: str, subject: str, body: str, source: str = "github") -> Record:
    return Record(
        stable_id=sid,
        source=source,
        subject=subject,
        body=body,
        tags=["src-tag"],
        created_at=_TS,
        updated_at=_TS,
    )


def _store(tmp_path, records) -> Store:
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    if records:
        s.upsert(list(records))
    return s


class FakeClaude:
    """A scripted stand-in for the ``claude`` subprocess.

    ``script`` is a list of outcomes, one per invocation: a ``dict`` is
    JSON-encoded to stdout (rc 0), a ``str`` goes to stdout verbatim (rc 0),
    an ``int`` is a nonzero exit, and an exception instance is raised. When
    the script runs out, the LAST entry repeats. ``"auto"`` answers every id
    in the prompt with three well-formed tags.
    """

    def __init__(self, script=("auto",)):
        self.script = list(script)
        self.calls: list[dict] = []

    def __call__(self, argv, **kwargs):
        prompt = kwargs.get("input", "")
        self.calls.append({"argv": list(argv), "prompt": prompt})
        step = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        if isinstance(step, BaseException):
            raise step
        if step == "auto":
            ids = _ids_in_prompt(prompt)
            step = {i: ["topic-one", "topic-two", "topic-three"] for i in ids}
        if isinstance(step, dict):
            return SimpleNamespace(
                returncode=0, stdout=json.dumps(step), stderr=""
            )
        if isinstance(step, int):
            return SimpleNamespace(returncode=step, stdout="", stderr="boom")
        return SimpleNamespace(returncode=0, stdout=str(step), stderr="")


def _ids_in_prompt(prompt: str) -> list[str]:
    # Record payloads are serialized as JSON with a "stable_id" key.
    return _extract_ids(prompt)


def _extract_ids(prompt: str) -> list[str]:
    ids = []
    for line in prompt.splitlines():
        marker = '"stable_id": "'
        start = 0
        while (i := line.find(marker, start)) != -1:
            j = line.index('"', i + len(marker))
            ids.append(line[i + len(marker) : j])
            start = j
    return ids


@pytest.fixture
def no_sleep(monkeypatch):
    naps: list[float] = []
    monkeypatch.setattr(cli.time, "sleep", naps.append)
    return naps


def _install(monkeypatch, fake: FakeClaude, which=None) -> None:
    """Stub the subprocess seam AND the PATH probe.

    The probe is stubbed so these tests pass on machines where no real
    ``claude`` is installed (CI): the refusal path is exercised explicitly by
    ``test_missing_claude_binary_aborts_loudly_without_retries``, which stubs
    the probe to find nothing.
    """
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setattr(
        cli.shutil, "which", which or (lambda prog: f"/stub/{prog}")
    )


def _row(store: Store, sid: str):
    return store._c().execute(
        "SELECT tags, llm_tags, llm_tags_src_hash, src_hash FROM records "
        "WHERE stable_id = ?",
        (sid,),
    ).fetchone()


# --- happy path -------------------------------------------------------------


def test_happy_path_writes_llm_tags_and_watermark(tmp_path, monkeypatch, capsys):
    s = _store(tmp_path, [_rec("github:a", "subj a", "body a")])
    fake = FakeClaude()
    _install(monkeypatch, fake)
    assert main(["tag"], _store=s) == 0
    row = _row(s, "github:a")
    assert json.loads(row["llm_tags"]) == ["topic-one", "topic-two", "topic-three"]
    assert row["llm_tags_src_hash"] == row["src_hash"]
    assert json.loads(row["tags"]) == ["src-tag"]  # source tags pristine
    # ...and the record is immediately reachable through the union.
    assert [r.stable_id for r in s.query(parse("tag:topic-one"))] == ["github:a"]
    assert "tagged=1" in capsys.readouterr().out


def test_prompt_carries_data_not_instructions_framing(tmp_path, monkeypatch):
    long_body = "beefcafe " * 1000  # > body cap
    s = _store(tmp_path, [_rec("github:p", "subj", long_body)])
    fake = FakeClaude()
    _install(monkeypatch, fake)
    assert main(["tag"], _store=s) == 0
    prompt = fake.calls[0]["prompt"]
    assert "github:p" in prompt
    assert "subj" in prompt
    # Body truncated to the cap — the full 9000 chars must not be shipped.
    payload_start = prompt.index("beefcafe")
    assert prompt.count("beefcafe") <= (cli._TAG_BODY_CAP // len("beefcafe ")) + 1
    assert payload_start > 0
    # Untrusted-data framing: the model is told record text is data to
    # describe, never instructions to follow.
    lowered = prompt.lower()
    assert "data" in lowered
    assert "not instructions" in lowered or "never" in lowered
    # And the invocation went to the claude CLI in print mode.
    argv = fake.calls[0]["argv"]
    assert argv[0] == "claude"
    assert "-p" in argv
    assert "haiku" in argv


def test_invocation_clamps_the_ambient_surface(tmp_path, monkeypatch):
    """The tagger is a pure text transform over untrusted bodies, so the
    invocation must define NO tools (allowlist-of-nothing, not a denylist),
    no MCP servers, no ambient settings — and must persist no transcript:
    tag-run transcripts under ~/.claude/projects would be re-ingested by the
    sessions source, feeding every record body back into the corpus."""
    s = _store(tmp_path, [_rec("github:clamp", "s", "b")])
    fake = FakeClaude()
    _install(monkeypatch, fake)
    assert main(["tag"], _store=s) == 0
    argv = fake.calls[0]["argv"]
    assert argv[argv.index("--tools") + 1] == ""
    assert "--strict-mcp-config" in argv
    assert json.loads(argv[argv.index("--mcp-config") + 1]) == {"mcpServers": {}}
    assert argv[argv.index("--setting-sources") + 1] == ""
    assert "--no-session-persistence" in argv


def test_batches_are_bounded(tmp_path, monkeypatch):
    s = _store(
        tmp_path,
        [_rec(f"github:{i}", f"s{i}", f"b{i}") for i in range(3)],
    )
    fake = FakeClaude()
    _install(monkeypatch, fake)
    assert main(["tag", "--batch-size", "2"], _store=s) == 0
    assert len(fake.calls) == 2  # 2 + 1


# --- the watermark ----------------------------------------------------------


def test_second_run_makes_zero_llm_calls(tmp_path, monkeypatch):
    s = _store(tmp_path, [_rec("github:w", "s", "b")])
    fake = FakeClaude()
    _install(monkeypatch, fake)
    assert main(["tag"], _store=s) == 0
    assert len(fake.calls) == 1
    assert main(["tag"], _store=s) == 0
    assert len(fake.calls) == 1  # watermark honored: nothing re-sent


def test_changed_record_is_retagged(tmp_path, monkeypatch):
    s = _store(tmp_path, [_rec("github:c", "s", "old body")])
    fake = FakeClaude()
    _install(monkeypatch, fake)
    assert main(["tag"], _store=s) == 0
    s.upsert([_rec("github:c", "s", "new body")])
    assert main(["tag"], _store=s) == 0
    assert len(fake.calls) == 2
    row = _row(s, "github:c")
    assert row["llm_tags_src_hash"] == row["src_hash"]


# --- per-record failure isolation -------------------------------------------


def test_malformed_json_isolates_the_batch_and_continues(
    tmp_path, monkeypatch, capsys
):
    s = _store(
        tmp_path,
        [_rec("github:bad", "s", "b"), _rec("github:good", "s2", "b2")],
    )
    # Newest-first ordering is by (updated_at DESC, stable_id): equal dates,
    # so "github:bad" goes first. First batch malformed, second fine.
    fake = FakeClaude(script=["this is not json", "auto"])
    _install(monkeypatch, fake)
    rc = main(["tag", "--batch-size", "1"], _store=s)
    assert rc == EXIT_COMPLETED_WITH_ERRORS
    assert _row(s, "github:bad")["llm_tags"] is None  # watermark NOT advanced
    assert _row(s, "github:good")["llm_tags"] is not None
    err = capsys.readouterr().err
    assert "github:bad" in err


def test_missing_stable_id_fails_that_record_only(tmp_path, monkeypatch, capsys):
    s = _store(
        tmp_path,
        [_rec("github:m1", "s", "b"), _rec("github:m2", "s2", "b2")],
    )
    fake = FakeClaude(script=[{"github:m1": ["only-one-answered", "x-y", "z-w"]}])
    _install(monkeypatch, fake)
    rc = main(["tag"], _store=s)
    assert rc == EXIT_COMPLETED_WITH_ERRORS
    assert json.loads(_row(s, "github:m1")["llm_tags"]) == [
        "only-one-answered",
        "x-y",
        "z-w",
    ]
    assert _row(s, "github:m2")["llm_tags"] is None
    assert "github:m2" in capsys.readouterr().err


def test_tag_shape_is_whitelisted_and_capped(tmp_path, monkeypatch, capsys):
    s = _store(
        tmp_path,
        [_rec("github:t1", "s", "b"), _rec("github:t2", "s2", "b2")],
    )
    many = [f"tag-{i}" for i in range(12)]
    fake = FakeClaude(
        script=[
            {
                # t1: violations discarded, survivors kept, capped at 8.
                "github:t1": ["Bad_Tag!", "UPPER", "ok-tag", *many],
                # t2: nothing valid at all -> per-record failure.
                "github:t2": ["", "-lead", "trail-", "x", "Спутник"],
            }
        ]
    )
    _install(monkeypatch, fake)
    rc = main(["tag"], _store=s)
    assert rc == EXIT_COMPLETED_WITH_ERRORS
    t1 = json.loads(_row(s, "github:t1")["llm_tags"])
    assert t1[0] == "ok-tag"
    assert len(t1) == 8  # capped
    assert all(cli._TAG_SHAPE_RE.match(t) for t in t1)
    assert _row(s, "github:t2")["llm_tags"] is None
    assert "github:t2" in capsys.readouterr().err


def test_non_list_tag_value_fails_that_record(tmp_path, monkeypatch):
    s = _store(tmp_path, [_rec("github:nl", "s", "b")])
    fake = FakeClaude(script=[{"github:nl": "a-string-not-a-list"}])
    _install(monkeypatch, fake)
    assert main(["tag"], _store=s) == EXIT_COMPLETED_WITH_ERRORS
    assert _row(s, "github:nl")["llm_tags"] is None


def test_markdown_fenced_json_is_accepted(tmp_path, monkeypatch):
    """Haiku wraps JSON in fences often enough that refusing them would turn
    a formatting tic into a failed batch; stripping ONE fence pair is not
    'loose parsing', everything inside is still strict json.loads."""
    s = _store(tmp_path, [_rec("github:f", "s", "b")])
    body = json.dumps({"github:f": ["fenced-tag", "two-two", "three-three"]})
    fake = FakeClaude(script=[f"```json\n{body}\n```"])
    _install(monkeypatch, fake)
    assert main(["tag"], _store=s) == 0
    assert json.loads(_row(s, "github:f")["llm_tags"])[0] == "fenced-tag"


# --- transient CLI failure: bounded retries, then loud ----------------------


def test_nonzero_exit_retries_with_backoff_then_fails_loudly(
    tmp_path, monkeypatch, capsys, no_sleep
):
    s = _store(tmp_path, [_rec("github:r", "s", "b")])
    fake = FakeClaude(script=[1])  # always exits 1
    _install(monkeypatch, fake)
    rc = main(["tag"], _store=s)
    assert rc == EXIT_COMPLETED_WITH_ERRORS
    assert len(fake.calls) == 1 + len(cli._TAG_BACKOFF_SECONDS)
    assert no_sleep == list(cli._TAG_BACKOFF_SECONDS)
    assert _row(s, "github:r")["llm_tags"] is None
    err = capsys.readouterr().err
    assert "github:r" in err


def test_timeout_is_transient_and_bounded(tmp_path, monkeypatch, no_sleep):
    s = _store(tmp_path, [_rec("github:to", "s", "b")])
    fake = FakeClaude(script=[subprocess.TimeoutExpired(cmd="claude", timeout=1)])
    _install(monkeypatch, fake)
    assert main(["tag"], _store=s) == EXIT_COMPLETED_WITH_ERRORS
    assert len(fake.calls) == 1 + len(cli._TAG_BACKOFF_SECONDS)


def test_a_failed_batch_does_not_stop_later_batches(
    tmp_path, monkeypatch, no_sleep
):
    s = _store(
        tmp_path,
        [_rec("github:f1", "s", "b"), _rec("github:f2", "s2", "b2")],
    )
    # Every attempt for batch 1 exits 1; batch 2 succeeds.
    failures = [1] * (1 + len(cli._TAG_BACKOFF_SECONDS))
    fake = FakeClaude(script=[*failures, "auto"])
    _install(monkeypatch, fake)
    rc = main(["tag", "--batch-size", "1"], _store=s)
    assert rc == EXIT_COMPLETED_WITH_ERRORS
    assert _row(s, "github:f1")["llm_tags"] is None
    assert _row(s, "github:f2")["llm_tags"] is not None


def test_missing_claude_binary_aborts_loudly_without_retries(
    tmp_path, monkeypatch, capsys
):
    """Not transient: no backoff loop can install a binary. The run refuses
    up front, names the env var that redirects it, and touches nothing."""
    s = _store(tmp_path, [_rec("github:nb", "s", "b")])
    fake = FakeClaude(script=[FileNotFoundError("claude")])
    _install(monkeypatch, fake, which=lambda prog: None)
    monkeypatch.setenv(cli.CLAUDE_COMMAND_ENV_VAR, "/nonexistent/claude")
    rc = main(["tag"], _store=s)
    assert rc not in (0, None)
    assert len(fake.calls) == 0  # refused before any invocation
    err = capsys.readouterr().err
    assert cli.CLAUDE_COMMAND_ENV_VAR in err or "claude" in err
    assert _row(s, "github:nb")["llm_tags"] is None


# --- resumability -----------------------------------------------------------


def test_a_new_run_does_not_redo_committed_batches(
    tmp_path, monkeypatch, no_sleep
):
    s = _store(
        tmp_path,
        [_rec("github:d1", "s", "b"), _rec("github:d2", "s2", "b2")],
    )
    # Run 1: first batch commits, second batch dies (all retries fail).
    fake1 = FakeClaude(script=["auto", 1])
    _install(monkeypatch, fake1)
    assert (
        main(["tag", "--batch-size", "1"], _store=s) == EXIT_COMPLETED_WITH_ERRORS
    )
    tagged_first = [
        sid for sid in ("github:d1", "github:d2") if _row(s, sid)["llm_tags"]
    ]
    assert len(tagged_first) == 1
    # Run 2: only the failed record is owed; the committed batch is not redone.
    fake2 = FakeClaude()
    _install(monkeypatch, fake2)
    assert main(["tag", "--batch-size", "1"], _store=s) == 0
    assert len(fake2.calls) == 1
    assert _extract_ids(fake2.calls[0]["prompt"]) == [
        sid for sid in ("github:d1", "github:d2") if sid not in tagged_first
    ]


# --- scope ------------------------------------------------------------------


def test_only_the_named_record_sources_are_walked(tmp_path, monkeypatch):
    s = _store(tmp_path, [_rec("research:s", "s", "b", source="research")])
    fake = FakeClaude()
    _install(monkeypatch, fake)
    assert main(["tag", "--sources", "github"], _store=s) == 0
    assert fake.calls == []  # research excluded by the filter
    assert main(["tag", "--sources", "research"], _store=s) == 0
    assert len(fake.calls) == 1


def test_unknown_source_is_a_usage_error(tmp_path, monkeypatch, capsys):
    s = _store(tmp_path, [])
    fake = FakeClaude()
    _install(monkeypatch, fake)
    rc = main(["tag", "--sources", "sessions"], _store=s)
    assert rc == 2
    assert fake.calls == []
    assert "sessions" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["", ",", " , "])
def test_empty_sources_is_a_usage_error_not_a_traceback(
    tmp_path, monkeypatch, capsys, value
):
    """``--sources ''`` (or ',') used to slip past the unknown-source check —
    an empty tuple has no unknown members — and then rendered ``source IN ()``
    in SQLite: a syntax-error traceback instead of a usage message."""
    s = _store(tmp_path, [_rec("github:e", "s", "b")])
    fake = FakeClaude()
    _install(monkeypatch, fake)
    rc = main(["tag", "--sources", value], _store=s)
    assert rc == 2
    assert fake.calls == []
    err = capsys.readouterr().err
    assert "--sources" in err


# --- the cross-batch circuit breaker ----------------------------------------


def test_consecutive_dead_batches_trip_the_breaker(
    tmp_path, monkeypatch, capsys, no_sleep
):
    """A dead CLI (revoked auth, uninstalled binary mid-run) fails EVERY
    batch identically. Per-batch isolation would grind through the whole
    backlog at ~minutes per batch for days; three consecutive invocation
    failures instead abort the run loudly."""
    s = _store(
        tmp_path,
        [_rec(f"github:cb{i}", f"s{i}", f"b{i}") for i in range(5)],
    )
    fake = FakeClaude(script=[1])  # every invocation exits 1, forever
    _install(monkeypatch, fake)
    rc = main(["tag", "--batch-size", "1"], _store=s)
    assert rc == 1  # aborted, not "completed with errors"
    attempts_per_batch = 1 + len(cli._TAG_BACKOFF_SECONDS)
    # Batches 4 and 5 were never attempted.
    assert len(fake.calls) == cli._TAG_BREAKER_BATCHES * attempts_per_batch
    out, err = capsys.readouterr()
    assert "consecutive" in err
    assert "tagged=0" in out  # counts still reported


def test_a_successful_batch_resets_the_breaker(tmp_path, monkeypatch, no_sleep):
    s = _store(
        tmp_path,
        [_rec(f"github:rs{i}", f"s{i}", f"b{i}") for i in range(5)],
    )
    dead_batch = [1] * (1 + len(cli._TAG_BACKOFF_SECONDS))
    # fail, success, fail, fail, success — never 3 consecutive.
    fake = FakeClaude(
        script=[*dead_batch, "auto", *dead_batch, *dead_batch, "auto"]
    )
    _install(monkeypatch, fake)
    rc = main(["tag", "--batch-size", "1"], _store=s)
    assert rc == EXIT_COMPLETED_WITH_ERRORS  # ran to completion
    assert len(fake.calls) == 3 * len(dead_batch) + 2


def test_parse_failures_do_not_count_toward_the_breaker(
    tmp_path, monkeypatch, no_sleep
):
    """A record the model mis-answers is a DATA problem — isolation is right
    for it. Only invocation failures (the CLI itself dying) feed the breaker."""
    s = _store(
        tmp_path,
        [_rec(f"github:pf{i}", f"s{i}", f"b{i}") for i in range(4)],
    )
    fake = FakeClaude(script=["this is not json"])
    _install(monkeypatch, fake)
    rc = main(["tag", "--batch-size", "1"], _store=s)
    assert rc == EXIT_COMPLETED_WITH_ERRORS
    assert len(fake.calls) == 4  # every batch was still attempted
