"""Every observation a source emits carries a provenance, or the cursor lies.

``provenance IS NULL`` is the backfill's cursor, so it has to mean exactly one
thing: "nobody has classified this row yet". A source that emits some rows
classified and some not would put freshly-ingested rows into a backlog they do
not belong in, and — worse — would make the backlog's emptiness meaningless.

The sessions source classifies from the RAW JSONL line, which is the only place
the vendor's structural fields still exist: ``_ParsedLine`` has already thrown
the dict away by the time ``ObservationRow`` is built. The two export sources
have no such fields and classify from type and body alone.
"""
from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from aggregator.core.provenance import AGENT, COMMAND, HOOK, HUMAN, SYSTEM
from aggregator.sources.base import ObservationRow
from aggregator.sources.claude_web import ClaudeWebSource
from aggregator.sources.sessions import SessionsSource


def _line(uuid: str, line_type: str = "user", body: str = "hi", **extra) -> dict:
    return {
        "sessionId": "sess-prov",
        "uuid": uuid,
        "parentUuid": None,
        "timestamp": "2026-07-25T10:00:00.000Z",
        "type": line_type,
        "cwd": "/home/u/proj",
        "gitBranch": "main",
        "message": {"role": line_type, "content": [{"type": "text", "text": body}]},
        **extra,
    }


def _write(root: Path, rel: str, lines: list[dict]) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(o) for o in lines), encoding="utf-8")
    old = time.time() - 24 * 60 * 60
    os.utime(p, (old, old))
    return p


def _observations(root: Path) -> dict[str, str | None]:
    src = SessionsSource(projects_root=str(root))
    errors: list[str] = []
    return {
        e.obs_id: e.provenance
        for e in src.iter_entities(errors=errors)
        if isinstance(e, ObservationRow)
    }


def test_a_plain_top_level_turn_is_human(tmp_path):
    _write(tmp_path, "proj/sess-prov.jsonl", [_line("u1", body="fix the thing")])
    assert _observations(tmp_path) == {"u1": HUMAN}


def test_an_sdk_brief_is_a_hook_even_with_no_marker_in_the_body(tmp_path):
    """The largest machine class carries no marker at all — 29% of the sample."""
    _write(
        tmp_path,
        "proj/sess-prov.jsonl",
        [_line("u1", body="You are a NixOS config drift analyzer.", promptSource="sdk")],
    )
    assert _observations(tmp_path) == {"u1": HOOK}


def test_a_resume_banner_is_a_hook_despite_typed_and_human(tmp_path):
    """43 of 43 are ``promptSource='typed'`` AND ``origin.kind='human'``."""
    _write(
        tmp_path,
        "proj/sess-prov.jsonl",
        [
            _line(
                "u1",
                body="Your context was just cleared to make room. Continue.",
                promptSource="typed",
                origin={"kind": "human"},
            )
        ],
    )
    assert _observations(tmp_path) == {"u1": HOOK}


def test_a_slash_command_is_a_command(tmp_path):
    _write(
        tmp_path,
        "proj/sess-prov.jsonl",
        [_line("u1", body="<command-name>/loop</command-name>")],
    )
    assert _observations(tmp_path) == {"u1": COMMAND}


def test_a_meta_line_is_system(tmp_path):
    _write(tmp_path, "proj/sess-prov.jsonl", [_line("u1", body="x", isMeta=True)])
    assert _observations(tmp_path) == {"u1": SYSTEM}


def test_an_assistant_turn_is_agent(tmp_path):
    _write(
        tmp_path,
        "proj/sess-prov.jsonl",
        [_line("u1", body="q"), _line("a1", line_type="assistant", body="a")],
    )
    assert _observations(tmp_path) == {"u1": HUMAN, "a1": AGENT}


def test_a_subagent_streams_user_turns_are_agent_authored(tmp_path):
    """The PARENT AGENT wrote those prompts. ~2,000 rows the census missed."""
    _write(
        tmp_path,
        "proj/sess-prov.jsonl",
        [_line("u1", body="spawn a researcher")],
    )
    _write(
        tmp_path,
        "proj/sess-prov/subagents/agent-a1.jsonl",
        [
            {
                **_line("s1", body="Research the thing and report back"),
                "sessionId": "sub-prov",
                "agentId": "a1",
                "isSidechain": True,
            }
        ],
    )
    assert _observations(tmp_path) == {"u1": HUMAN, "s1": AGENT}


def test_every_emitted_observation_is_classified(tmp_path):
    """No row leaves a source unclassified, whatever its type."""
    _write(
        tmp_path,
        "proj/sess-prov.jsonl",
        [
            _line("u1", body="ask"),
            _line("a1", line_type="assistant", body="answer"),
            _line("sy1", line_type="system", body="notice"),
            _line("at1", line_type="attachment", body="file"),
        ],
    )
    got = _observations(tmp_path)
    assert len(got) == 4
    assert all(v is not None for v in got.values())


# --- vendor chat exports ----------------------------------------------------


def test_claude_web_classifies_every_row(repo_root):
    """0 of 3,621 live ``claude-web`` user rows carry a machine marker, so the
    residual rule puts them where they belong without a structural field."""
    src = ClaudeWebSource(drops_dir=str(Path(repo_root) / "tests" / "fixtures" / "claude-web"))
    errors: list[str] = []
    got = {
        e.obs_id: e.provenance
        for e in src.iter_entities(errors=errors)
        if isinstance(e, ObservationRow)
    }
    assert got, "the export fixture must yield observations"
    assert all(v is not None for v in got.values())
    by_kind = set(got.values())
    assert HUMAN in by_kind and AGENT in by_kind


def test_an_observation_row_defaults_to_unclassified():
    """The dataclass default is NULL, i.e. "the backfill owns this row"."""
    row = ObservationRow(
        obs_id="o1",
        session_id="s1",
        root_session_id="s1",
        parent_obs_id=None,
        type="user",
        ts=datetime(2026, 7, 25, tzinfo=UTC),
        model=None,
        input_tokens=None,
        output_tokens=None,
        tool_name=None,
        tool_use_id=None,
        body="x",
    )
    assert row.provenance is None


@pytest.mark.parametrize("source_module", ["sessions", "claude_web", "chatgpt"])
def test_every_source_classifies_through_the_one_classifier(source_module):
    """No source may grow its own rules — there is one place this is decided.

    Three sources building ``ObservationRow`` means three chances to disagree,
    and a disagreement here is invisible: the rows all look classified. The
    backfill calls the same function, so ingest and backfill cannot drift
    either.
    """
    import importlib
    import inspect

    mod = importlib.import_module(f"aggregator.sources.{source_module}")
    src = inspect.getsource(mod)
    assert "from aggregator.core.provenance import" in src
    assert "classify(" in src
