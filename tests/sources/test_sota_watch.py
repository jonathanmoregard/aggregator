"""Tests for aggregator.sources.sota_watch (Chunk 7).

Records-shaped source over ``~/Repos/sota-watch/proposals/*.md`` (top-level
only — self-generated proposals, no quarantine subdir here).

Covers:

- source name + protocol shape.
- stable_id (``sota-watch:<filename stem>``).
- subject extraction from first ``# `` heading + stem fallback (incl. the
  50-line scan window).
- body ingested verbatim.
- ``since`` filter on file mtime (exclusive boundary).
- unreadable file → errors sink, iteration continues (decode='replace').
- ``AGGREGATOR_SOTA_WATCH_DIR`` env override.
- missing dir yields no records (no crash).
"""
from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from aggregator.sources.base import IngestResult, Record
from aggregator.sources.sota_watch import SotaWatchSource


@pytest.fixture
def proposals_dir(tmp_path, monkeypatch) -> Path:
    """Isolated proposals root, wired in via the env-var override seam."""
    d = tmp_path / "proposals"
    d.mkdir()
    monkeypatch.setenv("AGGREGATOR_SOTA_WATCH_DIR", str(d))
    return d


def _write(
    proposals_dir: Path, name: str, text: str, *, mtime: float | None = None
) -> Path:
    p = proposals_dir / name
    p.write_text(text, encoding="utf-8")
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return p


def _records(
    since: datetime | None = None, errors: list[str] | None = None
) -> list[Record]:
    return list(SotaWatchSource().iter_records(since, errors=errors))


# -- source identity / protocol shape ---------------------------------------


def test_source_name_is_sota_watch(proposals_dir):
    assert SotaWatchSource.name == "sota-watch"


def test_env_var_overrides_proposals_root(proposals_dir):
    _write(proposals_dir, "2026-07-31-tts.md", "# TTS\n\nbody\n")
    recs = _records()
    assert [r.stable_id for r in recs] == ["sota-watch:2026-07-31-tts"]


def test_ingest_returns_ingest_result_counts(proposals_dir):
    _write(proposals_dir, "2026-07-01-a.md", "# One\n")
    _write(proposals_dir, "2026-07-02-b.md", "# Two\n")
    result = SotaWatchSource().ingest(since=None)
    assert isinstance(result, IngestResult)
    assert result.added == 2
    assert result.errors == []


def test_record_shape_documents_fields(proposals_dir):
    shape = SotaWatchSource().record_shape()
    assert "path" in shape


# -- stable_id --------------------------------------------------------------


def test_stable_id_is_sota_watch_prefixed_stem(proposals_dir):
    _write(proposals_dir, "2026-07-31-model-landscape.md", "# Model landscape\n")
    (rec,) = _records()
    assert rec.stable_id == "sota-watch:2026-07-31-model-landscape"
    assert rec.source == "sota-watch"
    assert rec.tags == ["sota-watch"]


# -- subject extraction -----------------------------------------------------


def test_subject_is_first_markdown_heading_stripped(proposals_dir):
    _write(
        proposals_dir,
        "2026-07-31-prompt-injection.md",
        "intro line\n#  Prompt injection state of the art \nmore\n",
    )
    (rec,) = _records()
    assert rec.subject == "Prompt injection state of the art"


def test_subject_falls_back_to_stem_when_no_heading(proposals_dir):
    _write(proposals_dir, "2026-07-31-notes.md", "No heading here, prose only.\n")
    (rec,) = _records()
    assert rec.subject == "2026-07-31-notes"


def test_subject_ignores_heading_beyond_first_50_lines(proposals_dir):
    text = "\n" * 60 + "# Too late heading\n"
    _write(proposals_dir, "2026-07-31-deep.md", text)
    (rec,) = _records()
    assert rec.subject == "2026-07-31-deep"


# -- body verbatim ----------------------------------------------------------


def test_body_is_full_markdown_verbatim(proposals_dir):
    body = "# Header\n\nBullet:\n\n- item\n- item\n\nEnd.\n"
    _write(proposals_dir, "2026-07-31-verbatim.md", body)
    (rec,) = _records()
    assert rec.body == body


def test_extra_carries_source_path(proposals_dir):
    p = _write(proposals_dir, "2026-07-31-path.md", "# P\n")
    (rec,) = _records()
    assert rec.extra == {"path": str(p)}


# -- since filter -----------------------------------------------------------


def test_since_skips_files_with_mtime_at_or_before_since(proposals_dir):
    now = datetime.now(tz=UTC)
    old = now - timedelta(days=30)
    _write(proposals_dir, "2026-06-01-old.md", "# Old\n", mtime=old.timestamp())
    _write(proposals_dir, "2026-08-01-new.md", "# New\n", mtime=now.timestamp())
    cutoff = now - timedelta(days=1)
    recs = _records(since=cutoff)
    assert [r.stable_id for r in recs] == ["sota-watch:2026-08-01-new"]


def test_since_boundary_is_exclusive(proposals_dir):
    """mtime <= since → skip (spec wording), so mtime == since is skipped."""
    cutoff = datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC)
    _write(
        proposals_dir,
        "2026-07-01-exact.md",
        "# Exact\n",
        mtime=cutoff.timestamp(),
    )
    assert _records(since=cutoff) == []


# -- error handling ---------------------------------------------------------


def test_stray_bytes_survive_via_replace(proposals_dir):
    p = proposals_dir / "2026-07-31-stray.md"
    p.write_bytes(b"# Mostly fine\n\xff\xfe stray bytes\n")
    errors: list[str] = []
    (rec,) = _records(errors=errors)
    assert errors == []
    assert rec.subject == "Mostly fine"
    assert "�" in rec.body  # replacement char, not a crash


# -- missing dir ------------------------------------------------------------


def test_missing_proposals_dir_yields_nothing(tmp_path, monkeypatch):
    """A non-existent dir must not crash — glob just yields nothing."""
    missing = tmp_path / "nope"  # never mkdir'd
    monkeypatch.setenv("AGGREGATOR_SOTA_WATCH_DIR", str(missing))
    assert _records() == []


# -- mtime → UTC timestamps -------------------------------------------------


def test_mtime_becomes_utc_created_and_updated_at(proposals_dir):
    stamp = datetime(2026, 7, 15, 8, 30, 0, tzinfo=UTC)
    _write(
        proposals_dir,
        "2026-07-15-stamped.md",
        "# Stamped\n",
        mtime=stamp.timestamp(),
    )
    (rec,) = _records()
    assert rec.created_at == stamp
    assert rec.updated_at == stamp
    assert rec.created_at.tzinfo is not None
    assert rec.created_at.utcoffset().total_seconds() == 0
