from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pytest

from aggregator.sources.dropbox import (
    MAX_BODY_CHARS,
    MAX_TEXT_BYTES,
    DropboxRootUnavailableError,
    DropboxSource,
)


def _write(root, rel, content="body text"):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def test_finds_supported_extensions_only(tmp_path):
    _write(tmp_path, "notes/a.md")
    _write(tmp_path, "notes/b.txt")
    _write(tmp_path, "notes/c.markdown")
    _write(tmp_path, "code/d.js")
    _write(tmp_path, "code/e.json")
    src = DropboxSource(root=tmp_path)
    found = {p.name for p in src._iter_candidate_paths()}
    assert found == {"a.md", "b.txt", "c.markdown"}


def test_prunes_node_modules_and_git_and_dotdirs(tmp_path):
    _write(tmp_path, "keep.md")
    _write(tmp_path, "proj/node_modules/pkg/readme.md")
    _write(tmp_path, "proj/.git/COMMIT_EDITMSG.txt")
    _write(tmp_path, ".dropbox.cache/stale.md")
    _write(tmp_path, ".hidden/secret.md")
    src = DropboxSource(root=tmp_path)
    found = {p.name for p in src._iter_candidate_paths()}
    assert found == {"keep.md"}


def test_user_exclude_globs_are_applied(tmp_path):
    _write(tmp_path, "Public/ok.md")
    _write(tmp_path, "Private/secret.md")
    _write(tmp_path, "Health/report.md")
    src = DropboxSource(root=tmp_path, exclude="Private/*:Health")
    found = {p.name for p in src._iter_candidate_paths()}
    assert found == {"ok.md"}


def test_exclude_pattern_matches_whole_subtree(tmp_path):
    _write(tmp_path, "Private/deep/nested/secret.md")
    src = DropboxSource(root=tmp_path, exclude="Private")
    assert list(src._iter_candidate_paths()) == []


def test_exclude_read_from_env(tmp_path, monkeypatch):
    _write(tmp_path, "Private/secret.md")
    _write(tmp_path, "ok.md")
    monkeypatch.setenv("AGGREGATOR_DROPBOX_EXCLUDE", "Private")
    src = DropboxSource(root=tmp_path)
    found = {p.name for p in src._iter_candidate_paths()}
    assert found == {"ok.md"}


def test_root_read_from_env(tmp_path, monkeypatch):
    _write(tmp_path, "ok.md")
    monkeypatch.setenv("AGGREGATOR_DROPBOX_ROOT", str(tmp_path))
    src = DropboxSource()
    assert {p.name for p in src._iter_candidate_paths()} == {"ok.md"}


def test_record_fields(tmp_path):
    _write(tmp_path, "Blogg/post.md", "# My Title\n\nsome prose")
    src = DropboxSource(root=tmp_path)
    (rec,) = list(src.iter_records(None))
    assert rec.stable_id == "dropbox:Blogg/post.md"
    assert rec.source == "dropbox"
    assert rec.subject == "My Title"
    assert rec.body == "# My Title\n\nsome prose"
    assert set(rec.tags) == {"Blogg", "md"}
    assert rec.created_at is None
    assert rec.updated_at is not None
    assert rec.extra["relpath"] == "Blogg/post.md"
    assert rec.extra["ext"] == ".md"
    assert rec.extra["size_bytes"] > 0


def test_subject_falls_back_to_filename_stem(tmp_path):
    _write(tmp_path, "Recept/pannkakor.txt", "no heading here")
    src = DropboxSource(root=tmp_path)
    (rec,) = list(src.iter_records(None))
    assert rec.subject == "pannkakor"


def test_root_level_file_tags_have_extension_only(tmp_path):
    _write(tmp_path, "loose.md", "x")
    src = DropboxSource(root=tmp_path)
    (rec,) = list(src.iter_records(None))
    assert rec.tags == ["md"]


def test_oversized_text_file_skipped(tmp_path):
    _write(tmp_path, "wordlist.txt", "a" * (MAX_TEXT_BYTES + 1))
    _write(tmp_path, "small.txt", "fine")
    src = DropboxSource(root=tmp_path)
    subjects = {r.subject for r in src.iter_records(None)}
    assert subjects == {"small"}


def test_body_truncated_with_flag(tmp_path):
    _write(tmp_path, "long.txt", "b" * (MAX_BODY_CHARS + 500))
    src = DropboxSource(root=tmp_path)
    (rec,) = list(src.iter_records(None))
    assert len(rec.body) == MAX_BODY_CHARS
    assert rec.extra["truncated"] is True


def test_since_filters_on_mtime(tmp_path):
    old = _write(tmp_path, "old.md", "x")
    new = _write(tmp_path, "new.md", "y")
    stale = (datetime.now(UTC) - timedelta(days=30)).timestamp()
    os.utime(old, (stale, stale))
    since = datetime.now(UTC) - timedelta(days=1)
    src = DropboxSource(root=tmp_path)
    assert {r.subject for r in src.iter_records(since)} == {"new"}
    assert new.exists()


def test_corrupt_document_appends_error_and_continues(tmp_path):
    (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.4 truncated garbage")
    _write(tmp_path, "good.md", "fine")
    errors: list[str] = []
    src = DropboxSource(root=tmp_path)
    subjects = {r.subject for r in src.iter_records(None, errors=errors)}
    assert subjects == {"good"}
    assert len(errors) == 1
    assert "broken.pdf" in errors[0]


def test_image_only_pdf_skipped_without_error(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with (tmp_path / "scan.pdf").open("wb") as fh:
        writer.write(fh)
    errors: list[str] = []
    src = DropboxSource(root=tmp_path)
    assert list(src.iter_records(None, errors=errors)) == []
    assert errors == []


def test_missing_root_is_a_hard_failure_not_a_clean_empty_run(tmp_path):
    """An unmounted Dropbox must not be indistinguishable from "nothing changed".

    ``os.walk`` defaults to ``onerror=None``, which swallows the scandir error
    on the root: a laptop whose Dropbox is not mounted at 03:00 yielded 0
    records, ``errors=0`` and exit 0, forever. This source has no
    input-freshness seam either, so there was no second net.
    """
    errors: list[str] = []
    src = DropboxSource(root=tmp_path / "not-mounted")
    with pytest.raises(DropboxRootUnavailableError) as excinfo:
        list(src.iter_records(None, errors=errors))
    assert "not-mounted" in str(excinfo.value)


def test_root_that_is_a_file_is_a_hard_failure_too(tmp_path):
    """Same class of misconfiguration, different errno (NotADirectoryError)."""
    root = tmp_path / "Dropbox"
    root.write_text("not a directory", encoding="utf-8")
    with pytest.raises(DropboxRootUnavailableError):
        list(DropboxSource(root=root).iter_records(None, errors=[]))


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
def test_unreadable_subdirectory_is_reported_and_the_walk_continues(tmp_path):
    """A permission-denied subtree is a per-item error, not a reason to abort.

    The subtree is genuinely not indexed, so it must leave a record; the rest
    of the tree is fine, so partial ingest beats total loss.
    """
    _write(tmp_path, "ok/keep.md", "fine")
    blocked = tmp_path / "blocked"
    _write(tmp_path, "blocked/hidden.md", "unreachable")
    blocked.chmod(0o000)
    errors: list[str] = []
    src = DropboxSource(root=tmp_path)
    try:
        subjects = {r.subject for r in src.iter_records(None, errors=errors)}
    finally:
        blocked.chmod(0o755)
    assert subjects == {"keep"}
    assert len(errors) == 1
    assert "blocked" in errors[0]


def test_ingest_returns_counts(tmp_path):
    _write(tmp_path, "a.md", "x")
    _write(tmp_path, "b.md", "y")
    result = DropboxSource(root=tmp_path).ingest(None)
    assert result.added == 2
    assert result.errors == []
