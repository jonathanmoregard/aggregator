from __future__ import annotations

from aggregator.sources.dropbox import DropboxSource


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
