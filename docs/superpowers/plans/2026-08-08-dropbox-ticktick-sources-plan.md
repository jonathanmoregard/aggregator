# Dropbox + TickTick Sources Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (default) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Index the user's Dropbox prose/documents and their TickTick task history (including completed tasks) as searchable records in the aggregator.

**Architecture:** Two records-shaped sources following the `ResearchReportsSource` template in `aggregator/sources/research_reports.py`. `dropbox` walks the locally-synced `~/Dropbox` and extracts text from md/txt/docx/pdf. `ticktick` merges two legs — a CSV backup parser (authoritative, has completed tasks) and an Open API poll (fresh, open tasks only, infers completions by disappearance) — resolving conflicts by observation recency before yielding.

**Tech Stack:** Python 3.11+, `uv`, `pypdf`, `python-docx`, stdlib `urllib.request` and `csv`, pytest.

**Spec:** `docs/superpowers/specs/2026-08-08-dropbox-ticktick-sources-design.md`

**Constraints:** `tasks/session-constraints.md` — read it. Fail loudly; no silent degradation.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `aggregator/core/textextract.py` (create) | Pure text extraction: `.md`/`.txt`/`.markdown` direct, `.docx` via python-docx, `.pdf` via pypdf. Knows nothing about records or Dropbox. |
| `aggregator/sources/dropbox.py` (create) | Walk, filter, size caps, record mapping. |
| `aggregator/sources/ticktick_csv.py` (create) | Detect + parse TickTick backup CSVs into rows, map rows to records. |
| `aggregator/sources/ticktick_api.py` (create) | GET-only Open API client, open-task state file, completion inference. |
| `aggregator/sources/ticktick.py` (create) | Source class: runs both legs, merges by recency, archives CSVs. |
| `aggregator/cli.py` (modify) | Register both sources; exit 3 when a run completes with errors. |
| `pyproject.toml` (modify) | Add `pypdf`, `python-docx`. |

Tests mirror this: `tests/core/test_textextract.py`, `tests/sources/test_dropbox.py`, `tests/sources/test_ticktick_csv.py`, `tests/sources/test_ticktick_api.py`, `tests/sources/test_ticktick.py`, plus additions to `tests/test_cli_ingest.py` (create if absent).

**Gate after every task:** `uv run pytest -q && uv run ruff check .` — both exit 0.

---

## Task 1: Text extraction module

**Files:**
- Modify: `pyproject.toml:6-15`
- Create: `aggregator/core/textextract.py`
- Test: `tests/core/test_textextract.py`

- [ ] **Step 1: Add dependencies**

In `pyproject.toml`, replace the `dependencies` list with:

```toml
dependencies = [
  "fastmcp>=0.4",
  "presidio-analyzer>=2.2",
  "presidio-anonymizer>=2.2",
  # Dropbox source document extraction. Deliberately in the main dependency
  # list rather than an extras group: with extras, a missing install turns
  # ~590 Dropbox files into a silent gap in search results, and a silent gap
  # is the one failure mode this index cannot tolerate.
  "pypdf>=5.0",
  "python-docx>=1.1",
]
```

- [ ] **Step 2: Write the failing tests**

Create `tests/core/test_textextract.py`:

```python
from __future__ import annotations

import pytest

from aggregator.core.textextract import (
    SUPPORTED_EXTS,
    ExtractionError,
    extract_text,
    first_markdown_heading,
)


def test_supported_exts_are_lowercase_with_dots():
    assert SUPPORTED_EXTS == {".md", ".markdown", ".txt", ".docx", ".pdf"}


def test_extract_plain_text(tmp_path):
    p = tmp_path / "note.txt"
    p.write_text("hello world", encoding="utf-8")
    assert extract_text(p) == "hello world"


def test_extract_undecodable_bytes_degrade_not_raise(tmp_path):
    p = tmp_path / "note.txt"
    p.write_bytes(b"caf\xff")
    assert extract_text(p).startswith("caf")


def test_extract_unsupported_extension_raises(tmp_path):
    p = tmp_path / "image.png"
    p.write_bytes(b"\x89PNG")
    with pytest.raises(ExtractionError):
        extract_text(p)


def test_extract_docx(tmp_path):
    docx = pytest.importorskip("docx")
    p = tmp_path / "doc.docx"
    d = docx.Document()
    d.add_paragraph("first para")
    d.add_paragraph("second para")
    d.save(p)
    assert extract_text(p) == "first para\nsecond para"


def test_extract_corrupt_docx_raises_extraction_error(tmp_path):
    p = tmp_path / "broken.docx"
    p.write_bytes(b"not a zip file at all")
    with pytest.raises(ExtractionError):
        extract_text(p)


def test_extract_corrupt_pdf_raises_extraction_error(tmp_path):
    p = tmp_path / "broken.pdf"
    p.write_bytes(b"%PDF-1.4 truncated garbage")
    with pytest.raises(ExtractionError):
        extract_text(p)


def test_extract_pdf_with_no_text_layer_returns_empty(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    p = tmp_path / "scan.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with p.open("wb") as fh:
        writer.write(fh)
    assert extract_text(p).strip() == ""


def test_first_markdown_heading_found():
    assert first_markdown_heading("intro\n# Title\nbody") == "Title"


def test_first_markdown_heading_absent():
    assert first_markdown_heading("no heading here") is None


def test_first_markdown_heading_ignores_late_headings():
    text = "\n".join(["filler"] * 80 + ["# Late"])
    assert first_markdown_heading(text) is None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_textextract.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aggregator.core.textextract'`

- [ ] **Step 4: Implement the module**

Create `aggregator/core/textextract.py`:

```python
"""Text extraction for document-shaped source files.

Deliberately free of any record/source concepts so it can be tested in
isolation and reused. Callers own the policy decisions (size caps, what
counts as "too little text to be worth indexing"); this module only turns
a file into a string or raises.

No OCR. A scanned PDF has no text layer and legitimately yields "" — that
is an outcome, not an error, so it does not raise. Callers distinguish.
"""
from __future__ import annotations

from pathlib import Path

TEXT_EXTS = {".md", ".markdown", ".txt"}
DOCX_EXTS = {".docx"}
PDF_EXTS = {".pdf"}
SUPPORTED_EXTS = TEXT_EXTS | DOCX_EXTS | PDF_EXTS

# Only scan this many leading lines for an ATX heading. Matches the research
# source's window; a document whose first heading is 80 lines down is better
# titled by its filename anyway.
HEADING_SCAN_LINES = 50


class ExtractionError(Exception):
    """Raised when a file of a supported type cannot be parsed at all."""


def first_markdown_heading(text: str) -> str | None:
    """Return the first ATX (``# ``) heading within the leading scan window."""
    for line in text.splitlines()[:HEADING_SCAN_LINES]:
        if line.startswith("# "):
            return line.lstrip("#").strip()
    return None


def _extract_docx(path: Path) -> str:
    import docx  # imported lazily so a broken optional install fails per-file

    try:
        document = docx.Document(str(path))
    except Exception as e:  # python-docx raises a zoo of types on bad input
        raise ExtractionError(f"docx parse failed: {e}") from e
    return "\n".join(p.text for p in document.paragraphs)


def _extract_pdf(path: Path) -> str:
    import pypdf

    try:
        reader = pypdf.PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as e:
        raise ExtractionError(f"pdf parse failed: {e}") from e
    return "\n".join(pages)


def extract_text(path: Path) -> str:
    """Extract plain text from a supported file.

    Raises ExtractionError for unsupported extensions and unparseable files.
    Returns "" for a parseable file that genuinely holds no text.
    """
    ext = path.suffix.lower()
    if ext in TEXT_EXTS:
        return path.read_text(encoding="utf-8", errors="replace")
    if ext in DOCX_EXTS:
        return _extract_docx(path)
    if ext in PDF_EXTS:
        return _extract_pdf(path)
    raise ExtractionError(f"unsupported extension: {ext or '(none)'}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/core/test_textextract.py -q`
Expected: PASS (10 passed)

- [ ] **Step 6: Gate and commit**

```bash
uv run pytest -q && uv run ruff check .
git add pyproject.toml uv.lock aggregator/core/textextract.py tests/core/test_textextract.py
git commit -m "feat(textextract): plain/docx/pdf text extraction helper"
```

---

## Task 2: Dropbox source — discovery and filtering

**Files:**
- Create: `aggregator/sources/dropbox.py`
- Test: `tests/sources/test_dropbox.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/sources/test_dropbox.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/sources/test_dropbox.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aggregator.sources.dropbox'`

- [ ] **Step 3: Implement discovery**

Create `aggregator/sources/dropbox.py`:

```python
"""Dropbox source: prose and documents from the locally-synced ``~/Dropbox``.

Records-shaped, one Record per file. Dropbox syncs to local disk, so this is
a filesystem source — no API, no tokens, no export ritual.

The tree is ~25k files and 4 GB, the overwhelming majority of which is
checked-in source code, media, and node_modules from project backups. Only
prose and documents are indexed; see SUPPORTED_EXTS.

``AGGREGATOR_DROPBOX_EXCLUDE`` exists from day one rather than being deferred
to "when someone needs it": this index is exposed to Claude over MCP and the
tree contains contracts, health records, and coaching material. Users need a
way to keep a folder out without patching code.
"""
from __future__ import annotations

import fnmatch
import logging
import os
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

from aggregator.core.textextract import SUPPORTED_EXTS

log = logging.getLogger(__name__)

DEFAULT_ROOT = "~/Dropbox"

# Pruned unconditionally — never worth indexing, and node_modules alone is
# ~10k files of the tree.
SKIP_DIR_NAMES = {"node_modules", ".git", ".dropbox.cache"}


def _is_skipped_dir(name: str) -> bool:
    """Directories pruned during the walk: known junk plus any dot-directory."""
    return name in SKIP_DIR_NAMES or name.startswith(".")


def _matches_exclude(relpath: str, patterns: tuple[str, ...]) -> bool:
    """True when relpath is covered by a user exclude pattern.

    A pattern matches either the path itself or anything beneath it, so
    ``Private`` excludes ``Private/deep/nested/secret.md`` without the user
    having to write ``Private/**``.
    """
    for pat in patterns:
        stem = pat.rstrip("/")
        if fnmatch.fnmatch(relpath, stem) or fnmatch.fnmatch(relpath, f"{stem}/*"):
            return True
        if PurePosixPath(relpath).match(stem):
            return True
    return False


class DropboxSource:
    """Source implementation for local Dropbox prose and documents."""

    name = "dropbox"

    def __init__(
        self,
        root: str | os.PathLike[str] | None = None,
        exclude: str | None = None,
    ):
        self.root = Path(
            root
            or os.environ.get("AGGREGATOR_DROPBOX_ROOT")
            or os.path.expanduser(DEFAULT_ROOT)
        )
        raw = exclude if exclude is not None else os.environ.get("AGGREGATOR_DROPBOX_EXCLUDE", "")
        self.exclude: tuple[str, ...] = tuple(p for p in raw.split(":") if p)

    def _iter_candidate_paths(self) -> Iterator[Path]:
        """Yield indexable file paths, pruning junk directories during the walk.

        Pruning happens by mutating ``dirnames`` in place so os.walk never
        descends into node_modules at all — on this tree that is the
        difference between statting 25k files and statting 12k.
        """
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if not _is_skipped_dir(d))
            here = Path(dirpath)
            rel_dir = here.relative_to(self.root)
            if str(rel_dir) != "." and _matches_exclude(rel_dir.as_posix(), self.exclude):
                dirnames[:] = []
                continue
            for filename in sorted(filenames):
                path = here / filename
                if path.suffix.lower() not in SUPPORTED_EXTS:
                    continue
                rel = path.relative_to(self.root).as_posix()
                if _matches_exclude(rel, self.exclude):
                    continue
                yield path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/sources/test_dropbox.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Gate and commit**

```bash
uv run pytest -q && uv run ruff check .
git add aggregator/sources/dropbox.py tests/sources/test_dropbox.py
git commit -m "feat(dropbox): file discovery with pruning and exclude globs"
```

---

## Task 3: Dropbox source — records, size caps, error policy

**Files:**
- Modify: `aggregator/sources/dropbox.py`
- Test: `tests/sources/test_dropbox.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/sources/test_dropbox.py`:

```python
from datetime import UTC, datetime, timedelta

import pytest

from aggregator.sources.dropbox import MAX_BODY_CHARS, MAX_TEXT_BYTES, DropboxSource


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


def test_ingest_returns_counts(tmp_path):
    _write(tmp_path, "a.md", "x")
    _write(tmp_path, "b.md", "y")
    result = DropboxSource(root=tmp_path).ingest(None)
    assert result.added == 2
    assert result.errors == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/sources/test_dropbox.py -q`
Expected: FAIL — `AttributeError: 'DropboxSource' object has no attribute 'iter_records'`

- [ ] **Step 3: Implement records**

Add to the constants block in `aggregator/sources/dropbox.py`, after `SKIP_DIR_NAMES`:

```python
# A 2 MB text file is not prose. The only files over this limit in the tree
# today are two copies of a 5.4 MB Swedish wordlist.
MAX_TEXT_BYTES = 2 * 1024 * 1024
MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_BODY_CHARS = 200_000

# Below this, a PDF is an image scan with no text layer. Skipped and counted,
# not errored — we made a deliberate no-OCR choice, so this is an expected
# outcome and must not page anyone.
MIN_PDF_TEXT_CHARS = 50
```

Add the imports:

```python
from datetime import UTC, datetime

from aggregator.core.textextract import (
    PDF_EXTS,
    SUPPORTED_EXTS,
    ExtractionError,
    extract_text,
    first_markdown_heading,
)
from aggregator.sources.base import IngestResult, Record, stable_id_for
```

Add these methods to `DropboxSource`:

```python
    def record_shape(self) -> dict[str, str]:
        """DSL-facing field surface (M2 help generator)."""
        return {
            "subject": "str (first markdown heading, or filename stem)",
            "body": "str (extracted file text)",
            "relpath": "str (path relative to the Dropbox root)",
            "ext": "str (file extension, lowercased, with dot)",
            "size_bytes": "int (file size on disk)",
            "truncated": "bool (present only when the body was cut)",
        }

    def _size_limit(self, path: Path) -> int:
        return MAX_PDF_BYTES if path.suffix.lower() in PDF_EXTS else MAX_TEXT_BYTES

    def _to_record(self, path: Path, mtime: datetime, text: str, size: int) -> Record:
        rel = path.relative_to(self.root).as_posix()
        ext = path.suffix.lower()
        truncated = len(text) > MAX_BODY_CHARS
        body = text[:MAX_BODY_CHARS] if truncated else text
        subject = (first_markdown_heading(text) if ext in {".md", ".markdown"} else None) or path.stem
        top = PurePosixPath(rel).parts[0] if len(PurePosixPath(rel).parts) > 1 else None
        extra: dict[str, object] = {
            "relpath": rel,
            "ext": ext,
            "size_bytes": size,
        }
        if truncated:
            extra["truncated"] = True
        return Record(
            stable_id=stable_id_for(self.name, rel),
            source=self.name,
            subject=subject,
            body=body,
            # created_at deliberately unset: filesystem birth time is not
            # preserved across the Dropbox sync boundary, so it would be a
            # confident-looking lie.
            created_at=None,
            updated_at=mtime,
            tags=[t for t in (top, ext.lstrip(".")) if t],
            extra=extra,
        )

    def iter_records(
        self,
        since: datetime | None,
        errors: list[str] | None = None,
    ) -> Iterator[Record]:
        """Yield one Record per indexable Dropbox file.

        ``since``: files with mtime <= since are skipped before extraction, so
        an incremental run costs a stat per file rather than a parse.

        Error policy: a file that cannot be parsed appends to ``errors`` and is
        skipped — one corrupt PDF never aborts an ingest of 1600 files. An
        image-only PDF is NOT an error (see MIN_PDF_TEXT_CHARS).
        """
        since_utc: datetime | None = None
        if since is not None:
            since_utc = since if since.tzinfo is not None else since.replace(tzinfo=UTC)

        for path in self._iter_candidate_paths():
            try:
                stat = path.stat()
            except OSError as e:
                log.warning("stat failed for %s: %s", path, e)
                if errors is not None:
                    errors.append(f"{path}: stat failed: {e}")
                continue

            mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
            if since_utc is not None and mtime <= since_utc:
                continue
            if stat.st_size > self._size_limit(path):
                log.info("skipping oversized file %s (%d bytes)", path, stat.st_size)
                continue

            try:
                text = extract_text(path)
            except ExtractionError as e:
                log.warning("extraction failed for %s: %s", path, e)
                if errors is not None:
                    errors.append(f"{path}: {e}")
                continue
            except OSError as e:
                log.warning("read failed for %s: %s", path, e)
                if errors is not None:
                    errors.append(f"{path}: read failed: {e}")
                continue

            if path.suffix.lower() in PDF_EXTS and len(text.strip()) < MIN_PDF_TEXT_CHARS:
                log.info("skipping image-only pdf %s", path)
                continue
            if not text.strip():
                continue

            yield self._to_record(path, mtime, text, stat.st_size)

    def ingest(self, since: datetime | None) -> IngestResult:
        """Count-only path for protocol compat; persistence is the CLI's job."""
        errors: list[str] = []
        added = sum(1 for _ in self.iter_records(since, errors=errors))
        return IngestResult(added=added, updated=0, skipped=0, errors=errors)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/sources/test_dropbox.py -q`
Expected: PASS (15 passed)

- [ ] **Step 5: Gate and commit**

```bash
uv run pytest -q && uv run ruff check .
git add aggregator/sources/dropbox.py tests/sources/test_dropbox.py
git commit -m "feat(dropbox): record mapping, size caps, per-file error policy"
```

---

## Task 4: Register the dropbox source

**Files:**
- Modify: `aggregator/cli.py:84-100` (`_default_sources`), and the `ingest` subcommand help around `aggregator/cli.py:445-448`
- Test: `tests/test_cli_sources.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli_sources.py`:

```python
from __future__ import annotations

from aggregator.cli import _default_sources


def test_dropbox_registered():
    sources = _default_sources()
    assert "dropbox" in sources
    assert sources["dropbox"].name == "dropbox"


def test_every_registered_source_name_matches_its_key():
    for key, src in _default_sources().items():
        assert src.name == key
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_sources.py -q`
Expected: FAIL — `AssertionError: assert 'dropbox' in {...}`

- [ ] **Step 3: Register**

In `aggregator/cli.py`, add the import alongside the other source imports:

```python
from aggregator.sources.dropbox import DropboxSource
```

And add to the dict returned by `_default_sources()`:

```python
        "dropbox": DropboxSource(),
```

Update the `ingest` subcommand's `help=` string so `dropbox` appears in the list of valid source names.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli_sources.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Smoke it against the real Dropbox tree**

Run: `uv run aggregator ingest dropbox --since 2026-08-01`
Expected: a summary line reporting a non-zero record count and `errors=<n>`. This is real data — eyeball that the count is plausible (hundreds, not 25000) before committing.

- [ ] **Step 6: Gate and commit**

```bash
uv run pytest -q && uv run ruff check .
git add aggregator/cli.py tests/test_cli_sources.py
git commit -m "feat(cli): register dropbox source"
```

---

## Task 5: TickTick CSV backup parser

**Files:**
- Create: `aggregator/sources/ticktick_csv.py`
- Test: `tests/sources/test_ticktick_csv.py`

TickTick backup CSVs carry six metadata lines, then the header on line 7. Columns: `Folder Name, List Name, Title, Tags, Content, Is Check list, Start Date, Due Date, Reminder, Repeat, Priority, Status, Created Time, Completed Time, Order, Timezone, Is All Day, Is Floating, Column Name, Column Order, View Mode, taskId, parentId`. `Status`: `0` normal, `1` completed, `2` archived.

- [ ] **Step 1: Write the failing tests**

Create `tests/sources/test_ticktick_csv.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime

from aggregator.sources.ticktick_csv import (
    HEADER_LINE_INDEX,
    is_ticktick_backup,
    parse_backup,
    row_to_record,
)

HEADER = (
    "Folder Name,List Name,Title,Tags,Content,Is Check list,Start Date,Due Date,"
    "Reminder,Repeat,Priority,Status,Created Time,Completed Time,Order,Timezone,"
    "Is All Day,Is Floating,Column Name,Column Order,View Mode,taskId,parentId"
)


def _backup(tmp_path, rows, name="TickTick.csv"):
    preamble = "\n".join(f'"Date: line {i}"' for i in range(HEADER_LINE_INDEX))
    p = tmp_path / name
    p.write_text("\n".join([preamble, HEADER, *rows]) + "\n", encoding="utf-8")
    return p


def _row(**over):
    values = {
        "folder": "Personal",
        "list": "Inbox",
        "title": "Buy milk",
        "tags": "errand",
        "content": "from the good shop",
        "start": "",
        "due": "2026-08-02T09:00:00+0000",
        "repeat": "",
        "priority": "3",
        "status": "0",
        "created": "2026-08-01T08:00:00+0000",
        "completed": "",
        "task_id": "abc123",
        "parent_id": "",
    }
    values.update(over)
    return (
        f'{values["folder"]},{values["list"]},{values["title"]},{values["tags"]},'
        f'"{values["content"]}",false,{values["start"]},{values["due"]},,'
        f'{values["repeat"]},{values["priority"]},{values["status"]},'
        f'{values["created"]},{values["completed"]},0,UTC,false,false,,,,'
        f'{values["task_id"]},{values["parent_id"]}'
    )


def test_detects_ticktick_backup(tmp_path):
    assert is_ticktick_backup(_backup(tmp_path, [_row()]))


def test_rejects_unrelated_csv(tmp_path):
    p = tmp_path / "bank.csv"
    p.write_text("date,amount\n2026-01-01,42\n", encoding="utf-8")
    assert not is_ticktick_backup(p)


def test_rejects_binary_file_without_raising(tmp_path):
    p = tmp_path / "weird.csv"
    p.write_bytes(b"\x00\x01\x02\xff")
    assert not is_ticktick_backup(p)


def test_parses_rows(tmp_path):
    rows = parse_backup(_backup(tmp_path, [_row(), _row(task_id="def456", title="Call bank")]))
    assert [r["taskId"] for r in rows] == ["abc123", "def456"]
    assert rows[0]["Title"] == "Buy milk"


def test_parses_multiline_quoted_content(tmp_path):
    row = _row(content="line one\nline two")
    rows = parse_backup(_backup(tmp_path, [row]))
    assert rows[0]["Content"] == "line one\nline two"


def test_row_to_record_open_task(tmp_path):
    row = parse_backup(_backup(tmp_path, [_row()]))[0]
    rec = row_to_record(row, source_file="TickTick.csv")
    assert rec.stable_id == "ticktick:abc123"
    assert rec.source == "ticktick"
    assert rec.subject == "Buy milk"
    assert rec.body == "from the good shop"
    assert set(rec.tags) == {"errand", "Inbox", "Personal", "open"}
    assert rec.created_at == datetime(2026, 8, 1, 8, 0, tzinfo=UTC)
    assert rec.updated_at == rec.created_at
    assert rec.extra["provenance"] == "csv"
    assert rec.extra["status"] == "0"
    assert rec.extra["source_file"] == "TickTick.csv"


def test_row_to_record_completed_task_uses_completed_time(tmp_path):
    row = parse_backup(
        _backup(tmp_path, [_row(status="1", completed="2026-08-03T17:30:00+0000")])
    )[0]
    rec = row_to_record(row, source_file="TickTick.csv")
    assert "completed" in rec.tags
    assert rec.updated_at == datetime(2026, 8, 3, 17, 30, tzinfo=UTC)
    assert rec.extra.get("completed_time_approx") is None


def test_row_to_record_archived_status(tmp_path):
    row = parse_backup(_backup(tmp_path, [_row(status="2")]))[0]
    assert "archived" in row_to_record(row, source_file="x.csv").tags


def test_row_without_task_id_is_dropped(tmp_path):
    rows = parse_backup(_backup(tmp_path, [_row(task_id="")]))
    assert rows == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/sources/test_ticktick_csv.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aggregator.sources.ticktick_csv'`

- [ ] **Step 3: Implement the parser**

Create `aggregator/sources/ticktick_csv.py`:

```python
"""TickTick backup-CSV parsing.

The CSV export is the ONLY source of completed-task history: TickTick's
official Open API filters completed tasks out of every read endpoint, which
is why no TickTick MCP server can back a history index on its own. See the
design doc for the evidence.

Backups are detected by structure, not filename — six metadata lines, then
the header on line 7 — so an arbitrary CSV sitting in ~/Downloads is ignored
rather than half-parsed into garbage records.
"""
from __future__ import annotations

import csv
import logging
from datetime import UTC, datetime
from pathlib import Path

from aggregator.sources.base import Record, stable_id_for

log = logging.getLogger(__name__)

SOURCE_NAME = "ticktick"

# TickTick writes six metadata lines before the real header.
HEADER_LINE_INDEX = 6

# A file must have all of these on its header line to be treated as a backup.
REQUIRED_COLUMNS = frozenset({"Title", "taskId", "Status", "Created Time"})

STATUS_TAGS = {"0": "open", "1": "completed", "2": "archived"}


def _parse_dt(value: str | None) -> datetime | None:
    """Parse a TickTick timestamp, tolerating both ISO offsets and ``+0000``."""
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    for parse in (
        datetime.fromisoformat,
        lambda v: datetime.strptime(v, "%Y-%m-%dT%H:%M:%S%z"),
        lambda v: datetime.strptime(v, "%Y-%m-%d %H:%M:%S"),
    ):
        try:
            parsed = parse(text)
        except ValueError:
            continue
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    log.warning("unparseable ticktick timestamp: %r", value)
    return None


def _header_fields(path: Path) -> list[str] | None:
    """Return the backup's header fields, or None if this is not a backup."""
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            for index, row in enumerate(reader):
                if index < HEADER_LINE_INDEX:
                    continue
                return row
    except (OSError, UnicodeDecodeError, csv.Error) as e:
        log.debug("not a readable csv: %s (%s)", path, e)
    return None


def is_ticktick_backup(path: Path) -> bool:
    fields = _header_fields(path)
    return fields is not None and REQUIRED_COLUMNS.issubset(set(fields))


def parse_backup(path: Path) -> list[dict[str, str]]:
    """Return the backup's task rows. Rows without a taskId are dropped."""
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        for _ in range(HEADER_LINE_INDEX):
            next(reader, None)
        header = next(reader, None)
        if header is None:
            return []
        rows = [dict(zip(header, row, strict=False)) for row in reader if row]
    return [r for r in rows if r.get("taskId")]


def row_to_record(row: dict[str, str], source_file: str) -> Record:
    """Map one backup row to a Record."""
    status = (row.get("Status") or "0").strip()
    created = _parse_dt(row.get("Created Time"))
    completed = _parse_dt(row.get("Completed Time"))
    tags = [t.strip() for t in (row.get("Tags") or "").split(",") if t.strip()]
    for key in ("List Name", "Folder Name"):
        value = (row.get(key) or "").strip()
        if value:
            tags.append(value)
    tags.append(STATUS_TAGS.get(status, "open"))

    return Record(
        stable_id=stable_id_for(SOURCE_NAME, row["taskId"]),
        source=SOURCE_NAME,
        subject=(row.get("Title") or "").strip() or row["taskId"],
        body=row.get("Content") or "",
        tags=tags,
        created_at=created,
        updated_at=completed or created,
        extra={
            "provenance": "csv",
            "status": status,
            "priority": (row.get("Priority") or "").strip(),
            "due_date": (row.get("Due Date") or "").strip(),
            "start_date": (row.get("Start Date") or "").strip(),
            "repeat": (row.get("Repeat") or "").strip(),
            "parent_id": (row.get("parentId") or "").strip(),
            "source_file": source_file,
        },
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/sources/test_ticktick_csv.py -q`
Expected: PASS (10 passed)

- [ ] **Step 5: Gate and commit**

```bash
uv run pytest -q && uv run ruff check .
git add aggregator/sources/ticktick_csv.py tests/sources/test_ticktick_csv.py
git commit -m "feat(ticktick): backup-CSV detection and parsing"
```

---

## Task 6: TickTick Open API client (GET-only)

**Files:**
- Create: `aggregator/sources/ticktick_api.py`
- Test: `tests/sources/test_ticktick_api.py`

The token TickTick issues carries write scope — there is no read-only variant — so the compensating control is that this module structurally cannot issue a non-GET request. That guard is asserted in tests, mirroring how the GitHub source refuses write-capable tokens.

- [ ] **Step 1: Write the failing tests**

Create `tests/sources/test_ticktick_api.py`:

```python
from __future__ import annotations

import json
from urllib.error import HTTPError

import pytest

from aggregator.sources import ticktick_api


class _FakeResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_request_refuses_non_get():
    with pytest.raises(ticktick_api.WriteAttemptError):
        ticktick_api._request("POST", "https://api.ticktick.com/open/v1/task", token="t")


def test_get_sends_bearer_and_get_method(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["method"] = req.get_method()
        seen["auth"] = req.get_header("Authorization")
        seen["url"] = req.full_url
        return _FakeResponse({"ok": True})

    monkeypatch.setattr(ticktick_api.request, "urlopen", fake_urlopen)
    assert ticktick_api._request("GET", "https://x/y", token="tok") == {"ok": True}
    assert seen["method"] == "GET"
    assert seen["auth"] == "Bearer tok"


def test_fetch_open_tasks_walks_projects(monkeypatch):
    calls = []

    def fake_request(method, url, token, timeout=30):
        calls.append(url)
        if url.endswith("/project"):
            return [{"id": "p1", "name": "Work"}, {"id": "p2", "name": "Home"}]
        if url.endswith("/project/p1/data"):
            return {"tasks": [{"id": "t1", "title": "Ship it", "projectId": "p1"}]}
        return {"tasks": [{"id": "t2", "title": "Dishes", "projectId": "p2"}]}

    monkeypatch.setattr(ticktick_api, "_request", fake_request)
    tasks = ticktick_api.fetch_open_tasks("tok")
    assert {t["id"] for t in tasks} == {"t1", "t2"}
    assert {t["_projectName"] for t in tasks} == {"Work", "Home"}
    assert len(calls) == 3


def test_fetch_open_tasks_one_bad_project_does_not_abort(monkeypatch):
    def fake_request(method, url, token, timeout=30):
        if url.endswith("/project"):
            return [{"id": "p1", "name": "Work"}, {"id": "p2", "name": "Home"}]
        if url.endswith("/project/p1/data"):
            raise HTTPError(url, 500, "boom", {}, None)
        return {"tasks": [{"id": "t2", "title": "Dishes", "projectId": "p2"}]}

    monkeypatch.setattr(ticktick_api, "_request", fake_request)
    errors: list[str] = []
    tasks = ticktick_api.fetch_open_tasks("tok", errors=errors)
    assert {t["id"] for t in tasks} == {"t2"}
    assert len(errors) == 1


def test_task_to_record_shape():
    task = {
        "id": "t1",
        "title": "Ship it",
        "content": "details here",
        "priority": 5,
        "dueDate": "2026-08-09T12:00:00+0000",
        "tags": ["work"],
        "_projectName": "Work",
    }
    rec = ticktick_api.task_to_record(task)
    assert rec.stable_id == "ticktick:t1"
    assert rec.subject == "Ship it"
    assert rec.body == "details here"
    assert set(rec.tags) >= {"work", "Work", "open"}
    assert rec.extra["provenance"] == "api"
    assert rec.extra["status"] == "0"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/sources/test_ticktick_api.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aggregator.sources.ticktick_api'`

- [ ] **Step 3: Implement the client**

Create `aggregator/sources/ticktick_api.py`:

```python
"""TickTick Open API client — read-only by construction.

This is the repo's first direct HTTP call; the GitHub source shells out to
`gh` instead. Rather than introduce `requests` or `httpx` for a handful of
GETs, it uses stdlib urllib.

SECURITY: TickTick issues no read-only token — every token carries write
scope. The compensating control is that ``_request`` refuses any method
other than GET, so no code path in this repo can mutate the user's tasks.
Tested in tests/sources/test_ticktick_api.py.

COVERAGE LIMIT: the Open API filters completed tasks out of every read
endpoint. This module sees open tasks only; completions are inferred here
(by disappearance) and corrected later from the CSV backup.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from urllib import request
from urllib.error import URLError

from aggregator.sources.base import Record, stable_id_for

log = logging.getLogger(__name__)

BASE_URL = "https://api.ticktick.com/open/v1"
SOURCE_NAME = "ticktick"
DEFAULT_TIMEOUT = 30

# TickTick priority values: 0 none, 1 low, 3 medium, 5 high.
PRIORITY_NAMES = {0: "none", 1: "low", 3: "medium", 5: "high"}


class WriteAttemptError(RuntimeError):
    """Raised when any non-GET request is attempted. See module docstring."""


def _request(method: str, url: str, token: str, timeout: int = DEFAULT_TIMEOUT) -> object:
    if method != "GET":
        raise WriteAttemptError(
            f"refusing {method} {url}: the ticktick source is read-only by construction"
        )
    req = request.Request(url, method="GET")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/json")
    with request.urlopen(req, timeout=timeout) as response:  # noqa: S310 - fixed https base
        return json.loads(response.read().decode("utf-8"))


def fetch_open_tasks(token: str, errors: list[str] | None = None) -> list[dict]:
    """Return every currently-open task across all projects.

    A project that fails to fetch is recorded and skipped: one 500 must not
    cost us the other nine projects.
    """
    projects = _request("GET", f"{BASE_URL}/project", token)
    if not isinstance(projects, list):
        raise ValueError(f"unexpected /project payload: {type(projects).__name__}")

    tasks: list[dict] = []
    for project in projects:
        project_id = project.get("id")
        if not project_id:
            continue
        try:
            data = _request("GET", f"{BASE_URL}/project/{project_id}/data", token)
        except (URLError, OSError, ValueError) as e:
            log.warning("ticktick project %s fetch failed: %s", project_id, e)
            if errors is not None:
                errors.append(f"ticktick project {project_id}: {e}")
            continue
        for task in (data or {}).get("tasks", []):
            task["_projectName"] = project.get("name", "")
            tasks.append(task)
    return tasks


def task_to_record(
    task: dict,
    *,
    completed_at: datetime | None = None,
    provenance: str = "api",
) -> Record:
    """Map one API task payload to a Record.

    ``completed_at`` is set only for inferred completions, and always alongside
    ``provenance="api-inferred-complete"``.
    """
    inferred = completed_at is not None
    tags = [str(t) for t in (task.get("tags") or [])]
    project_name = task.get("_projectName") or ""
    if project_name:
        tags.append(project_name)
    tags.append("completed" if inferred else "open")

    extra: dict[str, object] = {
        "provenance": provenance,
        "status": "1" if inferred else "0",
        "priority": PRIORITY_NAMES.get(task.get("priority", 0), str(task.get("priority", ""))),
        "due_date": task.get("dueDate") or "",
        "start_date": task.get("startDate") or "",
        "repeat": task.get("repeatFlag") or "",
        "parent_id": task.get("parentId") or "",
        "project_id": task.get("projectId") or "",
    }
    if inferred:
        # Never let an approximate timestamp pass for a real one.
        extra["completed_time_approx"] = True

    return Record(
        stable_id=stable_id_for(SOURCE_NAME, str(task["id"])),
        source=SOURCE_NAME,
        subject=(task.get("title") or "").strip() or str(task["id"]),
        body=task.get("content") or "",
        tags=tags,
        created_at=_parse_api_dt(task.get("createdTime")),
        updated_at=completed_at or _parse_api_dt(task.get("modifiedTime")),
        extra=extra,
    )


def _parse_api_dt(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    for parse in (
        datetime.fromisoformat,
        lambda v: datetime.strptime(v, "%Y-%m-%dT%H:%M:%S%z"),
        lambda v: datetime.strptime(v, "%Y-%m-%dT%H:%M:%S.%f%z"),
    ):
        try:
            parsed = parse(value.strip())
        except ValueError:
            continue
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def resolve_token(token: str | None, token_file: str | None) -> str | None:
    """Return the bearer token, or None when the API leg should be skipped.

    An absent or empty token is a supported state, not an error: the source
    falls back to CSV-only. An unreadable token FILE is different — the user
    asked for the API leg and the secret is broken — so that raises.
    """
    if token:
        return token.strip() or None
    if token_file:
        from pathlib import Path

        content = Path(token_file).read_text(encoding="utf-8").strip()
        return content or None
    return None
```

Add `from collections.abc import Iterable` only if used; if ruff flags it as unused, delete the import.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/sources/test_ticktick_api.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Gate and commit**

```bash
uv run pytest -q && uv run ruff check .
git add aggregator/sources/ticktick_api.py tests/sources/test_ticktick_api.py
git commit -m "feat(ticktick): read-only Open API client"
```

---

## Task 7: Open-task state file and completion inference

**Files:**
- Modify: `aggregator/sources/ticktick_api.py`
- Test: `tests/sources/test_ticktick_api.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/sources/test_ticktick_api.py`:

```python
from datetime import UTC, datetime


def test_state_roundtrip(tmp_path):
    path = tmp_path / "open_tasks.json"
    now = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
    ticktick_api.save_state(path, [{"id": "t1", "title": "A"}], now)
    state = ticktick_api.load_state(path)
    assert state["t1"]["task"]["title"] == "A"
    assert state["t1"]["last_seen"] == now.isoformat()


def test_load_missing_state_returns_empty(tmp_path):
    assert ticktick_api.load_state(tmp_path / "nope.json") == {}


def test_load_corrupt_state_returns_empty(tmp_path):
    path = tmp_path / "open_tasks.json"
    path.write_text("{not json", encoding="utf-8")
    assert ticktick_api.load_state(path) == {}


def test_disappeared_task_becomes_inferred_completion(tmp_path):
    path = tmp_path / "open_tasks.json"
    first = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)
    second = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
    ticktick_api.save_state(
        path,
        [{"id": "t1", "title": "Gone", "_projectName": "Work"}, {"id": "t2", "title": "Stays"}],
        first,
    )
    prev = ticktick_api.load_state(path)
    records = ticktick_api.infer_completions(prev, current_ids={"t2"}, now=second)
    assert [r.stable_id for r in records] == ["ticktick:t1"]
    rec = records[0]
    assert rec.extra["provenance"] == "api-inferred-complete"
    assert rec.extra["completed_time_approx"] is True
    assert rec.updated_at == second
    assert "completed" in rec.tags


def test_no_inference_on_first_ever_poll(tmp_path):
    records = ticktick_api.infer_completions({}, current_ids={"t1"}, now=datetime.now(UTC))
    assert records == []


def test_default_state_path_respects_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    assert ticktick_api.default_state_path() == tmp_path / "aggregator" / "ticktick" / "open_tasks.json"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/sources/test_ticktick_api.py -q`
Expected: FAIL — `AttributeError: module 'aggregator.sources.ticktick_api' has no attribute 'save_state'`

- [ ] **Step 3: Implement state and inference**

Append to `aggregator/sources/ticktick_api.py`:

```python
def default_state_path() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
    return Path(base) / "aggregator" / "ticktick" / "open_tasks.json"


def load_state(path: Path) -> dict[str, dict]:
    """Return the previous poll's open tasks, or {} when unavailable.

    This is a cache, not a database: a missing or corrupt file costs one
    poll's worth of inference and then self-heals. Failing the whole ingest
    over it would be worse than the gap.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        log.warning("ticktick state unreadable at %s (%s); starting fresh", path, e)
        return {}
    return data if isinstance(data, dict) else {}


def save_state(path: Path, tasks: Iterable[dict], now: datetime) -> None:
    """Persist the current open-task set, atomically."""
    payload = {
        str(task["id"]): {"task": task, "last_seen": now.isoformat()}
        for task in tasks
        if task.get("id")
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def infer_completions(
    previous: dict[str, dict],
    current_ids: set[str],
    now: datetime,
) -> list[Record]:
    """Records for tasks that were open last poll and are absent now.

    The Open API cannot report completions, so disappearance is the only
    available signal. The resulting timestamp is approximate and is flagged
    as such; a later CSV backup overwrites it with the real Completed Time.
    """
    records = []
    for task_id, entry in sorted(previous.items()):
        if task_id in current_ids:
            continue
        task = entry.get("task") or {}
        if not task.get("id"):
            continue
        records.append(
            task_to_record(task, completed_at=now, provenance="api-inferred-complete")
        )
    return records
```

Add `import os` and `from pathlib import Path` to the module imports, and remove the local `from pathlib import Path` inside `resolve_token`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/sources/test_ticktick_api.py -q`
Expected: PASS (11 passed)

- [ ] **Step 5: Gate and commit**

```bash
uv run pytest -q && uv run ruff check .
git add aggregator/sources/ticktick_api.py tests/sources/test_ticktick_api.py
git commit -m "feat(ticktick): open-task state file and completion inference"
```

---

## Task 8: TickTick source — merge, archive, register

**Files:**
- Create: `aggregator/sources/ticktick.py`
- Modify: `aggregator/cli.py` (`_default_sources`, ingest help text)
- Test: `tests/sources/test_ticktick.py`, `tests/test_cli_sources.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/sources/test_ticktick.py`:

```python
from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

from aggregator.sources import ticktick_api
from aggregator.sources.ticktick import TickTickSource
from tests.sources.test_ticktick_csv import HEADER, _row


def _backup(path, rows):
    preamble = "\n".join(f'"Date: line {i}"' for i in range(6))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([preamble, HEADER, *rows]) + "\n", encoding="utf-8")
    return path


def _source(tmp_path, **kw):
    return TickTickSource(
        backup_dir=tmp_path / "downloads",
        archive_dir=tmp_path / "archive",
        state_file=tmp_path / "state.json",
        **kw,
    )


def test_csv_only_when_no_token(tmp_path):
    _backup(tmp_path / "downloads" / "TickTick.csv", [_row()])
    records = list(_source(tmp_path).iter_records(None))
    assert [r.stable_id for r in records] == ["ticktick:abc123"]
    assert records[0].extra["provenance"] == "csv"


def test_non_ticktick_csv_ignored(tmp_path):
    downloads = tmp_path / "downloads"
    downloads.mkdir(parents=True)
    (downloads / "bank.csv").write_text("date,amount\n2026-01-01,42\n", encoding="utf-8")
    assert list(_source(tmp_path).iter_records(None)) == []


def test_backup_is_archived(tmp_path):
    _backup(tmp_path / "downloads" / "TickTick.csv", [_row()])
    list(_source(tmp_path).iter_records(None))
    assert (tmp_path / "archive" / "TickTick.csv").exists()


def test_archived_backup_still_ingested_after_download_cleared(tmp_path):
    _backup(tmp_path / "downloads" / "TickTick.csv", [_row()])
    src = _source(tmp_path)
    list(src.iter_records(None))
    (tmp_path / "downloads" / "TickTick.csv").unlink()
    records = list(_source(tmp_path).iter_records(None))
    assert [r.stable_id for r in records] == ["ticktick:abc123"]


def test_since_skips_old_backups(tmp_path):
    path = _backup(tmp_path / "downloads" / "TickTick.csv", [_row()])
    stale = (datetime.now(UTC) - timedelta(days=30)).timestamp()
    os.utime(path, (stale, stale))
    src = _source(tmp_path)
    since = datetime.now(UTC) - timedelta(days=1)
    assert list(src.iter_records(since)) == []


def test_api_leg_merges_with_csv(tmp_path, monkeypatch):
    _backup(tmp_path / "downloads" / "TickTick.csv", [_row(task_id="csv1", title="From CSV")])
    monkeypatch.setattr(
        ticktick_api,
        "fetch_open_tasks",
        lambda token, errors=None: [{"id": "api1", "title": "From API", "_projectName": "Work"}],
    )
    records = {r.stable_id: r for r in _source(tmp_path, token="tok").iter_records(None)}
    assert set(records) == {"ticktick:csv1", "ticktick:api1"}
    assert records["ticktick:api1"].extra["provenance"] == "api"


def test_fresher_api_observation_beats_stale_csv(tmp_path, monkeypatch):
    path = _backup(tmp_path / "downloads" / "TickTick.csv", [_row(task_id="t1", status="1")])
    stale = (datetime.now(UTC) - timedelta(days=30)).timestamp()
    os.utime(path, (stale, stale))
    monkeypatch.setattr(
        ticktick_api,
        "fetch_open_tasks",
        lambda token, errors=None: [{"id": "t1", "title": "Reopened"}],
    )
    (rec,) = list(_source(tmp_path, token="tok").iter_records(None))
    assert rec.extra["provenance"] == "api"
    assert "open" in rec.tags


def test_fresher_csv_beats_api(tmp_path, monkeypatch):
    _backup(
        tmp_path / "downloads" / "TickTick.csv",
        [_row(task_id="t1", status="1", completed="2026-08-03T17:30:00+0000")],
    )
    monkeypatch.setattr(
        ticktick_api,
        "fetch_open_tasks",
        lambda token, errors=None: [{"id": "t1", "title": "Stale open view"}],
    )
    src = _source(tmp_path, token="tok")
    src._api_observed_at = datetime(2000, 1, 1, tzinfo=UTC)
    (rec,) = list(src.iter_records(None))
    assert rec.extra["provenance"] == "csv"


def test_completion_inferred_across_two_polls(tmp_path, monkeypatch):
    tasks = [{"id": "t1", "title": "Gone"}, {"id": "t2", "title": "Stays"}]
    monkeypatch.setattr(ticktick_api, "fetch_open_tasks", lambda token, errors=None: tasks)
    list(_source(tmp_path, token="tok").iter_records(None))

    tasks = [{"id": "t2", "title": "Stays"}]
    records = {r.stable_id: r for r in _source(tmp_path, token="tok").iter_records(None)}
    assert records["ticktick:t1"].extra["provenance"] == "api-inferred-complete"
    assert records["ticktick:t1"].extra["completed_time_approx"] is True
    assert records["ticktick:t2"].extra["provenance"] == "api"


def test_api_failure_records_error_and_keeps_csv(tmp_path, monkeypatch):
    _backup(tmp_path / "downloads" / "TickTick.csv", [_row()])

    def boom(token, errors=None):
        raise OSError("network down")

    monkeypatch.setattr(ticktick_api, "fetch_open_tasks", boom)
    errors: list[str] = []
    records = list(_source(tmp_path, token="tok").iter_records(None, errors=errors))
    assert [r.stable_id for r in records] == ["ticktick:abc123"]
    assert len(errors) == 1
    assert "network down" in errors[0]


def test_token_file_is_read(tmp_path, monkeypatch):
    token_file = tmp_path / "token"
    token_file.write_text("filetoken\n", encoding="utf-8")
    seen = {}

    def fake_fetch(token, errors=None):
        seen["token"] = token
        return []

    monkeypatch.setattr(ticktick_api, "fetch_open_tasks", fake_fetch)
    list(_source(tmp_path, token_file=str(token_file)).iter_records(None))
    assert seen["token"] == "filetoken"
```

Append to `tests/test_cli_sources.py`:

```python
def test_ticktick_registered():
    sources = _default_sources()
    assert "ticktick" in sources
    assert sources["ticktick"].name == "ticktick"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/sources/test_ticktick.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aggregator.sources.ticktick'`

- [ ] **Step 3: Implement the source**

Create `aggregator/sources/ticktick.py`:

```python
"""TickTick source: task history from CSV backups plus a live open-task poll.

Two legs, because neither is sufficient alone:

* CSV backup — the only place completed tasks exist, but manual and stale.
* Open API — always current, but structurally blind to completed tasks.

They are merged by observation recency (newest wins per task), which handles
the un-complete case for free: a task marked completed in last month's backup
but open in today's poll correctly reads as open.

Every emitted record carries ``extra.provenance`` so a search result never
hides which leg it came from.
"""
from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from aggregator.sources import ticktick_api
from aggregator.sources.base import IngestResult, Record
from aggregator.sources.ticktick_csv import is_ticktick_backup, parse_backup, row_to_record

log = logging.getLogger(__name__)

DEFAULT_BACKUP_DIR = "~/Downloads"


def _default_archive_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return Path(base) / "aggregator" / "ticktick" / "backups"


class TickTickSource:
    """Source implementation for TickTick tasks."""

    name = "ticktick"

    def __init__(
        self,
        backup_dir: str | os.PathLike[str] | None = None,
        token: str | None = None,
        token_file: str | None = None,
        state_file: str | os.PathLike[str] | None = None,
        archive_dir: str | os.PathLike[str] | None = None,
    ):
        self.backup_dir = Path(
            backup_dir
            or os.environ.get("AGGREGATOR_TICKTICK_DIR")
            or os.path.expanduser(DEFAULT_BACKUP_DIR)
        )
        self.archive_dir = Path(archive_dir) if archive_dir else _default_archive_dir()
        self.state_file = (
            Path(state_file) if state_file else ticktick_api.default_state_path()
        )
        self._token_arg = token if token is not None else os.environ.get("AGGREGATOR_TICKTICK_TOKEN")
        self._token_file = (
            token_file if token_file is not None else os.environ.get("AGGREGATOR_TICKTICK_TOKEN_FILE")
        )
        # Overridable in tests to exercise both sides of the precedence rule.
        self._api_observed_at = datetime.now(UTC)

    def record_shape(self) -> dict[str, str]:
        """DSL-facing field surface (M2 help generator)."""
        return {
            "subject": "str (task title)",
            "body": "str (task content/notes)",
            "provenance": "str (csv | api | api-inferred-complete)",
            "status": "str (0 open, 1 completed, 2 archived)",
            "priority": "str (none | low | medium | high)",
            "due_date": "str (ISO 8601, may be empty)",
            "parent_id": "str (parent taskId, may be empty)",
            "completed_time_approx": "bool (present when completion was inferred)",
        }

    def _backup_files(self, since: datetime | None) -> list[tuple[Path, datetime]]:
        """Return (path, mtime) for every TickTick backup CSV, newest last.

        Both the download dir and the archive are scanned so a `--rebuild`
        still sees the deep history after ~/Downloads has been cleared.
        """
        found: dict[str, tuple[Path, datetime]] = {}
        for directory in (self.archive_dir, self.backup_dir):
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.csv")):
                try:
                    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
                except OSError:
                    continue
                if since is not None and mtime <= since:
                    continue
                if not is_ticktick_backup(path):
                    continue
                found[path.name] = (path, mtime)
        return sorted(found.values(), key=lambda pair: pair[1])

    def _archive(self, path: Path) -> None:
        if path.parent == self.archive_dir:
            return
        try:
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, self.archive_dir / path.name)
        except OSError as e:
            log.warning("could not archive backup %s: %s", path, e)

    def _csv_candidates(
        self, since: datetime | None
    ) -> dict[str, tuple[datetime, Record]]:
        candidates: dict[str, tuple[datetime, Record]] = {}
        for path, mtime in self._backup_files(since):
            self._archive(path)
            for row in parse_backup(path):
                record = row_to_record(row, source_file=path.name)
                task_id = row["taskId"]
                if task_id not in candidates or candidates[task_id][0] <= mtime:
                    candidates[task_id] = (mtime, record)
        return candidates

    def _api_candidates(
        self, errors: list[str] | None
    ) -> dict[str, tuple[datetime, Record]]:
        token = ticktick_api.resolve_token(self._token_arg, self._token_file)
        if not token:
            log.info("no ticktick token configured; running CSV-only")
            return {}

        observed = self._api_observed_at
        try:
            tasks = ticktick_api.fetch_open_tasks(token, errors=errors)
        except Exception as e:  # network, auth, malformed payload
            log.warning("ticktick api poll failed: %s", e)
            if errors is not None:
                errors.append(f"ticktick api poll failed: {e}")
            return {}

        candidates: dict[str, tuple[datetime, Record]] = {}
        current_ids = {str(t["id"]) for t in tasks if t.get("id")}
        for task in tasks:
            if not task.get("id"):
                continue
            candidates[str(task["id"])] = (observed, ticktick_api.task_to_record(task))

        previous = ticktick_api.load_state(self.state_file)
        for record in ticktick_api.infer_completions(previous, current_ids, observed):
            candidates[record.stable_id.split(":", 1)[1]] = (observed, record)

        ticktick_api.save_state(self.state_file, tasks, observed)
        return candidates

    def iter_records(
        self,
        since: datetime | None,
        errors: list[str] | None = None,
    ) -> Iterator[Record]:
        """Yield one Record per task, newest observation winning per task id."""
        since_utc: datetime | None = None
        if since is not None:
            since_utc = since if since.tzinfo is not None else since.replace(tzinfo=UTC)

        merged = self._csv_candidates(since_utc)
        for task_id, (observed, record) in self._api_candidates(errors).items():
            if task_id not in merged or merged[task_id][0] < observed:
                merged[task_id] = (observed, record)

        for _, record in sorted(merged.items()):
            yield record[1]

    def ingest(self, since: datetime | None) -> IngestResult:
        """Count-only path for protocol compat; persistence is the CLI's job."""
        errors: list[str] = []
        added = sum(1 for _ in self.iter_records(since, errors=errors))
        return IngestResult(added=added, updated=0, skipped=0, errors=errors)
```

In `aggregator/cli.py`, add the import and the registration entry:

```python
from aggregator.sources.ticktick import TickTickSource
```
```python
        "ticktick": TickTickSource(),
```

Update the `ingest` subcommand's `help=` string to include `ticktick`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/sources/test_ticktick.py tests/test_cli_sources.py -q`
Expected: PASS (14 passed)

- [ ] **Step 5: Gate and commit**

```bash
uv run pytest -q && uv run ruff check .
git add aggregator/sources/ticktick.py aggregator/cli.py tests/sources/test_ticktick.py tests/test_cli_sources.py
git commit -m "feat(ticktick): merge CSV and API legs, register source"
```

---

## Task 9: Fail loudly — non-zero exit when a run completes with errors

Today `_cmd_ingest` prints the first five errors and then returns 0. A systemd timer sees success. Per `tasks/session-constraints.md` that is exactly the silent rot the user ruled out.

Exit codes after this task: `0` clean, `1` hard failure (unchanged), `2` usage error (unchanged — already used for unknown source, bad `--since`, unknown subcommand; see `aggregator/cli.py:200,261,268,386,392,508`), `3` completed but with errors.

`3`, not `2`: `2` is already the usage-error code throughout this file, and the systemd wrapper must be able to tell "you typed a bad source name" from "the run finished but dropped three PDFs" — those need different notification text and different human responses. A distinct code also keeps every existing `!= 0` caller seeing a failure.

**Files:**
- Modify: `aggregator/cli.py` — the records path around `aggregator/cli.py:356-364` and the entities path around `aggregator/cli.py:246-251`
- Test: `tests/test_cli_ingest_exit_codes.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli_ingest_exit_codes.py`:

```python
from __future__ import annotations

from datetime import datetime

from aggregator.cli import main
from aggregator.sources.base import Record


class _NoisySource:
    name = "noisy"

    def record_shape(self):
        return {"subject": "str", "body": "str"}

    def iter_records(self, since: datetime | None, errors: list[str] | None = None):
        if errors is not None:
            errors.append("some/file.pdf: pdf parse failed")
        yield Record(
            stable_id="noisy:1", source="noisy", subject="ok", body="fine"
        )


class _CleanSource(_NoisySource):
    name = "clean"

    def iter_records(self, since: datetime | None, errors: list[str] | None = None):
        yield Record(stable_id="clean:1", source="clean", subject="ok", body="fine")


def test_clean_run_exits_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "aggregator.cli._default_sources", lambda: {"clean": _CleanSource()}
    )
    rc = main(["--db", str(tmp_path / "c.db"), "ingest", "clean", "--yes"])
    assert rc == 0


def test_run_with_errors_exits_three(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "aggregator.cli._default_sources", lambda: {"noisy": _NoisySource()}
    )
    rc = main(["--db", str(tmp_path / "n.db"), "ingest", "noisy", "--yes"])
    assert rc == 3


def test_unknown_source_still_exits_two(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "aggregator.cli._default_sources", lambda: {"clean": _CleanSource()}
    )
    rc = main(["--db", str(tmp_path / "u.db"), "ingest", "nope", "--yes"])
    assert rc == 2
```

Adjust the `main([...])` argument lists to match the real CLI's flag names and required arguments — read `build_parser` in `aggregator/cli.py:417-483` first and use exactly what it defines.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli_ingest_exit_codes.py -q`
Expected: FAIL — `assert 0 == 3` on `test_run_with_errors_exits_three`. The other two tests should already pass.

- [ ] **Step 3: Change the exit codes**

Two edit sites, identical shape. Records path, currently `aggregator/cli.py:361-364`:

```python
    if errors:
        for e in errors[:5]:
            print(f"  error: {e}", file=sys.stderr)
    return 0
```

becomes:

```python
    if errors:
        for e in errors[:5]:
            print(f"  error: {e}", file=sys.stderr)
        # 3, not 0: a run that completed but dropped files is not a success.
        # A timer reporting success while the index rots is indistinguishable
        # from one with nothing to do. 3 rather than 2 because 2 is this
        # file's usage-error code and the systemd wrapper must tell them
        # apart.
        return 3
    return 0
```

Entities path, currently `aggregator/cli.py:248-251` — byte-identical text, same replacement, minus the comment (one comment in the file is enough; put it on the records path).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli_ingest_exit_codes.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Verify no existing test asserted the old behaviour**

Run: `uv run pytest -q`
Expected: PASS. If a pre-existing test asserts `rc == 0` for an ingest that produced errors, update that test to expect `2` and note it in the commit body — do not weaken the new behaviour to keep an old assertion green.

- [ ] **Step 6: Gate and commit**

```bash
uv run pytest -q && uv run ruff check .
git add aggregator/cli.py tests/test_cli_ingest_exit_codes.py
git commit -m "fix(cli): exit 3 when an ingest completes with errors"
```

---

## Task 10: Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document both sources**

Add a section covering:

- `dropbox`: what it indexes, `AGGREGATOR_DROPBOX_ROOT`, `AGGREGATOR_DROPBOX_EXCLUDE` syntax with a worked example, the no-OCR limitation, the 2 MB / 20 MB / 200k-char caps.
- `ticktick`: how to produce a backup (TickTick → Settings → Account → Backup & Import → Generate Backup), that it goes in `~/Downloads` by default, `AGGREGATOR_TICKTICK_DIR`, `AGGREGATOR_TICKTICK_TOKEN` / `AGGREGATOR_TICKTICK_TOKEN_FILE`, that the API leg is optional and CSV-only is a supported mode, and — prominently — that **completed tasks only ever arrive via a CSV backup**, so without periodic backups the completed history stops at the last export.
- Exit codes: `0` clean, `1` hard failure, `2` usage error, `3` completed with errors.

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: dropbox and ticktick source usage"
```

---

## Out of scope for this plan

The systemd user timers and the `notify-send` OnFailure units live in the
nixos-config repo and ship through its worktree/PR pipeline. The agenix
secret `ticktick-api-token` is already declared on branch
`feat/ticktick-secret`. `modules/nixos/aggregator-github-timer.nix` is the
template, and it currently only logs to the journal — it needs the same
OnFailure notification retrofit to satisfy the fail-loudly constraint.
