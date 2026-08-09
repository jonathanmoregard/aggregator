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
