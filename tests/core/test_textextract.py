from __future__ import annotations

import pytest

from aggregator.core.textextract import (
    SUPPORTED_EXTS,
    ExtractionError,
    extract_text,
    first_markdown_heading,
)


def test_supported_exts_are_lowercase_with_dots():
    assert {".md", ".markdown", ".txt", ".docx", ".pdf"} == SUPPORTED_EXTS


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
