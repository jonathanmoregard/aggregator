"""Tests for aggregator.sources.substack (Chunk 5, chat-exports plan).

Records-shaped source over Substack data-export zips (Settings → Exports).
Layout: ``posts/<post-id>.<slug>.html`` per post, with ``.delivers.csv`` and
``.opens.csv`` sidecars we ignore. Top-level ``posts.csv`` and
``email_list.<pub>.csv`` also ignored.

Covers:

- source name + protocol shape.
- discovery via ``exportdrops.discover_export_files('substack', ...)``:
  the zip contains ``posts/*.html`` → classified as substack; unrelated
  zips ignored.
- stable_id: numeric leading id → ``substack:<id>``; non-numeric stem →
  ``substack:<stem>``.
- subject: ``<h1>`` preferred over ``<title>`` preferred over slug
  title-case fallback.
- body: HTML text stripped, paragraph breaks preserved
  (``</p>``/``</div>``/``<br>`` → ``\n\n``), ``<script>``/``<style>``
  bodies excluded.
- tags: ``["substack", "published"]``; ``"stub"`` added when body text is
  < 200 bytes after stripping.
- ``.delivers.csv``/``.opens.csv`` sidecars are NOT emitted as records.
- ``since`` filter on member mtime (exclusive boundary).
- malformed HTML tolerated (parser keeps going).
- unreadable / corrupt zip skipped without polluting the errors sink.
"""
from __future__ import annotations

import os
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from aggregator.sources.base import IngestResult, Record
from aggregator.sources.substack import SubstackSource

# --- fixture builder -------------------------------------------------------


def _build_substack_zip(
    path: Path,
    posts: dict[str, str],
    *,
    include_sidecars: bool = True,
    include_top_level: bool = True,
    member_mtimes: dict[str, tuple[int, int, int, int, int, int]] | None = None,
) -> Path:
    """Write a substack-shaped zip.

    ``posts`` maps ``"<id>.<slug>.html"`` → HTML string.
    ``member_mtimes`` optionally overrides each member's stored date_time.
    """
    default_dt = (2026, 8, 1, 12, 0, 0)
    with zipfile.ZipFile(path, "w") as zf:
        if include_top_level:
            # A real substack export has these at the root; the source
            # must NOT trip on them.
            zi = zipfile.ZipInfo("posts.csv", default_dt)
            zf.writestr(zi, "post_id,title,is_published\n")
            zi = zipfile.ZipInfo("email_list.honest.csv", default_dt)
            zf.writestr(zi, "email,subscribed\n")
        for name, html in posts.items():
            member = f"posts/{name}"
            dt = (member_mtimes or {}).get(member, default_dt)
            zf.writestr(zipfile.ZipInfo(member, dt), html)
            if include_sidecars:
                stem = name.rsplit(".html", 1)[0]
                for sidecar in (f"{stem}.delivers.csv", f"{stem}.opens.csv"):
                    zf.writestr(
                        zipfile.ZipInfo(f"posts/{sidecar}", dt),
                        "email,sent_at\n",
                    )
    return path


@pytest.fixture
def drops(tmp_path, monkeypatch) -> Path:
    d = tmp_path / "drops"
    d.mkdir()
    monkeypatch.setenv("AGGREGATOR_DROPS_DIR", str(d))
    return d


@pytest.fixture
def downloads(tmp_path, monkeypatch) -> Path:
    d = tmp_path / "downloads"
    d.mkdir()
    monkeypatch.setenv("AGGREGATOR_DOWNLOADS_DIR", str(d))
    return d


def _records(
    since: datetime | None = None, errors: list[str] | None = None
) -> list[Record]:
    return list(SubstackSource().iter_records(since, errors=errors))


# -- identity / protocol ----------------------------------------------------


def test_source_name_is_substack():
    assert SubstackSource.name == "substack"


def test_record_shape_documents_fields():
    shape = SubstackSource().record_shape()
    assert "post_id" in shape
    assert "slug" in shape


def test_ingest_returns_ingest_result_counts(drops, downloads):
    _build_substack_zip(
        downloads / "sub.zip",
        {
            "1001.hello-world.html": "<h1>Hello world</h1><p>Body one.</p>",
            "1002.second-post.html": "<h1>Second post</h1><p>Body two.</p>",
        },
    )
    result = SubstackSource().ingest(since=None)
    assert isinstance(result, IngestResult)
    assert result.added == 2
    assert result.errors == []


# -- stable_id --------------------------------------------------------------


def test_stable_id_leading_numeric_wins(drops, downloads):
    _build_substack_zip(
        downloads / "sub.zip",
        {"92823208.second-order-unskillfulness.html": "<h1>Second-order</h1>"},
    )
    (rec,) = _records()
    assert rec.stable_id == "substack:92823208"
    assert rec.extra["post_id"] == "92823208"
    assert rec.extra["slug"] == "second-order-unskillfulness"


def test_stable_id_falls_back_to_full_stem_when_no_leading_digits(drops, downloads):
    _build_substack_zip(
        downloads / "sub.zip",
        {"draft-preview.html": "<h1>Draft preview</h1>"},
    )
    (rec,) = _records()
    assert rec.stable_id == "substack:draft-preview"


# -- subject ---------------------------------------------------------------


def test_subject_prefers_h1_over_slug(drops, downloads):
    _build_substack_zip(
        downloads / "sub.zip",
        {
            "111.slug-part.html":
                "<html><head><title>Ignored title</title></head>"
                "<body><h1>Real Heading</h1><p>...</p></body></html>",
        },
    )
    (rec,) = _records()
    assert rec.subject == "Real Heading"


def test_subject_uses_title_when_no_h1(drops, downloads):
    _build_substack_zip(
        downloads / "sub.zip",
        {
            "222.slug-part.html":
                "<html><head><title>Title Wins</title></head>"
                "<body><p>No h1 here.</p></body></html>",
        },
    )
    (rec,) = _records()
    assert rec.subject == "Title Wins"


def test_subject_falls_back_to_title_cased_slug(drops, downloads):
    _build_substack_zip(
        downloads / "sub.zip",
        {"333.second-order-unskillfulness.html": "<p>text only, no title</p>"},
    )
    (rec,) = _records()
    assert rec.subject == "Second Order Unskillfulness"


def test_subject_title_cased_slug_when_stem_no_digits(drops, downloads):
    _build_substack_zip(
        downloads / "sub.zip",
        {"my-first-post.html": "<p>text</p>"},
    )
    (rec,) = _records()
    assert rec.subject == "My First Post"


# -- body ------------------------------------------------------------------


def test_body_strips_tags_and_preserves_paragraphs(drops, downloads):
    html = (
        "<html><body>"
        "<h1>Ignored in body test</h1>"
        "<p>First paragraph text.</p>"
        "<p>Second paragraph text.</p>"
        "<div>Divider block content.</div>"
        "Line one<br>Line two"
        "</body></html>"
    )
    _build_substack_zip(downloads / "sub.zip", {"444.demo.html": html})
    (rec,) = _records()
    assert "<p>" not in rec.body
    assert "<h1>" not in rec.body
    assert "First paragraph text." in rec.body
    assert "Second paragraph text." in rec.body
    assert "Divider block content." in rec.body
    assert "\n\n" in rec.body
    assert "Line one" in rec.body and "Line two" in rec.body


def test_body_excludes_script_and_style(drops, downloads):
    html = (
        "<html><head>"
        "<style>body { color: red; } .leaked-style { }</style>"
        "</head><body>"
        "<script>var leaked_script = 42;</script>"
        "<p>Kept paragraph.</p>"
        "</body></html>"
    )
    _build_substack_zip(downloads / "sub.zip", {"555.demo.html": html})
    (rec,) = _records()
    assert "leaked_style" not in rec.body
    assert "leaked_script" not in rec.body
    assert "color: red" not in rec.body
    assert "Kept paragraph." in rec.body


# -- tags ------------------------------------------------------------------


def test_tags_always_include_substack_and_published(drops, downloads):
    _build_substack_zip(
        downloads / "sub.zip",
        {"666.regular.html": "<p>" + "x" * 500 + "</p>"},
    )
    (rec,) = _records()
    assert "substack" in rec.tags
    assert "published" in rec.tags
    assert "stub" not in rec.tags


def test_short_body_tagged_stub(drops, downloads):
    _build_substack_zip(
        downloads / "sub.zip",
        {"777.tiny.html": "<p>hi</p>"},  # < 200 bytes stripped
    )
    (rec,) = _records()
    assert "stub" in rec.tags


# -- sidecars are ignored --------------------------------------------------


def test_delivers_and_opens_csv_sidecars_not_emitted(drops, downloads):
    _build_substack_zip(
        downloads / "sub.zip",
        {"888.only-post.html": "<h1>Only</h1><p>Body.</p>"},
        include_sidecars=True,
    )
    recs = _records()
    assert len(recs) == 1
    assert recs[0].stable_id == "substack:888"


# -- since filter ----------------------------------------------------------


def test_since_filter_excludes_older_members(drops, downloads):
    zp = downloads / "sub.zip"
    _build_substack_zip(
        zp,
        {
            "1.old.html": "<h1>Old</h1><p>" + "x" * 300 + "</p>",
            "2.new.html": "<h1>New</h1><p>" + "x" * 300 + "</p>",
        },
        include_sidecars=False,
        member_mtimes={
            "posts/1.old.html": (2026, 1, 1, 0, 0, 0),
            "posts/2.new.html": (2026, 8, 1, 12, 0, 0),
        },
    )
    cutoff = datetime(2026, 6, 1, tzinfo=UTC)
    recs = _records(since=cutoff)
    assert [r.stable_id for r in recs] == ["substack:2"]


def test_since_boundary_is_exclusive(drops, downloads):
    """member mtime == since → skipped (mtime <= since means skip)."""
    zp = downloads / "sub.zip"
    _build_substack_zip(
        zp,
        {"9.exact.html": "<h1>E</h1><p>" + "x" * 300 + "</p>"},
        include_sidecars=False,
        member_mtimes={"posts/9.exact.html": (2026, 7, 1, 12, 0, 0)},
    )
    cutoff = datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC)
    assert _records(since=cutoff) == []


def test_zip_epoch_sentinel_falls_back_to_file_mtime(drops, downloads):
    zp = downloads / "sub.zip"
    _build_substack_zip(
        zp,
        {"1234.sample.html": "<h1>S</h1><p>" + "x" * 300 + "</p>"},
        include_sidecars=False,
        member_mtimes={"posts/1234.sample.html": (1980, 1, 1, 0, 0, 0)},
    )
    known = datetime(2026, 5, 5, 8, 0, 0, tzinfo=UTC)
    os.utime(zp, (known.timestamp(), known.timestamp()))
    (rec,) = _records()
    assert rec.created_at == known
    assert rec.updated_at == known


# -- robustness ------------------------------------------------------------


def test_malformed_html_tolerated(drops, downloads):
    _build_substack_zip(
        downloads / "sub.zip",
        {"321.busted.html": "<h1>Half open<p>Some text<div>never closes"},
    )
    (rec,) = _records()
    assert "Some text" in rec.body


def test_corrupt_zip_skipped_without_crash(drops, downloads):
    (downloads / "broken.zip").write_bytes(b"PK\x03\x04 not a zip")
    errors: list[str] = []
    assert _records(errors=errors) == []
    # Discovery quietly skips unreadable zips (chat-export policy).
    assert errors == []


def test_extra_carries_zip_path_member_and_post_id(drops, downloads):
    zp = _build_substack_zip(
        downloads / "sub.zip",
        {"555.demo-post.html": "<h1>H</h1><p>" + "x" * 300 + "</p>"},
        include_sidecars=False,
    )
    (rec,) = _records()
    assert rec.extra["zip_path"] == str(zp)
    assert rec.extra["member"] == "posts/555.demo-post.html"
    assert rec.extra["post_id"] == "555"
    assert rec.extra["slug"] == "demo-post"


# -- multiple zips ---------------------------------------------------------


def test_two_substack_zips_both_ingested(drops, downloads):
    _build_substack_zip(
        drops / "pub-a.zip",
        {"10.a-post.html": "<h1>A</h1><p>" + "x" * 300 + "</p>"},
        include_sidecars=False,
    )
    _build_substack_zip(
        downloads / "pub-b.zip",
        {"20.b-post.html": "<h1>B</h1><p>" + "x" * 300 + "</p>"},
        include_sidecars=False,
    )
    recs = _records()
    assert {r.stable_id for r in recs} == {"substack:10", "substack:20"}
