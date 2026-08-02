"""Substack export source (Chunk 5, chat-exports plan 2026-08-02).

Ingests Substack data-export drops (Settings → Exports → zip) as
**records-shaped** entities (one Record per post — units-of-work ontology,
like GitHub PRs / issues and the sibling ``research`` + ``sota-watch``
sources). Not a session/observation stream — a substack post is a finished
article, not a conversation.

Discovery: :mod:`aggregator.sources.exportdrops` classifies substack
zips by presence of any ``posts/*.html`` member (glob semantics — the
top-level ``posts.csv`` does not trip the check). Bare files are out of
scope for v1 (substack ships zip only). Zips are read via :mod:`zipfile`
without extraction; per-record ``extra`` carries both the zip path and
the internal member path so downstream tooling can round-trip to the
source.

Per post → one Record:

* ``stable_id`` = ``substack:<post-id>`` where post-id is the leading
  numeric portion of the filename stem
  (``92823208.second-order-unskillfulness.html`` → ``92823208``). If no
  leading digits, falls back to ``substack:<full stem>``.
* ``subject`` = ``<h1>`` (preferred) or ``<title>`` (fallback), stripped
  of tags. If neither present, derive a title-cased subject from the
  slug portion (``second-order-unskillfulness`` → ``Second Order
  Unskillfulness``) — matches how substack renders titles from slugs.
* ``body`` = HTML text content (tags stripped). Paragraph breaks are
  preserved: ``</p>``, ``</div>``, and ``<br>`` all emit ``\\n\\n``.
  ``<script>`` and ``<style>`` bodies are excluded entirely — those are
  not content, and their bodies would only pollute FTS.
* ``tags`` = ``["substack", "published"]``. All posts in a substack export
  are published (drafts don't ship). When the stripped body is under 200
  bytes, ``"stub"`` is appended — the vendor export leaves email-only
  micro-posts as tiny stubs and it's useful to filter them out at query
  time.
* ``created_at`` / ``updated_at`` = the zip MEMBER's mtime, converted to
  UTC. Zip epoch sentinel (1980-01-01) falls back to the zip FILE's mtime
  so we get a plausible timestamp even from tools that don't populate
  member dates.

Skipped: ``posts/*.delivers.csv`` and ``posts/*.opens.csv`` sidecars are
per-post email stats, not content. Top-level ``posts.csv`` and
``email_list.<pub>.csv`` are similarly ignored — only ``posts/*.html``
becomes a Record.

Robustness: malformed HTML → :class:`html.parser.HTMLParser` keeps going
(it tolerates most breakage). Unreadable zips are silently dropped by
the shared discovery helper (chat-export policy — Downloads legitimately
holds broken archives). Per-post errors are appended to the ``errors``
sink and the walk continues (partial ingest beats total loss).

Deps: stdlib only — :mod:`html.parser`. NOT BeautifulSoup.
"""
from __future__ import annotations

import logging
import os
import re
import zipfile
from collections.abc import Iterator
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

from aggregator.sources.base import IngestResult, Record, stable_id_for
from aggregator.sources.exportdrops import (
    DEFAULT_DROPS_DIR,
    ExportFile,
    discover_export_files,
    downloads_dir,
)

log = logging.getLogger(__name__)

# Bytes threshold below which we tag a post as "stub" (email-only
# micro-posts left as tiny placeholders by the vendor export).
STUB_BODY_BYTES = 200

# Post-id leading digits pattern; used to split ``92823208.slug.html`` into
# the numeric id + the slug portion.
_LEADING_DIGITS = re.compile(r"^(\d+)(?:[.-](.*))?$")

# Zip epoch sentinel: many zips get 1980-01-01T00:00:00 when the writer
# didn't populate the member date_time. We treat that as "unknown" and
# fall back to the zip file's mtime.
_ZIP_EPOCH = datetime(1980, 1, 1, tzinfo=UTC)


class _TextExtractor(HTMLParser):
    """Accumulate visible text out of a substack post HTML file.

    Paragraph-breakers (``</p>``, ``</div>``, ``<br>``) inject ``\\n\\n``
    so body text stays legible in FTS snippets. ``<script>`` and
    ``<style>`` bodies are dropped (a boolean gate tracks depth).

    Also captures the first ``<h1>`` inner text (``h1_text``) and the
    first ``<title>`` inner text (``title_text``) so the source can pick
    a subject without a second parser pass.
    """

    _PARA_BREAKERS = frozenset({"p", "div", "br", "li", "h1", "h2", "h3"})
    _SKIP_TAGS = frozenset({"script", "style"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._buf: list[str] = []
        self._skip_depth = 0
        # Subject capture state.
        self._in_h1 = False
        self._in_title = False
        self.h1_text: str | None = None
        self.title_text: str | None = None
        self._h1_buf: list[str] = []
        self._title_buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "br":
            self._buf.append("\n\n")
            return
        if tag == "h1" and self.h1_text is None:
            self._in_h1 = True
        elif tag == "title" and self.title_text is None:
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if tag in self._PARA_BREAKERS:
            self._buf.append("\n\n")
        if tag == "h1" and self._in_h1:
            self._in_h1 = False
            joined = "".join(self._h1_buf).strip()
            if joined:
                self.h1_text = joined
        elif tag == "title" and self._in_title:
            self._in_title = False
            joined = "".join(self._title_buf).strip()
            if joined:
                self.title_text = joined

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        # <br/> is common; treat as start (paragraph breaker) with no end.
        if tag == "br":
            self._buf.append("\n\n")
            return
        # <img/> etc: nothing to do.

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        self._buf.append(data)
        if self._in_h1:
            self._h1_buf.append(data)
        if self._in_title:
            self._title_buf.append(data)

    def get_text(self) -> str:
        """Return the accumulated body text with normalised whitespace.

        Collapses runs of blank lines into a single ``\\n\\n`` so the body
        reads cleanly regardless of the vendor's markup density.
        """
        raw = "".join(self._buf)
        # Collapse runs of >= 2 blank lines to exactly one paragraph break.
        text = re.sub(r"\n{3,}", "\n\n", raw)
        # Strip trailing whitespace on each line, then trim edges.
        text = "\n".join(line.rstrip() for line in text.splitlines())
        return text.strip()


def _split_stem(stem: str) -> tuple[str, str]:
    """Split ``92823208.second-order-unskillfulness`` → ("92823208", slug).

    Falls back to ``(stem, "")`` when the stem has no leading digits.
    Handles the occasional ``<id>.<slug>.<n>`` collision suffix by
    keeping everything after the first ``.`` as the slug portion.
    """
    m = _LEADING_DIGITS.match(stem)
    if m and m.group(1):
        return m.group(1), (m.group(2) or "")
    return stem, ""


def _slug_to_title(slug: str) -> str:
    """``second-order-unskillfulness`` → ``Second Order Unskillfulness``.

    Empty slug returns empty string; the caller decides what to substitute.
    """
    if not slug:
        return ""
    return " ".join(part.capitalize() for part in slug.split("-") if part)


def _member_mtime(info: zipfile.ZipInfo, zip_path: Path) -> datetime:
    """Return the member's mtime as an aware UTC datetime.

    Zips store member dates as a 6-tuple with 2-second precision and no
    timezone. We interpret them as UTC (vendor exports are produced by
    servers running UTC; the alternative — local time — makes cross-host
    ingest unstable). When the member carries the zip epoch sentinel
    (1980-01-01 00:00:00 — the ``ZipInfo`` default when the writer didn't
    populate it), we fall back to the zip FILE's mtime so the record has a
    plausible timestamp.
    """
    try:
        dt = datetime(*info.date_time, tzinfo=UTC)
    except (TypeError, ValueError):
        dt = _ZIP_EPOCH
    if dt == _ZIP_EPOCH:
        try:
            return datetime.fromtimestamp(zip_path.stat().st_mtime, tz=UTC)
        except OSError:
            return _ZIP_EPOCH
    return dt


class SubstackSource:
    """Records-shaped source over Substack data-export zips."""

    name = "substack"

    def __init__(self, drops_dir: str | None = None):
        raw = drops_dir or os.environ.get("AGGREGATOR_DROPS_DIR") or DEFAULT_DROPS_DIR
        self.drops_dir = Path(raw).expanduser()

    def record_shape(self) -> dict[str, str]:
        """DSL-facing field surface (M2 help generator)."""
        return {
            "subject": "str (h1 or <title>, or title-cased slug)",
            "body": "str (post HTML with tags stripped)",
            "post_id": "str (leading digits of filename stem, or full stem)",
            "slug": "str (slug portion of filename after post id)",
            "zip_path": "str (containing zip on disk)",
            "member": "str (path inside zip: posts/<id>.<slug>.html)",
        }

    # -- discovery ---------------------------------------------------------

    def _iter_export_files(self, errors: list[str]) -> Iterator[ExportFile]:
        yield from discover_export_files(
            "substack", dirs=[self.drops_dir, downloads_dir()], errors=errors
        )

    # -- iteration ---------------------------------------------------------

    def iter_records(
        self,
        since: datetime | None = None,
        errors: list[str] | None = None,
    ) -> Iterator[Record]:
        """Yield one Record per ``posts/*.html`` member across all substack
        zips discovered in drops dir + Downloads.

        ``since``: skip members whose zip-recorded mtime is <= since
        (exclusive boundary, matching the sibling records sources). Naive
        ``since`` is treated as UTC.
        """
        sink = errors if errors is not None else []
        since_utc: datetime | None = None
        if since is not None:
            since_utc = since if since.tzinfo is not None else since.replace(tzinfo=UTC)

        for ef in self._iter_export_files(sink):
            zip_path = ef.path
            try:
                zf = zipfile.ZipFile(zip_path)
            except (OSError, zipfile.BadZipFile) as e:
                sink.append(f"{zip_path}: unreadable zip: {e}")
                continue
            try:
                for info in sorted(zf.infolist(), key=lambda i: i.filename):
                    member = info.filename
                    if not member.startswith("posts/"):
                        continue
                    if not member.endswith(".html"):
                        continue
                    # Basename check — defensive against zips that nest
                    # deeper (posts/sub/x.html would fail the posts/*.html
                    # glob semantics we announce).
                    if "/" in member[len("posts/"):]:
                        continue
                    mtime = _member_mtime(info, zip_path)
                    if since_utc is not None and mtime <= since_utc:
                        continue
                    try:
                        raw = zf.read(info)
                    except (OSError, zipfile.BadZipFile, KeyError) as e:
                        sink.append(f"{zip_path}!{member}: read failed: {e}")
                        continue
                    try:
                        rec = self._html_to_record(
                            zip_path=zip_path,
                            member=member,
                            html_bytes=raw,
                            mtime=mtime,
                        )
                    except Exception as e:  # noqa: BLE001 -- degrade gracefully
                        log.warning(
                            "substack: parse failed for %s!%s: %s",
                            zip_path, member, e,
                        )
                        sink.append(f"{zip_path}!{member}: parse failed: {e}")
                        continue
                    yield rec
            finally:
                zf.close()

    def _html_to_record(
        self,
        *,
        zip_path: Path,
        member: str,
        html_bytes: bytes,
        mtime: datetime,
    ) -> Record:
        # Filename → post-id / slug. The basename shape is documented as
        # ``<post-id>.<slug>.html`` in the substack export; occasionally
        # ``<post-id>.<slug>.<n>.html`` if the writer added a collision
        # suffix. Stripping the ``.html`` suffix + splitting on the first
        # ``.`` covers both.
        basename = member.rsplit("/", 1)[-1]
        stem = basename[:-5] if basename.endswith(".html") else basename
        post_id, slug = _split_stem(stem)

        # HTML → text via stdlib parser.
        extractor = _TextExtractor()
        try:
            extractor.feed(html_bytes.decode("utf-8", errors="replace"))
            extractor.close()
        except Exception:  # noqa: BLE001 -- HTMLParser sometimes raises
            # HTMLParser occasionally raises on truly pathological input;
            # we still want a record with whatever we managed to collect.
            log.warning("substack: HTMLParser raised on %s", member)
        body = extractor.get_text()

        # Subject: h1 → title → slug title-case (falls back to a
        # title-cased stem when no leading numeric id was present, so
        # ``my-first-post.html`` → ``My First Post`` rather than the raw
        # stem).
        subject = extractor.h1_text or extractor.title_text or ""
        if not subject:
            subject = _slug_to_title(slug) or _slug_to_title(stem) or stem

        tags = ["substack", "published"]
        if len(body.encode("utf-8")) < STUB_BODY_BYTES:
            tags.append("stub")

        stable_id_source_id = post_id if post_id else stem
        extra = {
            "post_id": post_id,
            "slug": slug,
            "zip_path": str(zip_path),
            "member": member,
        }
        return Record(
            stable_id=stable_id_for(self.name, stable_id_source_id),
            source=self.name,
            subject=subject,
            body=body,
            tags=tags,
            created_at=mtime,
            updated_at=mtime,
            extra=extra,
        )

    def ingest(self, since: datetime | None) -> IngestResult:
        """Count-only path retained for protocol compat; persistence is the
        CLI's job (same split as the sibling record sources)."""
        errors: list[str] = []
        added = sum(1 for _ in self.iter_records(since, errors=errors))
        return IngestResult(added=added, updated=0, skipped=0, errors=errors)
