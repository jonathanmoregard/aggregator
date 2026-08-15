"""pypdf's chatter must never be mistaken for this pipeline's progress.

THE INCIDENT. On 2026-08-15 the first live timer run printed 102 lines of
``Ignoring wrong pointing object`` from pypdf and then went silent for two
hours. The silence was read as "pypdf is hung on a PDF" and cost the
investigation two hours and a wrong hypothesis (a per-file PDF timeout) before
the real fault — a missing watermark, so every run re-ingested 372k
observations — was found. The dropbox leg had in fact finished at 22:17; the
noise was simply the only thing talking, so its absence looked like a hang.

The rule that follows: a library's debug output must not be the loudest voice
in an unattended run, because then its absence reads as failure and its
presence reads as progress, and neither is true.
"""
from __future__ import annotations

import logging


def test_pypdf_logger_is_quieted_by_importing_textextract():
    """Importing the extractor is what silences pypdf, not the CLI.

    Set on IMPORT of the module that owns the pypdf call, so every entry point
    inherits it: the timer's ``ingest --all``, a hand-run ``ingest dropbox``,
    the MCP server, and a test. A ``logging.basicConfig`` in ``cli.main`` would
    have covered exactly one of those.
    """
    import aggregator.core.textextract  # noqa: F401 -- import IS the assertion

    assert logging.getLogger("pypdf").level == logging.ERROR


def test_pypdf_warnings_are_suppressed_but_errors_still_reach_a_handler():
    """ERROR, not CRITICAL and not disabled: a real pypdf failure stays loud.

    ``Ignoring wrong pointing object`` is emitted at WARNING and is per-object,
    so a single malformed PDF produces dozens of lines that say nothing a human
    can act on. An actual parse failure is already surfaced by
    ``ExtractionError`` into the run's ``errors`` list, but anything pypdf logs
    at ERROR is a second channel worth keeping.
    """
    import aggregator.core.textextract  # noqa: F401

    log = logging.getLogger("pypdf")
    assert not log.isEnabledFor(logging.WARNING)
    assert log.isEnabledFor(logging.ERROR)
