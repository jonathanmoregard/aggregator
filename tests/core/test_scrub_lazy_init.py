"""Presidio initialisation is lazy, happens exactly once, and never races.

WHY THIS IS NOT A CODE COMMENT. ``aggregator/core/scrub.py`` used to build
``AnalyzerEngine()`` and ``AnonymizerEngine()`` at module import. Importing the
module therefore imported ``presidio_analyzer`` (which pulls ``transformers``,
which pulls ``torch``) and ``spacy`` (which pulls ``thinc``, which pulls
``torch`` again): 49.0 s of the 57.6 s it took to import ``aggregator.mcp`` on
the user's laptop, measured with ``PYTHONPROFILEIMPORTTIME``. Claude Code gives
an MCP server 30 s to answer ``initialize``, so from 2026-09-04 every session
timed out and the recall server was simply absent.

Moving that work behind a lazy initialiser is only correct if the initialiser is
(a) run exactly once however many callers arrive at once, (b) blocking, so a
caller never sees a half-built engine, and (c) failure-tolerant in the same way
the old module-scope ``except (Exception, SystemExit)`` was. Each of those is a
test below.

NOTHING HERE TOUCHES A REAL ENGINE. Every test monkeypatches
``_build_presidio_engines``, which is the single function the heavy imports live
in. No model is downloaded and no network call is made, on any machine.
"""
import threading
import time
from types import SimpleNamespace

import pytest


class _FakeAnalyzer:
    """Stands in for ``presidio_analyzer.AnalyzerEngine``.

    Records how often it was asked to analyse, and always reports one US_SSN, so
    a test can tell "the Presidio branch ran" from "the regex branch ran".
    """

    def __init__(self):
        self.calls = 0

    def analyze(self, **_kwargs):
        self.calls += 1
        return [SimpleNamespace(entity_type="US_SSN")]


class _FakeAnonymizer:
    """Stands in for ``presidio_anonymizer.AnonymizerEngine``."""

    def anonymize(self, text, analyzer_results):
        return SimpleNamespace(text=f"{text} <anonymized:{len(analyzer_results)}>")


def _stub_builder(calls, delay=0.0):
    """A ``_build_presidio_engines`` replacement that counts and (optionally) stalls."""

    def _build():
        calls.append(1)
        if delay:
            time.sleep(delay)
        return _FakeAnalyzer(), _FakeAnonymizer()

    return _build


def test_importing_scrub_builds_no_engines(fresh_scrub_state):
    """Import must decide nothing. The whole point of the change."""
    mod = fresh_scrub_state
    assert mod._INIT_DONE.is_set() is False
    assert mod._analyzer is None
    assert mod._anonymizer is None


def test_the_initialiser_runs_exactly_once_under_concurrent_callers(
    fresh_scrub_state, monkeypatch
):
    """Eight threads, one engine build.

    A double-checked lock that checks the wrong thing builds the engines twice
    and leaves whichever finished last installed. The barrier makes all eight
    callers arrive inside the same instant, which is the only arrangement that
    can catch it.
    """
    mod = fresh_scrub_state
    calls: list[int] = []
    monkeypatch.setattr(mod, "_build_presidio_engines", _stub_builder(calls, delay=0.05))

    results: list[bool] = []
    results_lock = threading.Lock()
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait(timeout=10)
        ready = mod.ensure_presidio_ready()
        with results_lock:
            results.append(ready)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not any(t.is_alive() for t in threads), "a caller never returned"
    assert len(calls) == 1, f"engines built {len(calls)} times, expected exactly 1"
    assert results == [True] * 8


def test_a_second_call_does_not_rebuild(fresh_scrub_state, monkeypatch):
    """Idempotent in the boring sequential case too."""
    mod = fresh_scrub_state
    calls: list[int] = []
    monkeypatch.setattr(mod, "_build_presidio_engines", _stub_builder(calls))

    assert mod.ensure_presidio_ready() is True
    assert mod.ensure_presidio_ready() is True
    assert len(calls) == 1


def test_init_failure_falls_back_to_regex_and_still_scrubs(
    fresh_scrub_state, monkeypatch, caplog
):
    """A machine without the model degrades; it does not raise and does not stop
    scrubbing. The warning text is pinned because it is the user-facing
    explanation of why PII coverage narrowed."""
    mod = fresh_scrub_state

    def _boom():
        raise RuntimeError("no model here")

    monkeypatch.setattr(mod, "_build_presidio_engines", _boom)

    with caplog.at_level("WARNING", logger="aggregator.core.scrub"):
        assert mod.ensure_presidio_ready() is False

    assert mod._PRESIDIO_OK is False
    assert mod._analyzer is None
    assert mod._anonymizer is None
    assert "PII scrubbing will use regex fallback only" in caplog.text

    result = mod.scrub("write to bob@example.com and note 123-45-6789")
    assert "bob@example.com" not in result.text
    assert "123-45-6789" not in result.text
    assert result.counts.get("email", 0) >= 1
    assert result.counts.get("ssn", 0) >= 1


def test_init_failure_is_logged_once_not_once_per_scrub(
    fresh_scrub_state, monkeypatch, caplog
):
    """The old code decided at import, so the warning appeared once. A lazy
    initialiser that retried would print it on every stored row — 372k times on a
    full ingest — which is how a log stops being read."""
    mod = fresh_scrub_state

    def _boom():
        raise RuntimeError("no model here")

    monkeypatch.setattr(mod, "_build_presidio_engines", _boom)

    with caplog.at_level("WARNING", logger="aggregator.core.scrub"):
        for _ in range(5):
            mod.scrub("hello")

    assert caplog.text.count("PII scrubbing will use regex fallback only") == 1


def test_systemexit_from_engine_construction_is_caught(fresh_scrub_state, monkeypatch):
    """``spacy.cli.download`` calls ``sys.exit(1)`` on failure, and ``SystemExit``
    is a ``BaseException`` that ``except Exception`` does not catch. That took CI
    down with ``INTERNALERROR> SystemExit: 1`` in 2026-08. The widened clause has
    to survive the move into the initialiser."""
    mod = fresh_scrub_state

    def _exit_like_spacy_download():
        raise SystemExit(1)

    monkeypatch.setattr(mod, "_build_presidio_engines", _exit_like_spacy_download)

    assert mod.ensure_presidio_ready() is False
    assert mod.scrub("write to bob@example.com").counts.get("email", 0) >= 1


def test_the_init_event_is_set_even_when_the_builder_raises_baseexception(
    fresh_scrub_state, monkeypatch
):
    """The ``finally`` is the deadlock guard, not decoration.

    ``except (Exception, SystemExit)`` does not cover every ``BaseException``. If
    one escapes and the event is never set, every other thread — including the
    MCP server's request threads — waits on the lock forever.
    """
    mod = fresh_scrub_state

    class _Weird(BaseException):
        pass

    def _boom():
        raise _Weird("not an Exception, not a SystemExit")

    monkeypatch.setattr(mod, "_build_presidio_engines", _boom)

    with pytest.raises(_Weird):
        mod.ensure_presidio_ready()

    assert mod._INIT_DONE.is_set() is True
    assert mod._PRESIDIO_OK is False
    # And nothing is wedged: the next caller returns immediately.
    assert mod.ensure_presidio_ready() is False


def test_background_warmup_then_scrub_waits_and_uses_presidio(
    fresh_scrub_state, monkeypatch
):
    """The MCP arrangement, end to end: warm up on a thread, then scrub while the
    build is still running. ``scrub`` must BLOCK on the half-built engine rather
    than skip Presidio for that one call."""
    mod = fresh_scrub_state
    calls: list[int] = []
    monkeypatch.setattr(mod, "_build_presidio_engines", _stub_builder(calls, delay=0.3))

    thread = mod.start_background_init()
    assert thread is not None
    assert thread.daemon is True

    started = time.monotonic()
    result = mod.scrub("hello")
    elapsed = time.monotonic() - started

    thread.join(timeout=30)
    assert not thread.is_alive()

    assert len(calls) == 1, "the warm-up and the scrub must not both build"
    assert elapsed >= 0.2, "scrub returned before the engines existed"
    assert result.text.endswith("<anonymized:1>")
    assert result.counts.get("presidio_us_ssn", 0) == 1
    assert mod._analyzer.calls == 1


def test_start_background_init_is_idempotent(fresh_scrub_state, monkeypatch):
    """Two calls, one thread. And once initialisation is done it starts nothing."""
    mod = fresh_scrub_state
    calls: list[int] = []
    monkeypatch.setattr(mod, "_build_presidio_engines", _stub_builder(calls, delay=0.2))

    first = mod.start_background_init()
    second = mod.start_background_init()
    assert first is second

    first.join(timeout=30)
    assert not first.is_alive()
    assert len(calls) == 1

    assert mod.start_background_init() is None
    assert len(calls) == 1
