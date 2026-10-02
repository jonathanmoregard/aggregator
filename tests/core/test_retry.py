"""The shared transient-retry helper: retry what may heal, raise what won't."""

from __future__ import annotations

import pytest

from aggregator.core import retry


class TransientError(Exception):
    pass


class PermanentError(Exception):
    pass


def _flaky(failures: list[Exception], value="ok"):
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if failures:
            raise failures.pop(0)
        return value

    return fn, calls


def _is_transient(exc: BaseException) -> bool:
    return isinstance(exc, TransientError)


def test_returns_after_transient_failures_heal(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(retry, "_sleep", slept.append)
    fn, calls = _flaky([TransientError("500"), TransientError("500")])
    assert retry.call_with_retry(fn, is_transient=_is_transient, what="x") == "ok"
    assert calls["n"] == 3
    assert slept == list(retry.DEFAULT_DELAYS)


def test_permanent_failure_raises_at_once(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(retry, "_sleep", slept.append)
    fn, calls = _flaky([PermanentError("401")])
    with pytest.raises(PermanentError):
        retry.call_with_retry(fn, is_transient=_is_transient, what="x")
    assert calls["n"] == 1
    assert slept == []


def test_persistent_transient_failure_raises_the_last_error(monkeypatch):
    monkeypatch.setattr(retry, "_sleep", lambda _s: None)
    errs = [TransientError(str(i)) for i in range(len(retry.DEFAULT_DELAYS) + 1)]
    last = errs[-1]
    fn, calls = _flaky(list(errs))
    with pytest.raises(TransientError) as excinfo:
        retry.call_with_retry(fn, is_transient=_is_transient, what="x")
    assert excinfo.value is last
    assert calls["n"] == len(retry.DEFAULT_DELAYS) + 1


def test_each_retry_is_logged(monkeypatch, caplog):
    monkeypatch.setattr(retry, "_sleep", lambda _s: None)
    fn, _ = _flaky([TransientError("boom")])
    with caplog.at_level("WARNING", logger="aggregator.core.retry"):
        retry.call_with_retry(fn, is_transient=_is_transient, what="thing GET /x")
    assert "thing GET /x" in caplog.text
    assert "boom" in caplog.text
