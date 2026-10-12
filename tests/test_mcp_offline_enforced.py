""""The MCP path never downloads model weights" is enforced, not claimed.

The MCP server inherits the environment of whatever launched it, and
``downloads_allowed()`` reads ``os.environ`` and nothing else. This project's
own remediation text tells operators to export the opt-in
(``AGGREGATOR_ALLOW_MODEL_DOWNLOAD=1 aggregator seed-reranker``); export it
once in a shell that later starts the server and a single ``rerank=True``
query would fetch ~1.2 GB from the hub, inside the server's process, from a
tool whose annotations declare ``openWorldHint=False``. These tests pin that
the reranker construction is offline regardless of the inherited environment.

No real model is named anywhere here. The constructor is stubbed, and what is
asserted is the value ``downloads_allowed()`` reports AT CONSTRUCTION TIME,
which is the only moment the loader consults it.
"""

from __future__ import annotations

import pytest

import aggregator.core.rerank as rerank_mod
from aggregator import mcp


@pytest.fixture(autouse=True)
def _cold_singletons(monkeypatch):
    """``_get_reranker`` memoises; a warm one builds nothing."""
    monkeypatch.setattr(mcp, "_reranker", None)


class _RecordingModel:
    """Stands in for a loader and records the one thing that matters."""

    seen: list[bool] = []

    def __init__(self, *args, **kwargs):
        type(self).seen.append(rerank_mod.downloads_allowed())


@pytest.fixture
def recorder(monkeypatch):
    class _Stub(_RecordingModel):
        seen: list[bool] = []

    monkeypatch.setattr(rerank_mod, "Reranker", _Stub)
    return _Stub


def test_the_reranker_is_built_offline_despite_the_inherited_environment(
    monkeypatch, recorder
):
    """THE REPRO for the ~1.2 GB one: a single ``rerank=True`` call."""
    monkeypatch.setenv(rerank_mod.MODEL_DOWNLOAD_ENV, "1")

    mcp._get_reranker()

    assert recorder.seen == [False], (
        f"the cross-encoder was constructed with hub downloads permitted "
        f"({recorder.seen}); one rerank=True query would fetch ~1.2 GB inside "
        f"the editor's MCP process"
    )


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", " 1 "])
def test_every_spelling_of_the_opt_in_is_denied(monkeypatch, recorder, value):
    """Enforced by removing the variable, not by matching its value — so this
    cannot drift out of agreement with ``downloads_allowed``'s parsing."""
    monkeypatch.setenv(rerank_mod.MODEL_DOWNLOAD_ENV, value)
    mcp._get_reranker()
    assert recorder.seen == [False]


def test_the_denial_is_scoped_to_the_construction(monkeypatch, recorder):
    """The seeding path must keep working in the same interpreter.

    ``aggregator seed-reranker`` is the ONE sanctioned downloader and it
    builds its own ``Reranker`` directly. Clearing the variable
    process-wide would break the only supported way to obtain the weights, so
    the denial covers the construction and hands the environment back.
    """
    monkeypatch.setenv(rerank_mod.MODEL_DOWNLOAD_ENV, "1")

    mcp._get_reranker()

    assert rerank_mod.downloads_allowed() is True, (
        "the MCP path cleared the opt-in for the whole process; "
        "`aggregator seed-reranker` would then download nothing"
    )


def test_an_absent_opt_in_is_left_absent(monkeypatch, recorder):
    """No spurious variable is introduced on the way out."""
    monkeypatch.delenv(rerank_mod.MODEL_DOWNLOAD_ENV, raising=False)

    mcp._get_reranker()

    assert recorder.seen == [False]
    assert rerank_mod.MODEL_DOWNLOAD_ENV not in __import__("os").environ


def test_a_loader_that_raises_still_restores_the_environment(monkeypatch):
    """A model load that fails is the COMMON case on an unseeded machine."""
    monkeypatch.setenv(rerank_mod.MODEL_DOWNLOAD_ENV, "1")

    def _no_weights(*args, **kwargs):
        raise OSError("not in the local cache and local_files_only=True")

    monkeypatch.setattr(rerank_mod, "Reranker", _no_weights)

    with pytest.raises(OSError):
        mcp._get_reranker()

    assert rerank_mod.downloads_allowed() is True, (
        "a failed model load left the opt-in stripped from the environment"
    )
