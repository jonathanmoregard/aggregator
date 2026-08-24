""""The MCP path never sets AGGREGATOR_ALLOW_MODEL_DOWNLOAD" was a claim.

It was written in the module docstring as one of the three things that bound
the accepted torch-in-an-unsandboxed-process exposure, and it was not enforced
anywhere. The MCP server is a **stdio child of the editor**: it inherits the
environment of whatever shell launched the editor. ``downloads_allowed()``
reads ``os.environ`` and nothing else, and nothing in the MCP process cleared,
overrode or even looked at it.

So the property held only as long as no one exported the variable — and this
project's own remediation text tells operators to do exactly that
(``AGGREGATOR_ALLOW_MODEL_DOWNLOAD=1 aggregator embed --seed-models``). Export
it once in a shell that later starts the editor and a single ``rerank=True``
query fetches ~1.2 GB from the hub, inside the editor's own process, from a
tool whose annotations declare ``openWorldHint=False``.

A property that depends on the user's shell is not a property. These tests pin
it as code: the two model constructions this module performs are made offline
regardless of the inherited environment.

No real model is named anywhere here — an earlier round did that in a test and
pulled 15 GB off a CDN. The constructors are stubbed, and what is asserted is
the value ``downloads_allowed()`` reports AT CONSTRUCTION TIME, which is the
only moment either loader consults it (``Embedder.__init__`` and
``Reranker.__init__`` both resolve their weights eagerly).
"""

from __future__ import annotations

import pytest

import aggregator.core.embed as embed_mod
import aggregator.core.rerank as rerank_mod
from aggregator import mcp


@pytest.fixture(autouse=True)
def _cold_singletons(monkeypatch):
    """``_get_embedder``/``_get_reranker`` memoise; a warm one builds nothing."""
    monkeypatch.setattr(mcp, "_embedder", None)
    monkeypatch.setattr(mcp, "_reranker", None)


class _RecordingModel:
    """Stands in for a loader and records the one thing that matters."""

    seen: list[bool] = []

    def __init__(self, *args, **kwargs):
        type(self).seen.append(embed_mod.downloads_allowed())


@pytest.fixture
def recorder(monkeypatch):
    class _Stub(_RecordingModel):
        seen: list[bool] = []

    monkeypatch.setattr(embed_mod, "Embedder", _Stub)
    monkeypatch.setattr(rerank_mod, "Reranker", _Stub)
    return _Stub


def test_the_embedder_is_built_offline_despite_the_inherited_environment(
    monkeypatch, recorder
):
    """THE REPRO for the vector arm: one text query, once the index is warm."""
    monkeypatch.setenv(embed_mod.MODEL_DOWNLOAD_ENV, "1")

    mcp._get_embedder()

    assert recorder.seen == [False], (
        f"the query embedder was constructed with hub downloads permitted "
        f"({recorder.seen}) because the editor's shell exported "
        f"{embed_mod.MODEL_DOWNLOAD_ENV}=1"
    )


def test_the_reranker_is_built_offline_despite_the_inherited_environment(
    monkeypatch, recorder
):
    """THE REPRO for the ~1.2 GB one: a single ``rerank=True`` call."""
    monkeypatch.setenv(embed_mod.MODEL_DOWNLOAD_ENV, "1")

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
    monkeypatch.setenv(embed_mod.MODEL_DOWNLOAD_ENV, value)
    mcp._get_embedder()
    assert recorder.seen == [False]


def test_the_denial_is_scoped_to_the_construction(monkeypatch, recorder):
    """The seeding path must keep working in the same interpreter.

    ``aggregator embed --seed-models`` is the ONE sanctioned downloader and it
    builds its own ``Embedder``/``Reranker`` directly. Clearing the variable
    process-wide would break the only supported way to obtain the weights, so
    the denial covers the construction and hands the environment back.
    """
    monkeypatch.setenv(embed_mod.MODEL_DOWNLOAD_ENV, "1")

    mcp._get_embedder()

    assert embed_mod.downloads_allowed() is True, (
        "the MCP path cleared the opt-in for the whole process; "
        "`aggregator embed --seed-models` would then download nothing"
    )


def test_an_absent_opt_in_is_left_absent(monkeypatch, recorder):
    """No spurious variable is introduced on the way out."""
    monkeypatch.delenv(embed_mod.MODEL_DOWNLOAD_ENV, raising=False)

    mcp._get_embedder()

    assert recorder.seen == [False]
    assert embed_mod.MODEL_DOWNLOAD_ENV not in __import__("os").environ


def test_a_loader_that_raises_still_restores_the_environment(monkeypatch):
    """A model load that fails is the COMMON case on an unseeded machine."""
    monkeypatch.setenv(embed_mod.MODEL_DOWNLOAD_ENV, "1")

    def _no_weights(*args, **kwargs):
        raise OSError("not in the local cache and local_files_only=True")

    monkeypatch.setattr(embed_mod, "Embedder", _no_weights)

    with pytest.raises(OSError):
        mcp._get_embedder()

    assert embed_mod.downloads_allowed() is True, (
        "a failed model load left the opt-in stripped from the environment"
    )
