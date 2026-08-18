"""The gguf backend must obey the same offline gate as every other model load.

Round 1 made three of the four model-construction paths refuse to reach the
hub unless a human set ``AGGREGATOR_ALLOW_MODEL_DOWNLOAD``: the
sentence-transformers embedder, the cross-encoder reranker, and the seed
path. ``AGGREGATOR_EMBED_BACKEND=gguf`` was the fourth, and it was left
resolving weights with no gate at all — so selecting that backend inside the
MCP server (registered bare, no ``HF_HUB_OFFLINE``) could start a GB-scale
download from a tool that advertises ``openWorldHint=False``.

NOTHING HERE MAY TOUCH THE NETWORK. Both the loader and the hub resolver are
stubbed, and the assertions are about the arguments a resolution is made
with, never about whether a real file appears. A previous round's test named
a real model and pulled 15 GB from the CDN; that is the failure this file is
written to not repeat.
"""

from __future__ import annotations

import sys
import types

import pytest

from aggregator.core import embed as embed_mod


@pytest.fixture
def hub_calls(monkeypatch, tmp_path):
    """Record every weight resolution the gguf path makes, and block all I/O.

    Both plausible resolvers are captured — ``Llama.from_pretrained`` (which
    resolves the repo itself) and ``huggingface_hub.hf_hub_download`` — so the
    assertions describe the OFFLINE PROPERTY rather than one implementation of
    it, and stay meaningful whichever the loader ends up using.
    """
    calls: list[tuple[str, dict]] = []

    class FakeLlama:
        def __init__(self, **kwargs):
            calls.append(("Llama", kwargs))

        @classmethod
        def from_pretrained(cls, **kwargs):
            calls.append(("Llama.from_pretrained", kwargs))
            return cls()

        def embed(self, text):  # pragma: no cover - not exercised here
            return [0.0] * 1024

    fake_llama_cpp = types.ModuleType("llama_cpp")
    fake_llama_cpp.Llama = FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_llama_cpp)

    import huggingface_hub

    def fake_download(**kwargs):
        calls.append(("hf_hub_download", kwargs))
        path = tmp_path / "stub.gguf"
        path.write_bytes(b"")
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    # Belt and braces: even if a stub is somehow bypassed, the process must
    # not be able to fetch anything.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    return calls


def _resolutions(calls):
    """The calls that could reach the hub, i.e. everything but a local ctor."""
    return [(name, kw) for name, kw in calls if name != "Llama"]


def test_gguf_load_is_offline_when_downloads_are_not_allowed(hub_calls, monkeypatch):
    """THE REPRO. No opt-in env var, so no resolution may be network-capable."""
    monkeypatch.delenv(embed_mod.MODEL_DOWNLOAD_ENV, raising=False)

    embed_mod.Embedder(backend="gguf")

    resolutions = _resolutions(hub_calls)
    assert resolutions, f"the gguf path resolved no weights at all: {hub_calls!r}"
    for name, kwargs in resolutions:
        assert kwargs.get("local_files_only") is True, (
            f"{name} was called without an offline guard — the gguf backend "
            f"can reach the hub with no human opt-in. kwargs={kwargs!r}"
        )


def test_gguf_load_may_download_when_a_human_opted_in(hub_calls, monkeypatch):
    """The gate is a gate, not a wall: the seed path still has to work."""
    monkeypatch.setenv(embed_mod.MODEL_DOWNLOAD_ENV, "1")

    embed_mod.Embedder(backend="gguf")

    resolutions = _resolutions(hub_calls)
    assert resolutions, f"the gguf path resolved no weights at all: {hub_calls!r}"
    for name, kwargs in resolutions:
        assert kwargs.get("local_files_only") is False, (
            f"{name} stayed offline even though {embed_mod.MODEL_DOWNLOAD_ENV} "
            f"was set — the seed unit could never fetch gguf weights. "
            f"kwargs={kwargs!r}"
        )


def test_gguf_load_consults_the_one_download_switch(hub_calls, monkeypatch):
    """One switch for every model load, asked on this path too."""
    consulted = []
    real = embed_mod.downloads_allowed

    def counting():
        consulted.append(True)
        return real()

    monkeypatch.setattr(embed_mod, "downloads_allowed", counting)
    monkeypatch.delenv(embed_mod.MODEL_DOWNLOAD_ENV, raising=False)

    embed_mod.Embedder(backend="gguf")

    assert consulted, (
        "downloads_allowed() was never consulted by the gguf backend: the "
        "offline enforcement does not cover this model-construction path"
    )
