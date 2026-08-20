"""The gguf backend must obey the same offline gate as every other model load.

Round 1 made three of the four model-construction paths refuse to reach the
hub unless a human set ``AGGREGATOR_ALLOW_MODEL_DOWNLOAD``: the
sentence-transformers embedder, the cross-encoder reranker, and the seed
path. ``AGGREGATOR_EMBED_BACKEND=gguf`` was the fourth, and it was left
resolving weights with no gate at all — so selecting that backend inside the
MCP server (registered bare, no ``HF_HUB_OFFLINE``) could start a GB-scale
download from a tool that advertises ``openWorldHint=False``.

ROUND 3 NARROWED THE GATE INTO A WALL, DELIBERATELY, AND ONLY FOR NOW. Round 2
left this path resolving the ``-GGUF`` repo at ``main`` because it passed no
``revision=`` — a moving artifact under a deployment whose whole rule is that a
rev-pinned unit executes fixed bytes. ``QWEN3_EMBEDDING_REVISION`` cannot be
reused for it: that sha was read off the *safetensors* repo and is not a valid
ref in the separate ``-GGUF`` one, and no sha for the latter has been verified
yet. So ``QWEN3_EMBEDDING_GGUF_REVISION`` is ``None`` and an opted-in human is
REFUSED rather than handed an unpinned download.

That inverts what round 2 asserted here, so it is written down rather than left
as a test nobody could satisfy. The refusal is scoped as tightly as the problem:
it applies to the DOWNLOAD path only — loading an already-seeded cache still
works, since those bytes are on disk and not moving — and it lifts the moment
the constant holds a sha, which
``test_the_gate_reopens_once_a_revision_is_pinned`` proves. Round 2's actual
intent, that the seed path must work, is therefore still under test; it is
conditioned on the pin instead of assumed.

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


def test_an_opted_in_gguf_download_is_refused_while_unpinned(
    hub_calls, monkeypatch
):
    """Opting in buys a PINNED download, never an unpinned one.

    Renamed from ``test_gguf_load_may_download_when_a_human_opted_in``, which
    asserted the opposite. Round 2 was right that a gate must not become a
    permanent wall — see the test below, which holds exactly that — but it was
    written when this path passed no ``revision=`` at all. Consent to download
    is not consent to resolve ``main``: the operator asked for the model, and
    ``main`` is whatever the repo owner pushed most recently.

    The refusal is asserted by CONTENT, not merely by type. Its whole value is
    that the person who hit it is told what to do, and a bare ``pytest.raises``
    would stay green if the message decayed into "unsupported configuration".
    """
    monkeypatch.setattr(embed_mod, "QWEN3_EMBEDDING_GGUF_REVISION", None)
    monkeypatch.setenv(embed_mod.MODEL_DOWNLOAD_ENV, "1")

    with pytest.raises(RuntimeError) as excinfo:
        embed_mod.Embedder(backend="gguf")
    message = str(excinfo.value)

    for expected, why in (
        (
            "QWEN3_EMBEDDING_GGUF_REVISION",
            "the constant the operator has to fill in",
        ),
        ("aggregator/core/embed.py", "the file it lives in"),
        (
            "AGGREGATOR_EMBED_BACKEND=st",
            "the escape hatch — the default backend IS sha-pinned",
        ),
    ):
        assert expected in message, (
            f"the refusal never mentions {expected!r} ({why}), so the operator "
            f"is stopped without being told how to proceed. This is the one "
            f"path where an unpinned artifact would have entered a pinned "
            f"deployment, so the message is the entire remedy. Got: {message!r}"
        )

    # Refused BEFORE any resolution, not after one that already reached out.
    assert _resolutions(hub_calls) == [], (
        f"the gguf path resolved weights before refusing: {hub_calls!r}"
    )


def test_the_gate_reopens_once_a_revision_is_pinned(hub_calls, monkeypatch):
    """Round 2's contract, conditioned on the pin rather than abandoned.

    THE WALL MUST NOT OUTLIVE ITS REASON. The refusal above exists only
    because no sha for the ``-GGUF`` repo has been verified; the moment one is,
    an opted-in human must be able to fetch, or a stopgap has quietly become
    the permanent removal of a supported backend and nothing would say so.

    A dummy sha, because the assertion is about the GATE reopening, not about
    any particular revision — and because naming a real one here would be a
    standing invitation for some future edit to resolve it for real.
    """
    monkeypatch.setattr(embed_mod, "QWEN3_EMBEDDING_GGUF_REVISION", "0" * 40)
    monkeypatch.setenv(embed_mod.MODEL_DOWNLOAD_ENV, "1")

    embed_mod.Embedder(backend="gguf")

    resolutions = _resolutions(hub_calls)
    assert resolutions, f"the gguf path resolved no weights at all: {hub_calls!r}"
    for name, kwargs in resolutions:
        assert kwargs.get("local_files_only") is False, (
            f"{name} stayed offline even though {embed_mod.MODEL_DOWNLOAD_ENV} "
            f"was set and the revision is pinned — the seed unit could never "
            f"fetch gguf weights. kwargs={kwargs!r}"
        )
        assert kwargs.get("revision") == "0" * 40, (
            f"{name} was allowed to download without carrying the pin — "
            f"reopening the gate must not mean reopening it unpinned. "
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
