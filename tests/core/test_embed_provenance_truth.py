"""The provenance stamp must describe the embedder that did the work.

Round 3's M2. ``configured_model_id()`` took no arguments and read
``AGGREGATOR_EMBED_BACKEND`` only, while ``Embedder`` resolves from its own
``backend=`` / ``model_name=`` arguments first. So an ``Embedder`` built
non-default wrote vectors under one model while the stamp named another.

That stamp is not decoration. Round 1's H1 refuses to serve a vector index
whose stamp does not match this build, and round 2's S1 refuses to DELETE one
without explicit consent — both are decisions taken from the stamp, so a
stamp that can disagree with reality undermines both at once.

No weights are loaded anywhere here: the loaders are stubbed before any
Embedder is constructed, and the only repo ids named are the package's own
defaults (already cached) or deliberately nonexistent ones.
"""

import sys
import types

import pytest

import aggregator.core.embed as embed_mod

_BACKEND = "AGGREGATOR_EMBED_BACKEND"
_FAKE_MODEL = "aggregator-test/nonexistent-embedding-do-not-create"


@pytest.fixture
def stub_loaders(monkeypatch):
    """Neutralise both backends. Construction must reach no weights at all."""

    class _FakeST:
        def __init__(self, model_name, **kwargs):
            self.model_name = model_name

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeST)

    fake_llama = types.ModuleType("llama_cpp")

    class _Llama:
        def __init__(self, **kwargs):
            pass

    fake_llama.Llama = _Llama
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_llama)

    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda **kw: "/nonexistent/stub.gguf"
    )
    # The gguf pin refusal is M1's business, not this file's.
    monkeypatch.setattr(embed_mod, "QWEN3_EMBEDDING_GGUF_REVISION", "0" * 40)


def test_a_non_default_backend_is_not_stamped_as_the_default(
    monkeypatch, stub_loaders
):
    """The original lie, stated directly.

    Environment says nothing; the caller asked for gguf. The vectors are
    gguf vectors, so the stamp has to say gguf.
    """
    monkeypatch.delenv(_BACKEND, raising=False)
    embedder = embed_mod.Embedder(backend="gguf")

    assert embedder.model_id == embed_mod._DEFAULT_MODEL_GGUF
    assert embed_mod.configured_model_id(embedder) == embed_mod._DEFAULT_MODEL_GGUF
    # ...and the environment-derived answer really does differ, so this is a
    # live divergence rather than a pair of assertions that agree by accident.
    assert embed_mod.configured_model_id() != embedder.model_id


def test_a_caller_supplied_model_is_stamped_as_itself(monkeypatch, stub_loaders):
    monkeypatch.delenv(_BACKEND, raising=False)
    embedder = embed_mod.Embedder(model_name=_FAKE_MODEL)

    assert embedder.model_id == _FAKE_MODEL
    assert embed_mod.configured_model_id(embedder) == _FAKE_MODEL


@pytest.mark.parametrize("backend", ["st", "gguf"])
def test_the_two_resolutions_cannot_drift(monkeypatch, stub_loaders, backend):
    """The no-argument form must still mean what it claims to mean.

    It is the read path's question — "what would ``Embedder()`` load?" — asked
    before any embedder exists. If that answer ever stops matching what
    ``Embedder()`` actually resolves, the stamp goes back to vouching for a
    model nobody ran, which is the whole failure this file is about.
    """
    monkeypatch.setenv(_BACKEND, backend)
    assert embed_mod.configured_model_id() == embed_mod.Embedder().model_id


def test_the_stamp_is_readable_even_when_the_weights_fail_to_load(monkeypatch):
    """``model_id`` is a fact about configuration, not about a successful load.

    It is set before any weight is touched, so a caller diagnosing a failed
    load can still say which model was being asked for.
    """
    monkeypatch.delenv(_BACKEND, raising=False)

    def _boom(*a, **kw):
        raise OSError("weights not present")

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _boom)
    with pytest.raises(OSError):
        embed_mod.Embedder(model_name=_FAKE_MODEL)

    # The resolution itself is pure and independent of the load.
    assert embed_mod._resolve_model_id("st", _FAKE_MODEL) == _FAKE_MODEL
