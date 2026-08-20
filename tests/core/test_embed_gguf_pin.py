"""The gguf backend must be as pinned as the safetensors one — or say so.

Round 3's M1. ``Embedder``'s ``st`` branch passed ``revision=
QWEN3_EMBEDDING_REVISION``; its ``gguf`` branch called ``hf_hub_download``
with no ``revision=`` at all, so it resolved the ``-GGUF`` repo at ``main``.
That is a moving target inside a deployment whose entire rule is that a
rev-pinned unit executes fixed bytes, and it sat behind a comment admitting
the gap rather than behind anything that enforced it.

``QWEN3_EMBEDDING_REVISION`` is NOT the fix: it was read off the safetensors
repository and is not a valid ref in the separate ``-GGUF`` one. No sha for
that repo has been verified on this build, so the pin is a named hole
(``QWEN3_EMBEDDING_GGUF_REVISION = None``) and the DOWNLOAD path refuses
rather than quietly resolving ``main``.

NOTHING HERE CAN REACH THE NETWORK. ``llama_cpp`` is stubbed into
``sys.modules`` (it is an optional extra and is not installed), and
``hf_hub_download`` is replaced with a recorder before any Embedder is built.
A test about not downloading must not be able to download even when it fails
— see ``test_model_offline_default.py`` for the 15 GB this rule cost once.
"""

import sys
import types

import pytest

import aggregator.core.embed as embed_mod

_OPT_IN = "AGGREGATOR_ALLOW_MODEL_DOWNLOAD"

#: A repo id that cannot resolve, for the caller-supplied case. Deliberately
#: not a real model: if a stub ever fails to install, the blast radius is a
#: 404 rather than a disk full of weights.
_FAKE_REPO = "aggregator-test/nonexistent-gguf-do-not-create"


@pytest.fixture
def gguf(monkeypatch):
    """Stub both halves of the gguf load and record the download kwargs."""
    fake_llama = types.ModuleType("llama_cpp")

    class _Llama:
        def __init__(self, **kwargs):
            pass

    fake_llama.Llama = _Llama
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_llama)

    seen: dict = {}

    def _recorder(**kwargs):
        seen.update(kwargs)
        return "/nonexistent/stub.gguf"

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _recorder)
    return seen


def test_the_gguf_path_passes_a_revision_argument(monkeypatch, gguf):
    """The pin must be WIRED, not merely intended.

    Asserting the keyword is present — rather than only its value — is what
    catches the original defect. The bug was an absent argument, and an absent
    argument is invisible to any assertion that reads ``kwargs.get("revision")``
    and compares it to the ``None`` the constant currently holds.
    """
    monkeypatch.delenv(_OPT_IN, raising=False)
    embed_mod.Embedder(backend="gguf")
    assert "revision" in gguf, (
        "the gguf loader called hf_hub_download with no revision= at all, so "
        "it resolves the repo at 'main' — a moving artifact under a "
        "rev-pinned deployment. Pass the pin explicitly, even when it is None."
    )


def test_a_verified_gguf_sha_is_actually_forwarded(monkeypatch, gguf):
    """When the hole is filled, the value has to reach the hub call.

    Guards the other direction: a wired-but-ignored argument would satisfy the
    test above forever while pinning nothing.
    """
    sha = "0" * 40
    monkeypatch.setattr(embed_mod, "QWEN3_EMBEDDING_GGUF_REVISION", sha)
    monkeypatch.delenv(_OPT_IN, raising=False)
    embed_mod.Embedder(backend="gguf")
    assert gguf["revision"] == sha


def test_an_unpinned_gguf_download_is_refused_loudly(monkeypatch, gguf):
    """The gap is loud, not silent — on the one path where bytes can move."""
    monkeypatch.setattr(embed_mod, "QWEN3_EMBEDDING_GGUF_REVISION", None)
    monkeypatch.setenv(_OPT_IN, "1")
    with pytest.raises(RuntimeError) as excinfo:
        embed_mod.Embedder(backend="gguf")
    message = str(excinfo.value)
    # The message has to be actionable on its own: what is wrong, why the
    # obvious wrong fix is wrong, and where to put the right one.
    assert "QWEN3_EMBEDDING_GGUF_REVISION" in message
    assert "aggregator/core/embed.py" in message
    assert "AGGREGATOR_EMBED_BACKEND=st" in message
    # ...and it must not have fetched anything on the way to complaining.
    assert gguf == {}


def test_an_already_seeded_offline_load_still_works_unpinned(monkeypatch, gguf):
    """Refusing must not break a machine whose weights are already on disk.

    Those bytes are not moving. Breaking a working offline load to protest a
    missing pin would be a self-inflicted outage, not a safeguard.
    """
    monkeypatch.setattr(embed_mod, "QWEN3_EMBEDDING_GGUF_REVISION", None)
    monkeypatch.delenv(_OPT_IN, raising=False)
    embed_mod.Embedder(backend="gguf")
    assert gguf["revision"] is None
    assert gguf["local_files_only"] is True


def test_a_caller_supplied_gguf_repo_is_neither_pinned_nor_refused(
    monkeypatch, gguf
):
    """Symmetry with the ``st`` path, which round 2 settled.

    A pin taken from one repository vouches for nothing else, and someone who
    names their own repo has already chosen it. The rule this module enforces
    is about the repo it picks by default — extending the refusal to arbitrary
    caller-supplied ids would block a legitimate use with a message about a
    constant that could never apply to it.
    """
    monkeypatch.setenv(_OPT_IN, "1")
    embed_mod.Embedder(backend="gguf", model_name=_FAKE_REPO)
    assert gguf["repo_id"] == _FAKE_REPO
    assert gguf["revision"] is None
