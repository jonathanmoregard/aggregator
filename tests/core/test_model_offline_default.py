"""Loading a model must never reach the network unless someone asked it to.

``HF_HUB_OFFLINE=1`` is set on the timer-driven embed unit and nowhere else.
The MCP server is registered bare, with no environment wrapper, so the first
``rerank=True`` — or the first text query once the index is warm — would
resolve the hub and pull GB-scale weights inside the editor's MCP process,
from a tool whose annotations say ``openWorldHint=False``. The hardened path
was hardened and the interactive one was fail-open.

An env var cannot fix this from here: ``huggingface_hub`` reads
``HF_HUB_OFFLINE`` into a module constant at import time, and it is already
imported by the time ``aggregator.mcp`` finishes loading (via
``core.scrub`` → spaCy → thinc → transformers). So the refusal is passed
explicitly, per call, as ``local_files_only``.

Fail CLOSED, with one opt-in. ``aggregator-embed-seed.service`` is the single
place in the deployment allowed to fetch weights, and it is human-triggered;
it sets ``AGGREGATOR_ALLOW_MODEL_DOWNLOAD=1``. Everything else — the timer,
the MCP server, an ad-hoc CLI run — stays offline and fails loudly rather
than quietly downloading 1.2 GB.
"""

import pytest

import aggregator.core.embed as embed_mod
import aggregator.core.rerank as rerank_mod

_OPT_IN = "AGGREGATOR_ALLOW_MODEL_DOWNLOAD"


def _kwargs_from(monkeypatch, target, factory):
    seen = {}

    class _Fake:
        def __init__(self, model_name, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(target, _Fake)
    factory()
    return seen


def test_the_embedder_is_offline_by_default(monkeypatch):
    monkeypatch.delenv(_OPT_IN, raising=False)
    kwargs = _kwargs_from(
        monkeypatch, "sentence_transformers.SentenceTransformer", embed_mod.Embedder
    )
    assert kwargs["local_files_only"] is True


def test_the_reranker_is_offline_by_default(monkeypatch):
    monkeypatch.delenv(_OPT_IN, raising=False)
    kwargs = _kwargs_from(
        monkeypatch, "sentence_transformers.CrossEncoder", rerank_mod.Reranker
    )
    assert kwargs["local_files_only"] is True


def test_the_embedder_downloads_only_when_explicitly_allowed(monkeypatch):
    monkeypatch.setenv(_OPT_IN, "1")
    kwargs = _kwargs_from(
        monkeypatch, "sentence_transformers.SentenceTransformer", embed_mod.Embedder
    )
    assert kwargs["local_files_only"] is False


def test_the_reranker_downloads_only_when_explicitly_allowed(monkeypatch):
    monkeypatch.setenv(_OPT_IN, "1")
    kwargs = _kwargs_from(
        monkeypatch, "sentence_transformers.CrossEncoder", rerank_mod.Reranker
    )
    assert kwargs["local_files_only"] is False


@pytest.mark.parametrize("value", ["0", "", "no", "false"])
def test_only_an_affirmative_opt_in_counts(monkeypatch, value):
    """A variable that exists but says no must not be read as yes."""
    monkeypatch.setenv(_OPT_IN, value)
    kwargs = _kwargs_from(
        monkeypatch, "sentence_transformers.SentenceTransformer", embed_mod.Embedder
    )
    assert kwargs["local_files_only"] is True


def test_an_uncached_model_is_refused_rather_than_fetched(monkeypatch):
    """The end-to-end proof, against the real loader.

    A model that is genuinely not on this machine must raise instead of
    starting a multi-gigabyte download inside whatever process asked.

    THE REPO NAME IS DELIBERATELY NONEXISTENT, and that is a lesson paid for
    in bandwidth. An earlier draft of this test named a REAL model that simply
    was not cached locally (``Qwen/Qwen3-Embedding-8B``). Run against the
    unfixed code — which is the whole point of a red test — it did exactly
    what this test exists to forbid and pulled 15 GB onto the machine. A test
    for "must not download" must not be able to download even when it fails,
    so it names a repository that cannot resolve: offline it raises before any
    socket, and a regression to fail-open costs one 404 instead of a disk.
    """
    monkeypatch.delenv(_OPT_IN, raising=False)
    with pytest.raises(OSError):
        embed_mod.Embedder(
            model_name="aggregator-test/nonexistent-model-do-not-create"
        )


def test_the_seed_unit_is_the_only_place_that_opts_in(repo_root):
    """The opt-in must live in the one human-triggered unit, and only there."""
    from pathlib import Path

    module = (Path(repo_root) / "nix" / "aggregator.nix").read_text()
    assert module.count(_OPT_IN) == 1
    seeder = module.split("embedSeeder")[1]
    assert _OPT_IN in seeder.split("embedFailureNotify")[0]
