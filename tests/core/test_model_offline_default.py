"""Loading the reranker must never reach the network unless someone asked it to.

The reranker is constructed inside the MCP server on ``rerank=True``, so a
fail-open loader would pull GB-scale weights into that process, from a tool
whose annotations say ``openWorldHint=False``.

An env var cannot fix this from here: ``huggingface_hub`` reads
``HF_HUB_OFFLINE`` into a module constant at import time, and nothing in this
package controls when that import happens. So the refusal is passed
explicitly, per call, as ``local_files_only``.

Fail CLOSED, with one opt-in: ``AGGREGATOR_ALLOW_MODEL_DOWNLOAD=1``, which a
human sets when running ``aggregator seed-reranker``. Nothing in the deployment
sets it.
"""

import re
from pathlib import Path

import pytest

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


def test_the_reranker_is_offline_by_default(monkeypatch):
    monkeypatch.delenv(_OPT_IN, raising=False)
    kwargs = _kwargs_from(
        monkeypatch, "sentence_transformers.CrossEncoder", rerank_mod.Reranker
    )
    assert kwargs["local_files_only"] is True


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
        monkeypatch, "sentence_transformers.CrossEncoder", rerank_mod.Reranker
    )
    assert kwargs["local_files_only"] is True


def test_an_uncached_model_is_refused_rather_than_fetched(monkeypatch):
    """The end-to-end proof, against the real loader.

    A model that is genuinely not on this machine must raise instead of
    starting a multi-gigabyte download inside whatever process asked.

    THE REPO NAME IS DELIBERATELY NONEXISTENT. A test for "must not download"
    must not be able to download even when it fails, so it names a repository
    that cannot resolve: offline it raises before any socket, and a regression
    to fail-open costs one 404 instead of a disk.
    """
    monkeypatch.delenv(_OPT_IN, raising=False)
    with pytest.raises(OSError):
        rerank_mod.Reranker(model_name="aggregator-test/nonexistent-model-do-not-create")


#: An assignment is what ENABLES the opt-in. Matches ``FOO=1``,
#: ``export FOO=1`` and the ``"FOO=1"`` form a systemd ``Environment=`` list
#: would use.
_OPT_IN_ASSIGNMENT = re.compile(rf"\b{_OPT_IN}\s*=")


def test_no_deployed_unit_opts_in(repo_root):
    """Nothing the Nix module renders may ENABLE model downloads.

    Counts ASSIGNMENTS in non-comment lines, not mentions: documenting the
    rule in a comment is free.
    """
    module = (Path(repo_root) / "nix" / "aggregator.nix").read_text()
    code = "\n".join(
        line for line in module.splitlines() if not line.lstrip().startswith("#")
    )
    assert not _OPT_IN_ASSIGNMENT.search(code), (
        f"nix/aggregator.nix assigns {_OPT_IN}: an unattended unit must never "
        f"be able to start a GB-scale download"
    )
