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

import re
from pathlib import Path

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


def _strip_comment_lines(nix_source: str) -> str:
    """Drop whole-line comments — Nix ``#`` and, inside the ``''`` blocks, shell ``#``.

    Whole-line only, deliberately. A ``#`` mid-line can sit inside a string
    literal, and a stripper that guessed at those would be its own bug.
    """
    return "\n".join(
        line
        for line in nix_source.splitlines()
        if not line.lstrip().startswith("#")
    )


#: An assignment is what ENABLES the opt-in. Matches ``FOO=1``,
#: ``export FOO=1`` and the ``"FOO=1"`` form a systemd ``Environment=`` list
#: would use, capturing the assigned value.
_OPT_IN_ASSIGNMENT = re.compile(rf"\b{_OPT_IN}\s*=\s*([^\s\"']*)")


def test_the_seed_unit_is_the_only_place_that_opts_in(repo_root, monkeypatch):
    """Exactly one place in the deployment may ENABLE model downloads.

    This counts ASSIGNMENTS, not mentions.

    The previous version asserted ``module.count(_OPT_IN) == 1`` on the raw
    file, and went red the moment the module explained the invariant in
    prose. Two of the three occurrences that broke it were comments — and
    one of those comments exists precisely to record that the MCP server
    deliberately does NOT get the opt-in. A test that punishes documenting
    the rule is not measuring the rule. What the rule is actually about is
    what turns the variable ON.

    The invariant itself was re-confirmed against the rendered artifacts
    rather than assumed: of the four generated units, none sets the variable
    via ``Environment=``, and only ``aggregator-embed-seed.service``'s
    ExecStart script contains it at all (``export ...=1``).
    ``checks.<system>.aggregator-embed-unit-hygiene`` asserts that rendered
    half; this test asserts the source it is generated from, so the two
    cannot drift.
    """
    module = (Path(repo_root) / "nix" / "aggregator.nix").read_text()
    code = _strip_comment_lines(module)

    assignments = _OPT_IN_ASSIGNMENT.findall(code)
    assert len(assignments) == 1, (
        f"{_OPT_IN} must be assigned in exactly ONE place — the "
        f"human-triggered seed unit. Found {len(assignments)} "
        f"assignment(s): {assignments}. Every other caller (the timer, the "
        f"MCP server, an ad-hoc CLI run) must stay offline and fail loudly "
        f"rather than quietly pulling GBs of weights. Note this counts "
        f"assignments only: mentioning {_OPT_IN} in a comment is free."
    )

    # ...and that one assignment lives in the seeder, rather than the seeder
    # merely being the only unit that happens to mention it.
    seeder = code.split("embedSeeder")[1].split("embedFailureNotify")[0]
    assert _OPT_IN_ASSIGNMENT.search(seeder), (
        f"{_OPT_IN} is assigned outside the embedSeeder script. The opt-in "
        f"is what makes a 2.4 GB download consented to, and it is only "
        f"consent if a human started that unit by hand — no timer, no "
        f"[Install] section, no activation hook."
    )

    # The value the Nix module assigns and the predicate the Python side
    # reads must AGREE — asserted through `downloads_allowed` rather than
    # against a hardcoded "1", so that switching the module to `=true` (which
    # Python does accept) stays legal while `=on` or `=0` does not. Pinning
    # the literal would have been a second brittleness of exactly the kind
    # this test is being repaired for.
    monkeypatch.setenv(_OPT_IN, assignments[0])
    assert embed_mod.downloads_allowed() is True, (
        f"nix/aggregator.nix sets {_OPT_IN}={assignments[0]!r}, which "
        f"downloads_allowed() does not treat as enabled — the seed unit "
        f"would run with the gate shut and silently fetch nothing"
    )
