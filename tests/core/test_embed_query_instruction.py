"""The Qwen3 task prompt has to be the model's own, not a retyped copy of it.

CRITERION G, EMBEDDING HALF. Qwen3-Embedding is instruction-aware: the model
card puts the cost of omitting the query-side instruction at 1-5% of retrieval
performance, and says the instruction is written in English whatever language
the corpus is in. We were applying one — so the omission the criterion names was
already closed — but we were applying a HAND-TYPED PARAPHRASE of the one the
model ships, and the paraphrase differed in two places:

    ours    'Instruct: Given a search query, ...\\nQuery: '
    shipped 'Instruct: Given a web search query, ...\\nQuery:'

a missing word and a trailing space. That is the same failure Microsoft's Olive
recipe for this exact model documents — carrying an export without
``config_sentence_transformers.json`` costs ~20% of the retrieval benchmarks,
because the task prompts in that file are load-bearing weights-adjacent state.
Dropping the file and retyping its contents slightly wrong are the same bug
with different blast radii; the fix for both is to read the value rather than
restate it.

WHAT IS DELIBERATELY *NOT* HERE: the instruction is not in the string that keys
``chunk_embeddings``. See
``test_the_query_instruction_is_not_part_of_the_embedding_version`` for the
argument, which is the whole reason this file exists rather than a one-line
constant edit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import aggregator.core.embed as embed_mod
from aggregator.core.embed import QWEN3_QUERY_PREFIX, Embedder, embedding_version

_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots"
    / embed_mod.QWEN3_EMBEDDING_REVISION
)


def _shipped_prompts() -> dict:
    config = _SNAPSHOT / "config_sentence_transformers.json"
    if not config.is_file():
        pytest.skip("Qwen3-Embedding weights are not cached on this machine")
    return json.loads(config.read_text(encoding="utf-8"))["prompts"]


class _FakeST:
    """A sentence-transformers double. Names no real model, loads nothing."""

    def __init__(self, model_name, prompts=None, **kwargs):
        self.model_name = model_name
        if prompts is not None:
            self.prompts = prompts

    def encode(self, texts, **kwargs):  # pragma: no cover - not exercised
        return np.zeros((len(texts), 1024), dtype=np.float32)


def _stub_embedder(monkeypatch, *, prompts, model_name=None) -> Embedder:
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        lambda name, **kw: _FakeST(name, prompts=prompts),
    )
    return Embedder(backend="st", model_name=model_name)


# --- carry the file, do not retype it ---------------------------------------


def test_the_fallback_literal_matches_the_prompt_the_model_ships():
    """THE REPRO. Two divergences, both invisible without this comparison."""
    shipped = _shipped_prompts()["query"]

    assert shipped == QWEN3_QUERY_PREFIX, (
        "QWEN3_QUERY_PREFIX is a hand-typed paraphrase of the task prompt in "
        "config_sentence_transformers.json, not a copy of it. The model was "
        "trained and benchmarked with the shipped string; a near-miss is the "
        "same class of error as dropping the file entirely.\n"
        f"  ours    {QWEN3_QUERY_PREFIX!r}\n"
        f"  shipped {shipped!r}"
    )


def test_the_query_prompt_is_read_off_the_loaded_model(monkeypatch):
    """The literal is a FALLBACK. When the model carries its own registry --
    which every Qwen3-Embedding checkpoint does -- that is what gets used, so
    a checkpoint bump cannot leave a stale instruction glued to the query."""
    embedder = _stub_embedder(
        monkeypatch, prompts={"query": "Instruct: do the thing\nQuery:"}
    )
    assert embedder.query_prompt == "Instruct: do the thing\nQuery:"


def test_the_real_model_hands_over_its_own_prompt():
    """The end-to-end version of the two tests above, against the weights on
    this machine rather than against a double."""
    try:
        embedder = Embedder()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"embedder unavailable: {e}")

    assert embedder.query_prompt == _shipped_prompts()["query"]


def test_the_prompt_reaches_the_encoder_verbatim(monkeypatch):
    """``include_prompt: true`` in the model's pooling config is what makes
    plain concatenation identical to sentence-transformers' own
    ``prompt_name='query'`` path -- prompt tokens are pooled either way. So one
    code path serves both backends, and the gguf loader (which has no prompt
    registry at all) is not a second, quietly different implementation."""
    embedder = _stub_embedder(monkeypatch, prompts={"query": "PROMPT:"})
    seen: list[str] = []
    monkeypatch.setattr(
        embedder,
        "_encode",
        lambda texts: (
            seen.extend(texts),
            np.zeros((len(texts), 768), dtype=np.float32),
        )[1],
    )

    embedder.embed_query("power-on self test")

    assert seen == ["PROMPT:power-on self test"]


# --- do not glue Qwen's instruction onto somebody else's model --------------


def test_a_model_with_no_prompt_registry_gets_no_instruction(monkeypatch):
    """A caller-supplied model is not a Qwen3 checkpoint, and its instruction
    format is its own business. BGE, E5 and GTE each want a DIFFERENT prefix,
    and pasting Qwen's onto one of them is not a smaller error than omitting
    it -- the same reasoning that keeps QWEN3_EMBEDDING_REVISION off a
    caller-supplied model."""
    embedder = _stub_embedder(monkeypatch, prompts=None, model_name="acme/embed-x")

    assert embedder.query_prompt == ""

    seen: list[str] = []
    monkeypatch.setattr(
        embedder,
        "_encode",
        lambda texts: (
            seen.extend(texts),
            np.zeros((len(texts), 768), dtype=np.float32),
        )[1],
    )
    embedder.embed_query("power-on self test")
    assert seen == ["power-on self test"]


def test_the_default_model_still_gets_one_when_the_registry_is_missing(
    monkeypatch,
):
    """Belt and braces for the model this package actually pins: an older
    sentence-transformers, or a stripped export, exposes no ``.prompts``. That
    is the Olive case exactly, and it must degrade to the shipped literal
    rather than to no instruction at all."""
    embedder = _stub_embedder(monkeypatch, prompts=None)

    assert embedder.query_prompt == QWEN3_QUERY_PREFIX


# --- the document side, which is what the stored vectors are ----------------


def test_documents_are_encoded_exactly_as_given(monkeypatch):
    """THE INVARIANT THE VERSION STRING RESTS ON. Qwen3 ships
    ``prompts['document'] == ''``, and this path applies nothing at all. If
    that ever changes, every stored vector in the cache is invalidated and
    ``embedding_version`` MUST move with it -- so the day somebody adds a
    document-side instruction, this test is what stops them shipping it
    silently."""
    assert _shipped_prompts()["document"] == "", (
        "Qwen3-Embedding now ships a non-empty document prompt. Applying it "
        "changes every stored vector, so it belongs in embedding_version() "
        "and costs a full re-embed."
    )

    embedder = _stub_embedder(monkeypatch, prompts={"query": "Q:", "document": ""})
    seen: list[list[str]] = []
    monkeypatch.setattr(
        embedder,
        "_encode",
        lambda texts: (
            seen.append(list(texts)),
            np.zeros((len(texts), 768), dtype=np.float32),
        )[1],
    )

    embedder.embed_documents(["a document", "another"])

    assert seen == [["a document", "another"]]


def test_the_query_instruction_is_not_part_of_the_embedding_version():
    """THE CONCLUSION, PINNED SO IT CANNOT BE QUIETLY REVERSED.

    The research says the instruction is part of the embedding contract and so
    belongs in the version string that keys ``chunk_embeddings``. It is part of
    the contract. It does NOT belong in that key, and the two are not in
    tension once you ask what the key is FOR: it exists so a vector written by
    one build is never compared against a vector written by another.

    The query instruction cannot make two stored vectors incomparable, because
    no stored vector has ever seen it -- ``embed_documents`` applies nothing
    (asserted directly above). It transforms the QUERY, which is embedded fresh
    on every search, so a change takes effect immediately, uniformly, and
    against the existing index.

    Keying on it would mean a reworded instruction invalidates ~483k document
    vectors and starts a 25-30 day re-embed on this hardware
    (docs/embedding-throughput.md) to recompute bytes that provably do not
    change. That is exactly what ``embedding_version``'s own docstring already
    forbids: "WHAT IT MUST NEVER CONTAIN is anything that moves per deploy".

    What WOULD have to go in is a document-side instruction, and the test above
    is the tripwire for the day one appears.
    """
    version = embedding_version()

    assert QWEN3_QUERY_PREFIX not in version
    assert "Instruct:" not in version, (
        "the query-side instruction has been keyed into embedding_version(). "
        "Every reword now invalidates the whole document index and starts a "
        "multi-week re-embed, for a change that cannot alter one stored "
        f"vector: {version!r}"
    )
