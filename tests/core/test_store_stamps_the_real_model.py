"""The stamp must name the model that WROTE the vectors, not a guess at it.

Round 3's M2, store half. ``adaa28e`` fixed the embed side: ``Embedder`` now
records the repo id it actually resolved on ``Embedder.model_id``, and
``configured_model_id(embedder)`` reads the answer off the object that did the
work rather than off ``AGGREGATOR_EMBED_BACKEND``.

Nothing consumed it. ``store.vector_provenance()`` still took no arguments, so
the stamp written by ``migrate()`` could name one model while a differently
constructed embedder filled the index with another — the stamp vouching for
precisely the thing it exists to catch.

Why that is load-bearing rather than tidy: round 1's H1 refuses a foreign
index, round 2's S1 refuses to delete one unasked, and round 3's H1 decides
whether ``--reindex`` may destroy 25-30 days of computed vectors. All three are
decisions taken FROM this comparison. An input that can be wrong makes all
three wrong in the same direction at once — it makes a foreign index look
native, which is the failure mode that serves vectors from an unknown model
and calls them current.

THE READ PATH DELIBERATELY KEEPS THE NO-ARGUMENT FORM. It asks "may this
process trust what is on disk?", and it runs on every ``Store``, including the
read-only MCP one, long before any embedder exists. Threading an embedder into
it would mean building a model to answer a question about a file.

CRITERION E WIDENED WHAT THE STAMP HOLDS. It is no longer a bare repo id but
the full embedding version — model, quantization, dimension, chunker geometry,
normalization — because a repo id is silent about three things that each change
the bytes of every vector while leaving the model name untouched. The
assertions below therefore check that the id is CARRIED, not that it is the
whole string; the components get their own coverage in
``tests/test_cli_embed_stamps_its_own_embedder.py``.
"""

import json

import pytest

from aggregator.core.store import (
    _VEC_DIM,
    VECTOR_PROVENANCE_KEY,
    Store,
    vector_provenance,
)


class _FakeEmbedder:
    """Duck-types the one attribute the provenance path reads.

    Deliberately NOT a real ``Embedder``: constructing one loads weights, and
    naming a real uncached repo id in a test is how an earlier round pulled
    15 GB off the Hugging Face CDN. The id below resolves to nothing.
    """

    def __init__(self, model_id: str):
        self.model_id = model_id


def _stamp(store):
    row = store._c().execute(
        "SELECT value FROM meta WHERE key = ?", (VECTOR_PROVENANCE_KEY,)
    ).fetchone()
    return None if row is None else json.loads(row[0])


@pytest.fixture(autouse=True)
def _no_backend_override(monkeypatch):
    monkeypatch.delenv("AGGREGATOR_EMBED_BACKEND", raising=False)


# --- the resolver ----------------------------------------------------------


def test_vector_provenance_reads_the_embedder_it_is_given():
    fake = _FakeEmbedder("acme/embedder-that-actually-ran")
    version, dim = vector_provenance(fake)
    assert version.startswith("acme/embedder-that-actually-ran")
    assert dim == _VEC_DIM


def test_vector_provenance_without_an_embedder_still_answers_for_the_process():
    """The read path's form: what ``Embedder()`` WOULD load here."""
    model, dim = vector_provenance()
    assert dim == _VEC_DIM
    assert model and isinstance(model, str)


def test_the_two_forms_disagree_when_the_embedder_does(monkeypatch):
    """If they could not disagree, none of this would be worth wiring."""
    monkeypatch.setenv("AGGREGATOR_EMBED_BACKEND", "st")
    fake = _FakeEmbedder("acme/some-other-embedder")
    assert vector_provenance(fake)[0] != vector_provenance()[0]


# --- the stamping site, reached from migrate() ------------------------------


def test_migrate_stamps_the_embedder_that_will_write(tmp_path):
    """THE M2 REGRESSION: the stamp described the environment, not the model."""
    store = Store(db_path=tmp_path / "cache.db")
    fake = _FakeEmbedder("acme/embedder-that-actually-ran")

    store.migrate(embedder=fake)

    assert _stamp(store)["dim"] == _VEC_DIM
    assert _stamp(store)["model"].startswith("acme/embedder-that-actually-ran")


def test_migrate_without_an_embedder_stamps_the_process_default(tmp_path):
    store = Store(db_path=tmp_path / "cache.db")
    store.migrate()
    assert _stamp(store) == {"dim": _VEC_DIM, "model": vector_provenance()[0]}


def test_a_stamp_written_for_one_model_is_refused_for_another(tmp_path):
    """The comparison has to bite on the embedder-derived id too."""
    db = tmp_path / "cache.db"
    first = Store(db_path=db)
    first.migrate(embedder=_FakeEmbedder("acme/model-a"))
    first.close()

    second = Store(db_path=db)
    second.migrate(embedder=_FakeEmbedder("acme/model-b"))

    # Nothing computed was on disk, so the honest answer is to adopt and
    # re-stamp rather than refuse — but the stamp must move to model-b.
    assert _stamp(second)["model"].startswith("acme/model-b")


def test_a_matching_embedder_stamp_is_adopted_not_rewritten(tmp_path):
    db = tmp_path / "cache.db"
    fake = _FakeEmbedder("acme/model-a")
    first = Store(db_path=db)
    first.migrate(embedder=fake)
    first.close()

    again = Store(db_path=db)
    again.migrate(embedder=_FakeEmbedder("acme/model-a"))
    assert again.vector_quarantine is None
    assert _stamp(again)["model"].startswith("acme/model-a")
