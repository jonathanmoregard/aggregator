"""Task J — blackbox end-to-end: ingest → embed → query.

WHAT MAKES THIS FILE DIFFERENT FROM ``test_mcp_hybrid.py``. That file seeds the
store through ``Store.upsert_entities`` and writes vectors by hand, so it pins
the retrieval semantics precisely but never runs the two things that actually
put data in a user's cache: the ``aggregator ingest`` CLI and the
``aggregator embed`` worker. Everything here goes through those two commands
and then asks the MCP tool functions — ``aggregator_query`` /
``aggregator_capabilities`` — the way a client would. Nothing calls
``_apply_hybrid``, ``_fused_id_scope`` or ``Store.upsert_vec_*`` directly.

THE THREE SEAMS, and why they are the only ones:

* ``XDG_DATA_HOME`` is redirected at a ``tmp_path`` (the ``tmp_data_home``
  fixture), so the CLI and the MCP surface each resolve their OWN cache path.
  No test passes ``_store=``: a wrong path resolution is a bug this file
  should catch, and the live 1.2 GB cache must be unreachable even from a
  mistake.
* ``SessionsSource(projects_root=...)`` is pointed at a fixture corpus. It is
  the source's own documented constructor argument, not a test hook.
* ``Embedder`` is replaced by a deterministic stub, in ``aggregator.cli`` for
  the worker and at ``aggregator.mcp._get_embedder`` for the query path. Real
  Qwen3 weights would make the suite slow, and — worse — would make "is this
  document semantically near that query" a fact about a 600M-parameter model
  rather than a fact the test author chose. Both seams get the SAME stub, so
  the document vectors the worker writes and the query vector the retriever
  computes live in one deliberately-designed space.

THE SAD PATHS ARE THE POINT. The happy path is one test. The rest of the file
is the states a real cache is actually in: a vector index that is empty, one
that is half-built (the steady state — the worker runs on a 30-minute timer
against a 372k-row corpus, so "half" is where it lives), a backfill that lands
between two pages of one paginated query, a machine with no sqlite-vec, an
embedder that dies mid-run, a body with nothing in it, and a worker killed by
SIGKILL partway through a batch.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from aggregator.cli import main as cli_main
from aggregator.core import store as store_mod
from aggregator.core.store import Store
from aggregator.mcp import aggregator_capabilities, aggregator_query
from aggregator.sources.sessions import SessionsSource

# --- the fixture corpus ------------------------------------------------------
#
# One session per document, one observation per session, so a result id is a
# document id and every assertion below reads as a set of documents.
#
# Each body carries a ``docid-<name>`` marker. It is load-bearing twice: it
# gives the stub embedder a deterministic per-document distance rank (see
# ``_stub_vector``), and it is a token no query in this file ever searches for,
# so it cannot accidentally become an FTS5 hit.

_DOC_ORDER = ["gov", "vote2", "vote3", "vote4", "ballot", "pigeon", "long", "k8s"]

_LONG_PARAGRAPH = (
    "docid-long the pigeons roost on the parapet and refuse to move along, "
    "which is the sort of thing that fills a maintenance log. " * 12
)

CORPUS: list[tuple[str, str, int]] = [
    ("gov", "docid-gov quadratic voting mechanism design notes", 1),
    ("vote2", "docid-vote2 voting turnout statistics for the district", 2),
    ("vote3", "docid-vote3 voting machine audit log review", 3),
    ("vote4", "docid-vote4 voting rights litigation summary", 4),
    ("ballot", "docid-ballot ballot reform proposal for the council", 5),
    ("pigeon", "docid-pigeon the pigeons roost on the ledge each evening", 6),
    ("long", "\n\n".join([_LONG_PARAGRAPH] * 3), 7),
    ("k8s", "docid-k8s kubernetes ingress controller troubleshooting", 8),
    ("blank", "   ", 9),
]

# Documents whose body contains the literal token "voting". This is the
# ground truth the pre-v5 FTS5 path must return for dsl="voting", and it is
# derived from the corpus text above rather than from a previous run of the
# code under test.
FTS_VOTING = {"sess-gov", "sess-vote2", "sess-vote3", "sess-vote4"}

# No body contains "governance" or "columbidae" — those exist only in the
# stub's vector space, so a hit on them is proof the vector arm answered.
SEMANTIC_ONLY_GOVERNANCE = "governance"
SEMANTIC_ONLY_PIGEON = "columbidae"


def _write_corpus(root: Path, docs: list[tuple[str, str, int]]) -> Path:
    """Write ``docs`` as Claude Code session JSONL under ``root``.

    Back-dated past ``SessionsSource.LIVE_WINDOW_SECONDS``: the source skips
    files younger than five minutes because they may still be being written,
    and a test corpus created microseconds ago is exactly that shape.
    """
    proj = root / "proj-alpha"
    proj.mkdir(parents=True, exist_ok=True)
    old = time.time() - 24 * 60 * 60
    for name, body, day in docs:
        path = proj / f"sess-{name}.jsonl"
        path.write_text(
            json.dumps(
                {
                    "parentUuid": None,
                    "isSidechain": False,
                    "promptId": f"p-{name}",
                    "type": "user",
                    "message": {"role": "user", "content": body},
                    "uuid": f"obs-{name}",
                    "timestamp": f"2026-07-{day:02d}T08:00:00.000Z",
                    "cwd": "/home/u/proj-alpha",
                    "sessionId": f"sess-{name}",
                    "version": "2.1.92",
                    "gitBranch": "main",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        os.utime(path, (old, old))
    return proj


# --- the stub model space ----------------------------------------------------
#
# ``test_mcp_hybrid.py`` has a keyword→axis stub of the same family; this one
# is not importable from there because the e2e needs three things that one does
# not have — a per-document distance RANK (so "which neighbour is nearest" is a
# fact the test states rather than a coin-flip between tied vectors), an
# injectable failure point, and a call ledger keyed by row rather than by
# batch. The axis convention is deliberately the same so the two files read
# alike.

_DIM = 768
_SALT_AXIS = 700

# A word's semantic axis. Note "governance" and "columbidae" are QUERY-only
# words: they appear in no document body, so a query using one can only be
# answered by the vector arm.
_AXES = {
    "quadratic": 0,
    "voting": 0,
    "ballot": 0,
    "governance": 0,
    "pigeon": 1,
    "roost": 1,
    "ledge": 1,
    "columbidae": 1,
    "kubernetes": 2,
    "ingress": 2,
}
_DEFAULT_AXIS = 2


def _stub_vector(text: str) -> np.ndarray:
    """A 768-d vector whose distances are chosen, not learned.

    ``axis`` decides the semantic cluster. ``_SALT_AXIS`` carries a small
    per-document offset ordered by ``_DOC_ORDER``, so within one cluster the
    KNN ranking is total and known in advance — no ties for sqlite-vec to
    break however it likes. A query string carries no marker and therefore sits
    at salt 0, i.e. nearest to the earliest document of its cluster.
    """
    v = np.zeros(_DIM, dtype=np.float32)
    lowered = (text or "").lower()
    axis = _DEFAULT_AXIS
    for word, a in _AXES.items():
        if word in lowered:
            axis = a
            break
    v[axis] = 1.0
    for i, name in enumerate(_DOC_ORDER):
        if f"docid-{name}" in lowered:
            v[_SALT_AXIS] = 0.01 * (i + 1)
            break
    return v


#: ``{document name: semantic cluster}``, derived from the bodies rather than
#: restated, so a corpus edit cannot silently desynchronise the expectations.
DOC_CLUSTER = {name: int(np.argmax(_stub_vector(body)[:3])) for name, body, _ in CORPUS}


class StubEmbedder:
    """Deterministic embedder with a ledger and an injectable failure.

    ``embed_documents`` is called once per ROW by the worker (a row's chunks
    go in one call), so ``document_calls`` counts rows embedded — which is what
    "how much work was redone after the crash" is measured in.
    """

    def __init__(self, fail_on_document_call: int | None = None):
        self.query_calls = 0
        self.document_calls = 0
        self.documents: list[str] = []
        self.fail_on_document_call = fail_on_document_call

    def embed_documents(self, docs: list[str]) -> np.ndarray:
        self.document_calls += 1
        if self.fail_on_document_call == self.document_calls:
            raise RuntimeError("embedder died mid-batch")
        self.documents.extend(docs)
        return np.array([_stub_vector(d) for d in docs], dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        self.query_calls += 1
        return _stub_vector(query)


class EmbedderConstructed(BaseException):
    """Raised when a path that must not build a model builds one.

    DERIVES FROM ``BaseException``, AND THAT IS THE ENTIRE POINT. The retrieval
    code degrades to FTS5 on any embedding failure and does so with a blanket
    ``except Exception`` — correctly, because a broken model must cost the
    vector arm and never the answer. An ``AssertionError`` raised from a
    forbidden embedder is an ``Exception``, so it would be caught, logged, and
    the test would pass while the model it forbade was being constructed on
    every call. Verified: with the assertion-based version, breaking the
    routing predicate so it engages on every query left these tests green.
    """


class ForbiddenEmbedder:
    """Fails the test if anything constructs it.

    Used where the contract is "the model is never touched". An assertion on a
    call counter would prove the same thing one step later; this proves it at
    the moment of construction, with the traceback pointing at the caller.
    """

    def __init__(self, *a, **kw):
        raise EmbedderConstructed(
            "the embedder was constructed on a path that must not build one"
        )


class StubReranker:
    def __init__(self, prefer: str):
        self.prefer = prefer
        self.calls = 0

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        self.calls += 1
        return np.array([1.0 if self.prefer in d else 0.0 for d in docs], np.float32)


# --- driving the two commands ------------------------------------------------


@pytest.fixture
def cache(tmp_data_home):
    """The cache path the CLI and the MCP surface will each resolve on their own.

    Returned only so tests can inspect the database directly; no test hands it
    to ``_store=``.
    """
    return Path(tmp_data_home) / "aggregator" / "cache.db"


@pytest.fixture
def corpus(tmp_path):
    return _write_corpus(tmp_path / "projects", CORPUS)


@pytest.fixture
def stub_models(monkeypatch):
    """One stub shared by the worker and the query path.

    They must agree: the worker writes document vectors and the retriever
    computes the query vector, and a hybrid hit only means anything if both
    came out of the same space.
    """
    stub = StubEmbedder()
    monkeypatch.setattr("aggregator.cli.Embedder", lambda *a, **kw: stub)
    monkeypatch.setattr("aggregator.mcp._get_embedder", lambda: stub)
    return stub


def run_cli(argv: list[str], sources: dict | None = None) -> int:
    """Run one CLI command against the ``XDG_DATA_HOME`` cache, then close it.

    THE CLOSE IS NOT TIDINESS. The writable connection runs in WAL mode and the
    MCP recall path opens its cache with ``immutable=1``, which by design does
    not read the ``-wal`` sidecar. A real deployment gets the checkpoint for
    free because the CLI is a process that exits; in-process, the store has to
    be closed or the next query reads a database that is missing everything
    this command just wrote.
    """
    store = Store()
    try:
        return cli_main(argv, _store=store, _sources=sources)
    finally:
        store.close()


def ingest(corpus_root: Path) -> int:
    return run_cli(
        ["ingest", "sessions"],
        sources={"sessions": SessionsSource(projects_root=str(corpus_root))},
    )


def ids(result: dict) -> set[str]:
    return {r["stable_id"] for r in result["records"]}


def embedding_states(cache: Path, kind: str = "observations") -> dict[str, str | None]:
    """``{row id: embedding_state}`` straight out of SQLite."""
    store = Store(db_path=cache)
    try:
        col = "obs_id" if kind == "observations" else "stable_id"
        return {
            r[col]: r["embedding_state"]
            for r in store._c().execute(f"SELECT {col}, embedding_state FROM {kind}")  # noqa: S608
        }
    finally:
        store.close()


def vector_ids(cache: Path, kind: str = "observations") -> set[str]:
    """The ids the vector arm can actually reach, chunk suffixes stripped."""
    store = Store(db_path=cache)
    try:
        col = "obs_id" if kind == "observations" else "stable_id"
        table = f"vec_{kind}"
        rows = store._c().execute(f"SELECT {col} FROM {table}")  # noqa: S608
        out = set()
        for r in rows:
            raw = r[col]
            base, sep, tail = raw.rpartition(":")
            out.add(base if sep and base and tail.isdigit() else raw)
        return out
    finally:
        store.close()


# --- happy path --------------------------------------------------------------


def test_ingest_then_embed_then_query_finds_what_keywords_cannot(
    corpus, cache, stub_models
):
    """THE WHOLE FEATURE, in the order a user gets it.

    ``governance`` appears in no document. Before the backfill the query is
    answerable only by FTS5 and returns nothing at all; after
    ``aggregator embed --catchup`` the same query, through the same tool,
    reaches the document about quadratic voting.
    """
    assert ingest(corpus) == 0

    cold = aggregator_query(SEMANTIC_ONLY_GOVERNANCE)
    assert cold["ok"] is True
    assert cold["total"] == 0, "precondition: FTS5 alone cannot answer this"
    assert stub_models.query_calls == 0, "an empty index must not embed anything"

    assert run_cli(["embed", "--catchup"]) == 0

    warm = aggregator_query(SEMANTIC_ONLY_GOVERNANCE)
    assert warm["ok"] is True
    assert "sess-gov" in ids(warm)
    assert stub_models.query_calls == 1


def test_the_vector_arm_returns_the_nearest_documents_and_not_the_others(
    corpus, cache, stub_models, monkeypatch
):
    """Presence is not selection.

    With the shipped ``_VECTOR_ARM_K`` of 50 and a nine-document corpus the KNN
    returns everything, so a passing "the right row came back" assertion would
    also pass if the arm returned the corpus in file order. Narrowing k to 2
    makes the arm choose, and the stub's distance ranks say in advance which
    two it must choose: the query is in the pigeon cluster and carries no
    document salt, so the nearest are the two pigeon-cluster documents in
    ``_DOC_ORDER`` order.
    """
    assert ingest(corpus) == 0
    assert run_cli(["embed", "--catchup"]) == 0
    monkeypatch.setattr("aggregator.mcp._VECTOR_ARM_K", 2)

    result = aggregator_query(SEMANTIC_ONLY_PIGEON)
    assert result["ok"] is True
    assert ids(result) == {"sess-pigeon", "sess-long"}


def test_a_multi_chunk_document_is_reachable_through_the_vector_arm(
    corpus, cache, stub_models
):
    """A body over the chunker's 4000-char ceiling is stored as ``<id>:<n>``
    vectors, so the retriever has to widen a chunk id back to its row before
    the id can match anything. The long document exists to make that a real
    step in a real run rather than a unit test of ``_widen_chunk_ids``.
    """
    assert ingest(corpus) == 0
    assert run_cli(["embed", "--catchup"]) == 0

    store = Store(db_path=cache)
    try:
        raw = [
            r["obs_id"]
            for r in store._c().execute(
                "SELECT obs_id FROM vec_observations WHERE obs_id LIKE 'obs-long%'"
            )
        ]
    finally:
        store.close()
    assert len(raw) > 1, "precondition: the long body must have been chunked"
    assert all(":" in i for i in raw)

    result = aggregator_query(SEMANTIC_ONLY_PIGEON)
    assert "sess-long" in ids(result)


# --- sad path: the vector index is empty -------------------------------------


def test_an_empty_vector_index_answers_exactly_what_fts5_answered(corpus, cache):
    """Pre-v5 behaviour, byte for byte, and no model anywhere near it.

    The expected id set is read off the corpus text, not off a previous run, so
    this fails if hybrid routing ever engages on an empty index and quietly
    widens the answer.
    """
    assert ingest(corpus) == 0
    assert aggregator_capabilities()["vector_index"]["state"] == "not_started"

    result = aggregator_query("voting")
    assert result["ok"] is True
    assert ids(result) == FTS_VOTING
    assert result["total"] == len(FTS_VOTING)


def test_an_empty_vector_index_never_constructs_the_embedder(
    corpus, cache, monkeypatch
):
    monkeypatch.setattr("aggregator.mcp._get_embedder", ForbiddenEmbedder)
    assert ingest(corpus) == 0
    assert aggregator_query("voting")["ok"] is True


# --- sad path: the half-embedded corpus, which is the steady state -----------


def test_a_half_embedded_corpus_serves_the_built_part_and_loses_nothing(
    corpus, cache, stub_models
):
    """THE STATE THE CACHE IS ACTUALLY IN, almost always.

    The worker runs one batch per timer tick against a corpus of hundreds of
    thousands of rows, so "fully embedded" is a state this cache reaches
    roughly never. Two things have to hold at once while it is partway
    through, and they pull in opposite directions:

    * a keyword query must still return every keyword match, embedded or not —
      the vector arm may only ADD; and
    * a semantic query must reach the embedded rows of its own cluster and
      must not reach anything outside the embedded set — no phantom rows from
      a half-written index.

    The embedded set is read back out of ``embedding_state`` rather than
    hard-coded, so this test states a property and not a snapshot of one
    particular batch ordering. The semantic bound is two-sided rather than an
    equality for the same reason: a similarity cutoff, if one is ever added,
    may legitimately shrink the far half of the neighbour list, and it must
    still be unable to shrink the near half or to invent a row.
    """
    assert ingest(corpus) == 0
    assert run_cli(["embed", "--once", "--batch-size", "6"]) == 0

    states = embedding_states(cache)
    embedded = {i for i, s in states.items() if s == "ok"}
    pending = {i for i, s in states.items() if s is None}
    assert embedded and pending, "precondition: the backfill is genuinely partway"
    assert aggregator_capabilities()["vector_index"]["state"] == "backfilling"

    keyword = aggregator_query("voting")
    assert ids(keyword) >= FTS_VOTING, "a warm arm dropped a keyword match"

    semantic = aggregator_query(SEMANTIC_ONLY_GOVERNANCE)
    reachable = {f"sess-{i.removeprefix('obs-')}" for i in embedded}
    same_cluster = {f"sess-{n}" for n, c in DOC_CLUSTER.items() if c == 0}
    assert ids(semantic) <= reachable, "an unembedded row came back from the index"
    assert ids(semantic) >= reachable & same_cluster, "a built row went missing"


def test_the_backfill_finishes_and_the_watermark_says_so(corpus, cache, stub_models):
    """``--catchup`` after a partial ``--once`` picks up exactly the remainder."""
    assert ingest(corpus) == 0
    assert run_cli(["embed", "--once", "--batch-size", "6"]) == 0
    done_first = sum(1 for s in embedding_states(cache).values() if s is not None)

    before = stub_models.document_calls
    assert run_cli(["embed", "--catchup"]) == 0
    assert stub_models.document_calls - before == 9 - done_first

    assert all(s is not None for s in embedding_states(cache).values())
    assert aggregator_capabilities()["vector_index"]["state"] == "complete"


# --- sad path: a backfill lands between two pages of one query ---------------


def test_a_backfill_landing_between_pages_neither_skips_nor_repeats_a_row(
    corpus, cache, stub_models
):
    """THE HAZARD CHUNKS I AND L EXIST TO CLOSE, proved from outside.

    Page 1 is served with an empty vector index, so it is an offset into the
    FTS5 result set. The embed timer then fires — which, on a 30-minute timer,
    is the ordinary case and not an edge one — and page 2 is asked for. If the
    arm were re-decided per call, page 2 would be an offset into a candidate
    set that is now twice as large: the caller would silently re-read some rows
    and never see others, with nothing anywhere saying so.

    Two assertions, and the second is the one with teeth. Together the pages
    must cover the FTS5 result set exactly — no duplicate, no omission — and
    the query embedder must not have been called for page 2, which is the
    direct evidence that the token pinned the arm rather than the pages merely
    happening to line up.
    """
    assert ingest(corpus) == 0

    page1 = aggregator_query("voting", page_size=2)
    assert page1["total"] == len(FTS_VOTING)
    token = page1["next_page_token"]
    assert token

    assert run_cli(["embed", "--catchup"]) == 0
    assert aggregator_capabilities()["vector_index"]["state"] == "complete"
    calls_before_page_2 = stub_models.query_calls

    page2 = aggregator_query("voting", page_size=2, page_token=token)
    assert page2["ok"] is True
    assert stub_models.query_calls == calls_before_page_2, "page 2 re-decided the arm"

    seen = [r["stable_id"] for r in page1["records"]] + [
        r["stable_id"] for r in page2["records"]
    ]
    assert len(seen) == len(set(seen)), f"a row was served twice: {seen}"
    assert set(seen) == FTS_VOTING, f"a row was skipped: {FTS_VOTING - set(seen)}"


def test_pinning_the_arm_does_not_wedge_hybrid_off_for_the_next_query(
    corpus, cache, stub_models
):
    """The other half of the pin. A caller who starts a NEW query after the
    backfill lands gets the warm arm — otherwise "never break pagination"
    would have been bought by never using the vector index again.
    """
    assert ingest(corpus) == 0
    aggregator_query("voting", page_size=2)
    assert run_cli(["embed", "--catchup"]) == 0

    fresh = aggregator_query(SEMANTIC_ONLY_GOVERNANCE)
    assert fresh["ok"] is True
    assert "sess-gov" in ids(fresh)
    assert stub_models.query_calls == 1


def test_a_hybrid_page_token_keeps_paging_the_hybrid_result_set(
    corpus, cache, stub_models
):
    """The warm-start direction: page 1 minted by the vector arm, page 2
    continues in it. The pages must partition the fused set, not overlap it.

    ``total`` is read off the first page rather than asserted, so this stays a
    statement about pagination and does not quietly become a second assertion
    about how many neighbours the vector arm returns — that number belongs to
    the test below, where changing it is a deliberate act.
    """
    assert ingest(corpus) == 0
    assert run_cli(["embed", "--catchup"]) == 0

    page1 = aggregator_query(SEMANTIC_ONLY_GOVERNANCE, page_size=3)
    total = page1["total"]
    assert total > 3, "precondition: the result must not fit on one page"
    collected = [r["stable_id"] for r in page1["records"]]
    token = page1["next_page_token"]
    while token:
        page = aggregator_query(SEMANTIC_ONLY_GOVERNANCE, page_size=3, page_token=token)
        collected += [r["stable_id"] for r in page["records"]]
        token = page.get("next_page_token")
    assert len(collected) == len(set(collected)) == total


def test_a_warm_index_answers_a_query_that_matches_nothing(corpus, cache, stub_models):
    """KNOWN, ESCALATED, AND PINNED HERE SO A CHANGE TO IT IS DELIBERATE.

    The KNN is ``ORDER BY distance LIMIT k`` with no similarity cutoff, so once
    the index is warm every free-text query comes back with up to
    ``_VECTOR_ARM_K`` neighbours however far away they are. End to end that
    reads as: a query no document contains returned nothing before the
    backfill and returns the entire embedded corpus after it. Recall up,
    precision down — which is the trade hybrid retrieval IS — but the "no
    results" answer stops existing along the way, and on a recall tool that
    answer carries information.

    ``test_mcp_hybrid.py`` pins the same decision at the unit level; this is
    what it looks like from outside, which is the level at which somebody
    notices. If Task M sets a distance floor from the real-cache measurement,
    THIS is the test to change: the expected count becomes whatever the floor
    admits, and the change should be visible in a diff rather than absorbed by
    a loose bound somewhere else.
    """
    assert ingest(corpus) == 0
    assert aggregator_query(SEMANTIC_ONLY_GOVERNANCE)["total"] == 0

    assert run_cli(["embed", "--catchup"]) == 0
    warm = aggregator_query(SEMANTIC_ONLY_GOVERNANCE)
    assert warm["total"] == 8, "every embedded document; only the blank body has none"


# --- sad path: no sqlite-vec on this machine ---------------------------------


@pytest.fixture
def no_sqlite_vec(monkeypatch):
    """Every ``Store`` built after this point loads without the extension.

    Patched at the module function rather than on an instance because the
    point is that the CLI's writable store and the MCP surface's read-only one
    — neither of which the test constructs — both come up degraded.
    """

    def _boom(conn):
        raise store_mod.sqlite3.OperationalError("simulated sqlite-vec ABI mismatch")

    monkeypatch.setattr(store_mod, "_load_sqlite_vec", _boom)
    monkeypatch.setattr(store_mod, "_VEC_LOAD_WARNED", False)


def test_without_the_extension_keyword_search_still_answers(
    corpus, cache, no_sqlite_vec, monkeypatch
):
    """FTS5 has no vector dependency and must not acquire one. A machine whose
    sqlite-vec wheel is missing or ABI-mismatched keeps the recall tool it had
    before v5 — and ``VectorIndexUnavailableError`` never reaches the caller.
    """
    monkeypatch.setattr("aggregator.mcp._get_embedder", ForbiddenEmbedder)
    assert ingest(corpus) == 0

    result = aggregator_query("voting")
    assert result["ok"] is True
    assert ids(result) == FTS_VOTING


def test_without_the_extension_capabilities_says_unavailable_not_zero(
    corpus, cache, no_sqlite_vec
):
    """``vectors: None``, never ``0``.

    Zero is the answer that turns "this install is broken, fix it" into "the
    backfill has not started, wait" — the two need different humans doing
    different things, and only one of them resolves by waiting.
    """
    assert ingest(corpus) == 0
    vector_index = aggregator_capabilities()["vector_index"]

    assert vector_index["available"] is False
    assert vector_index["state"] == "unavailable"
    assert vector_index["reason"]
    assert vector_index["observations"]["vectors"] is None
    assert vector_index["records"]["vectors"] is None
    # The backlog is plain-column arithmetic and stays legible when the arm
    # is dead — that is precisely when an operator needs to read it.
    assert vector_index["observations"]["pending"] == 9


def test_without_the_extension_the_worker_refuses_and_marks_nothing(
    corpus, cache, no_sqlite_vec, capsys
):
    """Non-zero, on stderr, and the backlog untouched.

    With no extension the vector writes silently no-op. Marking the rows
    embedded anyway would advance the watermark past rows that have no vector,
    and nothing ever looks at a row a second time — the silent permanent loss
    the watermark rules exist to forbid. Refusing the whole run is the only
    safe answer.
    """
    assert ingest(corpus) == 0

    rc = run_cli(["embed", "--catchup"])
    assert rc != 0
    assert "sqlite-vec" in capsys.readouterr().err
    assert set(embedding_states(cache).values()) == {None}


# --- sad path: the embedder dies mid-run -------------------------------------


def test_an_embedder_that_dies_mid_batch_fails_loudly_and_strands_nothing(
    corpus, cache, monkeypatch
):
    """A run that cannot finish must not look like one that did.

    The failure is raised out of the CLI — non-zero exit with the traceback on
    stderr, which is what "fail loudly" means for an unattended timer — and the
    watermark stops where the vectors stop. The batch that was in flight when
    the model died wrote neither a vector nor a state, so the next run picks it
    up whole.
    """
    stub = StubEmbedder(fail_on_document_call=3)
    monkeypatch.setattr("aggregator.cli.Embedder", lambda *a, **kw: stub)
    assert ingest(corpus) == 0

    with pytest.raises(RuntimeError, match="died mid-batch"):
        run_cli(["embed", "--catchup", "--batch-size", "2"])

    states = embedding_states(cache)
    ok = {i for i, s in states.items() if s == "ok"}
    assert ok == {"obs-k8s"}, "the watermark advanced past a row with no vector"
    assert states["obs-blank"] == "skip"
    assert vector_ids(cache) == ok
    assert sum(1 for s in states.values() if s is None) == 7


def test_the_run_after_the_embedder_failure_drains_the_rest_of_the_backlog(
    corpus, cache, monkeypatch
):
    """The failure must be recoverable by re-running, with no row stranded and
    none embedded twice."""
    dying = StubEmbedder(fail_on_document_call=3)
    monkeypatch.setattr("aggregator.cli.Embedder", lambda *a, **kw: dying)
    assert ingest(corpus) == 0
    with pytest.raises(RuntimeError):
        run_cli(["embed", "--catchup", "--batch-size", "2"])

    healthy = StubEmbedder()
    monkeypatch.setattr("aggregator.cli.Embedder", lambda *a, **kw: healthy)
    assert run_cli(["embed", "--catchup", "--batch-size", "2"]) == 0

    assert healthy.document_calls == 7, "a completed row was embedded again"
    states = embedding_states(cache)
    assert all(s is not None for s in states.values())
    assert vector_ids(cache) == {i for i, s in states.items() if s == "ok"}


# --- sad path: a row with nothing to embed -----------------------------------


def test_an_empty_body_is_skipped_without_aborting_the_run(
    corpus, cache, stub_models
):
    """A whitespace-only body produces no chunks and therefore no vector.

    It still has to leave the backlog or the worker re-reads it on every tick
    forever — but as ``'skip'``, because "nothing to embed" and "embedded" are
    different facts and only one of them means the vector arm can find the row.
    The run around it completes normally.
    """
    assert ingest(corpus) == 0
    assert run_cli(["embed", "--catchup"]) == 0

    states = embedding_states(cache)
    assert states["obs-blank"] == "skip"
    assert "obs-blank" not in vector_ids(cache)
    assert {i for i, s in states.items() if s == "ok"} == set(states) - {"obs-blank"}


def test_the_skipped_row_does_not_come_back_on_the_next_tick(
    corpus, cache, stub_models
):
    """``'skip'`` has to be terminal. A row that stays selectable is a row the
    worker pays for on every one of the 48 ticks it runs each day, forever."""
    assert ingest(corpus) == 0
    assert run_cli(["embed", "--catchup"]) == 0
    before = stub_models.document_calls

    assert run_cli(["embed", "--catchup"]) == 0
    assert stub_models.document_calls == before


def test_the_skipped_row_is_still_reachable_by_keyword(corpus, cache, stub_models):
    """Skipping the vector is not dropping the row. FTS5 owns it."""
    assert ingest(corpus) == 0
    assert run_cli(["embed", "--catchup"]) == 0
    result = aggregator_query("source:sessions session:sess-blank")
    assert result["ok"] is True
    assert ids(result) == {"sess-blank"}


# --- rerank ------------------------------------------------------------------


def test_rerank_reorders_the_page_without_changing_who_is_on_it(
    corpus, cache, stub_models, monkeypatch
):
    """Rerank is an ordering, not a filter.

    If it could also drop hits it would be a second retrieval stage with no
    recall guarantee, and a caller could not reason about ``total`` at all.
    ``docid-gov`` is the OLDEST document, so recency ordering puts it last;
    preferring it moves it to the front, which no ordering the page already
    had would produce by accident.
    """
    reranker = StubReranker(prefer="docid-gov")
    monkeypatch.setattr("aggregator.mcp._get_reranker", lambda: reranker)
    assert ingest(corpus) == 0
    assert run_cli(["embed", "--catchup"]) == 0

    plain = aggregator_query("voting", fields="full")
    reranked = aggregator_query("voting", fields="full", rerank=True)

    assert reranker.calls == 1
    assert ids(plain) == ids(reranked)
    assert plain["total"] == reranked["total"]
    assert reranked["records"][0]["stable_id"] == "sess-gov"
    assert plain["records"][0]["stable_id"] != "sess-gov"


def test_rerank_off_by_default_never_loads_the_cross_encoder(
    corpus, cache, stub_models, monkeypatch
):
    loaded: list[int] = []
    monkeypatch.setattr(
        "aggregator.mcp._get_reranker",
        lambda: loaded.append(1) or StubReranker("docid-gov"),
    )
    assert ingest(corpus) == 0
    assert run_cli(["embed", "--catchup"]) == 0
    assert aggregator_query("voting")["ok"] is True
    assert loaded == []


# --- the query that must cost nothing ----------------------------------------


def test_a_filter_only_query_constructs_no_embedder_on_a_warm_index(
    corpus, cache, monkeypatch
):
    """Asserted on the seam, not on the clock.

    A pure-filter query has no text to embed, so the model must not be built —
    and "it was fast" is not evidence of that, because a warm process has the
    model in memory already and a cold one is slow for a dozen other reasons.
    ``ForbiddenEmbedder`` raises at construction, so the proof is a traceback
    at the exact call rather than an inference from a timer.
    """
    stub = StubEmbedder()
    monkeypatch.setattr("aggregator.cli.Embedder", lambda *a, **kw: stub)
    assert ingest(corpus) == 0
    assert run_cli(["embed", "--catchup"]) == 0
    assert aggregator_capabilities()["vector_index"]["state"] == "complete"

    monkeypatch.setattr("aggregator.mcp._get_embedder", ForbiddenEmbedder)
    for dsl in ("source:sessions", "from:2026-07-01 to:2026-07-31", "session:sess-gov"):
        assert aggregator_query(dsl)["ok"] is True, dsl


# --- sad path: the worker is killed mid-run ----------------------------------
#
# The embed timer carries ``TimeoutStartSec``, the machine reboots, and a
# 372k-row backfill runs for days. Being killed partway through is not an
# exotic failure for this worker; it is a weekly event. Two windows matter and
# they are not the same window, so each gets its own test:
#
#   1. killed BETWEEN batches — batch n is committed whole, batch n+1 has
#      written nothing;
#   2. killed BETWEEN the vector write and the watermark advance, the one
#      moment where the two halves of a batch disagree.
#
# In both, the invariant is the same and it is directional: vectors may run
# ahead of ``embedding_state`` (costing at most a repeated batch), and
# ``embedding_state`` may never run ahead of the vectors (costing rows that
# nothing will ever come back for).


def assert_watermark_not_ahead_of_data(cache: Path) -> None:
    states = embedding_states(cache)
    ok = {i for i, s in states.items() if s == "ok"}
    missing = ok - vector_ids(cache)
    assert not missing, f"marked embedded with no vector: {sorted(missing)}"


_KILL_RUNNER = '''\
"""Test harness — run the embed worker, then hang at a known point.

Executed as a real subprocess so the test can SIGKILL it: no ``finally``, no
atexit, no chance for the code under test to tidy up on the way out, which is
the whole point. The stub returns zero vectors because this run is about
durability rather than retrieval — nothing queries the rows it writes.
"""
import sys
import time
from pathlib import Path

import numpy as np

import aggregator.cli as cli

SENTINEL = Path(sys.argv[1])
BATCH_SIZE = sys.argv[2]
HANG_ON_CALL = int(sys.argv[3])


class HangingStub:
    def __init__(self, *a, **kw):
        self.calls = 0

    def embed_documents(self, docs):
        self.calls += 1
        if self.calls == HANG_ON_CALL:
            SENTINEL.write_text("ready")
            time.sleep(600)
        return np.zeros((len(docs), 768), dtype=np.float32)

    def embed_query(self, query):
        return np.zeros(768, dtype=np.float32)


cli.Embedder = HangingStub
sys.exit(cli.main(["embed", "--catchup", "--batch-size", BATCH_SIZE]))
'''


def test_a_sigkilled_catchup_resumes_without_redoing_a_finished_batch(
    corpus, cache, tmp_path, tmp_data_home, repo_root, monkeypatch
):
    """SIGKILL a real worker process, then restart it.

    The kill point is chosen, not raced: the child writes a sentinel file at
    the first row of the SECOND batch and then sleeps, so at the moment the
    signal arrives batch 1 is committed whole and batch 2 has touched nothing.
    A timing-based kill would make this test a coin flip about which assertions
    were even meaningful.
    """
    assert ingest(corpus) == 0

    runner = tmp_path / "kill_runner.py"
    runner.write_text(_KILL_RUNNER, encoding="utf-8")
    sentinel = tmp_path / "batch-2-started"
    env = {**os.environ, "XDG_DATA_HOME": str(tmp_data_home)}
    proc = subprocess.Popen(  # noqa: S603
        [sys.executable, str(runner), str(sentinel), "2", "2"],
        cwd=repo_root,
        env=env,
    )
    try:
        deadline = time.monotonic() + 120
        while not sentinel.exists():
            assert proc.poll() is None, f"worker exited early: rc={proc.returncode}"
            assert time.monotonic() < deadline, "worker never reached the second batch"
            time.sleep(0.05)
        proc.send_signal(signal.SIGKILL)
    finally:
        proc.wait(timeout=60)
    assert proc.returncode == -signal.SIGKILL

    states = embedding_states(cache)
    assert states["obs-blank"] == "skip", "batch 1 did not commit"
    assert states["obs-k8s"] == "ok"
    assert sum(1 for s in states.values() if s is None) == 7
    assert_watermark_not_ahead_of_data(cache)

    resumed = StubEmbedder()
    monkeypatch.setattr("aggregator.cli.Embedder", lambda *a, **kw: resumed)
    assert run_cli(["embed", "--catchup", "--batch-size", "2"]) == 0

    redone = resumed.document_calls - 7
    assert redone == 0
    assert redone <= 2, "more than one batch was redone"
    assert all(s is not None for s in embedding_states(cache).values())
    assert_watermark_not_ahead_of_data(cache)


def test_a_crash_between_the_vector_write_and_the_watermark_costs_one_batch(
    corpus, cache, monkeypatch
):
    """The narrow window, and the direction it has to fail in.

    ``_embed_batch`` commits the vectors and then advances the watermark. A
    death in between leaves vectors with no state — so the batch is selected
    again next run and redone. That costs one batch of embedding and nothing
    else, because the vec write is delete-then-insert and therefore idempotent.
    The mirrored ordering would leave rows marked embedded with no vector, and
    nothing ever looks at those rows again.
    """
    stub = StubEmbedder()
    monkeypatch.setattr("aggregator.cli.Embedder", lambda *a, **kw: stub)
    assert ingest(corpus) == 0

    real_mark = Store.mark_embedded
    crashed: list[int] = []

    def crash_once(self, kind, ids_, state):
        if not crashed:
            crashed.append(1)
            raise RuntimeError("power cut between the vectors and the watermark")
        return real_mark(self, kind, ids_, state)

    monkeypatch.setattr(Store, "mark_embedded", crash_once)
    with pytest.raises(RuntimeError, match="power cut"):
        run_cli(["embed", "--catchup", "--batch-size", "2"])

    assert set(embedding_states(cache).values()) == {None}
    assert vector_ids(cache) == {"obs-k8s"}, "the vectors of batch 1 are on disk"
    assert_watermark_not_ahead_of_data(cache)

    monkeypatch.setattr(Store, "mark_embedded", real_mark)
    resumed = StubEmbedder()
    monkeypatch.setattr("aggregator.cli.Embedder", lambda *a, **kw: resumed)
    assert run_cli(["embed", "--catchup", "--batch-size", "2"]) == 0

    assert resumed.document_calls == 8, "the whole backlog, with batch 1 redone"
    assert all(s is not None for s in embedding_states(cache).values())
    assert_watermark_not_ahead_of_data(cache)
    # Delete-then-insert, so redoing a batch cannot leave the row with two
    # vectors — a duplicate would double that row's weight in every KNN.
    store = Store(db_path=cache)
    try:
        n = store._c().execute(
            "SELECT COUNT(*) AS n FROM vec_observations WHERE obs_id = 'obs-k8s'"
        ).fetchone()["n"]
    finally:
        store.close()
    assert n == 1


def test_the_watermark_never_runs_ahead_of_the_vectors_at_any_batch_boundary(
    corpus, cache, stub_models
):
    """The invariant, checked after every batch of a real backfill rather than
    only at the end — a violation that self-heals by the final batch is still
    a window in which the vector arm serves a hole."""
    assert ingest(corpus) == 0
    for _ in range(9):
        assert run_cli(["embed", "--once", "--batch-size", "2"]) == 0
        assert_watermark_not_ahead_of_data(cache)
    assert all(s is not None for s in embedding_states(cache).values())
