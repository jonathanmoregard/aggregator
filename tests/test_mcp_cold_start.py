"""Cold start: importing the MCP module must not load the RAG model stack.

WHY THIS IS A TEST AND NOT A CODE COMMENT. ``aggregator.mcp`` is imported by
an MCP server the user's editor starts on demand, and every import that module
performs is paid before the first search returns. v5 made
``sentence-transformers`` (hence torch) a hard runtime dependency, so a single
convenience import at module scope — ``from aggregator.core.embed import
Embedder``, the obvious way to write it, and the way the 2026-08-08 draft did
write it — silently moves a multi-second model-stack import onto the recall
path of every session, including the ones that never run a vector query at
all. Nothing about that failure is visible in a unit test of the search
behaviour; it only shows up as "why is my editor slow to start".

Run in a SUBPROCESS on purpose. By the time the rest of the suite has run,
half these modules are already in the parent's ``sys.modules`` for unrelated
reasons, so an in-process assertion would be measuring test-ordering rather
than the import graph.

TWO INDEPENDENT LISTS, BECAUSE THEY FAIL FOR DIFFERENT REASONS.
``_RAG_ONLY_MODULES`` guards the lazy-import discipline inside ``mcp.py``: the
vector arm and the reranker are constructed inside ``_get_embedder`` /
``_get_reranker``, so a convenience import at module scope is the regression.
``_ML_STACK_MODULES`` guards ``aggregator.core.scrub``: it used to build
Presidio's ``AnalyzerEngine`` and ``AnonymizerEngine`` at module import, which
pulled ``presidio_analyzer`` -> ``transformers`` -> ``torch`` and ``spacy`` ->
``thinc`` -> ``torch``. That was 49.0 s of a 57.6 s cold start measured with
``PYTHONPROFILEIMPORTTIME`` on the user's laptop, Claude Code's connect cap is
30 s, and from 2026-09-04 every session failed to connect to this server at all.
An earlier version of this docstring recorded ``torch in sys.modules`` as
pre-existing and unfixable from the retrieval side. It was fixable from the
scrubbing side, and it is fixed: Presidio initialises on first use now, so the
assertion has teeth and is not to be softened back.
"""

import subprocess
import sys
import textwrap

# Modules that only the vector arm has any reason to load. ``sentence_
# transformers`` is the real cost centre (it is what pulls the model plumbing
# for both Embedder and Reranker); the two aggregator modules are the direct
# proof that the lazy-import discipline inside mcp.py is still in place.
_RAG_ONLY_MODULES = (
    "sentence_transformers",
    "aggregator.core.embed",
    "aggregator.core.rerank",
)

# The PII-scrubbing model stack. Not one of these may be on the import path of
# an MCP server that has 30 s to answer `initialize`. ``presidio_anonymizer``
# is listed even though a machine without the spaCy model never reaches its
# import today (``_build_presidio_engines`` raises on the model probe first) —
# the machine that matters is the one WITH the model installed.
_ML_STACK_MODULES = (
    "torch",
    "spacy",
    "thinc",
    "transformers",
    "presidio_analyzer",
    "presidio_anonymizer",
)


def _modules_after(statement: str, watched: tuple[str, ...] = _RAG_ONLY_MODULES) -> set[str]:
    """Return the subset of ``watched`` loaded by ``statement``."""
    probe = textwrap.dedent(
        f"""
        import sys
        {statement}
        watched = {watched!r}
        print(",".join(m for m in watched if m in sys.modules))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    return {m for m in out.stdout.strip().split(",") if m}


def test_importing_mcp_does_not_load_the_rag_model_stack():
    assert _modules_after("import aggregator.mcp") == set()


def test_importing_mcp_does_not_load_the_reranker():
    """The reranker is the expensive one — ~2 GB RSS — and it is opt-in per
    query via ``rerank=True``. A caller that never passes it must never pay
    for it, not even the import."""
    loaded = _modules_after("import aggregator.mcp")
    assert "aggregator.core.rerank" not in loaded


def test_the_guard_has_teeth():
    """Confirm the probe actually detects a module-scope import, so a green
    result above means "not imported" rather than "probe is broken"."""
    loaded = _modules_after("import aggregator.mcp; import aggregator.core.embed")
    assert "aggregator.core.embed" in loaded


def test_building_the_server_does_not_load_the_rag_model_stack():
    """``build_server`` runs at MCP connect time and reads the cache for the
    live inventory. It must not warm the model stack either."""
    loaded = _modules_after(
        "import aggregator.mcp; aggregator.mcp.build_server()"
    )
    assert loaded == set()


def test_importing_mcp_does_not_load_the_pii_model_stack():
    """The connect-timeout guard.

    ``aggregator.core.scrub`` built Presidio's engines at module scope, so
    importing this server imported torch twice over — 49.0 s of a 57.6 s cold
    start, against a 30 s connect cap. If this list ever comes back non-empty,
    the MCP server has stopped connecting for real users; it is not a style
    finding.
    """
    assert _modules_after("import aggregator.mcp", _ML_STACK_MODULES) == set()


def test_importing_scrub_does_not_load_the_pii_model_stack():
    """The same property one level down, so a failure names the culprit module
    instead of the 5000-line one that imports it."""
    assert _modules_after("import aggregator.core.scrub", _ML_STACK_MODULES) == set()


def test_building_the_server_does_not_load_the_pii_model_stack():
    """``build_server()`` is what runs during the ``initialize`` handshake — it
    reads the cache for the live inventory. Importing cheaply and then paying at
    connect time would be the same bug with an extra step."""
    loaded = _modules_after(
        "import aggregator.mcp; aggregator.mcp.build_server()", _ML_STACK_MODULES
    )
    assert loaded == set()


def test_the_pii_guard_has_teeth():
    """Confirm the probe detects these modules at all, so a green result above
    means "not imported" rather than "probe is broken". Deliberately imports
    ``spacy`` rather than calling the real initialiser: this must stay a
    seconds-long test, and building the engines takes ~50 s."""
    loaded = _modules_after(
        "import aggregator.mcp; import spacy", _ML_STACK_MODULES
    )
    assert "spacy" in loaded
    assert "thinc" in loaded
