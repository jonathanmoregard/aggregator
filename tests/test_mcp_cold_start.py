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

KNOWN, PRE-EXISTING, AND NOT WHAT THIS GUARDS: ``torch`` itself IS already in
``sys.modules`` after ``import aggregator.mcp``, and has been since long
before hybrid retrieval — ``aggregator.core.scrub`` probes for its spaCy model
at module scope, and spaCy imports thinc, which imports torch. That is roughly
4 s of the cold start and it belongs to the PII-scrubbing path, not the RAG
path. Asserting ``'torch' not in sys.modules`` here would fail for a reason no
change on the retrieval side can fix, and would then be deleted by whoever
next saw it fail. So this asserts the property the retrieval side actually
owns and can keep: the RAG model stack is not touched until a query needs it.
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


def _modules_after(statement: str) -> set[str]:
    """Return the subset of ``_RAG_ONLY_MODULES`` loaded by ``statement``."""
    probe = textwrap.dedent(
        f"""
        import sys
        {statement}
        watched = {_RAG_ONLY_MODULES!r}
        print(",".join(m for m in watched if m in sys.modules))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
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
