"""``scripts/measure_embed_rate.py`` grows a gguf-vs-st bench — offline part.

The gguf Q4_K_M backend exists in ``aggregator/core/embed.py`` but has never
been benchmarked, and its repo pin (``QWEN3_EMBEDDING_GGUF_REVISION``) is a
named hole: ``None``, with the download path refusing rather than resolving
``main``. Closing the hole takes a machine with network, so the script has to
let a HUMAN do the whole loop in one command — resolve the ``-GGUF`` repo's
sha, download at that sha, bench gguf against the shipping st fp32 backend at
the documented probe geometry, report vector fidelity, and print the exact
pin line to paste into ``embed.py``.

Everything here runs OFFLINE. The model layer is stubbed at the script's own
seams (``_load_embedder``, ``resolve_gguf_sha``); no test may construct a real
Embedder or touch the hub. A previous round's probe pulled 15 GB by naming an
uncached model — see ``tests/core/test_model_offline_default.py`` — and a
bench harness is exactly the kind of file where that mistake recurs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from aggregator.core import embed as embed_mod

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "measure_embed_rate.py"


@pytest.fixture(scope="module")
def harness():
    """Import the script by path. It is not a package, and must not become one."""
    spec = importlib.util.spec_from_file_location("_measure_embed_rate", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        sys.modules.pop(spec.name, None)


# ---------------------------------------------------------------------------
# Argument parsing + revision override
# ---------------------------------------------------------------------------


def test_no_arguments_still_means_the_legacy_sweep(harness):
    """The doc's 40 tok/s table was produced by the bare invocation.

    ``docs/embedding-throughput.md`` names ``scripts/measure_embed_rate.py``
    as the instrument behind every number in it, with no flags. Growing a CLI
    must not change what that exact command measures.
    """
    args = harness.parse_args([])
    assert args.backend is None


def test_backend_accepts_st_and_gguf_and_nothing_else(harness):
    assert harness.parse_args(["--backend", "st"]).backend == "st"
    assert harness.parse_args(["--backend", "gguf"]).backend == "gguf"
    with pytest.raises(SystemExit):
        harness.parse_args(["--backend", "onnx"])


def test_the_pin_can_be_supplied_for_validation_before_hardcoding(harness):
    """A flag and an env var, so a sha can be BENCHED before it is committed.

    ``embed.py`` is explicit that the production pin is a source constant and
    never an environment variable. The bench override is the step BEFORE that
    constant exists — the whole point is to validate a candidate sha without
    editing ``embed.py`` — so it lives here, in the harness, and nowhere near
    the production loader.
    """
    sha = "a" * 40
    args = harness.parse_args(["--backend", "gguf", "--gguf-revision", sha])
    assert args.gguf_revision == sha
    assert harness.parse_args([]).gguf_revision is None


def test_resolve_and_pin_is_a_flag(harness):
    assert harness.parse_args(["--resolve-and-pin"]).resolve_and_pin is True
    assert harness.parse_args([]).resolve_and_pin is False


def test_effective_revision_prefers_the_flag_over_the_env(harness):
    flag_sha = "b" * 40
    env_sha = "c" * 40
    env = {harness.GGUF_REVISION_ENV: env_sha}
    assert harness.effective_revision(flag_sha, env) == flag_sha
    assert harness.effective_revision(None, env) == env_sha
    assert harness.effective_revision(None, {}) is None


def test_effective_revision_rejects_a_malformed_sha_before_any_bench(harness):
    """A typo'd pin must fail in milliseconds, not after a 30-minute sweep."""
    for bad in ("main", "deadbeef", "g" * 40, "A" * 40, "0" * 39, "0" * 41):
        with pytest.raises(ValueError):
            harness.effective_revision(bad, {})
        with pytest.raises(ValueError):
            harness.effective_revision(None, {harness.GGUF_REVISION_ENV: bad})


# ---------------------------------------------------------------------------
# Cosine similarity — the fidelity half of the decision rule
# ---------------------------------------------------------------------------


def test_cosine_stats_on_identical_vectors_is_one(harness):
    vecs = np.eye(3, dtype=np.float32)
    stats = harness.cosine_stats(vecs, vecs.copy())
    assert stats["per_text"] == pytest.approx([1.0, 1.0, 1.0])
    assert stats["mean"] == pytest.approx(1.0)
    assert stats["min"] == pytest.approx(1.0)


def test_cosine_stats_on_orthogonal_vectors_is_zero(harness):
    a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    b = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    stats = harness.cosine_stats(a, b)
    assert stats["per_text"] == pytest.approx([0.0, 0.0], abs=1e-7)


def test_cosine_stats_is_row_wise_and_scale_invariant(harness):
    """Cosine, not dot: the stat must survive an unnormalized input.

    ``embed_documents`` L2-normalizes, so in practice dot == cosine — but a
    fidelity number that silently became a dot product would misreport the
    day someone feeds it raw vectors, and nothing downstream would notice.
    """
    a = np.array([[3.0, 0.0], [0.0, 5.0]], dtype=np.float32)
    b = np.array([[1.0, 1.0], [0.0, 2.0]], dtype=np.float32)
    stats = harness.cosine_stats(a, b)
    assert stats["per_text"] == pytest.approx([np.sqrt(2) / 2, 1.0])
    assert stats["mean"] == pytest.approx((np.sqrt(2) / 2 + 1.0) / 2)
    assert stats["min"] == pytest.approx(np.sqrt(2) / 2)


def test_cosine_stats_refuses_mismatched_shapes(harness):
    with pytest.raises(ValueError):
        harness.cosine_stats(np.eye(2, dtype=np.float32), np.eye(3, dtype=np.float32))


# ---------------------------------------------------------------------------
# The pin line — the artifact the whole run exists to produce
# ---------------------------------------------------------------------------


def test_pin_line_is_the_exact_source_line(harness):
    sha = "0123456789abcdef0123456789abcdef01234567"
    assert (
        harness.pin_line(sha)
        == f'QWEN3_EMBEDDING_GGUF_REVISION: str | None = "{sha}"'
    )


def test_pin_line_refuses_anything_but_a_40_hex_sha(harness):
    """A tag is repointable by the repo owner — the thing being defended
    against, per the constant's own docstring. The harness must be unable to
    print a pin line that ``embed.py``'s rule would reject."""
    for bad in ("main", "v1.0", "0" * 39, "0" * 41, "G" * 40, ""):
        with pytest.raises(ValueError):
            harness.pin_line(bad)


def test_pin_line_is_a_drop_in_replacement_for_the_line_embed_py_holds(harness):
    """Guard the two files against drifting apart.

    The printed line is only useful if it replaces the unpinned line verbatim.
    This reads ``embed.py`` off disk and asserts (a) the constant is still the
    named hole — ``None``, nothing invented — and (b) the harness's output is
    that same line with only the value changed.
    """
    source = Path(embed_mod.__file__).read_text()
    unpinned = "QWEN3_EMBEDDING_GGUF_REVISION: str | None = None"
    assert unpinned in source, (
        "embed.py no longer holds the unpinned constant line this harness "
        "formats a replacement for — update pin_line() to match, or the "
        "printed instruction is wrong"
    )
    sha = "f" * 40
    expected = unpinned[: -len("None")] + f'"{sha}"'
    assert harness.pin_line(sha) == expected
    assert embed_mod.QWEN3_EMBEDDING_GGUF_REVISION is None
