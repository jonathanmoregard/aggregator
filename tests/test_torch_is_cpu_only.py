"""``torch`` must resolve to the CPU build, and the lock must prove it.

WHY THIS FILE EXISTS. ``sentence-transformers`` pulls ``torch``, and the
default PyPI ``torch`` wheel declares the whole NVIDIA CUDA stack as hard
dependencies. This deployment is CPU-only — every throughput figure in
``docs/embedding-throughput.md`` is measured on a 15 W i7-1365U — so none of
that can ever execute. It is also not merely wasted bytes: packaging the
default wheel for NixOS FAILS, because ``nvidia-cufile`` ships
``libcufile_rdma.so.1`` linked against libmlx5 / librdmacm / libibverbs, and
auto-patchelf cannot satisfy those InfiniBand libraries. The whole
``aggregator-env`` derivation dies with it, which is how this was found.

The fix is `[tool.uv.sources] torch = { index = "pytorch-cpu" }` plus a DIRECT
``torch`` dependency, because uv binds sources only to direct dependencies and
a transitive torch silently ignores the pin. That combination is easy to
undo by accident: dropping the direct dependency as "redundant" (it looks
redundant — sentence-transformers already requires it) restores the CUDA
resolution with no other visible change.

So the assertion is on the LOCK, not on the installed interpreter. A test that
only checked ``torch.cuda.is_available()`` would pass on this laptop no matter
which wheel was pinned, since there is no NVIDIA device here to find.
"""

import re
import sys
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_LOCK = _ROOT / "uv.lock"
_PYPROJECT = _ROOT / "pyproject.toml"

_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

# Any distribution whose name marks it as part of the CUDA stack. Matched on
# the lock's `name = "..."` entries. `triton` is included deliberately: it is
# not named nvidia-*, it exists to compile GPU kernels, and it arrives with
# the CUDA torch build.
_GPU_NAME_RE = re.compile(r"^(nvidia[-_].*|cuda[-_].*|triton|.*-cu\d+)$")


def _lock() -> dict:
    if not _LOCK.exists():  # pragma: no cover - a checkout without a lock
        pytest.skip("uv.lock is absent")
    with _LOCK.open("rb") as fh:
        return tomllib.load(fh)


def _packages() -> list[dict]:
    return _lock().get("package", [])


def test_no_cuda_distribution_is_locked():
    """Nothing in the resolved set may be a GPU runtime.

    Names the offenders rather than asserting a count, because the useful
    thing at failure time is which package dragged the stack back in.
    """
    gpu = sorted(
        {p["name"] for p in _packages() if _GPU_NAME_RE.match(p.get("name", ""))}
    )
    assert not gpu, (
        "the lock resolved GPU runtime packages, so the aggregator-env "
        "derivation will fail on NixOS (nvidia-cufile wants libmlx5 / "
        "librdmacm / libibverbs and auto-patchelf cannot satisfy them). "
        "Check that `torch` is still a DIRECT dependency in pyproject.toml — "
        "uv binds [tool.uv.sources] only to direct dependencies, so a purely "
        f"transitive torch silently ignores the CPU index. Found: {gpu}"
    )


def test_torch_is_locked_against_the_cpu_index():
    """The positive half: torch is present, and it came from the CPU index."""
    torch_entries = [p for p in _packages() if p.get("name") == "torch"]
    assert torch_entries, "torch is not in the lock at all"

    sources = {
        p.get("source", {}).get("registry", "<no registry>") for p in torch_entries
    }
    assert _CPU_INDEX in sources, (
        f"torch is locked against {sources} — none of which is the CPU index "
        f"{_CPU_INDEX}. The `+cpu` local version is what carries no nvidia-* "
        "dependencies."
    )


def test_torch_is_a_direct_dependency():
    """The mechanism the two assertions above depend on.

    Pinned separately so its removal fails with the REASON attached rather
    than as a mysterious reappearance of the CUDA stack.
    """
    with _PYPROJECT.open("rb") as fh:
        pyproject = tomllib.load(fh)

    deps = pyproject["project"]["dependencies"]
    assert any(d == "torch" or d.startswith("torch") for d in deps), (
        "torch is not a direct dependency. It looks redundant because "
        "sentence-transformers already requires it — but [tool.uv.sources] "
        "binds only DIRECT dependencies, so removing it silently returns the "
        "resolution to the CUDA wheel on PyPI."
    )

    source = pyproject.get("tool", {}).get("uv", {}).get("sources", {}).get("torch")
    assert source == {"index": "pytorch-cpu"}, (
        f"torch's uv source is {source!r}, expected the pytorch-cpu index"
    )


@pytest.mark.skipif(
    "torch" not in sys.modules and not _LOCK.exists(),
    reason="nothing to check against",
)
def test_the_installed_torch_agrees_with_the_lock():
    """Weak on its own, kept as the one check that runs against reality.

    ``torch.version.cuda`` is None for a CPU build and a version string for a
    CUDA build, and unlike ``cuda.is_available()`` it does not depend on a GPU
    being present — so it distinguishes the WHEEL rather than the hardware.
    """
    torch = pytest.importorskip("torch")
    assert torch.version.cuda is None, (
        f"the installed torch is a CUDA build (cuda={torch.version.cuda}); "
        "the venv is out of sync with the lock — re-run `uv sync`"
    )
