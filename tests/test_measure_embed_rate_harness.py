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
import json
import os
import sys
import types
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


def test_validate_sha_refuses_a_sha_with_trailing_newline(harness):
    """``$`` tolerates a trailing newline; ``fullmatch`` semantics must not.

    ``"sha\\n"`` would flow into a pin line whose quoted value embeds the
    newline — a source line embed.py could never hold.
    """
    sha = "a" * 40
    for bad in (sha + "\n", sha + " ", "\n" + sha, sha + "\nx"):
        with pytest.raises(ValueError):
            harness.validate_sha(bad)
        with pytest.raises(ValueError):
            harness.pin_line(bad)


def test_the_env_read_is_stripped_but_the_flag_is_not(harness):
    """Env values sourced from files carry a trailing newline — the friendly
    contract is to strip the ENV read. The flag comes from argv and stays
    strict: a newline there is an anomaly worth refusing."""
    sha = "a" * 40
    env = {harness.GGUF_REVISION_ENV: sha + "\n"}
    assert harness.effective_revision(None, env) == sha
    env = {harness.GGUF_REVISION_ENV: f"  {sha}  \n"}
    assert harness.effective_revision(None, env) == sha
    with pytest.raises(ValueError):
        harness.effective_revision(sha + "\n", {})


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


# ---------------------------------------------------------------------------
# Bench orchestration — model layer stubbed at the script's own seams
# ---------------------------------------------------------------------------

_SHA = "0123456789abcdef0123456789abcdef01234567"


class _FakeEmbedder:
    """Just enough surface for the bench: a tokenizer and deterministic vectors.

    Both backends return the SAME unit vector per text, so the cos-sim the
    bench reports has a known value (1.0) without a single real weight.
    """

    def __init__(self, backend):
        self.backend = backend
        self._st_model = types.SimpleNamespace(
            tokenizer=lambda text: {"input_ids": [0] * max(1, len(text) // 5)}
        )

    def embed_documents(self, docs):
        out = np.zeros((len(docs), 8), dtype=np.float32)
        for i, doc in enumerate(docs):
            out[i, len(doc) % 8] = 1.0
        return out


@pytest.fixture
def fake_loaders(monkeypatch, harness):
    """Replace ``_load_embedder`` and record every construction."""
    loads = []

    def _fake_load(backend):
        loads.append(
            {
                "backend": backend,
                "pin_at_load": embed_mod.QWEN3_EMBEDDING_GGUF_REVISION,
            }
        )
        return _FakeEmbedder(backend)

    monkeypatch.setattr(harness, "_load_embedder", _fake_load)
    return loads


def test_a_gguf_run_prints_rates_speedup_cos_and_the_exact_pin_line(
    harness, fake_loaders, capsys
):
    """Criterion 3, end to end: everything a human needs is in the stdout of
    one successful run — tok/s per backend, the ratio, the fidelity numbers,
    and the line to paste into embed.py."""
    rc = harness.main(["--backend", "gguf", "--gguf-revision", _SHA])
    out = capsys.readouterr().out

    assert rc == 0
    for needle in ("st", "gguf", "tok/s", "speedup", "cos"):
        assert needle in out, f"summary never mentions {needle!r}"
    assert harness.pin_line(_SHA) in out, (
        "the run ended without printing the exact QWEN3_EMBEDDING_GGUF_REVISION "
        "source line — the one artifact the whole bench exists to produce"
    )
    # Machine-readable trailer, one line, for tooling and the docs table.
    summary_lines = [
        line for line in out.splitlines() if line.startswith("BENCH_SUMMARY ")
    ]
    assert len(summary_lines) == 1
    summary = json.loads(summary_lines[0][len("BENCH_SUMMARY ") :])
    assert set(summary["backends"]) == {"st", "gguf"}
    for backend in ("st", "gguf"):
        assert summary["backends"][backend]["tokens_per_s"] > 0
    assert summary["speedup"] > 0
    assert summary["cos"]["mean"] == pytest.approx(1.0)
    assert len(summary["cos"]["per_text"]) == len(harness.SIZE_CURVE_CHARS)
    assert summary["revision"] == _SHA


def test_the_candidate_sha_reaches_the_gguf_load_and_the_constant_is_restored(
    harness, fake_loaders
):
    """The override must exercise the SAME path the hardcoded pin will take —
    ``embed.py``'s own constant, injected for the load — and must leave no
    trace afterwards: the source constant stays None (criterion 6)."""
    rc = harness.main(["--backend", "gguf", "--gguf-revision", _SHA])
    assert rc == 0

    gguf_loads = [rec for rec in fake_loaders if rec["backend"] == "gguf"]
    assert len(gguf_loads) == 1
    assert gguf_loads[0]["pin_at_load"] == _SHA, (
        "the gguf Embedder was constructed without the candidate revision in "
        "place, so the bench validated nothing about the pin"
    )
    st_loads = [rec for rec in fake_loaders if rec["backend"] == "st"]
    assert len(st_loads) == 1
    assert st_loads[0]["pin_at_load"] is None, (
        "the injected pin leaked past the gguf load"
    )
    assert embed_mod.QWEN3_EMBEDDING_GGUF_REVISION is None


def test_backend_st_benches_st_alone_with_no_pin_talk(harness, fake_loaders, capsys):
    rc = harness.main(["--backend", "st"])
    out = capsys.readouterr().out

    assert rc == 0
    assert [rec["backend"] for rec in fake_loaders] == ["st"]
    assert "tok/s" in out
    assert "QWEN3_EMBEDDING_GGUF_REVISION" not in out


def test_resolve_and_pin_without_the_download_opt_in_is_refused_early(
    harness, fake_loaders, monkeypatch, capsys
):
    """Resolving a sha is a network call; the codebase has exactly ONE opt-in
    for model-related network and the harness must not sneak past it. The
    refusal happens before any resolution or load, and names the fix."""

    def _must_not_resolve():
        raise AssertionError("resolve_gguf_sha was called without the opt-in")

    monkeypatch.setattr(harness, "resolve_gguf_sha", _must_not_resolve)
    monkeypatch.delenv(embed_mod.MODEL_DOWNLOAD_ENV, raising=False)

    rc = harness.main(["--backend", "gguf", "--resolve-and-pin"])
    err = capsys.readouterr().err

    assert rc != 0
    assert embed_mod.MODEL_DOWNLOAD_ENV in err, (
        "the refusal never names the env var that unlocks the run"
    )
    assert fake_loaders == [], "a model was loaded on the way to refusing"


def test_resolve_and_pin_benches_at_the_resolved_sha(
    harness, fake_loaders, monkeypatch, capsys
):
    resolved = []

    def _fake_resolve():
        resolved.append(True)
        return _SHA

    monkeypatch.setattr(harness, "resolve_gguf_sha", _fake_resolve)
    monkeypatch.setenv(embed_mod.MODEL_DOWNLOAD_ENV, "1")

    rc = harness.main(["--backend", "gguf", "--resolve-and-pin"])
    out = capsys.readouterr().out

    assert rc == 0
    assert resolved == [True]
    gguf_loads = [rec for rec in fake_loaders if rec["backend"] == "gguf"]
    assert gguf_loads and gguf_loads[0]["pin_at_load"] == _SHA
    assert harness.pin_line(_SHA) in out


def test_an_explicit_revision_wins_and_skips_resolution(
    harness, fake_loaders, monkeypatch, capsys
):
    """``--resolve-and-pin --gguf-revision <sha>`` re-validates a KNOWN sha:
    no resolution call, no need for the download opt-in just to resolve."""

    def _must_not_resolve():
        raise AssertionError("resolved despite an explicit --gguf-revision")

    monkeypatch.setattr(harness, "resolve_gguf_sha", _must_not_resolve)
    monkeypatch.delenv(embed_mod.MODEL_DOWNLOAD_ENV, raising=False)

    rc = harness.main(["--backend", "gguf", "--gguf-revision", _SHA, "--resolve-and-pin"])
    out = capsys.readouterr().out

    assert rc == 0
    assert harness.pin_line(_SHA) in out


def test_gguf_flags_on_the_st_backend_are_refused_not_ignored(harness, capsys):
    rc = harness.main(["--backend", "st", "--resolve-and-pin"])
    assert rc != 0
    assert "gguf" in capsys.readouterr().err


def test_run_bench_leaves_os_environ_unchanged(harness, monkeypatch, capsys):
    """The scratch XDG_DATA_HOME must not leak out of the bench.

    run_bench pointed XDG_DATA_HOME at /tmp scratch via a process-global
    setdefault, so any test (or caller) invoking main() with the var unset
    inherited the mutation for the rest of the process. The scratch value
    must be in effect during the model load and gone afterwards.
    """
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    seen = []

    def _fake_load(backend):
        seen.append(os.environ.get("XDG_DATA_HOME"))
        return _FakeEmbedder(backend)

    monkeypatch.setattr(harness, "_load_embedder", _fake_load)

    rc = harness.main(["--backend", "st"])
    capsys.readouterr()

    assert rc == 0
    assert seen == ["/tmp/aggregator-measure-xdg"], (
        "the model load ran against the REAL data dir, not the scratch one"
    )
    assert "XDG_DATA_HOME" not in os.environ, (
        "run_bench leaked the scratch XDG_DATA_HOME into os.environ"
    )


def test_run_bench_respects_a_preset_xdg_data_home(harness, monkeypatch, capsys):
    """An operator's own XDG_DATA_HOME (the documented mktemp -d invocation)
    keeps winning, exactly as setdefault behaved."""
    monkeypatch.setenv("XDG_DATA_HOME", "/custom/xdg")
    seen = []

    def _fake_load(backend):
        seen.append(os.environ.get("XDG_DATA_HOME"))
        return _FakeEmbedder(backend)

    monkeypatch.setattr(harness, "_load_embedder", _fake_load)

    rc = harness.main(["--backend", "st"])
    capsys.readouterr()

    assert rc == 0
    assert seen == ["/custom/xdg"]
    assert os.environ.get("XDG_DATA_HOME") == "/custom/xdg"


def test_the_env_sha_on_the_st_backend_is_refused_like_the_flag(
    harness, fake_loaders, monkeypatch, capsys
):
    """--backend st refuses --gguf-revision; the env form must not slip past.

    An operator who exported AGGREGATOR_GGUF_BENCH_REVISION and benched st
    would have the sha silently unused — the flag form errors, so the env
    form must error identically, naming the env var.
    """
    monkeypatch.setenv(harness.GGUF_REVISION_ENV, _SHA)

    rc = harness.main(["--backend", "st"])
    err = capsys.readouterr().err

    assert rc != 0, "an env sha with --backend st was silently ignored"
    assert harness.GGUF_REVISION_ENV in err, (
        "the refusal never names the env var that triggered it"
    )
    assert fake_loaders == [], "a model was loaded on the way to refusing"


def test_resolve_and_pin_with_the_env_sha_set_is_refused_as_ambiguous(
    harness, fake_loaders, monkeypatch, capsys
):
    """A stale env sha must not masquerade as a fresh hub resolution.

    With AGGREGATOR_GGUF_BENCH_REVISION exported, --resolve-and-pin would
    silently bench (and print a pin line for) the OLD env sha while the
    operator believes it was just resolved. Ambiguous input — fail loudly.
    """

    def _must_not_resolve():
        raise AssertionError("resolved despite the ambiguous env sha")

    monkeypatch.setattr(harness, "resolve_gguf_sha", _must_not_resolve)
    monkeypatch.setenv(embed_mod.MODEL_DOWNLOAD_ENV, "1")
    monkeypatch.setenv(harness.GGUF_REVISION_ENV, _SHA)

    rc = harness.main(["--backend", "gguf", "--resolve-and-pin"])
    err = capsys.readouterr().err

    assert rc != 0, "--resolve-and-pin quietly used the env sha as the pin"
    assert harness.GGUF_REVISION_ENV in err
    assert "--resolve-and-pin" in err
    assert fake_loaders == [], "a model was loaded on the way to refusing"


def test_help_documents_the_flag_over_env_precedence(harness, capsys):
    """Finding C asks for the precedence contract in --help: the flag wins
    over the env on the non-resolve path, and the env conflicts with
    --resolve-and-pin."""
    with pytest.raises(SystemExit):
        harness.parse_args(["--help"])
    out = capsys.readouterr().out
    assert harness.GGUF_REVISION_ENV in out
    assert "flag wins" in out
    assert "--resolve-and-pin" in out


def test_a_bench_input_without_backend_is_refused_not_swept(
    harness, monkeypatch, capsys
):
    """A forgotten ``--backend gguf`` must not burn a ~30-minute st sweep.

    The bare invocation stays the legacy sweep ONLY when no bench-mode input
    is present. If --resolve-and-pin, --gguf-revision, or the env sha arrives
    without --backend, the run must exit non-zero naming the missing flag —
    not silently ignore the input and start sweeping.
    """

    def _must_not_sweep():
        raise AssertionError("legacy sweep started despite a bench-mode input")

    monkeypatch.setattr(harness, "legacy_sweep", _must_not_sweep)
    monkeypatch.delenv(harness.GGUF_REVISION_ENV, raising=False)

    for argv, env_sha in (
        (["--resolve-and-pin"], None),
        (["--gguf-revision", _SHA], None),
        ([], _SHA),
    ):
        if env_sha is not None:
            monkeypatch.setenv(harness.GGUF_REVISION_ENV, env_sha)
        rc = harness.main(argv)
        err = capsys.readouterr().err
        if env_sha is not None:
            monkeypatch.delenv(harness.GGUF_REVISION_ENV)

        assert rc != 0, f"{argv} with env={env_sha!r} fell through to the sweep"
        assert "--backend" in err, (
            f"the refusal for {argv} never names the missing flag"
        )
        if env_sha is not None:
            assert harness.GGUF_REVISION_ENV in err, (
                "the env-triggered refusal never names the env var"
            )


def test_bare_invocation_with_no_bench_input_still_sweeps(harness, monkeypatch):
    """The doc's 40 tok/s table came from the bare command; keep it working."""
    swept = []
    monkeypatch.setattr(harness, "legacy_sweep", lambda: (swept.append(True), 0)[1])
    monkeypatch.delenv(harness.GGUF_REVISION_ENV, raising=False)
    assert harness.main([]) == 0
    assert swept == [True]


def _real_unpinned_download_error() -> RuntimeError:
    """embed.py's genuine refusal, produced by the code that raises it.

    Generated rather than retyped so the harness's detection is tested
    against the message that will actually reach it — if embed.py rewords
    the refusal, this fails instead of silently testing a stale copy.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(embed_mod.MODEL_DOWNLOAD_ENV, "1")
        with pytest.raises(RuntimeError) as excinfo:
            embed_mod.Embedder._gguf_revision(embed_mod._DEFAULT_MODEL_GGUF)
    return excinfo.value


def test_the_unpinned_download_refusal_gets_the_harness_remedy(
    harness, monkeypatch, capsys
):
    """--backend gguf + download opt-in + no revision must not end in a raw
    traceback whose remedy says to edit the constant. The harness catches
    embed.py's refusal and exits with its OWN remedy: the bench validates a
    sha BEFORE the constant is edited, via --resolve-and-pin or
    --gguf-revision. embed.py itself stays unchanged."""
    error = _real_unpinned_download_error()

    def _fake_load(backend):
        if backend == "gguf":
            raise error
        return _FakeEmbedder(backend)

    monkeypatch.setattr(harness, "_load_embedder", _fake_load)
    monkeypatch.setenv(embed_mod.MODEL_DOWNLOAD_ENV, "1")
    monkeypatch.delenv(harness.GGUF_REVISION_ENV, raising=False)

    rc = harness.main(["--backend", "gguf"])
    err = capsys.readouterr().err

    assert rc != 0
    assert "--gguf-revision" in err and "--resolve-and-pin" in err, (
        "the harness's remedy never names its own flags — the operator is "
        "left with embed.py's edit-the-constant instruction"
    )


def test_an_unrelated_runtime_error_still_propagates(harness, monkeypatch):
    """Only the unpinned-download refusal gets translated; anything else out
    of the loader is a real failure and must keep its traceback."""

    def _fake_load(backend):
        raise RuntimeError("no embedder backend loaded")

    monkeypatch.setattr(harness, "_load_embedder", _fake_load)

    with pytest.raises(RuntimeError, match="no embedder backend loaded"):
        harness.main(["--backend", "gguf"])


def test_resolve_gguf_sha_asks_the_hub_for_the_default_gguf_repo(
    harness, monkeypatch
):
    """The repo id comes from ``embed.py``, not a second copy here — the pin
    must be resolved for exactly the repository the loader will fetch."""
    import huggingface_hub

    asked = []

    class _FakeApi:
        def model_info(self, repo_id):
            asked.append(repo_id)
            return types.SimpleNamespace(sha=_SHA)

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)
    assert harness.resolve_gguf_sha() == _SHA
    assert asked == [embed_mod._DEFAULT_MODEL_GGUF]


def test_resolve_gguf_sha_refuses_a_hub_answer_that_is_not_a_sha(
    harness, monkeypatch
):
    import huggingface_hub

    class _FakeApi:
        def model_info(self, repo_id):
            return types.SimpleNamespace(sha="main")

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)
    with pytest.raises(ValueError):
        harness.resolve_gguf_sha()
