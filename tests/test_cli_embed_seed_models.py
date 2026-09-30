"""``aggregator embed --seed-models`` — fetch the weights, touch nothing else.

``aggregator-embed-seed.service`` ran ``embed --once --source observations``,
which is the wrong command for the job in two separate ways.

It builds only the ``Embedder``. NOTHING in the deployment ever built the
``Reranker``, so its weights were never fetched by any path — and every model
load is offline unless a human opts in, which means ``rerank=True`` was
guaranteed to fail on this machine forever, on every surface, no matter how
long the seed unit ran.

And it embeds a real row of an untrusted corpus purely to warm a cache: it
opens the database, migrates it, takes the embed lock, claims a row, and
advances a watermark, all as a side effect of "make sure the model is
downloaded". Seeding a model cache has no business writing to the index.

So: a command that constructs both models, touches no database at all, and
— when the weights are absent and downloads are not permitted — says so and
names the command that fixes it.

NOTHING HERE MAY DOWNLOAD ANYTHING. Both constructors are stubbed in every
test; what is asserted is which constructors run and what the process does
with a failure, never whether a real file appears.
"""

from __future__ import annotations

import pytest

from aggregator import cli
from aggregator.core import embed as embed_mod


@pytest.fixture
def built(monkeypatch):
    """Record which models were constructed, and under which download gate."""
    built: list[tuple[str, bool]] = []

    class FakeEmbedder:
        def __init__(self, *a, **kw):
            built.append(("embedder", embed_mod.downloads_allowed()))

    class FakeReranker:
        def __init__(self, *a, **kw):
            built.append(("reranker", embed_mod.downloads_allowed()))

    import aggregator.core.rerank as rerank_mod

    monkeypatch.setattr(cli, "seed_embedder", FakeEmbedder)
    monkeypatch.setattr(rerank_mod, "Reranker", FakeReranker)
    return built


def test_the_seed_command_exists_under_the_name_the_unit_calls(capsys):
    """THE REPRO. The nix unit's ExecStart is this exact string."""
    with pytest.raises(SystemExit):
        cli.main(["embed", "--help"])
    assert "--seed-models" in capsys.readouterr().out


def test_it_builds_both_models(tmp_data_home, built, capsys):
    """The reranker is the one nothing else ever built."""
    rc = cli.main(["embed", "--seed-models"])

    assert rc == 0, capsys.readouterr().err
    assert [name for name, _ in built] == ["embedder", "reranker"]


def test_it_touches_no_database(tmp_data_home, built):
    """No open, no migrate, no lock, no row: this warms a model cache."""
    cli.main(["embed", "--seed-models"])

    assert not (tmp_data_home / "aggregator" / "cache.db").exists(), (
        "seeding the model cache created the index database"
    )


def test_it_is_offline_unless_the_one_switch_is_set(tmp_data_home, built, monkeypatch):
    """Reuses round 1's opt-in rather than inventing a second one."""
    monkeypatch.delenv(embed_mod.MODEL_DOWNLOAD_ENV, raising=False)
    cli.main(["embed", "--seed-models"])
    assert [allowed for _, allowed in built] == [False, False]

    built.clear()
    monkeypatch.setenv(embed_mod.MODEL_DOWNLOAD_ENV, "1")
    cli.main(["embed", "--seed-models"])
    assert [allowed for _, allowed in built] == [True, True]


def test_absent_weights_are_loud_and_name_the_fix(tmp_data_home, monkeypatch, capsys):
    """Never a silent no-op: the whole point is to know the cache is warm."""

    class Missing:
        def __init__(self, *a, **kw):
            raise OSError("not in the local cache and local_files_only=True")

    import aggregator.core.rerank as rerank_mod

    monkeypatch.setattr(cli, "seed_embedder", Missing)
    monkeypatch.setattr(rerank_mod, "Reranker", Missing)
    monkeypatch.delenv(embed_mod.MODEL_DOWNLOAD_ENV, raising=False)

    rc = cli.main(["embed", "--seed-models"])

    err = capsys.readouterr().err
    assert rc != 0, f"a failed seed exited 0; stderr={err!r}"
    assert embed_mod.MODEL_DOWNLOAD_ENV in err, (
        f"the refusal does not name the command a human should run: {err!r}"
    )
    assert "--seed-models" in err


def test_both_models_are_reported_not_just_the_first(
    tmp_data_home, monkeypatch, capsys
):
    """One run, one complete answer — not one fix per invocation."""

    class Missing:
        def __init__(self, *a, **kw):
            raise OSError("not in the local cache and local_files_only=True")

    import aggregator.core.rerank as rerank_mod

    monkeypatch.setattr(cli, "seed_embedder", Missing)
    monkeypatch.setattr(rerank_mod, "Reranker", Missing)

    cli.main(["embed", "--seed-models"])

    err = capsys.readouterr().err
    assert "embedder" in err.lower()
    assert "reranker" in err.lower()


def test_a_successful_seed_says_so(tmp_data_home, built, capsys):
    """Silence on success is how nobody notices the unit stopped working."""
    cli.main(["embed", "--seed-models"])

    out = capsys.readouterr().out
    assert "embedder" in out.lower()
    assert "reranker" in out.lower()


def test_seed_models_cannot_be_combined_with_a_batch_mode(capsys):
    """It is a mode, not a modifier: --once/--catchup mean something else."""
    with pytest.raises(SystemExit):
        cli.main(["embed", "--seed-models", "--once"])


# -- the server backend: the file the unit serves, fetched without the unit ---


@pytest.fixture
def hub(monkeypatch):
    """Record the hub call; stub the reranker. Nothing may be downloaded."""
    import huggingface_hub

    import aggregator.core.rerank as rerank_mod

    seen: dict = {}

    def _recorder(**kwargs):
        seen.update(kwargs)
        return f"/nonexistent/{kwargs.get('filename')}"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _recorder)
    monkeypatch.setattr(rerank_mod, "Reranker", lambda *a, **k: None)
    monkeypatch.delenv("AGGREGATOR_EMBED_BACKEND", raising=False)
    return seen


@pytest.mark.parametrize("allowed", [True, False])
def test_seeding_the_server_backend_fetches_the_pinned_file_without_the_server(
    tmp_data_home, hub, monkeypatch, capsys, allowed
):
    """The seed unit runs with the server STOPPED on a fresh machine — the
    server cannot start until this has run — so seeding must not need it.

    And what it fetches must be exactly the bytes the server unit will look
    for: the pinned revision of the pinned file, not whatever ``main`` holds.
    """
    from tests.embed_server_stub import closed_port_url

    monkeypatch.setenv(embed_mod.EMBED_URL_ENV, closed_port_url())
    if allowed:
        monkeypatch.setenv(embed_mod.MODEL_DOWNLOAD_ENV, "1")
    else:
        monkeypatch.delenv(embed_mod.MODEL_DOWNLOAD_ENV, raising=False)

    rc = cli.main(["embed", "--seed-models"])

    assert rc == 0, capsys.readouterr().err
    assert hub["repo_id"] == embed_mod._DEFAULT_MODEL_GGUF
    assert hub["filename"] == embed_mod.QWEN3_EMBEDDING_GGUF_FILENAME
    rev = embed_mod.QWEN3_EMBEDDING_GGUF_REVISION
    assert isinstance(rev, str) and len(rev) == 40 and int(rev, 16) >= 0
    assert hub["revision"] == rev
    assert hub["local_files_only"] is (not allowed)
