import os

import pytest


@pytest.fixture
def tmp_data_home(tmp_path, monkeypatch):
    """Point XDG_DATA_HOME at a temp dir so cache.db lives in an isolated location."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def _isolated_downloads_dir(tmp_path, monkeypatch):
    """Chat-export discovery also scans ``~/Downloads`` (Chunk 4 owner change).

    Tests must never touch the real Downloads dir — default the override env
    to a nonexistent path so discovery only sees dirs a test explicitly
    creates. Tests exercising the Downloads path monkeypatch
    ``AGGREGATOR_DOWNLOADS_DIR`` themselves (test-body setenv wins over this
    autouse default).
    """
    monkeypatch.setenv(
        "AGGREGATOR_DOWNLOADS_DIR", str(tmp_path / "downloads-nonexistent")
    )


@pytest.fixture(autouse=True)
def _isolated_state_home(tmp_path, monkeypatch):
    """No test may read or write the developer's own TickTick open-task baseline.

    ``ticktick_api.default_state_path()`` resolves under ``$XDG_STATE_HOME``,
    and ``aggregator status`` now reads that file to list the uncovered
    projects it is holding. Unset, that is the real ``~/.local/state`` copy of
    the user's live task payloads — titles and notes — so a status test's
    output would depend on whose machine ran it, and a test that saved state
    would edit the user's real baseline. Same reasoning as
    ``_isolated_downloads_dir``; tests that care about the fallback delete or
    blank the variable in their own body, which wins over this.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state-home"))


@pytest.fixture
def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
