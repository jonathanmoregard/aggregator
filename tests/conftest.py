import os

import pytest


@pytest.fixture
def tmp_data_home(tmp_path, monkeypatch):
    """Point XDG_DATA_HOME at a temp dir so cache.db lives in an isolated location."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
