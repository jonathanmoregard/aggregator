from __future__ import annotations

from aggregator.cli import _default_sources


def test_dropbox_registered():
    sources = _default_sources()
    assert "dropbox" in sources
    assert sources["dropbox"].name == "dropbox"


def test_every_registered_source_name_matches_its_key():
    for key, src in _default_sources().items():
        assert src.name == key
