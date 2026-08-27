"""A missing cache must arrive as a named error, not a bare SQLite one.

``Store.vector_available`` reads like a free boolean and is not: touching it
opens the database and loads a native extension. Against a cache that is not
there — which is the ordinary state of a fresh machine, and of the MCP server
started before the first ingest — it raised
``sqlite3.OperationalError: unable to open database file``.

That is the same exception type a malformed query raises, so no caller could
tell "there is no cache yet, run an ingest" apart from "that query was
wrong". Exactly the distinction ``VectorIndexUnavailableError`` already exists
to draw for the vector arm.

``CacheUnavailableError`` subclasses ``sqlite3.OperationalError`` on purpose:
every handler already written against the broad type keeps working unchanged,
and callers that want to be specific now can be.
"""

import sqlite3

import pytest

from aggregator.core.store import SCHEMA_VERSION, CacheUnavailableError, Store


@pytest.fixture
def missing_cache(tmp_path):
    return tmp_path / "does-not-exist" / "cache.db"


def test_vector_available_names_the_missing_cache(missing_cache):
    store = Store(db_path=missing_cache, read_only=True)
    with pytest.raises(CacheUnavailableError) as excinfo:
        store.vector_available  # noqa: B018 - the property access IS the call
    assert str(missing_cache) in str(excinfo.value)


def test_schema_version_names_the_missing_cache(missing_cache):
    store = Store(db_path=missing_cache, read_only=True)
    with pytest.raises(CacheUnavailableError):
        store.schema_version()


def test_the_error_says_what_to_do_about_it(missing_cache):
    store = Store(db_path=missing_cache, read_only=True)
    with pytest.raises(CacheUnavailableError) as excinfo:
        store.vector_available  # noqa: B018
    assert "ingest" in str(excinfo.value).lower()


def test_existing_broad_handlers_still_catch_it(missing_cache):
    """The MCP layer catches sqlite3.OperationalError; it must keep working."""
    store = Store(db_path=missing_cache, read_only=True)
    with pytest.raises(sqlite3.OperationalError):
        store.vector_available  # noqa: B018


def test_a_writable_store_still_creates_its_cache(tmp_path):
    """Only the read-only path may refuse — ingest is what CREATES the cache."""
    store = Store(db_path=tmp_path / "cache.db")
    store.migrate()
    assert store.schema_version() == SCHEMA_VERSION


def test_a_present_cache_is_unaffected(tmp_path):
    db = tmp_path / "cache.db"
    Store(db_path=db).migrate()
    reader = Store(db_path=db, read_only=True)
    assert reader.vector_available in (True, False)
    assert reader.schema_version() == SCHEMA_VERSION


def test_the_mcp_surface_reports_a_missing_cache_as_a_structured_refusal(
    tmp_path, monkeypatch
):
    """The end a user actually sees: a reason, not a traceback."""
    from aggregator.mcp import aggregator_query

    store = Store(db_path=tmp_path / "gone" / "cache.db", read_only=True)
    result = aggregator_query("anything", _store=store)
    assert result["ok"] is False
    assert "cache" in result["reason"].lower()
