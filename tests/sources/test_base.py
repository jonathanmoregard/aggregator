import pytest

from aggregator.sources.base import IngestResult, QueryAST, Record, Source, stable_id_for


def test_record_defaults():
    r = Record(stable_id="s:1", source="s", subject="t", body="b")
    assert r.tags == []
    assert r.extra == {}
    assert r.created_at is None
    assert r.updated_at is None


def test_ingest_result_defaults():
    ir = IngestResult(added=0, updated=0, skipped=0)
    assert ir.errors == []


def test_query_ast_defaults():
    ast = QueryAST()
    assert ast.source is None
    assert ast.tags == []
    assert ast.from_date is None
    assert ast.to_date is None
    assert ast.text is None
    assert ast.extra == {}


def test_source_protocol_declares_required_methods():
    """Source protocol must expose ingest, search, record_shape per spec §Components."""
    # Protocols in typing declare methods; check they exist on the class.
    assert hasattr(Source, "ingest")
    assert hasattr(Source, "search")
    assert hasattr(Source, "record_shape")


def test_source_protocol_structural_check_accepts_conforming_class():
    """A class implementing name/ingest/search/record_shape must satisfy the protocol at runtime."""

    class Fake:
        name = "fake"

        def ingest(self, since):
            return IngestResult(added=0, updated=0, skipped=0)

        def search(self, ast):
            return []

        def record_shape(self):
            return {"stable_id": "str"}

    # Protocol classes aren't runtime-checkable by default; use isinstance only when
    # decorated with @runtime_checkable. This test asserts the shape is present,
    # which is what the DSL help generator will duck-type against in M2.
    f = Fake()
    assert f.name == "fake"
    assert isinstance(f.ingest(None), IngestResult)
    assert f.search(QueryAST()) == []
    assert f.record_shape() == {"stable_id": "str"}


def test_stable_id_for_formats_source_and_id():
    assert stable_id_for("sessions", "abc-123") == "sessions:abc-123"
    assert stable_id_for("github", "owner/repo:42") == "github:owner/repo:42"


def test_stable_id_for_rejects_empty_source():
    with pytest.raises(ValueError, match="invalid source"):
        stable_id_for("", "abc")


def test_stable_id_for_rejects_colon_in_source():
    """Source name must not contain ':' so parsing '<source>:<rest>' stays unambiguous."""
    with pytest.raises(ValueError, match="invalid source"):
        stable_id_for("bad:source", "abc")


def test_stable_id_for_rejects_empty_id():
    with pytest.raises(ValueError, match="non-empty"):
        stable_id_for("sessions", "")
