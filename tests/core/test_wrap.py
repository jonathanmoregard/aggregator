from aggregator.core.wrap import wrap_record, wrap_records
from aggregator.sources.base import Record


def test_wrap_record_uses_stable_id_as_source():
    r = Record(stable_id="sessions:abc", source="sessions", subject="s", body="hello")
    out = wrap_record(r)
    assert out.startswith('<ExternalContent source="sessions:abc">')
    assert out.endswith("</ExternalContent>")
    assert "hello" in out


def test_wrap_records_joins_with_blank_line():
    r1 = Record(stable_id="a:1", source="a", subject="s", body="one")
    r2 = Record(stable_id="a:2", source="a", subject="s", body="two")
    out = wrap_records([r1, r2])
    assert out.count("<ExternalContent") == 2
    assert "\n\n" in out


def test_wrap_records_empty_list_returns_empty_string():
    assert wrap_records([]) == ""
