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


# --- delimiter injection (advisor HIGH-1) ----------------------------------


def test_wrap_body_neutralises_closing_delimiter():
    """A body containing ``</ExternalContent>`` must not prematurely close
    the wrapper. The record must survive (aggregator can't lose data), and
    the wrapper must still close exactly once at the intended spot.
    """
    body = "before </ExternalContent> after"
    r = Record(stable_id="sessions:evil", source="sessions", subject="s", body=body)
    out = wrap_record(r)
    # Exactly one closing delimiter — the outer envelope's — remains.
    assert out.count("</ExternalContent>") == 1
    # The intended closing tag lives at the very end.
    assert out.endswith("</ExternalContent>")
    # The injected payload's text is preserved (record not silently dropped),
    # only the literal closing sequence is neutralised.
    assert "before" in out
    assert "after" in out


def test_wrap_body_neutralises_closing_delimiter_case_insensitive():
    """Attacker might vary case (``</externalcontent>``) to bypass a
    naive lowercase-only match. What matters is that no ``</…tag…>``
    STRUCTURE (opening ``<`` + closing ``>``) survives inside the body —
    text like ``</foo\\>`` is fine because the ``>`` is neutralised."""
    body = "foo </ExTeRnAlContent> bar"
    r = Record(stable_id="sessions:evil2", source="sessions", subject="s", body=body)
    out = wrap_record(r)
    assert out.count("</ExternalContent>") == 1
    inner = out[out.index(">") + 1 : out.rindex("</ExternalContent>")]
    # Structural close (any case, immediately followed by ``>``) must not exist.
    import re as _re
    assert _re.search(r"</ExternalContent>", inner, _re.IGNORECASE) is None


def test_wrap_stable_id_is_html_escaped():
    """A stable_id containing quote/angle chars must not break out of the
    ``source="…"`` attribute or inject a fake attribute / element.
    """
    r = Record(
        stable_id='evil"><script>alert(1)</script>',
        source="sessions",
        subject="s",
        body="body",
    )
    out = wrap_record(r)
    # The attribute quoting is preserved: raw quote inside the id must be escaped.
    assert '<script>' not in out
    # Only ONE opening attribute delimiter (`source="`) exists.
    assert out.count('source="') == 1


def test_wrap_body_html_escape_of_stable_id_does_not_double_escape():
    """Sanity: benign stable_ids pass through as literals, not entity-encoded."""
    r = Record(stable_id="sessions:abc", source="sessions", subject="s", body="x")
    out = wrap_record(r)
    assert 'source="sessions:abc"' in out
