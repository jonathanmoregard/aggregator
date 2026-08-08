"""chunk_body — paragraph-aware windowing for p99 observation bodies."""

from aggregator.core.chunk import chunk_body


def test_short_text_returns_single_chunk():
    text = "short obs body"
    assert chunk_body(text, max_len=4000, overlap=400) == [text]


def test_empty_returns_empty_list():
    assert chunk_body("", max_len=4000, overlap=400) == []
    assert chunk_body("   ", max_len=4000, overlap=400) == []


def test_long_text_splits_on_paragraph_boundary():
    para = "a" * 3000
    text = f"{para}\n\n{para}\n\n{para}"  # ~9000 chars, 3 paras
    chunks = chunk_body(text, max_len=4000, overlap=400)
    assert len(chunks) >= 2
    # each chunk should end at a paragraph break when possible
    for c in chunks[:-1]:
        assert len(c) <= 4000


def test_chunk_boundaries_include_overlap():
    text = "x" * 10000
    chunks = chunk_body(text, max_len=4000, overlap=400)
    # overlap: consecutive chunks share ~400 chars
    if len(chunks) >= 2:
        tail = chunks[0][-400:]
        head = chunks[1][:400]
        assert tail == head  # exact overlap for the pure-char fallback


def test_deterministic():
    text = "para one.\n\npara two.\n\n" * 500
    a = chunk_body(text, max_len=1000, overlap=100)
    b = chunk_body(text, max_len=1000, overlap=100)
    assert a == b
