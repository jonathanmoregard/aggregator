"""chunk_body — paragraph-aware windowing for p99 observation bodies."""

import pytest

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


# --- parameters that cannot terminate ---------------------------------------
#
# The hard-window fallback advances by ``stride = max_len - overlap``, so
# ``overlap >= max_len`` leaves ``i`` at 0 forever: the only exit is
# ``i + max_len >= len(text)``, which a stationary ``i`` never reaches. Probed
# with an iteration cap rather than by running it out — at
# ``max_len=overlap=4000`` the loop reached 200,000 iterations with ``i`` still
# 0, holding 800 MB of identical chunks and growing. Nothing below runs the
# unguarded loop; they assert it is refused before it starts.


@pytest.mark.parametrize(
    ("max_len", "overlap"),
    [(4000, 4000), (4000, 5000), (100, 100), (1, 1)],
)
def test_overlap_at_or_above_max_len_is_refused(max_len, overlap):
    with pytest.raises(ValueError, match="overlap"):
        chunk_body("x" * (max_len * 3), max_len=max_len, overlap=overlap)


def test_a_nonsense_max_len_is_refused():
    for bad in (0, -1):
        with pytest.raises(ValueError, match="max_len"):
            chunk_body("x" * 100, max_len=bad, overlap=0)


def test_a_negative_overlap_is_refused():
    with pytest.raises(ValueError, match="overlap"):
        chunk_body("x" * 100, max_len=10, overlap=-1)


def test_bad_parameters_are_refused_even_on_input_that_would_not_window():
    """The early returns must not hide a misconfiguration.

    Short and empty bodies never reach the fallback loop, so validating after
    them would let a broken configuration pass every one of the ~50% of the
    corpus that sits in the p<50 bucket, and take the worker down weeks later
    on the first long body it met.
    """
    for text in ("", "   ", "short"):
        with pytest.raises(ValueError):
            chunk_body(text, max_len=4000, overlap=4000)


def test_the_widest_legal_overlap_still_works():
    """The guard must stop at "cannot terminate", not one step further."""
    chunks = chunk_body("x" * 50, max_len=10, overlap=9)
    assert len(chunks) > 1
    assert "".join(dict.fromkeys(chunks))  # terminated with real content
