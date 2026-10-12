"""Does any document on a reranked page stand out from the rest?

The confidence signal behind ``mcp._maybe_rerank``: the cross-encoder scores a
page, and :func:`has_standout` asks whether the best score clears what the
largest of that many scores would reach by chance. Pure Python, so the MCP
server can import it at module scope without paying for the model stack.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from statistics import NormalDist


def relative_z(values: Sequence[float], *, min_sample: int) -> list[float] | None:
    """Per-query z-scores, one per input value.

    Returns ``None`` when the sample cannot support the estimate — fewer than
    ``min_sample`` values, or no spread at all.

    ``min_sample`` IS THE CALLER'S TO STATE, and is required rather than
    defaulted: a shared default is how two unrelated rules end up with one
    knob.

    ``None`` AND NOT A LIST OF ZEROS. "Undecidable" and "measured, nothing
    stands out" are opposite facts and must not share a representation: a
    caller that read 0.0 out of an unmeasurable sample would abstain on
    evidence it never had.
    """
    if len(values) < min_sample:
        return None
    spread = statistics.pstdev(values)
    if spread == 0.0:
        return None
    mean = statistics.fmean(values)
    return [(v - mean) / spread for v in values]


#: How much a reranker score has to beat its own page's NULL MAXIMUM by before
#: it counts as standing out.
#:
#: The judgement half of the bar. The other half is the null term
#: (:func:`expected_max_z`), a fact about the sample size: 1.87 at ``n = 20``,
#: which with this margin reproduces the 2.5 the bar was first calibrated at.
RERANK_STANDOUT_MARGIN = 0.63


def expected_max_z(n: int) -> float:
    """Where the largest of ``n`` draws sits, in sd above their own mean.

    WITH NOTHING RELEVANT AMONG THEM — this is the null, and it is the number a
    standout has to beat. Blom's plotting position, the standard approximation
    to the expected value of the largest normal order statistic: it gives 1.87
    at ``n = 20``.

    Grows with ``n``, which is the whole point: a fixed bar over a variable
    window silently changes what it is asserting every time the page size
    changes.
    """
    if n < 2:
        return 0.0
    return NormalDist().inv_cdf((n - 0.375) / (n + 0.25))


def standout_z_threshold(n: int) -> float:
    """The bar for a page of ``n`` scored documents."""
    return expected_max_z(n) + RERANK_STANDOUT_MARGIN


def _smallest_judgeable_sample() -> int:
    """The smallest page this rule can answer ``True`` about for real reasons.

    TWO CONDITIONS, BOTH ABOUT THE SAME HEADROOM. A standout has to clear the
    null maximum by :data:`RERANK_STANDOUT_MARGIN`; and the sample has to be
    large enough that clearing it is not the same thing as BEING the largest
    value the sample can hold. That ceiling is ``sqrt(n-1)`` — the z of one
    value among ``n-1`` identical others — and a page of identical
    cross-encoder scores is not a thing that happens, so a bar within a hair of
    it is a bar nothing can clear. Requiring the same margin between the bar
    and the ceiling is what keeps ``True`` reachable rather than theoretical.

    DERIVED AT IMPORT AND NOT WRITTEN DOWN, so it cannot drift away from the
    margin it comes from; ``tests/core/test_rerank_standout.py`` pins the value
    and both conditions.
    """
    n = 2
    while math.sqrt(n - 1) < standout_z_threshold(n) + RERANK_STANDOUT_MARGIN:
        n += 1
        if n > 1_000:  # pragma: no cover — the loop converges by n≈9
            raise RuntimeError("no sample size satisfies the standout margin")
    return n


#: Fewer scored documents than this and :func:`has_standout` answers ``None``.
RERANK_STANDOUT_MIN_SAMPLE = _smallest_judgeable_sample()

#: What the rule resolves to at a full ``mcp._RERANK_WINDOW`` page. Kept as a
#: reference point for the docs and pinned by a test; nothing reads it to make
#: a decision.
RERANK_STANDOUT_Z = standout_z_threshold(20)


def has_standout(
    values: Sequence[float], *, z_threshold: float | None = None
) -> bool | None:
    """Does anything in ``values`` stand out from the rest? ``None`` = can't say.

    ``values`` are scores where higher means more relevant. ``z_threshold``
    defaults to :func:`standout_z_threshold` FOR THIS SAMPLE; passing one
    overrides it, which is the escape hatch for a caller with its own
    calibration.

    THREE-VALUED ON PURPOSE. "Too few scores to judge" is not "nothing was
    relevant", and a caller that collapsed them would report low confidence for
    a three-hit page that simply had nothing to compare against.

    NO SPREAD ANSWERS ``False``, NOT ``None``. A cross-encoder that scored
    twenty documents identically has told you something: none of them stands
    out.
    """
    if len(values) < RERANK_STANDOUT_MIN_SAMPLE:
        return None
    bar = standout_z_threshold(len(values)) if z_threshold is None else z_threshold
    zs = relative_z(values, min_sample=RERANK_STANDOUT_MIN_SAMPLE)
    if zs is None:
        return False
    return max(zs) >= bar
