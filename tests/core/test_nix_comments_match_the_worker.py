"""The Nix module's claims about the worker must still be true of the worker.

Round 3's L1. ``nix/aggregator.nix`` justified ``TimeoutStopSec=5min`` with
two statements about the embed worker — "the worker installs no SIGTERM
handler" and "correctness comes from the commit ordering (vectors first,
watermark second)". Round 2 falsified both: ``cli.py`` wraps the embed loop in
``graceful_shutdown()``, and ``Store.commit_embed_batch`` became a single
transaction. The comment kept reading as an explanation while describing code
that no longer existed, which is the most expensive kind of stale comment —
the next person to touch that timeout would have reasoned from it.

A comment cannot be type-checked, but THIS one can be cross-referenced: it
makes falsifiable claims about two named behaviours in two named files. So
the assertion is a conditional — if the behaviour is present in the Python,
the Nix module may not claim its absence.

Deliberately narrow. It does not try to validate prose in general; it pins
the two specific sentences that went stale, in the direction they went stale.
Neither ``cli.py`` nor ``store.py`` is modified by this test — they are read.
"""

from pathlib import Path

import pytest


@pytest.fixture
def sources(repo_root):
    root = Path(repo_root)
    return {
        "nix": (root / "nix" / "aggregator.nix").read_text(),
        "cli": (root / "aggregator" / "cli.py").read_text(),
        "store": (root / "aggregator" / "core" / "store.py").read_text(),
    }


def test_the_module_does_not_deny_a_sigterm_handler_that_exists(sources):
    """``graceful_shutdown()`` around the embed loop IS a SIGTERM handler.

    It installs one that sets a flag rather than one that unwinds, which is
    the entire reason ``TimeoutStopSec`` buys anything: without a handler the
    stop would be immediate and the window would be dead weight.
    """
    installs_handler = "with graceful_shutdown() as stop:" in sources["cli"]
    assert installs_handler, (
        "aggregator/cli.py no longer wraps a loop in graceful_shutdown(). If "
        "the embed worker genuinely stopped handling SIGTERM, this test needs "
        "rewriting — but check first, because nix/aggregator.nix's "
        "TimeoutStopSec rationale depends on the handler existing."
    )
    assert "installs no SIGTERM handler" not in sources["nix"], (
        "nix/aggregator.nix says the embed worker installs no SIGTERM "
        "handler, but aggregator/cli.py wraps the embed loop in "
        "graceful_shutdown(), which installs one. The comment justifies "
        "TimeoutStopSec, so a reader reasoning from it reaches the wrong "
        "conclusion about what a stop costs."
    )


def test_the_module_does_not_claim_a_commit_ordering_that_was_removed(sources):
    """One transaction, not vectors-then-watermark in two commits.

    The distinction is load-bearing for the timeout's rationale: with two
    ordered commits the worst case was a repeated batch, so the window only
    bought re-work. With one transaction the batch rolls back cleanly and
    what the window actually protects is the in-flight row claim.
    """
    one_transaction = "ONE TRANSACTION, ONE GUARD" in sources["store"]
    assert one_transaction, (
        "aggregator/core/store.py's commit_embed_batch no longer documents "
        "itself as one transaction. If the ordering came back, nix/"
        "aggregator.nix's TimeoutStopSec comment needs revisiting with it."
    )
    for stale in ("vectors\n          # first, watermark second", "vectors first, watermark second"):
        assert stale not in sources["nix"], (
            f"nix/aggregator.nix still explains TimeoutStopSec with {stale!r}, "
            f"but commit_embed_batch commits the vectors and the watermark in "
            f"ONE transaction — there is no ordering between them to rely on, "
            f"and the failure the ordering guarded against is now impossible "
            f"by construction rather than by sequencing."
        )
