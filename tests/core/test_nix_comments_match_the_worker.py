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

A third claim in the same comment was false from the start rather than gone
stale: "A row is sub-second, so 5min is wildly generous on purpose". A row's
encode is one ``embed_documents`` call over all of that row's chunks and the
live cache holds rows of 257 chunks — about 86 minutes — so the window does
not cover a long row and never did. The chunk-bounded-batches branch then put
the true version ("a ``TimeoutStopSec`` is far shorter than that") in
``_embed_batch``'s docstring and left the false one standing next to it. The
third test below pins the comfort, not the number: ``5min`` is a deliberate
bound on a wedged worker and nothing here argues about its value.

A fourth claim went wrong the same way and cost an outage rather than a
misreading. ``interval``'s description explained concurrent-write safety as
"WAL journal mode plus a 30s busy_timeout … one short write transaction per
batch rather than one long one per run" — true of the embed worker, false of
ingest, which writes a whole source in one transaction (measured holding the
write lock for 139 s). ``aggregator-embed.service`` then failed on every tick
for at least five hours on 2026-08-30. The last two tests pin the replacement:
the falsified claim may not return, and the lock the module now credits has to
be one the store actually takes.

Deliberately narrow. It does not try to validate prose in general; it pins
the specific sentences that went wrong, in the direction they went wrong.
Neither ``cli.py`` nor ``store.py`` is modified by this test — they are read.
"""

import re
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


def _unwrapped(text):
    """One long line, so a pinned sentence survives being re-wrapped.

    The claims below are prose inside a Nix string, and Nix prose gets
    reflowed by anyone who edits the paragraph around it. Matching against the
    literal source would make these tests fail on a reflow and pass on a
    rewrite, which is exactly backwards.
    """
    return re.sub(r"\s+", " ", text)


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


#: Wordings that make a row sound short enough for ``TimeoutStopSec`` to cover.
#: Each is affirmative, so stating the truth in the negative ("a row is not
#: short") does not trip it — with the deliberate exception of ``sub-second``
#: itself, which is the literal stale claim and is worth catching even inside a
#: denial. Say what a row actually costs instead of negating the old word.
_COMFORTING_ABOUT_ROW_LENGTH = (
    "sub-second",
    "subsecond",
    "a row is short",
    "rows are short",
    "a row is quick",
    "wildly generous",
    "generous on purpose",
)


def test_the_module_does_not_call_a_row_short_enough_for_the_stop_window(sources):
    """A row is ONE ``embed_documents`` call, and the 5min window cannot cover it.

    Measured read-only against the live cache at the chunker's
    ``chunk-4000-400`` geometry and the measured ~20 s per chunk: 1348 rows
    (1298 observations + 50 records) each exceed 300 s in a single call, and
    the largest is 257 chunks — about 86 minutes. "A row is sub-second" was
    three orders of magnitude out.

    The comfort is the dangerous half, not the arithmetic. It reads as a
    reason not to worry about a stop landing mid-row, and a stop landing
    mid-row is precisely what escalates to SIGKILL, leaves the claim on disk,
    and gets ``_blame_crashed_row`` to book a good row into the poison ledger.
    So this pins two things: the comfort may not come back, and the gap must
    stay named.
    """
    encode_is_interruptible = (
        "for at in range(0, len(missing), _MAX_CHUNKS_PER_ENCODE):"
        in sources["cli"]
    )
    assert encode_is_interruptible, (
        "aggregator/cli.py no longer slices a row's encode, so the stop flag "
        "is only read once the whole row returns from the model. That is the "
        "86-minute window this comment used to document as an open gap: a "
        "stop landing inside it escalates to SIGKILL, the claim survives, and "
        "_blame_crashed_row condemns a good row. If the slicing was removed "
        "deliberately, nix/aggregator.nix's TimeoutStopSec rationale has to go "
        "back to naming the gap in the same commit."
    )
    nix = sources["nix"].lower()
    for comforting in _COMFORTING_ABOUT_ROW_LENGTH:
        assert comforting not in nix, (
            f"nix/aggregator.nix has reacquired {comforting!r} in its "
            f"TimeoutStopSec rationale. A row's encode is one "
            f"embed_documents() call over every chunk of that row, and the "
            f"live cache holds 1348 rows over 300 s plus one of 257 chunks "
            f"(~86 minutes), so no wording that makes a row sound short — or "
            f"the 5min window sound sufficient for one — is true. The comment "
            f"has to keep naming the gap instead: a stop mid-row escalates to "
            f"SIGKILL, the claim survives, and _blame_crashed_row condemns a "
            f"good row."
        )
    # Anchored on the evidence and on where the fix actually lives, rather
    # than on the prose around them: rewording the paragraph does not fail
    # this, deleting either half does.
    for required in ("257 chunks", "_MAX_CHUNKS_PER_ENCODE"):
        assert required in sources["nix"], (
            f"nix/aggregator.nix no longer carries {required!r} beside "
            f"TimeoutStopSec. Both halves have to stay. The measured worst "
            f"case is what makes 'a row is short' checkable rather than a "
            f"matter of taste — and it is still true, because 5min still does "
            f"not cover an 86-minute row. What changed is that it stopped "
            f"mattering, and the pointer to _MAX_CHUNKS_PER_ENCODE is the only "
            f"thing that tells the next reader WHY: the safety is in the "
            f"worker's sliced encode, not in this number. Lose that pointer "
            f"and someone reasonably concludes the window is what protects "
            f"them, then edits it. If the corpus was re-measured, update the "
            f"number here and there together."
        )


def test_the_module_does_not_credit_the_busy_timeout_for_concurrent_safety(sources):
    """The claim that failed in production, pinned so it cannot come back.

    ``interval``'s description used to end: "What makes overlap safe is on the
    store side - WAL journal mode plus a 30s busy_timeout - and on the worker
    side, one short write transaction per batch rather than one long one per
    run." The first half is a backstop and the second half was never true of
    ingest: ``cli._ingest_entities`` hands a whole source to ONE
    ``upsert_entities`` call, and ``_do_write_entities`` chunks the hash probe
    without chunking the commit.

    Measured on the live cache 2026-08-31 by sampling ``BEGIN IMMEDIATE`` at
    1Hz through one ordinary ingest run: contiguous write-lock holds of 49s,
    29s and 139s. Against 139s a 30s timeout is not unlucky, it is arithmetic
    - and ``aggregator-embed.service`` duly died on every tick for at least
    five hours on 2026-08-30.

    So the sentence is banned in the affirmative. Quoting it as history, which
    the replacement text does, reads differently and must stay allowed - that
    is why the ban is on the CLAIM ("what makes overlap safe is ... busy") and
    not on the words "busy_timeout".
    """
    nix = _unwrapped(sources["nix"]).lower()
    assert "what makes overlap safe is on the store side" not in nix, (
        "nix/aggregator.nix has re-credited WAL + busy_timeout with making "
        "ingest/embed overlap safe. It does not: a busy_timeout is a deadline, "
        "and the thing it is waiting on is another aggregator process doing "
        "unbounded legitimate work (139 s measured, on a corpus that only "
        "grows). What makes overlap safe is Store._CacheWriteLock. If the lock "
        "was removed, this test and the module have to change together - and "
        "the removal needs a story for the 2026-08-30 outage."
    )
    assert "one short write transaction per batch rather than one long one per run" not in nix, (
        "nix/aggregator.nix again describes the writers as using one short "
        "transaction per batch. That is true of the embed worker and false of "
        "ingest, which is the asymmetry the whole outage lived in."
    )


def test_the_lock_the_module_credits_is_the_lock_the_store_takes(sources):
    """A conditional, like the rest of this file: claim it, then own it.

    The module now points a reader at ``Store._CacheWriteLock`` as the reason
    two units may share a cache. That pointer is only worth having while the
    lock is real AND is actually taken on the write path - a lock defined but
    never acquired would leave the module confidently describing a mechanism
    that does nothing, which is the failure mode this file exists for.
    """
    if "_CacheWriteLock" not in sources["nix"]:
        pytest.skip("the module no longer credits the lock; nothing to check")

    assert "class _CacheWriteLock:" in sources["store"], (
        "nix/aggregator.nix credits Store._CacheWriteLock for concurrent-write "
        "safety, but aggregator/core/store.py no longer defines it."
    )
    ensure_writable = sources["store"].split("def _ensure_writable(self)")[1]
    body = ensure_writable.split("\n    def ")[0]
    assert "take_cache_write_lock()" in body, (
        "Store._ensure_writable no longer takes the cross-process write lock. "
        "It is the choke point every write method passes through, so an "
        "acquire anywhere else leaves some write path racing SQLite directly - "
        "which is the 2026-08-30 failure. If the acquire moved, move this "
        "assertion with it rather than deleting it."
    )
    assert "flock" in sources["store"], (
        "the write lock no longer uses flock, so it is no longer released by "
        "the kernel when a process is SIGKILLed mid-transaction. That property "
        "is what keeps a killed embed worker from wedging ingest."
    )
