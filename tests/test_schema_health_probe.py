"""The detector for the skew that killed recall for three days and told nobody.

THE INCIDENT THESE TESTS ENCODE. On 2026-08-27 ``SCHEMA_VERSION`` went 5 -> 6
in the working tree (commit ``3c9a29d``). The MCP reader runs from that tree,
so it began refusing every recall call with "cache schema version 5 is older
than required version 6". The WRITER — ``aggregator-ingest.timer``, and the
``aggregator`` on ``$PATH`` — runs a Nix build pinned at ``4cb66f1a``, still
at 5. It re-stamped ``PRAGMA user_version = 5`` every 30 minutes and **exited
0** every time. ``OnFailure=`` never fires on a success, so for three days the
only symptom was an agent quietly grepping transcripts instead of recalling.

Nothing in the system could see this, because seeing it requires comparing
three quantities that live in three different places and no component holds
more than two of them.

WHY THESE TESTS MATTER MORE THAN USUAL. A health check is the one kind of code
whose bug is silence, and a test for a health check is the one kind of test
whose bug is passing. Every assertion below was run RED against a probe that
did not exist and then against deliberately broken variants — a probe that
returns FINE unconditionally passes none of them. In particular
``test_probe_never_writes_the_cache`` is the regression lock for the trap that
makes this whole class of bug self-concealing: ``cli.py`` calls ``migrate()``
on every subcommand but ``embed``, and ``migrate()`` WRITES ``user_version``,
so probing the cache with ``aggregator status`` re-stamps the very value it
was asked to report. A probe built the obvious way performs the damage it
exists to detect.
"""
from __future__ import annotations

import importlib
import itertools
import json
import os
import re
import sqlite3
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from aggregator.core.store import SCHEMA_VERSION
from aggregator.health import schema_probe as sp

# --- fixtures: the three quantities, each forgeable independently -----------
#
# The probe's whole job is to disagree with itself across three sources, so
# every test needs to set them apart. These build each one the way the real
# thing is built, not the way the probe reads it — a fixture that wrote what
# the probe expects to read would test nothing.


def _stamp_cache(path: Path, version: int, *, meta: int | None = None) -> Path:
    """A cache.db stamped exactly the way ``Store.migrate()`` stamps one.

    Both stamps, because ``migrate()`` writes both (``store.py`` at the
    ``PRAGMA user_version`` line and the ``meta`` upsert right after) and the
    probe cross-checks them against each other. ``meta`` defaults to matching
    ``version``; pass it explicitly to forge the disagreement.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    try:
        con.execute(f"PRAGMA user_version = {int(version)}")
        con.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)")
        con.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(version if meta is None else meta),),
        )
        con.commit()
    finally:
        con.close()
    return path


def _fake_tree(root: Path, version: int) -> Path:
    """A reader checkout: ``<root>/aggregator/core/store.py`` with a constant.

    Written as a real source file rather than a stub the probe is handed,
    because the probe reads the constant out of that file by text — the one
    way to learn the reader's requirement without importing the package (and
    dragging torch into a session-start budget).
    """
    core = root / "aggregator" / "core"
    core.mkdir(parents=True, exist_ok=True)
    (core / "store.py").write_text(
        "import os\n"
        "\n"
        f"SCHEMA_VERSION = {int(version)}\n"
        "\n"
        "class Store:\n    pass\n",
        encoding="utf-8",
    )
    return root


def _fake_writer(
    root: Path,
    version: int | None,
    *,
    name: str = "aggregator",
    outer_dir: str = "wrapper",
) -> Path:
    """A Nix-shaped install, wrapper indirection and all.

    Reproduces the real chain measured on this host: ``bin/<name>`` is a shell
    wrapper whose last line execs a second ``bin/<name>`` inside an env
    derivation, and only that env carries
    ``lib/python3.11/site-packages/aggregator/core/store.py``. A probe that only
    handled the direct case would report UNKNOWN against every real NixOS
    install, which is the only kind this host has.

    ONE fixture builds both sides, because the reader's chain has the identical
    shape — ``/etc/profiles/.../bin/aggregator-mcp`` -> a wrapper in an
    ``aggregator-0.0.1`` derivation -> a console script in an
    ``aggregator-env`` derivation that owns site-packages. ``name`` picks the
    program, and ``outer_dir`` puts the outer wrapper in a real ``bin/`` when
    the test needs it found on a ``PATH``. Mirrors the source, which resolves
    both through one walker: if the two ever needed different fixtures, they
    would need different walkers, and that is the drift this design refuses.

    ``version=None`` builds the wrapper chain but no packaged source, i.e. an
    install whose version cannot be determined.
    """
    env = root / "env"
    binroot = root / outer_dir
    (env / "bin").mkdir(parents=True, exist_ok=True)
    binroot.mkdir(parents=True, exist_ok=True)

    inner = env / "bin" / name
    inner.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    inner.chmod(0o755)

    outer = binroot / name
    outer.write_text(
        "#!/bin/sh\n"
        "PYTHONPATH=${PYTHONPATH%':'}\n"
        "export PYTHONPATH\n"
        f'exec "{inner}"  "$@"\n',
        encoding="utf-8",
    )
    outer.chmod(0o755)

    if version is not None:
        pkg = env / "lib" / "python3.11" / "site-packages" / "aggregator" / "core"
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "store.py").write_text(
            f"SCHEMA_VERSION = {int(version)}\n", encoding="utf-8"
        )
    return outer


def _claude_json(home: Path, entry: object | None) -> Path:
    """A ``~/.claude.json`` shaped like this host's, with one aggregator entry.

    ``entry=None`` means the file exists and configures no aggregator server.
    Anything else is written verbatim, including shapes that are not objects at
    all — the probe has to survive a hand-edited config, not only a valid one.

    The two decorations are not decoration. The real file carries FIVE
    top-level servers, so a probe that took "the only server" or "the first
    server" would pass a one-entry fixture and fail on the host. And it carries
    a stale PROJECT-scoped block —
    ``projects["/home/jonathan/Repos/gdocs-review-mcp"].mcpServers.aggregator``
    with ``args: ["not", "found"]``, left over from an unrelated repo — which
    applies only to sessions started in that directory. Both are in every
    fixture so a probe that ever starts reading either fails here rather than on
    the machine.
    """
    home.mkdir(parents=True, exist_ok=True)
    doc: dict = {
        "projects": {
            "/home/jonathan/Repos/gdocs-review-mcp": {
                "mcpServers": {
                    "aggregator": {"command": "aggregator-mcp", "args": ["not", "found"]}
                }
            }
        },
        "mcpServers": {"research-agent": {"command": "true", "args": []}},
    }
    if entry is not None:
        doc["mcpServers"]["aggregator"] = entry
    path = home / ".claude.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _env(home: Path, *, path: str = "") -> dict[str, str]:
    """The two environment values the reader resolution reads, and nothing else.

    A plain dict rather than a monkeypatched ``os.environ``: every resolver in
    the probe takes its environment as an argument, and that is the property
    that makes a PATH lookup testable without a shell and without leaking one
    test's PATH into the next.
    """
    return {"HOME": str(home), "PATH": path}


@pytest.fixture
def world(tmp_path):
    """Build a whole three-quantity world and probe it.

    Defaults to the healthy case so each test states only its own skew; a test
    that had to spell all three every time would drift from the thing it is
    about.
    """
    base = tmp_path

    # Every build gets its OWN subtree. Found the hard way: two builds in one
    # test sharing a writer dir meant the second still saw the first's
    # site-packages, so the "writer version unreadable" case quietly measured
    # the previous case's version instead. A fixture that leaks between calls
    # is a fixture that can make a real regression look like a passing test.
    counter = itertools.count()

    def build(*, cache=None, reader=None, writer=None, meta=None):
        tmp_path = base / f"w{next(counter)}"
        cache_db = tmp_path / "share" / "aggregator" / "cache.db"
        if cache is not None:
            _stamp_cache(cache_db, cache, meta=meta)
        reader_dir = _fake_tree(tmp_path / "reader", reader if reader is not None else 6)
        writer_bin = _fake_writer(tmp_path / "writer", writer)
        return sp.probe(
            cache_db=cache_db, reader_dir=reader_dir, writer_bin=writer_bin
        )

    return build


# --- the predicate ----------------------------------------------------------


def test_matching_versions_are_fine_and_silent(world):
    """The one state that is allowed to say nothing.

    Silence is the check's whole budget: a detector that speaks every run is
    one nobody reads by the time it matters. So FINE must be reachable and
    must be genuinely quiet — no findings, exit 0.
    """
    v = world(cache=6, reader=6, writer=6)
    assert v.state == sp.FINE, v.explain()
    assert v.severity == sp.SILENT, v.explain()
    assert v.findings == [], v.explain()
    assert v.exit_code() == 0


def test_cache_one_behind_the_reader_is_dead(world):
    """The live incident's first half, minimised to a single version step.

    One behind is the whole bug — the MCP gate is ``version < SCHEMA_VERSION``,
    so 5-against-6 refuses exactly as hard as 0-against-6.
    """
    v = world(cache=5, reader=6, writer=6)
    assert sp.DEAD in v.states, v.explain()
    assert v.state == sp.DEAD, v.explain()
    assert v.severity == sp.DOWN, v.explain()
    assert v.exit_code() == sp.EXIT_DEAD
    assert "5" in v.explain() and "6" in v.explain()


def test_writer_behind_the_reader_will_rot(world):
    """The second half, and the one no component can self-diagnose.

    The cache is fine *today* — recall works this minute. But the writer is
    the thing that stamps it, so the next ingest tick pulls it back down. A
    check that only compared the cache to the reader would call this healthy
    right up until the tick, and then call it DEAD with no explanation of why
    a cache that was fine an hour ago is not.
    """
    v = world(cache=6, reader=6, writer=5)
    assert sp.WILL_ROT in v.states, v.explain()
    assert v.state == sp.WILL_ROT, v.explain()
    assert v.severity == sp.REDUCED, v.explain()
    assert v.exit_code() == sp.EXIT_WILL_ROT


def test_the_live_incident_reports_both_states_at_once(world):
    """cache 5, reader 6, writer 5 — what this host actually looked like.

    Both diagnoses have to survive into the report. Collapsing them loses the
    remedy: fixing only the cache leaves a writer that re-breaks it in thirty
    minutes, and the operator who ran a one-off migration and watched it
    revert is exactly the person this check exists to spare.
    """
    v = world(cache=5, reader=6, writer=5)
    assert sp.DEAD in v.states, v.explain()
    assert sp.WILL_ROT in v.states, v.explain()
    assert v.state == sp.DEAD, "DEAD outranks WILL-ROT: recall is refusing NOW"
    assert v.exit_code() == sp.EXIT_DEAD


def test_a_cache_ahead_of_the_reader_is_dead_too(world):
    """The mirror of the incident, and the half the probe used to call healthy.

    THIS TEST USED TO ASSERT THE OPPOSITE, on the strength of ``mcp.py``'s gate
    being ``version < SCHEMA_VERSION``. That gate is now ``!=`` — see
    ``_ensure_cache_ready`` and ``_ahead_cache_response``, which exist because
    serving a cache the reader cannot describe is the WORSE half of the two
    failures: it answers, with rows out of tables this build has no description
    of, so nothing about it prompts anyone to look. A probe that reported
    "healthy, exit 0" while every ``aggregator_search_memory`` call came back
    ``ok:false`` would be the original incident wearing the other shoe.

    The probe must mirror the gate it reports on. Stricter than the reader
    trains an operator to ignore it; LOOSER than the reader is the failure this
    module exists to prevent.
    """
    v = world(cache=7, reader=6, writer=7)
    assert v.state != sp.FINE, v.explain()
    assert sp.DEAD in v.states, v.explain()
    assert v.exit_code() == sp.EXIT_DEAD
    assert "7" in v.explain() and "6" in v.explain()


def test_the_cache_ahead_remedy_moves_the_reader_never_the_cache(world):
    """Opposite cause, opposite fix — and one fix here is actively destructive.

    In the stale direction the writer lags and a newer writer is deployed. Here
    the CACHE is the current side: whatever wrote it is already ahead, and the
    reader is the lagging one. Running an older writer against it re-stamps
    ``user_version`` DOWNWARD, which turns a cache one component cannot read
    into a cache that is wrong for all of them. So the remedy must send the
    operator at the reader binary Claude Code launches, and must not name the
    cache or a downgrade as an option at all.
    """
    v = world(cache=7, reader=6, writer=7)
    fix = " ".join(f.remedy for f in v.findings if f.state == sp.DEAD)
    assert "restart" in fix.lower(), fix
    assert "reader" in fix.lower(), fix
    assert "downgrad" not in fix.lower(), fix


def _remedy_targets(verdict) -> set[int]:
    """Every version number the SKEW remedies tell an operator to reach.

    Remedies only — never ``detail``. The details legitimately quote numbers
    that are below the cache, because that is what they measured; it is the
    instructions that must not send anyone backwards. UNKNOWN findings are
    excluded too: their remedies carry ``lib/python3*`` and byte counts, which
    are not versions.
    """
    return {
        int(n)
        for f in verdict.findings
        if f.state in (sp.DEAD, sp.WILL_ROT)
        for n in re.findall(r"\d+", f.remedy)
    }


def test_a_three_way_skew_does_not_advise_undoing_its_own_other_remedy(world):
    """cache 7, reader 6, writer 5 — and the two remedies used to disagree.

    Each finding was computed against the READER alone, so with three distinct
    versions the advice forked. The DEAD finding said the cache is the current
    side, leave it alone, bring the reader to 7. The WILL_ROT finding said bump
    the writer to "at least 6" — and a schema-6 writer re-stamps that
    schema-7 cache DOWN on the next tick, destroying the thing the other half
    of the same message had just called authoritative.

    An operator who follows both instructions must not end up worse off than
    one who follows either. So every remedy names ONE target: the highest
    version anything here is at, because that is the only number nothing has to
    move backwards to reach.
    """
    v = world(cache=7, reader=6, writer=5)
    assert sp.DEAD in v.states and sp.WILL_ROT in v.states, v.explain()

    targets = _remedy_targets(v)
    assert targets == {7}, v.explain()
    assert min(targets) >= v.cache_version, "a remedy advised down-stamping the cache"


def test_the_mirror_three_way_skew_agrees_with_itself_too(world):
    """cache 5, reader 6, writer 7. Same defect, pointing the other way.

    Here the DEAD finding used to say "bring the WRITER up to at least 6" about
    a writer already at 7 — which is either inert or, read literally, an
    instruction to install a SIXES writer in place of the seven and start
    down-stamping. Meanwhile the WILL_ROT finding correctly asked for a reader
    at 7. One number, derived from all three quantities, is what makes those
    the same instruction.
    """
    v = world(cache=5, reader=6, writer=7)
    assert sp.DEAD in v.states and sp.WILL_ROT in v.states, v.explain()

    targets = _remedy_targets(v)
    assert targets == {7}, v.explain()
    assert min(targets) >= v.cache_version, "a remedy advised down-stamping the cache"


def test_two_way_skews_keep_naming_the_reader_s_requirement(world):
    """The target only moves when a third version exists to move it.

    Every two-quantity world — the live incident included — has its highest
    version at the reader or the cache, so the number in these messages is the
    one that was always there. Stated so the generalisation is visibly a
    generalisation and not a rewrite of the texts the incident produced.
    """
    assert _remedy_targets(world(cache=5, reader=6, writer=6)) == {6}
    assert _remedy_targets(world(cache=6, reader=6, writer=5)) == {6}
    assert _remedy_targets(world(cache=5, reader=6, writer=5)) == {6}
    assert _remedy_targets(world(cache=7, reader=6, writer=7)) == {7}


def test_writer_ahead_of_the_reader_will_rot(world):
    """The writer stamps what the reader must accept, so ahead rots too.

    ALSO INVERTED FROM WHAT IT ONCE ASSERTED, and for the same reason: while
    the gate was ``<``, a writer past the reader was the sanctioned forward
    fix and flagging it would have argued against the probe's own remedy. Under
    ``!=`` it is a countdown. ``migrate()`` ends by stamping the writer's own
    constant, so the next ingest tick puts the cache at 7 against a reader that
    requires exactly 6 — recall works this minute and is refused within one
    timer period, which is the WILL_ROT shape exactly.
    """
    v = world(cache=6, reader=6, writer=7)
    assert v.state != sp.FINE, v.explain()
    assert sp.WILL_ROT in v.states, v.explain()
    assert v.exit_code() == sp.EXIT_WILL_ROT


# --- unknown ⇒ warn, never "fine" -------------------------------------------
#
# Every branch below is a way the probe can fail to measure. The rule, taken
# from the guard-health check that models this one: silence must mean
# "verified", never "could not tell". A probe allowed to shrug at an
# unreadable input is a probe that reports healthy on a machine where the
# cache has been deleted.


def test_absent_cache_warns_rather_than_reporting_healthy(world):
    """No cache.db at all. The tempting reading is "nothing is broken yet".

    It is wrong twice: the MCP opens the cache ``mode=ro`` and cannot create
    one, so recall is refusing right now; and a cache that vanished is a
    bigger incident than a stale one, not a smaller one.
    """
    v = world(cache=None, reader=6, writer=6)
    assert v.state != sp.FINE, "an absent cache must never read as healthy"
    assert sp.UNKNOWN in v.states, v.explain()
    assert v.severity == sp.DOWN, v.explain()
    assert v.exit_code() == sp.EXIT_UNKNOWN
    assert v.cache_version is None


def test_unreadable_cache_warns(tmp_path):
    """A file that exists at the cache path and is not a database.

    Truncation, a half-written restore, a filesystem fault. SQLite raises
    ``DatabaseError`` rather than returning a version, and the probe has to
    treat "the pragma raised" the same as "there is no file" — both mean it
    did not measure.
    """
    cache_db = tmp_path / "cache.db"
    cache_db.write_bytes(b"this is not a sqlite database, not even close")
    v = sp.probe(
        cache_db=cache_db,
        reader_dir=_fake_tree(tmp_path / "reader", 6),
        writer_bin=_fake_writer(tmp_path / "writer", 6),
    )
    assert v.state != sp.FINE, "a corrupt cache must never read as healthy"
    assert sp.UNKNOWN in v.states, v.explain()
    assert v.cache_version is None


def test_unreadable_reader_warns(tmp_path):
    """No reader checkout — the requirement cannot be known.

    Without it there is no number to compare against, so *every* other
    reading is uninterpretable. This is the branch most likely to be written
    as an early ``return FINE`` by someone tidying up, which is why it has
    its own test.
    """
    _stamp_cache(tmp_path / "cache.db", 6)
    v = sp.probe(
        cache_db=tmp_path / "cache.db",
        reader_dir=tmp_path / "no-such-checkout",
        writer_bin=_fake_writer(tmp_path / "writer", 6),
    )
    assert v.state != sp.FINE, "an unknown requirement must never read as healthy"
    assert sp.UNKNOWN in v.states, v.explain()
    assert v.reader_version is None


def test_unresolvable_writer_warns(tmp_path):
    """``aggregator`` is not on PATH, or resolves to something unreadable.

    The writer is the quantity this host got wrong, so failing to read it is
    the failure that matters most. It must not degrade into "cache matches
    reader, therefore fine" — that is precisely the two-quantity blindness
    the whole check exists to remove.
    """
    _stamp_cache(tmp_path / "cache.db", 6)
    v = sp.probe(
        cache_db=tmp_path / "cache.db",
        reader_dir=_fake_tree(tmp_path / "reader", 6),
        writer_bin=tmp_path / "no-such-binary",
    )
    assert v.state != sp.FINE, "an unknown writer must never read as healthy"
    assert sp.UNKNOWN in v.states, v.explain()
    assert v.writer_version is None


def test_writer_wrapper_without_packaged_source_warns(tmp_path):
    """The wrapper resolves but carries no ``store.py`` to read.

    A broken or half-built install. Distinguished from "not on PATH" only in
    the message; both are UNKNOWN, because the point is that the number was
    not obtained and nothing may be concluded from its absence.
    """
    _stamp_cache(tmp_path / "cache.db", 6)
    v = sp.probe(
        cache_db=tmp_path / "cache.db",
        reader_dir=_fake_tree(tmp_path / "reader", 6),
        writer_bin=_fake_writer(tmp_path / "writer", None),
    )
    assert v.state != sp.FINE, v.explain()
    assert v.writer_version is None


def test_unknown_outranks_will_rot_but_not_dead(world):
    """Severity ordering, stated once so the notifier's headline is settled.

    A probe that cannot measure is DOWN, not REDUCED — but a probe that
    measured a refusing reader has something more actionable to say, so DEAD
    still takes the headline.
    """
    both = world(cache=None, reader=6, writer=5)
    assert sp.UNKNOWN in both.states and sp.WILL_ROT in both.states, both.explain()
    assert both.state == sp.UNKNOWN, both.explain()

    dead_too = world(cache=5, reader=6, writer=None)
    assert sp.DEAD in dead_too.states and sp.UNKNOWN in dead_too.states
    assert dead_too.state == sp.DEAD, dead_too.explain()


# --- corroboration ----------------------------------------------------------


def test_meta_row_disagreeing_with_the_pragma_is_reported(world):
    """``migrate()`` writes two stamps; they are supposed to agree.

    The gate reads only ``PRAGMA user_version``, so that is what decides the
    verdict — but a cache where the two have come apart has been written by
    something that is not ``migrate()``, and saying so is free. Two readings
    corroborating each other is the same discipline the guard-health check
    uses, and for the same reason: either one alone is a single point of rot.
    """
    v = world(cache=6, reader=6, writer=6, meta=5)
    assert v.state != sp.FINE, "disagreeing stamps must not read as healthy"
    assert "meta" in v.explain().lower(), v.explain()


# --- the trap ---------------------------------------------------------------


def test_probe_never_writes_the_cache(tmp_path):
    """THE regression lock. Probing must not perform the damage it reports.

    ``cli.py`` runs ``store.migrate()`` on every subcommand except ``embed``,
    and ``migrate()`` ends by stamping ``PRAGMA user_version = SCHEMA_VERSION``.
    So the obvious probe — shell out to ``aggregator status`` and read what it
    prints — re-stamps the cache at the prober's own version. From the
    schema-5 build that is exactly the write that has been erasing the
    evidence every thirty minutes.

    Checked three ways because each alone can be fooled: the version itself
    (a same-version rewrite would pass a version check), the mtime (a write
    that happened to restore the value would still touch it), and the absence
    of the WAL/journal sidecars SQLite leaves behind when a connection is
    opened writable at all.
    """
    cache_db = _stamp_cache(tmp_path / "cache.db", 5)
    before_mtime = cache_db.stat().st_mtime_ns
    before_bytes = cache_db.read_bytes()

    v = sp.probe(
        cache_db=cache_db,
        reader_dir=_fake_tree(tmp_path / "reader", 6),
        writer_bin=_fake_writer(tmp_path / "writer", 5),
    )
    assert v.state == sp.DEAD, v.explain()

    con = sqlite3.connect(f"file:{cache_db}?mode=ro", uri=True)
    try:
        assert con.execute("PRAGMA user_version").fetchone()[0] == 5, (
            "the probe re-stamped the cache it was asked to report on — this "
            "is the `aggregator status` trap, reintroduced"
        )
    finally:
        con.close()
    assert cache_db.stat().st_mtime_ns == before_mtime, "the probe touched cache.db"
    assert cache_db.read_bytes() == before_bytes, "the probe rewrote cache.db bytes"
    for sidecar in ("cache.db-wal", "cache.db-journal", "cache.db-shm"):
        assert not (tmp_path / sidecar).exists(), (
            f"the probe left {sidecar} behind, so it opened the cache writable"
        )


def test_probe_does_not_execute_the_writer(tmp_path):
    """Never invoke the broken thing to test the broken thing.

    The writer is the component under suspicion. Running it to ask its
    version would migrate the cache, would hang if the install is wedged, and
    would report nothing at all if the binary is the part that is broken. The
    version is read out of the packaged source instead — so a writer binary
    that fails outright on execution must still be measurable.
    """
    _stamp_cache(tmp_path / "cache.db", 6)
    writer = _fake_writer(tmp_path / "writer", 5)
    # Make the wrapper fatal to run. A probe that executes it gets nothing;
    # a probe that reads it gets 5.
    writer.write_text(
        writer.read_text(encoding="utf-8").replace("#!/bin/sh\n", "#!/bin/sh\nexit 127\n"),
        encoding="utf-8",
    )
    v = sp.probe(
        cache_db=tmp_path / "cache.db",
        reader_dir=_fake_tree(tmp_path / "reader", 6),
        writer_bin=writer,
    )
    assert v.writer_version == 5, (
        "the writer's version must be READ from its packaged source, never "
        f"obtained by running it: {v.explain()}"
    )
    assert v.state == sp.WILL_ROT, v.explain()


# --- one walker, two callers ------------------------------------------------


def test_resolve_package_dir_walks_a_nix_wrapper_chain_to_site_packages(tmp_path):
    """The two-hop chain every install on this host is reached through.

    ``bin/aggregator`` is a shell wrapper; only the env derivation it execs
    carries ``lib/python3.11/site-packages``. What comes back is the DIRECTORY,
    not the version, because the verdict reports that path to a human who then
    has to go and look at the file the number came from.
    """
    writer = _fake_writer(tmp_path / "writer", 6)
    found = sp.resolve_package_dir(writer)
    assert found == tmp_path / "writer" / "env" / "lib" / "python3.11" / "site-packages"
    assert sp.read_reader_version(found) == 6


def test_resolve_package_dir_is_none_when_no_packaged_source_is_reachable(tmp_path):
    """A chain that ends nowhere useful, a path that is not a file, and nothing.

    All three are "could not tell". Returning a directory that carries no
    ``store.py`` would name a place the number did not come from, which is the
    same lie one layer further from the operator.
    """
    assert sp.resolve_package_dir(_fake_writer(tmp_path / "empty", None)) is None
    assert sp.resolve_package_dir(tmp_path / "no-such-binary") is None
    assert sp.resolve_package_dir(None) is None


def test_the_writer_reads_through_the_one_shared_walker(tmp_path):
    """The writer keeps its answer, and gets it from the shared helper.

    The reader is reached through a chain of the identical shape, so a second
    copy of this walk would be a second thing to drift — and the half that
    drifted would be the half nobody was looking at, which is precisely how the
    incident this file detects lasted three days.
    """
    writer = _fake_writer(tmp_path / "writer", 5)
    assert sp.read_writer_version(writer) == 5

    # The directory the writer's answer came out of, spelled independently
    # rather than re-derived from the helper under test — asserting
    # ``read_writer_version(w) == read_reader_version(resolve_package_dir(w))``
    # would restate that function's one-line definition and hold even if the
    # walk found nothing at all.
    site_packages = tmp_path / "writer" / "env" / "lib" / "python3.11" / "site-packages"
    assert sp.resolve_package_dir(writer) == site_packages
    assert (site_packages / "aggregator" / "core" / "store.py").read_text(
        encoding="utf-8"
    ) == "SCHEMA_VERSION = 5\n"


def test_the_writer_is_the_one_beside_the_reader_not_the_one_on_path(tmp_path):
    """PATH is the probe's PATH, and in a checkout that is the wrong writer.

    THE WRITER UNDER TEST IS THE DEPLOYED ONE. On this host the ingest timer
    execs ``pkgs.aggregator`` — the same package the profile's
    ``aggregator-mcp`` comes from, so the two sit in one ``bin/``. A bare
    ``_which("aggregator")`` instead answers with whatever is first on the
    PATH the probe happens to inherit, and inside a checkout, a ``uv run`` or
    a devShell that is the checkout's own venv CLI. Nobody's ingest timer runs
    that binary.

    The failure is quiet in both directions: the probe reports the checkout's
    version as "the writer", so a genuinely lagging deployed writer never
    fires WILL_ROT, and a checkout that happens to be ahead invents a skew
    nobody has. Both are the two-quantity blindness this module exists to
    remove, re-entering through the PATH.
    """
    home = tmp_path / "home"
    reader = _fake_writer(
        tmp_path / "profile", 6, name="aggregator-mcp", outer_dir="bin"
    )
    _fake_writer(tmp_path / "profile", 6, name="aggregator", outer_dir="bin")
    checkout = _fake_writer(
        tmp_path / "checkout", 7, name="aggregator", outer_dir="bin"
    )
    _claude_json(home, {"command": str(reader), "args": []})

    env = _env(home, path=str(checkout.parent))
    assert sp.resolve_writer_bin(env) == tmp_path / "profile" / "bin" / "aggregator"
    assert sp.read_writer_version(sp.resolve_writer_bin(env)) == 6

    # And end to end: the checkout at 7 must not become a skew report.
    v = sp.probe(cache_db=_stamp_cache(tmp_path / "cache.db", 6), env=env)
    assert v.state == sp.FINE, v.explain()
    assert v.writer_version == 6, v.explain()
    assert v.writer_bin == str(tmp_path / "profile" / "bin" / "aggregator")


def test_the_sibling_lookup_yields_to_the_override_and_falls_back_to_path(tmp_path):
    """The new step is a middle one: it must not swallow the two either side.

    ``AGGREGATOR_WRITER_BIN`` stays first — every input here is overridable,
    and a probe that could only look at the live machine could not be tested.
    And a profile that ships ``aggregator-mcp`` without the CLI beside it has
    no sibling to find, so the PATH search still has to answer; a middle step
    that returned "nothing" rather than deferring would turn a working
    single-package install into UNKNOWN.
    """
    home = tmp_path / "home"
    reader = _fake_writer(
        tmp_path / "profile", 6, name="aggregator-mcp", outer_dir="bin"
    )
    _fake_writer(tmp_path / "profile", 6, name="aggregator", outer_dir="bin")
    on_path = _fake_writer(tmp_path / "elsewhere", 7, name="aggregator", outer_dir="bin")
    _claude_json(home, {"command": str(reader), "args": []})

    forced = _fake_writer(tmp_path / "forced", 9, name="aggregator", outer_dir="bin")
    env = _env(home, path=str(on_path.parent))
    env[sp.WRITER_BIN_ENV] = str(forced)
    assert sp.resolve_writer_bin(env) == forced

    # No CLI beside the reader: PATH answers, exactly as before.
    lonely = _fake_writer(
        tmp_path / "mcp-only", 6, name="aggregator-mcp", outer_dir="bin"
    )
    _claude_json(home, {"command": str(lonely), "args": []})
    assert sp.resolve_writer_bin(_env(home, path=str(on_path.parent))) == on_path


def test_a_directory_reader_takes_no_sibling(tmp_path):
    """``uv run --directory <checkout>``: the command is ``uv``, not a reader.

    Its neighbours are uv's own install, which has no aggregator in it and no
    business being measured as one. The sibling step is only meaningful when
    the ``command`` IS the reader — anywhere else it would name a binary
    chosen by coincidence of directory layout.
    """
    home = tmp_path / "home"
    checkout = _fake_tree(tmp_path / "checkout", 6)
    uv_bin = tmp_path / "uv" / "bin"
    uv_bin.mkdir(parents=True)
    for name in ("uv", "aggregator"):
        exe = uv_bin / name
        exe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        exe.chmod(0o755)
    on_path = _fake_writer(tmp_path / "deployed", 6, name="aggregator", outer_dir="bin")
    _claude_json(
        home,
        {"command": str(uv_bin / "uv"), "args": ["run", "--directory", str(checkout)]},
    )

    env = _env(home, path=str(on_path.parent))
    assert sp.resolve_reader_dir(env) == checkout
    assert sp.resolve_writer_bin(env) == on_path


def test_resolve_writer_bin_searches_the_path_it_is_handed(tmp_path):
    """``shutil.which`` semantics, over the environment passed in.

    This is a characterization test, and it is expected to pass BEFORE the code
    it describes is touched: the next step moves this search into a helper the
    reader shares, and a refactor with no test underneath it is how the writer
    half of this file would quietly stop finding anything.

    The environment is a plain dict, never the process's. The systemd unit and
    a Claude Code session have different ``PATH``s, and a resolver that
    consulted the wrong one would measure a binary nobody runs.

    EVERY env HERE CARRIES AN EMPTY ``HOME``, and that is not tidiness. The
    lookup now consults ``~/.claude.json`` for a reader to stand beside, and
    ``_claude_config_path`` falls back to ``Path.home()`` when the dict it was
    handed names no HOME — so the HOME-less version of this test read the
    developer's real config and answered
    ``/etc/profiles/per-user/jonathan/bin/aggregator``. It found the live host
    from inside a unit test, which is the machine's answer and not the
    fixture's.
    """
    home = tmp_path / "home"
    home.mkdir()
    binroot = tmp_path / "bin"
    binroot.mkdir()
    exe = binroot / "aggregator"
    exe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    exe.chmod(0o755)

    assert sp.resolve_writer_bin(_env(home, path=str(binroot))) == exe
    assert sp.resolve_writer_bin(_env(home, path=str(tmp_path / "nowhere"))) is None
    assert sp.resolve_writer_bin(_env(home)) is None
    assert sp.resolve_writer_bin({"HOME": str(home)}) is None

    # A file a shell would not run is not the writer.
    exe.chmod(0o644)
    assert sp.resolve_writer_bin(_env(home, path=str(binroot))) is None


# --- which reader is under test ---------------------------------------------


def test_reader_dir_comes_from_the_command_when_there_is_no_directory_arg(tmp_path):
    """THE BUG. ``{"command": "<nix wrapper>", "args": []}`` — this host, today.

    ``args`` is empty, so the old code found no ``--directory`` and fell back to
    the checkout the probe file happened to sit in. On 2026-09-05 that checkout
    was at schema 7 while the deployed reader, the writer and the cache were all
    at 6, so every new session was told AGGREGATOR RECALL IS DEAD about a
    machine where recall was fine. The requirement has to come from the binary
    Claude Code actually execs, and the answer is the site-packages the chain
    ends in — the directory the number was read from.
    """
    home = tmp_path / "home"
    wrapper = _fake_writer(
        tmp_path / "install", 6, name="aggregator-mcp", outer_dir="bin"
    )
    _claude_json(
        home, {"type": "stdio", "command": str(wrapper), "args": [], "env": {}}
    )

    found = sp.resolve_reader_dir(_env(home))
    assert found == tmp_path / "install" / "env" / "lib" / "python3.11" / "site-packages"
    assert sp.read_reader_version(found) == 6
    assert found != Path(sp.__file__).resolve().parent.parent.parent


def test_a_bare_command_name_is_looked_up_on_the_path(tmp_path):
    """``{"command": "aggregator-mcp"}`` with no slash in it.

    Claude Code runs that through a PATH search, so the probe must too — over
    the PATH it was handed, because the systemd health unit's PATH is not the
    session's and resolving against the wrong one would measure the wrong
    binary while looking like it worked.
    """
    home = tmp_path / "home"
    wrapper = _fake_writer(
        tmp_path / "install", 6, name="aggregator-mcp", outer_dir="bin"
    )
    _claude_json(home, {"command": "aggregator-mcp", "args": []})

    found = sp.resolve_reader_dir(_env(home, path=str(wrapper.parent)))
    assert found == tmp_path / "install" / "env" / "lib" / "python3.11" / "site-packages"


def test_a_command_that_resolves_nowhere_is_unknown_not_this_checkout(tmp_path):
    """An entry exists and cannot be resolved. The answer is "I could not tell".

    This is the case the systemd health unit hits: its PATH carries no
    aggregator at all, so a bare name resolves to nothing. UNKNOWN is loud and
    correct there. Falling back to this checkout would compare the writer
    against a tree the reader has never run — the false alarm being fixed,
    re-entering through the door it left by.
    """
    home = tmp_path / "home"
    _claude_json(home, {"command": "aggregator-mcp", "args": []})
    assert sp.resolve_reader_dir(_env(home, path=str(tmp_path / "empty-bin"))) is None


def test_a_directory_argument_still_wins_over_the_command(tmp_path):
    """``uv run --directory <checkout> aggregator-mcp`` — the dev shape, kept.

    A session pointed at a checkout IS running that checkout, and resolving
    ``uv`` through the wrapper walk would find uv's own install and report
    nothing useful. So the explicit directory stays first: it names a tree a
    human chose, while the command is what to consult when nobody chose.
    """
    home = tmp_path / "home"
    checkout = _fake_tree(tmp_path / "checkout", 7)
    wrapper = _fake_writer(
        tmp_path / "install", 6, name="aggregator-mcp", outer_dir="bin"
    )

    _claude_json(
        home,
        {
            "command": "uv",
            "args": ["run", "--directory", str(checkout), "aggregator-mcp"],
        },
    )
    assert sp.resolve_reader_dir(_env(home)) == checkout

    # And when BOTH could resolve, the argument is still the answer.
    _claude_json(home, {"command": str(wrapper), "args": ["--directory", str(checkout)]})
    assert sp.read_reader_version(sp.resolve_reader_dir(_env(home))) == 7


def test_the_joined_directory_spelling_is_read_too(tmp_path):
    """``--directory=<dir>``, which ``uv run`` accepts exactly as readily.

    Only the split form was parsed, so the joined one fell through to the
    ``command`` branch — where the command is ``uv``, the wrapper walk finds
    uv's own install, no ``aggregator/core/store.py`` is there, and the answer
    is UNKNOWN. A session pointed at a perfectly good checkout got told its
    recall health could not be verified, on the strength of a space.

    This config is hand-edited and the two spellings are interchangeable to the
    tool that consumes it, so which one an operator typed cannot be allowed to
    decide whether the check works.
    """
    home = tmp_path / "home"
    checkout = _fake_tree(tmp_path / "checkout", 7)
    _claude_json(
        home,
        {"command": "uv", "args": ["run", f"--directory={checkout}", "aggregator-mcp"]},
    )
    assert sp.resolve_reader_dir(_env(home)) == checkout
    assert sp.read_reader_version(sp.resolve_reader_dir(_env(home))) == 7

    # The split form keeps working, and an empty value is not a directory.
    _claude_json(home, {"command": "uv", "args": ["run", "--directory", str(checkout)]})
    assert sp.resolve_reader_dir(_env(home)) == checkout
    _claude_json(home, {"command": "uv", "args": ["run", "--directory="]})
    assert sp.resolve_reader_dir(_env(home)) is None


def test_no_entry_at_all_falls_back_to_this_checkout(tmp_path):
    """No ``mcpServers.aggregator`` anywhere: a dev tree with no MCP wiring.

    The fallback that caused the false alarm survives for exactly the case it
    was written for, and is narrowed to it. Still gated on a ``pyproject.toml``
    beside the package: installed into site-packages this module also sits next
    to a ``core/store.py``, and reading THAT as the reader's requirement would
    compare the writer against itself and call every skew healthy.
    """
    home = tmp_path / "home"
    _claude_json(home, None)
    repo_root = Path(sp.__file__).resolve().parent.parent.parent
    assert (repo_root / "pyproject.toml").is_file()
    assert sp.resolve_reader_dir(_env(home)) == repo_root


def test_a_missing_claude_json_falls_back_to_this_checkout(tmp_path):
    """No file at all: there is no entry to be read, so the fallback holds.

    Deliberate, and the narrow reading was chosen on evidence: a HOME with no
    ``.claude.json`` is every CI run and every fresh clone, and turning those
    into UNKNOWN would make the probe cry wolf where nothing is wrong — which
    spends the silence budget rule 2 exists to protect. Absence is the only
    unreadable shape that gets this benefit; see the next test.
    """
    home = tmp_path / "home"
    home.mkdir()
    repo_root = Path(sp.__file__).resolve().parent.parent.parent
    assert sp.resolve_reader_dir(_env(home)) == repo_root


def test_a_present_but_unparseable_claude_json_is_unknown(tmp_path):
    """The file IS there and cannot be read. Guessing here is the original bug.

    A config that exists says an operator configured something; that it will not
    parse says nobody knows what. Falling back to this checkout then re-creates
    the 2026-09-05 false alarm through a second door — the tree would be
    measured and announced as the reader on a host where the real entry, sitting
    unparsed in that very file, names a Nix wrapper at a different version.
    "Fail loudly": a broken config is a fault to report, never a silent licence
    to substitute a different measurement.

    Absence keeps the fallback (previous test); presence-and-broken does not.
    """
    home = tmp_path / "home"
    home.mkdir()

    (home / ".claude.json").write_text("{not json", encoding="utf-8")
    assert sp.resolve_reader_dir(_env(home)) is None

    # Valid JSON, but not an object at the top level.
    (home / ".claude.json").write_text("[1, 2, 3]", encoding="utf-8")
    assert sp.resolve_reader_dir(_env(home)) is None

    # An object whose `mcpServers` is not a block of servers, so an entry could
    # be hiding in there and this probe cannot see it.
    (home / ".claude.json").write_text('{"mcpServers": "nope"}', encoding="utf-8")
    assert sp.resolve_reader_dir(_env(home)) is None


def _oversized_claude_json(home: Path, wrapper: Path) -> Path:
    """A ``~/.claude.json`` that is entirely VALID and larger than the read cap.

    Not a synthetic blob: the shape is the real file's. Claude Code stores
    per-project history in this same document — ``projects[<dir>].history`` —
    and on a machine with a few long-lived projects it grows without bound,
    which is the only reason a JSON config ever passes eight-figure byte
    counts. The aggregator entry sits AFTER the filler on purpose: it is the
    part a truncated read loses, so a probe that parses the prefix is not
    merely unlucky, it is systematically blind to the thing it came to read.
    """
    home.mkdir(parents=True, exist_ok=True)
    doc = {
        "projects": {
            "/home/jonathan/Repos/something": {"history": ["x" * 4096] * 4200}
        },
        "mcpServers": {
            "research-agent": {"command": "true", "args": []},
            "aggregator": {"command": str(wrapper), "args": []},
        },
    }
    path = home / ".claude.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert path.stat().st_size > sp._CLAUDE_CONFIG_LIMIT, path.stat().st_size
    return path


def test_an_oversized_but_valid_claude_json_names_the_cap_not_a_repair(tmp_path):
    """A config too big to read is not a config that is broken.

    The probe reads a bounded prefix of ``~/.claude.json`` — deliberately, and
    it must keep doing so: both consumers run on budgets of a few seconds and
    this file has no business being scanned without limit. But a bounded read
    of a valid 17 MB document yields a buffer cut mid-token, ``json.loads``
    fails on it exactly the way it fails on a corrupt file, and the operator is
    handed "repair ~/.claude.json — it must be valid JSON" about a file that
    already is. That remedy is unfollowable: there is nothing to repair, so the
    check announces an impossible chore every session until someone reads this
    source to find out why.

    The two causes have to be told apart at the point where they are still
    distinguishable — before the parse, by the size — and the message has to
    name the number, the cap, and the override that gets past both.
    """
    home = tmp_path / "home"
    wrapper = _fake_writer(
        tmp_path / "install", 6, name="aggregator-mcp", outer_dir="bin"
    )
    path = _oversized_claude_json(home, wrapper)
    env = _env(home)
    env[sp.WRITER_BIN_ENV] = str(_fake_writer(tmp_path / "writer", 6))

    v = sp.probe(cache_db=_stamp_cache(tmp_path / "cache.db", 6), env=env)
    assert v.state == sp.UNKNOWN, v.explain()
    assert v.reader_version is None
    assert v.reader_dir is None, "a truncated config must not become a guess"

    said = " ".join(f.text() for f in v.findings if f.state == sp.UNKNOWN)
    assert str(path) in said, said
    assert str(path.stat().st_size) in said, said
    assert str(sp._CLAUDE_CONFIG_LIMIT) in said, said
    assert sp.READER_DIR_ENV in said, said
    # The corrupt-file remedy must NOT be the one shown: this file parses.
    assert "repair" not in said.lower(), said


def test_a_config_under_the_cap_still_parses(tmp_path):
    """The bound is a bound, not a new failure mode.

    A padded-but-legal config — well under the cap and far larger than the
    fixtures everything else uses — has to resolve the reader normally. A size
    check that shipped with an off-by-one, or that stat()ed the wrong thing,
    would turn every real ``~/.claude.json`` on this host into UNKNOWN, which
    is the false alarm this module has already paid for once.
    """
    home = tmp_path / "home"
    home.mkdir()
    wrapper = _fake_writer(
        tmp_path / "install", 6, name="aggregator-mcp", outer_dir="bin"
    )
    doc = {
        "projects": {"/home/jonathan/Repos/x": {"history": ["y" * 4096] * 64}},
        "mcpServers": {"aggregator": {"command": str(wrapper), "args": []}},
    }
    (home / ".claude.json").write_text(json.dumps(doc), encoding="utf-8")
    assert (home / ".claude.json").stat().st_size < sp._CLAUDE_CONFIG_LIMIT

    assert sp.read_reader_version(sp.resolve_reader_dir(_env(home))) == 6


def test_an_unparseable_claude_json_says_so_and_names_the_file(tmp_path):
    """The remedy has to send the operator at the FILE, not at the install.

    Two causes reach the same UNKNOWN state and they have opposite fixes: a
    config this probe could not parse is the operator's file to repair, while a
    config that resolves to an install with no packaged source is the install's
    problem. One generic message for both leaves the reader guessing which,
    and an announcement nobody can act on gets acknowledged and forgotten.
    """
    home = tmp_path / "home"
    home.mkdir()
    (home / ".claude.json").write_text("{not json", encoding="utf-8")
    env = _env(home)
    env[sp.WRITER_BIN_ENV] = str(_fake_writer(tmp_path / "writer", 6))

    v = sp.probe(cache_db=_stamp_cache(tmp_path / "cache.db", 6), env=env)
    assert v.state == sp.UNKNOWN, v.explain()
    assert v.reader_version is None
    assert v.reader_dir is None
    said = " ".join(f.text() for f in v.findings if f.state == sp.UNKNOWN)
    assert str(home / ".claude.json") in said, said
    assert "parse" in said, said


def test_an_entry_that_is_present_but_unusable_is_unknown(tmp_path):
    """``mcpServers.aggregator`` exists and carries nothing resolvable.

    Something IS configured, so this checkout is not the reader, and "could not
    tell" is the only honest answer. Rule 2 at the top of the probe: unknown
    warns, never "fine" — and never "here is a different thing I measured
    instead".
    """
    home = tmp_path / "home"

    _claude_json(home, {})
    assert sp.resolve_reader_dir(_env(home)) is None

    _claude_json(home, {"command": "", "args": []})
    assert sp.resolve_reader_dir(_env(home)) is None

    _claude_json(home, "not-an-object")
    assert sp.resolve_reader_dir(_env(home)) is None

    # And the trap that matters most on a real host: a command that RESOLVES,
    # through a wrapper chain that carries no ``aggregator/core/store.py``. A
    # half-installed reader must not silently become this checkout — the
    # binary exists, so the fallback looks defensible right up until it reports
    # the wrong version.
    hollow = _fake_writer(
        tmp_path / "hollow", None, name="aggregator-mcp", outer_dir="bin"
    )
    _claude_json(home, {"command": str(hollow), "args": []})
    assert sp.resolve_reader_dir(_env(home)) is None


def test_the_reader_dir_override_beats_everything(tmp_path):
    """``AGGREGATOR_READER_DIR``: the escape hatch, still first.

    Every input to this probe is overridable, because a health check that can
    only ever look at the live machine cannot be tested at all — which for a
    health check is the failure mode, not an inconvenience.
    """
    home = tmp_path / "home"
    wrapper = _fake_writer(
        tmp_path / "install", 6, name="aggregator-mcp", outer_dir="bin"
    )
    _claude_json(home, {"command": str(wrapper), "args": []})
    forced = _fake_tree(tmp_path / "forced", 9)

    env = _env(home)
    env[sp.READER_DIR_ENV] = str(forced)
    assert sp.resolve_reader_dir(env) == forced


def test_the_project_scoped_entry_is_never_read(tmp_path):
    """``~/.claude.json`` also carries per-project blocks. They are not ours.

    This host has a stale one from an unrelated repo whose command resolves to
    nothing. It applies only to sessions started in that directory. A probe that
    read it would report on a server nothing runs — while the top-level entry
    sat right there and resolved.
    """
    home = tmp_path / "home"
    wrapper = _fake_writer(
        tmp_path / "install", 6, name="aggregator-mcp", outer_dir="bin"
    )
    path = _claude_json(home, {"command": str(wrapper), "args": []})

    stale = json.loads(path.read_text(encoding="utf-8"))["projects"]
    assert stale["/home/jonathan/Repos/gdocs-review-mcp"]["mcpServers"]["aggregator"][
        "args"
    ] == ["not", "found"]
    assert sp.read_reader_version(sp.resolve_reader_dir(_env(home))) == 6


def test_the_unknown_reader_fix_names_every_place_it_looked(tmp_path):
    """A remedy has to be actionable at 03:00 by someone who did not write this.

    The entry now has two shapes — a ``command`` and a ``--directory``
    argument — and the old text named only the second, so an operator whose
    command had gone stale was sent looking for an argument their config does
    not contain. An announcement with no action is one that gets acknowledged
    and forgotten, which is how the original incident survived three days of a
    tool returning ``ok: false`` on every call.
    """
    home = tmp_path / "home"
    _claude_json(home, {"command": "aggregator-mcp", "args": []})
    env = _env(home, path=str(tmp_path / "empty-bin"))
    env[sp.WRITER_BIN_ENV] = str(_fake_writer(tmp_path / "writer", 6))

    v = sp.probe(cache_db=_stamp_cache(tmp_path / "cache.db", 6), env=env)
    assert v.state == sp.UNKNOWN, v.explain()
    assert v.reader_version is None
    remedy = " ".join(f.remedy for f in v.findings if f.state == sp.UNKNOWN)
    assert "command" in remedy, remedy
    assert "--directory" in remedy, remedy
    assert sp.READER_DIR_ENV in remedy, remedy


def test_probe_reports_the_directory_it_actually_read(tmp_path):
    """``reader_dir`` in the verdict is evidence, not a guess.

    A human reading the JSON has to be able to go and open the file the number
    came from. In the deployed case that is the site-packages inside the env
    derivation — not the profile entry, not the wrapper, and emphatically not a
    checkout nobody ran.
    """
    home = tmp_path / "home"
    wrapper = _fake_writer(
        tmp_path / "install", 6, name="aggregator-mcp", outer_dir="bin"
    )
    _claude_json(home, {"command": str(wrapper), "args": []})
    env = _env(home)
    env[sp.WRITER_BIN_ENV] = str(_fake_writer(tmp_path / "writer", 6))

    v = sp.probe(cache_db=_stamp_cache(tmp_path / "cache.db", 6), env=env)
    assert v.state == sp.FINE, v.explain()
    assert v.reader_version == 6
    assert v.reader_dir == str(
        tmp_path / "install" / "env" / "lib" / "python3.11" / "site-packages"
    )
    assert v.to_dict()["reader_dir"] == v.reader_dir


def test_the_false_alarm_of_2026_09_05_does_not_reproduce(tmp_path):
    """The incident this task exists for, end to end.

    Deployed reader 6, writer 6, cache 6 — a healthy machine — while the
    checkout this file lives in is at some other version entirely. The old
    resolution measured the checkout and reported DEAD. The verdict must be
    FINE, and it must not have been reached by reading this tree.
    """
    home = tmp_path / "home"
    wrapper = _fake_writer(
        tmp_path / "install", 6, name="aggregator-mcp", outer_dir="bin"
    )
    _claude_json(home, {"command": str(wrapper), "args": []})
    env = _env(home)
    env[sp.WRITER_BIN_ENV] = str(_fake_writer(tmp_path / "writer", 6))

    v = sp.probe(cache_db=_stamp_cache(tmp_path / "cache.db", 6), env=env)
    assert v.state == sp.FINE, v.explain()
    assert sp.DEAD not in v.states, v.explain()
    assert v.reader_dir != str(Path(sp.__file__).resolve().parent.parent.parent), (
        "the checkout was consulted; it is not the reader"
    )


def test_the_script_resolves_the_reader_from_claude_json(tmp_path):
    """The whole path, through the invocation the consumers actually use.

    Everything above calls into the module. The systemd unit and the
    SessionStart hook run ``python3 schema_probe.py --json`` as a bare script
    with no aggregator on ``sys.path``, and the reader resolution now depends on
    HOME and PATH — two things a systemd unit trims. So it is exercised in a
    real subprocess with a forged environment, not through an import.
    """
    home = tmp_path / "home"
    wrapper = _fake_writer(
        tmp_path / "install", 6, name="aggregator-mcp", outer_dir="bin"
    )
    _claude_json(home, {"command": "aggregator-mcp", "args": []})
    _stamp_cache(tmp_path / "cache.db", 5)

    env = dict(os.environ)
    env.update(
        {
            "HOME": str(home),
            "PATH": str(wrapper.parent),
            "AGGREGATOR_CACHE_DB": str(tmp_path / "cache.db"),
            "AGGREGATOR_WRITER_BIN": str(_fake_writer(tmp_path / "writer", 5)),
        }
    )
    env.pop("AGGREGATOR_READER_DIR", None)
    env.pop("PYTHONPATH", None)

    proc = subprocess.run(
        [sys.executable, sp.__file__, "--json"],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert proc.returncode == sp.EXIT_DEAD, proc.stderr
    doc = json.loads(proc.stdout)
    assert doc["reader_version"] == 6
    assert doc["cache_version"] == 5
    assert doc["reader_dir"] == str(
        tmp_path / "install" / "env" / "lib" / "python3.11" / "site-packages"
    )


# --- the machine-readable verdict -------------------------------------------


def _run_cli(tmp_path, *, cache, reader, writer):
    """Run the probe as the consumers run it: a bare file, plain python3.

    Deliberately NOT ``uv run`` and deliberately not an import. Both real
    callers — a systemd user unit and a SessionStart hook on a few-second
    budget — invoke it as a standalone stdlib script, so that invocation is
    what gets tested. An import-only test would pass while the shipped
    entrypoint was broken.
    """
    cache_db = tmp_path / "cache.db"
    if cache is not None:
        _stamp_cache(cache_db, cache)
    env = dict(os.environ)
    env.update(
        {
            "AGGREGATOR_CACHE_DB": str(cache_db),
            "AGGREGATOR_READER_DIR": str(_fake_tree(tmp_path / "reader", reader)),
            "AGGREGATOR_WRITER_BIN": str(_fake_writer(tmp_path / "writer", writer)),
        }
    )
    # PYTHONPATH is stripped on purpose: the script must not need the
    # aggregator package on sys.path to run.
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, sp.__file__, "--json"],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )


def test_cli_emits_json_and_the_documented_exit_code(tmp_path):
    proc = _run_cli(tmp_path, cache=5, reader=6, writer=5)
    assert proc.returncode == sp.EXIT_DEAD, proc.stderr
    doc = json.loads(proc.stdout)
    assert doc["state"] == sp.DEAD
    assert doc["severity"] == sp.DOWN
    assert sorted(doc["states"]) == sorted([sp.DEAD, sp.WILL_ROT])
    assert doc["cache_version"] == 5
    assert doc["reader_version"] == 6
    assert doc["writer_version"] == 5
    assert doc["findings"], "a non-FINE verdict with no findings says nothing"


def test_cli_is_silent_and_exits_zero_when_fine(tmp_path):
    proc = _run_cli(tmp_path, cache=6, reader=6, writer=6)
    assert proc.returncode == 0, proc.stderr
    doc = json.loads(proc.stdout)
    assert doc["state"] == sp.FINE
    assert doc["findings"] == []


def test_cli_runs_without_the_aggregator_package_importable(tmp_path):
    """The script must be stdlib-only, standalone, and free of package imports.

    Both consumers run it outside any venv. If it ever grows
    ``from aggregator.core.store import SCHEMA_VERSION`` it will import torch
    on a session-start budget and time out — and a timed-out hook has its
    output discarded, which for a health check is the same as never noticing.
    """
    proc = _run_cli(tmp_path, cache=6, reader=6, writer=6)
    assert proc.returncode == 0, proc.stderr
    source = Path(sp.__file__).read_text(encoding="utf-8")
    assert "from aggregator" not in source and "import aggregator" not in source, (
        "the probe imported its own package; it must stay stdlib-only"
    )


def test_probe_agrees_with_the_reader_it_is_installed_beside():
    """Ties the fixtures to reality: this checkout is a reader, so read it.

    Every other test forges the three quantities. This one runs the real
    resolution path against the real tree and asserts it recovers the same
    constant ``mcp.py`` enforces — so a change to how ``store.py`` spells
    ``SCHEMA_VERSION`` breaks this test rather than silently turning the
    probe's reader reading into ``None`` and every verdict into UNKNOWN.
    """
    repo_root = Path(__file__).resolve().parent.parent
    assert sp.read_reader_version(repo_root) == SCHEMA_VERSION


# --- the packaged entry point -----------------------------------------------


def test_the_console_script_entry_point_is_declared_and_resolves():
    """``aggregator-schema-probe`` must exist as a packaged console script.

    The SessionStart hook and the systemd health unit have to be able to run the
    PRODUCTION probe — installed next to ``aggregator-mcp``, built from the same
    rev as the reader it measures — rather than reaching into a developer
    checkout. Reaching into a checkout is the coupling that produced a false
    alarm in the first place, and a unit that executes a working tree is the
    deployment bug this project has already been bitten by once. This
    declaration is necessary but not sufficient: the profile only carries the
    script once nixos-config's overlay enumerates the name too.

    Resolved the way a console script resolves it: import the module named
    before the colon, look up the attribute named after it.
    """
    repo_root = Path(__file__).resolve().parent.parent
    with open(repo_root / "pyproject.toml", "rb") as fh:
        pyproject = tomllib.load(fh)

    target = pyproject["project"]["scripts"]["aggregator-schema-probe"]
    assert target == "aggregator.health.schema_probe:main"

    module_name, _, attr = target.partition(":")
    entry = getattr(importlib.import_module(module_name), attr)
    assert callable(entry)


def test_main_returns_the_exit_code_a_console_script_needs(tmp_path, capsys, monkeypatch):
    """``main(argv)`` takes an argv and RETURNS an int.

    A console script calls ``main()`` with no arguments and hands the result to
    ``sys.exit``. Returning ``None`` there would exit 0 on every verdict — a
    health check that always reports success, which is worse than none at all
    and is exactly the shape of the original incident: exit 0, every time,
    while recall was dead.
    """
    monkeypatch.setenv("AGGREGATOR_CACHE_DB", str(_stamp_cache(tmp_path / "cache.db", 5)))
    monkeypatch.setenv("AGGREGATOR_READER_DIR", str(_fake_tree(tmp_path / "reader", 6)))
    monkeypatch.setenv(
        "AGGREGATOR_WRITER_BIN", str(_fake_writer(tmp_path / "writer", 6))
    )

    code = sp.main(["--json"])
    assert code == sp.EXIT_DEAD

    doc = json.loads(capsys.readouterr().out)
    assert doc["state"] == sp.DEAD
    assert doc["cache_version"] == 5
    assert doc["reader_version"] == 6


def test_the_packaged_entry_point_imports_nothing_heavy():
    """The precondition for shipping this as a console script, made permanent.

    The entry point walks ``aggregator`` -> ``aggregator.health`` -> this
    module: three ``__init__``-shaped opportunities to pull in the dependency
    tree. Both callers run on budgets measured in single-digit seconds, and a
    SessionStart hook that overruns has its output DISCARDED — so the light
    import is not an optimisation, it is the feature. Asserted in a subprocess
    against ``sys.modules``, because in-process the check would pass on anything
    pytest had already imported for another test.
    """
    code = (
        "import sys\n"
        "import aggregator.health.schema_probe\n"
        "heavy = sorted(m for m in sys.modules if m.split('.')[0] in {\n"
        "    'torch', 'spacy', 'thinc', 'transformers', 'presidio_analyzer',\n"
        "    'presidio_anonymizer', 'sentence_transformers', 'numpy',\n"
        "    'sqlite_vec', 'fastmcp',\n"
        "})\n"
        "assert not heavy, heavy\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
