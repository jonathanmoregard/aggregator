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

import itertools
import json
import os
import sqlite3
import subprocess
import sys
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


def test_a_cache_ahead_of_the_reader_is_not_a_fault(world):
    """Mirrors the gate's ``<``, which is deliberate and not a typo.

    ``mcp.py`` refuses only a cache OLDER than it requires. A newer cache is
    readable, so the probe must not invent a fault the reader does not have —
    a check stricter than the thing it checks trains its operator to ignore
    it.
    """
    v = world(cache=7, reader=6, writer=7)
    assert v.state == sp.FINE, v.explain()
    assert v.exit_code() == 0


def test_writer_ahead_of_the_reader_is_not_a_fault(world):
    """The forward-fix direction, which must never be flagged.

    Bringing the writer up past the reader is the sanctioned repair for this
    incident. If the probe called that a fault it would argue against its own
    remedy while the operator was applying it.
    """
    v = world(cache=6, reader=6, writer=7)
    assert v.state == sp.FINE, v.explain()


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
    assert sp.read_writer_version(writer) == sp.read_reader_version(
        sp.resolve_package_dir(writer)
    )


def test_resolve_writer_bin_searches_the_path_it_is_handed(tmp_path):
    """``shutil.which`` semantics, over the environment passed in.

    This is a characterization test, and it is expected to pass BEFORE the code
    it describes is touched: the next step moves this search into a helper the
    reader shares, and a refactor with no test underneath it is how the writer
    half of this file would quietly stop finding anything.

    The environment is a plain dict, never the process's. The systemd unit and
    a Claude Code session have different ``PATH``s, and a resolver that
    consulted the wrong one would measure a binary nobody runs.
    """
    binroot = tmp_path / "bin"
    binroot.mkdir()
    exe = binroot / "aggregator"
    exe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    exe.chmod(0o755)

    assert sp.resolve_writer_bin({"PATH": str(binroot)}) == exe
    assert sp.resolve_writer_bin({"PATH": str(tmp_path / "nowhere")}) is None
    assert sp.resolve_writer_bin({"PATH": ""}) is None
    assert sp.resolve_writer_bin({}) is None

    # A file a shell would not run is not the writer.
    exe.chmod(0o644)
    assert sp.resolve_writer_bin({"PATH": str(binroot)}) is None


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


def test_an_unreadable_claude_json_is_treated_as_no_entry(tmp_path):
    """Missing or malformed: there is no entry to be read, so the fallback holds.

    Deliberate, and the stricter reading was rejected on evidence: a HOME with
    no ``.claude.json`` is every CI run and every fresh clone, and turning those
    into UNKNOWN would make the probe cry wolf where nothing is wrong — which
    spends the silence budget rule 2 exists to protect.
    """
    home = tmp_path / "home"
    home.mkdir()
    repo_root = Path(sp.__file__).resolve().parent.parent.parent
    assert sp.resolve_reader_dir(_env(home)) == repo_root

    (home / ".claude.json").write_text("{not json", encoding="utf-8")
    assert sp.resolve_reader_dir(_env(home)) == repo_root


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
