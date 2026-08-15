"""Round-2 MEDIUM-2: the ``--rebuild`` refusal matrix was incoherent.

``--rebuild`` adds exactly one thing over a plain ingest: it DELETEs the rows
the re-scan did not reproduce. That is only ever safe when the scan can
reproduce everything the store holds — i.e. when something on this machine
keeps the input current.

``substack`` reads a zip a human downloads from Settings → Exports, exactly
like ``chatgpt`` and ``claude-web``, and nothing on this machine refreshes it.
Its ``--rebuild`` was allowed anyway, because the refusal was decided by a
hand-kept name list plus an accident of shape (entity-shaped sources were
refused; substack is record-shaped). An old or partial archive plus the
guards' slack therefore deleted last-copy rows at exit 0.

The rule now follows the PROPERTY: a source that declares
``manual_export_input()`` (``sources.base.ReadsManualExport``) cannot be
rebuilt, whatever its shape and whatever its name.
"""
from __future__ import annotations

from datetime import UTC, datetime

from aggregator.cli import _default_sources, _rebuild_refusal, main
from aggregator.core.store import Store
from aggregator.sources.base import QueryAST, Record
from aggregator.sources.substack import SubstackSource


class _PartialSubstack(SubstackSource):
    """The REAL source (so it carries whatever the class declares), reading an
    archive that has gone partial — the exact case the guards do not catch."""

    def __init__(self, tmp_path, records: list[Record]) -> None:
        super().__init__(drops_dir=str(tmp_path / "no-drops"))
        self._records = records

    def iter_records(self, since, errors=None):
        yield from self._records


def _store(tmp_path) -> Store:
    store = Store(db_path=tmp_path / "cache.db")
    store.migrate()
    return store


def _post(i: int) -> Record:
    return Record(
        stable_id=f"substack:{i}",
        source="substack",
        subject=f"post {i}",
        body="body",
        created_at=datetime(2026, 6, 1, tzinfo=UTC),
        updated_at=datetime(2026, 6, 1, tzinfo=UTC),
    )


def _ids(store: Store) -> set[str]:
    return {r.stable_id for r in store.query(QueryAST(source="substack"))}


def test_substack_rebuild_is_refused(tmp_path, capsys):
    """THE finding. 140 posts re-scanned against 150 stored is a 6.7% shrink —
    nowhere near the 20% guard — and the 10 rows it drops have no other copy:
    the zip that held them is not on this machine any more."""
    store = _store(tmp_path)
    store.upsert([_post(i) for i in range(150)])
    src = _PartialSubstack(tmp_path, [_post(i) for i in range(140)])

    rc = main(
        ["ingest", "substack", "--rebuild"],
        _store=store,
        _sources={"substack": src},
    )

    assert rc == 2
    assert len(_ids(store)) == 150, "no last-copy row may be deleted"
    err = capsys.readouterr().err
    assert "--rebuild" in err
    assert "substack" in err


def test_plain_substack_ingest_still_refreshes_rows(tmp_path):
    """Refusing --rebuild costs nothing an operator wanted: the everyday
    upsert already overwrites every row a re-scan can produce."""
    store = _store(tmp_path)
    store.upsert([_post(0)])
    fixed = Record(
        stable_id="substack:0",
        source="substack",
        subject="reparsed",
        body="reparsed body",
    )

    rc = main(
        ["ingest", "substack"],
        _store=store,
        _sources={"substack": _PartialSubstack(tmp_path, [fixed])},
    )

    assert rc == 0
    subjects = {
        r.stable_id: r.subject for r in store.query(QueryAST(source="substack"))
    }
    assert subjects["substack:0"] == "reparsed"


# -- round 3 D1: the DEFAULT was the dangerous one -------------------------
#
# Every check above decides refusal from evidence AGAINST a rebuild. A source
# that declares nothing at all — the normal state of one whose author never
# read ``_rebuild_refusal`` — fell through them and got the DELETE. The rule is
# now inverted: allowed only for a source that declares
# ``sources.base.SupportsRebuild``.


class _SourceTen:
    """A new record-shaped source. Its input happens to be a vendor export a
    human downloads, but nobody told the CLI that."""

    name = "source-ten"

    def __init__(self, records: list[Record]) -> None:
        self._records = records

    def iter_records(self, since, errors=None):
        yield from self._records


def _ten(i: int) -> Record:
    return Record(
        stable_id=f"source-ten:{i}",
        source="source-ten",
        subject=f"p{i}",
        body="body",
        created_at=datetime(2026, 6, 1, tzinfo=UTC),
        updated_at=datetime(2026, 6, 1, tzinfo=UTC),
    )


def test_a_source_that_declares_nothing_is_refused(tmp_path, capsys):
    """THE round-3 finding. Measured before the fix: 150 stored, 140
    re-scanned, ``added=0 updated=140 skipped=0 errors=0``, exit 0, and the 10
    rows that had no other copy were gone."""
    store = _store(tmp_path)
    store.upsert([_ten(i) for i in range(150)])
    src = _SourceTen([_ten(i) for i in range(140)])

    rc = main(
        ["ingest", "source-ten", "--rebuild"],
        _store=store,
        _sources={"source-ten": src},
    )

    assert rc == 2
    kept = {r.stable_id for r in store.query(QueryAST(source="source-ten"))}
    assert len(kept) == 150, "no last-copy row may be deleted"
    err = capsys.readouterr().err
    assert "rebuild_input" in err, "the refusal has to say how to opt in"


def test_declaring_the_property_is_what_allows_the_rebuild(tmp_path):
    """The other direction, so the default is a gate and not a wall. Same
    source, same run, one added declaration."""

    class _Declared(_SourceTen):
        def rebuild_input(self) -> str:
            return "an API this machine can re-read in full"

    store = _store(tmp_path)
    store.upsert([_ten(i) for i in range(150)])

    rc = main(
        ["ingest", "source-ten", "--rebuild", "--force", "--yes"],
        _store=store,
        _sources={"source-ten": _Declared([_ten(i) for i in range(140)])},
    )

    assert rc == 0
    assert len({r.stable_id for r in store.query(QueryAST(source="source-ten"))}) == 140


def test_a_plain_ingest_from_an_undeclared_source_is_untouched(tmp_path):
    """Refusing --rebuild must not cost the everyday path anything."""
    store = _store(tmp_path)
    store.upsert([_ten(0)])

    rc = main(
        ["ingest", "source-ten"],
        _store=store,
        _sources={"source-ten": _SourceTen([_ten(1)])},
    )

    assert rc == 0
    assert len({r.stable_id for r in store.query(QueryAST(source="source-ten"))}) == 2


def test_every_source_allowed_to_rebuild_says_what_refreshes_its_input():
    """The declaration's whole value is the written claim. An empty phrase
    passes ``isinstance`` and tells a reviewer nothing."""
    allowed = {
        name: src
        for name, src in _default_sources().items()
        if _rebuild_refusal(name, src) is None
    }
    assert allowed, "the inversion must not have refused everything"
    for name, src in allowed.items():
        assert src.rebuild_input().strip(), f"{name} declared an empty reason"


def test_the_refusal_follows_the_property_not_a_hand_kept_list():
    """The coherence check. Every source whose input is an export a human
    downloads is refused; every source with a live or machine-refreshed input
    is allowed. Membership is derived, so a new export-archive source cannot
    be forgotten off a list."""
    sources = _default_sources()
    refused = {
        name for name, src in sources.items() if _rebuild_refusal(name, src)
    }

    assert refused == {"chatgpt", "claude-web", "substack", "ticktick"}


def test_every_refused_source_says_what_refreshes_its_input():
    """The message has to name the thing that is not going to be regenerated,
    or an operator just reaches for --force."""
    for name, src in _default_sources().items():
        refusal = _rebuild_refusal(name, src)
        if refusal is None:
            continue
        assert "--rebuild" in refusal
        assert name in refusal
