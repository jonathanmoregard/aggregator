"""LLM topic tags on records: additive columns, union search, truthful counts.

THE DESIGN RULE IS SEPARATION. Source-written ``records.tags`` stays PRISTINE
— the tagger never writes into it — and the LLM's tags live in their own
additive ``llm_tags`` column with ``llm_tags_src_hash`` recording which
``src_hash`` the tags were computed from. The two meet only at query time:
``tag:`` filters against the UNION, and the ``records_fts`` ``tags`` column
indexes the space-joined union, so a record reachable by an LLM tag is
reachable by exactly the same search paths as one reachable by a source tag.

THE WATERMARK LIVES IN THE ROWS IT DESCRIBES, like ``provenance IS NULL`` and
``embedding_state IS NULL`` before it: a record needs tagging when
``llm_tags_src_hash`` is NULL or disagrees with the row's current
``src_hash``. No sidecar, so a kill can never leave the ledger ahead of the
data.

TRUTHFULNESS: a partially-tagged corpus must never look fully tagged, which
is what ``llm_tag_progress_by_source`` exists for — the same argument as
``embed_progress_by_source`` one abstraction over.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from aggregator.core.dsl import parse
from aggregator.core.store import Store
from aggregator.sources.base import Record

_TS = datetime(2026, 7, 25, tzinfo=UTC)


def _rec(sid: str, subject: str, body: str, tags=(), source: str = "github") -> Record:
    return Record(
        stable_id=sid,
        source=source,
        subject=subject,
        body=body,
        tags=list(tags),
        created_at=_TS,
        updated_at=_TS,
    )


def _store(tmp_path) -> Store:
    s = Store(db_path=tmp_path / "cache.db")
    s.migrate()
    return s


def _columns(store: Store, table: str) -> set[str]:
    return {
        row[1] for row in store._c().execute(f"PRAGMA table_info({table})")
    }


# --- column ensure ----------------------------------------------------------


def test_llm_tags_columns_exist_on_a_fresh_database(tmp_path):
    s = _store(tmp_path)
    cols = _columns(s, "records")
    assert "llm_tags" in cols
    assert "llm_tags_src_hash" in cols


def test_llm_tags_ensure_is_idempotent(tmp_path):
    s = _store(tmp_path)
    s.migrate()  # runs on every CLI invocation; a second pass must not raise
    s.migrate()
    cols = [
        row[1]
        for row in s._c().execute("PRAGMA table_info(records)")
        if row[1] in ("llm_tags", "llm_tags_src_hash")
    ]
    assert sorted(cols) == ["llm_tags", "llm_tags_src_hash"]


def test_llm_tags_columns_added_to_a_pre_v7_shaped_table(tmp_path):
    """A database whose records table predates the columns converges."""
    s = _store(tmp_path)
    c = s._c()
    c.execute("ALTER TABLE records DROP COLUMN llm_tags")
    c.execute("ALTER TABLE records DROP COLUMN llm_tags_src_hash")
    c.commit()
    assert "llm_tags" not in _columns(s, "records")
    s2 = Store(db_path=tmp_path / "cache.db")
    s2.migrate()
    cols = _columns(s2, "records")
    assert "llm_tags" in cols
    assert "llm_tags_src_hash" in cols


# --- tag: DSL matches the union --------------------------------------------


def test_tag_dsl_matches_a_source_tag_and_an_llm_tag(tmp_path):
    s = _store(tmp_path)
    s.upsert([_rec("github:a", "A", "body a", tags=["alpha"])])
    s.upsert([_rec("github:b", "B", "body b", tags=[])])
    s.write_llm_tags([("github:b", ["beta-topic"], "h-b")])

    by_source_tag = s.query(parse("tag:alpha"))
    assert [r.stable_id for r in by_source_tag] == ["github:a"]

    by_llm_tag = s.query(parse("tag:beta-topic"))
    assert [r.stable_id for r in by_llm_tag] == ["github:b"]

    assert s.count(parse("tag:beta-topic")) == 1


def test_a_record_with_only_llm_tags_is_found_by_tag_filter(tmp_path):
    s = _store(tmp_path)
    s.upsert([_rec("github:only-llm", "S", "b", tags=[])])
    s.write_llm_tags([("github:only-llm", ["orphan-topic"], "h1")])
    assert [r.stable_id for r in s.query(parse("tag:orphan-topic"))] == [
        "github:only-llm"
    ]


def test_llm_tags_visible_on_the_returned_record(tmp_path):
    s = _store(tmp_path)
    s.upsert([_rec("github:vis", "S", "b", tags=["src-tag"])])
    s.write_llm_tags([("github:vis", ["llm-topic"], "h1")])
    (rec,) = s.query(parse("tag:llm-topic"))
    assert rec.tags == ["src-tag"]  # pristine
    assert rec.llm_tags == ["llm-topic"]


def test_tag_filter_is_exact_not_a_like_wildcard(tmp_path):
    """``%`` and ``_`` in a tag value must match literally, not as LIKE
    wildcards: ``tag:a_b`` used to match a record tagged ``axb`` because the
    filter bound the raw value into a LIKE pattern."""
    s = _store(tmp_path)
    s.upsert(
        [
            _rec("github:lit", "S", "b", tags=["a_b"]),
            _rec("github:wild", "S2", "b2", tags=["axb"]),
            _rec("github:pct", "S3", "b3", tags=["50%"]),
            _rec("github:pfx", "S4", "b4", tags=["50-percent"]),
        ]
    )
    assert [r.stable_id for r in s.query(parse("tag:a_b"))] == ["github:lit"]
    assert [r.stable_id for r in s.query(parse("tag:50%"))] == ["github:pct"]
    # And on the llm side of the union too.
    s.upsert([_rec("github:llm-w", "S5", "b5", tags=[])])
    s.write_llm_tags([("github:llm-w", ["cxd"], "h1")])
    assert s.query(parse("tag:c_d")) == []


def test_non_ascii_source_tag_is_reachable(tmp_path):
    """Source tags are stored ``json.dumps``-escaped (``\\uXXXX``), so a LIKE
    against the raw JSON text could never see ``tag:blåbär``. json_each
    decodes the escapes back to text before comparing."""
    s = _store(tmp_path)
    s.upsert([_rec("github:sv", "S", "b", tags=["blåbär"])])
    assert [r.stable_id for r in s.query(parse("tag:blåbär"))] == ["github:sv"]
    assert s.count(parse("tag:blåbär")) == 1


def test_tag_filter_keeps_ascii_case_insensitivity(tmp_path):
    """LIKE folded ASCII case; callers may have leaned on it (a github label
    ``Bug`` found by ``tag:bug``), so the exact-match rewrite pins it."""
    s = _store(tmp_path)
    s.upsert([_rec("github:case", "S", "b", tags=["Bug"])])
    assert [r.stable_id for r in s.query(parse("tag:bug"))] == ["github:case"]


def test_capabilities_tag_inventory_includes_llm_only_tags(tmp_path):
    """``tags_by_source`` advertises what ``tag:`` can filter on, and ``tag:``
    filters the UNION — so a value only ``llm_tags`` carries must appear in
    the inventory, or the one place callers discover tags hides half of them."""
    s = _store(tmp_path)
    s.upsert([_rec("github:u1", "S", "b", tags=["src-only"])])
    s.upsert([_rec("github:u2", "S2", "b2", tags=[])])
    s.write_llm_tags([("github:u2", ["llm-only", "src-only"], "h1")])
    tags = s.capabilities()["tags_by_source"]["github"]
    assert "llm-only" in tags
    # Popularity counts the union per record: both records carry src-only.
    assert tags.index("src-only") < tags.index("llm-only")


# --- records_fts carries the union ------------------------------------------


def test_writing_llm_tags_updates_the_fts_row(tmp_path):
    s = _store(tmp_path)
    s.upsert([_rec("github:f", "release notes", "the body text", tags=[])])
    assert s.query(parse("quantization-tricks")) == []
    s.write_llm_tags([("github:f", ["quantization-tricks"], "h1")])
    hits = s.query(parse("quantization-tricks"))
    assert [r.stable_id for r in hits] == ["github:f"]
    # The rewrite must not cost the row its subject/body indexing.
    assert [r.stable_id for r in s.query(parse("release notes"))] == ["github:f"]


def test_fts_row_is_not_duplicated_by_a_tag_write(tmp_path):
    s = _store(tmp_path)
    s.upsert([_rec("github:d", "subject", "body", tags=["alpha"])])
    s.write_llm_tags([("github:d", ["beta"], "h1")])
    n = s._c().execute(
        "SELECT COUNT(*) FROM records_fts WHERE stable_id = 'github:d'"
    ).fetchone()[0]
    assert n == 1


def test_reingest_keeps_llm_tags_in_the_fts_union(tmp_path):
    """A re-upsert of the record must not drop the LLM tags from FTS.

    The row-write path rebuilds the record's ``records_fts`` row; if it wrote
    only the source tags, every ingest tick would silently strip the LLM tags
    from the lexical index while the columns still claimed the record tagged.
    """
    s = _store(tmp_path)
    s.upsert([_rec("github:r", "subject", "body one", tags=["alpha"])])
    s.write_llm_tags([("github:r", ["gamma-topic"], "h1")])
    # Content change → new src_hash → row rewritten.
    s.upsert([_rec("github:r", "subject", "body two", tags=["alpha"])])
    assert [r.stable_id for r in s.query(parse("gamma-topic"))] == ["github:r"]
    assert [r.stable_id for r in s.query(parse("tag:gamma-topic"))] == ["github:r"]


# --- source tags stay pristine ----------------------------------------------


def test_source_tags_are_never_clobbered_by_llm_tag_writes(tmp_path):
    s = _store(tmp_path)
    s.upsert([_rec("github:p", "S", "b", tags=["source-tag", "another"])])
    before = s._c().execute(
        "SELECT tags, src_hash FROM records WHERE stable_id = 'github:p'"
    ).fetchone()
    s.write_llm_tags([("github:p", ["llm-one", "llm-two"], "h1")])
    after = s._c().execute(
        "SELECT tags, src_hash FROM records WHERE stable_id = 'github:p'"
    ).fetchone()
    assert after["tags"] == before["tags"]
    assert json.loads(after["tags"]) == ["source-tag", "another"]
    assert after["src_hash"] == before["src_hash"]


def test_llm_tag_write_does_not_reset_embedding_state(tmp_path):
    """Tags are additive metadata; they must not re-queue the vector arm."""
    s = _store(tmp_path)
    s.upsert([_rec("github:e", "S", "b")])
    c = s._c()
    c.execute(
        "UPDATE records SET embedding_state = 'ok' WHERE stable_id = 'github:e'"
    )
    c.commit()
    s.write_llm_tags([("github:e", ["a-topic"], "h1")])
    row = c.execute(
        "SELECT embedding_state FROM records WHERE stable_id = 'github:e'"
    ).fetchone()
    assert row["embedding_state"] == "ok"


# --- the watermark ----------------------------------------------------------


def _needing(s: Store, sources=("github",)) -> list[str]:
    return s.ids_needing_llm_tags(sources)


def test_untagged_records_are_selected(tmp_path):
    s = _store(tmp_path)
    s.upsert([_rec("github:w1", "S", "b")])
    assert _needing(s) == ["github:w1"]


def test_a_tagged_record_with_matching_hash_is_skipped(tmp_path):
    s = _store(tmp_path)
    s.upsert([_rec("github:w2", "S", "b")])
    src_hash = s._c().execute(
        "SELECT src_hash FROM records WHERE stable_id = 'github:w2'"
    ).fetchone()["src_hash"]
    s.write_llm_tags([("github:w2", ["t-one"], src_hash)])
    assert _needing(s) == []


def test_a_changed_record_is_reselected(tmp_path):
    s = _store(tmp_path)
    s.upsert([_rec("github:w3", "S", "old body")])
    src_hash = s._c().execute(
        "SELECT src_hash FROM records WHERE stable_id = 'github:w3'"
    ).fetchone()["src_hash"]
    s.write_llm_tags([("github:w3", ["t-one"], src_hash)])
    s.upsert([_rec("github:w3", "S", "new body")])
    assert _needing(s) == ["github:w3"]


def test_a_null_src_hash_row_is_tagged_once_not_forever(tmp_path):
    """Pre-v4 rows carry ``src_hash IS NULL``; tagging must still converge.

    The write stores a sentinel for the missing hash so the row is not
    re-selected on every run — and the moment ingest stamps a real hash the
    disagreement re-queues it, which is the correct outcome: the tags were
    computed from text the hash does not describe.
    """
    s = _store(tmp_path)
    s.upsert([_rec("github:w4", "S", "b")])
    c = s._c()
    c.execute("UPDATE records SET src_hash = NULL WHERE stable_id = 'github:w4'")
    c.commit()
    assert _needing(s) == ["github:w4"]
    s.write_llm_tags([("github:w4", ["t-one"], None)])
    assert _needing(s) == []
    # A later ingest stamps the real hash → disagreement → re-tag.
    c.execute("UPDATE records SET src_hash = 'real' WHERE stable_id = 'github:w4'")
    c.commit()
    assert _needing(s) == ["github:w4"]


def test_needing_ids_respects_the_source_filter(tmp_path):
    s = _store(tmp_path)
    s.upsert([_rec("github:s1", "S", "b", source="github")])
    s.upsert([_rec("research:s2", "S", "b", source="research")])
    assert set(_needing(s, ("github",))) == {"github:s1"}
    assert set(_needing(s, ("github", "research"))) == {
        "github:s1",
        "research:s2",
    }


def test_records_for_tagging_returns_the_fields_the_prompt_needs(tmp_path):
    s = _store(tmp_path)
    s.upsert([_rec("github:rf", "the subject", "the body", tags=["x"])])
    (row,) = s.records_for_tagging(["github:rf"])
    assert row["stable_id"] == "github:rf"
    assert row["source"] == "github"
    assert row["subject"] == "the subject"
    assert row["body"] == "the body"
    assert row["src_hash"] is not None


# --- truthful coverage counts ----------------------------------------------


def test_llm_tag_progress_counts_a_partially_tagged_corpus(tmp_path):
    s = _store(tmp_path)
    s.upsert(
        [
            _rec("github:c1", "S", "b", source="github"),
            _rec("github:c2", "S", "b", source="github"),
        ]
    )
    s.upsert([_rec("research:c3", "S", "b", source="research")])
    h = s._c().execute(
        "SELECT src_hash FROM records WHERE stable_id = 'github:c1'"
    ).fetchone()["src_hash"]
    s.write_llm_tags([("github:c1", ["t-one"], h)])

    progress = {row["source"]: row for row in s.llm_tag_progress_by_source()}
    assert progress["github"]["total"] == 2
    assert progress["github"]["tagged"] == 1
    assert progress["github"]["pending"] == 1
    assert progress["github"]["state"] == "in_progress"
    assert progress["research"]["total"] == 1
    assert progress["research"]["tagged"] == 0
    assert progress["research"]["state"] == "not_started"


def test_llm_tag_progress_reports_complete_only_when_complete(tmp_path):
    s = _store(tmp_path)
    s.upsert([_rec("github:cc", "S", "b", source="github")])
    h = s._c().execute(
        "SELECT src_hash FROM records WHERE stable_id = 'github:cc'"
    ).fetchone()["src_hash"]
    s.write_llm_tags([("github:cc", ["t-one"], h)])
    progress = {row["source"]: row for row in s.llm_tag_progress_by_source()}
    assert progress["github"]["state"] == "complete"


def test_capabilities_exposes_llm_tag_coverage(tmp_path):
    s = _store(tmp_path)
    s.upsert([_rec("github:cap", "S", "b", source="github")])
    caps = s.capabilities()
    assert "llm_tag_coverage" in caps
    by_source = {row["source"]: row for row in caps["llm_tag_coverage"]}
    assert by_source["github"]["total"] == 1
    assert by_source["github"]["tagged"] == 0
    assert by_source["github"]["state"] == "not_started"
