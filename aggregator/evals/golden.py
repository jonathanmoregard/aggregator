"""The frozen golden query set, and the loader that refuses a broken one.

IT IS A DATA FILE IN THE REPO, next to this module, and NOT a table in the
user's cache. A golden set stored inside the database it measures moves
whenever that database moves — a re-ingest, a schema migration, a model swap —
and a baseline that drifts with the thing it is baselining measures nothing.
In the repo it is reviewable in a diff, and freezing a new query is a commit
somebody has to make on purpose.

WHAT IS IN IT, and why each part earns its place (research report §8):

* **Identifier queries.** The corpus is saturated with symbol names, error
  codes, PR numbers and package versions. They are also where the FTS5
  tokenizer and the embedding model disagree most.
* **Natural-language queries.** What the agent actually types.
* **Negative queries** — queries this corpus genuinely cannot answer. The
  report is blunt that almost nobody includes these, and they are the ONLY way
  to test abstention: for a negative query you want the metrics near zero, and
  a run that starts returning hits for one is a regression that needs no
  labels to detect.

The five rows of the reported FTS5 failure table are frozen verbatim and
tagged ``fts5-failure``. They currently return nothing, and that is the point:
the baseline records today's abstention, so the fix shows up as maximum drift
on exactly those queries and on nothing else.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

#: The shapes a golden query can have. ``negative`` is not a flavour of
#: ``natural`` — it inverts the expectation, so it has to be its own kind.
KINDS = frozenset({"identifier", "natural", "negative"})

#: Where a query came from. Provenance is kept because "we made this up" and
#: "the agent really typed this" are worth different amounts when a metric
#: moves.
ORIGINS = frozenset({"repo", "authored", "report", "search-miss"})


class GoldenSetError(ValueError):
    """Raised on a malformed golden set or label file.

    Loud on purpose. A golden set that silently loads as empty would make
    every later regression run report a clean bill of health.
    """


@dataclass(frozen=True)
class GoldenQuery:
    id: str
    query: str
    kind: str
    origin: str
    note: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_negative(self) -> bool:
        """Whether this query is expected to return nothing at all."""
        return self.kind == "negative"


def golden_set_path() -> Path:
    return Path(__file__).with_name("golden_queries.json")


def labels_path() -> Path:
    """Where hand-made relevance labels live, once a human has made them.

    Absent by default — labelling ~30 queries is the human step recorded in
    ``pending_for_human.md``. Its absence must never be silently scored.
    """
    return Path(__file__).with_name("golden_labels.json")


def load_golden_queries(path: Path | None = None) -> list[GoldenQuery]:
    """Load and validate the frozen query set."""
    path = Path(path) if path is not None else golden_set_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise GoldenSetError(f"cannot read golden query set at {path}: {e}") from e

    raw = payload.get("queries") if isinstance(payload, dict) else None
    if not isinstance(raw, list) or not raw:
        raise GoldenSetError(
            f"golden query set at {path} is empty or has no 'queries' list; "
            "an empty set would make every regression run report success"
        )

    queries: list[GoldenQuery] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise GoldenSetError(f"golden query entry is not an object: {item!r}")
        qid = str(item.get("id", "")).strip()
        text = str(item.get("query", ""))
        kind = str(item.get("kind", ""))
        origin = str(item.get("origin", ""))
        if not qid:
            raise GoldenSetError(f"golden query entry has no id: {item!r}")
        if qid in seen:
            raise GoldenSetError(f"duplicate golden query id {qid!r}")
        if not text.strip():
            raise GoldenSetError(f"golden query {qid!r} has an empty query string")
        if kind not in KINDS:
            raise GoldenSetError(
                f"unknown kind {kind!r} for golden query {qid!r}; "
                f"expected one of {sorted(KINDS)}"
            )
        if origin not in ORIGINS:
            raise GoldenSetError(
                f"unknown origin {origin!r} for golden query {qid!r}; "
                f"expected one of {sorted(ORIGINS)}"
            )
        seen.add(qid)
        queries.append(
            GoldenQuery(
                id=qid,
                query=text,
                kind=kind,
                origin=origin,
                note=str(item.get("note", "")),
                tags=tuple(item.get("tags", ())),
            )
        )
    return queries


def load_labels(path: Path | None = None) -> dict[str, dict[str, int]]:
    """Load hand-made relevance labels, or ``{}`` when there are none.

    ``{}`` is the honest answer to "nobody has labelled anything", and the
    harness renders it as "no labels" rather than as a score. A malformed
    label file is a different thing entirely and raises.
    """
    path = Path(path) if path is not None else labels_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise GoldenSetError(f"cannot read labels at {path}: {e}") from e
    raw = payload.get("labels") if isinstance(payload, dict) else None
    if not isinstance(raw, dict):
        raise GoldenSetError(f"labels at {path} must be an object under 'labels'")
    labels: dict[str, dict[str, int]] = {}
    for qid, grades in raw.items():
        if not isinstance(grades, dict):
            raise GoldenSetError(
                f"labels for query {qid!r} must map result id -> integer grade, "
                f"got {type(grades).__name__}"
            )
        try:
            labels[str(qid)] = {str(rid): int(g) for rid, g in grades.items()}
        except (TypeError, ValueError) as e:
            raise GoldenSetError(f"non-integer grade in labels for {qid!r}: {e}") from e
    return labels


def suggest_from_misses(
    misses: Iterable[Mapping[str, object]],
    existing: Iterable[GoldenQuery],
) -> list[str]:
    """Zero-result queries that are not yet frozen, in first-seen order.

    Closes the loop the report describes: the ``search_misses`` log is "the
    cheapest source of golden-set queries you will ever get". Suggestions are
    PRINTED, never auto-appended — freezing a query has to stay a deliberate
    commit, or the golden set silently follows whatever the retriever is
    currently bad at.
    """
    frozen = {q.query for q in existing}
    out: list[str] = []
    for miss in misses:
        text = str(miss.get("query_text", ""))
        if not text or text in frozen or text in out:
            continue
        out.append(text)
    return out
