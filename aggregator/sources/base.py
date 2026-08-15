"""Abstract source protocol. Every ingester (sessions, github, future) implements Source.

Security note (spec §Security): sources MUST use read-only credentials only. The
github source enforces this via `gh auth` scope inspection (see spec constraint 1).
Every source's `ingest()` scrubs pre-store (Presidio + gitleaks) via core/scrub.py.

Schema v2 (Langfuse-derived, Schema B from
``research-agent/reports/0212132731c649a99d54eaf72c6e220c.md``): sessions now
emit two entity kinds — ``SessionRow`` (Langfuse "trace") and ``ObservationRow``
(Langfuse "observation"). GitHub keeps ``Record`` because PRs/issues are a
different ontology (units-of-work, not conversation streams); the sessions vs
records distinction is intentional and documented in ``store.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


@dataclass
class Record:
    """Uniform record shape for row-per-unit-of-work sources (GitHub PRs/issues).

    stable_id: mint once on first cache, never mutate (Gooen stable-ID discipline).
    Format: "<source>:<source-specific-id>", e.g. "github:owner/repo:42".
    Enforced by `stable_id_for()` below and by the store's upsert path.

    v2 note: sessions no longer use ``Record`` — they emit ``SessionRow`` +
    ``ObservationRow`` via ``iter_entities``. ``Record`` remains the shape for
    GitHub (and any future issue/PR-shaped source).
    """

    stable_id: str
    source: str
    subject: str  # short label for triage (session summary line, PR title, etc.)
    body: str  # full text for FTS indexing (already scrubbed by ingest pipeline)
    tags: list[str] = field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # source-specific metadata


# --- v2 Langfuse-derived entities (Schema B) ------------------------------
#
# Sessions source emits these two shapes via ``iter_entities``. The store
# writes them to ``sessions`` + ``observations`` tables respectively.
# ``root_session_id`` is denormalized onto every observation so
# "everything under X" is a single indexed equality (WHERE root_session_id=X),
# no recursion — the SOTA trick documented in the research report §2.


@dataclass
class SessionRow:
    """One row in the ``sessions`` table (Langfuse "trace").

    ``kind='session'`` for top-level JSONL files; ``kind='subagent'`` for
    ``<sessionId>/subagents/agent-*.jsonl`` files. For subagents the
    ``session_id`` is synthesized as ``<parent_sessionId>:<agentId>`` so it's
    unique across the two kinds without needing a compound key.

    v3: ``origin`` distinguishes where the stream came from —
    ``'claude-code'`` (default; local JSONLs) | ``'chatgpt'`` |
    ``'claude-web'`` (vendor data-export drops). Kept LAST with a default so
    existing constructors (all keyword-based) and any positional
    construction stay valid without changes.
    """

    session_id: str          # sessionId (top-level) or sessionId:agentId (subagent)
    root_session_id: str     # = session_id for top-level, parent's for subagents
    parent_session_id: str | None
    kind: str                # 'session' | 'subagent'
    agent_id: str | None
    agent_type: str | None
    spawned_by_tool_use_id: str | None  # best-effort recovered Task tool_use id
    cwd: str | None
    git_branch: str | None
    first_ts: datetime       # min timestamp across observations
    last_ts: datetime        # max timestamp across observations
    jsonl_path: str
    origin: str = "claude-code"  # 'claude-code' | 'chatgpt' | 'claude-web'


@dataclass
class ObservationRow:
    """One row in the ``observations`` table (Langfuse "observation").

    ``root_session_id`` is denormalized from the owning session — Langfuse
    pattern for "everything under X" as an indexed equality without recursion.
    ``parent_obs_id`` mirrors JSONL ``parentUuid`` — advisory only, may be null
    or point to an unwritten uuid (see anthropics/claude-code#22526).

    Granularity: one row per JSONL line. Multi-block ``message.content`` is
    collapsed: first text block into ``body``, first tool_use/tool_result
    block into ``tool_name``/``tool_use_id``. Documented SOTA row-per-message
    shape.
    """

    obs_id: str              # message uuid
    session_id: str          # FK to sessions.session_id
    root_session_id: str     # denormalized from the owning session
    parent_obs_id: str | None  # parentUuid; advisory only
    type: str                # 'user' | 'assistant' | 'tool_use' | 'tool_result' | 'system' | 'other'
    ts: datetime
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    tool_name: str | None
    tool_use_id: str | None
    body: str


# Type alias for the tagged-union yield from ``Source.iter_entities``.
SessionEntity = SessionRow | ObservationRow


@dataclass
class IngestResult:
    added: int
    updated: int
    skipped: int
    errors: list[str] = field(default_factory=list)


@dataclass
class QueryAST:
    """Parsed DSL query. Populated by core/dsl.py.

    v2 keys (Schema B) let the DSL address the sessions ontology directly:

    * ``session_id`` — everything under a session root (uses ``root_session_id``
      filter, so subagents included).
    * ``top_session_id`` — just the top-level, no subagents.
    * ``agent_id`` — just one subagent.
    * ``obs_type`` — filter observations by type (user/assistant/tool_use/...).
    * ``active_from``/``active_to`` — activity-window overlap:
      ``first_ts <= active_to AND last_ts >= active_from``. Different from
      ``from_date``/``to_date`` which are point-in-time-created.
    """

    source: str | None = None
    tags: list[str] = field(default_factory=list)
    from_date: datetime | None = None
    to_date: datetime | None = None
    text: str | None = None  # freeform FTS terms
    extra: dict[str, str] = field(default_factory=dict)  # per-source keys
    # v2 session-scoped filters (Schema B).
    session_id: str | None = None       # matches root_session_id (incl. subagents)
    top_session_id: str | None = None   # matches session_id (top-level only)
    agent_id: str | None = None
    obs_type: str | None = None
    active_from: datetime | None = None
    active_to: datetime | None = None


@runtime_checkable
class Source(Protocol):
    """Every ingester conforms to this shape. runtime_checkable so the DSL help
    generator (M2) can duck-type registered sources via isinstance().

    v2: sources come in two shapes now.

    * ``Record``-shaped (GitHub) — implements ``iter_records`` and CLI writes
      via ``store.upsert(records)``.
    * Entity-shaped (sessions) — implements ``iter_entities`` yielding
      ``SessionRow | ObservationRow`` and CLI writes via
      ``store.upsert_entities(entities)``.

    The CLI ingest dispatch checks for ``iter_entities`` first, then falls
    back to ``iter_records`` — additive so we don't break GitHub.
    """

    name: str

    def ingest(self, since: datetime | None) -> IngestResult: ...

    def record_shape(self) -> dict[str, str]:
        """Return {field_name: type_description}. Used by DSL help generator."""
        ...


@runtime_checkable
class ReadsManualExport(Protocol):
    """A source whose input is an archive a HUMAN downloads, by hand.

    The property that decides whether ``--rebuild`` can mean what it says.
    ``--rebuild`` adds exactly one thing over a plain ingest: it DELETEs the
    rows the re-scan did not reproduce. That is only ever safe when something
    on this machine keeps the input current — a live API, a synced directory,
    a tool that writes the files. When the input is a vendor export a person
    downloads occasionally, the stored rows are a SUPERSET of what any scan can
    produce (last month's export is gone from ~/Downloads), so the DELETE can
    only destroy data.

    Declared by the source itself, and checked structurally, so the rule is
    derived rather than hand-kept. It was hand-kept until round 2, and
    ``substack`` — the same Settings → Exports zip as ``chatgpt`` and
    ``claude-web`` — was missing from the list: its ``--rebuild`` was allowed
    and deleted last-copy rows at exit 0.

    Returns a short human phrase naming the export and who refreshes it (i.e.
    nobody), which goes into the refusal message. An operator who is only told
    "not supported" reaches for ``--force``.
    """

    def manual_export_input(self) -> str: ...


@runtime_checkable
class SupportsRebuild(Protocol):
    """A source that AFFIRMATIVELY declares ``--rebuild`` can mean what it says.

    The opt-in half of the rule, and the default is the reason it exists.
    ``--rebuild`` adds one thing to a plain ingest: it DELETEs every row the
    re-scan did not reproduce. Refusal used to be decided by evidence AGAINST —
    a ``ReadsManualExport`` declaration, a name on a list, an entity shape — so
    a source that declared nothing at all got the destructive path. That is
    fails-dangerous in exactly the direction that matters: forgetting a
    declaration is normal, and the cost of forgetting was permanent deletion of
    rows nothing can regenerate, at exit 0.

    So the question is inverted. A source is refused unless it says, in its own
    code, that a re-scan reproduces everything the DELETE can reach. Forgetting
    now costs a refusal an operator can read and fix, which is recoverable;
    the old default was not.

    Return a short human phrase naming what keeps the input current — a live
    API, a directory another tool syncs, a scan of files this machine owns. The
    phrase is the review artefact: it is where the claim is written down, and a
    reviewer who cannot believe it has found the bug before the data is gone.
    """

    def rebuild_input(self) -> str: ...


def stable_id_for(source: str, source_specific_id: str) -> str:
    """Mint a stable local ID. Enforces the "<source>:<id>" convention centrally
    so no source module hand-rolls the format inconsistently. See spec §Components
    and non-negotiable #5.
    """
    if not source or ":" in source:
        raise ValueError(f"invalid source name: {source!r} (must be non-empty, no colons)")
    if not source_specific_id:
        raise ValueError("source_specific_id must be non-empty")
    return f"{source}:{source_specific_id}"
