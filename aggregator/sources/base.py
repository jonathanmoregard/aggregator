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

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
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

    v6: ``provenance`` says WHO COMPOSED THE TEXT, which ``type`` does not —
    ``type='user'`` is the channel a line arrived on, and 59% of the rows on
    that channel were written by a machine. See
    :mod:`aggregator.core.provenance` for the enum and why ``human`` is a
    residual rather than a claim. Kept LAST with a default so every existing
    keyword constructor stays valid; ``None`` means "not classified here" and
    the standalone backfill owns those rows.
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
    provenance: str | None = None  # v6; see aggregator.core.provenance


# Type alias for the tagged-union yield from ``Source.iter_entities``.
SessionEntity = SessionRow | ObservationRow


@dataclass
class IngestResult:
    added: int
    updated: int
    skipped: int
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PermanentFault:
    """Input a source can never parse, named precisely enough to be REMEMBERED.

    THE DISTINCTION THIS TYPE EXISTS TO DRAW. A source's ``errors`` list mixes
    two populations that look identical in it and are opposites in every way
    that matters. A locked database, an expired token, an unreachable directory
    are TRANSIENT: they must be reported on every run until somebody fixes
    them, because the next run might succeed. Two malformed lines in a JSONL
    file are PERMANENT: they will never parse, no run will ever succeed on
    them, and reporting them as a fresh failure every 30 minutes is a
    permanently-red alarm — which is the alarm an operator learns to dismiss
    unread, at the cost of the next real failure's audience.

    So a source DECLARES the second kind, one fault at a time, and everything
    it does not declare stays loud forever. Declaring is affirmative work; the
    default is loud. That direction is not negotiable — see
    ``imports/ingest_state.PoisonLedger``.

    THE FIELDS ARE THE IDENTITY, and each one is load-bearing:

    * ``scope`` — the artifact the fault is IN, as a filesystem path. It is what
      the ledger re-stats to notice that the file was rewritten, so it must name
      a real file rather than a logical id.
    * ``stamp`` — that file's identity at the moment the fault was found (see
      :func:`fault_stamp`). A stored fault whose scope no longer carries this
      stamp is about a file that has since changed.
    * ``reason`` — the CLASS of damage, e.g. ``"corrupt line(s)"``. Two
      different kinds of damage in one file are two faults.
    * ``detail`` — which records, EXACTLY and UNCAPPED (``"318,328"``). The
      rendered line caps its examples at five so one wrecked file cannot bury
      the run report; the identity must not, or a sixth bad line in a file that
      already has five would inherit the fifth's silence.
    * ``count`` — how many records this fault costs the index. Reported by
      ``aggregator status``, because a quarantined record nobody can count is a
      gap that reads as full coverage.
    * ``line`` — the exact error text this fault renders as, so the runner can
      move THAT line out of the run's errors and nothing else.
    """

    scope: str
    stamp: str
    reason: str
    detail: str
    count: int
    line: str

    @property
    def key(self) -> str:
        """The stable identity, hashed. What "already known" is decided by.

        Over scope + reason + detail and deliberately NOT over ``count``,
        ``stamp`` or ``line``: suppressing by count alone is what would let a
        different bad line inherit a known one's silence, and hashing the
        rendered text would make every future rewording re-alarm the whole
        ledger. ``\\x00`` as the separator because it cannot occur in a path,
        a reason or a line list, so no two different faults can collide by
        splicing at a delimiter.
        """
        raw = f"{self.scope}\x00{self.reason}\x00{self.detail}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def fault_stamp(path: Path | str) -> str:
    """A file's identity right now: ``"<mtime_ns>:<size>"``, or ``""`` if gone.

    WHAT IT IS FOR. A permanent fault goes quiet, so something has to notice
    when it stops being true — otherwise ``aggregator status`` reports a
    quarantine that no longer exists, which is the same lie as reporting none.
    The rule the ledger applies is "the file changed AND the fault was not
    re-reported", and this is the "changed" half.

    mtime AND size, because either alone is forgeable by accident: a rewrite
    within the filesystem's timestamp granularity keeps the mtime, and an edit
    that swaps one character keeps the size. Nanoseconds because whole seconds
    are exactly the granularity a fast rewrite hides inside.

    A missing or unreadable file answers ``""``, which never equals a stored
    stamp — so it reads as "changed", the loud direction: the fault is dropped
    and re-reported if it ever comes back.
    """
    try:
        st = Path(path).stat()
    except OSError:
        return ""
    return f"{st.st_mtime_ns}:{st.st_size}"


@dataclass
class QueryAST:
    """Parsed DSL query. Populated by core/dsl.py.

    v2 keys (Schema B) let the DSL address the sessions ontology directly:

    * ``session_id`` — everything under a session root (uses ``root_session_id``
      filter, so subagents included).
    * ``top_session_id`` — just the top-level, no subagents.
    * ``agent_id`` — just one subagent.
    * ``obs_type`` — filter observations by type (user/assistant/tool_use/...).
      A TRANSPORT ROLE, not an authorship claim — see ``provenance``.
    * ``provenance`` — filter observations by WHO COMPOSED them (``by:``).
      One of ``aggregator.core.provenance.PROVENANCE_VALUES``, or the
      shorthand ``'machine'`` for any non-human member. ``None`` means NO
      FILTER, exactly like ``obs_type``: nothing in this codebase applies a
      default here, because a default would silently narrow
      ``_first_user_prompt``, the frozen eval baseline and every
      ``matching_observations`` count at once.
    * ``active_from``/``active_to`` — activity-window overlap:
      ``first_ts <= active_to AND last_ts >= active_from``. Different from
      ``from_date``/``to_date`` which are point-in-time-created.

    v5 adds ``id_scope``, and it is the one field with NO DSL key. The hybrid
    retriever fuses an FTS5 arm and a vector arm into a ranked id list that no
    FTS5 MATCH expression can express, so the MCP layer computes those ids and
    hands them down here while every other filter still applies normally. The
    DSL surface is unchanged: nothing parses this, and no caller can set it.

    ``None`` means "no id filter". The EMPTY frozenset means "nothing
    matched" and is a real, different value — it renders as a no-match clause,
    because SQL ``IN ()`` is a syntax error rather than an empty result.
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
    # v6 authorship filter (``by:``). See the class docstring.
    provenance: str | None = None
    # v5 hybrid retrieval: internal-only id filter. See the class docstring.
    id_scope: frozenset[str] | None = None


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
