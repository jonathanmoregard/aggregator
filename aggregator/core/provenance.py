"""Who composed an observation. ``human`` is the residual, never a claim.

WHY THIS EXISTS. ``type='user'`` is a TRANSPORT ROLE — it says which channel a
line arrived on, not who wrote it. Measured against the vendor's own structural
fields over a fixed 3,000-file sample of ``~/.claude/projects``, **59% of
``type='user'`` observations were composed by a machine**: hook-injected
classifier prompts, headless SDK briefs, subagent task briefs, slash-command
output, and client notices. A recall tool that presents those as the user's own
words hands back a majority of noise and calls it history — which is what a
search for "when did the user first ask for X" actually returned.

THE VENDOR HAS NO AUTHORSHIP FIELD, and this is the finding the whole design is
shaped around. ``promptSource='typed'`` and ``origin.kind='human'`` look like
the answer and are the same bug one layer down: the self-compact resume banner
is **43 of 43** 'typed' *and* 'human', because the harness injects it through
the interactive input channel and the client honestly records how it arrived.
``entrypoint`` is a worse trap — 2.1.222+ reports ``sdk-cli`` for ordinary
interactive sessions, and a classifier built on it measured 67.6% machine and
had to be thrown away.

SO THE RULE IS ASYMMETRIC, and it is the only thing keeping this honest:

    Structure and markers may only ever produce a positive MACHINE claim.
    ``human`` is whatever is left.

That asymmetry is also the failure mode. If the vendor renames or drops
``promptSource`` and ``origin`` tomorrow, the classifier loses machine claims
and degrades toward ``human`` — under-claiming, never mislabelling.

COVERAGE IS BIMODAL AND WILL IMPROVE ON ITS OWN. The structural fields are
present on 91-96% of lines from client 2.1.220+ and on 10-26% of anything
older, so structure decides roughly 47% of today's corpus and a larger share of
every future session. The closed marker set below covers the rest: 93.5%
precision, 23.3% recall on its own. It is a backstop for old lines, not the
mechanism.

``NULL`` IS NOT A MEMBER. It means "not classified yet" and is the backfill's
cursor, exactly as ``embedding_state IS NULL`` is the embed worker's. Never
store ``'unknown'``: a value that means "we looked and could not tell" and a
value that means "nobody has looked" are different facts, and collapsing them
loses the only thing that makes the backfill resumable.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

#: Nothing claimed a machine composed it. THE RESIDUAL — see the module
#: docstring. It is never returned because something positively identified a
#: human, because nothing in the data can.
HUMAN = "human"

#: Another Claude composed the text: a subagent task brief, a peer or
#: coordinator message, a task-notification relay, or any assistant turn.
AGENT = "agent"

#: A hook, skill or wrapper program on this machine composed it: Stop-hook
#: classifier prompts, ``claude -p`` briefs, self-compact resume banners.
HOOK = "hook"

#: A slash command and the local output it splices back in.
COMMAND = "command"

#: The Claude Code client injected it: system reminders, compaction
#: continuations, interrupt and API-error notices, tool results.
SYSTEM = "system"

#: The closed enum, in the order the design lists it. ``NULL`` is reserved and
#: is deliberately absent from this tuple: it is not a classification.
PROVENANCE_VALUES: tuple[str, ...] = (HUMAN, AGENT, HOOK, COMMAND, SYSTEM)

#: Everything that is a positive machine claim. This is what the DSL's
#: convenience value ``by:machine`` expands to, and it is NOT the same as
#: ``provenance IS NULL`` — unclassified is not a claim about authorship.
MACHINE_VALUES: tuple[str, ...] = (AGENT, HOOK, COMMAND, SYSTEM)

#: The DSL shorthand for :data:`MACHINE_VALUES`. A query word, never a stored
#: value: no row's ``provenance`` column ever holds this string.
MACHINE = "machine"

#: ``origin.kind`` values that name a machine author. ``human`` is absent on
#: purpose and must stay absent: see the module docstring's 43-of-43.
_AGENT_ORIGIN_KINDS = frozenset({"task-notification", "coordinator", "peer"})

#: Substrings that identify a hook-authored prompt riding the human channel.
#: Closed set, measured. ``ASSISTANT:`` is deliberately NOT here: it fires on
#: any body quoting a transcript, including one a human pasted, and it is the
#: marker to drop first whenever precision matters more than recall.
_HOOK_MARKERS = (
    "You are watching a Claude Code session",
    "Your context was just cleared to make room",
    "You also asked yourself to focus on this first",
)

#: A slash command's own envelope. Unambiguous: the client writes these tags.
_COMMAND_MARKERS = (
    "<command-name>",
    "<local-command-stdout>",
    "<local-command-caveat>",
)

#: Client-injected text that can appear anywhere in a body.
_SYSTEM_MARKERS = (
    "<system-reminder>",
    "This session is being continued from a previous",
)

#: Client notices, matched at the START of the body only. A human pasting
#: ``API Error: 529`` back into the chat to ask about it is a human turn, and
#: an unanchored match would relabel exactly the turns most worth finding.
_SYSTEM_PREFIXES = (
    "[Request interrupted",
    "API Error",
)

#: Observation types the model itself authored.
_AGENT_TYPES = frozenset({"assistant", "tool_use"})


@dataclass(frozen=True)
class LineStructure:
    """The vendor's structural fields on one JSONL line, and only those.

    Every field here can produce a positive MACHINE claim and none of them can
    produce a human one — that asymmetry is the module's whole contract. The
    defaults are the "field absent" state, which is what an older client's line
    looks like, so an unpopulated instance says nothing at all.
    """

    #: 100% coverage on every client version seen. True for subagent streams.
    is_sidechain: bool = False
    #: ~10% coverage. Present only when True; a client-injected line.
    is_meta: bool = False
    #: ~0.4% coverage. Present only when True; an auto-compaction summary.
    is_compact_summary: bool = False
    #: ``typed`` | ``queued`` | ``sdk`` | ``system``. ~40% coverage, client
    #: 2.1.201+. Only ``sdk`` and ``system`` are ever consulted.
    prompt_source: str | None = None
    #: ``origin.kind``: ``human`` | ``task-notification`` | ``coordinator`` |
    #: ``peer``. ~27% coverage, client 2.1.20x+. ``human`` is never consulted.
    origin_kind: str | None = None

    @classmethod
    def from_jsonl(cls, obj: Mapping[str, Any]) -> LineStructure:
        """Read the fields off a raw JSONL line, tolerating every absence.

        A wrong-shaped or missing field becomes the default, i.e. silence.
        Raising here would make an undocumented vendor rename an ingest
        failure; degrading makes it a loss of machine claims, which the
        residual rule already handles.
        """
        origin = obj.get("origin")
        origin_kind = None
        if isinstance(origin, Mapping):
            kind = origin.get("kind")
            if isinstance(kind, str):
                origin_kind = kind
        prompt_source = obj.get("promptSource")
        if not isinstance(prompt_source, str):
            prompt_source = None
        return cls(
            is_sidechain=bool(obj.get("isSidechain")),
            is_meta=bool(obj.get("isMeta")),
            is_compact_summary=bool(obj.get("isCompactSummary")),
            prompt_source=prompt_source,
            origin_kind=origin_kind,
        )


def classify(
    obs_type: str,
    body: str | None,
    *,
    structure: LineStructure | None = None,
    session_kind: str | None = None,
) -> str:
    """Return one of :data:`PROVENANCE_VALUES` for one observation.

    EVERY OBSERVATION, not just ``type='user'``. That is what keeps
    ``provenance IS NULL`` meaning exactly one thing — "not classified yet" —
    so the backfill can use it as its cursor. It also makes ``by:human`` on its
    own select precisely the human turns across the whole corpus, without
    needing ``type:user`` alongside it.

    ``structure`` is the JSONL evidence and is absent on the DB-only route,
    which has the body and the owning session's ``kind`` and nothing else.
    ``session_kind='subagent'`` is how that route recovers the ``agent`` claim
    that ``isSidechain`` carries at ingest.

    ORDER IS PRECEDENCE, and it is deliberate: a subagent's brief is
    agent-authored however many hook markers the brief happens to quote, and a
    client-injected line is the client's however it reads.
    """
    if obs_type in _AGENT_TYPES:
        return AGENT
    if obs_type == "hook":
        # The line names its own author; no other type does.
        return HOOK
    if obs_type != "user":
        # tool_result, attachment, system, progress, other, and every control
        # type: the client wrote the record. ``attachment`` is the loosest fit
        # (it holds both UserPromptSubmit hook results and file uploads, two
        # authors under one type) and it is 125k rows, so if this ever gets
        # refined that is where to start.
        return SYSTEM

    st = structure or LineStructure()
    if (
        session_kind == "subagent"
        or st.is_sidechain
        or st.origin_kind in _AGENT_ORIGIN_KINDS
    ):
        return AGENT
    if st.is_meta or st.is_compact_summary or st.prompt_source == "system":
        return SYSTEM
    if st.prompt_source == "sdk":
        # The largest machine class has NO body marker at all: headless briefs
        # supplied through the SDK, 29% of the sampled corpus.
        return HOOK

    text = body or ""
    if any(m in text for m in _COMMAND_MARKERS):
        return COMMAND
    if any(m in text for m in _HOOK_MARKERS):
        return HOOK
    if any(m in text for m in _SYSTEM_MARKERS):
        return SYSTEM
    if text.lstrip().startswith(_SYSTEM_PREFIXES):
        return SYSTEM
    return HUMAN
