"""Who composed an observation — and the two fields that must never answer it.

`type:user` is a TRANSPORT ROLE, not an authorship claim. Measured against the
vendor's own structural fields on a fixed 3,000-file sample, 59% of
``type='user'`` observations were composed by a machine: hook-injected
classifier prompts, headless SDK briefs, subagent task briefs, slash-command
output, client notices. The substring markers the first census used score 93.5%
precision but only 23.3% recall, so they undercount that population threefold.

THE VENDOR HAS NO AUTHORSHIP FIELD, which is the finding this whole module is
shaped around. ``promptSource='typed'`` and ``origin.kind='human'`` are
transport labels with exactly the same bug one layer down: the self-compact
resume banner is 43 of 43 'typed' AND 'human', because the harness injects it
through the interactive input channel and the client honestly records how it
arrived. So structure may only ever produce a positive MACHINE claim, and
``human`` is the residual — whatever nothing claimed.
"""

import pytest

from aggregator.core.provenance import (
    AGENT,
    COMMAND,
    HOOK,
    HUMAN,
    MACHINE_VALUES,
    PROVENANCE_VALUES,
    SYSTEM,
    LineStructure,
    classify,
)


def test_the_enum_is_closed_and_human_is_not_machine():
    assert PROVENANCE_VALUES == (HUMAN, AGENT, HOOK, COMMAND, SYSTEM)
    assert set(MACHINE_VALUES) == set(PROVENANCE_VALUES) - {HUMAN}
    assert "unknown" not in PROVENANCE_VALUES


def test_an_ordinary_typed_turn_is_human_by_residual():
    assert classify("user", "make the fan stop blasting please") == HUMAN


def test_an_empty_user_turn_is_still_human():
    """Nothing claimed a machine wrote it, so nothing may claim one did."""
    assert classify("user", "") == HUMAN


# --- the two fields that must never be believed -----------------------------


def test_typed_prompt_source_is_not_evidence_of_a_human():
    """43 of 43 self-compact resume banners are 'typed'. It is transport."""
    banner = "Your context was just cleared to make room. Resume from the handoff."
    assert (
        classify(
            "user",
            banner,
            structure=LineStructure(prompt_source="typed", origin_kind="human"),
        )
        == HOOK
    )


def test_origin_human_is_not_evidence_of_a_human():
    focus = "You also asked yourself to focus on this first: finish the gate."
    assert (
        classify("user", focus, structure=LineStructure(origin_kind="human")) == HOOK
    )


def test_a_typed_label_on_a_plain_turn_changes_nothing():
    """It cannot promote either — ``human`` was already the residual answer."""
    plain = "do sota research on aggregator systems"
    assert classify("user", plain) == HUMAN
    assert (
        classify("user", plain, structure=LineStructure(prompt_source="typed"))
        == HUMAN
    )


@pytest.mark.parametrize("prompt_source", [None, "typed", "queued"])
@pytest.mark.parametrize("origin_kind", [None, "human"])
@pytest.mark.parametrize(
    "body",
    [
        "an ordinary typed turn",
        "Your context was just cleared to make room",
        "<command-name>/loop</command-name>",
        "<system-reminder>x</system-reminder>",
        "",
    ],
)
def test_a_human_leaning_transport_label_never_changes_the_answer(
    prompt_source, origin_kind, body
):
    """THE ASYMMETRY, stated as a property rather than as a comment.

    ``typed``, ``queued`` and ``origin.kind='human'`` are the labels that look
    like a human claim. None of them may move the verdict in any direction:
    the answer with the label must be byte-identical to the answer without it.
    A future edit that "improves" the classifier by trusting one of them fails
    here rather than quietly relabelling 43 resume banners as the user.
    """
    assert classify(
        "user",
        body,
        structure=LineStructure(prompt_source=prompt_source, origin_kind=origin_kind),
    ) == classify("user", body)


# --- positive machine claims from structure ---------------------------------


def test_sidechain_is_an_agent_brief():
    assert (
        classify("user", "Research X and report back", structure=LineStructure(is_sidechain=True))
        == AGENT
    )


def test_a_subagent_session_makes_its_user_turns_agent_authored():
    """The DB-only route has no ``isSidechain``; ``sessions.kind`` says it."""
    assert classify("user", "Research X", session_kind="subagent") == AGENT


@pytest.mark.parametrize("kind", ["task-notification", "coordinator", "peer"])
def test_machine_origin_kinds_are_agent(kind):
    assert classify("user", "done", structure=LineStructure(origin_kind=kind)) == AGENT


def test_sdk_prompt_source_is_a_hook():
    """Headless briefs have NO body marker at all — 29% of the sample."""
    assert (
        classify(
            "user",
            "You are a NixOS config drift analyzer.",
            structure=LineStructure(prompt_source="sdk"),
        )
        == HOOK
    )


@pytest.mark.parametrize(
    "structure",
    [
        LineStructure(is_meta=True),
        LineStructure(is_compact_summary=True),
        LineStructure(prompt_source="system"),
    ],
)
def test_client_injected_lines_are_system(structure):
    assert classify("user", "whatever the body says", structure=structure) == SYSTEM


# --- markers, for the ~53% of the corpus structure cannot decide ------------


@pytest.mark.parametrize(
    "body",
    [
        "You are watching a Claude Code session for a specific failure mode",
        "Your context was just cleared to make room for more work",
        "You also asked yourself to focus on this first",
    ],
)
def test_hook_markers(body):
    assert classify("user", body) == HOOK


@pytest.mark.parametrize(
    "body",
    ["<command-name>/loop</command-name>", "<local-command-stdout>ok", "<local-command-caveat>x"],
)
def test_command_markers(body):
    assert classify("user", body) == COMMAND


@pytest.mark.parametrize(
    "body",
    [
        "<system-reminder>context follows</system-reminder>",
        "This session is being continued from a previous conversation",
    ],
)
def test_system_markers(body):
    assert classify("user", body) == SYSTEM


@pytest.mark.parametrize("body", ["[Request interrupted by user]", "API Error: 529"])
def test_client_notices_are_matched_at_the_start_only(body):
    assert classify("user", body) == SYSTEM


def test_a_quoted_client_notice_mid_body_is_not_a_client_notice():
    """A human pasting an error back is not the client injecting one."""
    assert classify("user", "I keep getting API Error: 529, what gives?") == HUMAN


def test_assistant_colon_is_not_a_marker():
    """It fires on any body quoting a transcript, including a pasted one.

    Measured as the riskiest marker in the set and deliberately left out: the
    classifier-prompt population it would catch is already caught by
    ``promptSource='sdk'`` and by the hook markers.
    """
    assert classify("user", "ASSISTANT: Eval green. Apply:") == HUMAN


# --- the two markers the frozen design set missed ---------------------------
#
# Measured on the live cache read-only after the first classifier landed: 888
# residual-``human`` rows carry a background-task envelope and 342 carry the
# Stop-hook classifier's own preamble. Neither is in the design's §2.2 set, and
# ``golden-queries.md`` names both as top pollutants in four fixtures
# (pol-gymma-ben, pol-unwedged, fn-branch-protection,
# fn-natural-language-question). Left out, criterion E freezes ranks against a
# corpus that still calls them the user's words.


def test_a_background_task_envelope_is_agent_authored():
    """The harness relays another agent's result on the user channel.

    Classed ``agent`` rather than ``system`` so the marker route and the
    structural route give the SAME answer for the same row: ``origin.kind =
    'task-notification'`` is already in ``_AGENT_ORIGIN_KINDS``. A row whose
    class flips depending on which route reached it first is worse than either
    class, because the backfill mixes both routes over one column.
    """
    body = "<task-notification>\n<task-id>a934e875</task-id>\n<status>completed</status>"
    assert classify("user", body) == AGENT


def test_the_envelope_outranks_a_client_reminder_wrapped_around_it():
    """Precedence, and it matches the structural block one layer up.

    The client wraps the relay in its own reminder, so a real row carries both
    tags. ``agent`` is the inner author and the more specific claim; ``system``
    would be true of the wrapper and useless about the content.
    """
    body = "<system-reminder>\n<task-notification>agent finished</task-notification>\n"
    assert classify("user", body) == AGENT


def test_the_classifier_prompts_own_preamble_is_a_hook_marker():
    """Distinct from ``ASSISTANT:``, which is deliberately excluded above.

    ``ASSISTANT:`` fires on any body quoting a transcript, a human-pasted one
    included. This phrase is a role instruction a wrapper program writes to
    open a headless prompt — a human writing *about* it says "the classifier",
    not "You are a classifier".
    """
    body = "You are a classifier. Read the session below and answer with one word."
    assert classify("user", body) == HOOK


# --- every observation gets classified, not just type='user' ----------------


@pytest.mark.parametrize("obs_type", ["assistant", "tool_use"])
def test_model_authored_types_are_agent(obs_type):
    assert classify(obs_type, "some body") == AGENT


@pytest.mark.parametrize(
    "obs_type",
    ["tool_result", "attachment", "system", "progress", "other", "queue-operation"],
)
def test_client_authored_types_are_system(obs_type):
    assert classify(obs_type, "some body") == SYSTEM


def test_a_hook_typed_line_names_its_own_author():
    assert classify("hook", "PostToolUse fired") == HOOK


def test_every_answer_is_a_member_of_the_enum():
    """No route may invent a sixth value, and none may return ``'unknown'``."""
    bodies = ["", "plain", "<command-name>x", "API Error: 1", "<system-reminder>"]
    types = ["user", "assistant", "tool_use", "tool_result", "system", "made-up"]
    for t in types:
        for b in bodies:
            assert classify(t, b) in PROVENANCE_VALUES


# --- reading the structure off a raw JSONL line -----------------------------


def test_line_structure_reads_the_vendor_fields():
    st = LineStructure.from_jsonl(
        {
            "isSidechain": True,
            "isMeta": True,
            "isCompactSummary": True,
            "promptSource": "sdk",
            "origin": {"kind": "task-notification"},
        }
    )
    assert st == LineStructure(
        is_sidechain=True,
        is_meta=True,
        is_compact_summary=True,
        prompt_source="sdk",
        origin_kind="task-notification",
    )


def test_line_structure_degrades_to_silence_on_an_old_line():
    """``promptSource``/``origin`` only exist on ~2.1.20x and later.

    Absent, they must say NOTHING rather than something wrong — which is what
    makes the classifier degrade toward the ``human`` residual if the vendor
    ever drops them, instead of mislabelling.
    """
    st = LineStructure.from_jsonl({"type": "user", "entrypoint": "cli"})
    assert st == LineStructure()
    assert classify("user", "an old plain turn", structure=st) == HUMAN


def test_line_structure_ignores_a_wrong_shaped_origin():
    st = LineStructure.from_jsonl({"origin": "human"})
    assert st.origin_kind is None
