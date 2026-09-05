"""Criterion E — the frozen regression suite for the whole mission.

WHAT THIS IS. Twenty-five ``(query -> known-correct row)`` pairs, mined from a
real incident and verified against the user's live cache read-only, replayed
here against a SYNTHETIC corpus that reproduces each pair's SHAPE. Every query
string below is the real one, typed by the real caller. Every body below is
written for this file.

WHY THE CORPUS IS SYNTHETIC AND THE QUERIES ARE NOT. The pairs were frozen as
``(query, obs_id)`` against the developer's own history, and this repository is
public: committing the bodies, subjects or session ids behind them would
publish a personal corpus, which is disqualifying rather than merely risky. It
would also be a flaky test by construction — the ingest timer is live, and the
counts behind those pairs were measured drifting inside forty minutes. The
query strings are what carry the signal (they are the phrasings that failed),
so they are kept verbatim and only the corpus is replaced.

WHAT "RANK IN TOP 5" MEANS HERE, AND WHAT IT DOES NOT. This path has NO
relevance ordering. ``drilldown=True`` is ``ORDER BY ts ASC`` and
``drilldown=False`` is ``ORDER BY last_ts DESC``; ``bm25`` is never consulted,
and the observation vector arm is cold (0 of 563,991 rows embedded on the live
cache), so ``_apply_hybrid`` reports ``engaged=False`` and the route is pure
FTS5. A rank assertion is therefore a statement about **what the filter
admits**, not about scoring: the target ranks where it does because the rows
that would have preceded it are no longer matched. That is exactly the property
worth pinning — every rank that moved in this mission moved because a filter
changed, never because a score did — but it must not be read as a ranking
guarantee.

WHY EVERY OTHER FIXTURE CARRIES A QUOTED PHRASE. The defect criterion D
actually found was that ``fts5_match_query`` shattered a caller's quoted phrase
into independent words. It shipped in ``b4eab9b`` and nothing caught it for six
days because **0 of the 86 frozen golden queries in
``aggregator/evals/golden_queries.json`` contains a quote**. A regression suite
without quoted phrases has that same blind spot by construction.

THE INDEX DOES NOT STEM. ``tokenize='unicode61 remove_diacritics 2'``, no
``porter`` — ``report`` and ``reports`` are different terms, and turning that on
would mean rebuilding ``obs_fts`` over half a million rows. Several fixtures
here fail *because* of it; none is written to need it.

RE-FREEZE THESE NUMBERS WHEN THE SESSIONS EMBED BACKFILL RUNS. Sessions and
subagents are fourth in the user's stated backfill order, so there is time — but
once observations carry vectors the hybrid arm engages, ``ORDER BY ts ASC``
stops being the ordering, and every rank in this file is stale. The corpus is
local and deterministic; the *meaning* of the ranks is not.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

import aggregator.mcp as mcp
from aggregator.core.provenance import (
    HOOK,
    HUMAN,
    MACHINE_VALUES,
    LineStructure,
    classify,
)
from aggregator.core.store import Store
from aggregator.sources.base import ObservationRow, SessionRow

# ---------------------------------------------------------------------------
# The corpus.
#
# Shapes, not contents. Each family below reproduces one thing that made a real
# pair interesting: an answer with near-misses around it, a machine-authored
# row that satisfies the same words because its body quotes a transcript, a
# single enormous body that buried two separate answers, or a phrasing whose
# right answer is nothing at all.
# ---------------------------------------------------------------------------

#: What a headless ``claude -p`` brief carries on its JSONL line. It is the
#: largest machine class and has NO body marker, so the fixtures whose real
#: pollutants are headless briefs have to supply the structural evidence a real
#: ingest would — otherwise they would be testing the marker backstop instead
#: of the mechanism. See ``test_the_marker_backstop_does_not_cover_a_headless_
#: brief`` for the half of this that is a measured gap rather than a choice.
_SDK = LineStructure(prompt_source="sdk")


def _at(month: int, day: int, hour: int = 12, minute: int = 0, second: int = 0):
    """A synthetic 2026 timestamp. The month spread matches the real set's
    (2026-03 .. 2026-08); the times are invented, because ``ts`` is the sort
    key and therefore IS the rank, but the user's actual clock is not.
    """
    return datetime(2026, month, day, hour, minute, second, tzinfo=UTC)


def _hook_watch(quoted: str, pad: int = 60) -> str:
    """A Stop-hook classifier prompt. THE ARCHETYPAL POLLUTANT: it satisfies
    every loose word of a query because it quotes the transcript back at the
    model, and it arrives on the user channel, so ``type:user`` cannot tell it
    apart from the turn it is quoting."""
    return (
        "You are watching a Claude Code session for a specific failure mode. "
        "TRANSCRIPT EXCERPT FOLLOWS, quoted verbatim for you to judge: "
        f"{quoted} "
        "Decide whether the assistant complied with the standing instruction. "
        + "Additional judging guidance that pads this prompt out to the size a "
          "real hook prompt reaches. " * pad
    )


def _classifier(quoted: str, pad: int = 60) -> str:
    """The other hook shape, and the one the user chose to add to the frozen
    marker set after it was measured leaving 342 rows filed as human."""
    return (
        "You are a classifier for a Claude Code Stop hook. You decide if the "
        "assistant stopped its turn too early. TRANSCRIPT: "
        f"{quoted} "
        + "Grading rubric filler that pads the prompt to a realistic size. " * pad
    )


def _task_notification(quoted: str, pad: int = 8) -> str:
    """The harness's envelope around another agent's result, relayed on the
    user channel. Classed ``agent`` so the marker route and the structural
    route agree on the same row."""
    return (
        "<task-notification>An agent you dispatched has finished. Its final "
        f"report quoted the request: {quoted}"
        + " Relay the result to the user. " * pad
        + "</task-notification>"
    )


def _resume_banner(quoted: str, pad: int = 90) -> str:
    """A self-compact resume banner: the client's own summary of a previous
    session, injected through the interactive channel, which is why the
    vendor's ``promptSource='typed'`` and ``origin.kind='human'`` are both true
    of it and both useless."""
    return (
        "This session is being continued from a previous conversation that ran "
        "out of context. The summary below covers what was decided: "
        f"{quoted} "
        + "Continue from where the summary leaves off without repeating work. " * pad
    )


def _research_label(quoted: str, pad: int = 70) -> str:
    """A headless labelling brief. Carries no marker the frozen set covers —
    it is classified from the JSONL structure or not at all."""
    return (
        "You are reviewing a real Claude Code transcript excerpt to produce a "
        f"research label. EXCERPT: {quoted} "
        + "Label taxonomy notes that pad this prompt to its real size. " * pad
    )


_FILLER = (
    "building llm powered applications with claude: reference prose that "
    "carries none of the words any query below is looking for. "
)

#: The only reachable part of the giant body, buried in its middle so a snippet
#: taken from the head would show nothing. Every word here is deliberately
#: NON-ADJACENT to the phrase some fixture searches for: that is the whole
#: point — under a shattered phrase this row satisfies the query, under an
#: adjacency phrase it does not.
_SKILL_MIDDLE = (
    "cost control: keep usage inside the plan, and note that a low ceiling on "
    "the monthly cap is a budget decision rather than a technical one. the "
    "tool permission surface generates too many separate prompts unless the "
    "config groups them. billing integrations such as stripe deliver a "
    "webhook whose signature must be verified before the payload is trusted. "
    "deployment state is held in terraform, and a lock on that state file "
    "prevents two applies at once. a kubernetes cluster installed with helm "
    "keeps a chart revision so a rollback is one command. "
)

#: ~353 000 characters, the order of the real one (342,459). A single body of
#: this size, stored as ``type='user'``, sat at rank 1 of two different
#: fixtures and made ``page_size=2`` a 343 KB payload. Criterion B exists
#: because of this row and criterion A's snippet is what makes it readable.
_SKILL_BODY = _FILLER * 1400 + _SKILL_MIDDLE + _FILLER * 1400

#: ~76 000 characters, the order of the drift-analyzer prompt that filled ranks
#: 1-4 of the terraform over-eagerness canary when it returned 1,845 rows for a
#: phrase the corpus does not contain anywhere.
_DRIFT_BODY = (
    "You are a NixOS config drift analyzer. Compare the declared state against "
    "the machine and report every divergence. Infrastructure elsewhere uses "
    "terraform; this host does not, and the lock file it would need is absent. "
) * 260

_CONFIG_TURN = (
    "I want to open up your config file! the basic idea: allow everything, "
    "then use hooks to block bad stuff."
)


def _build_corpus(db_path) -> Store:
    """Its own temporary database. NEVER the live cache — a test that touches
    it is a defect even if it passes."""
    store = Store(db_path=db_path)
    store.migrate()
    rows: list[ObservationRow] = []

    def add(obs_id, session_id, ts, body, obs_type="user", structure=None):
        rows.append(ObservationRow(
            obs_id=obs_id, session_id=session_id, root_session_id=session_id,
            parent_obs_id=None, type=obs_type, ts=ts, model=None,
            input_tokens=None, output_tokens=None, tool_name=None,
            tool_use_id=None, body=body,
            # Classified by the SHIPPED classifier rather than hand-labelled,
            # so a regression in criterion C's marker set or precedence order
            # shows up here as a moved rank instead of as nothing at all.
            provenance=classify(obs_type, body, structure=structure),
        ))

    # --- the clickable-links incident -------------------------------------
    # Two older rows carrying both words NON-ADJACENTLY. Under the shattered
    # phrase they match and, being older, outrank the answer; under an
    # adjacency phrase they are gone. This is what makes the acceptance test's
    # rank 1 a result rather than an accident of there being nothing else.
    add("o-noise-both-1", "sess-early-noise", _at(3, 22, 8, 0),
        "make the buttons clickable and then collect the sitemap links into "
        "one file so the crawler has something to chew on")
    add("o-noise-both-2", "sess-early-noise", _at(4, 2, 9, 30),
        "the footer is clickable now; the broken links are listed in the "
        "report I left on the branch")

    # The answer, and a session whose FIRST turn is about something else — so
    # a card that showed its subject instead of its hit would show nothing.
    add("o-report-open", "sess-report-style", _at(7, 28, 9, 0),
        "start a fresh branch for the delivery-notes parser")
    add("o-report-answer", "sess-report-style", _at(7, 28, 15, 32),
        "link prs for me. also: whenever you hand me a status report at the "
        "end of a task, put clickable links in it instead of bare numbers.")

    # The pollutants. They quote the caller's own question back, which is why
    # a natural-language query finds them and not the answer.
    add("o-hook-report-a", "sess-hook-a", _at(8, 2, 10, 0),
        _classifier(
            "USER: when did I first ask for clickable links in status "
            "reports? ASSISTANT: you asked me to link the PR in every report "
            "on the status of the branch."))
    add("o-hook-report-b", "sess-hook-b", _at(8, 2, 11, 0),
        _classifier(
            "USER: I want clickable links. did you ask first, or did you just "
            "hand back a report on the status of the run? there were no "
            "status reports for the whole hour."))
    add("o-label-report", "sess-label-a", _at(8, 2, 12, 0),
        _research_label(
            "USER: when did I first ask for clickable links in status "
            "reports"),
        structure=_SDK)

    # A session that carries each phrase in a DIFFERENT turn: nothing for the
    # default scope, and the only thing ``scope:session`` has to offer.
    add("o-split-1", "sess-split", _at(6, 14, 10, 0),
        "please include a PR link every time you finish something")
    add("o-split-2", "sess-split", _at(6, 14, 13, 0),
        "and put it in the status report as well, not just in chat")

    # --- the giant body, and the two answers it buried --------------------
    add("o-skill-llm-apps", "sess-skill-import", _at(4, 11, 7, 0), _SKILL_BODY)

    add("o-usage-cap", "sess-usage", _at(8, 2, 20, 13),
        "disable the nightly loop, I have low usage cap this week. we need to "
        "measure its impact more carefully before turning it back on.")
    add("o-usage-relay", "sess-usage-relay", _at(8, 5, 9, 0),
        _task_notification("the run was paused because of a low usage cap"))

    add("o-permission-prompts", "sess-permissions", _at(8, 1, 6, 54),
        "I have too many permission prompts. I want a userspace config where "
        "every prompt is recorded and then evaluated in one batch.")

    # --- hook pollution around a human turn -------------------------------
    add("o-hook-gym-1", "sess-gym-hook-1", _at(4, 23, 10, 0),
        _hook_watch("the habit nudge said gymma ben was unchecked"))
    add("o-gymma-ben", "sess-gym", _at(6, 11, 16, 19),
        "the habit nudge is off: it just told me gymma ben was unchecked, and "
        "it is not!")
    add("o-hook-gym-2", "sess-gym-hook-2", _at(6, 11, 18, 0),
        _hook_watch("the user reported that gymma ben was wrongly unchecked"))
    add("o-hook-gym-3", "sess-gym-hook-3", _at(6, 25, 9, 0),
        _hook_watch("gymma ben is still being reported as unchecked"))
    add("o-hook-gym-4", "sess-gym-hook-4", _at(6, 25, 15, 0),
        _classifier("the nudge for gymma ben fired twice in one evening"))

    add("o-unwedged-note", "sess-wedge", _at(5, 23, 9, 0),
        "the browser unwedged itself after a restart, so nothing to do today")
    add("o-hook-wedge-1", "sess-wedge-hook-1", _at(5, 23, 11, 0),
        _hook_watch("the user said the browser has unwedged before"))
    add("o-unwedged", "sess-wedge-2", _at(6, 11, 11, 7),
        "I want a permanent fix for this: historically the browser has "
        "unwedged on its own. I want it to stop wedging in the first place.")
    add("o-hook-wedge-2", "sess-wedge-hook-2", _at(6, 11, 14, 0),
        _classifier("asked for a permanent fix so the browser never unwedged "
                    "again"))

    # --- result sets that are 100% machine --------------------------------
    add("o-resume-fan", "sess-resume", _at(8, 27, 11, 3),
        _resume_banner(
            "the user said that when it runs the fan blasts, so limit it a "
            "bit more even if it takes a bit longer"))

    for i, ts in enumerate((_at(8, 2, 13, 0), _at(8, 2, 14, 0),
                            _at(8, 3, 9, 0))):
        add(f"o-label-embed-{i}", f"sess-label-embed-{i}", ts,
            _research_label(
                "the backfill order the user gave was dropbox first, then the "
                "blog, then the llm exports, and claude code sessions last"),
            structure=_SDK)

    # --- an answer buried under six relays --------------------------------
    for i, ts in enumerate((_at(4, 11, 21, 47), _at(4, 12, 8, 0),
                            _at(4, 13, 8, 0), _at(4, 14, 8, 0),
                            _at(4, 15, 8, 0), _at(4, 16, 8, 0))):
        add(f"o-jhana-machine-{i}", f"sess-jhana-machine-{i}", ts,
            _task_notification(
                "the user asked about the jhanas-maxxing notes and wanted a "
                "skill scheduled from them"))
    add("o-jhanas-maxxing", "sess-jhana", _at(4, 17, 9, 0),
        "look at the jhanas-maxxing notes and, using what is there, schedule "
        "a skill creation that embeds the approach")

    # --- the plain positives, each with its own near-miss -----------------
    add("o-survival-noise", "sess-survival-noise", _at(3, 10, 9, 0),
        "survival of the build depends on which corpuses we index first")
    add("o-survival", "sess-survival", _at(3, 22, 20, 52),
        "this workspace is for building survival corpuses for collapse "
        "scenarios! go go")
    for i, ts in enumerate((_at(5, 2, 9, 0), _at(6, 3, 9, 0))):
        add(f"o-survival-relay-{i}", f"sess-survival-relay-{i}", ts,
            _task_notification("the workspace is about survival corpuses"))

    add("o-rag-noise", "sess-rag-noise", _at(5, 5, 9, 0),
        "the RAG demo needs a way to implement paging over the results, so "
        "hand it to me when it works")
    add("o-implement-rag", "sess-rag", _at(8, 8, 7, 28),
        "do sota research on aggregator systems and then implement RAG over it")
    add("o-rag-relay", "sess-rag-relay", _at(8, 9, 7, 0),
        _task_notification("the brief was to implement RAG over it after the "
                           "research"))

    add("o-booking-noise", "sess-booking-noise", _at(5, 6, 9, 0),
        "double check the booking flow before the demo on friday")
    add("o-double-booking", "sess-calendar", _at(7, 6, 18, 48),
        "keeping an eye on their calendar to avoid double booking, for free, "
        "makes planning much easier")
    add("o-hook-calendar", "sess-calendar-hook", _at(7, 7, 9, 0),
        _hook_watch("the pitch was about avoiding double booking "
                    "automatically"))

    add("o-keys-noise", "sess-keys-noise", _at(4, 1, 9, 0),
        "scrolling: page down should move down one page and page up should "
        "move up, and nothing else should change")
    add("o-page-up-down", "sess-keys", _at(4, 17, 5, 9),
        "I only want the global one, and I want it bound to the control page "
        "up and page down keys")

    add("o-onnx-brief", "sess-onnx-brief", _at(4, 12, 9, 0),
        _task_notification("the agent was told to check whether onnx still "
                           "exports"))
    add("o-onnx-resume-1", "sess-onnx-resume-1", _at(4, 13, 9, 0),
        _resume_banner("the onnx export had been left half finished"))
    add("o-onnx-resume-2", "sess-onnx-resume-2", _at(4, 14, 9, 0),
        _resume_banner("onnx was the thing eating the disk"))
    add("o-onnx", "sess-onnx", _at(4, 14, 17, 44),
        "first, does it still export? if so, fix it so we do not get more "
        "onnx. then look through the disk for hogs")

    add("o-hook-qv-1", "sess-qv-hook-1", _at(7, 11, 9, 0),
        _hook_watch("the user asked about quadratic voting and liquid "
                    "funding"))
    add("o-hook-qv-2", "sess-qv-hook-2", _at(7, 11, 12, 0),
        _classifier("recollection of the quadratic voting conversation was "
                    "requested"))
    add("o-quadratic-voting", "sess-qv", _at(8, 2, 19, 27),
        "do you have any recollection of our talk on quadratic voting and "
        "liquid funding")

    # Three near-identical re-sends of one prompt, seconds apart: the
    # interrupt-and-resend artifact that makes a rank-1 assertion a four-way
    # tie unless the ordering is the timestamp. Here it is.
    add("o-block-bad-stuff", "sess-config", _at(4, 2, 12, 52, 36), _CONFIG_TURN)
    for i, sec in enumerate((42, 44, 45)):
        add(f"o-block-resend-{i}", "sess-config", _at(4, 2, 12, 52, sec),
            _CONFIG_TURN)

    # The two slogans under comparison. Only adjacency tells them apart, and
    # which one was chosen is the entire question the query asks.
    add("o-slogan-draft", "sess-slogan", _at(7, 6, 17, 20),
        "meet more, plan less. drop the reminder copy, and stop the nag in "
        "the hero while you are there")
    add("o-slogan-final", "sess-slogan", _at(7, 6, 17, 44),
        "wdyt? Meet more, nag less. the app finds the best time to meet, and "
        "your guests give availability in ten seconds")
    add("o-hook-slogan", "sess-slogan-hook", _at(7, 6, 18, 30),
        _hook_watch("the options were meet more, plan less and meet more, "
                    "nag less"))

    for i, ts in enumerate((_at(4, 19, 9, 0), _at(4, 19, 15, 0),
                            _at(4, 27, 9, 0), _at(5, 3, 9, 0))):
        add(f"o-hook-bp-{i}", f"sess-bp-hook-{i}", ts,
            _hook_watch("the user said not to force merge and to keep branch "
                        "protection on main"))
    add("o-branch-protection", "sess-bp", _at(5, 3, 22, 2),
        "no force merge, use the script for branch protection. a force merge "
        "by an admin is ok, but do not override branch protection for a push "
        "to main")

    # --- what the negatives have to be able to reach ----------------------
    # A negative is only worth anything if every token of it EXISTS. Otherwise
    # "0 results" proves the corpus is small, not that the conjunction is
    # selective. ``test_the_negatives_are_real_conjunctions`` enforces it.
    add("o-drift", "sess-infra", _at(5, 10, 9, 0), _DRIFT_BODY)
    add("o-infra-notes", "sess-infra", _at(5, 10, 10, 0),
        "the deploy is stuck behind a lock; check the state of the queue "
        "before you retry")
    add("o-infra-k8s", "sess-infra-2", _at(5, 11, 9, 0),
        "read up on kubernetes, and on helm, and on what a chart is, then "
        "tell me whether a rollback is cheap")
    add("o-infra-pay", "sess-infra-2", _at(5, 11, 10, 0),
        "stripe is the payment processor; a webhook arrives per event and "
        "each one carries a signature")

    # Sessions are derived from the turns they own, so ``first_ts``/``last_ts``
    # cannot drift away from the rows and quietly reorder the card page.
    spans: dict[str, list[datetime]] = {}
    for row in rows:
        spans.setdefault(row.session_id, []).append(row.ts)
    sessions = [
        SessionRow(
            session_id=sid, root_session_id=sid, parent_session_id=None,
            kind="session", agent_id=None, agent_type=None,
            spawned_by_tool_use_id=None,
            cwd=("/home/dev/.claude/continue-nudge" if "hook" in sid
                 else "/home/dev/project"),
            git_branch="main", first_ts=min(stamps), last_ts=max(stamps),
            jsonl_path=f"/tmp/{sid}.jsonl",
        )
        for sid, stamps in spans.items()
    ]
    store.upsert_entities([*sessions, *rows])
    return store


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """Built once: the giant bodies cost a third of a megabyte of FTS5
    indexing and nothing in this file writes to the store."""
    return _build_corpus(tmp_path_factory.mktemp("legibility") / "cache.db")


def _query(store, dsl, **kw):
    kw.setdefault("fields", "summary")
    kw.setdefault("drilldown", True)
    kw.setdefault("page_size", 50)
    result = mcp.aggregator_query(dsl=dsl, _store=store, **kw)
    assert result["ok"] is True, result
    return result


def _ids(result) -> list[str]:
    return [r["obs_id"] for r in result["records"]]


def _rank(result, obs_id: str) -> int | None:
    ids = _ids(result)
    return ids.index(obs_id) + 1 if obs_id in ids else None


def _inner(content: str) -> str:
    """The payload inside the ``<ExternalContent>`` wrapper."""
    return content.partition(">\n")[2].rpartition("\n</ExternalContent>")[0]


def _payload(result) -> int:
    return len(json.dumps(result, default=str, ensure_ascii=False))


# ---------------------------------------------------------------------------
# The frozen table.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pair:
    """One frozen ``(query -> known-correct row)`` pair.

    ``criterion`` is what a failure here should be blamed on FIRST. It is not
    decorative: six of the mined pairs move by zero under criterion D and their
    remedy is criterion C's, and a suite that cannot say which is which credits
    a change with a gain it did not produce.
    """

    id: str
    dsl: str
    obs_id: str
    rank: int
    total: int
    criterion: str


#: Every pair whose answer must come back inside the first five rows, carrying
#: a snippet a caller can read without a second call. Fourteen pairs; criterion
#: E asks for at least ten.
TOP5 = (
    Pair("pos-clickable-links",
         'source:sessions type:user "clickable links"',
         "o-report-answer", 1, 4, "D+A"),
    Pair("pos-survival-corpuses",
         'source:sessions type:user "survival corpuses"',
         "o-survival", 1, 3, "A"),
    Pair("pos-implement-rag",
         'source:sessions type:user "implement RAG over it"',
         "o-implement-rag", 1, 2, "A"),
    Pair("pos-double-booking",
         'source:sessions type:user "double booking"',
         "o-double-booking", 1, 2, "A"),
    Pair("pos-page-up-down",
         'source:sessions type:user "page up and page down"',
         "o-page-up-down", 1, 1, "D"),
    Pair("pos-onnx",
         "source:sessions type:user onnx",
         "o-onnx", 4, 4, "C"),
    Pair("pos-quadratic-voting",
         'source:sessions type:user "quadratic voting"',
         "o-quadratic-voting", 3, 3, "C"),
    Pair("pos-block-bad-stuff",
         'source:sessions type:user "block bad stuff"',
         "o-block-bad-stuff", 1, 4, "A"),
    Pair("pol-gymma-ben",
         'source:sessions type:user "gymma ben"',
         "o-gymma-ben", 2, 5, "C"),
    Pair("pol-unwedged",
         "source:sessions type:user unwedged",
         "o-unwedged", 3, 4, "C"),
    Pair("fn-low-usage-cap",
         'source:sessions type:user "low usage cap"',
         "o-usage-cap", 1, 2, "D"),
    Pair("fn-meet-more-nag-less",
         'source:sessions type:user "Meet more, nag less"',
         "o-slogan-final", 1, 2, "D"),
    Pair("fn-branch-protection",
         'source:sessions type:user "branch protection" force merge',
         "o-branch-protection", 5, 5, "D+C"),
    Pair("fn-permission-prompts",
         'source:sessions type:user "too many permission prompts"',
         "o-permission-prompts", 1, 1, "D+B"),
)


@pytest.mark.parametrize("pair", TOP5, ids=lambda p: p.id)
def test_the_answer_is_in_the_top_five_and_legible(corpus, pair):
    """CRITERION E, THE HEADLINE ASSERTION: rank in top 5 AND a non-empty
    snippet, for every frozen pair.

    The second half is the one that was failing on every single row before
    criterion A: the retrieval was already right and the response format threw
    the answer away. The first half is a statement about what the FILTER
    admits — see the module docstring — not about scoring.
    """
    result = _query(corpus, pair.dsl)
    assert result["total"] == pair.total, (pair.id, _ids(result))
    assert _rank(result, pair.obs_id) == pair.rank, (pair.id, _ids(result))
    assert pair.rank <= 5, pair.id

    row = result["records"][pair.rank - 1]
    assert row["obs_id"] == pair.obs_id
    # A: legible.
    assert row["content"], f"{pair.id}: summary mode returned an empty content"
    assert row["content"].startswith('<ExternalContent source="'), row["content"]
    assert row["content"].endswith("</ExternalContent>"), row["content"]
    assert mcp._SNIPPET_MARK_OPEN in row["content"], row["content"]
    # A: bounded. The budget is per row, not per page.
    assert len(_inner(row["content"])) <= mcp._SNIPPET_CHARS * 2, row["content"]
    # B: an untouched row still states the fact rather than omitting the key.
    assert row["truncated"] is False, row
    assert row["content_length"] == len(row["content"])
    # C: authorship rides on the row, always.
    assert "provenance" in row, row


@pytest.mark.parametrize("pair", TOP5, ids=lambda p: p.id)
def test_the_answer_is_the_users_own_turn(corpus, pair):
    """Every frozen answer is human-authored. Stated separately from the rank
    so a classifier regression reads as a classifier regression."""
    result = _query(corpus, pair.dsl)
    row = result["records"][pair.rank - 1]
    assert row["provenance"] == HUMAN, (pair.id, row["provenance"])


def test_the_frozen_set_is_large_enough_and_covers_every_shape():
    """Criterion E's own floor, pinned so a future trim is visible."""
    assert len(TOP5) >= 10
    assert len(ABSTENTIONS) >= 1
    assert {p.id for p in TOP5} >= {"pos-clickable-links"}
    # The blind spot that let the phrase defect ship: a suite with no quoted
    # query cannot see it.
    quoted = [p for p in TOP5 if '"' in p.dsl]
    assert len(quoted) >= 8, [p.id for p in quoted]


def test_every_frozen_pair_names_the_criterion_to_blame_first():
    """THE ATTRIBUTION COLUMN IS NOT DECORATIVE, and this is what keeps it from
    rotting into a free-text field nobody maintains.

    Six of the mined pairs move by exactly zero under criterion D — every one
    is a single conjunct or unquoted, so there is no phrase to restore — and
    their remedy is criterion C's. A suite that cannot say which is which
    credits a change with a gain it did not produce, which is the failure the
    mission's own sequencing rule was written to prevent.
    """
    known = {"A", "B", "C", "D", "—"}
    for pair in (*TOP5, *ABSTENTIONS):
        parts = pair.criterion.split("+")
        assert parts and all(p in known for p in parts), pair
    named = {p for pair in (*TOP5, *ABSTENTIONS)
             for p in pair.criterion.split("+")}
    assert {"A", "B", "C", "D"} <= named, named


# ---------------------------------------------------------------------------
# Abstention: nothing answers it, and the page says so.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Abstention:
    id: str
    dsl: str
    #: True when the query has two or more conjuncts, so the empty page owes
    #: the caller an explanation of what was ANDed.
    conjunction_notice: bool
    criterion: str


ABSTENTIONS = (
    Abstention("neg-helm-rollback",
               'source:sessions type:user "kubernetes helm chart rollback"',
               False, "D"),
    Abstention("neg-cassandra-compaction",
               'source:sessions type:user "cassandra compaction strategy"',
               False, "—"),
    Abstention("neg-terraform-state-lock",
               'source:sessions type:user "terraform state lock"',
               False, "D"),
    Abstention("neg-stripe-webhook",
               'source:sessions type:user "stripe webhook signature"',
               False, "D+B"),
)

#: The two multi-conjunct abstentions v7 RESCUES. Pre-relaxation they were
#: the mission's second-best outcome — nothing, with an explanation — and
#: they are kept here, not deleted, because the ranking's FIRST outcome is
#: returning the right turn: under OR relaxation both now surface the mined
#: pair's answer row, flagged so the looser match cannot pose as exact. The
#: four single-phrase negatives above are untouched by relaxation on
#: purpose — a quoted phrase has one conjunct, so there is no OR tier, and
#: the prefix tier preserves adjacency (``"terraform state lock"*`` still
#: requires the phrase) — which is what keeps them honest abstentions.
RESCUED = (
    ("fn-pr-link-status-report",
     'source:sessions type:user "PR link" "status report"',
     "o-report-answer"),
    ("fn-spec-long-phrase",
     "source:sessions type:user hand back control only when done "
     "clickable PR link executive summary",
     "o-report-answer"),
)


@pytest.mark.parametrize("case", ABSTENTIONS, ids=lambda c: c.id)
def test_an_unanswerable_query_returns_nothing_and_says_so(corpus, case):
    """MISSION RANKING OF OUTCOMES: the right turn is best, nothing WITH an
    explanation is acceptable, and a silent pile of irrelevant rows is the
    worst. These are the second outcome, made explicit — and v7's relaxation
    must NOT dilute them: each is a single quoted phrase, so no OR tier
    exists and the phrase-prefix tier still demands the adjacency the corpus
    does not contain."""
    result = _query(corpus, case.dsl)
    assert result["total"] == 0, (case.id, _ids(result))
    assert result["records"] == []
    assert "lexical_relaxation" not in result, case.id
    notice = result.get("notice") or ""
    assert "abstention" in notice, (case.id, notice)


@pytest.mark.parametrize("case", RESCUED, ids=lambda c: c[0])
def test_a_formerly_abstaining_conjunction_is_rescued_and_flagged(corpus, case):
    """The mission's FIRST outcome, reached by relaxation: the AND-dead gist
    query now returns the answer row — marked, never posing as exact."""
    case_id, dsl, answer_obs = case
    result = _query(corpus, dsl)
    assert result["total"] > 0, case_id
    assert answer_obs in _ids(result), (case_id, _ids(result))
    assert result["lexical_relaxation"] == "or", case_id
    assert "RELAXATION" in result["notice"], (case_id, result["notice"])


def test_the_negatives_are_real_conjunctions_not_an_empty_corpus(corpus):
    """A NEGATIVE THAT NOTHING COULD MATCH PROVES NOTHING. Each token of the
    three infrastructure negatives exists in this corpus on its own; only the
    adjacency is absent. That is what makes them over-eagerness canaries — the
    terraform one returned 1,845 rows against the real corpus before criterion
    D, for a phrase it does not contain anywhere.

    ``cassandra`` is the deliberate exception and is asserted as such: it is
    the pure-absence negative, present so the set holds one of each.
    """
    for token in ("terraform", "state", "lock", "kubernetes", "helm", "chart",
                  "rollback", "stripe", "webhook", "signature"):
        result = _query(corpus, f"source:sessions type:user {token}")
        assert result["total"] > 0, f"{token} is absent, so its negative is free"
    for absent in ("cassandra", "compaction"):
        assert _query(corpus, f"source:sessions type:user {absent}")["total"] == 0


# ---------------------------------------------------------------------------
# Criterion D — a quoted phrase is ONE conjunct.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phrase:
    id: str
    quoted: str
    unquoted: str
    obs_id: str | None
    quoted_total: int
    quoted_rank: int | None
    unquoted_total: int
    unquoted_rank: int | None


#: The same query with and without the caller's quotes. ``fts5_match_query``
#: used to discard them, so the left column is what the caller asked for and
#: the right column is what the index was given. Eight pairs, because none of
#: the 86 queries in the repo's existing frozen set carries a quote at all.
PHRASES = (
    Phrase("pos-clickable-links",
           'source:sessions type:user "clickable links"',
           "source:sessions type:user clickable links",
           "o-report-answer", 4, 1, 6, 3),
    Phrase("pos-survival-corpuses",
           'source:sessions type:user "survival corpuses"',
           "source:sessions type:user survival corpuses",
           "o-survival", 3, 1, 4, 2),
    Phrase("pos-page-up-down",
           'source:sessions type:user "page up and page down"',
           "source:sessions type:user page up and page down",
           "o-page-up-down", 1, 1, 2, 2),
    Phrase("fn-low-usage-cap",
           'source:sessions type:user "low usage cap"',
           "source:sessions type:user low usage cap",
           "o-usage-cap", 2, 1, 3, 2),
    Phrase("fn-meet-more-nag-less",
           'source:sessions type:user "Meet more, nag less"',
           "source:sessions type:user Meet more, nag less",
           "o-slogan-final", 2, 1, 3, 2),
    Phrase("fn-permission-prompts",
           'source:sessions type:user "too many permission prompts"',
           "source:sessions type:user too many permission prompts",
           "o-permission-prompts", 1, 1, 2, 2),
    Phrase("neg-terraform-state-lock",
           'source:sessions type:user "terraform state lock"',
           "source:sessions type:user terraform state lock",
           None, 0, None, 2, None),
    Phrase("neg-stripe-webhook",
           'source:sessions type:user "stripe webhook signature"',
           "source:sessions type:user stripe webhook signature",
           None, 0, None, 2, None),
)


@pytest.mark.parametrize("case", PHRASES, ids=lambda c: c.id)
def test_the_quotes_the_caller_typed_survive_to_the_index(corpus, case):
    """THE DEFECT CRITERION D ACTUALLY FOUND, pinned from both sides.

    Dropping the quotes must widen the result set — that is what proves the
    quotes narrowed it — and the widened set is the one the mission calls the
    worst outcome: more rows, the answer further down or gone, and nothing
    saying why.
    """
    quoted = _query(corpus, case.quoted)
    unquoted = _query(corpus, case.unquoted)
    assert quoted["total"] == case.quoted_total, _ids(quoted)
    assert unquoted["total"] == case.unquoted_total, _ids(unquoted)
    assert unquoted["total"] > quoted["total"], (
        f"{case.id}: the quotes bought nothing, so nothing enforces them"
    )
    if case.obs_id:
        assert _rank(quoted, case.obs_id) == case.quoted_rank, _ids(quoted)
        assert _rank(unquoted, case.obs_id) == case.unquoted_rank, _ids(unquoted)


def test_an_adjacency_phrase_separates_two_drafts_of_one_slogan(corpus):
    """WHY ADJACENCY IS NOT A DETAIL. The query asks which of two slogans was
    chosen. Both drafts contain every word of the other, so the token rewrite
    could not tell them apart and returned the REJECTED variant first."""
    quoted = _query(corpus, 'source:sessions type:user "Meet more, nag less"')
    assert "o-slogan-draft" not in _ids(quoted)
    unquoted = _query(corpus, "source:sessions type:user Meet more nag less")
    assert _ids(unquoted)[0] == "o-slogan-draft"


def test_scope_session_is_the_widening_and_it_is_asked_for_by_name(corpus):
    """The default is one observation, and it always was — ``obs_fts`` holds
    one row per observation, so ``MATCH`` has never spanned turns. What was
    missing was a way to ASK for the wider unit, and a page that says the
    wider unit would have found something.

    v7 note: the AND-dead default-scope page is no longer empty — relaxation
    fills it with ANY-phrase rows, FLAGGED. ``scope:session`` is still the
    named way to ask the exact-conjunction question, and its answer is still
    exact: no relaxation marker, only the rows carrying every conjunct.
    """
    dsl = 'source:sessions type:user "PR link" "status report"'
    relaxed = _query(corpus, dsl)
    assert relaxed["lexical_relaxation"] == "or"
    widened = _query(corpus, "source:sessions type:user scope:session "
                             '"PR link" "status report"')
    assert sorted(_ids(widened)) == ["o-split-1", "o-split-2"]
    assert "lexical_relaxation" not in widened


def test_the_rescued_page_still_names_the_widening_that_answers_exactly(corpus):
    """THE MOST USEFUL SENTENCE THIS PAGE CAN CARRY, v7 edition. Pre-v7 the
    AND-dead page was empty and the notice pointed at ``scope:session``; now
    relaxation fills the page with ANY-phrase rows, and the same pointer must
    survive on the rescued page — the looser rows are leads, and the
    exact-conjunction answer the caller actually asked for is still one
    ``scope:session`` away."""
    result = _query(corpus,
                    'source:sessions type:user "PR link" "status report"')
    assert result["total"] > 0
    assert result["lexical_relaxation"] == "or"
    notice = result["notice"]
    assert "RELAXATION" in notice, notice
    assert "scope:session" in notice, notice
    assert "1 session" in notice, notice
    assert "does not stem" not in notice, notice


# ---------------------------------------------------------------------------
# Criterion C — authorship on the row, and what it buys.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Precision:
    id: str
    dsl: str
    obs_id: str
    rank: int
    total: int
    machine: int
    human_rank: int
    human_total: int


#: Six pairs where the top of the page is machine-authored and ``by:human`` is
#: the whole remedy. These are the ones criterion D moved by ZERO — every one
#: is a single conjunct or unquoted, so there is no phrase to restore — and
#: crediting D with them would be crediting a change with a gain it did not
#: produce.
PRECISION = (
    Precision("pos-onnx", "source:sessions type:user onnx",
              "o-onnx", 4, 4, 3, 1, 1),
    Precision("pos-quadratic-voting",
              'source:sessions type:user "quadratic voting"',
              "o-quadratic-voting", 3, 3, 2, 1, 1),
    Precision("pol-gymma-ben", 'source:sessions type:user "gymma ben"',
              "o-gymma-ben", 2, 5, 4, 1, 1),
    Precision("pol-unwedged", "source:sessions type:user unwedged",
              "o-unwedged", 3, 4, 2, 2, 2),
    Precision("fn-branch-protection",
              'source:sessions type:user "branch protection" force merge',
              "o-branch-protection", 5, 5, 4, 1, 1),
    Precision("fn-hyphenated-identifier",
              "source:sessions type:user jhanas-maxxing",
              "o-jhanas-maxxing", 7, 7, 6, 1, 1),
)


@pytest.mark.parametrize("case", PRECISION, ids=lambda c: c.id)
def test_by_human_is_what_lifts_the_answer_off_the_pollution(corpus, case):
    """Criterion C's precision claim, measured per fixture rather than
    asserted in aggregate."""
    default = _query(corpus, case.dsl)
    assert default["total"] == case.total, _ids(default)
    assert _rank(default, case.obs_id) == case.rank, _ids(default)
    machine = [r for r in default["records"]
               if r["provenance"] in MACHINE_VALUES]
    assert len(machine) == case.machine, [r["provenance"]
                                          for r in default["records"]]

    human = _query(corpus, case.dsl.replace("type:user ", "type:user by:human "))
    assert human["total"] == case.human_total, _ids(human)
    assert _rank(human, case.obs_id) == case.human_rank, _ids(human)


@pytest.mark.parametrize("case", PRECISION, ids=lambda c: c.id)
def test_the_notice_names_the_machine_count_and_the_filter(corpus, case):
    """A silent narrowing is "plausible but wrong", which this codebase already
    refuses for page tokens. So the majority stays on the page and gets NAMED —
    the benefit is delivered loudly rather than behind the caller's back."""
    result = _query(corpus, case.dsl)
    notice = result["notice"]
    assert f"{case.machine} of {case.total}" in notice, notice
    assert "by:human" in notice, notice
    assert "not who wrote it" in notice, notice


def test_a_hyphenated_identifier_neither_crashes_nor_is_findable_by_luck(corpus):
    """Two halves, and only the first was ever fixed. ``b4eab9b`` stopped
    ``jhanas-maxxing`` raising ``no such column: maxxing``; the answer still
    lands at rank 7, under six machine relays, and ``by:human`` is what
    recovers it. The repo's existing golden set pins the crash and nothing
    pinned the rank."""
    result = _query(corpus, "source:sessions type:user jhanas-maxxing")
    assert result["ok"] is True
    assert _rank(result, "o-jhanas-maxxing") == 7, _ids(result)


# ---------------------------------------------------------------------------
# Result sets that are 100% machine-authored.
# ---------------------------------------------------------------------------


MACHINE_ONLY = (
    ("pol-fan-blasts", 'source:sessions type:user "the fan blasts"', 1),
    ("pol-embed-order",
     'source:sessions type:user dropbox blog llm "claude code"', 3),
    ("fn-natural-language-question",
     "source:sessions type:user when did I first ask for clickable links in "
     "status reports", 2),
)


@pytest.mark.parametrize("case", MACHINE_ONLY, ids=lambda c: c[0])
def test_a_wholly_machine_page_is_labelled_and_by_human_abstains(corpus, case):
    """THE WORST SHAPE IN THE MINED SET, and the one that names the whole
    incident: ``when did I first ask for clickable links in status reports`` is
    verbatim the question the mission opens with, and the tool answered it with
    hook prompts and empty strings.

    After criterion C the rows still come back — nothing is narrowed behind the
    caller's back — but every one of them says a machine wrote it, the notice
    says how many, and ``by:human`` turns the pile into an honest abstention.
    """
    fixture_id, dsl, total = case
    result = _query(corpus, dsl)
    assert result["total"] == total, _ids(result)
    assert all(r["provenance"] in MACHINE_VALUES for r in result["records"]), [
        (r["obs_id"], r["provenance"]) for r in result["records"]
    ]
    assert f"{total} of {total}" in result["notice"], result["notice"]
    assert "by:human" in result["notice"]

    human = _query(corpus, dsl.replace("type:user ", "type:user by:human "))
    assert human["total"] == 0, _ids(human)
    assert "abstention" in (human.get("notice") or ""), human.get("notice")


def test_the_marker_backstop_does_not_cover_a_headless_brief(corpus):
    """A MEASURED GAP, PINNED SO IT IS NOT REDISCOVERED AS A BUG.

    The two 100%-machine fixtures above are headless labelling briefs, and
    ``aggregator.core.provenance``'s frozen marker set does not contain their
    preamble. They are classified from the JSONL structure — ``promptSource
    ='sdk'`` — which is why this corpus supplies that structure rather than
    relying on the body.

    Where the structure is absent, which is old sessions, the same body falls
    to the ``human`` residual. That is the design's stated degradation
    direction and it is the safe one (under-claiming machine, never
    mislabelling a human), but it means ``by:human`` does NOT drop these rows
    on a pre-2.1.20x session. Measured on the live cache: 276 rows carrying
    this preamble, plus 1,405 carrying "This session is ending" and 606
    carrying "ASSISTANT:", all still filed as human.
    """
    body = _research_label("USER: whatever the excerpt happened to say")
    assert classify("user", body) == HUMAN
    assert classify("user", body, structure=_SDK) == HOOK


# ---------------------------------------------------------------------------
# Criterion A — the snippet, under the pressure a 353 KB body applies.
# ---------------------------------------------------------------------------


def test_the_snippet_is_centred_on_the_hit_inside_a_third_of_a_megabyte(corpus):
    """The reachable words sit in the MIDDLE of a 353 000-character body. A
    head slice — which is what ``o.body[:120]`` gives the subject — shows the
    reader none of them."""
    result = _query(corpus, "source:sessions type:user permission")
    row = next(r for r in result["records"] if r["obs_id"] == "o-skill-llm-apps")
    inner = _inner(row["content"])
    assert inner.startswith("…"), inner
    assert "[[permission]]" in inner, inner
    assert len(inner) <= mcp._SNIPPET_CHARS * 2, len(inner)
    # ...and it does not open on a word fragment.
    assert inner.lstrip("…").split(" ", 1)[0] in _SKILL_BODY.split()


def test_a_session_card_shows_what_matched_not_its_own_opening_turn(corpus):
    """Criterion A's other half. ``matching_observations: N`` advertises that
    something matched without ever showing what, and the card's ``subject`` is
    the session's FIRST user turn, which is usually about something else. Here
    the answer session opens on a branch name and matches on its second turn.
    """
    result = mcp.aggregator_query(
        dsl='source:sessions "clickable links"', fields="summary",
        _store=corpus,
    )
    assert result["ok"] is True, result
    card = next(r for r in result["records"]
                if r["stable_id"] == "sess-report-style")
    assert card["subject"].startswith("start a fresh branch"), card["subject"]
    assert card["matching_observations"] == 1, card
    assert "[[clickable]]" in card["content"], card["content"]
    assert card["content"].startswith('<ExternalContent source="')
    for other in result["records"]:
        assert other["content"], other


def test_the_card_page_is_headed_by_hook_sessions_whose_subject_is_a_prompt(
    corpus,
):
    """The single best one-shot demonstration of A and C together: the answer
    is in the result set and completely invisible. Cards are ordered
    ``last_ts DESC``, so the three machine sessions sit above it, and each of
    their subjects is the classifier prompt itself."""
    result = mcp.aggregator_query(
        dsl='source:sessions "clickable links"', fields="summary",
        _store=corpus,
    )
    ids = [r["stable_id"] for r in result["records"]]
    assert ids.index("sess-report-style") == 3, ids
    for machine in ids[:3]:
        card = next(r for r in result["records"] if r["stable_id"] == machine)
        assert card["subject"].startswith("You are "), card["subject"]


# ---------------------------------------------------------------------------
# Criterion B — the payload ceiling, on the fixture that justified it.
# ---------------------------------------------------------------------------


def test_the_two_display_budgets_are_pinned_at_their_measured_numbers():
    """EVERY BOUND IN THIS FILE IS WRITTEN AGAINST THESE TWO CONSTANTS, so
    raising either would loosen the whole suite in silence — which is exactly
    how a governor stops governing. Both were derived rather than picked, and
    the derivation belongs beside the number.

    ``_SNIPPET_CHARS`` — at ~3.4 characters per token, measured on this
    corpus's own JSON payloads, 200 characters is ~60 tokens per row, so a
    20-row page costs ~1 200 tokens of body: readable inline with the task
    still in hand.

    ``_MAX_RESPONSE_CHARS`` — the client's 25 000-token output cap times ~2.5
    characters per token for UUID-dense JSON is a 62 500-character break-even,
    less 20% headroom.
    """
    assert mcp._SNIPPET_CHARS == 200
    assert mcp._MAX_RESPONSE_CHARS == 50_000


def test_a_page_of_oversized_bodies_is_bounded_and_every_cut_is_declared(corpus):
    """``page_size`` bounds ROWS. Two rows here are 353 000 and 76 000
    characters, so a page of two is 429 KB before the envelope — which is how
    ``page_size=8`` produced a 282 110-character spill in the field report.

    Both are cut, both declare it, and both report the length of the body that
    EXISTS rather than of what came back. Truncation is the opposite call from
    the one this module makes for page tokens, where it refuses rather than
    truncates; the declaration is the only thing that keeps it honest.
    """
    result = mcp.aggregator_query(
        dsl="source:sessions type:user terraform", fields="full",
        drilldown=True, page_size=8, _store=corpus,
    )
    assert result["ok"] is True, result
    assert sorted(_ids(result)) == ["o-drift", "o-skill-llm-apps"]
    assert _payload(result) <= mcp._MAX_RESPONSE_CHARS, _payload(result)

    lengths = {r["obs_id"]: r["content_length"] for r in result["records"]}
    assert lengths["o-skill-llm-apps"] >= len(_SKILL_BODY)
    assert lengths["o-drift"] >= len(_DRIFT_BODY)
    for row in result["records"]:
        assert row["truncated"] is True, row["obs_id"]
        assert len(row["content"]) < row["content_length"]
        assert row["content"].count("<ExternalContent") == 1
        assert "</ExternalContent>" in row["content"]
        # The in-band marker sits OUTSIDE the closed wrapper: it is the tool's
        # own words about the row, and putting it inside would file the
        # server's statement as untrusted body text.
        assert row["content"].index(mcp._CONTENT_TRUNCATION_MARKER) > row[
            "content"
        ].index("</ExternalContent>")

    # One giant row must not starve the other: fair-share gives every row what
    # it asks for until the budget binds, and only then splits the remainder.
    assert min(len(r["content"]) for r in result["records"]) > 1000, [
        (r["obs_id"], len(r["content"])) for r in result["records"]
    ]


def test_a_page_carrying_the_giant_body_stays_under_the_ceiling(corpus):
    """``fn-permission-prompts`` in one call: a 388-character answer sitting
    behind a 342 459-character skill body, so ``page_size=2`` was already a
    343 KB payload and ``page_size`` was never the governor.

    The cut is declared, on the cut row and the whole one alike, and
    ``content_length`` is the length of the body that EXISTS rather than of
    what came back — otherwise it says nothing ``len()`` could not.
    """
    result = mcp.aggregator_query(
        dsl="source:sessions type:user too many permission prompts",
        fields="full", drilldown=True, page_size=8, _store=corpus,
    )
    assert result["ok"] is True, result
    assert len(result["records"]) == 2, _ids(result)
    assert _payload(result) <= mcp._MAX_RESPONSE_CHARS, _payload(result)

    giant = next(r for r in result["records"]
                 if r["obs_id"] == "o-skill-llm-apps")
    assert giant["truncated"] is True
    assert len(giant["content"]) < giant["content_length"]
    assert giant["content_length"] >= len(_SKILL_BODY), giant["content_length"]
    assert mcp._CONTENT_TRUNCATION_MARKER in giant["content"]
    # Truncation must not cut the untrusted-content boundary open.
    assert giant["content"].count("<ExternalContent") == 1
    assert "</ExternalContent>" in giant["content"]

    answer = next(r for r in result["records"]
                  if r["obs_id"] == "o-permission-prompts")
    assert answer["truncated"] is False
    assert answer["content_length"] == len(answer["content"])

    notice = result["notice"]
    assert "truncat" in notice.lower(), notice
    assert "content_length" in notice, notice


def test_the_phrase_fix_is_what_keeps_the_payload_bomb_off_the_page(corpus):
    """The two changes compose, and this is where it shows. ``neg-stripe-
    webhook``'s right answer is nothing, and before criterion D it returned two
    rows — one of them 922 982 characters — for a phrase the corpus does not
    contain. Criterion B would have truncated it; criterion D means it is never
    fetched."""
    quoted = _query(corpus, 'source:sessions type:user "stripe webhook signature"')
    assert quoted["total"] == 0
    unquoted = mcp.aggregator_query(
        dsl="source:sessions type:user stripe webhook signature",
        fields="full", drilldown=True, page_size=8, _store=corpus,
    )
    assert "o-skill-llm-apps" in _ids(unquoted)
    assert _payload(unquoted) <= mcp._MAX_RESPONSE_CHARS


# ---------------------------------------------------------------------------
# The two acceptance tests, named as such.
# ---------------------------------------------------------------------------


def test_acceptance_1_one_call_summary_mode_no_spill(corpus):
    """MISSION ACCEPTANCE TEST 1, verbatim:

        dsl='source:sessions type:user "clickable links"' fields=summary
        drilldown=true

    must return rows carrying a legible snippet, oldest-first orderable,
    readable inline. It returned five empty strings; the correct row was
    ranked first and unreadable, and the caller spent six calls, two
    output-limit spills and ~40k tokens getting nothing.
    """
    result = mcp.aggregator_query(
        dsl='source:sessions type:user "clickable links"',
        fields="summary", drilldown=True, _store=corpus,
    )
    assert result["ok"] is True, result

    answer = result["records"][0]
    assert answer["obs_id"] == "o-report-answer"
    assert "[[clickable]]" in answer["content"], answer["content"]
    assert "[[links]]" in answer["content"], answer["content"]
    assert answer["provenance"] == HUMAN

    # Oldest-first orderable: every row still carries its timestamp.
    assert all(r["ts"] for r in result["records"])
    stamps = [r["ts"] for r in result["records"]]
    assert stamps == sorted(stamps)

    # No spill, and nothing was cut to achieve that.
    assert _payload(result) <= mcp._MAX_RESPONSE_CHARS, _payload(result)
    assert all(r["truncated"] is False for r in result["records"])

    # The notice no longer sends the caller to fields=full, which is the
    # instruction that produced the 282 110-character spill.
    assert "Re-call with fields=full to include observation bodies" not in (
        result["notice"]
    )


def test_acceptance_2_the_caller_who_does_not_know_the_exact_phrase(corpus):
    """MISSION ACCEPTANCE TEST 2, verbatim: a caller who does not know the
    exact phrase tries ``"PR link" "status report"``. It must return the
    answer turn, OR return nothing WITH a notice explaining same-observation
    conjunction and suggesting single-phrase queries. **Silently returning 36
    irrelevant sessions is the worst of the three outcomes.**

    v7 reaches the FIRST outcome. Strict AND still matches nothing — the
    answer says "link prs", not "PR link", and a quoted run means ADJACENCY;
    stemming folds ``prs`` into ``pr`` but cannot reorder a phrase — so the
    OR tier fires and the answer turn comes back ON the page, flagged
    ``lexical_relaxation: "or"`` so the loose match cannot pose as exact.
    The loose (unquoted) four-word query is also transformed: porter carries
    it to the human answer row, where pre-v7 it drowned in hook prompts
    quoting a transcript.
    """
    result = mcp.aggregator_query(
        dsl='source:sessions type:user "PR link" "status report"',
        fields="summary", drilldown=True, _store=corpus,
    )
    assert result["ok"] is True, result
    assert result["total"] > 0
    assert "o-report-answer" in _ids(result), _ids(result)
    assert result["lexical_relaxation"] == "or"

    notice = result["notice"]
    assert "RELAXATION" in notice, notice
    assert "NOT exact" in notice, notice
    assert "scope:session" in notice, notice

    # The loose query, for contrast. Pre-v7 this page held ONLY machine rows
    # quoting a transcript — the silent worst outcome. Under porter stemming
    # "prs" and "PR" are one term, so the same four loose words now reach the
    # human answer row as well. The machine rows still ride along; the quoted
    # form above is still what asks the precise question.
    silent = _query(corpus, "source:sessions type:user PR link status report")
    assert silent["total"] > 0
    by_id = {r["obs_id"]: r["provenance"] for r in silent["records"]}
    assert by_id.get("o-report-answer") == "human", by_id
