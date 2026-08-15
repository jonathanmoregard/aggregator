"""Round-6 HIGH 1: the default no-op notifier counted as delivery.

Third round on one bug, and the same conceptual error each time — INFERRING
DELIVERY FROM THE ABSENCE OF AN ERROR. An uncovered TickTick project must be
reported once and then go quiet, and the receipt that makes it quiet was stamped:

  R4: whenever the report was EMITTED.                  -> stamp after notify
  R5: when ``notify`` returned without raising.         -> a separate barrier
  R6: ...and the DEFAULT notifier does nothing at all, which returns without
      raising. So every run with no notifier configured stamped receipts with no
      human channel in existence, and the single alert the mechanism exists to
      raise was suppressed by a record of its own delivery.

The shape changed rather than the check: a hook now RETURNS
``Delivery.DELIVERED`` or it did not deliver. ``None`` is the default answer and
is what a function that does nothing returns, so the no-op notifier cannot stamp
a receipt — there is no gate to forget, because there is no gate.

Stubs first: this file drives a real ``TickTickSource`` and must never reach the
live API, the developer's credential, or the real baseline.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from aggregator import cli
from aggregator.imports.port import Delivery, WriteCounts
from aggregator.imports.runner import RunReport, _no_notification, run_imports
from aggregator.imports.ticktick import TickTickAdapter
from aggregator.sources import ticktick_api
from aggregator.sources.ticktick import TickTickSource

UNCOVERED = "never covered"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("test attempted a real network call")

    monkeypatch.setattr(ticktick_api, "_open", _forbidden)


@pytest.fixture(autouse=True)
def _no_real_credentials(monkeypatch, tmp_path):
    monkeypatch.setattr(ticktick_api, "DEFAULT_ENV_FILE", str(tmp_path / "no-such-env"))
    for var in (
        "TICKTICK_ACCESS_TOKEN",
        "TICKTICK_TOKEN_EXPIRES_AT",
        "AGGREGATOR_TICKTICK_TOKEN",
        "AGGREGATOR_TICKTICK_TOKEN_FILE",
        "AGGREGATOR_TICKTICK_DIR",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _live_project_only(monkeypatch):
    """The poll covers ``live`` and serves ``t9`` from it. The baseline's ``t1``
    sits in ``gone``, which the poll never looked at — so t1 is retained, never
    inferred completed, and REPORTED once."""
    monkeypatch.setattr(
        ticktick_api,
        "poll_open_tasks",
        lambda token, errors=None: ticktick_api.OpenTaskPoll(
            [{"id": "t9", "title": "still open", "projectId": "live"}],
            complete=True,
            covered_project_ids=frozenset({"live"}),
        ),
    )


@pytest.fixture
def state_file(tmp_path):
    path = tmp_path / "open_tasks.json"
    path.write_text(
        json.dumps({"t1": {"task": {"id": "t1", "projectId": "gone"}, "last_seen": "x"}}),
        encoding="utf-8",
    )
    return path


class _Sink:
    def write(self, items) -> WriteCounts:
        return WriteCounts(added=len(items))


def _source(tmp_path, state_file) -> TickTickSource:
    return TickTickSource(
        backup_dir=tmp_path / "no-downloads",
        archive_dir=tmp_path / "no-archive",
        token="fake-token",
        state_file=state_file,
    )


def _run(tmp_path, state_file, **kwargs):
    """One whole run through the runner, on a fresh adapter — what a timer does.

    No ``notify=`` means the shipped default, which is the configuration this
    finding is about.
    """
    adapter = TickTickAdapter(source=_source(tmp_path, state_file))
    return asyncio.run(run_imports([adapter], _Sink(), **kwargs))


def _reported(report) -> list[str]:
    return [e for e in report.errors if UNCOVERED in e]


def _delivering(heard: list[str]):
    """A notifier that reached a human AND SAYS SO."""

    def notify(report) -> Delivery:
        heard.extend(report.errors)
        return Delivery.DELIVERED

    return notify


def _broken(report) -> None:
    raise FileNotFoundError("notify-send: command not found")


# -- HIGH 1 ---------------------------------------------------------------


def test_the_default_no_op_notifier_is_not_a_delivery_channel(tmp_path, state_file):
    """THE repro. ``run_imports``' default ``notify`` does nothing at all — it
    returns without raising, which was read as "the report reached a human", so
    the receipt was stamped and every later poll went quiet about a
    disappearance nobody was ever told about."""
    first = _run(tmp_path, state_file)
    assert _reported(first), first.errors

    second = _run(tmp_path, state_file)

    assert _reported(second), (
        "permanent silence: nothing was configured that could reach a human, "
        f"yet the receipt says one was told. run 2 reported {second.errors}"
    )


def test_a_hook_that_does_nothing_has_nothing_to_declare_delivery_with():
    """Why a fourth layer cannot be written the same way as the first three.

    Delivery is a value the hook returns, and the do-nothing hooks are annotated
    ``-> None``: there is no expression in either of them that could produce
    ``Delivery.DELIVERED``. The previous three fixes were all conditions at the
    call site, and a condition is a thing the next call site forgets."""
    report = RunReport()

    assert _no_notification(report) is None
    assert cli._silent_notification(report) is None
    assert None is not Delivery.DELIVERED


def test_a_truthy_return_value_is_not_a_declaration(tmp_path, state_file):
    """Identity, not truthiness. The obvious hook —
    ``lambda report: subprocess.run(...)`` — returns a CompletedProcess, which
    is truthy even when the notifier exited non-zero."""
    first = _run(tmp_path, state_file, notify=lambda report: "sent, honest")
    assert _reported(first)

    assert _reported(_run(tmp_path, state_file)), "a truthy value passed for delivery"


# -- the contract, end to end ---------------------------------------------


def test_the_delivery_contract_end_to_end(tmp_path, state_file):
    """ONE test for the whole rule, because this bug has now grown three layers
    by each fix pinning only the layer in front of it.

    No channel at all: reported on every run, forever, until there is one.
    A channel that raised: the same — nobody heard it, so it stands.
    A channel that delivered: reported exactly once, then quiet.

    Anything that breaks the rule breaks this, whatever route it takes to the
    receipt."""
    heard: list[str] = []

    # 1. Nothing configured. The disappearance is unreported and stays that way.
    assert _reported(_run(tmp_path, state_file)), "run 1 must raise it"
    assert _reported(_run(tmp_path, state_file)), "and keep raising it"

    # 2. A notifier that blew up reached nobody either (round 5, unregressed).
    raised = _run(tmp_path, state_file, notify=_broken)
    assert _reported(raised)
    assert any("notify hook failed" in e for e in raised.errors)
    assert _reported(_run(tmp_path, state_file)), "a raise is not a delivery"

    # 3. One that worked, and said so.
    delivered = _run(tmp_path, state_file, notify=_delivering(heard))
    assert _reported(delivered)
    assert [e for e in heard if UNCOVERED in e], f"the human was told: {heard}"

    # 4. Exactly once (round 4, unregressed): quiet from here, whatever channel
    #    the next run has. An alarm on every 30-minute tick is one an operator
    #    learns to ignore, which costs the next real failure its audience.
    assert _reported(_run(tmp_path, state_file)) == []
    assert _reported(_run(tmp_path, state_file, notify=_delivering(heard))) == []


def test_the_baseline_still_advances_with_no_channel_at_all(tmp_path, state_file):
    """The regression the loud-forever direction must not smuggle in. Only the
    REPORT waits for delivery; the write barrier may never wait for anything. A
    frozen baseline never learns about a task created after the freeze, so that
    task's later disappearance is invisible to every future poll — unbounded
    loss, versus one duplicated line."""
    _run(tmp_path, state_file)

    baseline = json.loads(state_file.read_text())
    assert "t9" in baseline, "the poll's own tasks must land whatever notify did"
    assert "t1" in baseline, "and the uncovered task is still never inferred done"
    assert ticktick_api.uncovered_mark(baseline["t1"]) is None
