"""``--once``'s documentation must describe the units that actually exist.

Round 3's L1. Two places claimed a wiring that no longer exists:

* ``--once``'s ``--help`` said it was "used by the human-triggered seed unit
  as a live check". The seed unit runs ``embed --seed-models`` — round 2 moved
  it precisely so that warming a model cache stops opening the database,
  taking the embed lock and advancing a watermark.
* ``_cmd_embed``'s docstring said ``--once`` was "what the systemd timer
  runs". The timer runs ``embed --catchup --source both``, and the nix module
  explains at length why: at batch-size 500 against 483k observations,
  ``--once`` per tick is ~970 ticks, about three weeks before the last row is
  even attempted.

Stale docs on an operational flag are not cosmetic. This is the text someone
reads at 2 a.m. deciding which command to run against a stalled index, and
both sentences pointed at a unit that would not do what they said.

These tests read ``nix/aggregator.nix`` rather than restating it, so the claim
is checked against the deployment instead of against another comment. The nix
module is owned elsewhere; this only reads it.
"""

import pathlib
import re

import pytest

from aggregator.cli import _cmd_embed, build_parser


@pytest.fixture(scope="module")
def nix_module():
    path = (
        pathlib.Path(__file__).resolve().parent.parent / "nix" / "aggregator.nix"
    )
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def once_help():
    for action in build_parser()._subparsers._group_actions[0].choices[
        "embed"
    ]._actions:
        if "--once" in (action.option_strings or []):
            return action.help
    raise AssertionError("embed has no --once flag")


# --- what the deployment actually runs --------------------------------------


def test_the_timer_runs_catchup_not_once(nix_module):
    runner = re.search(
        r'embedRunner = pkgs\.writeShellScript.*?\n  \'\';', nix_module, re.S
    )
    assert runner, "could not find embedRunner in nix/aggregator.nix"
    assert "embed --catchup --source both" in runner.group(0)
    assert "--once" not in runner.group(0)


def test_the_seed_unit_runs_seed_models_not_once(nix_module):
    seeder = re.search(
        r'embedSeeder = pkgs\.writeShellScript.*?\n  \'\';', nix_module, re.S
    )
    assert seeder, "could not find embedSeeder in nix/aggregator.nix"
    assert "embed --seed-models" in seeder.group(0)
    assert "--once" not in seeder.group(0)


def test_only_the_exec_lines_are_checked_for_once(nix_module):
    """``--once`` still appears in the module, and correctly so.

    Both units carry a comment explaining why they do NOT run it — the timer
    because ~970 ticks is three weeks to first coverage, the seeder because a
    weight download has no business writing to the corpus. Those comments are
    the record of the decision, so the two tests above scope themselves to the
    ``writeShellScript`` bodies rather than banning the string outright.
    """
    assert "--once" in nix_module
    assert nix_module.count("exec ${aggregatorBin} embed --once") == 0


# --- and what the CLI says about it -----------------------------------------


def test_the_once_help_does_not_claim_the_seed_unit_runs_it(once_help):
    """THE L1 REGRESSION, half one.

    The old text was "used by the human-triggered seed unit as a live check,
    not by the timer". Mentioning the seed unit is fine — and the new text
    does, to say what it runs INSTEAD. What may not survive is the claim that
    it runs this flag.
    """
    assert not re.search(r"used by[^.]*seed unit", once_help), once_help
    assert "seed unit runs --seed-models" in once_help, once_help


def test_the_once_help_says_plainly_that_nothing_deployed_runs_it(once_help):
    assert "no deployed unit runs this" in once_help, once_help
    assert "the timer runs --catchup" in once_help, once_help


def test_the_once_help_points_at_the_flag_that_is_deployed(once_help):
    """Naming --catchup is the useful half: it is what a stalled index needs."""
    assert "--catchup" in once_help, once_help


def test_the_command_docstring_does_not_call_once_the_timers_command():
    """THE L1 REGRESSION, half two."""
    doc = _cmd_embed.__doc__ or ""
    match = re.search(r"``--once``[^.]*what the systemd timer runs", doc)
    assert match is None, doc
    assert "--catchup" in doc
