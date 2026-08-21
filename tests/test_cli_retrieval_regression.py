"""``aggregator retrieval-regression`` — criterion A's missing surface.

The harness landed first, on purpose: it gates every other retrieval decision
on this branch. But ``retrieval_regression_command`` had no caller. A gate that
can only be invoked from a Python REPL is one nobody runs before a change and
nobody runs after it, which is the same as not having one — and the freeze/run
ORDER is where its entire value lives. Freeze before the change, run after; a
baseline taken afterwards has baselined the bug.

THE IMPORT DIRECTION IS PART OF THE DESIGN AND IS PINNED BELOW. ``cli`` imports
``evals``; ``evals`` imports nothing from ``cli``. Wiring is exactly the moment
that gets reversed by reflex — a helper in ``cli`` looks handy from the harness
— and reversing it would make the eval package unimportable from anywhere the
CLI is not, tests included.
"""

from __future__ import annotations

import pytest

from aggregator import cli


def test_the_command_exists_with_both_actions(tmp_data_home, capsys):
    """THE REPRO: the entry point had no argv that reached it."""
    with pytest.raises(SystemExit):
        cli.main(["retrieval-regression", "--help"])
    out = " ".join(capsys.readouterr().out.split())

    assert "freeze" in out and "run" in out, (
        f"--help names neither action; freeze-then-run IS the tool: {out!r}"
    )
    assert "--mode" in out
    assert "--drift-threshold" in out


def test_the_action_mode_and_threshold_reach_the_harness(
    tmp_data_home, monkeypatch
):
    """A flag that parses and then does not arrive is worse than no flag."""
    seen: dict = {}

    def fake(action="run", **kwargs):
        seen["action"] = action
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "retrieval_regression_command", fake)

    rc = cli.main(
        [
            "retrieval-regression",
            "freeze",
            "--mode",
            "hybrid",
            "--drift-threshold",
            "0.25",
        ]
    )

    assert rc == 0
    assert seen["action"] == "freeze"
    assert seen["mode"] == "hybrid"
    assert seen["drift_threshold"] == 0.25


def test_run_is_the_default_action(tmp_data_home, monkeypatch):
    """Typing the subcommand alone must not be a usage error: 'run' is what a
    person means by 'check retrieval', and freezing by accident would overwrite
    the baseline the check compares against."""
    seen: dict = {}
    monkeypatch.setattr(
        cli,
        "retrieval_regression_command",
        lambda action="run", **kw: (seen.update({"action": action, **kw}), 0)[1],
    )

    assert cli.main(["retrieval-regression"]) == 0
    assert seen["action"] == "run"
    assert seen["mode"] == "lexical"
    assert seen["drift_threshold"] is None


def test_the_harness_exit_code_is_the_process_exit_code(
    tmp_data_home, monkeypatch
):
    """1 is a regression, 2 is "the harness could not run". Collapsing either
    into 0 would make this green in a script that never looked at the text."""
    for code in (0, 1, 2):
        monkeypatch.setattr(
            cli, "retrieval_regression_command", lambda *a, code=code, **kw: code
        )
        assert cli.main(["retrieval-regression", "run"]) == code


def test_the_command_measures_this_machines_cache(tmp_data_home, monkeypatch):
    """The eval must run against the store the rest of the CLI opened, not
    against whatever default the harness would pick on its own."""
    seen: dict = {}
    monkeypatch.setattr(
        cli,
        "retrieval_regression_command",
        lambda *a, **kw: (seen.update(kw), 0)[1],
    )
    from aggregator.core.store import Store

    store = Store()
    store.migrate()

    cli.main(["retrieval-regression", "run"], _store=store)

    assert seen["db_path"] == store.db_path


def test_the_evals_package_does_not_import_the_cli(tmp_data_home):
    """One direction only. See this module's docstring."""
    import ast
    from pathlib import Path

    package = Path(cli.__file__).parent / "evals"
    offenders: list[str] = []
    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            offenders += [
                f"{path.name}:{node.lineno} imports {n}"
                for n in names
                if n == "aggregator.cli" or n.startswith("aggregator.cli.")
            ]
    assert offenders == [], (
        "aggregator.evals must not import aggregator.cli; the CLI depends on "
        f"the eval package, never the reverse: {offenders}"
    )
