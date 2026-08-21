"""Every path from user text to FTS5 ``MATCH``, enumerated MECHANICALLY.

WHY THIS FILE EXISTS AND WHY IT SCANS SOURCE RATHER THAN BEHAVIOUR. The
unescaped-``MATCH`` defect has now been found three separate times on this
branch, each time in a site a sibling of the one just fixed:

* ``b4eab9b`` whitelisted the six statements in ``aggregator.core.store``.
* ``29b2e8b`` fixed ``scripts/rag_rollout_smoke.py:_fts_ranked`` and pinned it
  with a test whose own docstring claimed it covered "the ONE helper standing
  between a raw query string and ``MATCH``".
* There were two. ``store_fts_all`` in the same file still bound raw user text,
  and its only caller swallowed the resulting ``OperationalError`` into an
  empty set — so six of seven realistic queries failed and nothing went red.

The pattern is not "somebody forgot". It is that every previous fix was scoped
by *reasoning* about which sites exist, and a reasoned enumeration of call
sites in a 5000-line module is worth nothing. So this file does not reason. It
parses every module under ``aggregator/`` and ``scripts/`` and finds every
string literal containing the SQL keyword ``MATCH``, then requires each one to
fall into exactly one of three declared buckets. A site nobody thought of is
not "missed" — it lands in no bucket and this file goes red, naming the file
and line.

THE PARTITION IS THE POINT. ``test_every_match_literal_is_classified`` is what
makes the enumeration complete rather than merely long: it asserts that the
three buckets cover the whole set, so a new MATCH shape (a differently-named
FTS table, an f-string that interpolates a table variable under another name,
a MATCH built by concatenation) cannot pass silently. It forces a decision.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCANNED_DIRS = ("aggregator", "scripts")

#: The SQL keyword, as a word. ``rematch``/``obj.match`` are not it.
_MATCH_WORD = re.compile(r"(?<![\w.])MATCH(?![\w])")

#: An FTS5 site: the left operand of ``MATCH`` is an FTS5 table. Both the
#: literal table names this project uses and the f-string form the smoke
#: script builds. A NEW table name matching ``\w+_fts`` is picked up
#: automatically; anything else falls through to the unclassified bucket.
_FTS_SITE = re.compile(r"(?:\b\w*_fts\b|\{table\})\s+MATCH\b")

#: A sqlite-vec KNN site: the left operand is the ``embedding`` column and the
#: bound parameter is a float32 blob, never user text.
_VEC_SITE = re.compile(r"\bembedding\s+MATCH\b")

#: Literals that mention MATCH but are not SQL at all. Frozen deliberately:
#: adding one is a decision, not a formality.
_NOT_SQL = {
    (
        "aggregator/core/store.py",
        "Store._fts_rows",
        "FTS5 MATCH failed for %r (rewritten to %r) — the lexical arm "
        "contributes nothing to this query",
    ),
}

#: The whitelist every FTS5 site's text must be rewritten through.
_SANITIZER = "fts5_match_query"


class _MatchLiteral:
    __slots__ = ("path", "lineno", "func", "text")

    def __init__(self, path: str, lineno: int, func: str, text: str) -> None:
        self.path = path
        self.lineno = lineno
        self.func = func
        self.text = text

    def __repr__(self) -> str:  # pragma: no cover - failure messages only
        return f"{self.path}:{self.lineno} in {self.func or '<module>'}"


def _literal_text(node: ast.AST) -> str | None:
    """The constant part of a string literal, f-string included.

    An f-string's interpolations are dropped, which is exactly right here:
    ``f"WHERE {table} MATCH ?"`` still shows ``MATCH ?`` and still shows the
    ``{table}`` placeholder, so the site is visible without evaluating it.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                parts.append("{" + ast.unparse(value.value) + "}")
        return "".join(parts)
    return None


def _docstring_ids(tree: ast.AST) -> set[int]:
    out: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            out.add(id(body[0].value))
    return out


class _Collector(ast.NodeVisitor):
    def __init__(self, path: str, docstrings: set[int]) -> None:
        self.path = path
        self.docstrings = docstrings
        self.stack: list[str] = []
        self.found: list[_MatchLiteral] = []
        #: qualified function name -> set of plain function names it calls
        self.calls: dict[str, set[str]] = {}

    def _scoped(self, node: ast.AST) -> None:
        self.stack.append(node.name)  # type: ignore[attr-defined]
        self.calls.setdefault(".".join(self.stack), set())
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scoped(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._scoped(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scoped(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name:
            for depth in range(len(self.stack), 0, -1):
                self.calls.setdefault(".".join(self.stack[:depth]), set()).add(name)
        self.generic_visit(node)

    def _literal(self, node: ast.AST) -> None:
        if id(node) in self.docstrings:
            return
        text = _literal_text(node)
        if text and _MATCH_WORD.search(text):
            self.found.append(
                _MatchLiteral(
                    self.path,
                    node.lineno,  # type: ignore[attr-defined]
                    ".".join(self.stack),
                    " ".join(text.split()),
                )
            )

    def visit_Constant(self, node: ast.Constant) -> None:
        self._literal(node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        self._literal(node)
        # Do not descend: its Constant parts are the same literal.


def _scan() -> tuple[list[_MatchLiteral], dict[str, dict[str, set[str]]]]:
    literals: list[_MatchLiteral] = []
    calls: dict[str, dict[str, set[str]]] = {}
    for directory in _SCANNED_DIRS:
        for file in sorted((_REPO / directory).rglob("*.py")):
            rel = file.relative_to(_REPO).as_posix()
            tree = ast.parse(file.read_text())
            collector = _Collector(rel, _docstring_ids(tree))
            collector.visit(tree)
            literals.extend(collector.found)
            calls[rel] = collector.calls
    return literals, calls


_LITERALS, _CALLS = _scan()

_FTS_LITERALS = [lit for lit in _LITERALS if _FTS_SITE.search(lit.text)]


def test_the_scan_covers_every_non_test_python_file_in_the_repo():
    """The enumeration is only complete if the SEARCH is.

    ``_SCANNED_DIRS`` is two names, and two names are a guess the moment
    somebody adds a third package. This walks the repo instead and fails if any
    non-test Python file lives outside them — which is the only way the claim
    "no MATCH site is left" can be checked rather than asserted.
    """
    skip = {"tests", "result", ".git", ".venv", "__pycache__", "node_modules"}
    stray: list[str] = []
    for entry in sorted(_REPO.iterdir()):
        if entry.name.startswith(".") or entry.name in skip or entry.is_symlink():
            continue
        if entry.is_file() and entry.suffix == ".py":
            stray.append(entry.name)
        elif entry.is_dir() and entry.name not in _SCANNED_DIRS:
            stray.extend(
                p.relative_to(_REPO).as_posix()
                for p in entry.rglob("*.py")
                if "__pycache__" not in p.parts
            )
    assert not stray, (
        f"Python outside {_SCANNED_DIRS} is unscanned for FTS5 MATCH sites: "
        f"{stray}. Add its directory to _SCANNED_DIRS."
    )


def test_the_scan_found_the_sites_it_is_supposed_to_be_watching():
    """A scanner that silently matches nothing passes every other test here.

    The floor is deliberately concrete rather than ``> 0``: the store alone has
    six FTS5 statements and the smoke script has three, and a refactor that
    drops the count without deleting the feature is itself worth a red test.
    """
    assert len(_FTS_LITERALS) >= 9, [repr(lit) for lit in _FTS_LITERALS]
    files = {lit.path for lit in _FTS_LITERALS}
    assert "aggregator/core/store.py" in files
    assert "scripts/rag_rollout_smoke.py" in files


def test_every_match_literal_is_classified():
    """THE COMPLETENESS PROPERTY. Three buckets, and they cover everything.

    Not "we fixed this one and now there are five". A MATCH literal that is
    neither an FTS5 site nor a sqlite-vec KNN nor a declared non-SQL string is
    a site nobody classified, and it fails here with its file and line.
    """
    unclassified = [
        lit
        for lit in _LITERALS
        if not _FTS_SITE.search(lit.text)
        and not _VEC_SITE.search(lit.text)
        and (lit.path, lit.func, lit.text) not in _NOT_SQL
    ]
    assert not unclassified, (
        "unclassified MATCH literal(s) — every string reaching FTS5 MATCH must "
        "be rewritten by fts5_match_query, and every string that is not an "
        "FTS5 site must say what it is: "
        + "; ".join(f"{lit!r}: {lit.text!r}" for lit in unclassified)
    )


@pytest.mark.parametrize("lit", _FTS_LITERALS, ids=repr)
def test_every_fts5_match_site_is_reached_through_the_whitelist(lit):
    """The enclosing function must call :func:`fts5_match_query` itself.

    Asserted on the function that OWNS the statement rather than somewhere up
    the call chain, because "a caller sanitizes it" is precisely the belief
    that produced all three instances of this defect. A helper that takes an
    already-rewritten expression should take it as a parameter from a function
    that does the rewrite — and that function is the one this rule names.
    """
    owner = lit.func
    called = _CALLS[lit.path].get(owner, set())
    assert _SANITIZER in called, (
        f"{lit!r} binds text to FTS5 MATCH without calling {_SANITIZER}(). "
        "Raw user text reaching MATCH is a syntax error for any query "
        "containing '-', '.', '#', '+', ':' or a quote."
    )


def _reaches_match() -> set[str]:
    """Plain function names that reach an FTS5 ``MATCH``, transitively.

    Seeded from the enumeration above — the functions that OWN a MATCH
    statement — then closed over the whole call graph of both scanned trees,
    so ``fts_obs_ranked`` (which only calls ``_fts_ranked``) and
    ``aggregator.mcp`` calling ``store._fts_obs_ids`` are both in the set
    without anybody listing them.
    """
    reaching = {lit.func.rsplit(".", 1)[-1] for lit in _FTS_LITERALS}
    edges = [
        (qualified.rsplit(".", 1)[-1], callees)
        for per_file in _CALLS.values()
        for qualified, callees in per_file.items()
    ]
    changed = True
    while changed:
        changed = False
        for caller, callees in edges:
            if caller not in reaching and callees & reaching:
                reaching.add(caller)
                changed = True
    return reaching


_REACHES_MATCH = _reaches_match()


def _handler_reports(handler: ast.ExceptHandler) -> bool:
    """Does this handler tell anyone the lexical arm dropped out?

    Three acceptable shapes, all of which leave a trace a human can find:
    re-raise; log it; or hand the exception back to the caller (the
    ``return [], str(e)`` shape ``_fts_ranked`` uses). Anything else is the
    empty-result-looks-like-success failure this project bans by name.
    """
    for node in ast.walk(handler):
        if isinstance(node, ast.Raise):
            return True
        if isinstance(node, ast.Call):
            rendered = ast.unparse(node.func).lower()
            if any(
                word in rendered
                for word in ("log", "warn", "error", "exception", "print", "notify")
            ):
                return True
        if handler.name and isinstance(node, ast.Name) and node.id == handler.name:
            return True
    return False


def _empty_binding(handler: ast.ExceptHandler) -> str | None:
    for node in ast.walk(handler):
        if not isinstance(node, ast.Assign | ast.AnnAssign):
            continue
        value = node.value
        if value is None:
            continue
        empty_call = (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in {"set", "list", "dict", "tuple", "frozenset"}
            and not value.args
        )
        empty_literal = isinstance(
            value, ast.List | ast.Dict | ast.Set | ast.Tuple
        ) and not (getattr(value, "elts", None) or getattr(value, "keys", None))
        if empty_call or empty_literal:
            return f"line {node.lineno}: {ast.unparse(node)}"
    return None


def test_no_operational_error_on_the_match_path_is_swallowed_into_an_empty_result():
    """Fail loudly: an emptied set is indistinguishable from "no matches".

    The smoke script's ``cmd_distances`` caught the ``OperationalError`` that
    ``store_fts_all`` raised on six of seven realistic queries and assigned
    ``set()``, which silently emptied the "already reachable by FTS5"
    exclusion feeding ``cosine_distance.vector_only_relevant`` — the
    distribution the distance floor is judged against. Six sevenths of that
    exclusion vanished and the JSON looked exactly the same.

    Scoped to the MATCH path by the CALL GRAPH rather than by a hand-listed
    set of functions, so this cannot be defeated by adding one more wrapper.
    Handlers elsewhere are none of this test's business: ``_v4_surface``
    treating a missing ``ingest_state`` as "no watermarks" is the right answer
    to a different question.
    """
    offenders: list[str] = []
    for directory in _SCANNED_DIRS:
        for file in sorted((_REPO / directory).rglob("*.py")):
            rel = file.relative_to(_REPO).as_posix()
            tree = ast.parse(file.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Try):
                    continue
                guarded = {
                    call.func.id
                    if isinstance(call.func, ast.Name)
                    else call.func.attr
                    for stmt in node.body
                    for call in ast.walk(stmt)
                    if isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name | ast.Attribute)
                }
                if not guarded & _REACHES_MATCH:
                    continue
                for handler in node.handlers:
                    if "OperationalError" not in ast.unparse(
                        handler.type or ast.Pass()
                    ):
                        continue
                    empty = _empty_binding(handler)
                    if empty and not _handler_reports(handler):
                        offenders.append(f"{rel}:{empty}")
    assert not offenders, (
        "sqlite3.OperationalError from the FTS5 MATCH path swallowed into an "
        f"empty result, with nothing logged, raised or returned: {offenders}"
    )
