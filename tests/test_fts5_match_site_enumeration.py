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

AND THEN THE SCANNER ITSELF HAD THE SAME CLASS OF HOLE, which is the fourth
instance and the reason the last section of this file exists. ``_MATCH_WORD``
was case-SENSITIVE, so ``WHERE obs_fts match ?`` — valid SQLite, raw user text,
a real FTS5 MATCH — was collected by nothing and classified by nothing. Every
test above still passed, because every test above asserts something about the
literals the scanner FOUND. A mechanical enumeration is only worth what its
collector is worth, so the collector is now itself tested, against planted
sites in shapes nobody here wrote: lowercase, mixed case, lowercase inside an
f-string, lowercase sqlite-vec. That is the difference between "we parse the
source" and "we would catch it".
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCANNED_DIRS = ("aggregator", "scripts")

#: The SQL keyword, as a word. ``rematch``/``obj.match`` are not it.
#:
#: CASE-INSENSITIVE, AND THAT IS THE WHOLE POINT OF THE FLAG. This pattern
#: shipped without ``re.IGNORECASE`` and the omission was a one-character hole
#: straight through the completeness property: ``WHERE obs_fts match ?`` is
#: valid SQLite, binds raw user text to a real FTS5 MATCH, and was collected by
#: nothing and classified by nothing, so the file went green on a smuggled site.
#: SQLite does not care about the case of a keyword and neither may this.
#: ``test_the_scanner_is_not_defeated_by_lowercasing_the_operator`` plants one
#: of each shape.
_MATCH_WORD = re.compile(r"(?<![\w.])MATCH(?![\w])", re.IGNORECASE)

#: An FTS5 site: the left operand of ``MATCH`` is an FTS5 table. Both the
#: literal table names this project uses and the f-string form the smoke
#: script builds. A NEW table name matching ``\w+_fts`` is picked up
#: automatically; anything else falls through to the unclassified bucket.
#:
#: Case-insensitive for the same reason as ``_MATCH_WORD``, and it has to move
#: WITH it: collecting a lowercase site while the bucket regexes stayed
#: case-sensitive would drop every one of them into the unclassified bucket, so
#: the file would go red on real, correctly-sanitized statements and the fix
#: would be reverted within a day.
_FTS_SITE = re.compile(r"(?:\b\w*_fts\b|\{table\})\s+MATCH\b", re.IGNORECASE)

#: A sqlite-vec KNN site: the left operand is the ``embedding`` column and the
#: bound parameter is a float32 blob, never user text.
_VEC_SITE = re.compile(r"\bembedding\s+MATCH\b", re.IGNORECASE)

#: Literals that mention MATCH but are not SQL at all. Frozen deliberately:
#: adding one is a decision, not a formality.
#:
#: THE PRICE OF CASE-INSENSITIVITY, MEASURED BEFORE IT WAS PAID. Matching any
#: case also collects ordinary English containing the word "match", which then
#: has to be classified here. Counted over every non-docstring string literal in
#: ``aggregator/`` and ``scripts/`` at the time this landed: the flag added
#: exactly ONE literal to the set, and the confidence-surface work in the same
#: wave added a second. Two declared entries is a cheaper guarantee than a
#: scanner with a hole in it, and the alternative — a heuristic that only reads
#: lowercase MATCH when the literal "looks like SQL" — was rejected because the
#: shape it would miss is ``f"... {table} match '{q}'"``, i.e. precisely the
#: interpolated-user-text injection this file exists to catch.
_NOT_SQL = {
    (
        "aggregator/core/store.py",
        "Store._fts_rows",
        "FTS5 MATCH failed for %r (rewritten to %r) — the lexical arm "
        "contributes nothing to this query",
    ),
    # Both of these are sentences shown to the AGENT, in ``low_confidence_reason``.
    # They say "no match" / "DID match" about the keyword arm in English and
    # touch no SQL at all.
    (
        "aggregator/mcp.py",
        "_note_confidence",
        "the keyword arm was UNAVAILABLE for this query — it failed while "
        "running, so it contributed nothing and these rows come from the "
        "semantic arm alone. This is NOT the same as the keyword arm finding "
        "no match: nothing here has been checked against your words at all, "
        "and a re-run may answer differently",
    ),
    (
        "aggregator/mcp.py",
        "_note_confidence",
        "the keyword arm ran and DID match rows for this query, but the second "
        "lookup that maps its hits onto these session cards failed, so whether "
        "it corroborated the rows on this page is UNKNOWN. This is weaker than "
        "either 'corroborated' or 'not corroborated' — a re-run may answer "
        "differently",
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


def _collect(rel: str, source: str) -> tuple[list[_MatchLiteral], dict[str, set[str]]]:
    """Run the collector over ONE module's source.

    Split out of :func:`_scan` so the scanner can be pointed at a synthetic
    module. A scanner is a claim about what it would catch, and the only way to
    check that claim is to hand it something it is supposed to catch — see
    ``test_the_scanner_is_not_defeated_by_lowercasing_the_operator``, which
    exists because it was.
    """
    tree = ast.parse(source)
    collector = _Collector(rel, _docstring_ids(tree))
    collector.visit(tree)
    return collector.found, collector.calls


def _scan() -> tuple[list[_MatchLiteral], dict[str, dict[str, set[str]]]]:
    literals: list[_MatchLiteral] = []
    calls: dict[str, dict[str, set[str]]] = {}
    for directory in _SCANNED_DIRS:
        for file in sorted((_REPO / directory).rglob("*.py")):
            rel = file.relative_to(_REPO).as_posix()
            found, per_file = _collect(rel, file.read_text())
            literals.extend(found)
            calls[rel] = per_file
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


# --- the scanner's own teeth ------------------------------------------------
#
# Everything above asserts things about THIS repo. Nothing above asserts that
# the scanner would catch a site it has never seen, and that is the only
# property the file is actually selling: "a site nobody thought of lands in no
# bucket and this file goes red". The tests below hand it sites nobody thought
# of.

#: One smuggled FTS5 site per shape, all of them valid SQLite. ``MATCH`` is a
#: keyword and SQLite does not care about its case, so every one of these binds
#: raw text to a real FTS5 MATCH; none of them calls the sanitizer.
_SMUGGLED = {
    "lowercase operator": (
        'def sneak(con, q):\n'
        '    return con.execute("SELECT obs_id FROM obs_fts WHERE obs_fts match ?", (q,))\n'
    ),
    "mixed case operator": (
        'def sneak(con, q):\n'
        '    return con.execute("SELECT obs_id FROM obs_fts WHERE obs_fts Match ?", (q,))\n'
    ),
    "lowercase in an f-string": (
        'def sneak(con, q, table):\n'
        '    return con.execute(f"SELECT rowid FROM {table} WHERE {table} match ?", (q,))\n'
    ),
    "lowercase vec KNN": (
        'def sneak(con, v):\n'
        '    return con.execute("SELECT rowid FROM vec0 WHERE embedding match ?", (v,))\n'
    ),
}


@pytest.mark.parametrize("shape", sorted(_SMUGGLED))
def test_the_scanner_is_not_defeated_by_lowercasing_the_operator(shape):
    """THE HOLE THIS FILE SHIPPED WITH, and it was one character wide.

    ``_MATCH_WORD`` was ``re.compile(r"(?<![\\w.])MATCH(?![\\w])")`` — case
    SENSITIVE. ``WHERE obs_fts match ?`` is valid SQLite, binds raw user text to
    a real FTS5 MATCH, and was collected by nothing and classified by nothing,
    so the completeness property the module docstring sells ("a site nobody
    thought of lands in no bucket and this file goes red") silently did not hold
    for it. Demonstrated by planting exactly these shapes: lowercase went green,
    the same site with the operator uppercased went red.

    No site in the repo exploited it, so this was a hole in the guarantee rather
    than a live bug — which is the reason to fix it rather than a reason not to:
    the guarantee is the whole product here.
    """
    found, _calls = _collect("scripts/smuggled.py", _SMUGGLED[shape])
    assert found, (
        f"the {shape} site was collected by nothing — it is invisible to every "
        "other test in this file, including the completeness property"
    )
    lit = found[0]
    assert _FTS_SITE.search(lit.text) or _VEC_SITE.search(lit.text), (
        f"the {shape} site was collected but fell in no bucket: {lit.text!r}"
    )


def test_a_smuggled_site_is_held_to_the_whitelist_rule_like_any_other():
    """Collected and bucketed is not enough — the rule has to bite.

    The point of classifying a lowercase site as an FTS5 site is that
    ``test_every_fts5_match_site_is_reached_through_the_whitelist`` then demands
    its owning function call ``fts5_match_query``. This checks the demand
    actually fails for a function that does not.
    """
    found, calls = _collect(
        "scripts/smuggled.py", _SMUGGLED["lowercase operator"]
    )
    lit = found[0]
    assert _SANITIZER not in calls.get(lit.func, set()), (
        "the planted site does not sanitize, so the whitelist rule must be the "
        "thing that catches it"
    )


def test_prose_that_merely_says_match_is_not_mistaken_for_sql():
    """The cost of case-insensitivity, bounded and paid deliberately.

    Making the scanner case-insensitive means it also collects ordinary English
    string literals containing the word "match", which then have to be
    classified. Measured over ``aggregator/`` and ``scripts/`` when this landed:
    ONE such literal existed. So the noise is a declared entry in ``_NOT_SQL``,
    not a reason to keep the hole — but a scanner that classified prose as an
    FTS5 site would be worse than either, so that is checked here.
    """
    source = (
        'def report():\n'
        '    return "the keyword arm found no match: nothing was checked"\n'
    )
    found, _calls = _collect("aggregator/prose.py", source)
    assert found, "case-insensitive collection has to see it at all"
    lit = found[0]
    assert not _FTS_SITE.search(lit.text) and not _VEC_SITE.search(lit.text), (
        f"English prose classified as a SQL site: {lit.text!r}"
    )
