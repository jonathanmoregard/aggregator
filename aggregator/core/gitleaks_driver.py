"""Run gitleaks 8.30 over many texts and return, per text, the strings to redact.

Invariants:
- Each text is judged alone: one scanned file per piece, so a rule's keywords never cross texts.
- gitleaks reads every byte or the call fails: no file is skipped as binary or cut mid-token,
  and gitleaks' own read count is checked against what was written.
- Nothing in or around the input can silence a finding (no ignore files, gitleaks:allow ignored).
- Any scanner fault raises GitleaksError; an empty result means "read it all, found nothing".
- Results are plain target strings (never containing a redaction marker); no coordinates leave.
- ``redact`` replaces every occurrence of every target; overlapping ones become one marker.
Stdlib only: vendored byte-identical into the aggregator repo.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from collections.abc import Callable

# BMP magic: gitleaks sniffs each file and silently skips application/* types (%PDF, MZ, ...).
HEADER = b"BM gitleaks-driver\n\n"
# HEADER + piece stays under gitleaks' 100,000-byte read, so gitleaks never cuts a file itself.
PIECE_BYTES = 96_000
# Neighbouring pieces share this many bytes, so a secret cut by one piece's edge is whole in
# the other (an armored private key is well under 16 KB).
OVERLAP_BYTES = 16_000
# A Secret capture shorter than this is too generic to redact everywhere; use the full Match.
MIN_SECRET_CHARS = 8
# Decoded findings report the DECODED secret; the encoded runs on that line are what to redact.
ENCODED_RUN = re.compile(r"[A-Za-z0-9+/=_%\-]{16,}")
# A target containing a marker is a match on an earlier redaction: dropping it keeps reruns stable.
MARKER = "[REDACTED:"
# An existing marker on a decoded line is not an encoded run (its rule id could look like one).
MARKER_SPAN = re.compile(r"\[REDACTED:[^\]\n]*\]")
# gitleaks' log line with its own read count.
SCANNED = re.compile(r"scanned ~(\d+) bytes")
# gitleaks colours its fatal log lines even under --no-color.
ANSI = re.compile(r"\x1b\[[0-9;]*m")
# tmpfs where available, so unredacted text is not written to disk.
RAM_DIR = "/dev/shm" if os.access("/dev/shm", os.W_OK) else None


class GitleaksError(Exception):
    """The scan did not complete; the texts are NOT known to be clean. Never carries text."""


def pieces(data: bytes) -> list[bytes]:
    """``data`` as PIECE_BYTES windows, each overlapping the next by OVERLAP_BYTES."""
    step = PIECE_BYTES - OVERLAP_BYTES
    return [data[i : i + PIECE_BYTES] for i in range(0, max(len(data) - OVERLAP_BYTES, 1), step)]


def find_secrets(texts: list[str], *, gitleaks: str, rules: str, timeout: float,
                 env: dict[str, str] | None = None) -> list[dict[str, str]]:
    """Per text, ``{target string: rule id}``, from one gitleaks run of at most ``timeout`` s."""
    files = [(i, HEADER + p) for i, t in enumerate(texts) if t
             for p in pieces(t.encode("utf-8", "replace"))]
    result: list[dict[str, str]] = [{} for _ in texts]
    if not files:
        return result
    report = _run(gitleaks, rules, [data for _, data in files], timeout, env)
    try:
        for finding in report:
            i, data = files[int(os.path.basename(finding["File"]).removesuffix(".txt"))]
            for target in _targets(finding, data):
                if MARKER not in target:  # several rules on one target: the same one every run
                    result[i][target] = min(finding["RuleID"], result[i].get(target, finding["RuleID"]))
    except (ValueError, KeyError, TypeError, IndexError) as e:
        raise GitleaksError("gitleaks report was unreadable") from e
    return result


def _run(gitleaks: str, rules: str, files: list[bytes], timeout: float,
         env: dict[str, str] | None) -> list:
    """gitleaks' JSON report over ``files`` (written as in/<n>.txt), checked to be complete."""
    cmd = [gitleaks, "dir", "in", "--config", os.path.abspath(rules),
           "--report-format", "json", "--report-path", "-", "--exit-code", "0",
           "--no-banner", "--no-color", "--log-level", "info",
           "--ignore-gitleaks-allow", "--gitleaks-ignore-path", os.devnull]
    try:
        # A fresh 0700 dir as cwd AND scan root: no .gitleaksignore can be found in either.
        with tempfile.TemporaryDirectory(prefix="gitleaks-driver-", dir=RAM_DIR) as work:
            os.mkdir(os.path.join(work, "in"), 0o700)
            for n, data in enumerate(files):
                with open(os.path.join(work, "in", f"{n}.txt"), "wb") as fh:
                    fh.write(data)
            proc = subprocess.run(cmd, cwd=work, capture_output=True, timeout=timeout, check=False,
                                  stdin=subprocess.DEVNULL, env={**os.environ, **(env or {})})
    except subprocess.TimeoutExpired as e:
        raise GitleaksError(f"gitleaks timed out after {timeout:.1f}s") from e
    except OSError as e:
        raise GitleaksError(f"gitleaks could not run: {e.strerror or type(e).__name__} ({gitleaks})") from e
    stderr = ANSI.sub("", proc.stderr.decode("utf-8", "replace"))
    if proc.returncode != 0:
        fatal = next((ln for ln in stderr.splitlines() if "FTL" in ln), "")
        raise GitleaksError(f"gitleaks exited {proc.returncode} {fatal[:200]}".rstrip())
    scanned = SCANNED.search(stderr)
    written = sum(map(len, files))
    if scanned is None or int(scanned[1]) < written:
        raise GitleaksError(f"gitleaks read {scanned[1] if scanned else '?'} of {written} bytes")
    try:
        report = json.loads(proc.stdout)
    except ValueError as e:
        raise GitleaksError("gitleaks report was unreadable") from e
    if not isinstance(report, list):
        raise GitleaksError("gitleaks report was unreadable")
    return report


def _targets(finding: dict, data: bytes) -> list[str]:
    """The strings in the scanned file ``data`` that redact this finding."""
    if any(str(tag).startswith("decoded:") for tag in finding.get("Tags") or ()):
        lines = data.decode("utf-8", "replace").split("\n")
        n = finding["StartLine"] - 1
        if not 0 <= n < len(lines):
            raise GitleaksError("gitleaks placed a decoded finding outside its file")
        return ENCODED_RUN.findall(MARKER_SPAN.sub(" ", lines[n]))
    secret = finding.get("Secret") or ""
    target = secret if len(secret) >= MIN_SECRET_CHARS else finding.get("Match") or secret
    return [target] if target else []


def redact(text: str, targets: dict[str, str], marker: Callable[[str, int], str]) -> str:
    """``text`` with each occurrence of each target (``{target: rule id}``) replaced by
    ``marker(rule, length)``. Occurrences that share characters become one marker, named
    by the longest target at its start, so no fragment of either survives between them."""
    # Hand-rolled on purpose: the hooks run stdlib-only `python3 -S -E`, where pyahocorasick
    # (in nixpkgs) is not importable, and an `re` alternation took 19 s for 1,000 targets
    # over 8 MB. One str.find pass per target took 0.7 s.
    spans = sorted(((start, end, rule) for t, rule in targets.items() if t for start, end in _spans(text, t)),
                   key=lambda span: (span[0], -span[1]))
    merged: list[list] = []
    for start, end, rule in spans:
        if merged and start < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end, rule])
    out, pos = [], 0
    for start, end, rule in merged:
        out += [text[pos:start], marker(rule, end - start)]
        pos = end
    return "".join(out) + text[pos:]


def _spans(text: str, target: str):
    """(start, end) of each run of overlapping occurrences of ``target``: one span per run,
    so a target repeated through a long run of one character costs no memory."""
    start = text.find(target)
    while start != -1:
        end = start + len(target)
        nxt = text.find(target, start + 1)
        while nxt != -1 and nxt < end:
            end = nxt + len(target)
            nxt = text.find(target, nxt + 1)
        yield start, end
        start = nxt
