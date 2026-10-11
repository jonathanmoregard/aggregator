"""Run gitleaks 8.30 over many texts and return, per text, the strings to redact.

Invariants:
- Each text is judged alone: one scanned file per piece, so a rule's keywords never cross texts.
- gitleaks reads every byte or the call fails: no piece is skipped as binary or cut mid-token,
  and the "scanned ~N bytes" count is checked against what was written.
- Nothing in or around the input can silence a finding (no ignore files, gitleaks:allow ignored).
- Any scanner fault raises GitleaksError; an empty result means "read it all, found nothing".
- Results are plain target strings (never containing a redaction marker); no coordinates leave.
Stdlib only: vendored byte-identical into the aggregator repo.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time

# BMP magic: gitleaks sniffs each file and silently skips application/* types (%PDF, MZ, ...).
HEADER = "BM gitleaks-driver\n\n"
# Under gitleaks' 100,000-byte read, so a file is one fragment and never cut mid-token.
PIECE_BYTES = 96_000
# A secret straddling a piece edge lies whole inside the neighbouring piece.
OVERLAP_CHARS = 16_000
# A Secret capture shorter than this is too generic to redact everywhere; use the full Match.
MIN_SECRET_CHARS = 8
# Decoded findings report the DECODED secret; the encoded run on that line is what to redact.
ENCODED_RUN = re.compile(r"[A-Za-z0-9+/=_%\-]{16,}")
# A target containing a marker is a match on an earlier redaction: dropping it keeps reruns stable.
MARKER = "[REDACTED:"
# gitleaks' own read count; fewer bytes than written means something was skipped.
SCANNED = re.compile(r"scanned ~(\d+) bytes")
# gitleaks colours its log lines even under --no-color.
ANSI = re.compile(r"\x1b\[[0-9;]*m")
# Used when the caller gives no deadline; only a hung scanner reaches it.
DEFAULT_TIMEOUT_S = 300.0
# tmpfs where available, so unredacted text is not written to disk.
RAM_DIR = "/dev/shm" if os.access("/dev/shm", os.W_OK) else None
# Where Nix puts gitleaks when PATH is minimal (systemd units, hooks).
FALLBACK_DIRS = ("/etc/profiles/per-user/{user}/bin", "{home}/.nix-profile/bin",
                 "/run/current-system/sw/bin")


class GitleaksError(Exception):
    """The scan did not complete; the texts are NOT known to be clean. Never carries text."""


def gitleaks_binary(env_var: str | None = None) -> str | None:
    """The gitleaks to run: ``$env_var`` if set (and nothing else), else PATH, else Nix dirs."""
    if env_var and os.environ.get(env_var):
        path = os.environ[env_var]
        return path if os.path.isfile(path) and os.access(path, os.X_OK) else None
    found = shutil.which("gitleaks")
    if found:
        return found
    user, home = os.environ.get("USER", ""), os.path.expanduser("~")
    for d in FALLBACK_DIRS:
        path = os.path.join(d.format(user=user, home=home), "gitleaks")
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def pieces(text: str) -> list[str]:
    """``text`` cut into overlapping pieces that each fit one file, preferring newline cuts."""
    budget = PIECE_BYTES - len(HEADER.encode())
    out, start = [], 0
    while True:
        end = min(len(text), start + budget)
        while len(text[start:end].encode("utf-8", "replace")) > budget:
            end = start + (end - start) * 9 // 10
        if end < len(text):
            nl = text.rfind("\n", start + OVERLAP_CHARS + 1, end)
            end = nl + 1 if nl != -1 else end
        out.append(text[start:end])
        if end >= len(text):
            return out
        start = max(end - OVERLAP_CHARS, start + 1)


def find_secrets(texts: list[str], *, rules: str, binary: str, deadline: float | None = None,
                 env: dict | None = None) -> list[dict[str, str]]:
    """Per text, ``{target string: rule id}``. ``deadline`` is a time.monotonic() instant."""
    owner = [(i, p) for i, t in enumerate(texts) if t for p in pieces(t)]
    result: list[dict[str, str]] = [{} for _ in texts]
    if not owner:
        return result
    timeout = DEFAULT_TIMEOUT_S if deadline is None else deadline - time.monotonic()
    if timeout <= 0:
        raise GitleaksError("deadline passed before the scan")
    cmd = [binary, "dir", "in", "--config", os.path.abspath(rules),
           "--report-format", "json", "--report-path", "-", "--exit-code", "0",
           "--no-banner", "--no-color", "--log-level", "info",
           "--ignore-gitleaks-allow", "--gitleaks-ignore-path", os.devnull]
    written = 0
    try:
        # Fresh 0700 dir as cwd AND scan root: no .gitleaksignore can be found in either.
        with tempfile.TemporaryDirectory(prefix="gitleaks-driver-", dir=RAM_DIR) as work:
            os.mkdir(os.path.join(work, "in"), 0o700)
            for n, (_, piece) in enumerate(owner):
                data = (HEADER + piece).encode("utf-8", "replace")
                with open(os.path.join(work, "in", f"{n}.txt"), "wb") as fh:
                    fh.write(data)
                written += len(data)
            proc = subprocess.run(cmd, cwd=work, capture_output=True, timeout=timeout, check=False,
                                  stdin=subprocess.DEVNULL, env={**os.environ, **(env or {})})
    except subprocess.TimeoutExpired as e:
        raise GitleaksError(f"gitleaks timed out after {timeout:.1f}s") from e
    except OSError as e:
        raise GitleaksError(f"gitleaks could not run ({type(e).__name__})") from e
    stderr = ANSI.sub("", proc.stderr.decode("utf-8", "replace"))
    if proc.returncode != 0:
        fatal = next((ln for ln in stderr.splitlines() if "FTL" in ln), "")
        raise GitleaksError(f"gitleaks exited {proc.returncode} {fatal[:200]}".rstrip())
    scanned = SCANNED.search(stderr)
    if scanned is None or int(scanned.group(1)) < written:
        raise GitleaksError(f"gitleaks read {scanned.group(1) if scanned else '?'} of {written} bytes")
    try:
        report = json.loads(proc.stdout)
        for f in report:
            n = int(os.path.basename(f["File"]).removesuffix(".txt"))
            i, piece = owner[n]
            for target in _targets(f, piece):
                if MARKER not in target:
                    result[i].setdefault(target, f["RuleID"])
    except (ValueError, KeyError, TypeError, IndexError) as e:
        raise GitleaksError("gitleaks report was unreadable") from e
    return result


def _targets(finding: dict, piece: str) -> list[str]:
    """The strings in ``piece`` that redact this finding."""
    if any(str(tag).startswith("decoded:") for tag in finding.get("Tags") or ()):
        lines = piece.split("\n")
        n = finding["StartLine"] - HEADER.count("\n") - 1
        return ENCODED_RUN.findall(lines[n]) if 0 <= n < len(lines) else []
    secret = finding.get("Secret") or ""
    target = secret if len(secret) >= MIN_SECRET_CHARS else finding.get("Match") or secret
    return [target] if target else []
