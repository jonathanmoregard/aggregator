"""A stand-in for nixos-config's ``secret-redact`` CLI, for the unit tests.

Same contract (stdin {"texts", "timeout"} -> stdout {"texts", "counts"}, one
counts object per text), but a single prefix regex instead of gitleaks, so the
suite runs without the real scanner. The real CLI's behaviour is tested in
nixos-config (checks.x86_64-linux.secret-redact); ``tests/core/
test_scrub_secret_redact.py`` runs the real one when it is on PATH.
"""
import json
import re
import sys

TOKEN = re.compile(
    r"(?<![A-Za-z0-9_+/-])(?:gh[pousr]_|github_pat_|glpat-|sk-|AKIA|eyJ|tvly-|xox[abprs]-|GOCSPX-)"
    r"[A-Za-z0-9_.\-]{8,}"
)


def main() -> int:
    request = json.load(sys.stdin)
    texts, counts = [], []
    for text in request["texts"]:
        n = len(TOKEN.findall(text))
        texts.append(TOKEN.sub(lambda m: f"[REDACTED:stub:{len(m[0])}chars]", text))
        counts.append({"stub": n} if n else {})
    json.dump({"texts": texts, "counts": counts}, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
