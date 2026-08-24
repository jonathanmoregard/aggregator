#!/usr/bin/env python3
"""Count the embedding backfill's REAL token bill, and price the levers.

THE NUMBER IN ``docs/embedding-throughput.md`` HAS TO BE RE-RUNNABLE. That file
originally estimated the bill from a row count and an assumed mean body length
("~240k embeddable rows", "≈14 days if the mean body is nearer 200 tokens").
Both halves were wrong, and wrong in the direction that made the backfill look
tractable: the real corpus is 65.2% embeddable, not ~50%, and the real mean is
423 tokens per chunk for observations and 866 for records. This script is what
replaced the assumption, and it lives here rather than in a scratch directory
because a document that cites a measurement nobody can re-run is a document
that will drift from it.

THE ONE THING TO GET RIGHT HERE IS WHAT THE EMBEDDER ACTUALLY SEES. Observations
embed their body alone; RECORDS embed ``subject + "\n\n" + body`` (``cli.py``'s
backlog loop). Counting the record bill from ``body`` alone reports 1,242
title-only rows — mostly TickTick tasks with no notes — as unembeddable, when
every one of them is indexed on its title. See :func:`_embed_text`; this file
shipped with that bug for one commit.

Three subcommands:

* ``total`` — the whole bill. Tokenizer only, no forward passes, so it costs
  minutes rather than the weeks the actual embedding does.
* ``by-source`` — the same bill split by BACKFILL GROUP, in the user's priority
  order (``dropbox -> substack -> claude-web -> sessions -> subagents ->
  rest``). This is the subcommand that matters for planning: the day columns are
  CUMULATIVE, so the answer to "when does the vector arm become useful" is a
  prefix of the table and not its last row.
* ``bench-int8`` — does int8 dynamic quantization of the CACHED model buy
  anything? Measures throughput AND geometry drift together, because a speedup
  that moves the vectors is not free: ``hybrid.VECTOR_FLOOR_MAX_DISTANCE`` is
  calibrated in this embedding space.

Everything here reads the model from the LOCAL Hugging Face cache with
``local_files_only=True`` and never downloads weights. ``total`` and
``by-source`` load only the tokenizer.

Usage::

    uv run python scripts/embedding_token_bill.py total --db PATH
    uv run python scripts/embedding_token_bill.py by-source --db PATH
    uv run python scripts/embedding_token_bill.py bench-int8
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time

MODEL = "Qwen/Qwen3-Embedding-0.6B"

#: Tokenizer batch size. Purely a throughput knob for this script.
_BATCH = 2000

#: The measured encoder rate on this machine, and two ESTIMATES scaled by
#: non-embedding parameter count. Only the first is measured; see
#: ``docs/embedding-throughput.md``.
RATES = {
    "Qwen3-0.6B (measured 40 t/s)": 40.0,
    "~300M class (est 176 t/s)": 176.0,
    "~33M class (est 530 t/s)": 530.0,
}

#: The user's backfill priority order, as cache source names. Groups not named
#: here are appended in sorted order — they are LATER, not unimportant.
BACKFILL_ORDER = (
    "dropbox",
    "substack",
    "claude-web",
    "chatgpt",
    "sessions",
    "subagents",
)

# Observations carry no source column of their own; the group is a property of
# the session they belong to. LEFT JOIN so an observation whose session row is
# missing is still counted rather than silently dropped from the bill.
_OBS_SQL = """
SELECT CASE
         WHEN s.origin = 'claude-code' AND s.kind = 'session'  THEN 'sessions'
         WHEN s.origin = 'claude-code' AND s.kind = 'subagent' THEN 'subagents'
         WHEN s.origin IN ('claude-web', 'chatgpt') THEN s.origin
         ELSE '(other-obs)'
       END AS src,
       o.body AS body
FROM observations o LEFT JOIN sessions s ON s.session_id = o.session_id
"""
_REC_SQL = "SELECT source AS src, subject, body FROM records"


def _embed_text(kind: str, row: sqlite3.Row) -> str:
    """The string the embedder would actually see for this row.

    MUST TRACK ``cli.py``'s backlog loop, and the records branch is the reason
    this is a function rather than a column in the SQL. Observations embed their
    body alone; RECORDS embed ``subject + "\\n\\n" + body``, so a record with an
    empty body and a real subject (a TickTick task with no notes — 1,176 of
    them) IS embeddable. Counting the body alone reported those as skipped and
    undercounted the records bill.
    """
    if kind == "observations":
        return row["body"] or ""
    return f"{row['subject']}\n\n{row['body']}"


def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL, local_files_only=True)


def _connect(db: str) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _count_tokens(tok, batch: list[str]) -> int:
    return sum(len(ids) for ids in tok(batch, add_special_tokens=True)["input_ids"])


def cmd_total(args: argparse.Namespace) -> int:
    """The whole bill, split by table."""
    from aggregator.core.chunk import chunk_body

    tok = _tokenizer()
    con = _connect(args.db)
    totals: dict[str, dict[str, int]] = {}

    def account(kind: str, sql: str) -> None:
        bucket = totals.setdefault(
            kind, {"rows": 0, "embeddable": 0, "chunks": 0, "tokens": 0}
        )
        batch: list[str] = []
        t0 = time.time()
        for row in con.execute(sql):
            bucket["rows"] += 1
            body = _embed_text(kind, row)
            if not body or not body.strip():
                continue
            chunks = chunk_body(body)
            if not chunks:
                continue
            bucket["embeddable"] += 1
            bucket["chunks"] += len(chunks)
            batch.extend(chunks)
            if len(batch) >= _BATCH:
                bucket["tokens"] += _count_tokens(tok, batch)
                batch = []
            if bucket["rows"] % 50000 == 0:
                print(
                    f"  {kind}: {bucket['rows']} rows, {bucket['tokens']} tokens, "
                    f"{time.time() - t0:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
        if batch:
            bucket["tokens"] += _count_tokens(tok, batch)

    account("observations", "SELECT body FROM observations")
    account("records", "SELECT subject, body FROM records")

    grand = {
        key: sum(b[key] for b in totals.values())
        for key in ("rows", "embeddable", "chunks", "tokens")
    }
    print()
    for kind, b in totals.items():
        per = b["tokens"] / b["chunks"] if b["chunks"] else 0
        print(
            f"{kind:14s} rows={b['rows']:>7d} embeddable={b['embeddable']:>7d} "
            f"chunks={b['chunks']:>7d} tokens={b['tokens']:>10d} "
            f"mean_tokens_per_chunk={per:.1f}"
        )
    print()
    print(f"TOTAL rows        {grand['rows']}")
    pct = 100 * grand["embeddable"] / grand["rows"] if grand["rows"] else 0
    print(f"TOTAL embeddable  {grand['embeddable']}  ({pct:.1f}%)")
    print(f"TOTAL chunks      {grand['chunks']}")
    print(f"TOTAL tokens      {grand['tokens']}")
    print()
    for label, rate in RATES.items():
        days = grand["tokens"] / rate / 86400
        print(f"  {label:32s} -> {days:7.2f} days of continuous CPU")
    return 0


def cmd_by_source(args: argparse.Namespace) -> int:
    """The bill by backfill group, with CUMULATIVE day columns."""
    from aggregator.core.chunk import chunk_body

    tok = _tokenizer()
    con = _connect(args.db)
    bill: dict[str, dict[str, int]] = {}
    pending: dict[str, list[str]] = {}

    def flush(src: str) -> None:
        batch = pending.get(src)
        if not batch:
            return
        bill[src]["tokens"] += _count_tokens(tok, batch)
        pending[src] = []

    for kind, sql in (("observations", _OBS_SQL), ("records", _REC_SQL)):
        for row in con.execute(sql):
            src = str(row["src"])
            b = bill.setdefault(src, {"rows": 0, "chunks": 0, "tokens": 0})
            b["rows"] += 1
            body = _embed_text(kind, row)
            if not body or not body.strip():
                continue
            chunks = chunk_body(body)
            if not chunks:
                continue
            b["chunks"] += len(chunks)
            pending.setdefault(src, []).extend(chunks)
            if len(pending[src]) >= _BATCH:
                flush(src)
        for src in list(pending):
            flush(src)

    rest = sorted(s for s in bill if s not in BACKFILL_ORDER)
    header = "  ".join(f"{k.split(' ')[0]:>10s}" for k in RATES)
    print(f"{'group':14s} {'rows':>7s} {'chunks':>7s} {'tokens':>11s}   {header}")
    cum = 0
    for src in list(BACKFILL_ORDER) + rest:
        if src not in bill:
            continue
        b = bill[src]
        cum += b["tokens"]
        days = "  ".join(f"{cum / r / 86400:10.2f}" for r in RATES.values())
        print(
            f"{src:14s} {b['rows']:>7d} {b['chunks']:>7d} {b['tokens']:>11d}   {days}"
        )
    print()
    print("(day columns are CUMULATIVE — time until that group is FINISHED)")
    print(
        "rates: 40 t/s measured; the others are scaled by non-embedding "
        "parameter count and are ESTIMATES"
    )
    return 0


# --- the int8 lever ---------------------------------------------------------

_BENCH_BODY = (
    "the ingest timer fired again and re-read every observation from scratch "
    "because since=None was passed, so the run could not finish inside "
    "TimeoutStartSec=4h and got SIGTERMed at about 44 percent. "
)

#: Measured characters per token on this corpus. Only used to report a tok/s
#: figure comparable with ``docs/embedding-throughput.md``.
_CHARS_PER_TOKEN = 4.97


def cmd_bench_int8(args: argparse.Namespace) -> int:
    """Throughput AND geometry drift, together — neither alone decides this."""
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    def texts(chars: int, n: int) -> list[str]:
        return [(f"docid-{i} " + _BENCH_BODY * 200)[:chars] for i in range(n)]

    def bench(model, chars: int, n: int) -> tuple[float, object]:
        batch = texts(chars, n)
        model.encode(batch[:1], show_progress_bar=False)  # warm
        t0 = time.perf_counter()
        vecs = model.encode(
            batch, show_progress_bar=False, normalize_embeddings=True
        )
        return time.perf_counter() - t0, vecs

    sizes = tuple(int(x) for x in args.sizes.split(","))
    n = args.repeats
    print(f"torch {torch.__version__}  threads={torch.get_num_threads()}")
    print(f"engine={torch.backends.quantized.engine}")

    fp32 = SentenceTransformer(MODEL, local_files_only=True)
    fp32.max_seq_length = args.max_seq_length
    results: dict[int, tuple[float, object]] = {}
    for chars in sizes:
        dt, vecs = bench(fp32, chars, n)
        toks = chars / _CHARS_PER_TOKEN
        results[chars] = (dt, vecs)
        print(
            f"fp32  {chars:5d} chars x{n}: {dt:7.2f}s total, "
            f"{dt / n:6.2f}s/chunk, {toks * n / dt:6.1f} tok/s"
        )

    q = SentenceTransformer(MODEL, local_files_only=True)
    q.max_seq_length = args.max_seq_length
    q[0].auto_model = torch.ao.quantization.quantize_dynamic(
        q[0].auto_model, {torch.nn.Linear}, dtype=torch.qint8
    )
    for chars in sizes:
        dt, vecs = bench(q, chars, n)
        toks = chars / _CHARS_PER_TOKEN
        fdt, fvecs = results[chars]
        cos = float(np.mean(np.sum(vecs * fvecs, axis=1)))
        print(
            f"int8  {chars:5d} chars x{n}: {dt:7.2f}s total, "
            f"{dt / n:6.2f}s/chunk, {toks * n / dt:6.1f} tok/s   "
            f"speedup={fdt / dt:4.2f}x   cos(fp32,int8)={cos:.5f}"
        )
    print()
    print(
        "A speedup near 1.00x means the lever is dead on this hardware. "
        "cos near 1.00000 means the geometry did NOT move, so the abstention "
        "floor would survive — which only matters if the speedup is real."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("total", help="the whole bill, by table")
    t.add_argument("--db", required=True, help="cache.db (read-only snapshot is fine)")
    t.set_defaults(func=cmd_total)

    s = sub.add_parser("by-source", help="the bill by backfill group, cumulative")
    s.add_argument("--db", required=True, help="cache.db (read-only snapshot is fine)")
    s.set_defaults(func=cmd_by_source)

    b = sub.add_parser("bench-int8", help="is int8 dynamic quantization worth it?")
    b.add_argument("--sizes", default="1800,4000", help="chunk sizes in characters")
    b.add_argument("--repeats", type=int, default=4)
    b.add_argument("--max-seq-length", type=int, default=1024)
    b.set_defaults(func=cmd_bench_int8)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
