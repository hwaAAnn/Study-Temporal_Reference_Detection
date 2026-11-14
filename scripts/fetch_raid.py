"""Keep one domain of the RAID benchmark (Dugan et al., ACL 2024; MIT licence; huggingface.co/datasets/liamdugan/raid)
while the 11.8 GB train.csv streams in, so the full file is never stored (experiment 04).

Reads the CSV on stdin and writes the rows of --domain to --out (same columns). Prints counts by model, decoding,
repetition penalty and attack at the end.

Example (the download goes straight into the filter):
  curl -sL https://huggingface.co/datasets/liamdugan/raid/resolve/main/train.csv \
    | python -I scripts/fetch_raid.py --domain abstracts --out data/raw/raid/abstracts_train.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from collections import Counter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="abstracts")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    csv.field_size_limit(1 << 30)

    src = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", newline="")
    reader = csv.DictReader(src)
    need = {"domain", "model", "decoding", "repetition_penalty", "attack", "generation"}
    if not need <= set(reader.fieldnames or []):
        raise SystemExit(f"unexpected columns: {reader.fieldnames}")
    seen = kept = 0
    counts: Counter = Counter()
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=reader.fieldnames)
        w.writeheader()
        for row in reader:
            seen += 1
            if row["domain"] == args.domain:
                w.writerow(row)
                kept += 1
                counts[(row["model"], row["decoding"], row["repetition_penalty"], row["attack"])] += 1
            if seen % 500000 == 0:
                print(f"read {seen:,} rows, kept {kept:,}", file=sys.stderr, flush=True)
    print(f"read {seen:,} rows, kept {kept:,} ({args.domain})")
    for i, name in enumerate(["model", "decoding", "repetition_penalty", "attack"]):
        field: Counter = Counter()
        for key, c in counts.items():
            field[key[i]] += c
        print(f"{name}: " + ", ".join(f"{k}={v}" for k, v in sorted(field.items())))


if __name__ == "__main__":
    main()
