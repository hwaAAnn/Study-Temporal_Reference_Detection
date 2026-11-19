"""Keep the unattacked rows of every RAID domain while the 11.8 GB train.csv streams in (experiment 15).

RAID (Dugan et al., ACL 2024; MIT licence; huggingface.co/datasets/liamdugan/raid). fetch_raid.py (04) keeps one
domain with every attack; here one pass keeps, for each domain in --domains, the human rows and the generator rows
with no attack, sampling decoding and no repetition penalty. Writes the same columns to --out and prints counts
by domain and model.

Example (the download goes straight into the filter):
  curl -sL https://huggingface.co/datasets/liamdugan/raid/resolve/main/train.csv \
    | python -I scripts/fetch_raid_domains.py --out data/raw/raid/domains_none_sampling.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from collections import Counter

DOMAINS = ["abstracts", "books", "news", "poetry", "recipes", "reddit", "reviews", "wiki"]


def keep(row: dict, domains: set[str]) -> bool:
    if row["domain"] not in domains or row["attack"] != "none":
        return False
    if row["model"] == "human":
        return True
    return row["decoding"] == "sampling" and row["repetition_penalty"] == "no"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=DOMAINS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    csv.field_size_limit(1 << 30)
    domains = set(args.domains)

    src = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", newline="")
    reader = csv.DictReader(src)
    need = {"domain", "model", "decoding", "repetition_penalty", "attack", "generation", "source_id"}
    if not need <= set(reader.fieldnames or []):
        raise SystemExit(f"unexpected columns: {reader.fieldnames}")
    seen = kept = 0
    counts: Counter = Counter()
    all_domains: Counter = Counter()
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=reader.fieldnames)
        w.writeheader()
        for row in reader:
            seen += 1
            all_domains[row["domain"]] += 1
            if keep(row, domains):
                w.writerow(row)
                kept += 1
                counts[(row["domain"], row["model"])] += 1
            if seen % 500000 == 0:
                print(f"read {seen:,} rows, kept {kept:,}", file=sys.stderr, flush=True)
    print(f"read {seen:,} rows, kept {kept:,}")
    print("domains in the file: " + ", ".join(f"{k}={v}" for k, v in sorted(all_domains.items())))
    for d in sorted({k[0] for k in counts}):
        print(f"{d}: " + ", ".join(f"{m}={c}" for (dd, m), c in sorted(counts.items()) if dd == d))


if __name__ == "__main__":
    main()
