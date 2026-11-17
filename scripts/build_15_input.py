"""Experiment 15 input: RAID sources from every domain, each with its human text and eleven generators' texts.

Reads fetch_raid_domains.py output (no attack, sampling, no repetition penalty). A source is complete when its
human text and all eleven generators' texts are present and have at least --min-words words. Per domain, a fixed
random sample of --n-sources complete sources (seed 0, from the sorted source ids). Whitespace is collapsed as in
convert_raid.py (04).

Rows: {"id": "raid:<RAID id>", "text", "group": "<domain>.H" | "<domain>.<model>", "label", "pair_id": <source id>,
       "generator", "domain": "raid_<domain>"}

Example:
  python -I scripts/build_15_input.py --csv data/raw/raid/domains_none_sampling.csv --out data/15_input.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

CHAT = ["chatgpt", "gpt4", "llama-chat", "mistral-chat", "mpt-chat", "cohere-chat"]
BASE = ["mistral", "mpt"]  # Mistral-7B-v0.1 and MPT-30B, base versions of mistral-chat and mpt-chat
OLD = ["gpt2"]  # GPT-2 XL (2019)
INSTRUCT = ["cohere", "gpt3"]  # Cohere command via generate, instruction-tuned davinci: not chat, not base
GENS = CHAT + BASE + OLD + INSTRUCT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-sources", type=int, default=150)
    ap.add_argument("--min-words", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    csv.field_size_limit(1 << 30)

    texts: dict[tuple[str, str], dict[str, tuple[str, str]]] = defaultdict(dict)  # (domain, source) -> model -> (id, text)
    dup: Counter = Counter()
    with open(args.csv, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["model"] != "human" and r["model"] not in GENS:
                continue
            key = (r["domain"], r["source_id"])
            if r["model"] in texts[key]:
                dup[r["model"]] += 1
                continue
            texts[key][r["model"]] = (r["id"], " ".join(r["generation"].split()))
    if dup:
        print(f"duplicate rows ignored (first kept): {dict(dup)}")

    out, report = [], []
    for domain in sorted({d for d, _ in texts}):
        sources = sorted(s for d, s in texts if d == domain)
        complete, short, missing = [], 0, 0
        for s in sources:
            m = texts[(domain, s)]
            if not all(k in m for k in ["human"] + GENS):
                missing += 1
            elif min(len(m[k][1].split()) for k in ["human"] + GENS) < args.min_words:
                short += 1
            else:
                complete.append(s)
        pick = sorted(random.Random(args.seed).sample(complete, min(args.n_sources, len(complete))))
        report.append(f"{domain:<10} sources {len(sources):>6}  missing a text {missing:>5}  short {short:>5}  "
                      f"complete {len(complete):>6}  sampled {len(pick)}")
        for s in pick:
            for k in ["human"] + GENS:
                rid, text = texts[(domain, s)][k]
                human = k == "human"
                out.append({"id": f"raid:{rid}", "text": text, "group": f"{domain}.{'H' if human else k}",
                            "label": 0 if human else 1, "pair_id": s, "generator": "" if human else k,
                            "domain": f"raid_{domain}"})
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print("\n".join(report))
    print(f"wrote {len(out)} rows to {args.out}")


if __name__ == "__main__":
    main()
