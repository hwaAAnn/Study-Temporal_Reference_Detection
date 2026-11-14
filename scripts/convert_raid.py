"""RAID abstracts (fetch_raid.py output) -> score_texts.py rows for experiment 04.

A fixed random sample of source abstracts (seed 0); for each: the human abstract, every generator's text with
greedy and with sampling decoding (no repetition penalty), and, for the chat generators listed in
--attack-models (sampling), the --attacks versions together with the same attack applied to the human abstract.
RAID's human abstracts are arXiv abstracts, which are almost absent from FineWeb-Edu (02), so no DatedGPT
vintage trained on them.

Rows: {"id": "raid:<RAID id>", "text", "group", "label", "pair_id": <source id>, "generator", "decoding",
       "attack", "domain": "raid_abstracts"}
  group: RAID.H (human), RAID.H.<attack>, RAID.<model>.<decoding>, RAID.<model>.<decoding>.<attack>

Example:
  python -I scripts/convert_raid.py --csv data/raw/raid/abstracts_train.csv --out data/raid04.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-sources", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attacks", nargs="+", default=["paraphrase", "synonym"])
    ap.add_argument("--attack-models", nargs="+", default=["chatgpt", "gpt4", "llama-chat", "mistral-chat"])
    args = ap.parse_args()
    csv.field_size_limit(1 << 30)

    plain, attacked = [], []
    with open(args.csv, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["model"] != "human" and r["repetition_penalty"] != "no":
                continue
            if r["attack"] == "none":
                plain.append(r)
            elif r["attack"] in args.attacks and (r["model"] == "human" or
                                                  (r["model"] in args.attack_models and r["decoding"] == "sampling")):
                attacked.append(r)
    humans = sorted(r["id"] for r in plain if r["model"] == "human")
    keep = set(random.Random(args.seed).sample(humans, args.n_sources))

    def row(r: dict) -> dict:
        human = r["model"] == "human"
        att = "" if r["attack"] == "none" else f".{r['attack']}"
        group = ("RAID.H" if human else f"RAID.{r['model']}.{r['decoding']}") + att
        return {"id": f"raid:{r['id']}", "text": " ".join(r["generation"].split()), "group": group,
                "label": 0 if human else 1, "pair_id": r["source_id"], "generator": "" if human else r["model"],
                "decoding": r["decoding"], "attack": r["attack"], "domain": "raid_abstracts"}

    out = [row(r) for r in plain + attacked if r["source_id"] in keep and r["generation"].strip()]
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    counts = Counter(r["group"] for r in out)
    print(f"{len(out)} rows from {len(keep)} sources; {len(counts)} groups")
    for g, c in sorted(counts.items()):
        print(f"  {g:<40} {c}")


if __name__ == "__main__":
    main()
