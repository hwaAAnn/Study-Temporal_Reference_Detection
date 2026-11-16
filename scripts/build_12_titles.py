"""Experiment 12 titles: 08's H19 and B human abstracts (data/01_all.jsonl pair rows) with their titles.

Writes one row per human abstract: {pair_id, group (H19 or B), title, human}. The pod generates an AI abstract
per title (generate_recent.py) and build_12_input.py pairs it with the human abstract by pair_id.

Example:
  python scripts/build_12_titles.py --input data/01_all.jsonl --out data/12_titles.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out, n = [], {"H19": 0, "B": 0}
    for line in Path(args.input).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "human" not in r:
            continue
        g = r.get("human_group", "B")
        if g not in n:
            continue
        out.append(json.dumps({"pair_id": str(r["id"]), "group": g, "title": r["title"], "human": r["human"]},
                              ensure_ascii=False))
        n[g] += 1
    if n != {"H19": 400, "B": 400}:
        raise SystemExit(f"expected 400 H19 and 400 B, got {n}")
    Path(args.out).write_text("".join(x + "\n" for x in out), encoding="utf-8")
    print(f"wrote {len(out)} titles to {args.out} {n}")


if __name__ == "__main__":
    main()
