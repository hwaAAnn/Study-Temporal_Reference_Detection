"""Experiment 15 pipeline check: RAID texts that 04 already scored must get the same log p again.

subset:  rows of the 15 input whose id is also in 04's score file (RAID abstracts), at most --max rows
compare: per text, mean log p (2022 and 2024) of the new scores vs 04's yc04_<year>.jsonl; same token count,
         largest absolute difference <= 0.02 (08's tolerance)

Example:
  python scripts/check_15.py subset --input data/15_input.jsonl --ref outputs/yc04_2022.jsonl --out data/15_check.jsonl
  python scripts/check_15.py compare --new outputs/s15_check.jsonl --ref outputs/yc04_2022.jsonl outputs/yc04_2024.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TOL = 0.02


def read(path: str) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("subset")
    s.add_argument("--input", required=True)
    s.add_argument("--ref", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--max", type=int, default=48)
    c = sub.add_parser("compare")
    c.add_argument("--new", required=True)
    c.add_argument("--ref", nargs="+", required=True)
    args = ap.parse_args()

    if args.cmd == "subset":
        ref = {json.loads(x)["id"] for x in Path(args.ref).read_text(encoding="utf-8").splitlines() if x.strip()}
        rows = [r for r in read(args.input) if r["id"] in ref][: args.max]
        Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"{len(rows)} overlapping rows -> {args.out}")
        return

    new = {r["id"]: r for r in read(args.new)}
    worst, n = 0.0, 0
    for path in args.ref:
        for r in read(path):
            if r["id"] not in new:
                continue
            m = new[r["id"]]
            if m["n_tokens"] != r["n_tokens"]:
                raise SystemExit(f"{r['id']}: {m['n_tokens']} tokens now, {r['n_tokens']} in {path}")
            for y, lp in r["lp"].items():
                d = abs(float(np.mean(m["lp"][y])) - float(np.mean(lp)))
                worst, n = max(worst, d), n + 1
    if n == 0:
        raise SystemExit("no overlapping texts")
    print(f"{n} text x vintage comparisons, largest |mean log p difference| {worst:.5f} (tolerance {TOL})")
    print("PASS" if worst <= TOL else "FAIL")
    if worst > TOL:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
