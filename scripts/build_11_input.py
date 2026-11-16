"""Experiment 11 input: 08's abstracts (part A) and 09's main pools (part B) in one file for the pod.

Part A = data/01_all.jsonl groups H19, C19, H24, C24, B, C (2,398 texts, as 08).
Part B = data/04_new.jsonl groups RAID.H, RAID.<chat model>.sampling, APT.H, APT.<polisher>.{slight_major,major}
(5,596 texts, as 09's main pools). Rows go through trd.data.normalise_rows (01 stores human/AI pairs) and are
written as text rows {id, pair_id, group, label, text, generator, domain}, which normalise_rows reads back
unchanged. Ids must be unique across the two files.

Example:
  python scripts/build_11_input.py --a data/01_all.jsonl --b data/04_new.jsonl --out data/11_input.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trd.data import normalise_rows  # noqa: E402

A_GROUPS = {"H19", "C19", "H24", "C24", "B", "C"}
B_PATTERN = re.compile(r"^(RAID\.H|APT\.H|RAID\.(chatgpt|gpt4|cohere-chat|llama-chat|mistral-chat|mpt-chat)\.sampling"
                       r"|APT\..+\.(slight_major|major))$")
EXPECT = {"A": 2398, "B": 5596}
FIELDS = ("id", "pair_id", "group", "label", "text", "generator", "domain")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out, ids, count = [], set(), Counter()
    for part, path, keep in (("A", args.a, lambda g: g in A_GROUPS), ("B", args.b, lambda g: bool(B_PATTERN.match(g)))):
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        for it in normalise_rows(rows):
            if not keep(it["group"]):
                continue
            if it["id"] in ids:
                raise SystemExit(f"duplicate id {it['id']}")
            ids.add(it["id"])
            out.append(json.dumps({k: it[k] for k in FIELDS}, ensure_ascii=False))
            count[part] += 1
    for part, n in EXPECT.items():
        if count[part] != n:
            raise SystemExit(f"part {part}: {count[part]} texts, expected {n}")
    Path(args.out).write_text("".join(x + "\n" for x in out), encoding="utf-8")
    print(f"wrote {len(out)} texts to {args.out} (A {count['A']}, B {count['B']})")


if __name__ == "__main__":
    main()
