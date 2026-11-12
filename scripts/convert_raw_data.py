"""Convert Fast-DetectGPT style `{dataset}_{generator}.raw_data.json` files into pairs.jsonl rows.

Used for the pilot on existing data (Fast-DetectGPT exp_gpt3to4 and the local continuations from
Study-AIText_Detection). Human texts there are from before 2021, so both DatedGPT vintages may have
seen them: the pilot is a quick read of the signal, not the clean test.

Example:
  python scripts/convert_raw_data.py --out data/pilot_pairs.jsonl \
      ../Study-AIText_Detection/data/fastdetectgpt/raw/*.raw_data.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for f in sorted(args.files):
        p = Path(f)
        stem = p.name.removesuffix(".raw_data.json")
        if re.search(r"\.limit\d+$", stem):  # debug subsets, e.g. xsum_qwen25-1.5b-instruct.limit8
            continue
        dataset, generator = stem.split("_", 1)
        d = json.loads(p.read_text(encoding="utf-8"))
        for i, (h, a) in enumerate(zip(d["original"], d["sampled"])):
            rows.append({"id": f"{dataset}-{i}", "human": h, "ai": a, "generator": generator, "domain": dataset})
        print(f"{p.name}: {len(d['original'])} pairs")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} pairs to {out}")


if __name__ == "__main__":
    main()
