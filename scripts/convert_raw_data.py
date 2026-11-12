"""Convert Fast-DetectGPT style `{dataset}_{generator}.raw_data.json` files into pair rows for score_texts.py.

Used for the pilot on existing data (Fast-DetectGPT exp_gpt3to4 and the local continuations from
Study-AIText_Detection). The human texts there were written before 2021, so they are group A (every
vintage >= 2021 may have seen them), not B. Generators released before ChatGPT (davinci = GPT-3) are
group D'; chat models are group C. Pair ids are unique per (dataset, generator, index), because each
raw file has its own human side.

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
    ap.add_argument("--human-group", default="A")
    ap.add_argument("--pre-chatgpt", nargs="*", default=["davinci"], help="generators that go to group D")
    args = ap.parse_args()

    rows = []
    for f in sorted(args.files):
        p = Path(f)
        stem = p.name.removesuffix(".raw_data.json")
        if re.search(r"\.limit\d+$", stem):  # debug subsets, e.g. xsum_qwen25-1.5b-instruct.limit8
            continue
        dataset, generator = stem.split("_", 1)
        ai_group = "D" if generator in args.pre_chatgpt else "C"
        d = json.loads(p.read_text(encoding="utf-8"))
        for i, (h, a) in enumerate(zip(d["original"], d["sampled"])):
            rows.append({"id": f"{dataset}-{generator}-{i}", "human": h, "ai": a, "generator": generator,
                         "domain": dataset, "human_group": args.human_group, "ai_group": ai_group})
        print(f"{p.name}: {len(d['original'])} pairs (human {args.human_group}, AI {ai_group})")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} pairs to {out}")


if __name__ == "__main__":
    main()
