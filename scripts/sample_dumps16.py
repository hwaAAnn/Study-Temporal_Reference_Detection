"""Experiment 16: per-document marker counts in random row groups of every FineWeb(-Edu) crawl year.

For each year, --rg-per-year row groups (1,000 documents each) are drawn round-robin over that year's Common
Crawl dumps; within a dump the parquet file and the row group are random (seed). Only counts are kept, never the
text. Measures (fixed before any count was seen, docs/experiments/16-crawl-ai-share.md):
  mark   05's 18 marker stems (polish_calibration.MARK)
  rare   the 9 rarer of them: delve intricate showcas pivotal meticulous commendable notably realm underscore
  phrase ChatGPT signature phrases (PHRASES below)
Documents with a phrase hit also go to --examples (url + first 300 characters, at most 300 rows) for reading.

Example:
  python scripts/sample_dumps16.py --repo fineweb-edu --out outputs/s16_edu.jsonl --examples outputs/s16_edu_examples.jsonl
  python scripts/sample_dumps16.py --repo fineweb --out outputs/s16_fw.jsonl --examples outputs/s16_fw_examples.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from polish_calibration import MARK  # noqa: E402

RARE = ["delve", "intricate", "showcas", "pivotal", "meticulous", "commendable", "notably", "realm", "underscore"]
PHRASES = ["delve into", "delves into", "delving into", "a testament to", "in today's fast-paced",
           "navigating the complexities", "navigate the complexities", "plays a crucial role", "play a crucial role",
           "in the realm of", "embark on a journey", "it's important to note", "it is important to note",
           "unlock the potential", "rich tapestry", "a pivotal role", "as an ai language model", "foster a sense of"]
MARK_RE = re.compile(r"\b(" + "|".join(MARK) + r")", re.I)
RARE_RE = re.compile(r"\b(" + "|".join(RARE) + r")", re.I)
PHRASE_RE = re.compile(r"\b(" + "|".join(re.escape(p) for p in PHRASES) + r")\b", re.I)


def counts(text: str) -> dict:
    t = text.replace("’", "'")
    return {"words": len(t.split()), "mark": len(MARK_RE.findall(t)), "rare": len(RARE_RE.findall(t)),
            "phrase": len(PHRASE_RE.findall(t))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", choices=["fineweb-edu", "fineweb"], required=True)
    ap.add_argument("--years", nargs="+", type=int, default=list(range(2013, 2026)))
    ap.add_argument("--rg-per-year", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--examples", required=True)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    repo = f"datasets/HuggingFaceFW/{args.repo}"
    dumps = sorted(p.split("/")[-1] for p in fs.ls(f"{repo}/data", detail=False))
    rng = random.Random(args.seed)
    out = open(args.out, "w", encoding="utf-8")
    ex = open(args.examples, "w", encoding="utf-8")
    n_ex = 0
    t0 = time.time()
    for y in args.years:
        ydumps = [d for d in dumps if d.startswith(f"CC-MAIN-{y}-")]
        if not ydumps:
            print(f"{y}: no dumps", flush=True)
            continue
        files = {d: sorted(fs.glob(f"{repo}/data/{d}/*.parquet")) for d in ydumps}
        taken: set = set()
        for k in range(args.rg_per_year):
            d = ydumps[k % len(ydumps)]
            for _ in range(20):
                fi = rng.randrange(len(files[d]))
                pf = pq.ParquetFile(fs.open(files[d][fi], block_size=8 * 2**20))
                rg = rng.randrange(pf.num_row_groups)
                if (d, fi, rg) not in taken:
                    break
            taken.add((d, fi, rg))
            for attempt in range(3):
                try:
                    tab = pf.read_row_group(rg, columns=["text", "url", "id"]).to_pylist()
                    break
                except Exception as e:  # network hiccup: retry the same row group
                    print(f"  retry {d} file {fi} rg {rg}: {e}", flush=True)
                    time.sleep(5)
            else:
                raise SystemExit(f"could not read {d} file {fi} rg {rg}")
            for r in tab:
                c = counts(r["text"])
                out.write(json.dumps({"repo": args.repo, "year": y, "dump": d, "file": fi, "rg": rg, "id": r["id"], **c})
                          + "\n")
                if c["phrase"] and n_ex < 300 and rng.random() < 0.2:
                    ex.write(json.dumps({"year": y, "dump": d, "url": r["url"], "head": r["text"][:300]},
                                        ensure_ascii=False) + "\n")
                    n_ex += 1
        print(f"{y}: {len(ydumps)} dumps, {args.rg_per_year} row groups ({time.time() - t0:.0f}s)", flush=True)
    out.close()
    ex.close()
    print("done")


if __name__ == "__main__":
    main()
