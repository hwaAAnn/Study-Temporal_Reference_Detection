"""Was a paper's arXiv abstract page crawled into FineWeb-Edu? (actual-membership check for experiment 02)

DatedGPT trains on FineWeb-Edu documents split by crawl year, so an abstract can only be in the
2024 model's data if its page is in a FineWeb-Edu dump crawled in or before 2024. This reads only the
`url` column of the chosen dumps (parquet range requests, no text download) and records every
arxiv.org/abs/<id> page found, with the dump it came from.

Presence here is a lower bound on exposure (the same abstract can also appear on other sites) and an
upper bound on training (DatedGPT samples ~100B tokens from what is eligible).

Example:
  python scripts/fineweb_presence.py --dumps CC-MAIN-2022-49 CC-MAIN-2024-10 --out data/fineweb_arxiv_abs.jsonl
  python scripts/fineweb_presence.py --dumps CC-MAIN-2022-49 --max-files 1        # quick size/time check
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

REPO = "datasets/HuggingFaceFW/fineweb-edu"
ABS = re.compile(r"arxiv\.org/abs/(\d{4}\.\d{4,5})(?:v\d+)?", re.I)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="+", required=True, help="e.g. CC-MAIN-2022-49")
    ap.add_argument("--out", default="data/fineweb_arxiv_abs.jsonl")
    ap.add_argument("--max-files", type=int, default=None, help="only the first N parquet files per dump")
    args = ap.parse_args()

    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():  # resume: skip files already scanned
        for l in out.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            if r.get("scanned_file"):
                done.add(r["scanned_file"])
    with out.open("a", encoding="utf-8") as f:
        for dump in args.dumps:
            files = sorted(fs.glob(f"{REPO}/data/{dump}/*.parquet"))[: args.max_files]
            for path in files:
                if path in done:
                    continue
                t0 = time.time()
                pf = pq.ParquetFile(fs.open(path, block_size=8 * 2**20))
                n_rows, hits = 0, 0
                for rg in range(pf.num_row_groups):
                    urls = pf.read_row_group(rg, columns=["url"]).column("url")
                    n_rows += len(urls)
                    mask = pc.match_substring_regex(urls, r"arxiv\.org/abs/", ignore_case=True)
                    for u in pc.filter(urls, mask).to_pylist():
                        m = ABS.search(u)
                        if m:
                            hits += 1
                            f.write(json.dumps({"arxiv_id": m.group(1), "url": u, "dump": dump}) + "\n")
                f.write(json.dumps({"scanned_file": path, "dump": dump, "rows": n_rows, "abs_pages": hits}) + "\n")
                f.flush()
                print(f"{path.rsplit('/', 1)[-1]} ({dump}): rows={n_rows} abs_pages={hits} {time.time() - t0:.0f}s",
                      flush=True)


if __name__ == "__main__":
    main()
