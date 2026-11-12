"""Sample human web documents straight from a FineWeb-Edu dump (experiment 02-b).

Documents taken from FineWeb-Edu are, by construction, in the pool DatedGPT trained on for every
vintage whose crawl window includes the dump. A date path in the URL (e.g. /2021/) tells roughly when
the page was first published, so it cannot be in pools crawled before that year.

Row groups of one parquet file are visited in random order; text is read only for row groups whose
URLs match, and the scan stops once --oversample x --n documents are found.

On Git Bash for Windows set MSYS_NO_PATHCONV=1, or a regex such as /2022/ is rewritten into a path.

Example (pool of the 2021 model, not of the 2018 one; pre-ChatGPT):
  python scripts/sample_fineweb.py --dump CC-MAIN-2021-49 --url-regex "/202[01]/" --n 450 --out data/fw_2021.jsonl
Example (in neither the 2018 nor the 2021 pool; still pre-ChatGPT):
  python scripts/sample_fineweb.py --dump CC-MAIN-2022-40 --url-regex "/2022/" --n 450 --out data/fw_2022.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

REPO = "datasets/HuggingFaceFW/fineweb-edu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--url-regex", required=True, help="date path the URL must contain, e.g. /2022/")
    ap.add_argument("--n", type=int, default=450)
    ap.add_argument("--file-index", type=int, default=0, help="which parquet file of the dump to use")
    ap.add_argument("--min-words", type=int, default=250, help="enough text for 200 scored tokens")
    ap.add_argument("--oversample", type=float, default=2.0, help="collect this many times --n, then sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    path = sorted(fs.glob(f"{REPO}/data/{args.dump}/*.parquet"))[args.file_index]
    pf = pq.ParquetFile(fs.open(path, block_size=8 * 2**20))
    t0 = time.time()
    rng = random.Random(args.seed)
    order = list(range(pf.num_row_groups))
    rng.shuffle(order)  # visit row groups in random order and stop once enough documents are found
    rows, scanned, n_match = [], 0, 0
    for rg in order:
        urls = pf.read_row_group(rg, columns=["url"]).column("url")
        scanned += 1
        hit = pc.indices_nonzero(pc.match_substring_regex(urls, args.url_regex)).to_pylist()
        if not hit:
            continue
        n_match += len(hit)
        t = pf.read_row_group(rg, columns=["id", "url", "dump", "text", "score", "token_count"]).to_pylist()
        for i in hit:
            r = t[i]
            text = re.sub(r"[ \t]+", " ", r["text"]).strip()
            if len(text.split()) >= args.min_words:
                rows.append({"id": r["id"], "url": r["url"], "dump": r["dump"], "edu_score": r["score"],
                             "token_count": r["token_count"], "text": text})
        if len(rows) >= args.oversample * args.n:
            break
    print(f"{path.rsplit('/', 1)[-1]}: scanned {scanned}/{pf.num_row_groups} row groups, {n_match} URLs match "
          f"{args.url_regex!r}, {len(rows)} with >= {args.min_words} words ({time.time() - t0:.0f}s)", flush=True)
    rng.shuffle(rows)
    rows = rows[: args.n]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} documents to {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
