"""Experiment 17: candidate papers for the literature audit, from one fixed arXiv API query, in a fixed random order.

The query (QUERY below) and the date window were fixed in docs/experiments/17-literature-audit.md before any paper
was read. Candidates are deduplicated by arXiv id (version dropped), kept if first submitted in the window, and
shuffled with seed 0. Coding then goes down this order (screening rule in the doc).

Output JSONL: {"rank", "id", "title", "published", "summary"}

Example:
  python -I scripts/fetch_17_candidates.py --out docs/experiments/17-candidates.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

API = "https://export.arxiv.org/api/query"
QUERY = ('abs:"zero-shot" AND abs:detect* AND (abs:"machine-generated" OR abs:"AI-generated" OR abs:"LLM-generated" '
         'OR abs:"machine generated" OR abs:"AI generated" OR abs:"LLM generated")')
WINDOW = ("2023-01-01", "2026-10-31")
NS = {"a": "http://www.w3.org/2005/Atom"}


def page(start: int, n: int) -> list[dict]:
    url = API + "?" + urllib.parse.urlencode({"search_query": QUERY, "start": start, "max_results": n,
                                              "sortBy": "submittedDate", "sortOrder": "ascending"})
    req = urllib.request.Request(url, headers={"User-Agent": "trd-research-script/0.1"})
    for wait in (3, 10, 30, 60):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                root = ET.fromstring(r.read())
            break
        except Exception as e:  # arXiv rate limits: back off and retry
            print(f"retry after {wait}s: {e}", flush=True)
            time.sleep(wait)
    else:
        raise SystemExit("arXiv API failed")
    out = []
    for e in root.findall("a:entry", NS):
        out.append({"id": e.find("a:id", NS).text.rsplit("/abs/", 1)[-1].split("v")[0],
                    "title": " ".join(e.find("a:title", NS).text.split()),
                    "published": e.find("a:published", NS).text[:10],
                    "summary": " ".join(e.find("a:summary", NS).text.split())[:400]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows, start = {}, 0
    while True:
        got = page(start, 200)
        for r in got:
            rows.setdefault(r["id"], r)
        print(f"start {start}: {len(got)} entries, {len(rows)} unique", flush=True)
        if len(got) < 200:
            break
        start += 200
        time.sleep(3)
    keep = sorted((r for r in rows.values() if WINDOW[0] <= r["published"] <= WINDOW[1]), key=lambda r: r["id"])
    random.Random(args.seed).shuffle(keep)
    Path(args.out).write_text("".join(json.dumps({"rank": k + 1, **r}, ensure_ascii=False) + "\n"
                                      for k, r in enumerate(keep)), encoding="utf-8")
    print(f"{len(keep)} candidates in {WINDOW} -> {args.out}")


if __name__ == "__main__":
    main()
