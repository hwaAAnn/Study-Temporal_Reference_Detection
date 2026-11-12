"""Collect human-written arXiv cs abstracts submitted after both models' data cutoffs.

For each month in [--start, --end] we read a few random windows of the arXiv API listing (sorted by
submission date), keep papers whose *primary* category is cs.*, and sample --per-month of them.
The submittedDate filter refers to the first version, so every abstract was first posted after 2024.

Example:
  python scripts/fetch_arxiv.py --start 2025-07 --end 2026-10 --per-month 40 --out data/human_arxiv.jsonl
"""

from __future__ import annotations

import argparse
import calendar
import json
import random
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom",
      "os": "http://a9.com/-/spec/opensearch/1.1/"}
DELAY = 3.5  # arXiv asks for at most one request every 3 seconds


def query(month: str, start: int, n: int, order: str = "ascending") -> ET.Element | None:
    """One API page, retried with exponential backoff (10 s, 20 s, ... ). None if every try fails.

    The API returns HTTP 500 for start >= 10000, so callers keep start below that (see window()).
    """
    y, m = map(int, month.split("-"))
    last = calendar.monthrange(y, m)[1]
    q = f"cat:cs.* AND submittedDate:[{y}{m:02d}010000 TO {y}{m:02d}{last}2359]"
    url = API + "?" + urllib.parse.urlencode({"search_query": q, "start": start, "max_results": n,
                                               "sortBy": "submittedDate", "sortOrder": order})
    req = urllib.request.Request(url, headers={"User-Agent": "trd-research-script/0.1"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                root = ET.fromstring(r.read())
            if root.findall("a:entry", NS) or start == 0:
                return root
            reason = "empty page"
        except Exception as e:  # network hiccup or API throttling (429 / 503)
            reason = f"{type(e).__name__} {getattr(e, 'code', '')}".strip()
        wait = 10 * 2 ** attempt
        print(f"  {month} start={start}: {reason}, retry in {wait}s", flush=True)
        time.sleep(wait)
    return None


def window(pos: int, total: int, size: int) -> tuple[int, str]:
    """(start, order) that reads listing positions [pos, pos + size) without a deep offset.

    Windows in the second half of the month are read from the end (descending order), so the offset
    stays below total / 2 and the whole month is still reachable.
    """
    cap = 10000 - size  # months with more than ~20k papers: the middle of the listing is read a bit more often
    if pos < total // 2:
        return min(pos, cap), "ascending"
    return min(max(0, total - size - pos), cap), "descending"


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def keep(title: str, abstract: str, primary: str) -> bool:
    words = len(abstract.split())
    return (primary.startswith("cs.") and 150 <= words <= 400 and abstract.count("$") <= 4
            and "withdraw" not in abstract.lower() and len(title.split()) >= 4)


def months(start: str, end: str) -> list[str]:
    y, m = map(int, start.split("-"))
    ye, me = map(int, end.split("-"))
    out = []
    while (y, m) <= (ye, me):
        out.append(f"{y}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-07")
    ap.add_argument("--end", default="2026-10")
    ap.add_argument("--per-month", type=int, default=40)
    ap.add_argument("--windows", type=int, default=2, help="random listing windows read per month")
    ap.add_argument("--window-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/human_arxiv.jsonl")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()] if out.exists() else []
    done_months = {r["month"] for r in done}
    seen = {r["id"] for r in done}
    if done:
        print(f"resuming: {len(done)} abstracts already in {out} ({len(done_months)} months)")

    for month in months(args.start, args.end):
        rng = random.Random(f"{args.seed}-{month}")  # per-month seed, so resuming gives the same sample
        if month in done_months:
            continue
        head = query(month, 0, 1)
        if head is None:
            print(f"{month}: skipped (API unavailable)")
            continue
        total = int(head.find("os:totalResults", NS).text)
        time.sleep(DELAY)
        cands = []
        for _ in range(args.windows):
            start, order = window(rng.randrange(0, max(1, total - args.window_size)), total, args.window_size)
            root = query(month, start, args.window_size, order)
            time.sleep(DELAY)
            if root is None:
                continue
            for e in root.findall("a:entry", NS):
                aid = e.find("a:id", NS).text.rsplit("/abs/", 1)[-1]
                base = re.sub(r"v\d+$", "", aid)
                title = clean(e.find("a:title", NS).text or "")
                abstract = clean(e.find("a:summary", NS).text or "")
                prim = e.find("arxiv:primary_category", NS)
                primary = prim.get("term") if prim is not None else ""
                if base in seen or not keep(title, abstract, primary):
                    continue
                seen.add(base)
                cands.append({"id": base, "title": title, "abstract": abstract, "primary_category": primary,
                              "published": e.find("a:published", NS).text, "month": month})
        picked = rng.sample(cands, min(args.per_month, len(cands)))
        with out.open("a", encoding="utf-8") as f:  # saved month by month, so a crash loses at most one month
            for r in picked:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        done.extend(picked)
        print(f"{month}: total={total} candidates={len(cands)} picked={len(picked)}", flush=True)
    print(f"{len(done)} abstracts in {out}")


if __name__ == "__main__":
    main()
