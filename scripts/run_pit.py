"""Experiment 06: score texts with every PIT vintage (1B and 4B, 2013-2024) on a cloud GPU.

One score_texts.py run per (size, vintage), each pinned to the commit in pit_revisions.tsv. Downloads
run ahead of scoring in a background thread (--ahead snapshots), and a snapshot is deleted from the HF
cache once its vintage is scored, so disk holds at most ahead + 1 models. Every vintage writes its own
file (outputs/pit1b_2013.jsonl ...), and finished files are skipped on a rerun.

Example (on the pod):
  python scripts/run_pit.py --sizes 1B 4B --dtype bfloat16
  # precision check: the 2022 vintages on the first 800 texts, both precisions, snapshots kept
  python scripts/run_pit.py --years 2022 --limit 800 --dtype float32 --tag _chk32 --keep
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

BATCH = {"1B": 32, "4B": 16}


def read_revisions(path: str) -> list[tuple[str, int, str, str]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            size, year, repo, sha = line.split("\t")
            rows.append((size, int(year), repo, sha))
    return rows


def fetch(repo: str, sha: str) -> str:
    from huggingface_hub import snapshot_download

    t0 = time.time()
    path = snapshot_download(repo, revision=sha)
    gb = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e9
    print(f"[fetch] {repo}@{sha[:8]} {gb:.1f} GB in {time.time() - t0:.0f}s", flush=True)
    return path


def drop(snapshot: str) -> None:
    repo_dir = Path(snapshot).parents[1]  # .../models--Diamegs--PIT-1B-201312
    if repo_dir.name.startswith("models--"):
        shutil.rmtree(repo_dir, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revisions", default="scripts/pit_revisions.tsv")
    ap.add_argument("--input", default="data/01_all.jsonl")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--sizes", nargs="+", default=["1B", "4B"])
    ap.add_argument("--years", nargs="+", type=int, default=list(range(2013, 2025)))
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--ahead", type=int, default=2, help="snapshots downloaded ahead of the one being scored")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N texts")
    ap.add_argument("--tag", default="", help="suffix of the output names")
    ap.add_argument("--keep", action="store_true", help="keep snapshots after scoring")
    args = ap.parse_args()

    jobs = []
    for size, year, repo, sha in read_revisions(args.revisions):
        if size not in args.sizes or year not in args.years:
            continue
        out = Path(args.out_dir) / f"pit{size.lower()}_{year}{args.tag}.jsonl"
        if out.exists() and out.stat().st_size > 0:
            print(f"[skip] {out}", flush=True)
            continue
        jobs.append((size, year, repo, sha, out))
    jobs.sort(key=lambda j: (args.sizes.index(j[0]), j[1]))
    if not jobs:
        print("nothing to do")
        return

    pool = ThreadPoolExecutor(max_workers=1)
    futures: dict[int, Future] = {}

    def ensure(k: int) -> None:
        if k < len(jobs) and k not in futures:
            futures[k] = pool.submit(fetch, jobs[k][2], jobs[k][3])

    t_all = time.time()
    for k, (size, year, repo, sha, out) in enumerate(jobs):
        for a in range(k, k + args.ahead + 1):
            ensure(a)
        snapshot = futures[k].result()
        tmp = out.with_name(out.name + ".tmp")
        cmd = [sys.executable, "scripts/score_texts.py", "--input", args.input, "--out", str(tmp),
               "--models", f"{repo}@{sha}", "--dtype", args.dtype, "--batch-size", str(BATCH[size]),
               "--no-pair-truncate", "--trust-remote-code", "--prepend-bos"]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        t0 = time.time()
        print(f"[score] {size} {year} {args.dtype} -> {out}", flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            raise SystemExit(f"score_texts failed for {repo} (exit {r.returncode})")
        tmp.replace(out)
        print(f"[done] {size} {year} in {time.time() - t0:.0f}s (total {(time.time() - t_all) / 60:.1f} min)", flush=True)
        if not args.keep:
            drop(snapshot)
    pool.shutdown()
    print(f"[all done] {len(jobs)} vintages in {(time.time() - t_all) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
