"""Check that GPU fp16 scoring matches CPU fp32 for a DatedGPT pair (trained in bf16; Pascal GPUs lack bf16).

Scores the first --n texts of a pairs file with both precisions and reports the largest per-token
log-prob difference and the largest Delta difference.

Example:
  python scripts/check_precision.py --pairs data/pilot_pairs.jsonl --n 16
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from trd.scoring import delta_score, token_logprobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--old", default="datedgpt/datedgpt-2021-base")
    ap.add_argument("--new", default="datedgpt/datedgpt-2024-base")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=200)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.pairs).read_text(encoding="utf-8").splitlines()[: (args.n + 1) // 2]]
    texts = [t for r in rows for t in (r["human"], r["ai"])][: args.n]
    lp = {}
    for prec, dev in (("float16", "cuda"), ("float32", "cpu")):
        for m in (args.old, args.new):
            t0 = time.time()
            lp[prec, m], _ = token_logprobs(texts, m, args.max_tokens, prec, device=dev, batch_size=4)
            print(f"{m} {prec} on {dev}: {time.time() - t0:.1f}s for {len(texts)} texts", flush=True)

    for m in (args.old, args.new):
        d = max(np.abs(a - b).max() for a, b in zip(lp["float16", m], lp["float32", m]))
        mean_d = max(abs(a.mean() - b.mean()) for a, b in zip(lp["float16", m], lp["float32", m]))
        print(f"{m}: max |token lp diff| = {d:.4f}, max |mean lp diff| = {mean_d:.5f}")
    d16 = np.array([delta_score(n, o) for n, o in zip(lp["float16", args.new], lp["float16", args.old])])
    d32 = np.array([delta_score(n, o) for n, o in zip(lp["float32", args.new], lp["float32", args.old])])
    print(f"Delta: max |fp16 - fp32| = {np.abs(d16 - d32).max():.5f}; Delta range {d32.min():+.3f} .. {d32.max():+.3f}")
    print(f"Spearman-free sanity: corr(fp16, fp32) = {np.corrcoef(d16, d32)[0, 1]:.6f}")


if __name__ == "__main__":
    main()
