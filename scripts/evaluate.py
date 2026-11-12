"""Compare Delta with single-model baselines on a scored file.

Scores (higher = more AI-like):
  delta   mean(lp_new - lp_old) over kept tokens
  ll_new  mean lp_new  (classic log-likelihood baseline)
  ll_old  mean lp_old

Example:
  python scripts/evaluate.py --scores outputs/k0.jsonl --k 0 0.05 0.1 0.2
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from trd.metrics import auroc, pair_win_rate, tpr_at_fpr
from trd.scoring import delta_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--k", type=float, nargs="+", default=[0.0])
    args = ap.parse_args()

    recs = [json.loads(l) for l in Path(args.scores).read_text(encoding="utf-8").splitlines() if l.strip()]
    labels = np.array([r["label"] for r in recs])
    lp_old = [np.array(r["lp_old"]) for r in recs]
    lp_new = [np.array(r["lp_new"]) for r in recs]

    table = {}
    for k in args.k:
        table[f"delta(k={k:g})"] = np.array([delta_score(n, o, k) for n, o in zip(lp_new, lp_old)])
    table["ll_new"] = np.array([n.mean() for n in lp_new])
    table["ll_old"] = np.array([o.mean() for o in lp_old])

    print(f"n={len(recs)} (AI={int(labels.sum())}, human={int((1 - labels).sum())})")
    print(f"{'score':<16}{'AUROC':>8}{'TPR@1%':>9}{'pairwin':>9}")
    for name, s in table.items():
        by_id = defaultdict(dict)
        for r, v in zip(recs, s):
            by_id[(r["id"], r["generator"])][r["label"]] = v
        pairs = [(d[1], d[0]) for d in by_id.values() if 0 in d and 1 in d]
        win = pair_win_rate([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else float("nan")
        print(f"{name:<16}{auroc(labels, s):>8.3f}{tpr_at_fpr(labels, s):>9.3f}{win:>9.3f}")

    d0 = table[f"delta(k={args.k[0]:g})"]
    r = np.corrcoef(d0, table["ll_new"])[0, 1]
    print(f"corr(delta, ll_new) = {r:.3f}   (H3: |r| < 0.8 means Delta carries new information)")


if __name__ == "__main__":
    main()
