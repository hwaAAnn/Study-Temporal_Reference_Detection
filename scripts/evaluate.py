"""Evaluate Delta on a scored file from score_texts.py.

Three reports:
  1. detection   AI (label 1) vs human (label 0) in the chosen groups: AUROC [95% CI], AUPRC,
                 TPR@1%/5% FPR, pair win rate, against ll_new / ll_old baselines
  2. groups      mean Delta per group (A/B/C/D) with 95% CI, Cohen's d of C vs B and D vs B
  3. DiD         (gap between AI and human under the post-AI pair) - (same gap under a pre-AI pair
                 of equal width). Positive and CI above 0 = the gap is not just general temporal change.

Example:
  python scripts/evaluate.py --scores outputs/k0.jsonl --new 2024 --old 2021 \
      --pre-new 2021 --pre-old 2018 --k 0
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from trd.metrics import (auprc, auroc, cluster_bootstrap_ci, cohens_d, pair_win_rate,
                         tpr_at_fpr)
from trd.scoring import delta_score


def deltas(recs, new, old, k):
    return np.array([delta_score(np.array(r["lp"][new]), np.array(r["lp"][old]), k) for r in recs])


def ci_str(lo, hi):
    return f"[{lo:.3f}, {hi:.3f}]"


def detection_report(recs, scores: dict, n_boot: int):
    labels = np.array([r["label"] for r in recs])
    clusters = np.array([r["pair_id"] for r in recs])
    print(f"\n## 1. detection  n={len(recs)} (AI={int(labels.sum())}, human={int((1 - labels).sum())})")
    print(f"{'score':<16}{'AUROC':>8} {'95% CI':<17}{'AUPRC':>7}{'TPR@1%':>8}{'TPR@5%':>8}{'pairwin':>9}")
    for name, s in scores.items():
        lo, hi = cluster_bootstrap_ci(clusters, lambda idx: auroc(labels[idx], s[idx]), n_boot)
        by_pair = defaultdict(dict)
        for r, v in zip(recs, s):
            by_pair[r["pair_id"]][r["label"]] = v
        pairs = [(d[1], d[0]) for d in by_pair.values() if 0 in d and 1 in d]
        win = pair_win_rate(*zip(*pairs)) if pairs else float("nan")
        print(f"{name:<16}{auroc(labels, s):>8.3f} {ci_str(lo, hi):<17}{auprc(labels, s):>7.3f}"
              f"{tpr_at_fpr(labels, s, 0.01):>8.3f}{tpr_at_fpr(labels, s, 0.05):>8.3f}{win:>9.3f}")


def group_report(recs, d, n_boot: int, title: str):
    groups = np.array([r["group"] for r in recs])
    clusters = np.array([r["pair_id"] for r in recs])
    print(f"\n## 2. groups ({title})")
    print(f"{'group':<7}{'n':>5}{'mean Δ':>10} {'95% CI':<17}")
    for g in sorted(set(groups)):
        m = groups == g
        lo, hi = cluster_bootstrap_ci(clusters[m], lambda idx, v=d[m]: float(v[idx].mean()), n_boot)
        print(f"{g:<7}{int(m.sum()):>5}{d[m].mean():>10.4f} {ci_str(lo, hi):<17}")
    if "B" in groups:
        for g in ("C", "D", "A"):
            if g in groups:
                print(f"Cohen's d ({g} vs B) = {cohens_d(d[groups == g], d[groups == 'B']):.3f}")


def did_report(recs, d_post, d_pre, n_boot: int, ai_group="C", human_group="B"):
    groups = np.array([r["group"] for r in recs])
    clusters = np.array([r["pair_id"] for r in recs])

    def gap(v, idx):
        g = groups[idx]
        a, h = v[idx][g == ai_group], v[idx][g == human_group]
        if len(a) == 0 or len(h) == 0:
            raise ValueError
        return float(a.mean() - h.mean())

    full = np.arange(len(recs))
    post, pre = gap(d_post, full), gap(d_pre, full)
    lo, hi = cluster_bootstrap_ci(clusters, lambda idx: gap(d_post, idx) - gap(d_pre, idx), n_boot)
    print(f"\n## 3. difference-in-differences ({ai_group} - {human_group})")
    print(f"gap under post-AI pair = {post:.4f}")
    print(f"gap under pre-AI pair  = {pre:.4f}")
    print(f"DiD = {post - pre:.4f}  95% CI {ci_str(lo, hi)}  (CI above 0 supports the AI-exposure hypothesis)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--new", default="2024")
    ap.add_argument("--old", default="2021")
    ap.add_argument("--pre-new", default=None, help="e.g. 2021, for the pre-AI pair of the DiD")
    ap.add_argument("--pre-old", default=None, help="e.g. 2018")
    ap.add_argument("--k", type=float, nargs="+", default=[0.0])
    ap.add_argument("--detect-groups", nargs="+", default=["B", "C"],
                    help="groups used for the detection table (default: matched B vs C)")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    recs = [json.loads(l) for l in Path(args.scores).read_text(encoding="utf-8").splitlines() if l.strip()]
    available = set(recs[0]["lp"])
    for key in (args.new, args.old, args.pre_new, args.pre_old):
        if key is not None and key not in available:
            raise SystemExit(f"model key {key!r} not in scores (have {sorted(available)})")

    det = [r for r in recs if r["group"] in set(args.detect_groups)]
    scores = {f"Δ{args.new}-{args.old}(k={k:g})": deltas(det, args.new, args.old, k) for k in args.k}
    scores["ll_new"] = np.array([np.mean(r["lp"][args.new]) for r in det])
    scores["ll_old"] = np.array([np.mean(r["lp"][args.old]) for r in det])
    detection_report(det, scores, args.n_boot)

    d0 = next(iter(scores.values()))
    r = np.corrcoef(d0, scores["ll_new"])[0, 1]
    print(f"corr(Δ, ll_new) = {r:.3f}   (|r| < 0.8: Δ carries information beyond plain likelihood)")

    d_post = deltas(recs, args.new, args.old, args.k[0])
    group_report(recs, d_post, args.n_boot, f"Δ{args.new}-{args.old}, k={args.k[0]:g}")
    if args.pre_new and args.pre_old:
        d_pre = deltas(recs, args.pre_new, args.pre_old, args.k[0])
        group_report(recs, d_pre, args.n_boot, f"Δ{args.pre_new}-{args.pre_old}, k={args.k[0]:g}")
        did_report(recs, d_post, d_pre, args.n_boot)


if __name__ == "__main__":
    main()
