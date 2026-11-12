"""Evaluate Delta on a scored file from score_texts.py.

Three reports:
  1. detection   AI (label 1) vs human (label 0) in the chosen groups: AUROC [95% CI], AUPRC,
                 TPR@1%/5% FPR, pair win rate, against ll_new / ll_old baselines
                 (pooled, then per generator with --by-generator)
  2. groups      mean Delta per group (A/B/C/D) with 95% CI, Cohen's d of C vs B and D vs B
  3. DiD         (gap between AI and human under the post-AI pair) - (same gap under a pre-AI pair
                 of equal width). Positive and CI above 0 = the gap is not just general temporal change.
                 Reported for C vs B (pooled and per generator) and D vs B.

Ends with the H1-H4 verdicts of experiment 00; --json writes every number.

Example:
  python scripts/evaluate.py --scores outputs/k0.jsonl --new 2024 --old 2021 \
      --pre-new 2021 --pre-old 2018 --k 0 --by-generator --json outputs/k0_eval.json
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


def detection_report(recs, scores: dict, n_boot: int, title: str = "") -> dict:
    labels = np.array([r["label"] for r in recs])
    clusters = np.array([r["pair_id"] for r in recs])
    print(f"\n## 1. detection{title}  n={len(recs)} (AI={int(labels.sum())}, human={int((1 - labels).sum())})")
    print(f"{'score':<16}{'AUROC':>8} {'95% CI':<17}{'AUPRC':>7}{'TPR@1%':>8}{'TPR@5%':>8}{'pairwin':>9}")
    out = {}
    for name, s in scores.items():
        lo, hi = cluster_bootstrap_ci(clusters, lambda idx: auroc(labels[idx], s[idx]), n_boot)
        by_pair = defaultdict(dict)
        for r, v in zip(recs, s):
            by_pair[r["pair_id"]][r["label"]] = v
        pairs = [(d[1], d[0]) for d in by_pair.values() if 0 in d and 1 in d]
        win = pair_win_rate(*zip(*pairs)) if pairs else float("nan")
        m = {"auroc": auroc(labels, s), "auroc_ci": [lo, hi], "auprc": auprc(labels, s),
             "tpr@1%": tpr_at_fpr(labels, s, 0.01), "tpr@5%": tpr_at_fpr(labels, s, 0.05),
             "pair_win": win, "n_pairs": len(pairs)}
        print(f"{name:<16}{m['auroc']:>8.3f} {ci_str(lo, hi):<17}{m['auprc']:>7.3f}"
              f"{m['tpr@1%']:>8.3f}{m['tpr@5%']:>8.3f}{win:>9.3f}")
        out[name] = m
    return out


def group_report(recs, d, n_boot: int, title: str, human_group: str = "B") -> dict:
    groups = np.array([r["group"] for r in recs])
    clusters = np.array([r["pair_id"] for r in recs])
    print(f"\n## 2. groups ({title})")
    print(f"{'group':<7}{'n':>5}{'mean Δ':>10} {'95% CI':<17}")
    out = {"mean": {}, "cohens_d_vs_human": {}, "human_group": human_group}
    for g in sorted(set(groups)):
        m = groups == g
        lo, hi = cluster_bootstrap_ci(clusters[m], lambda idx, v=d[m]: float(v[idx].mean()), n_boot)
        print(f"{g:<7}{int(m.sum()):>5}{d[m].mean():>10.4f} {ci_str(lo, hi):<17}")
        out["mean"][g] = {"n": int(m.sum()), "mean": float(d[m].mean()), "ci": [lo, hi]}
    if human_group in groups:
        for g in ("C", "D", "A", "B"):
            if g in groups and g != human_group:
                cd = cohens_d(d[groups == g], d[groups == human_group])
                print(f"Cohen's d ({g} vs {human_group}) = {cd:.3f}")
                out["cohens_d_vs_human"][g] = cd
    return out


def did_report(recs, d_post, d_pre, n_boot: int, ai_group="C", human_group="B", title="") -> dict | None:
    groups = np.array([r["group"] for r in recs])
    clusters = np.array([r["pair_id"] for r in recs])
    if not ((groups == ai_group).any() and (groups == human_group).any()):
        return None

    def gap(v, idx):
        g = groups[idx]
        a, h = v[idx][g == ai_group], v[idx][g == human_group]
        if len(a) == 0 or len(h) == 0:
            raise ValueError
        return float(a.mean() - h.mean())

    full = np.arange(len(recs))
    post, pre = gap(d_post, full), gap(d_pre, full)
    lo, hi = cluster_bootstrap_ci(clusters, lambda idx: gap(d_post, idx) - gap(d_pre, idx), n_boot)
    print(f"\n## 3. difference-in-differences ({ai_group} - {human_group}){title}")
    print(f"gap under post-AI pair = {post:.4f}")
    print(f"gap under pre-AI pair  = {pre:.4f}")
    print(f"DiD = {post - pre:.4f}  95% CI {ci_str(lo, hi)}  (CI above 0 supports the AI-exposure hypothesis)")
    return {"gap_post": post, "gap_pre": pre, "did": post - pre, "ci": [lo, hi]}


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
    ap.add_argument("--human-group", default="B", help="human reference group for Cohen's d and DiD (pilot: A)")
    ap.add_argument("--generators", nargs="*", default=None,
                    help="keep only records of these generators (their pairs' human texts included)")
    ap.add_argument("--by-generator", action="store_true", help="repeat detection and DiD per AI generator")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--json", default=None, help="also write all numbers here")
    args = ap.parse_args()

    recs = [json.loads(l) for l in Path(args.scores).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.generators:
        recs = [r for r in recs if r["generator"] in set(args.generators)]
    available = set(recs[0]["lp"])
    for key in (args.new, args.old, args.pre_new, args.pre_old):
        if key is not None and key not in available:
            raise SystemExit(f"model key {key!r} not in scores (have {sorted(available)})")
    result = {"scores_file": args.scores, "new": args.new, "old": args.old,
              "pre_new": args.pre_new, "pre_old": args.pre_old, "k": args.k}

    det_groups = set(args.detect_groups)
    det = [r for r in recs if r["group"] in det_groups]

    def score_table(rs):
        t = {f"Δ{args.new}-{args.old}(k={k:g})": deltas(rs, args.new, args.old, k) for k in args.k}
        t["ll_new"] = np.array([np.mean(r["lp"][args.new]) for r in rs])
        t["ll_old"] = np.array([np.mean(r["lp"][args.old]) for r in rs])
        return t

    scores = score_table(det)
    result["detection"] = detection_report(det, scores, args.n_boot)
    d0 = next(iter(scores.values()))
    lab = np.array([r["label"] for r in det])
    r_all = float(np.corrcoef(d0, scores["ll_new"])[0, 1])
    r_in = {int(c): float(np.corrcoef(d0[lab == c], scores["ll_new"][lab == c])[0, 1]) for c in (0, 1)}
    print(f"corr(Δ, ll_new) = {r_all:.3f}   (|r| < 0.8: Δ carries information beyond plain likelihood)"
          f"   within human {r_in[0]:+.3f}, within AI {r_in[1]:+.3f}")
    result["corr_delta_llnew"] = {"pooled": r_all, "within_human": r_in[0], "within_ai": r_in[1]}

    gens = sorted({r["generator"] for r in det if r["label"] == 1})
    if args.by_generator and len(gens) > 1:
        result["detection_by_generator"] = {}
        for g in gens:
            sub = [r for r in det if r["generator"] == g]
            result["detection_by_generator"][g] = detection_report(sub, score_table(sub), args.n_boot, f" [{g}]")

    d_post = deltas(recs, args.new, args.old, args.k[0])
    result["groups_post"] = group_report(recs, d_post, args.n_boot, f"Δ{args.new}-{args.old}, k={args.k[0]:g}", args.human_group)
    if args.pre_new and args.pre_old:
        d_pre = deltas(recs, args.pre_new, args.pre_old, args.k[0])
        result["groups_pre"] = group_report(recs, d_pre, args.n_boot, f"Δ{args.pre_new}-{args.pre_old}, k={args.k[0]:g}", args.human_group)
        hg = args.human_group
        result["did"] = {"C": did_report(recs, d_post, d_pre, args.n_boot, human_group=hg)}
        if args.by_generator and len(gens) > 1:
            for g in gens:
                # this generator's C texts against the B texts they were paired with (same topics)
                idx = [i for i, r in enumerate(recs) if r["group"] in (hg, "C") and r["generator"] == g]
                sub = [recs[i] for i in idx]
                result["did"][f"C[{g}]"] = did_report(sub, d_post[idx], d_pre[idx], args.n_boot, human_group=hg, title=f" [{g}]")
        result["did"]["D"] = did_report(recs, d_post, d_pre, args.n_boot, ai_group="D", human_group=hg)

    # verdicts of experiment 00 (docs/experiments/00-k0-signal.md)
    m = result["detection"][next(iter(scores))]
    cd = result["groups_post"]["cohens_d_vs_human"].get("C", float("nan"))
    print("\n## verdicts (experiment 00)")
    v = {"H1": bool(m["pair_win"] > 0.5 and cd > 0), "H2": bool(m["auroc"] >= 0.70 and m["auroc_ci"][0] > 0.5),
         "H3": bool(abs(r_all) < 0.8)}
    print(f"H1 C > {args.human_group}: pair win {m['pair_win']:.3f} > 0.5 and Cohen's d {cd:.3f} > 0          -> {'yes' if v['H1'] else 'no'}")
    print(f"H2 AUROC {m['auroc']:.3f} >= 0.70 and CI low {m['auroc_ci'][0]:.3f} > 0.5            -> {'yes' if v['H2'] else 'no'}")
    print(f"H3 |corr(Δ, ll_new)| = {abs(r_all):.3f} < 0.8                               -> {'yes' if v['H3'] else 'no'}")
    did_c = result.get("did", {}).get("C")
    if did_c:
        v["H4"] = bool(did_c["did"] > 0 and did_c["ci"][0] > 0)
        print(f"H4 DiD {did_c['did']:.4f} > 0 and CI low {did_c['ci'][0]:.4f} > 0                  -> {'yes' if v['H4'] else 'no'}")
    result["verdicts"] = v

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
