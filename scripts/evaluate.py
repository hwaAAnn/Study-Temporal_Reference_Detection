"""Compare Delta with single-model baselines on a scored file.

Scores (higher = more AI-like):
  delta   mean(lp_new - lp_old) over kept tokens
  ll_new  mean lp_new  (classic log-likelihood baseline)
  ll_old  mean lp_old

Prints the pooled table with pair-bootstrap 95% CIs, a breakdown per group (--by), and the H1-H3
verdicts of experiment 00. --json writes the same numbers for the experiment log.

Example:
  python scripts/evaluate.py --scores outputs/k0.jsonl --by generator
  python scripts/evaluate.py --scores outputs/pilot.jsonl --by domain generator --k 0 0.1
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from trd.metrics import auroc, cluster_bootstrap, pair_win_rate, tpr_at_fpr
from trd.scoring import delta_score


def score_table(recs, ks):
    lp_old = [np.array(r["lp_old"]) for r in recs]
    lp_new = [np.array(r["lp_new"]) for r in recs]
    table = {f"delta(k={k:g})": np.array([delta_score(n, o, k) for n, o in zip(lp_new, lp_old)]) for k in ks}
    table["ll_new"] = np.array([n.mean() for n in lp_new])
    table["ll_old"] = np.array([o.mean() for o in lp_old])
    return table


def matched_pairs(recs, s) -> np.ndarray:
    """(AI score, human score) rows for every (id, generator) that has both texts."""
    by_id = defaultdict(dict)
    for r, v in zip(recs, s):
        by_id[(r["id"], r["generator"])][r["label"]] = v
    return np.array([(d[1], d[0]) for d in by_id.values() if 0 in d and 1 in d]).reshape(-1, 2)


def win_rate_ci(pairs: np.ndarray, n_boot: int, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    wins = pairs[:, 0] > pairs[:, 1]
    boots = [wins[rng.integers(0, len(wins), len(wins))].mean() for _ in range(n_boot)]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(lo), float(hi)


def evaluate(recs, ks, n_boot):
    labels = np.array([r["label"] for r in recs])
    clusters = np.array([f'{r["id"]}|{r["generator"]}' for r in recs])
    nan2 = (float("nan"), float("nan"))
    rows = {}
    for name, s in score_table(recs, ks).items():
        pairs = matched_pairs(recs, s)
        has_pairs = len(pairs) > 0
        rows[name] = {
            "auroc": auroc(labels, s),
            "auroc_ci": cluster_bootstrap(auroc, labels, s, clusters, n_boot) if n_boot else nan2,
            "tpr@1%fpr": tpr_at_fpr(labels, s),
            "pair_win": pair_win_rate(pairs[:, 0], pairs[:, 1]) if has_pairs else float("nan"),
            "pair_win_ci": win_rate_ci(pairs, n_boot) if n_boot and has_pairs else nan2,
            "mean_gap": float(np.mean(pairs[:, 0] - pairs[:, 1])) if has_pairs else float("nan"),
            "n_pairs": len(pairs),
        }
    return rows, labels


def print_rows(rows, title):
    print(f"\n{title}")
    print(f"{'score':<14}{'AUROC':>7}  {'95% CI':<15}{'TPR@1%':>7}{'pairwin':>9}  {'95% CI':<15}{'AI-human':>9}")
    for name, m in rows.items():
        print(f"{name:<14}{m['auroc']:>7.3f}  [{m['auroc_ci'][0]:.3f},{m['auroc_ci'][1]:.3f}]"
              f"{m['tpr@1%fpr']:>8.3f}{m['pair_win']:>9.3f}  [{m['pair_win_ci'][0]:.3f},{m['pair_win_ci'][1]:.3f}]"
              f"{m['mean_gap']:>+9.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--k", type=float, nargs="+", default=[0.0])
    ap.add_argument("--by", nargs="*", default=["generator"], help="record fields to break results down by")
    ap.add_argument("--bootstrap", type=int, default=1000, help="pair-bootstrap resamples (0 = skip CIs)")
    ap.add_argument("--json", default=None, help="also write all numbers here")
    args = ap.parse_args()

    recs = [json.loads(l) for l in Path(args.scores).read_text(encoding="utf-8").splitlines() if l.strip()]
    result = {"scores_file": args.scores, "n_texts": len(recs), "groups": {}}

    pooled, labels = evaluate(recs, args.k, args.bootstrap)
    print(f"n={len(recs)} (AI={int(labels.sum())}, human={int((1 - labels).sum())})")
    print_rows(pooled, "pooled")
    result["pooled"] = pooled

    if args.by:
        groups = defaultdict(list)
        for r in recs:
            groups[" / ".join(str(r.get(f, "")) for f in args.by)].append(r)
        if len(groups) > 1:
            for g, rs in sorted(groups.items()):
                rows, _ = evaluate(rs, args.k, args.bootstrap)
                print_rows(rows, f"{' / '.join(args.by)} = {g}  (n_pairs={rows['ll_new']['n_pairs']})")
                result["groups"][g] = rows

    d0 = score_table(recs, args.k[:1])
    s_delta, s_ll = d0[f"delta(k={args.k[0]:g})"], d0["ll_new"]
    r_all = float(np.corrcoef(s_delta, s_ll)[0, 1])
    r_within = {lab: float(np.corrcoef(s_delta[labels == lab], s_ll[labels == lab])[0, 1]) for lab in (0, 1)}
    m = pooled[f"delta(k={args.k[0]:g})"]
    h1 = m["pair_win"] > 0.5 and m["pair_win_ci"][0] > 0.5
    h2 = m["auroc"] >= 0.70
    h3 = abs(r_all) < 0.8
    print(f"\ncorr(delta, ll_new): pooled {r_all:+.3f}, within human {r_within[0]:+.3f}, within AI {r_within[1]:+.3f}")
    print(f"H1 AI > human in pairs (win > 0.5, CI above 0.5): {'yes' if h1 else 'no'}  ({m['pair_win']:.3f})")
    print(f"H2 AUROC(delta) >= 0.70:                          {'yes' if h2 else 'no'}  ({m['auroc']:.3f})")
    print(f"H3 |corr(delta, ll_new)| < 0.8:                   {'yes' if h3 else 'no'}  ({r_all:+.3f})")
    result.update({"corr_delta_llnew": r_all, "corr_within": r_within, "H1": h1, "H2": h2, "H3": h3})

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
