"""Gate R1 on the 07 TiC-LM scores: does single-model likelihood detection step up at the post-ChatGPT cutoff?

The rule was fixed on 2026-11-15 16:00 (session scratchpad, calibrated on the already-seen DatedGPT scores),
before TiC-LM scoring began at 18:19; it was not committed before the scores existed (see 08 doc).
  A(c)   AUROC of mean log p under cutoff c alone (AI vs human, paired sets)
  J_k    logit A(c_k) - logit A(c_{k-1});  J_post = 202301 -> 202407, J_pre = the five earlier steps
  z_J    (J_post - mean J_pre) / (sd J_pre * sqrt(1 + 1/5)), 4 df
  holds  z_J > 2.776 and the paired pair-bootstrap 95% CI of A(202407) - A(202301) > 0 (2000 draws, seed 0)
Main set H19/C19, secondary B/C.

Example:
  python scripts/tic_r1_gate.py --outputs outputs > outputs/tic_r1_gate.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CUTS = ["201305", "201501", "201701", "201901", "202101", "202301", "202407"]


def auroc(pos, neg):
    x = np.concatenate([pos, neg])
    ranks = x.argsort().argsort() + 1.0
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", required=True, help="directory with tic1b_<cutoff>.jsonl")
    args = ap.parse_args()
    mean_lp, meta = {}, {}
    for c in CUTS:
        d = {}
        for line in (Path(args.outputs) / f"tic1b_{c}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            (lp,) = r["lp"].values()
            if lp:
                d[r["id"]] = float(np.mean(lp))
            meta[r["id"]] = (r["group"], r["pair_id"], r["generator"])
        mean_lp[c] = d
    logit = lambda p: np.log(p / (1 - p))  # noqa: E731
    rng = np.random.default_rng(0)
    for hg, ag in [("H19", "C19"), ("B", "C")]:
        h = {(m[1], m[2]): i for i, m in meta.items() if m[0] == hg}
        a = {(m[1], m[2]): i for i, m in meta.items() if m[0] == ag}
        pr = [(h[k], a[k]) for k in sorted(set(h) & set(a))
              if all(h[k] in mean_lp[c] and a[k] in mean_lp[c] for c in CUTS)]
        H = {c: np.array([mean_lp[c][x] for x, y in pr]) for c in CUTS}
        C = {c: np.array([mean_lp[c][y] for x, y in pr]) for c in CUTS}
        A = np.array([auroc(C[c], H[c]) for c in CUTS])
        J = np.diff(logit(A))
        jpre, jpost = J[:-1], J[-1]
        z = (jpost - jpre.mean()) / (jpre.std(ddof=1) * np.sqrt(1 + 1 / len(jpre)))
        n = len(pr)
        d = []
        for _ in range(2000):
            i = rng.integers(0, n, n)
            d.append(auroc(C["202407"][i], H["202407"][i]) - auroc(C["202301"][i], H["202301"][i]))
        lo, hi = np.percentile(d, [2.5, 97.5])
        ok = z > 2.776 and lo > 0
        print(f"\n=== {hg}/{ag}  n={n} ===")
        print("AUROC  " + "  ".join(f"{c}:{x:.4f}" for c, x in zip(CUTS, A)))
        print(f"J_pre {np.round(jpre, 3).tolist()}  J_post {jpost:+.3f}")
        print(f"dA(202407-202301) {A[-1] - A[-2]:+.4f}  CI [{lo:+.4f},{hi:+.4f}]  z_J {z:+.1f} -> "
              f"{'HOLDS' if ok else 'fails'}")


if __name__ == "__main__":
    main()
