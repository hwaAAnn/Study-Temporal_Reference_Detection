"""Experiment 02: how much does 'the newer model may have trained on this human text' move Delta?

Each period p has human abstracts H_p and same-title AI abstracts C_p (same generators as in 00).
C_p is the topic- and knowledge-matched anchor: whether the human text was crawled into a model's
data does not change the AI text, so a change of the C-H gap across periods reflects what happened
to the human side (training membership, AI-assisted writing).

    gap_m(p) = mean Delta_m(C_p) - mean Delta_m(H_p)          m = post (2024-2021) or pre (2021-2018)

Membership contrasts (positive = membership raises the human text's Delta):
    mia_pre        = gap_pre(22) - gap_pre(19)     H19 eligible for 2021 only, H22 for neither; no AI writing
    mia_post_clean = gap_post(19) - gap_post(22)   H22 eligible for 2024 only, H19 for both;    no AI writing
    mia_minus_polish_B = gap_post(25) - gap_post(22)   membership(H22) minus AI-assisted writing in B (2025+)
    polish_24      = gap_post(22) - gap_post(24)   both eligible for 2024; H24 may contain AI-assisted text

CIs: periods are independent samples, so the bootstrap resamples pair_ids within each period.

Example:
  python scripts/mia_contrast.py --scores outputs/mia.jsonl \
      --periods 19:H19:C19 22:H22:C22 24:H24:C24 25:B:C --json outputs/mia_eval.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from trd.metrics import auroc

CONTRASTS = {
    "mia_pre": (("pre", "22"), ("pre", "19")),
    "mia_post_clean": (("post", "19"), ("post", "22")),
    "mia_minus_polish_B": (("post", "25"), ("post", "22")),
    "polish_24": (("post", "22"), ("post", "24")),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--periods", nargs="+", default=["19:H19:C19", "22:H22:C22", "24:H24:C24", "25:B:C"],
                    help="label:human_group:ai_group")
    ap.add_argument("--post", default="2024-2021")
    ap.add_argument("--pre", default="2021-2018")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--contrasts", nargs="*", default=None,
                    help="name=m:label-m:label, e.g. mia_pre=pre:w22-pre:w21 (default: the arXiv set above)")
    ap.add_argument("--diffs", nargs="*", default=["mia_post_clean-mia_pre"],
                    help="differences of two contrasts with a joint bootstrap CI, e.g. mia_post_web-mia_pre_web")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    contrast_defs = dict(CONTRASTS)
    if args.contrasts:
        contrast_defs = {}
        for spec in args.contrasts:
            name, rhs = spec.split("=")
            a, b = rhs.split("-")
            contrast_defs[name] = (tuple(a.split(":")), tuple(b.split(":")))

    recs = [json.loads(l) for l in Path(args.scores).read_text(encoding="utf-8").splitlines() if l.strip()]
    pairs = {"post": args.post.split("-"), "pre": args.pre.split("-")}
    periods = [p.split(":") for p in args.periods]

    # per period: arrays of Delta for human and AI, indexed by pair (pairs with both texts only)
    data = {}
    for label, hg, cg in periods:
        by_pair = defaultdict(dict)
        for r in recs:
            if r["group"] in (hg, cg):
                d = {m: float(np.mean(np.array(r["lp"][a]) - np.array(r["lp"][b]))) for m, (a, b) in pairs.items()}
                by_pair[r["pair_id"]]["h" if r["group"] == hg else "c"] = d
        keep = [v for v in by_pair.values() if "h" in v and "c" in v]
        data[label] = {m: (np.array([v["h"][m] for v in keep]), np.array([v["c"][m] for v in keep])) for m in pairs}
        data[label]["n"] = len(keep)

    def gaps(idx_by_period):
        out = {}
        for label, _, _ in periods:
            idx = idx_by_period[label]
            for m in pairs:
                h, c = data[label][m]
                out[m, label] = float(c[idx].mean() - h[idx].mean())
        return out

    def contrasts(g):
        return {name: g[a] - g[b] for name, (a, b) in contrast_defs.items() if a in g and b in g}

    full = {label: np.arange(data[label]["n"]) for label, _, _ in periods}
    g0, c0 = gaps(full), None
    c0 = contrasts(g0)
    rng = np.random.default_rng(args.seed)
    boots_g, boots_c = defaultdict(list), defaultdict(list)
    for _ in range(args.n_boot):
        idx = {label: rng.integers(0, data[label]["n"], data[label]["n"]) for label, _, _ in periods}
        g = gaps(idx)
        for k, v in g.items():
            boots_g[k].append(v)
        for k, v in contrasts(g).items():
            boots_c[k].append(v)
    ci = lambda xs: (float(np.quantile(xs, 0.025)), float(np.quantile(xs, 0.975)))

    result = {"periods": {}, "contrasts": {}}
    print(f"{'period':<8}{'n':>5}{'H post':>9}{'C post':>9}{'gap post':>10} {'95% CI':<17}"
          f"{'H pre':>8}{'C pre':>8}{'gap pre':>9} {'95% CI':<17}{'AUROC post':>11}")
    for label, hg, cg in periods:
        hp, cp = data[label]["post"]
        hq, cq = data[label]["pre"]
        y = np.r_[np.zeros(len(hp)), np.ones(len(cp))]
        au = auroc(y, np.r_[hp, cp])
        cpost, cpre = ci(boots_g["post", label]), ci(boots_g["pre", label])
        print(f"{label:<8}{data[label]['n']:>5}{hp.mean():>9.3f}{cp.mean():>9.3f}{g0['post', label]:>+10.3f} "
              f"[{cpost[0]:+.3f}, {cpost[1]:+.3f}]{hq.mean():>8.3f}{cq.mean():>8.3f}{g0['pre', label]:>+9.3f} "
              f"[{cpre[0]:+.3f}, {cpre[1]:+.3f}]{au:>11.3f}")
        result["periods"][label] = {"human_group": hg, "ai_group": cg, "n_pairs": data[label]["n"],
                                    "human_post": float(hp.mean()), "ai_post": float(cp.mean()),
                                    "human_pre": float(hq.mean()), "ai_pre": float(cq.mean()),
                                    "gap_post": g0["post", label], "gap_post_ci": cpost,
                                    "gap_pre": g0["pre", label], "gap_pre_ci": cpre, "auroc_post": au}
    print()
    for name, v in c0.items():
        lo, hi = ci(boots_c[name])
        print(f"{name:<20}{v:>+8.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
        result["contrasts"][name] = {"value": v, "ci": [lo, hi]}
    for spec in args.diffs or []:
        a, b = spec.split("-")
        if a in c0 and b in c0:
            diff = np.array(boots_c[a]) - np.array(boots_c[b])
            v = c0[a] - c0[b]
            lo, hi = ci(diff)
            print(f"{spec:<20}{v:>+8.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
            result["contrasts"][spec] = {"value": v, "ci": [lo, hi]}
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
