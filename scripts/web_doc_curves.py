"""Experiment 01, exploratory (not pre-registered): the human side alone of the web-document and arXiv sets,
per vintage, to read Q5 (is each vintage's data its own year's crawl, or cumulative?) without the AI
continuation side, which shifts after ChatGPT.

For each human set (W21, W22: FineWeb-Edu documents in the 2021-49 / 2022-40 crawls, in the pool;
H19, H22, H24, B: arXiv abstracts, outside it), l_v = mean log p_v over the first n tokens, n = the
length shared with the set's AI partner (as in year_curve.py). Prints each set relative to its
2013-2019 mean, D(v) = l(W21) - l(W22), and the change after the crawl-year vintage with 95% bootstrap
CIs over documents (the same resample for every vintage).

Example:
  python scripts/web_doc_curves.py --scores outputs/yc_20*.jsonl > outputs/yc_q5_human_only.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from year_curve import YEARS, Scores  # noqa: E402

SETS = [("W21", "CW21"), ("W22", "CW22"), ("H19", "C19"), ("H22", "C22"), ("H24", "C24"), ("B", "C")]
# (set, crawl-year vintage, later vintages): web sets are in the pool; arXiv sets of the same years are not
CHANGES = [("W21", 2021, [2022, 2024]), ("W22", 2022, [2023, 2024]),
           ("H19", 2021, [2022, 2024]), ("H22", 2022, [2023, 2024])]
YI = {y: k for k, y in enumerate(YEARS)}
PRE7 = slice(0, 7)  # 2013-2019


def human_side(sc: Scores, hg: str, ag: str) -> np.ndarray:
    hs, as_ = sc.by_group(hg), sc.by_group(ag)
    return np.array([sc.l(hs[p], min(sc.n(hs[p]), sc.n(as_[p]))) for p in sorted(set(hs) & set(as_))])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True, help="one score file per vintage (2013-2024)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    sc = Scores(args.scores)
    sets = {h: human_side(sc, h, a) for h, a in SETS}
    print("human text only: mean l_v minus its 2013-2019 mean")
    print("set   n    " + " ".join(f"{y:>7}" for y in YEARS))
    for k, a in sets.items():
        m = a.mean(0)
        print(f"{k:<4}{len(a):>4}   " + " ".join(f"{x - m[PRE7].mean():+7.3f}" for x in m))

    def boot(a: np.ndarray) -> np.ndarray:
        return np.array([a[rng.integers(0, len(a), len(a))].mean(0) for _ in range(args.n_boot)])

    print("\nchange after the crawl-year vintage, 95% CI over documents")
    for k, peak, later in CHANGES:
        bs = boot(sets[k])
        m = sets[k].mean(0)
        for t in later:
            d = bs[:, YI[t]] - bs[:, YI[peak]]
            print(f"  {k}  l({t}) - l({peak})  {m[YI[t]] - m[YI[peak]]:+.3f} "
                  f"[{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]")

    a, b = sets["W21"], sets["W22"]
    ba, bb = boot(a), boot(b)
    D, bD = a.mean(0) - b.mean(0), ba - bb
    print("\nD(v) = l(W21) - l(W22), minus its 2013-2019 mean")
    print("  " + " ".join(f"{y}:{x - D[PRE7].mean():+.3f}" for y, x in zip(YEARS, D)))
    for name, f in [("D(2021) - D(2022)", lambda x: x[..., YI[2021]] - x[..., YI[2022]]),
                    ("D(2024) - D(2022)", lambda x: x[..., YI[2024]] - x[..., YI[2022]])]:
        v = f(bD)
        print(f"  {name}  {f(D):+.3f} [{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]")


if __name__ == "__main__":
    main()
