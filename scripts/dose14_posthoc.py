"""Experiment 14, post hoc (written after the results): does a second term, the calendar distance between the two
slots' crawl dates, explain what the dose alone misses (D2: older pre-ChatGPT references give less gain)?

Per set, family (DatedGPT, TiC 5x5 grids) and detector, OLS over the 25 cells of s14_eval.json:
  M1  d ~ 1 + [r = s] + (G(s) - G(r))                      (= D5)
  M2  d ~ 1 + [r = s] + (G(s) - G(r)) + |t(s) - t(r)|       t = crawl date in years
  M3  d ~ 1 + [r = s] + G(s) + G(r)                         do the two slots carry opposite, equal weights?
Prints coefficients and R^2. Also the leave-one-cell-out prediction of the key cell (pre_new) under M2.

Example:
  python scripts/dose14_posthoc.py --eval ../Study-Temporal_Reference_Detection/outputs/s14_eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

T = {
    "DG": {"2013": 2013.0, "2019": 2019.0, "2022": 2022.0, "2023": 2023.0, "2024": 2024.0},
    "TiC": {"201305": 2013.33, "201901": 2019.0, "202101": 2021.0, "202301": 2023.0, "202407": 2024.5},
}
KEY = {"DG": "2022_2024", "TiC": "202301_202407"}


def fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b, 1 - float(((y - X @ b) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    res = json.loads(Path(ap.parse_args().eval).read_text(encoding="utf-8"))
    G = res["dose"]
    for key, s in res["sets"].items():
        for fam in ("DG", "TiC"):
            ms = list(T[fam])
            cells = [(r, t) for r in ms for t in ms]
            gd = np.array([G[fam][t] - G[fam][r] for r, t in cells])
            dist = np.array([abs(T[fam][t] - T[fam][r]) for r, t in cells])
            diag = np.array([float(r == t) for r, t in cells])
            X1 = np.column_stack([np.ones(25), diag, gd])
            X2 = np.column_stack([X1, dist])
            for det in ("F", "Bi"):
                y = np.array([s["families"][fam][det]["d"][f"{r}_{t}"] for r, t in cells])
                b1, r2_1 = fit(X1, y)
                b2, r2_2 = fit(X2, y)
                X3 = np.column_stack([np.ones(25), diag, [G[fam][t] for r, t in cells], [G[fam][r] for r, t in cells]])
                b3, r2_3 = fit(X3, y)
                k = [f"{r}_{t}" for r, t in cells].index(KEY[fam])
                keep = np.arange(25) != k
                bk, _ = fit(X2[keep], y[keep])
                print(f"{key:8} {fam:3} {det:2}  M1 R2 {r2_1:.2f} (dose {b1[2]:+.2f})   "
                      f"M2 R2 {r2_2:.2f} (dose {b2[2]:+.2f}, distance {b2[3]:+.3f}/yr, diag {b2[1]:+.2f})   "
                      f"LOO {KEY[fam]}: pred {X2[k] @ bk:.2f} vs obs {y[k]:.2f}   "
                      f"M3 G(s) {b3[2]:+.2f}, G(r) {b3[3]:+.2f}, R2 {r2_3:.2f}")


if __name__ == "__main__":
    main()
