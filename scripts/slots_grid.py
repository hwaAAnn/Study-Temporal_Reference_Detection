"""Experiment 08 analysis: AUROC of Fast-DetectGPT and Binoculars in every (reference r, scorer s) cell of the
DatedGPT grid, four contrasts, and the direction verdict fixed before scoring.

Sets (human / AI, paired by pair_id): H19/C19 (main), B/C (replication), H24/C24 (exploratory).
Scores, higher = more AI: F = fdgpt, Bi = -bino. For reference: ll = mean log p under s alone, D = ll_s - ll_r.
A(r_s) = AUROC of cell r_s; E = 1 - A.
  H1 scorer slot     A(2022_2024) - A(2022_2022), holds if CI lower > 0
  H2 reference slot  A(2022_2024) - A(2024_2024), holds if CI lower > 0 and E(2024_2024) / E(2022_2024) >= 1.3;
                     a set where both cells are >= 0.998 is at the ceiling and is skipped
  H3 boundary        A(2023_2024) - A(2022_2024), holds if CI upper < 0
  H4 control         A(2022_2024) - A(2013_2024), holds if CI upper > 0 (not lower)
  diag (described)   A(2024_2024) - A(2022_2022), the usual single-model setting updated from 2022 to 2024
Verdict per detector: H2 holds on H19/C19 and on B/C (sets at the ceiling skipped; both at the ceiling ->
undecidable). Direction: H2 holds for both F and Bi -> adopt; for one -> partial; for neither -> drop.
CIs: 2000 bootstrap draws over pairs (seed 0); every cell uses the same draws.

Example:
  python scripts/slots_grid.py --grid outputs/s08_grid.jsonl --json outputs/s08_eval.json > outputs/s08_eval.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

SETS = [("H19", "C19", "main"), ("B", "C", "replication"), ("H24", "C24", "exploratory")]
DETS = ["F", "Bi"]
CONTRASTS = {
    "H1": ("2022_2024", "2022_2022"),
    "H2": ("2022_2024", "2024_2024"),
    "H3": ("2023_2024", "2022_2024"),
    "H4": ("2022_2024", "2013_2024"),
    "diag": ("2024_2024", "2022_2022"),
}
RATIO, CEIL = 1.3, 0.998


def score(row: dict, det: str, cell: str) -> float:
    r, s = cell.split("_")
    c = row["cells"]
    if det == "F":
        return c[cell]["fdgpt"]
    if det == "Bi":
        return -c[cell]["bino"]
    if det == "ll":
        return c[f"{s}_{s}"]["ll"]
    if det == "D":
        return c[f"{s}_{s}"]["ll"] - c[f"{r}_{r}"]["ll"]
    raise ValueError(det)


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(pos > neg) + 0.5 P(tie), from average ranks."""
    x = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2)[inv]
    n_p, n_n = len(pos), len(neg)
    return float((ranks[:n_p].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


def tpr_at(pos: np.ndarray, neg: np.ndarray, fpr: float = 0.05) -> float:
    thr = np.quantile(neg, 1 - fpr)
    return float(np.mean(pos > thr))


def pairs(rows: list[dict], hg: str, ag: str) -> list[tuple[dict, dict]]:
    h = {r["pair_id"]: r for r in rows if r["group"] == hg}
    a = {r["pair_id"]: r for r in rows if r["group"] == ag}
    return [(h[p], a[p]) for p in sorted(set(h) & set(a))]


def evaluate(rows: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    cells = sorted(rows[0]["cells"])
    vint = sorted({c.split("_")[0] for c in cells})
    need = sorted({c for pair in CONTRASTS.values() for c in pair})
    rng = np.random.default_rng(seed)
    res = {"vintages": vint, "sets": {}}
    for hg, ag, role in SETS:
        pr = pairs(rows, hg, ag)
        if not pr:
            continue
        n = len(pr)
        draws = [rng.integers(0, n, n) for _ in range(n_boot)]
        out = {"role": role, "n_pairs": n, "dets": {}}
        for det in DETS + ["ll", "D"]:
            H = {c: np.array([score(h, det, c) for h, a in pr]) for c in cells}
            C = {c: np.array([score(a, det, c) for h, a in pr]) for c in cells}
            grid = {c: auroc(C[c], H[c]) for c in cells}
            tpr = {c: tpr_at(C[c], H[c]) for c in cells}
            d = {"auroc": grid, "tpr5": tpr}
            if det in DETS:
                boot = {c: np.array([auroc(C[c][i], H[c][i]) for i in draws]) for c in need}
                con = {}
                for name, (x, y) in CONTRASTS.items():
                    diff = boot[x] - boot[y]
                    con[name] = {"cells": [x, y], "a": [grid[x], grid[y]], "diff": grid[x] - grid[y],
                                 "ci": [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))]}
                e_mix, e_new = 1 - grid["2022_2024"], 1 - grid["2024_2024"]
                con["H2"]["error_ratio"] = float(e_new / e_mix) if e_mix > 0 else float("inf")
                d["contrasts"] = con
                gens = {}
                for g in sorted({a["generator"] for h, a in pr}):
                    idx = np.array([k for k, (h, a) in enumerate(pr) if a["generator"] == g])
                    gens[g] = {c: auroc(C[c][idx], H[c][idx]) for c in need}
                d["by_generator"] = gens
            out["dets"][det] = d
        res["sets"][f"{hg}/{ag}"] = out
    res["verdict"] = judge(res)
    return res


def h_status(name: str, con: dict) -> str:
    lo, hi = con["ci"]
    if name == "H1":
        return "holds" if lo > 0 else "fails"
    if name == "H2":
        if min(con["a"]) >= CEIL:
            return "ceiling"
        return "holds" if lo > 0 and con["error_ratio"] >= RATIO else "fails"
    if name == "H3":
        return "holds" if hi < 0 else "fails"
    if name == "H4":
        return "holds" if hi > 0 else "fails"
    return "-"


def judge(res: dict) -> dict:
    v = {"per_set": {}, "H2": {}}
    for key, s in res["sets"].items():
        v["per_set"][key] = {det: {h: h_status(h, s["dets"][det]["contrasts"][h]) for h in ("H1", "H2", "H3", "H4")}
                             for det in DETS}
    deciding = [k for k, s in res["sets"].items() if s["role"] in ("main", "replication")]
    for det in DETS:
        st = [v["per_set"][k][det]["H2"] for k in deciding]
        st = [x for x in st if x != "ceiling"]
        v["H2"][det] = "undecidable" if not st else ("holds" if all(x == "holds" for x in st) else "fails")
    n = sum(v["H2"][d] == "holds" for d in DETS)
    if all(v["H2"][d] == "undecidable" for d in DETS):
        v["direction"] = "undecidable"
    else:
        v["direction"] = "adopt" if n == len(DETS) else ("partial" if n else "drop")
    return v


def print_report(res: dict) -> None:
    vint = res["vintages"]
    for key, s in res["sets"].items():
        print(f"\n=== {key} ({s['role']}), {s['n_pairs']} pairs ===")
        for det in DETS + ["ll", "D"]:
            g = s["dets"][det]["auroc"]
            print(f"\n{det}: AUROC, rows = reference r, columns = scorer s")
            print("r \\ s  " + "  ".join(f"{x:>6}" for x in vint))
            for r in vint:
                print(f"{r}   " + "  ".join(f"{g[f'{r}_{x}']:.4f}" for x in vint))
        for det in DETS:
            print(f"\n{det} contrasts:")
            for name, c in s["dets"][det]["contrasts"].items():
                extra = f"  E ratio {c['error_ratio']:.2f}" if name == "H2" else ""
                st = h_status(name, c)
                print(f"  {name:4} {c['cells'][0]} - {c['cells'][1]}: {c['a'][0]:.4f} - {c['a'][1]:.4f} = "
                      f"{c['diff']:+.4f} [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]{extra}  {st}")
    v = res["verdict"]
    print("\n=== verdict ===")
    for key, d in v["per_set"].items():
        print(f"{key}: " + "; ".join(f"{det} " + ", ".join(f"{h} {x}" for h, x in hs.items()) for det, hs in d.items()))
    print("H2 per detector: " + ", ".join(f"{d} {x}" for d, x in v["H2"].items()))
    print(f"direction: {v['direction']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rows = [json.loads(line) for line in Path(args.grid).read_text(encoding="utf-8").splitlines() if line.strip()]
    res = evaluate(rows, args.n_boot, args.seed)
    print_report(res)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
