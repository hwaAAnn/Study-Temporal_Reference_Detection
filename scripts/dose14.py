"""Experiment 14: does the chat absorption measured in 01/07 (G) predict the reference-slot loss seen in 08/10/11?

No new scoring. Inputs are the slot grids already on disk, all on the same 2,398 texts of 08:
  DatedGPT  outputs/s08_grid.jsonl  r, s in 2013 2019 2022 2023 2024
  TiC-LM    outputs/s10_A.jsonl     r, s in 201305 201901 202101 202301 202407
  Falcon    outputs/s11_grid.jsonl  r, s in base inst (rows of the 08 groups only)
Cell keys are "r_s": r = reference slot (Fast-DetectGPT sampling model, Binoculars observer), s = scorer slot.

Dose (predictor, one number per model m, fixed before the outcome is computed):
  G(m) = mean over B/C pairs of ll_m(C) - ll_m(B), ll_m = mean log p of m over the first 200 tokens (the
  ll field of diagonal cell m_m). Same quantity as 01/07's G, but without truncating a pair to its shorter text.
  x(r) = G(s*) - G(r): how much less chat-absorbed reference r is than the family's newest scorer s*.
Outcome (per set, detector, cell): d(r_s) = Cohen's d of the detector score, AI minus human, pooled SD.
  Scores, higher = more AI: F = fdgpt, Bi = -bino. AUROC is reported next to it.
  y(r) = d(r_s*) - d(s*_s*): gain of reference r over using the newest model in both slots.
Tests (criteria in docs/experiments/14-dose-response.md, fixed before any outcome was computed):
  D1 intermediate dose, DatedGPT. f = y(2023) / y(2022), predicted f_hat = x(2023) / x(2022).
     status: undecidable if CI of y(2022) includes 0; graded if CI of f is inside (0, 1);
     proportional if graded and f in [f_hat / 2, 2 f_hat]; step if CI upper < 1 and lower <= 0 (2023 already
     loses the whole gain); no-dose if CI lower > 0 and upper >= 1; wide if the CI covers both 0 and 1.
     Per detector: the status must be the same on H19/C19 and B/C; otherwise "mixed".
  D2 pre-ChatGPT references, DatedGPT: y(2013) / y(2022), y(2019) / y(2022) against x ratios; flat if in [0.7, 1.3].
  D3 scorer side, DatedGPT: g = [d(2022_2023) - d(2022_2022)] / [d(2022_2024) - d(2022_2022)],
     g_hat = [G(2023) - G(2022)] / [G(2024) - G(2022)].
  D4 one scale across families: k = y(r_pre) / x(r_pre) with r_pre = 2022 / 202301 / base, ratio rho = k / k_DG;
     same scale if rho in [0.5, 2].
  D5 (exploratory) per family, OLS over the 25 cells: d(r_s) ~ 1 + [r = s] + (G(s) - G(r)); R^2 with and
     without the dose term.
CIs: 2000 bootstrap draws over pair_id (seed 0); the same draws for every cell and every family.

Example:
  python scripts/dose14.py --outputs ../Study-Temporal_Reference_Detection/outputs --dose-only
  python scripts/dose14.py --outputs ../Study-Temporal_Reference_Detection/outputs \
      --json ../Study-Temporal_Reference_Detection/outputs/s14_eval.json > ../Study-Temporal_Reference_Detection/outputs/s14_eval.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FAMILIES = {
    "DG": {"file": "s08_grid.jsonl", "models": ["2013", "2019", "2022", "2023", "2024"], "new": "2024", "pre": "2022"},
    "TiC": {"file": "s10_A.jsonl", "models": ["201305", "201901", "202101", "202301", "202407"], "new": "202407",
            "pre": "202301"},
    "Falcon": {"file": "s11_grid.jsonl", "models": ["base", "inst"], "new": "inst", "pre": "base"},
}
# Published G (nats/token, B/C, pairs truncated to the shorter text): 01 results table and 07 results table.
PUBLISHED_G = {
    "DG": {"2013": 0.971, "2019": 0.978, "2022": 0.945, "2023": 1.152, "2024": 1.219},
    "TiC": {"201305": 1.110, "201901": 1.103, "202101": 1.095, "202301": 1.080, "202407": 1.273},
}
SETS = [("H19", "C19", "main"), ("B", "C", "replication"), ("H24", "C24", "exploratory")]
GROUPS = {g for hg, ag, _ in SETS for g in (hg, ag)}
DETS = ["F", "Bi"]
FLAT = (0.7, 1.3)
SCALE = (0.5, 2.0)


def load(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r["group"] in GROUPS:
                rows[r["id"]] = r
    return rows


def score(row: dict, det: str, cell: str) -> float:
    c = row["cells"][cell]
    if det == "F":
        return c["fdgpt"]
    if det == "Bi":
        return -c["bino"]
    raise ValueError(det)


def pair_ids(fams: dict[str, dict], hg: str, ag: str) -> list[str]:
    """pair_ids present as both human and AI text in every family's grid."""
    common = None
    for rows in fams.values():
        h = {r["pair_id"] for r in rows.values() if r["group"] == hg}
        a = {r["pair_id"] for r in rows.values() if r["group"] == ag}
        common = (h & a) if common is None else common & h & a
    return sorted(common or [])


def by_pair(rows: dict[str, dict], group: str) -> dict[str, dict]:
    return {r["pair_id"]: r for r in rows.values() if r["group"] == group}


def dose(rows: dict[str, dict], models: list[str], pids: list[str]) -> dict[str, float]:
    h, a = by_pair(rows, "B"), by_pair(rows, "C")
    return {m: float(np.mean([a[p]["cells"][f"{m}_{m}"]["ll"] - h[p]["cells"][f"{m}_{m}"]["ll"] for p in pids]))
            for m in models}


def cohens_d_rows(ai: np.ndarray, hu: np.ndarray) -> np.ndarray:
    """Cohen's d along the last axis (pooled SD), for one row per bootstrap draw."""
    na, nh = ai.shape[-1], hu.shape[-1]
    sp = np.sqrt(((na - 1) * ai.var(axis=-1, ddof=1) + (nh - 1) * hu.var(axis=-1, ddof=1)) / (na + nh - 2))
    return (ai.mean(axis=-1) - hu.mean(axis=-1)) / sp


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    x = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2)[inv]
    n_p, n_n = len(pos), len(neg)
    return float((ranks[:n_p].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


def ci(v: np.ndarray) -> list[float]:
    v = v[np.isfinite(v)]
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]


def d1_status(f_ci: list[float], f: float, f_hat: float, y_pre_ci: list[float]) -> str:
    lo, hi = f_ci
    if y_pre_ci[0] <= 0:
        return "undecidable"
    if lo <= 0 and hi >= 1:
        return "wide"
    if lo <= 0:
        return "step"
    if hi >= 1:
        return "no-dose"
    return "proportional" if f_hat / 2 <= f <= 2 * f_hat else "graded"


def combine(statuses: list[str]) -> str:
    return statuses[0] if len(set(statuses)) == 1 else "mixed (" + ", ".join(statuses) + ")"


def evaluate(fams: dict[str, dict], n_boot: int = 2000, seed: int = 0) -> dict:
    bc = pair_ids(fams, "B", "C")
    G = {f: dose(rows, FAMILIES[f]["models"], bc) for f, rows in fams.items()}
    res: dict = {"dose": G, "published_G": PUBLISHED_G, "sets": {}}
    rng = np.random.default_rng(seed)
    for hg, ag, role in SETS:
        pids = pair_ids(fams, hg, ag)
        if not pids:
            continue
        n = len(pids)
        draws = np.stack([rng.integers(0, n, n) for _ in range(n_boot)])
        sres: dict = {"role": role, "n_pairs": n, "families": {}}
        D: dict = {}   # D[fam][det][cell] = (point d, boot d array)
        A: dict = {}   # AUROC points
        for fam, rows in fams.items():
            ms = FAMILIES[fam]["models"]
            h, a = by_pair(rows, hg), by_pair(rows, ag)
            gen = np.array([a[p]["generator"] for p in pids])
            D[fam], A[fam] = {}, {}
            for det in DETS:
                D[fam][det], A[fam][det] = {}, {}
                for r in ms:
                    for s in ms:
                        cell = f"{r}_{s}"
                        hs = np.array([score(h[p], det, cell) for p in pids])
                        as_ = np.array([score(a[p], det, cell) for p in pids])
                        point = float(cohens_d_rows(as_[None, :], hs[None, :])[0])
                        boot = cohens_d_rows(as_[draws], hs[draws])
                        by_gen = {g: float(cohens_d_rows(as_[gen == g][None, :], hs[gen == g][None, :])[0])
                                  for g in sorted(set(gen))}
                        D[fam][det][cell] = (point, boot, by_gen)
                        A[fam][det][cell] = auroc(as_, hs)
            sres["families"][fam] = {
                det: {"d": {c: v[0] for c, v in D[fam][det].items()},
                      "d_by_generator": {c: v[2] for c, v in D[fam][det].items()},
                      "auroc": A[fam][det]} for det in DETS}
        sres["tests"] = tests(D, A, G)
        res["sets"][f"{hg}/{ag}"] = sres
    res["verdict"] = judge(res)
    res["regression"] = regression(res, G)
    return res


def gain(Df: dict, r: str, s: str) -> tuple[float, np.ndarray]:
    """y(r) = d(r_s) - d(s_s), point and bootstrap."""
    return Df[f"{r}_{s}"][0] - Df[f"{s}_{s}"][0], Df[f"{r}_{s}"][1] - Df[f"{s}_{s}"][1]


def tests(D: dict, A: dict, G: dict) -> dict:
    out: dict = {}
    gd = G["DG"]
    x = {r: gd["2024"] - gd[r] for r in gd}
    for det in DETS:
        Dg = D["DG"][det]
        t: dict = {}
        y22, y22b = gain(Dg, "2022", "2024")
        y23, y23b = gain(Dg, "2023", "2024")
        f, fb = y23 / y22, y23b / y22b
        f_hat = x["2023"] / x["2022"]
        a = A["DG"][det]
        f_auc = ((a["2023_2024"] - a["2024_2024"]) / (a["2022_2024"] - a["2024_2024"])
                 if a["2022_2024"] != a["2024_2024"] else float("nan"))
        t["D1"] = {"y2022": y22, "y2022_ci": ci(y22b), "y2023": y23, "y2023_ci": ci(y23b), "f": f, "f_ci": ci(fb),
                   "f_hat": f_hat, "f_auroc": f_auc,
                   "auroc": {c: a[c] for c in ("2022_2024", "2023_2024", "2024_2024")}}
        t["D1"]["status"] = d1_status(t["D1"]["f_ci"], f, f_hat, t["D1"]["y2022_ci"])
        t["D1"]["by_generator"] = {}
        for g in Dg["2022_2024"][2]:
            yy22 = Dg["2022_2024"][2][g] - Dg["2024_2024"][2][g]
            yy23 = Dg["2023_2024"][2][g] - Dg["2024_2024"][2][g]
            t["D1"]["by_generator"][g] = {"y2022": yy22, "y2023": yy23, "f": yy23 / yy22 if yy22 else float("nan")}
        t["D2"] = {}
        for r in ("2013", "2019"):
            yr, yrb = gain(Dg, r, "2024")
            ratio = yr / y22
            t["D2"][r] = {"ratio": ratio, "ci": ci(yrb / y22b), "pred": x[r] / x["2022"],
                          "status": "flat" if FLAT[0] <= ratio <= FLAT[1] else "not flat"}
        num = Dg["2022_2023"][0] - Dg["2022_2022"][0]
        den = Dg["2022_2024"][0] - Dg["2022_2022"][0]
        numb = Dg["2022_2023"][1] - Dg["2022_2022"][1]
        denb = Dg["2022_2024"][1] - Dg["2022_2022"][1]
        t["D3"] = {"g": num / den, "ci": ci(numb / denb),
                   "g_hat": (gd["2023"] - gd["2022"]) / (gd["2024"] - gd["2022"])}
        k = {}
        for fam, spec in FAMILIES.items():
            gf = G[fam]
            xf = gf[spec["new"]] - gf[spec["pre"]]
            yf, yfb = gain(D[fam][det], spec["pre"], spec["new"])
            k[fam] = {"x": xf, "y": yf, "y_ci": ci(yfb), "k": yf / xf, "kb": yfb / xf}
            k[fam]["points"] = {r: {"x": gf[spec["new"]] - gf[r], "y": gain(D[fam][det], r, spec["new"])[0]}
                                for r in spec["models"]}
        t["D4"] = {}
        for fam in FAMILIES:
            rho, rhob = k[fam]["k"] / k["DG"]["k"], k[fam]["kb"] / k["DG"]["kb"]
            t["D4"][fam] = {"x": k[fam]["x"], "y": k[fam]["y"], "y_ci": k[fam]["y_ci"], "k": k[fam]["k"],
                            "rho": rho, "rho_ci": ci(rhob), "points": k[fam]["points"],
                            "status": "same scale" if SCALE[0] <= rho <= SCALE[1] else "different scale"}
        out[det] = t
    return out


def judge(res: dict) -> dict:
    v: dict = {"D1": {}, "D4": {}}
    deciding = [k for k, s in res["sets"].items() if s["role"] in ("main", "replication")]
    for det in DETS:
        v["D1"][det] = combine([res["sets"][k]["tests"][det]["D1"]["status"] for k in deciding])
        v["D4"][det] = {fam: combine([res["sets"][k]["tests"][det]["D4"][fam]["status"] for k in deciding])
                        for fam in ("TiC", "Falcon")}
    st = list(v["D1"].values())
    if all(s == "proportional" for s in st):
        v["dose_response"] = "quantitative"
    elif all(s in ("proportional", "graded") for s in st):
        v["dose_response"] = "monotone"
    elif all(s == "step" for s in st):
        v["dose_response"] = "step"
    elif all(s == "no-dose" for s in st):
        v["dose_response"] = "no-dose"
    else:
        v["dose_response"] = "mixed"
    return v


def regression(res: dict, G: dict) -> dict:
    """D5: per set, family (5x5 grids), detector: R^2 of d ~ 1 + diag + dose, and of d ~ 1 + diag."""
    out: dict = {}
    for key, s in res["sets"].items():
        out[key] = {}
        for fam in ("DG", "TiC"):
            ms = FAMILIES[fam]["models"]
            cells = [(r, t) for r in ms for t in ms]
            X1 = np.array([[1.0, float(r == t), G[fam][t] - G[fam][r]] for r, t in cells])
            X0 = X1[:, :2]
            out[key][fam] = {}
            for det in DETS:
                y = np.array([s["families"][fam][det]["d"][f"{r}_{t}"] for r, t in cells])
                tss = float(((y - y.mean()) ** 2).sum())
                b1, *_ = np.linalg.lstsq(X1, y, rcond=None)
                b0, *_ = np.linalg.lstsq(X0, y, rcond=None)
                r2 = 1 - float(((y - X1 @ b1) ** 2).sum()) / tss
                r2_0 = 1 - float(((y - X0 @ b0) ** 2).sum()) / tss
                out[key][fam][det] = {"beta_dose": float(b1[2]), "beta_diag": float(b1[1]), "r2": r2, "r2_no_dose": r2_0}
    return out


def print_dose(G: dict) -> None:
    print("Dose G(m), nats/token, B/C pairs (grid ll, first 200 tokens, no pair truncation)")
    for fam, gm in G.items():
        pub = PUBLISHED_G.get(fam, {})
        print(f"  {fam}: " + "  ".join(f"{m} {v:.3f}" + (f" (01/07 {pub[m]:.3f})" if m in pub else "")
                                       for m, v in gm.items()))
    gd = G["DG"]
    x = {r: gd["2024"] - gd[r] for r in gd}
    print("  DG x(r) = G(2024) - G(r): " + "  ".join(f"{r} {v:.3f}" for r, v in x.items()))
    print(f"  f_hat = x(2023)/x(2022) = {x['2023'] / x['2022']:.3f};  "
          f"g_hat = [G(2023)-G(2022)]/[G(2024)-G(2022)] = {(gd['2023'] - gd['2022']) / (gd['2024'] - gd['2022']):.3f}")
    for fam, spec in FAMILIES.items():
        gf = G[fam]
        print(f"  {fam}: x(pre) = G({spec['new']}) - G({spec['pre']}) = {gf[spec['new']] - gf[spec['pre']]:.3f}")


def print_report(res: dict) -> None:
    print_dose(res["dose"])
    for key, s in res["sets"].items():
        print(f"\n=== {key} ({s['role']}), {s['n_pairs']} pairs ===")
        for det in DETS:
            t = s["tests"][det]
            d1 = t["D1"]
            print(f"\n{det}:")
            print(f"  D1 y(2022) {d1['y2022']:+.3f} {fmt(d1['y2022_ci'])}  y(2023) {d1['y2023']:+.3f} {fmt(d1['y2023_ci'])}"
                  f"  f {d1['f']:.3f} {fmt(d1['f_ci'])}  f_hat {d1['f_hat']:.3f}  -> {d1['status']}")
            print(f"     AUROC 2022_2024 {d1['auroc']['2022_2024']:.4f}  2023_2024 {d1['auroc']['2023_2024']:.4f}"
                  f"  2024_2024 {d1['auroc']['2024_2024']:.4f}  f(AUROC) {d1['f_auroc']:.3f}")
            print("     by generator: " + "; ".join(f"{g} y22 {v['y2022']:+.3f} y23 {v['y2023']:+.3f} f {v['f']:.3f}"
                                                   for g, v in d1["by_generator"].items()))
            for r, v in t["D2"].items():
                print(f"  D2 y({r})/y(2022) {v['ratio']:.3f} {fmt(v['ci'])}  pred {v['pred']:.3f}  -> {v['status']}")
            print(f"  D3 g {t['D3']['g']:.3f} {fmt(t['D3']['ci'])}  g_hat {t['D3']['g_hat']:.3f}")
            for fam, v in t["D4"].items():
                pts = "  ".join(f"{r}:({p['x']:.3f},{p['y']:+.3f})" for r, p in v["points"].items())
                print(f"  D4 {fam:6} x {v['x']:.3f}  y {v['y']:+.3f} {fmt(v['y_ci'])}  k {v['k']:.2f}"
                      f"  rho {v['rho']:.2f} {fmt(v['rho_ci'])}  -> {v['status']}")
                print(f"     (x, y) per reference, scorer = newest: {pts}")
        for det in DETS:
            print(f"\n{det}: Cohen's d, rows = reference r, columns = scorer s")
            for fam in FAMILIES:
                ms = FAMILIES[fam]["models"]
                dd = s["families"][fam][det]["d"]
                print(f"  {fam}: r \\ s " + " ".join(f"{m:>7}" for m in ms))
                for r in ms:
                    print(f"    {r:>7} " + " ".join(f"{dd[f'{r}_{t}']:7.3f}" for t in ms))
    print("\n=== D5 (exploratory): d ~ 1 + diag + (G(s) - G(r)) over 25 cells ===")
    for key, fams in res["regression"].items():
        for fam, dets in fams.items():
            print(f"  {key} {fam}: " + "; ".join(
                f"{det} beta {v['beta_dose']:+.2f}, R2 {v['r2']:.2f} (no dose {v['r2_no_dose']:.2f})"
                for det, v in dets.items()))
    v = res["verdict"]
    print("\n=== verdict ===")
    print("D1 per detector: " + ", ".join(f"{d} {x}" for d, x in v["D1"].items()))
    print("D4 per detector: " + "; ".join(f"{d} " + ", ".join(f"{f} {x}" for f, x in fs.items())
                                         for d, fs in v["D4"].items()))
    print(f"dose-response: {v['dose_response']}")


def fmt(c: list[float]) -> str:
    return f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k != "kb"}
    if isinstance(o, (list, tuple)):
        return [strip(v) for v in o]
    if isinstance(o, np.ndarray):
        return None
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--outputs", required=True, help="directory holding s08_grid.jsonl, s10_A.jsonl, s11_grid.jsonl")
    ap.add_argument("--dose-only", action="store_true", help="print the predictor G and stop (no outcome computed)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--json")
    args = ap.parse_args()
    out = Path(args.outputs)
    fams = {f: load(out / spec["file"]) for f, spec in FAMILIES.items()}
    if args.dose_only:
        bc = pair_ids(fams, "B", "C")
        print(f"B/C pairs common to all three grids: {len(bc)}")
        print_dose({f: dose(rows, FAMILIES[f]["models"], bc) for f, rows in fams.items()})
        return
    res = evaluate(fams, n_boot=args.n_boot)
    print_report(res)
    if args.json:
        Path(args.json).write_text(json.dumps(strip(res), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
