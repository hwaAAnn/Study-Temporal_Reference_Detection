"""Experiment 07: the 01 curves on the TiC-LM Oracle series (seven from-scratch models, 2013-05 ... 2024-07).

Same curves and pairing as year_curve.py (G, per-generator G, G_D, G_p, S, U, U_B, M_y), on the cutoff time
axis t(v) = end of the cutoff month. Per curve:
  trend   line through the six pre-ChatGPT cutoffs (2013-05 ... 2023-01) against t(v)
  r       curve(2024-07) - trend(t(2024-07));  z = r / (s * sqrt(1 + 1/6 + (t - tbar)^2 / Sxx)), 4 df
  J       differences between neighbouring cutoffs; J_post = curve(2024-07) - curve(2023-01), compared with
          the largest |J| of the five pre-ChatGPT steps; J / (years between cutoffs) is reported too
  robust  (i) trend through 2013-05 ... 2021-01 only (3 df), (ii) mean of 2019-01, 2021-01, 2023-01
No change point (too few points). CIs: 2000 bootstrap draws over pairs, as in 01.

Example:
  python scripts/tic_curve.py --scores outputs/tic1b_20*.jsonl --polish data/polish_2022.jsonl \
      --json outputs/tic_eval.json > outputs/tic_eval.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from trd.metrics import auroc, tpr_at_fpr  # noqa: E402
from trd.tic import CUTOFFS, PRE, POST, cutoff_time  # noqa: E402
from year_curve import GENS, PERIODS, Set, ci, pair_set, polish_set  # noqa: E402

K = len(CUTOFFS)
CI_ = {c: k for k, c in enumerate(CUTOFFS)}
T = np.array([cutoff_time(c) for c in CUTOFFS])
NPRE = len(PRE)
T975 = {4: 2.776, 3: 3.182}


class Scores:
    """Cumulative per-token log p sums for every (text, cutoff); same interface as year_curve.Scores."""

    def __init__(self, paths: list[str]):
        per: dict[str, dict[str, np.ndarray]] = {}
        self.meta: dict[str, dict] = {}
        for p in paths:
            for line in Path(p).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                d = per.setdefault(r["id"], {})
                for c, v in r["lp"].items():
                    if c in d:
                        raise SystemExit(f"cutoff {c} given twice for {r['id']}")
                    d[c] = np.cumsum(np.asarray(v, dtype=np.float64))
                m = {k: r.get(k) for k in ("pair_id", "group", "label", "generator", "n_tokens")}
                if r["id"] in self.meta and self.meta[r["id"]]["n_tokens"] != m["n_tokens"]:
                    raise SystemExit(f"{r['id']}: token counts differ between score files")
                self.meta[r["id"]] = m
        got = sorted({c for d in per.values() for c in d})
        if got != CUTOFFS:
            raise SystemExit(f"need every cutoff {CUTOFFS}, got {got}")
        self.cs = {}
        for i, d in per.items():
            if len(d) != K:
                raise SystemExit(f"{i}: scored at {len(d)} cutoffs, not {K}")
            self.cs[i] = np.stack([d[c] for c in CUTOFFS])

    def l(self, i: str, n: int) -> np.ndarray:
        n = min(n, self.cs[i].shape[1])
        return self.cs[i][:, n - 1] / n

    def n(self, i: str) -> int:
        return self.cs[i].shape[1]

    def by_group(self, g: str) -> dict[str, str]:
        return {m["pair_id"]: i for i, m in self.meta.items() if m["group"] == g}


def trend_r(c: np.ndarray, idx: list[int]) -> tuple[float, float, float, float]:
    """Departure of the last cutoff from a line through the cutoffs idx; returns r, z, slope per year, residual s."""
    x, y = T[idx], c[idx]
    b, a = np.polyfit(x, y, 1)
    res = y - (a + b * x)
    dof = len(x) - 2
    s = float(np.sqrt((res ** 2).sum() / dof))
    sxx = float(((x - x.mean()) ** 2).sum())
    t = T[-1]
    r = float(c[-1] - (a + b * t))
    se = s * np.sqrt(1 + 1 / len(x) + (t - x.mean()) ** 2 / sxx)
    return r, (r / se if se > 0 else np.inf), float(b), s


def curve_stats(c: np.ndarray) -> dict:
    r, z, slope, s = trend_r(c, list(range(NPRE)))
    r5, z5, _, _ = trend_r(c, list(range(NPRE - 1)))
    J = np.diff(c)
    dt = np.diff(T)
    jp = J[:NPRE - 1]  # the five pre-ChatGPT steps
    sj = float(jp.std(ddof=1))
    zj = float((J[-1] - jp.mean()) / (sj * np.sqrt(1 + 1 / len(jp)))) if sj > 0 else np.inf
    return {"r": r, "z": z, "slope_pre": slope, "s_pre": s, "zJ": zj,
            "r_no2023": r5, "z_no2023": z5,
            "r_mean19": float(c[-1] - c[[CI_["201901"], CI_["202101"], CI_["202301"]]].mean()),
            "J": [float(j) for j in J], "J_rate": [float(j) for j in J / dt],
            "maxabsJ_pre": float(np.abs(J[:NPRE - 1]).max()), "J_post": float(J[-1]),
            "maxabsJrate_pre": float(np.abs((J / dt)[:NPRE - 1]).max()), "J_rate_post": float((J / dt)[-1])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--polish", required=True, help="05-a versions with orig_id and edit_sim")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    sc = Scores(args.scores)
    print(f"texts: {len(sc.cs)}; cutoffs {', '.join(CUTOFFS)}")
    sets: dict[str, Set] = {}
    sets["G"], bc_pairs = pair_set(sc, "B", "C", "G")
    for g in GENS:
        sets[f"G[{g}]"], _ = pair_set(sc, "B", "C", f"G[{g}]", gen=g)
    sets["G_D"], _ = pair_set(sc, "B", "D", "G_D")
    for name, hg, ag in PERIODS[:-1]:
        sets[f"G_{name}"], _ = pair_set(sc, hg, ag, f"G_{name}")
    sets["S"] = polish_set(sc, args.polish)
    for y in ("21", "22"):
        st, _ = pair_set(sc, f"W{y}", f"CW{y}", f"M_{y}")
        st.num = -st.num
        sets[f"M_{y}"] = st
    for k, st in sets.items():
        print(f"  {k:<28} rows {len(st.cl):>5}  clusters {st.K:>4}")

    def derived(cv):
        out = dict(cv)
        out["U"] = cv["G_2022"] - cv["G_2024"]
        out["U_B"] = cv["G_2022"] - cv["G"]
        return out

    curves = derived({k: st.curve() for k, st in sets.items()})
    stats = {k: curve_stats(c) for k, c in curves.items()}
    rng = np.random.default_rng(args.seed)
    keys = ["r", "z", "zJ", "slope_pre", "r_no2023", "r_mean19", "maxabsJ_pre", "J_post", "J_rate_post"]
    boot = {k: {q: [] for q in keys + ["J"]} for k in curves}
    boot_curves = {k: [] for k in curves}
    for _ in range(args.n_boot):
        cv = {}
        for k, st in sets.items():
            wk = np.bincount(rng.integers(0, st.K, st.K), minlength=st.K).astype(float)
            cv[k] = st.curve(wk)
        cv = derived(cv)
        for k, c in cv.items():
            s = curve_stats(c)
            for q in keys + ["J"]:
                boot[k][q].append(s[q])
            boot_curves[k].append(c)

    res = {"cutoffs": CUTOFFS, "t": T.tolist(), "curves": {}, "n": {k: int(len(st.cl)) for k, st in sets.items()}}
    for k, c in curves.items():
        bc = np.array(boot_curves[k])
        e = {"curve": c.tolist(), "curve_ci": [ci(bc[:, j]) for j in range(K)], **stats[k]}
        for q in keys:
            e[f"{q}_ci"] = ci(boot[k][q])
        J = np.array(boot[k]["J"])
        e["J_ci"] = [ci(J[:, j]) for j in range(J.shape[1])]
        res["curves"][k] = e

    print("\n=== curves: value per cutoff (nats/token) ===")
    print(f"{'curve':<26}" + "".join(f"{c:>9}" for c in CUTOFFS))
    for k in curves:
        print(f"{k:<26}" + "".join(f"{v:>+9.3f}" for v in curves[k]))
    print("\n=== departure of 2024-07 from the pre-ChatGPT line (95% bootstrap CI over texts; z: 4 df, crit 2.776) ===")
    print(f"{'curve':<26}{'slope/yr':>10}{'s_pre':>8}{'r':>8}{'CI':>18}{'z':>7}{'r (no 2023-01)':>16}{'z':>7}{'r (mean 19-23)':>16}")
    for k, e in res["curves"].items():
        print(f"{k:<26}{e['slope_pre']:>+10.4f}{e['s_pre']:>8.3f}{e['r']:>+8.3f}  [{e['r_ci'][0]:+.3f}, {e['r_ci'][1]:+.3f}]"
              f"{e['z']:>7.1f}{e['r_no2023']:>+16.3f}{e['z_no2023']:>7.1f}{e['r_mean19']:>+16.3f}")
    print("\n=== steps between cutoffs J (per-year rate in brackets) ===")
    steps = [f"{a[2:4]}{a[4:]}>{b[2:4]}{b[4:]}" for a, b in zip(CUTOFFS[:-1], CUTOFFS[1:])]
    print(f"{'curve':<26}" + "".join(f"{s:>16}" for s in steps) + f"{'max|J| pre':>12}")
    for k, e in res["curves"].items():
        print(f"{k:<26}" + "".join(f"{j:>+8.3f} ({jr:+.3f})" for j, jr in zip(e["J"], e["J_rate"])) + f"{e['maxabsJ_pre']:>12.3f}  zJ {e['zJ']:+.1f}")

    y = np.r_[np.zeros(len(bc_pairs)), np.ones(len(bc_pairs))]
    lh = np.array([sc.l(h, n) for _, h, _, n in bc_pairs])
    la = np.array([sc.l(a, n) for _, _, a, n in bc_pairs])
    det = {"ll": {}, "delta_last": {}}
    for c in CUTOFFS:
        x = np.r_[lh[:, CI_[c]], la[:, CI_[c]]]
        det["ll"][c] = {"auroc": auroc(y, x), "tpr5": tpr_at_fpr(y, x, 0.05)}
        if c != CUTOFFS[-1]:
            x = np.r_[lh[:, -1] - lh[:, CI_[c]], la[:, -1] - la[:, CI_[c]]]
            det["delta_last"][c] = {"auroc": auroc(y, x), "tpr5": tpr_at_fpr(y, x, 0.05)}
    res["detection"], res["levels"] = det, {"B": lh.mean(0).tolist(), "C": la.mean(0).tolist()}
    print("\n=== detection C vs B (exploratory): AUROC (TPR@5%FPR) ===")
    print(f"{'cutoff v':<12}{'log p_v':>18}{'Delta 2024-07 - v':>22}")
    for c in CUTOFFS:
        d2 = det["delta_last"].get(c)
        print(f"{c:<12}{det['ll'][c]['auroc']:>11.3f} ({det['ll'][c]['tpr5']:.2f})"
              + (f"{d2['auroc']:>15.3f} ({d2['tpr5']:.2f})" if d2 else ""))

    e = res["curves"]
    verdict = {
        "H1": e["G"]["r_ci"][0] > 0 and e["G"]["z"] > T975[4],
        # synthetic null check (11/15): "J_post > max |J_pre|" alone held by chance, so the step must also
        # leave the 95% prediction interval of the five pre-ChatGPT steps (4 df)
        "H2": e["G"]["J_ci"][-1][0] > 0 and e["G"]["J_post"] > e["G"]["maxabsJ_pre"] and e["G"]["zJ"] > T975[4],
        "H3a": e["S"]["r_ci"][0] > 0 and e["S"]["z"] > T975[4],
        "H3b": e["G_D"]["r_ci"][1] < e["G"]["r"] / 3,
        "H4": e["U"]["r_ci"][0] > 0 and e["U"]["z"] > T975[4],
    }
    res["verdict"] = {k: bool(v) for k, v in verdict.items()}
    print("\n=== hypotheses ===")
    for k, v in res["verdict"].items():
        print(f"{k}: {'holds' if v else 'fails'}")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
