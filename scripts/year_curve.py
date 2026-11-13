"""Experiment 01: twelve-vintage curves (DatedGPT 2013-2024) of how much each vintage prefers AI-written text,
measured against its pre-ChatGPT trend, with the change point and a check of how the vintages' data was built.

Input: one score file per vintage from score_texts.py (--no-pair-truncate), merged by text id. Every
comparison cuts its two texts to the shorter of the two lengths (exact for a causal LM).

Curves (one value per vintage v; l_v(x) = mean log p_v over the first n tokens):
  G(v)    mean over B/C pairs of l_v(c) - l_v(b)                     main: preference for same-title AI text
  G_D(v)  B / D' (GPT-Neo) pairs                                       machine text without chat style
  G_p(v)  arXiv pairs by period: 2019, 2022, 2024, 2025+ (= G)
  S(v)    sum e * (l_v(p) - l_v(h)) / sum e^2 over the 05-a versions   polishing slope (same text, no knowledge shift)
  U(v)    G_2022(v) - G_2024(v); U_B(v) = G_2022(v) - G_2025+(v)       drift of human abstracts toward AI
  M_y(v)  FineWeb-Edu document minus its AI continuation, y = 21, 22   membership fingerprint
  f       [M_21(2021) - M_21(2022)] / [M_21(2021) - mean M_21(2013..2019)]   1 = per-year data, 0 = cumulative

Per curve: a line fitted to the pre-ChatGPT vintages 2013-2022; departures r(2023), r(2024) and their
z = r / prediction s.e. (8 df: the model-to-model noise seen before ChatGPT); year-to-year jumps J;
broken-stick change point tau (slope changes after vintage tau, tau = 2015..2023, least squares);
robustness: r from a 2017-2022 line and from the 2019-2022 mean.
CIs: 2000 bootstrap draws over pairs (papers for polishing, documents for the web), each data set apart.
They are the text-sampling error with the 12 models fixed; z carries the model noise.

Example:
  python scripts/year_curve.py --scores outputs/yc_20*.jsonl --polish data/polish_2022.jsonl \
      --json outputs/yc_eval.json > outputs/yc_eval.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trd.metrics import auroc, tpr_at_fpr

YEARS = list(range(2013, 2025))
YI = {y: k for k, y in enumerate(YEARS)}
PRE = [y for y in YEARS if y <= 2022]  # every crawl before (or within ten days of) ChatGPT's release
POST = [2023, 2024]
TAUS = list(range(2015, 2024))
T975_8 = 2.306  # t quantile, 8 df (10 pre-ChatGPT vintages, line fit)
PERIODS = [("2019", "H19", "C19"), ("2022", "H22", "C22"), ("2024", "H24", "C24"), ("2025+", "B", "C")]
GENS = ["qwen2.5-1.5b-instruct", "llama3.2-1b-instruct"]


class Scores:
    """Cumulative sums of per-token log p for every (text, vintage), so a mean over the first n tokens is O(1)."""

    def __init__(self, paths: list[str]):
        per: dict[str, dict[str, np.ndarray]] = {}
        self.meta: dict[str, dict] = {}
        for p in paths:
            for line in Path(p).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                d = per.setdefault(r["id"], {})
                for y, v in r["lp"].items():
                    if y in d:
                        raise SystemExit(f"vintage {y} given twice for {r['id']}")
                    d[y] = np.cumsum(np.asarray(v, dtype=np.float64))
                m = {k: r.get(k) for k in ("pair_id", "group", "label", "generator", "n_tokens")}
                if r["id"] in self.meta and self.meta[r["id"]]["n_tokens"] != m["n_tokens"]:
                    raise SystemExit(f"{r['id']}: token counts differ between score files")
                self.meta[r["id"]] = m
        years = sorted({y for d in per.values() for y in d})
        if years != [str(y) for y in YEARS]:
            raise SystemExit(f"need every vintage {YEARS[0]}-{YEARS[-1]}, got {years}")
        self.cs = {}
        for i, d in per.items():
            if len(d) != len(YEARS):
                raise SystemExit(f"{i}: scored by {len(d)} vintages, not {len(YEARS)}")
            self.cs[i] = np.stack([d[str(y)] for y in YEARS])  # (12, n)

    def l(self, i: str, n: int) -> np.ndarray:
        """Mean log p over the first n tokens, one value per vintage."""
        n = min(n, self.cs[i].shape[1])
        return self.cs[i][:, n - 1] / n

    def n(self, i: str) -> int:
        return self.cs[i].shape[1]

    def by_group(self, g: str) -> dict[str, str]:
        """pair_id -> text id for the texts of group g."""
        return {m["pair_id"]: i for i, m in self.meta.items() if m["group"] == g}


class Set:
    """Rows of per-vintage values with bootstrap clusters; curve(w) = sum w*num / sum w*den."""

    def __init__(self, num, den, cl, label: str):
        self.num, self.den = np.asarray(num, float), np.asarray(den, float)
        cl = np.asarray(cl)
        _, self.cl = np.unique(cl, return_inverse=True)
        self.K, self.label = int(self.cl.max()) + 1, label

    def curve(self, wk: np.ndarray | None = None) -> np.ndarray:
        w = np.ones(len(self.cl)) if wk is None else wk[self.cl]
        return (w[:, None] * self.num).sum(0) / (w * self.den).sum()


def pair_set(sc: Scores, hg: str, ag: str, label: str, gen: str | None = None) -> tuple[Set, list]:
    """Mean over pairs of l_v(ai) - l_v(human), both cut to the pair's shorter length."""
    hs, as_ = sc.by_group(hg), sc.by_group(ag)
    rows, keep = [], []
    for pid in sorted(set(hs) & set(as_)):
        h, a = hs[pid], as_[pid]
        if gen and sc.meta[a]["generator"] != gen:
            continue
        n = min(sc.n(h), sc.n(a))
        rows.append(sc.l(a, n) - sc.l(h, n))
        keep.append((pid, h, a, n))
    if not rows:
        raise SystemExit(f"no {hg}/{ag} pairs")
    return Set(rows, np.ones(len(rows)), [k[0] for k in keep], label), keep


def polish_set(sc: Scores, polish_path: str) -> Set:
    """Slope through 0 of l_v(polished) - l_v(original) on the share of words changed e (05-a)."""
    num, den, cl = [], [], []
    for line in Path(polish_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        p, h = str(r["id"]), f"{r['orig_id']}:h"
        if p not in sc.cs or h not in sc.cs:
            continue
        e = 1.0 - float(r["edit_sim"])
        n = min(sc.n(p), sc.n(h))
        num.append(e * (sc.l(p, n) - sc.l(h, n)))
        den.append(e * e)
        cl.append(r["orig_id"])
    if not num:
        raise SystemExit("no polished versions matched the scores")
    return Set(num, den, cl, "S")


def line_fit(x, y):
    b, a = np.polyfit(x, y, 1)
    return a, b


def curve_stats(c: np.ndarray) -> dict:
    """Pre-ChatGPT trend, departures, jumps, change point and robustness for one 12-vintage curve."""
    x = np.array(PRE, float)
    y = c[[YI[v] for v in PRE]]
    a, b = line_fit(x, y)
    res = y - (a + b * x)
    s = float(np.sqrt((res ** 2).sum() / (len(x) - 2)))
    sxx = float(((x - x.mean()) ** 2).sum())
    out = {"slope_pre": float(b), "s_pre": s}
    for v in POST:
        r = float(c[YI[v]] - (a + b * v))
        se = s * np.sqrt(1 + 1 / len(x) + (v - x.mean()) ** 2 / sxx)
        out[f"r{v}"], out[f"z{v}"] = r, (r / se if se > 0 else np.inf)
        a2, b2 = line_fit(np.arange(2017, 2023, dtype=float), c[[YI[w] for w in range(2017, 2023)]])
        out[f"r{v}_trend17"] = float(c[YI[v]] - (a2 + b2 * v))
        out[f"r{v}_mean19"] = float(c[YI[v]] - c[[YI[w] for w in range(2019, 2023)]].mean())
    J = np.diff(c)  # J[k] = c(YEARS[k+1]) - c(YEARS[k])
    out["J"] = [float(j) for j in J]
    pre_J = J[: YI[2022]]  # J(2014)..J(2022)
    out["maxabsJ_pre"] = float(np.abs(pre_J).max())
    out["maxJ_post"] = float(max(J[YI[2023] - 1], J[YI[2024] - 1]))
    xs = np.array(YEARS, float)
    sse = []
    for tau in TAUS:
        X = np.c_[np.ones_like(xs), xs - 2013, np.maximum(0, xs - tau)]
        coef, *_ = np.linalg.lstsq(X, c, rcond=None)
        sse.append(float(((c - X @ coef) ** 2).sum()))
    out["sse_tau"] = sse
    out["tau"] = TAUS[int(np.argmin(sse))]
    return out


def ci(xs, alpha=0.05):
    xs = np.asarray(xs, float)
    xs = xs[np.isfinite(xs)]
    return [float(np.quantile(xs, alpha / 2)), float(np.quantile(xs, 1 - alpha / 2))] if len(xs) else [np.nan, np.nan]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True, help="one score file per vintage (or files with several)")
    ap.add_argument("--polish", required=True, help="05-a versions with orig_id and edit_sim")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    sc = Scores(args.scores)
    print(f"texts: {len(sc.cs)}; vintages {YEARS[0]}-{YEARS[-1]}")

    # ---- data sets (each bootstrapped on its own clusters) ----
    sets: dict[str, Set] = {}
    sets["G"], bc_pairs = pair_set(sc, "B", "C", "G")
    for g in GENS:
        sets[f"G[{g}]"], _ = pair_set(sc, "B", "C", f"G[{g}]", gen=g)
    sets["G_D"], _ = pair_set(sc, "B", "D", "G_D")
    for name, hg, ag in PERIODS[:-1]:
        sets[f"G_{name}"], _ = pair_set(sc, hg, ag, f"G_{name}")
    sets["S"] = polish_set(sc, args.polish)
    # membership fingerprint: human document minus its AI continuation (sign flipped: + = human more likely)
    for y in ("21", "22"):
        st, _ = pair_set(sc, f"W{y}", f"CW{y}", f"M_{y}")
        st.num = -st.num
        sets[f"M_{y}"] = st
    for k, st in sets.items():
        print(f"  {k:<28} rows {len(st.cl):>5}  clusters {st.K:>4}")

    def derived(cv: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = dict(cv)
        out["G_2025+"] = cv["G"]
        out["U"] = cv["G_2022"] - cv["G_2024"]
        out["U_B"] = cv["G_2022"] - cv["G"]
        return out

    def f_drop(m21: np.ndarray) -> float:
        base = m21[[YI[v] for v in range(2013, 2020)]].mean()
        den = m21[YI[2021]] - base
        return float((m21[YI[2021]] - m21[YI[2022]]) / den) if den > 0 else np.nan

    curves = derived({k: st.curve() for k, st in sets.items()})
    stats = {k: curve_stats(c) for k, c in curves.items()}
    f_point = f_drop(curves["M_21"])

    # ---- bootstrap ----
    rng = np.random.default_rng(args.seed)
    keys = ["slope_pre", "r2023", "r2024", "r2023_trend17", "r2024_trend17", "r2023_mean19", "r2024_mean19",
            "maxabsJ_pre", "maxJ_post"]
    boot = {k: {q: [] for q in keys + ["tau", "J"]} for k in curves}
    boot_curves = {k: [] for k in curves}
    boot_f = []
    for _ in range(args.n_boot):
        cv = {}
        for k, st in sets.items():
            wk = np.bincount(rng.integers(0, st.K, st.K), minlength=st.K).astype(float)
            cv[k] = st.curve(wk)
        cv = derived(cv)
        for k, c in cv.items():
            s = curve_stats(c)
            for q in keys + ["tau", "J"]:
                boot[k][q].append(s[q])
            boot_curves[k].append(c)
        boot_f.append(f_drop(cv["M_21"]))

    # ---- report ----
    res = {"years": YEARS, "curves": {}, "f": {"value": f_point, "ci": ci(boot_f)}, "n": {k: int(len(st.cl)) for k, st in sets.items()}}
    for k, c in curves.items():
        bc = np.array(boot_curves[k])
        s = stats[k]
        entry = {"curve": c.tolist(), "curve_ci": [ci(bc[:, j]) for j in range(len(YEARS))], **s}
        for q in keys:
            entry[f"{q}_ci"] = ci(boot[k][q])
        J = np.array(boot[k]["J"])
        entry["J_ci"] = [ci(J[:, j]) for j in range(J.shape[1])]
        taus = np.array(boot[k]["tau"])
        entry["tau_boot"] = {str(t): float((taus == t).mean()) for t in TAUS}
        res["curves"][k] = entry

    print("\n=== curves: value per vintage (nats/token) ===")
    print(f"{'curve':<26}" + "".join(f"{y:>8}" for y in YEARS))
    for k in curves:
        print(f"{k:<26}" + "".join(f"{v:>+8.3f}" for v in curves[k]))

    print("\n=== departure from the 2013-2022 line (95% bootstrap CI over texts; z = model-noise scale, 8 df) ===")
    hdr = f"{'curve':<26}{'slope/yr':>10}{'s_pre':>8}{'r2023':>8}{'CI':>18}{'z':>7}{'r2024':>8}{'CI':>18}{'z':>7}{'tau+1':>7}{'P(boot)':>9}"
    print(hdr)
    for k, e in res["curves"].items():
        t = e["tau"]
        print(f"{k:<26}{e['slope_pre']:>+10.4f}{e['s_pre']:>8.3f}{e['r2023']:>+8.3f}"
              f"  [{e['r2023_ci'][0]:+.3f}, {e['r2023_ci'][1]:+.3f}]{e['z2023']:>7.1f}{e['r2024']:>+8.3f}"
              f"  [{e['r2024_ci'][0]:+.3f}, {e['r2024_ci'][1]:+.3f}]{e['z2024']:>7.1f}{t + 1:>7}{e['tau_boot'][str(t)]:>9.2f}")

    print("\n=== robustness: r from a 2017-2022 line / from the 2019-2022 mean ===")
    for k, e in res["curves"].items():
        print(f"{k:<26} r2023 {e['r2023_trend17']:+.3f} [{e['r2023_trend17_ci'][0]:+.3f}, {e['r2023_trend17_ci'][1]:+.3f}]"
              f" / {e['r2023_mean19']:+.3f}   r2024 {e['r2024_trend17']:+.3f} [{e['r2024_trend17_ci'][0]:+.3f}, "
              f"{e['r2024_trend17_ci'][1]:+.3f}] / {e['r2024_mean19']:+.3f} [{e['r2024_mean19_ci'][0]:+.3f}, {e['r2024_mean19_ci'][1]:+.3f}]")

    print("\n=== year-to-year jumps J(v) = curve(v) - curve(v-1) ===")
    print(f"{'curve':<26}" + "".join(f"{y:>8}" for y in YEARS[1:]) + f"{'max|J| pre':>12}")
    for k, e in res["curves"].items():
        print(f"{k:<26}" + "".join(f"{j:>+8.3f}" for j in e["J"]) + f"{e['maxabsJ_pre']:>12.3f}")

    print("\n=== change point: SSE of the broken line, slope change after vintage tau ===")
    print(f"{'curve':<26}" + "".join(f"{t:>8}" for t in TAUS))
    for k, e in res["curves"].items():
        print(f"{k:<26}" + "".join(f"{v:>8.4f}" for v in e["sse_tau"]))

    m21 = curves["M_21"]
    peak = YEARS[int(np.argmax(m21))]
    peak_pre = PRE[int(np.argmax(m21[[YI[v] for v in PRE]]))]
    m22_pre = PRE[int(np.argmax(curves["M_22"][[YI[v] for v in PRE]]))]
    print(f"\n=== data per vintage: per-year or cumulative ===\nM_21 peak: {peak} (among 2013-2022: {peak_pre}); "
          f"M_22 peak among 2013-2022: {m22_pre}")
    print(f"f = drop of the 2021 peak at 2022 = {f_point:.2f} [{res['f']['ci'][0]:.2f}, {res['f']['ci'][1]:.2f}]"
          "   (1 = per-year crawls only, 0 = cumulative)")
    res.update({"M21_peak": peak, "M21_peak_pre": peak_pre, "M22_peak_pre": m22_pre})

    # ---- sanity: the 3-model numbers of 00/02/05 ----
    print("\n=== check against 00/02/05 (same texts, same lengths) ===")
    for k, ref in (("G", "00 B/C gap_post 0.263, gap_pre -0.024"), ("S", "05 s_post 0.216, s_pre -0.013"),
                   ("G_2019", "02 0.380 / +0.015"), ("G_2022", "02 0.356 / -0.014"), ("G_2024", "02 0.281 / -0.015"),
                   ("U", "02 polish_24 +0.074")):
        c = curves[k]
        print(f"{k:<8} v2024-v2021 {c[YI[2024]] - c[YI[2021]]:+.3f}   v2021-v2018 {c[YI[2021]] - c[YI[2018]]:+.3f}   ({ref})")

    # ---- detection (exploratory): B vs C at the pair's common length ----
    y = np.r_[np.zeros(len(bc_pairs)), np.ones(len(bc_pairs))]
    lh = np.array([sc.l(h, n) for _, h, _, n in bc_pairs])  # (pairs, 12)
    la = np.array([sc.l(a, n) for _, _, a, n in bc_pairs])
    det = {"ll": {}, "delta2024": {}}
    for v in YEARS:
        x = np.r_[lh[:, YI[v]], la[:, YI[v]]]
        det["ll"][str(v)] = {"auroc": auroc(y, x), "tpr5": tpr_at_fpr(y, x, 0.05)}
        if v < 2024:
            x = np.r_[lh[:, YI[2024]] - lh[:, YI[v]], la[:, YI[2024]] - la[:, YI[v]]]
            det["delta2024"][str(v)] = {"auroc": auroc(y, x), "tpr5": tpr_at_fpr(y, x, 0.05)}
    res["detection"] = det
    res["levels"] = {"B": lh.mean(0).tolist(), "C": la.mean(0).tolist()}
    print("\n=== detection C vs B (exploratory): AUROC (TPR@5%FPR) ===")
    print(f"{'vintage v':<12}{'log p_v':>18}{'Delta 2024-v':>18}")
    for v in YEARS:
        d2 = det["delta2024"].get(str(v))
        print(f"{v:<12}{det['ll'][str(v)]['auroc']:>11.3f} ({det['ll'][str(v)]['tpr5']:.2f})"
              + (f"{d2['auroc']:>11.3f} ({d2['tpr5']:.2f})" if d2 else ""))

    # ---- hypotheses (pre-registered criteria) ----
    e = res["curves"]
    g24 = e["G"]["r2024"]
    verdict = {
        "H1": e["G"]["r2024_ci"][0] > 0 and e["G"]["z2024"] > T975_8,
        "H2": e["G"]["tau"] in (2022, 2023) and e["G"]["maxJ_post"] > e["G"]["maxabsJ_pre"],
        "H3a": e["S"]["r2024_ci"][0] > 0 and e["S"]["z2024"] > T975_8,
        "H3b": e["G_D"]["r2024_ci"][1] < g24 / 3,
        "H4": e["U"]["r2024_ci"][0] > 0 and e["U"]["z2024"] > T975_8,
    }
    verdict = {k: bool(v) for k, v in verdict.items()}
    if not np.isfinite(f_point):
        q5 = "undetermined (no 2021 peak in M_21)"
    else:
        q5 = "per-year" if f_point >= 0.5 else ("cumulative" if f_point <= 0.2 else "mixed")
    res["verdict"], res["q5"] = verdict, q5
    print("\n=== hypotheses ===")
    for k, v in verdict.items():
        print(f"{k}: {'holds' if v else 'fails'}")
    print(f"Q5 (data per vintage): {q5}")

    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
