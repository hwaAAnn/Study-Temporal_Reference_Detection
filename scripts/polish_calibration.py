"""Experiment 05: express the post-ChatGPT rise of human Delta in units of LLM polishing.

Calibration (05-a). Pre-ChatGPT abstracts h_i (H22, H19) were polished by local chat models at three
intensities (polish_texts.py). Scores come from score_texts.py --no-pair-truncate; every comparison
below is cut here to a common length (exact for causal LMs). For paper i and polished version v:
    d_iv = Delta(p_iv) - Delta(h_i)                                      paired shift, equal length
    g_iv = [Delta(c_i) - Delta(h_i)] - [Delta(c_i) - Delta(p_iv)]          gap shift, each gap cut as in 02
    e_iv = 1 - edit_sim(h_i, p_iv)                                       share of the text changed
    s    = sum(e d) / sum(e^2)                                           Delta shift per unit edit (line through 0)
Transfer (05-b). APT-Eval abstracts polished by large LLMs and by our local models, same 51 originals:
    R_pol = s_pol / s_local
Population (02's scores and estimator, unchanged): shift_T = gap(H22) - gap(T), T = H24 (2024), B (2025+)
    beta_T     = shift_T / s              AI-edited share of the text, local-model polishing
    beta_T,pol = shift_T / (s * R_pol)    ... if authors polished like `pol`
Also reported: per-level shifts and the share of abstracts they imply, the AI-free placebo (2021-2018),
AI-marker words, detection of polished text, a pre-ChatGPT trend and an LLM-topic-free check, and the
shape of the 2024 distribution (a mixture of polished and untouched abstracts, or a uniform shift).

Bootstrap: H22 papers (slope and gap(H22) together), T papers, H19 papers and APT originals are each
resampled with all of their versions.

Post hoc (added after the first scored run and labelled so in the output; earlier numbers unchanged):
the curvature d = b1 e + b2 e^2 through 0, and detection with every text cut to its first 100 tokens.

Example:
  python scripts/polish_calibration.py --scores05 outputs/polish.jsonl --scores02 outputs/mia.jsonl \
      --texts data/02_all.jsonl data/polish_2022.jsonl data/polish_2019.jsonl data/apt_originals.jsonl \
              data/apt_polished.jsonl data/polish_apt.jsonl --json outputs/polish_eval.json
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from trd.data import normalise_rows
from trd.metrics import auroc, tpr_at_fpr

LOCAL = ["qwen2.5-1.5b-instruct", "llama3.2-1b-instruct"]
APT_LARGE = ["GPT-4o", "Llama3.1-70B", "DeepSeek-V3", "Llama3-8B"]
APT_OTHER = ["Llama2-7B"]  # often answers off-task in APT-Eval; reported apart
LEVELS = ["proof", "polish", "rewrite"]
APT_DEGREES = ["extreme_minor", "minor", "slight_major", "major"]
BINS = np.array([0, .05, .1, .2, .3, .4, .5, .6, .7, 1.01])  # share-of-text-changed bins for the dose-response
# words reported as markers of LLM-assisted writing (same 18 stems as 02)
MARK = ["delve", "intricate", "showcas", "pivotal", "meticulous", "commendable", "notably", "realm", "underscore",
        "leverag", "enhanc", "comprehensive", "crucial", "various", "seamless", "innovative", "robust", "ensur"]
MARK_RE = re.compile(r"\b(" + "|".join(MARK) + r")", re.I)
LLM_TOPIC = re.compile(r"\b(LLMs?|large language models?|language models?|GPT-?\d*|ChatGPT|prompt\w*|"
                       r"in-context|instruction[- ]tun\w*|chatbots?|generative AI)\b", re.I)


class Store:
    """Per-text cumulative sums of the per-token log-ratios, so a mean over the first n tokens is O(1)."""

    def __init__(self, path: str):
        self.cs, self.n, self.group = {}, {}, {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            lp = {k: np.asarray(v, dtype=np.float64) for k, v in r["lp"].items()}
            self.n[r["id"]] = len(lp["2024"])
            self.group[r["id"]] = r["group"]
            self.cs[r["id"]] = {"post": np.cumsum(lp["2024"] - lp["2021"]),
                                "pre": np.cumsum(lp["2021"] - lp["2018"]), "ll": np.cumsum(lp["2024"])}

    def mean(self, i: str, key: str, n: int | None = None) -> float:
        n = self.n[i] if n is None else min(n, self.n[i])
        return float(self.cs[i][key][n - 1] / n)

    def ids(self, group: str) -> list[str]:
        return [i for i, g in self.group.items() if g == group]


def slope(e: np.ndarray, d: np.ndarray) -> float:
    return float((e * d).sum() / (e * e).sum())


def frac_year(s: str) -> float:
    t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return t.year + (t.timetuple().tm_yday - 0.5) / 365.25


def marker_rate(text: str) -> float:
    return 1000 * len(MARK_RE.findall(text)) / max(1, len(text.split()))


def ci(xs, alpha=0.05):
    xs = np.asarray(xs, dtype=float)
    xs = xs[np.isfinite(xs)]
    return [float(np.quantile(xs, alpha / 2)), float(np.quantile(xs, 1 - alpha / 2))] if len(xs) else [np.nan, np.nan]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores05", required=True, help="score_texts.py --no-pair-truncate output (H/C/P/APT texts)")
    ap.add_argument("--scores02", required=True, help="02's pair-truncated scores (H22, H24, B and anchors)")
    ap.add_argument("--texts", nargs="+", required=True, help="data files with the texts (pairs and text rows)")
    ap.add_argument("--base", default="22:H22:C22", help="pre-ChatGPT reference period label:human:ai")
    ap.add_argument("--trend-from", default="19:H19:C19", help="earlier pre-ChatGPT period for the trend check")
    ap.add_argument("--targets", nargs="+", default=["24:H24:C24", "25:B:C"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # ---------- texts and metadata ----------
    text, meta = {}, {}
    for p in args.texts:
        rows = [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
        pair_rows = [r for r in rows if "human" in r and "ai" in r]
        for it in normalise_rows(pair_rows):
            text[it["id"]] = it["text"]
        for r in pair_rows:
            meta[str(r["id"])] = {"title": r.get("title", ""), "published": r.get("published", "")}
        for r in rows:
            if "text" in r:
                text[str(r["id"])] = r["text"]
                meta[str(r["id"])] = r

    s05, s02 = Store(args.scores05), Store(args.scores02)
    base_lbl, base_h, base_c = args.base.split(":")

    # ---------- 05-a: local polishing of pre-ChatGPT abstracts ----------
    def versions(period_tag: str):
        """rows of (paper, level, generator, e, d_post, d_pre, g_post, g_pre, d_ll, d_mark)."""
        out = []
        for lv in LEVELS:
            for pid_v in s05.ids(f"P{period_tag}.{lv}"):
                paper = pid_v.split("~")[0]
                h, c = f"{paper}:h", f"{paper}:a"
                if h not in s05.n or c not in s05.n:
                    continue
                m = meta[pid_v]
                nph, nhc, npc = min(s05.n[pid_v], s05.n[h]), min(s05.n[h], s05.n[c]), min(s05.n[pid_v], s05.n[c])
                row = {"paper": paper, "level": lv, "gen": m["generator"], "e": 1 - m["edit_sim"],
                       "d_mark": marker_rate(text[pid_v]) - marker_rate(text[h])}
                for k in ("post", "pre"):
                    row[f"d_{k}"] = s05.mean(pid_v, k, nph) - s05.mean(h, k, nph)
                    row[f"g_{k}"] = ((s05.mean(c, k, nhc) - s05.mean(h, k, nhc))
                                     - (s05.mean(c, k, npc) - s05.mean(pid_v, k, npc)))
                row["d_ll"] = s05.mean(pid_v, "ll", nph) - s05.mean(h, "ll", nph)
                out.append(row)
        return out

    v22 = versions(base_lbl)
    v19 = versions(args.trend_from.split(":")[0])
    papers22 = sorted({r["paper"] for r in v22} | {i[:-2] for i in s02.ids(base_h)})

    # check: 05's full-length scores reproduce 02's gap(H22) when cut the same way
    g05 = np.mean([s05.mean(f"{p}:a", "post", min(s05.n[f"{p}:h"], s05.n[f"{p}:a"]))
                   - s05.mean(f"{p}:h", "post", min(s05.n[f"{p}:h"], s05.n[f"{p}:a"]))
                   for p in papers22 if f"{p}:h" in s05.n])
    g02 = np.mean([s02.mean(f"{p}:a", "post") - s02.mean(f"{p}:h", "post") for p in papers22 if f"{p}:h" in s02.n])
    print(f"gap_post({base_h}): 05 rescored {g05:.4f} vs 02 {g02:.4f}")

    # from-scratch point: the same-title AI anchor as a 'complete rewrite' of h
    scratch = []
    for p in papers22:
        h, c = f"{p}:h", f"{p}:a"
        if h in s05.n and c in s05.n:
            n = min(s05.n[h], s05.n[c])
            scratch.append({"e": 1 - difflib.SequenceMatcher(
                None, text[h].lower().split(), text[c].lower().split(), autojunk=False).ratio(),
                "d_post": s05.mean(c, "post", n) - s05.mean(h, "post", n),
                "d_pre": s05.mean(c, "pre", n) - s05.mean(h, "pre", n)})

    # ---------- 05-b: APT-Eval ----------
    apt = defaultdict(list)  # polisher -> rows
    for i, g in s05.group.items():
        if not g.startswith("APT.") or g == "APT.H":
            continue
        m = meta[i]
        o = m["orig_id"]
        if o not in s05.n:
            continue
        if not 0.5 <= m.get("word_ratio", 1.0) <= 2.0:
            continue
        n = min(s05.n[i], s05.n[o])
        apt[m["polisher"]].append({"orig": o, "level": m["level"], "e": 1 - m["edit_sim"],
                                   "d_post": s05.mean(i, "post", n) - s05.mean(o, "post", n),
                                   "d_pre": s05.mean(i, "pre", n) - s05.mean(o, "pre", n),
                                   "d_mark": marker_rate(text[i]) - marker_rate(text[o]), "id": i})
    apt_origs = sorted({r["orig"] for rows in apt.values() for r in rows})
    apt["local"] = [r for g in LOCAL for r in apt.get(g, [])]
    apt["others"] = [r for g in APT_LARGE if g != "GPT-4o" for r in apt.get(g, [])]  # pooled, for the figure

    # ---------- population shifts (02's estimator) ----------
    def period_gap_items(store, hg):
        items = []
        for i in store.ids(hg):
            p = i[:-2]
            c = f"{p}:a"
            if c in store.n:
                items.append((p, store.mean(c, "post") - store.mean(i, "post"), store.mean(i, "post"),
                              store.mean(c, "post")))
        return items

    base_items = {p: (g, h, c) for p, g, h, c in period_gap_items(s02, base_h)}
    targets = {}
    for spec in args.targets:
        lbl, hg, cg = spec.split(":")
        targets[lbl] = {"items": period_gap_items(s02, hg), "hg": hg}
    t_lbl, t_h, _ = args.trend_from.split(":")
    trend_items = period_gap_items(s02, t_h)

    def mean_year(hg):
        ys = [frac_year(meta[i[:-2]]["published"]) for i in s02.ids(hg) if meta.get(i[:-2], {}).get("published")]
        return float(np.mean(ys)) if ys else np.nan

    years = {base_lbl: mean_year(base_h), t_lbl: mean_year(t_h)}
    for lbl, t in targets.items():
        years[lbl] = mean_year(t["hg"])
    llm_topic = {p for p, m in meta.items() if isinstance(m, dict) and LLM_TOPIC.search(m.get("title", "") or "")}

    # ---------- arrays for the bootstrap ----------
    P = {p: k for k, p in enumerate(papers22)}
    NP = len(papers22)

    def per_paper(rows, key, sel=lambda r: True):
        A, B = np.zeros(NP), np.zeros(NP)
        for r in rows:
            if sel(r):
                A[P[r["paper"]]] += r["e"] * r[key]
                B[P[r["paper"]]] += r["e"] ** 2
        return A, B

    AB = {("all", k): per_paper(v22, k) for k in ("d_post", "g_post", "d_pre", "d_mark")}
    # post hoc (added after scoring): curvature, d = b1 e + b2 e^2 through 0, from per-paper sums
    quad = {k: np.zeros(NP) for k in ("e2", "e3", "e4", "ed", "e2d")}
    for r in v22:
        k, e_, d_ = P[r["paper"]], r["e"], r["d_post"]
        for name, v in (("e2", e_ ** 2), ("e3", e_ ** 3), ("e4", e_ ** 4), ("ed", e_ * d_), ("e2d", e_ ** 2 * d_)):
            quad[name][k] += v
    for g in LOCAL:
        AB[(g, "d_post")] = per_paper(v22, "d_post", lambda r, g=g: r["gen"] == g)
    lvl_sum = {}
    for lv in LEVELS:
        S, N = np.zeros(NP), np.zeros(NP)
        for r in v22:
            if r["level"] == lv:
                S[P[r["paper"]]] += r["g_post"]
                N[P[r["paper"]]] += 1
        lvl_sum[lv] = (S, N)
    bin_sum = {}  # binned dose-response, so each bin mean gets a paper-bootstrap interval
    for b, (lo, hi) in enumerate(zip(BINS, BINS[1:])):
        for key in ("e", "d_post", "d_pre"):
            S, N = np.zeros(NP), np.zeros(NP)
            for r in v22:
                if lo <= r["e"] < hi:
                    S[P[r["paper"]]] += r[key]
                    N[P[r["paper"]]] += 1
            bin_sum[(b, key)] = (S, N)
    base_gap =np.array([base_items[p][0] if p in base_items else np.nan for p in papers22])
    base_has = np.isfinite(base_gap)
    base_gap0 = np.where(base_has, base_gap, 0.0)

    O = {o: k for k, o in enumerate(apt_origs)}
    NO = len(apt_origs)
    apt_AB = {}
    for pol, rows in apt.items():
        for key in ("d_post", "d_pre", "d_mark"):
            A, B = np.zeros(NO), np.zeros(NO)
            for r in rows:
                A[O[r["orig"]]] += r["e"] * r[key]
                B[O[r["orig"]]] += r["e"] ** 2
            apt_AB[(pol, key)] = (A, B)

    def hits_words(i):
        return len(MARK_RE.findall(text[i])), len(text[i].split())

    base_hw = np.array([hits_words(f"{p}:h") if p in base_items else (0, 0) for p in papers22], dtype=float)
    tgt_hw = {lbl: np.array([hits_words(f"{p}:h") for p, _, _, _ in t["items"]], dtype=float)
              for lbl, t in targets.items()}
    tgt_arr = {lbl: np.array([g for _, g, _, _ in t["items"]]) for lbl, t in targets.items()}
    tgt_trim = {lbl: np.array([p not in llm_topic for p, _, _, _ in t["items"]]) for lbl, t in targets.items()}
    base_trim = np.array([p not in llm_topic for p in papers22])
    trend_arr = np.array([g for _, g, _, _ in trend_items])

    def estimates(ip, io, it, itr):
        """All bootstrap-able quantities for one draw of papers (ip), APT originals (io), targets (it), trend (itr)."""
        out = {}
        w = np.bincount(ip, minlength=NP).astype(float)
        for (name, key), (A, B) in AB.items():
            out[f"s_{name}_{key}"] = float((w * A).sum() / (w * B).sum())
        for lv, (S, N) in lvl_sum.items():
            out[f"level_{lv}"] = float((w * S).sum() / (w * N).sum())
        for (b, key), (S, N) in bin_sum.items():
            den = (w * N).sum()
            out[f"bin{b}_{key}"] = float((w * S).sum() / den) if den > 0 else np.nan
        q = {k: float((w * v).sum()) for k, v in quad.items()}
        out["quad_b1"], out["quad_b2"] = (float(x) for x in np.linalg.solve(
            [[q["e2"], q["e3"]], [q["e3"], q["e4"]]], [q["ed"], q["e2d"]]))
        rb = 1000 * (w * base_hw[:, 0]).sum() / (w * base_hw[:, 1]).sum()
        out["mark_rate_base"] = rb
        gb =float((w * base_gap0).sum() / (w * base_has).sum())
        gb_trim = float((w * base_gap0 * base_trim).sum() / (w * base_has * base_trim).sum())
        out["gap_base"] = gb
        wo = np.bincount(io, minlength=NO).astype(float)
        for (pol, key), (A, B) in apt_AB.items():
            out[f"apt_s_{pol}_{key}"] = float((wo * A).sum() / (wo * B).sum()) if (wo * B).sum() > 0 else np.nan
        g_tr = float(trend_arr[itr].mean())
        tau = (gb - g_tr) / (years[base_lbl] - years[t_lbl])  # gap change per year before ChatGPT
        out["trend_per_year"] = tau
        s_loc = out["s_all_d_post"]
        for lbl, idx in it.items():
            gt = tgt_arr[lbl][idx]
            shift = gb - gt.mean()
            out[f"shift_{lbl}"] = shift
            out[f"shift_{lbl}_trend"] = shift + tau * (years[lbl] - years[base_lbl])
            keep = tgt_trim[lbl][idx]
            out[f"shift_{lbl}_nollm"] = gb_trim - gt[keep].mean()
            out[f"beta_{lbl}_local"] = shift / s_loc
            out[f"beta_{lbl}_local_trend"] = out[f"shift_{lbl}_trend"] / s_loc
            out[f"beta_{lbl}_local_nollm"] = out[f"shift_{lbl}_nollm"] / s_loc
            for pol in APT_LARGE + APT_OTHER + ["others"]:
                r = out.get(f"apt_s_{pol}_d_post", np.nan) / out["apt_s_local_d_post"]
                out[f"R_{pol}"] = r
                out[f"beta_{lbl}_{pol}"] = shift / (s_loc * r)
            for lv in LEVELS:
                out[f"alpha_{lbl}_{lv}"] = shift / out[f"level_{lv}"]
            hw = tgt_hw[lbl][idx]
            rt = 1000 * hw[:, 0].sum() / hw[:, 1].sum()
            out[f"mark_rate_{lbl}"] = rt
            out[f"betaW_{lbl}_local"] = (rt - rb) / out["s_all_d_mark"]
            for pol in APT_LARGE:
                out[f"betaW_{lbl}_{pol}"] = (rt - rb) / out.get(f"apt_s_{pol}_d_mark", np.nan)
        return out

    full = {lbl: np.arange(len(t["items"])) for lbl, t in targets.items()}
    est = estimates(np.arange(NP), np.arange(NO), full, np.arange(len(trend_items)))
    boots = defaultdict(list)
    for _ in range(args.n_boot):
        b = estimates(rng.integers(0, NP, NP), rng.integers(0, NO, NO),
                      {lbl: rng.integers(0, len(t["items"]), len(t["items"])) for lbl, t in targets.items()},
                      rng.integers(0, len(trend_items), len(trend_items)))
        for k, v in b.items():
            boots[k].append(v)
    res = {"estimates": {k: {"value": v, "ci": ci(boots[k])} for k, v in est.items()}, "years": years}
    E = lambda k: f"{est[k]:+.3f} [{ci(boots[k])[0]:+.3f}, {ci(boots[k])[1]:+.3f}]"
    E2 = lambda k: f"{est[k]:.2f} [{ci(boots[k])[0]:.2f}, {ci(boots[k])[1]:.2f}]"

    # ---------- report ----------
    print(f"\n== 05-a: local polishing of {base_h} (gap-cut shift g, paired shift d; Delta 2024-2021) ==")
    print(f"{'level':<9}{'generator':<24}{'n':>5}{'e (changed)':>12}{'g post':>9}{'d post':>9}{'d pre':>9}"
          f"{'d log p24':>10}{'marker/1k':>10}")
    res["levels"] = {}
    for lv in LEVELS:
        for g in LOCAL + ["both"]:
            rows = [r for r in v22 if r["level"] == lv and (g == "both" or r["gen"] == g)]
            if not rows:
                continue
            f = lambda k: np.mean([r[k] for r in rows])
            print(f"{lv:<9}{g:<24}{len(rows):>5}{f('e'):>12.3f}{f('g_post'):>+9.3f}{f('d_post'):>+9.3f}"
                  f"{f('d_pre'):>+9.3f}{f('d_ll'):>+10.3f}{f('d_mark'):>+10.2f}")
            res["levels"][f"{lv}/{g}"] = {k: float(f(k)) for k in ("e", "g_post", "d_post", "d_pre", "d_ll", "d_mark")} | {"n": len(rows)}
    if v19:
        f = lambda k: np.mean([r[k] for r in v19])
        print(f"{'polish':<9}{'both (H19)':<24}{len(v19):>5}{f('e'):>12.3f}{f('g_post'):>+9.3f}{f('d_post'):>+9.3f}"
              f"{f('d_pre'):>+9.3f}{f('d_ll'):>+10.3f}{f('d_mark'):>+10.2f}")
        res["levels"]["polish/H19"] = {k: float(f(k)) for k in ("e", "g_post", "d_post", "d_pre", "d_ll", "d_mark")} | {"n": len(v19)}
    sc = {k: float(np.mean([r[k] for r in scratch])) for k in ("e", "d_post", "d_pre")}
    print(f"{'scratch':<9}{'(same-title AI anchor)':<24}{len(scratch):>5}{sc['e']:>12.3f}{'':>9}{sc['d_post']:>+9.3f}{sc['d_pre']:>+9.3f}")
    res["scratch"] = sc | {"n": len(scratch)}

    print("\nslope through 0 of shift on e (pooled levels and generators):")
    for key, lab in (("d_post", "paired d, 2024-2021"), ("g_post", "gap-cut g, 2024-2021"),
                     ("d_pre", "paired d, 2021-2018 (placebo)"), ("d_mark", "AI-marker words /1k")):
        print(f"  {lab:<32} {E(f's_all_{key}')}")
    for g in LOCAL:
        print(f"  {'paired d, ' + g:<32} {E(f's_{g}_d_post')}")
    print(f"  post hoc, d = b1 e + b2 e^2:     b1 {E('quad_b1')}, b2 {E('quad_b2')}")
    e_all =np.array([r["e"] for r in v22])
    res["bins_local_H22"], parts = [], []
    for b, (lo, hi) in enumerate(zip(BINS, BINS[1:])):
        n = int(((e_all >= lo) & (e_all < hi)).sum())
        if n < 5:
            continue
        res["bins_local_H22"].append({"lo": float(lo), "hi": float(hi), "n": n, "e": est[f"bin{b}_e"],
                                      "d": est[f"bin{b}_d_post"], "ci_d": ci(boots[f"bin{b}_d_post"]),
                                      "d_pre": est[f"bin{b}_d_pre"], "ci_d_pre": ci(boots[f"bin{b}_d_pre"])})
        parts.append(f"[{lo:.2f},{hi:.2f}) {n} {est[f'bin{b}_e']:.2f} {est[f'bin{b}_d_post']:+.3f}")
    print("  binned means (e bin: n, mean e, mean d):", "; ".join(parts))

    print(f"\n== 05-b: APT-Eval, {len(apt_origs)} originals (paired d, Delta 2024-2021) ==")
    print(f"{'polisher':<24}{'n':>5}{'slope d/e':>24}{'R vs local':>22}{'slope pre':>11}{'marker slope':>14}")
    res["apt"] = {}
    for pol in APT_LARGE + APT_OTHER + LOCAL + ["local", "others"]:
        if pol not in apt:
            continue
        rstr = E2(f"R_{pol}") if f"R_{pol}" in est else ""
        print(f"{pol:<24}{len(apt[pol]):>5}{E(f'apt_s_{pol}_d_post'):>24}{rstr:>22}"
              f"{est[f'apt_s_{pol}_d_pre']:>+11.3f}{est[f'apt_s_{pol}_d_mark']:>+14.2f}")
        res["apt"][pol] = {"n": len(apt[pol])}
    print(f"{'polisher':<24}" + "".join(f"{lv:>22}" for lv in APT_DEGREES + LEVELS) + "   (mean e / mean d)")
    for pol in APT_LARGE + APT_OTHER + LOCAL:
        cells = []
        for lv in sorted({r["level"] for r in apt.get(pol, [])}):  # percent levels are stored, not printed
            rows = [r for r in apt[pol] if r["level"] == lv]
            res["apt"][pol][lv] = {"n": len(rows)} | {k: float(np.mean([r[k] for r in rows]))
                                                      for k in ("e", "d_post", "d_pre", "d_mark")}
        for lv in APT_DEGREES + LEVELS:
            s = res["apt"].get(pol, {}).get(lv)
            cells.append(f"{s['e']:.2f} / {s['d_post']:+.3f}" if s else "")
        print(f"{pol:<24}" + "".join(f"{c:>22}" for c in cells))

    print(f"\n== population: shift_T = gap({base_h}) - gap(T) (02's scores) ==")
    print(f"mean first-version dates: " + ", ".join(f"{k} {v:.2f}" for k, v in years.items()))
    print(f"pre-ChatGPT gap trend {t_lbl}->{base_lbl}: {E('trend_per_year')} per year")
    for lbl in targets:
        print(f"\nT = {lbl}: shift {E(f'shift_{lbl}')}, trend-adjusted {E(f'shift_{lbl}_trend')}, "
              f"no LLM-topic titles {E(f'shift_{lbl}_nollm')}")
        print(f"  beta (AI-edited share of text), local polishing:  {E2(f'beta_{lbl}_local')}"
              f"   trend-adj {E2(f'beta_{lbl}_local_trend')}   no-LLM-topic {E2(f'beta_{lbl}_local_nollm')}")
        for pol in APT_LARGE + APT_OTHER:
            print(f"  beta if polished like {pol:<14} {E2(f'beta_{lbl}_{pol}')}")
        print("  share of abstracts if each AI-touched one got local level L: "
              + ", ".join(f"{lv} {E2(f'alpha_{lbl}_{lv}')}" for lv in LEVELS))

    # ---------- markers: populations (rates pooled over words; bootstrap draws as above) ----------
    res["markers"] = {"base": est["mark_rate_base"]}
    print(f"\n== AI-marker words (/1k words) ==\n{base_h}: {est['mark_rate_base']:.2f}")
    for lbl, t in targets.items():
        r_t = est[f"mark_rate_{lbl}"]
        res["markers"][lbl] = r_t
        print(f"{t['hg']}: {r_t:.2f} (rise {r_t - est['mark_rate_base']:+.2f}); beta_W local {E2(f'betaW_{lbl}_local')}, "
              + ", ".join(f"{pol} {E2(f'betaW_{lbl}_{pol}')}" for pol in APT_LARGE))

    # ---------- detection of polished text (each text on its own, up to 200 tokens) ----------
    print("\n== detection: polished vs original, AUROC (TPR@5%FPR) ==")
    res["detection"] = {}
    h_ids = [i for i in s05.ids(base_h)]
    for lv in LEVELS:
        p_ids = s05.ids(f"P{base_lbl}.{lv}")
        row = {}
        y = np.r_[np.zeros(len(h_ids)), np.ones(len(p_ids))]
        for key, n in (("post", None), ("pre", None), ("ll", None), ("post_100", 100), ("ll_100", 100)):
            k = key.split("_")[0]
            x = np.r_[[s05.mean(i, k, n) for i in h_ids], [s05.mean(i, k, n) for i in p_ids]]
            row[key] = (auroc(y, x), tpr_at_fpr(y, x, 0.05))
        res["detection"][lv] = row
        print(f"{lv:<9} Delta24-21 {row['post'][0]:.3f} ({row['post'][1]:.3f})   Delta21-18 {row['pre'][0]:.3f}   "
              f"log p24 {row['ll'][0]:.3f} ({row['ll'][1]:.3f})   | post hoc, first 100 tokens: "
              f"Delta {row['post_100'][0]:.3f} ({row['post_100'][1]:.3f}), log p24 {row['ll_100'][0]:.3f} ({row['ll_100'][1]:.3f})")
    apt_h = s05.ids("APT.H")
    for pol in APT_LARGE + APT_OTHER:
        for lv in ("minor", "major"):
            p_ids = [r["id"] for r in apt.get(pol, []) if r["level"] == lv]
            if not p_ids:
                continue
            y = np.r_[np.zeros(len(apt_h)), np.ones(len(p_ids))]
            xa = np.r_[[s05.mean(i, "post") for i in apt_h], [s05.mean(i, "post") for i in p_ids]]
            xl = np.r_[[s05.mean(i, "ll") for i in apt_h], [s05.mean(i, "ll") for i in p_ids]]
            res["detection"][f"APT {pol} {lv}"] = {"post": (auroc(y, xa), tpr_at_fpr(y, xa, 0.05)), "ll": (auroc(y, xl), tpr_at_fpr(y, xl, 0.05))}
            print(f"APT {pol:<13}{lv:<6} Delta24-21 {auroc(y, xa):.3f}   log p24 {auroc(y, xl):.3f}")

    # ---------- shape: mixture of polished/untouched abstracts vs a uniform shift ----------
    print("\n== shape of the 2024+ human distribution (Delta 2024-2021, anchor drift removed) ==")
    qs = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    xb = np.array([h for _, (_, h, _) in base_items.items()])
    cb = np.array([c for _, (_, _, c) in base_items.items()])
    res["shape"] = {}
    for lbl, t in targets.items():
        xt = np.array([h for _, _, h, _ in t["items"]])
        ct = np.array([c for _, _, _, c in t["items"]])
        adj = xt - (ct.mean() - cb.mean())
        obs = np.quantile(adj, qs) - np.quantile(xb, qs)
        bq = []
        for _ in range(args.n_boot):
            a = rng.integers(0, len(xt), len(xt))
            b = rng.integers(0, len(xb), len(xb))
            bq.append(np.quantile(xt[a] - (ct[a].mean() - cb[b].mean()), qs) - np.quantile(xb[b], qs))
        lo, hi = np.quantile(bq, 0.025, axis=0), np.quantile(bq, 0.975, axis=0)
        print(f"T = {lbl}: quantile shift " + ", ".join(f"q{int(q * 100)} {o:+.3f} [{l:+.3f}, {h:+.3f}]"
                                                         for q, o, l, h in zip(qs, obs, lo, hi)))
        shape = {"q": qs.tolist(), "obs": obs.tolist(), "lo": lo.tolist(), "hi": hi.tolist(), "models": {}}
        Fobs = np.searchsorted(np.sort(adj), adj, side="right") / len(adj)

        def cvm(F_model_at_adj):
            return float(np.mean((Fobs - F_model_at_adj) ** 2))

        shift = adj.mean() - xb.mean()
        F_shift = np.searchsorted(np.sort(xb + shift), adj, side="right") / len(xb)
        shape["models"]["uniform shift"] = {"cvm": cvm(F_shift), "pred": (np.quantile(xb + shift, qs) - np.quantile(xb, qs)).tolist()}
        print(f"  uniform shift {shift:+.3f}: CvM {cvm(F_shift):.4f}")
        for lv in LEVELS:
            xp = []
            for i in s05.ids(f"P{base_lbl}.{lv}"):
                paper = i.split("~")[0]
                c = f"{paper}:a"
                if c in s05.n:
                    xp.append(s05.mean(i, "post", min(s05.n[i], s05.n[c])))
            xp = np.array(xp)
            a = shift / (xp.mean() - xb.mean())
            if not 0 < a <= 1:
                print(f"  mixture with {lv}: share {a:.2f} outside (0, 1], skipped")
                shape["models"][f"mixture {lv}"] = {"share": float(a)}
                continue
            F_mix = ((1 - a) * np.searchsorted(np.sort(xb), adj, side="right") / len(xb)
                     + a * np.searchsorted(np.sort(xp), adj, side="right") / len(xp))
            # mixture quantiles by inverting the mixture CDF on a fine grid
            g = np.linspace(min(xb.min(), xp.min()) - 0.1, max(xb.max(), xp.max()) + 0.1, 4001)
            Fg = (1 - a) * np.searchsorted(np.sort(xb), g, side="right") / len(xb) + a * np.searchsorted(np.sort(xp), g, side="right") / len(xp)
            qm = np.array([g[np.searchsorted(Fg, q)] for q in qs])
            shape["models"][f"mixture {lv}"] = {"share": float(a), "cvm": cvm(F_mix), "pred": (qm - np.quantile(xb, qs)).tolist()}
            print(f"  mixture with {lv:<8} (share {a:.2f}): CvM {cvm(F_mix):.4f}; predicted quantile shift "
                  + ", ".join(f"{v:+.3f}" for v in qm - np.quantile(xb, qs)))
        res["shape"][lbl] = shape

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
