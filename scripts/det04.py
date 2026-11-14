"""Experiment 04: detection on public benchmarks (RAID abstracts, APT-Eval) and the dev sets, for the
time-pair scores (DatedGPT) and the baseline zero-shot detectors; complementarity; time-boundary control.

Per text (each on its own length, at most 200 tokens; every score is "higher = more AI"):
  ll24, ll21           mean log p under DatedGPT-2024 / -2021 (likelihood)
  delta                mean (log p_2024 - log p_2022)
  delta_k05            the same over tokens kept by the surprisal filter (k = 0.05, mask from 2022; 03)
  s075, s09            mean log p_2024 - lambda * mean log p_2022, lambda = 0.75 / 0.9 (fixed on dev)
  ctrl_s075            mean log p_2021 - 0.75 * mean log p_2018 (the same formula on a pre-ChatGPT pair)
  bino                 minus Binoculars (Llama-3.2-1B observer / -Instruct performer)
  fdgpt                Fast-DetectGPT analytic discrepancy (GPT-Neo-2.7B)
  fusion               mean of the within-comparison percentile ranks of s075 and bino (label-free)
Comparisons: AI group(s) vs human group(s) (RAID attacks vs humans with the same attack). Metrics: AUROC and
TPR at 5% FPR (threshold from the comparison's human texts). Pools carry 95% CIs from 2000 bootstrap draws over
sources (RAID source abstract, APT original, dev paper), the same draws for every detector, so differences are
paired. Rescue rate: share of the AI texts a baseline misses at its 5% FPR threshold that s075 flags at its own.

Example:
  python scripts/det04.py --input data/04_all.jsonl --dated outputs/yc04_20*.jsonl outputs/yc_2018.jsonl \
      outputs/yc_2021.jsonl outputs/yc_2022.jsonl outputs/yc_2024.jsonl \
      --binoculars outputs/b04_binoculars.jsonl --fdgpt outputs/b04_fdgpt.jsonl \
      --json outputs/det04.json > outputs/det04.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trd.scoring import keep_mask  # noqa: E402

VINT = ["2018", "2021", "2022", "2024"]
DETS = ["ll24", "delta", "delta_k05", "s075", "s09", "bino", "fdgpt", "fusion", "ll21", "ctrl_s075"]
SHOW = ["ll24", "delta", "delta_k05", "s075", "s09", "bino", "fdgpt", "fusion"]
CHAT = ["chatgpt", "gpt4", "cohere-chat", "llama-chat", "mistral-chat", "mpt-chat"]
PRECHAT = ["gpt2", "gpt3", "cohere", "mistral", "mpt"]
ATTACK_MODELS = ["chatgpt", "gpt4", "llama-chat", "mistral-chat"]
LEVELS = ["extreme_minor", "minor", "slight_major", "major"]


# ---------------- metrics ----------------
def auroc(h: np.ndarray, a: np.ndarray) -> float:
    r = rankdata(np.r_[h, a])
    return float((r[len(h):].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(h)))


def threshold5(h: np.ndarray) -> float:
    """Score above which at most 5% of the human texts lie."""
    hs = np.sort(h)
    k = int(np.floor(0.05 * len(hs)))
    return float(hs[len(hs) - k - 1])


def tpr5(h: np.ndarray, a: np.ndarray) -> float:
    return float(np.mean(a > threshold5(h)))


# ---------------- loading ----------------
def read_meta(path: str) -> dict[str, dict]:
    meta = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "human" in r and "ai" in r:  # pair rows (00/02)
            pid = str(r["id"])
            meta[f"{pid}:h"] = {"group": r.get("human_group", "B"), "label": 0, "cluster": pid}
            meta[f"{pid}:a"] = {"group": r.get("ai_group", "C"), "label": 1, "cluster": pid,
                                "generator": r.get("generator", "")}
        else:
            m = {k: v for k, v in r.items() if k != "text"}
            m["cluster"] = str(r.get("orig_id") or r.get("pair_id") or r["id"])
            meta[str(r["id"])] = m
    return meta


def load_dated(paths: list[str], ids: set[str]) -> dict[str, dict[str, np.ndarray]]:
    lp: dict[str, dict[str, np.ndarray]] = {}
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["id"] not in ids:
                continue
            for y, v in r["lp"].items():
                if y in VINT:
                    lp.setdefault(r["id"], {})[y] = np.asarray(v, dtype=np.float64)
    return lp


def text_scores(meta, lp, bino_path, fd_path) -> dict[str, dict[str, float]]:
    bino = {json.loads(x)["id"]: json.loads(x) for x in Path(bino_path).read_text(encoding="utf-8").splitlines() if x}
    fd = {json.loads(x)["id"]: json.loads(x) for x in Path(fd_path).read_text(encoding="utf-8").splitlines() if x}
    out, missing = {}, []
    for i in meta:
        if i not in lp or sorted(lp[i]) != VINT or i not in bino or i not in fd:
            missing.append(i)
            continue
        m = {y: lp[i][y].mean() for y in VINT}
        keep = keep_mask(lp[i]["2022"], 0.05)
        out[i] = {"ll24": m["2024"], "ll21": m["2021"], "delta": m["2024"] - m["2022"],
                  "delta_k05": float(np.mean(lp[i]["2024"][keep] - lp[i]["2022"][keep])),
                  "s075": m["2024"] - 0.75 * m["2022"], "s09": m["2024"] - 0.9 * m["2022"],
                  "ctrl_s075": m["2021"] - 0.75 * m["2018"],
                  "bino": -bino[i]["binoculars"], "fdgpt": fd[i]["fdgpt"]}
    if missing:
        raise SystemExit(f"{len(missing)} texts lack a score (e.g. {missing[:3]})")
    return out


# ---------------- comparisons ----------------
class Comp:
    """One comparison: arrays of every detector's scores, labels and source clusters."""

    def __init__(self, name, h_ids, a_ids, sc, meta):
        if not h_ids or not a_ids:
            raise SystemExit(f"{name}: empty side ({len(h_ids)} human, {len(a_ids)} AI)")
        self.name, ids = name, h_ids + a_ids
        self.y = np.r_[np.zeros(len(h_ids)), np.ones(len(a_ids))].astype(bool)
        self.S = {d: np.array([sc[i][d] for i in ids]) for d in DETS if d != "fusion"}
        _, self.cl = np.unique([meta[i]["cluster"] for i in ids], return_inverse=True)
        self.K = int(self.cl.max()) + 1
        self.members = [np.flatnonzero(self.cl == k) for k in range(self.K)]

    def metrics(self, idx=None) -> dict[str, dict[str, float]]:
        idx = np.arange(len(self.y)) if idx is None else idx
        y = self.y[idx]
        S = {d: v[idx] for d, v in self.S.items()}
        S["fusion"] = (rankdata(S["s075"]) + rankdata(S["bino"])) / 2
        out = {}
        for d, v in S.items():
            h, a = v[~y], v[y]
            out[d] = {"auroc": auroc(h, a), "tpr5": tpr5(h, a)}
        flag = {d: S[d][y] > threshold5(S[d][~y]) for d in ("s075", "bino", "fdgpt")}
        for base in ("bino", "fdgpt"):
            miss = ~flag[base]
            out[f"rescue_{base}"] = {"rate": float(flag["s075"][miss].mean()) if miss.any() else float("nan"),
                                     "n_missed": int(miss.sum())}
        return out

    def draw(self, rng) -> np.ndarray:
        pick = rng.integers(0, self.K, self.K)
        return np.concatenate([self.members[k] for k in pick])


def derived(m: dict) -> dict[str, float]:
    a = {d: m[d]["auroc"] for d in DETS}
    return {"s075-ll24": a["s075"] - a["ll24"], "s09-ll24": a["s09"] - a["ll24"], "s09-bino": a["s09"] - a["bino"],
            "s075-bino": a["s075"] - a["bino"], "s09-fdgpt": a["s09"] - a["fdgpt"],
            "fusion-max": a["fusion"] - max(a["bino"], a["s075"]),
            "time_gain": (a["s075"] - a["ll24"]) - (a["ctrl_s075"] - a["ll21"]),
            "rescue_bino": m["rescue_bino"]["rate"], "rescue_fdgpt": m["rescue_fdgpt"]["rate"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="data/04_all.jsonl")
    ap.add_argument("--dated", nargs="+", required=True, help="DatedGPT score files (2018, 2021, 2022, 2024)")
    ap.add_argument("--binoculars", required=True)
    ap.add_argument("--fdgpt", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    meta = read_meta(args.input)
    sc = text_scores(meta, load_dated(args.dated, set(meta)), args.binoculars, args.fdgpt)
    by: dict[str, list[str]] = {}
    for i, m in meta.items():
        by.setdefault(m["group"], []).append(i)
    print(f"texts: {len(sc)}; groups: {len(by)}")
    G = lambda *gs: [i for g in gs for i in by.get(g, [])]  # noqa: E731

    # ---- comparisons: single groups (point estimates) and pools (with CIs) ----
    single, pools = {}, {}
    for m in CHAT + PRECHAT:
        for dec in ("greedy", "sampling"):
            single[f"RAID {m} {dec}"] = (G("RAID.H"), G(f"RAID.{m}.{dec}"))
    for att in ("paraphrase", "synonym"):
        for m in ATTACK_MODELS:
            single[f"RAID {m} sampling +{att}"] = (G(f"RAID.H.{att}"), G(f"RAID.{m}.sampling.{att}"))
    pols = sorted({meta[i]["polisher"] for i in G(*[g for g in by if g.startswith("APT.") and g != "APT.H"])})
    for lv in LEVELS:
        single[f"APT {lv}"] = (G("APT.H"), G(*[f"APT.{p}.{lv}" for p in pols]))
    strong = lambda ids: [i for i in ids if meta[i].get("level") in ("slight_major", "major")]  # noqa: E731
    apt_ai = G(*[g for g in by if g.startswith("APT.") and g != "APT.H"])
    for p in pols:
        single[f"APT strong, {p}"] = (G("APT.H"), strong([i for i in apt_ai if meta[i]["polisher"] == p]))
    for dom in sorted({meta[i]["domain"] for i in G("APT.H")}):
        hd = [i for i in G("APT.H") if meta[i]["domain"] == dom]
        single[f"APT strong, {dom}"] = (hd, strong([i for i in apt_ai if meta[i]["domain"] == dom]))
    single["dev C vs B"] = (G("B"), G("C"))
    for lv in ("proof", "polish", "rewrite"):
        single[f"dev P22.{lv} vs H22"] = (G("H22"), G(f"P22.{lv}"))

    pools["RAID chat, sampling"] = (G("RAID.H"), G(*[f"RAID.{m}.sampling" for m in CHAT]))
    pools["RAID chat, greedy"] = (G("RAID.H"), G(*[f"RAID.{m}.greedy" for m in CHAT]))
    pools["RAID chat, all"] = (G("RAID.H"), G(*[f"RAID.{m}.{d}" for m in CHAT for d in ("greedy", "sampling")]))
    pools["RAID pre-chat, all"] = (G("RAID.H"), G(*[f"RAID.{m}.{d}" for m in PRECHAT for d in ("greedy", "sampling")]))
    for dec in ("greedy", "sampling"):
        pools[f"RAID pre-chat, {dec}"] = (G("RAID.H"), G(*[f"RAID.{m}.{dec}" for m in PRECHAT]))
    pools["RAID 4 chat, sampling (no attack)"] = (G("RAID.H"), G(*[f"RAID.{m}.sampling" for m in ATTACK_MODELS]))
    for att in ("paraphrase", "synonym"):
        pools[f"RAID 4 chat, sampling +{att}"] = (G(f"RAID.H.{att}"),
                                                   G(*[f"RAID.{m}.sampling.{att}" for m in ATTACK_MODELS]))
    pools["APT weak (extreme_minor+minor)"] = (G("APT.H"), [i for i in apt_ai if meta[i]["level"] in LEVELS[:2]])
    pools["APT strong (slight_major+major)"] = (G("APT.H"), strong(apt_ai))

    res = {"single": {}, "pools": {}}
    hdr = f"{'comparison':<44}{'n_h':>5}{'n_ai':>6}" + "".join(f"{d:>13}" for d in SHOW)
    print("\n=== AUROC (TPR@5%FPR), point estimates ===")
    print(hdr)
    for name, (h, a) in single.items():
        m = Comp(name, h, a, sc, meta).metrics()
        res["single"][name] = m
        print(f"{name:<44}{len(h):>5}{len(a):>6}" + "".join(f"{m[d]['auroc']:>7.3f} ({m[d]['tpr5']:.2f})" for d in SHOW))

    print("\n=== pools, 95% bootstrap CI over sources ===")
    print(hdr)
    for name, (h, a) in pools.items():
        c = Comp(name, h, a, sc, meta)
        m = c.metrics()
        boots = [c.metrics(c.draw(rng)) for _ in range(args.n_boot)]
        ci = {d: np.percentile([b[d]["auroc"] for b in boots], [2.5, 97.5]).tolist() for d in DETS}
        ci_t = {d: np.percentile([b[d]["tpr5"] for b in boots], [2.5, 97.5]).tolist() for d in DETS}
        dv = derived(m)
        dci = {k: np.nanpercentile([derived(b)[k] for b in boots], [2.5, 97.5]).tolist() for k in dv}
        res["pools"][name] = {"metrics": m, "auroc_ci": ci, "tpr5_ci": ci_t, "derived": dv, "derived_ci": dci,
                              "n_h": len(h), "n_ai": len(a)}
        print(f"{name:<44}{len(h):>5}{len(a):>6}" + "".join(f"{m[d]['auroc']:>7.3f} ({m[d]['tpr5']:.2f})" for d in SHOW))
        print(f"{'':<55}" + "".join(f"  [{ci[d][0]:.3f},{ci[d][1]:.3f}]" for d in SHOW))
        print(f"{'':<55}" + "; ".join(f"{k} {v:+.3f} [{dci[k][0]:+.3f}, {dci[k][1]:+.3f}]" for k, v in dv.items()))

    # ---- hypotheses ----
    P = res["pools"]
    d = lambda pool, k: (P[pool]["derived"][k], *P[pool]["derived_ci"][k])  # noqa: E731
    cs, ca, ap_s = "RAID chat, sampling", "RAID chat, all", "APT strong (slight_major+major)"
    v, lo, _ = d(cs, "s075-ll24")
    verdict = {"H1": bool(v >= 0.03 and lo > 0)}
    verdict["H2"] = bool(P["RAID pre-chat, all"]["metrics"]["delta"]["auroc"] < 0.65
                         and P[ca]["metrics"]["delta"]["auroc"] >= 0.80)
    verdict["H3a"] = bool(d(ap_s, "s09-ll24")[1] > 0)
    verdict["H3b"] = bool(d(ap_s, "s09-bino")[1] > -0.03)
    verdict["H4"] = bool(all(P[p]["derived"]["rescue_bino"] >= 0.25 for p in (ca, ap_s))
                         and any(d(p, "fusion-max")[1] > 0 for p in (ca, ap_s)))
    verdict["H5"] = bool(all(d(p, "time_gain")[1] > 0 for p in (cs, ap_s)))
    res["verdict"] = verdict
    print("\n=== hypotheses ===")
    for k, x in verdict.items():
        print(f"{k}: {'holds' if x else 'fails'}")

    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
