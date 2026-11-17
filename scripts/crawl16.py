"""Experiment 16: does the AI-style share of DatedGPT's training data rise with the 2023 crawls, like G(v) does?

Input: sample_dumps16.py counts for FineWeb-Edu (DatedGPT's data) and FineWeb (no educational filter).
Per data set, measure (mark, rare, phrase) and crawl year y: m(y) = 1000 * sum(count) / sum(words) over the
sampled documents of that year's dumps. The 2013-2024 curve gets 01's statistics (year_curve.curve_stats): line
fitted to 2013-2022, departures r(2023), r(2024) with z on 8 df, change point. 2025 is reported, not judged.
CIs: 2000 bootstrap draws over row groups (1,000 documents each), resampled within each year.

Pre-registered criteria (docs/experiments/16-crawl-ai-share.md), on FineWeb-Edu, measure mark:
  C1  r(2024) CI > 0 and z(2024) > 2.306
  C2  r(2023) CI > 0 and z(2023) > 2.306
  C3  r(2023) / r(2024) in [0.44, 1.04] (01's G: 0.74 +- 0.3)
  C4  relative rise rel = r(2024) / mean(m, 2019-2022) is larger in FineWeb-Edu than in FineWeb (CI of difference > 0)
  verdict  C1 and C2: "first link supported"; C1 only: "visible only in 2024"; otherwise "not visible"
Descriptive: implied AI-equivalent share r(y) / (m_AI - m_human), calibrated on RAID chat vs human texts (15 input)
and on 00's B/C abstracts; correlation of m(y) with 01's G(v).

Example:
  python scripts/crawl16.py --edu outputs/s16_edu.jsonl --fw outputs/s16_fw.jsonl --g outputs/yc_eval.json \
      --calib-raid data/15_input.jsonl --calib-bc data/pairs.jsonl --json outputs/s16_eval.json > outputs/s16_eval.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import year_curve as yc  # noqa: E402
from sample_dumps16 import counts  # noqa: E402

MEASURES = ["mark", "rare", "phrase"]
CHAT = ["chatgpt", "gpt4", "llama-chat", "mistral-chat", "mpt-chat", "cohere-chat"]  # RAID chat (15)
LO, HI = 0.74 - 0.3, 0.74 + 0.3


class Counts:
    """Row-group sums of words and measure counts, by year."""

    def __init__(self, rows: list[dict]):
        agg: dict = defaultdict(lambda: defaultdict(float))
        for r in rows:
            a = agg[(r["year"], r["dump"], r["file"], r["rg"])]
            a["words"] += r["words"]
            for m in MEASURES:
                a[m] += r[m]
        self.years = sorted({k[0] for k in agg})
        self.by_year = {y: {q: np.array([v[q] for k, v in agg.items() if k[0] == y]) for q in ["words"] + MEASURES}
                        for y in self.years}
        self.n_docs = {y: sum(1 for r in rows if r["year"] == y) for y in self.years}

    def rate(self, m: str, y: int, idx: np.ndarray | None = None) -> float:
        d = self.by_year[y]
        w, c = (d["words"], d[m]) if idx is None else (d["words"][idx], d[m][idx])
        return float(1000 * c.sum() / w.sum())

    def curve(self, m: str, rng: np.random.Generator | None = None) -> dict[int, float]:
        out = {}
        for y in self.years:
            n = len(self.by_year[y]["words"])
            idx = None if rng is None else rng.integers(0, n, n)
            out[y] = self.rate(m, y, idx)
        return out


def stats12(cur: dict[int, float]) -> dict:
    c = np.array([cur[y] for y in yc.YEARS])
    s = yc.curve_stats(c)
    s["share2023"] = float(s["r2023"] / s["r2024"]) if s["r2024"] > 0 else float("nan")
    s["rel2024"] = float(s["r2024"] / np.mean([cur[y] for y in range(2019, 2023)]))
    return s


def analyse(edu: Counts, fw: Counts, n_boot: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    res: dict = {"sets": {}}
    keys = ["r2023", "r2024", "share2023", "rel2024"]
    boot_rel = {}
    for name, cn in (("edu", edu), ("fw", fw)):
        for m in MEASURES:
            cur = cn.curve(m)
            s = stats12(cur)
            boot = {q: [] for q in keys}
            for _ in range(n_boot):
                b = stats12(cn.curve(m, rng))
                for q in keys:
                    boot[q].append(b[q])
            e = {"curve": {str(y): v for y, v in cur.items()}, **{k: v for k, v in s.items() if k != "sse_tau"}}
            for q in keys:
                e[f"{q}_ci"] = yc.ci(boot[q])
            res["sets"][f"{name}:{m}"] = e
            boot_rel[(name, m)] = np.array(boot["rel2024"])
    for m in MEASURES:
        d = boot_rel[("edu", m)] - boot_rel[("fw", m)]
        res["sets"][f"edu:{m}"]["rel_minus_fw_ci"] = yc.ci(d)
    e = res["sets"]["edu:mark"]
    c1 = bool(e["r2024_ci"][0] > 0 and e["z2024"] > yc.T975_8)
    c2 = bool(e["r2023_ci"][0] > 0 and e["z2023"] > yc.T975_8)
    res["criteria"] = {"C1": c1, "C2": c2, "C3": bool(LO <= e["share2023"] <= HI),
                       "C4": bool(e["rel_minus_fw_ci"][0] > 0)}
    res["verdict"] = "first link supported" if c1 and c2 else ("visible only in 2024" if c1 else "not visible")
    res["n_docs"] = {"edu": edu.n_docs, "fw": fw.n_docs}
    return res


def calib_rates(texts_ai: list[str], texts_h: list[str]) -> dict:
    out = {}
    for m in MEASURES:
        def rate(ts):
            cs = [counts(t) for t in ts]
            return 1000 * sum(c[m] for c in cs) / sum(c["words"] for c in cs)
        out[m] = {"ai": rate(texts_ai), "human": rate(texts_h)}
    return out


def calibrations(raid_path: str | None, bc_path: str | None) -> dict:
    cal = {}
    if raid_path:
        rows = [json.loads(x) for x in Path(raid_path).read_text(encoding="utf-8").splitlines() if x.strip()]
        cal["RAID chat vs human"] = calib_rates([r["text"] for r in rows if r["generator"] in CHAT],
                                                [r["text"] for r in rows if r["label"] == 0])
    if bc_path:
        rows = [json.loads(x) for x in Path(bc_path).read_text(encoding="utf-8").splitlines() if x.strip()]
        cal["B/C abstracts"] = calib_rates([r["ai"] for r in rows], [r["human"] for r in rows])
    return cal


def report(res: dict, g: dict | None, cal: dict) -> None:
    years_all = sorted({int(y) for e in res["sets"].values() for y in e["curve"]})
    print("=== rate per 1,000 words by crawl year ===")
    print(f"{'set':<14}" + "".join(f"{y:>8}" for y in years_all))
    for k, e in res["sets"].items():
        print(f"{k:<14}" + "".join(f"{e['curve'].get(str(y), float('nan')):>8.4f}" for y in years_all))
    print("\n=== departure from the 2013-2022 line (95% CI over row groups; z on 8 df) ===")
    print(f"{'set':<14}{'slope/yr':>10}{'r2023':>9}{'CI':>21}{'z':>7}{'r2024':>9}{'CI':>21}{'z':>7}"
          f"{'r23/r24':>8}{'rel2024':>9}{'tau+1':>6}")
    for k, e in res["sets"].items():
        print(f"{k:<14}{e['slope_pre']:>+10.5f}{e['r2023']:>+9.4f}  [{e['r2023_ci'][0]:+.4f}, {e['r2023_ci'][1]:+.4f}]"
              f"{e['z2023']:>7.1f}{e['r2024']:>+9.4f}  [{e['r2024_ci'][0]:+.4f}, {e['r2024_ci'][1]:+.4f}]{e['z2024']:>7.1f}"
              f"{e['share2023']:>8.2f}{e['rel2024']:>+9.3f}{e['tau'] + 1:>6}")
    for m in MEASURES:
        c = res["sets"][f"edu:{m}"]["rel_minus_fw_ci"]
        print(f"rel2024 FineWeb-Edu - FineWeb ({m}): CI [{c[0]:+.3f}, {c[1]:+.3f}]")
    if g is not None:
        gc = np.array(g["curves"]["G"]["curve"])
        print(f"\n=== link to 01's G (share r2023/r2024 = {g['curves']['G']['r2023'] / g['curves']['G']['r2024']:.2f}) ===")
        for k, e in res["sets"].items():
            mc = np.array([e["curve"][str(y)] for y in yc.YEARS])
            print(f"{k:<14} corr over 2013-2024 with G: {np.corrcoef(mc, gc)[0, 1]:+.2f}")
    print("\n=== implied AI-equivalent share r(y) / (m_AI - m_human), FineWeb-Edu ===")
    for name, c in cal.items():
        for m in MEASURES:
            gap = c[m]["ai"] - c[m]["human"]
            e = res["sets"][f"edu:{m}"]
            print(f"{name:<20} {m:<7} m_AI {c[m]['ai']:.3f} m_H {c[m]['human']:.3f}  "
                  f"2023 {e['r2023'] / gap:+.3f}  2024 {e['r2024'] / gap:+.3f}")
    print(f"\ncriteria: {res['criteria']}")
    print(f"verdict: {res['verdict']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edu", required=True)
    ap.add_argument("--fw", required=True)
    ap.add_argument("--g", default=None, help="01's yc_eval.json")
    ap.add_argument("--calib-raid", default=None)
    ap.add_argument("--calib-bc", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    def load(p):
        return Counts([json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()])

    edu, fw = load(args.edu), load(args.fw)
    print(f"documents: FineWeb-Edu {sum(edu.n_docs.values()):,}, FineWeb {sum(fw.n_docs.values()):,}\n")
    res = analyse(edu, fw, args.n_boot)
    g = json.loads(Path(args.g).read_text(encoding="utf-8")) if args.g else None
    cal = calibrations(args.calib_raid, args.calib_bc)
    res["calibration"] = cal
    report(res, g, cal)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
