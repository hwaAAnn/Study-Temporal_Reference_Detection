"""Experiment 15: 01's twelve-vintage curve on RAID's eight domains, with base vs chat versions of the same models.

Input: one score file per vintage from score_texts.py (--no-pair-truncate) on build_15_input.py's rows. Every
comparison cuts its two texts to the shorter length (as in 01). l_v(x) = mean log p_v over those tokens.

Curves per domain d (bootstrap clusters: RAID source documents):
  G_chat[d](v)  mean over sources and the six chat generators of l_v(chat) - l_v(human)      01's G
  K[d](v)       mean over sources and mistral/mpt of l_v(chat version) - l_v(base version)
                (no human text: the two texts differ only by the chat tuning of the generator)
  G_old[d](v)   mean over sources and the two base models + gpt2 of l_v(text) - l_v(human)
  G[d|gen](v)   one generator against human (descriptive)
and the same pooled over domains (clusters: domain:source). Each curve gets 01's statistics: a line fitted to
2013-2022, departures r(2023), r(2024) with z on 8 df, jumps J, change point tau (year_curve.curve_stats).

Pre-registered criteria (docs/experiments/15-raid-domains.md):
  D1[d]  G_chat departs: r2024 CI > 0 and z2024 > 2.306
  D2[d]  timing: tau in (2022, 2023) and the largest post-ChatGPT jump exceeds every pre-ChatGPT jump
  D3[d]  K departs: r2024 CI > 0 and z2024 > 2.306
  D4[d]  G_old stays low: r2024 CI upper end < G_chat r2024 / 3
  verdict  D1 in >= 7 domains: "general (chat style)" if D3 and D4 each hold in >= 6, else "general (AI text)";
           D1 in 4-6: "domain-dependent"; D1 in <= 3: "narrow"
  D5     pooled over domains, chatgpt and gpt4 each depart (r2024 CI > 0, z2024 > 2.306)

Example:
  python scripts/domains15.py --scores outputs/yc15_20*.jsonl --json outputs/s15_eval.json > outputs/s15_eval.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import year_curve as yc
from build_15_input import BASE, CHAT, GENS

FAMS = ["mistral", "mpt"]  # true base/chat pairs in RAID; the chat version is "<fam>-chat"
OLD_POOL = BASE + ["gpt2"]
ALL_GENS = GENS
KEYS = ["r2023", "r2024", "maxabsJ_pre", "maxJ_post"]


def sources(sc: yc.Scores) -> dict[str, dict[str, dict[str, str]]]:
    """domain -> source -> model ("H" for human) -> text id."""
    out: dict = {}
    for i, m in sc.meta.items():
        dom, model = m["group"].split(".", 1)
        out.setdefault(dom, {}).setdefault(m["pair_id"], {})[model] = i
    return out


def diff_set(sc: yc.Scores, src: dict[str, dict[str, dict[str, str]]], pairs: list[tuple[str, str]],
             label: str) -> yc.Set:
    """Rows l_v(a) - l_v(b) for every (a, b) model pair in every source of the given domains (cut to the shorter)."""
    rows, cl = [], []
    for dom, ss in src.items():
        for s, m in ss.items():
            for a, b in pairs:
                if a in m and b in m:
                    ia, ib = m[a], m[b]
                    n = min(sc.n(ia), sc.n(ib))
                    rows.append(sc.l(ia, n) - sc.l(ib, n))
                    cl.append(f"{dom}:{s}")
    if not rows:
        raise SystemExit(f"no rows for {label}")
    return yc.Set(rows, np.ones(len(rows)), cl, label)


def build_sets(sc: yc.Scores) -> tuple[dict[str, yc.Set], list[str]]:
    src = sources(sc)
    doms = sorted(src)
    sets: dict[str, yc.Set] = {}
    for scope in doms + ["all"]:
        sub = src if scope == "all" else {scope: src[scope]}
        sets[f"G_chat[{scope}]"] = diff_set(sc, sub, [(g, "H") for g in CHAT], f"G_chat[{scope}]")
        sets[f"K[{scope}]"] = diff_set(sc, sub, [(f"{f}-chat", f) for f in FAMS], f"K[{scope}]")
        sets[f"G_old[{scope}]"] = diff_set(sc, sub, [(g, "H") for g in OLD_POOL], f"G_old[{scope}]")
        for g in ALL_GENS:
            sets[f"G[{scope}|{g}]"] = diff_set(sc, sub, [(g, "H")], f"G[{scope}|{g}]")
    return sets, doms


def departs(e: dict) -> bool:
    return bool(e["r2024_ci"][0] > 0 and e["z2024"] > yc.T975_8)


def judge(res: dict, doms: list[str]) -> dict:
    c = res["curves"]
    per = {}
    for d in doms:
        g, k, o = c[f"G_chat[{d}]"], c[f"K[{d}]"], c[f"G_old[{d}]"]
        per[d] = {
            "D1": departs(g),
            "D2": bool(g["tau"] in (2022, 2023) and g["maxJ_post"] > g["maxabsJ_pre"]),
            "D3": departs(k),
            "D4": bool(o["r2024_ci"][1] < g["r2024"] / 3),
        }
    n = {q: sum(per[d][q] for d in doms) for q in ("D1", "D2", "D3", "D4")}
    if n["D1"] >= 7:
        verdict = "general (chat style)" if n["D3"] >= 6 and n["D4"] >= 6 else "general (AI text)"
    elif n["D1"] >= 4:
        verdict = "domain-dependent"
    else:
        verdict = "narrow"
    d5 = {g: departs(c[f"G[all|{g}]"]) for g in ("chatgpt", "gpt4")}
    return {"per_domain": per, "counts": n, "verdict": verdict, "D5": d5}


def analyse(sc: yc.Scores, n_boot: int = 2000, seed: int = 0) -> dict:
    sets, doms = build_sets(sc)
    curves = {k: st.curve() for k, st in sets.items()}
    rng = np.random.default_rng(seed)
    res: dict = {"years": yc.YEARS, "domains": doms, "n": {k: int(len(st.cl)) for k, st in sets.items()},
                 "clusters": {k: int(st.K) for k, st in sets.items()}, "curves": {}}
    for k, st in sets.items():
        s = yc.curve_stats(curves[k])
        boot = {q: [] for q in KEYS + ["tau"]}
        for _ in range(n_boot):
            wk = np.bincount(rng.integers(0, st.K, st.K), minlength=st.K).astype(float)
            b = yc.curve_stats(st.curve(wk))
            for q in KEYS + ["tau"]:
                boot[q].append(b[q])
        e = {"curve": curves[k].tolist(), **s}
        for q in KEYS:
            e[f"{q}_ci"] = yc.ci(boot[q])
        taus = np.array(boot["tau"])
        e["tau_boot"] = float((taus == s["tau"]).mean())
        e["share2023"] = float(s["r2023"] / s["r2024"]) if s["r2024"] > 0 else float("nan")
        res["curves"][k] = e
    for d in doms + ["all"]:
        g, kk = res["curves"][f"G_chat[{d}]"]["r2024"], res["curves"][f"K[{d}]"]["r2024"]
        res["curves"][f"K[{d}]"]["ratio_to_G_chat"] = float(kk / g) if g > 0 else float("nan")
    res["judge"] = judge(res, doms)
    return res


def fmt_ci(c: list[float]) -> str:
    return f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def report(res: dict) -> None:
    doms = res["domains"]
    c = res["curves"]
    print("=== curves: value per vintage (nats/token) ===")
    print(f"{'curve':<26}" + "".join(f"{y:>8}" for y in res["years"]))
    for k, e in c.items():
        print(f"{k:<26}" + "".join(f"{v:>+8.3f}" for v in e["curve"]))
    print("\n=== departure from the 2013-2022 line (95% bootstrap CI over sources; z on 8 df) ===")
    print(f"{'curve':<26}{'n':>6}{'slope/yr':>10}{'s_pre':>7}{'r2023':>8}{'CI':>19}{'z':>7}{'r2024':>8}{'CI':>19}"
          f"{'z':>7}{'tau+1':>6}{'P':>6}{'r23/r24':>8}")
    for k, e in c.items():
        print(f"{k:<26}{res['n'][k]:>6}{e['slope_pre']:>+10.4f}{e['s_pre']:>7.3f}{e['r2023']:>+8.3f} "
              f"{fmt_ci(e['r2023_ci'])}{e['z2023']:>7.1f}{e['r2024']:>+8.3f} {fmt_ci(e['r2024_ci'])}{e['z2024']:>7.1f}"
              f"{e['tau'] + 1:>6}{e['tau_boot']:>6.2f}{e['share2023']:>8.2f}")
    print("\n=== K / G_chat at 2024 (predicted 0.5-1.0) ===")
    print("  ".join(f"{d} {c[f'K[{d}]']['ratio_to_G_chat']:.2f}" for d in doms + ["all"]))
    j = res["judge"]
    print("\n=== criteria per domain ===")
    print(f"{'domain':<10}{'D1':>6}{'D2':>6}{'D3':>6}{'D4':>6}")
    for d in doms:
        p = j["per_domain"][d]
        print(f"{d:<10}" + "".join(f"{('yes' if p[q] else 'no'):>6}" for q in ("D1", "D2", "D3", "D4")))
    print(f"counts: {j['counts']}")
    print(f"verdict: {j['verdict']}")
    print(f"D5 (pooled, departs): {j['D5']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    sc = yc.Scores(args.scores)
    print(f"texts: {len(sc.cs)}; vintages {yc.YEARS[0]}-{yc.YEARS[-1]}\n")
    res = analyse(sc, args.n_boot, args.seed)
    report(res)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
