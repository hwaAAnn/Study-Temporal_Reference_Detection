"""Experiment 14, post hoc (after part B): is the right dose generator-specific?

Part B's f (share of the 2022 reference's gain kept by the 2023 reference) was far above part A's f_hat for
2025-2026 generators. Idea: a vintage's absorption is of the chat styles on the web by its crawl date, so the dose
should be measured on the generator's own text. Per (set, generator), with ll_v from the diagonal cell v_v:
  G_gen(v) = mean over title pairs of ll_v(AI) - ll_v(human)
  f_hat_gen = [G_gen(2024) - G_gen(2023)] / [G_gen(2024) - G_gen(2022)]
and the observed f = [d(2023_2024) - d(2024_2024)] / [d(2022_2024) - d(2024_2024)] (Cohen's d, as in dose14).
In sample (dose and outcome on the same texts), so a description, not a test. Pools whose y(2022) CI includes 0
in part B (gpt-oss on NB) are skipped.

Part C mode (registered before part C's outcome existed, from the A+B relation): the same quantities on the
six RAID chat generators and five APT polishers, with G_gen as mean ll_v(AI) - mean ll_v(human) (APT is not
one-to-one) and f, y(2022) CI from dose14_ext's part C output. Pools whose y(2022) CI includes 0 are skipped.
Verdict: F holds if corr(f_hat_gen, f) >= 0.5 and median f / f_hat_gen is in [0.67, 1.5]; Bi holds if corr >= 0.5.

Example:
  python scripts/dose14_gen_posthoc.py --outputs <main>/outputs > <main>/outputs/s14_gen_posthoc.txt
  python scripts/dose14_gen_posthoc.py --part C --outputs <main>/outputs --input <main>/data/04_new.jsonl       > <main>/outputs/s14_gen_C.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import dose14 as dz
import dose14_ext as dx


def by_gen(rows: dict, hg: str, ag_of: dict[str, str]) -> dict[str, list[tuple[dict, dict]]]:
    """generator -> list of (human row, AI row) pairs sharing pair_id."""
    h = {r["pair_id"]: r for r in rows.values() if r["group"] == hg}
    out: dict[str, list] = {}
    for gen, ag in ag_of.items():
        a = {r["pair_id"]: r for r in rows.values() if r["group"] == ag and (r["generator"] == gen or ag.endswith(gen))}
        out[gen] = [(h[p], a[p]) for p in sorted(set(h) & set(a))]
    return out


def stats(pairs: list[tuple[dict, dict]]) -> dict:
    G = {v: float(np.mean([a["cells"][f"{v}_{v}"]["ll"] - h["cells"][f"{v}_{v}"]["ll"] for h, a in pairs]))
         for v in ("2022", "2023", "2024")}
    out = {"G": G, "f_hat_gen": (G["2024"] - G["2023"]) / (G["2024"] - G["2022"])}
    for det in dz.DETS:
        d = {}
        for c in ("2022_2024", "2023_2024", "2024_2024"):
            hs = np.array([dz.score(h, det, c) for h, _ in pairs])
            as_ = np.array([dz.score(a, det, c) for _, a in pairs])
            d[c] = float(dz.cohens_d_rows(as_[None, :], hs[None, :])[0])
        y22, y23 = d["2022_2024"] - d["2024_2024"], d["2023_2024"] - d["2024_2024"]
        out[det] = {"f": y23 / y22, "y2022": y22}
    return out


def part_c(out: Path, input_path: Path) -> None:
    pools, fams = dx.build_c(out, input_path)
    cev = json.loads((out / "s14_Cev.json").read_text(encoding="utf-8"))["pools"]
    rows = fams["DG"]
    print(f"{'pool':28} {'G22':>6} {'G23':>6} {'G24':>6} {'f_hat_gen':>9} {'f F':>6} {'f Bi':>6}  y22 F CI")
    pts = []
    for name, pool in pools.items():
        if not (name.startswith("RAID ") or name.startswith("APT strong ")):
            continue
        G = {}
        for v in ("2022", "2023", "2024"):
            ll = np.array([rows[i]["cells"][f"{v}_{v}"]["ll"] for i in pool.ids])
            G[v] = float(ll[pool.y].mean() - ll[~pool.y].mean())
        fh = (G["2024"] - G["2023"]) / (G["2024"] - G["2022"])
        e = cev[name]
        ok = {det: e[det]["D1"]["y2022_ci"][0] > 0 for det in dz.DETS}
        print(f"{name:28} {G['2022']:6.3f} {G['2023']:6.3f} {G['2024']:6.3f} {fh:9.3f} {e['F']['D1']['f']:6.3f} "
              f"{e['Bi']['D1']['f']:6.3f}  {dz.fmt(e['F']['D1']['y2022_ci'])}" + ("" if all(ok.values()) else "  (skipped)"))
        if all(ok.values()):
            pts.append((name, fh, {det: e[det]["D1"]["f"] for det in dz.DETS}))
    raid = [fh for n, fh, _ in pts if n.startswith("RAID ")]
    print(f"RAID mean f_hat_gen {np.mean(raid):.3f} (predicted 0.20-0.45)")
    fh = np.array([p[1] for p in pts])
    for det in dz.DETS:
        f = np.array([p[2][det] for p in pts])
        r = float(np.corrcoef(fh, f)[0, 1])
        med = float(np.median(f / fh))
        ok = r >= 0.5 and (0.67 <= med <= 1.5 if det == "F" else True)
        slope, icpt = np.polyfit(fh, f, 1)
        print(f"{det}: {len(pts)} pools, corr {r:.2f}, f = {icpt:+.2f} + {slope:.2f} f_hat_gen, "
              f"median f / f_hat_gen {med:.2f} -> {'holds' if ok else 'fails'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--part", choices=["AB", "C"], default="AB")
    ap.add_argument("--input")
    args = ap.parse_args()
    out = Path(args.outputs)
    if args.part == "C":
        part_c(out, Path(args.input))
        return
    a_rows = dz.load(out / "s08_grid.jsonl")
    b_rows = dx.read_jsonl(out / "s12_dg.jsonl")
    gens_a = {"qwen2.5-1.5b-instruct": None, "llama3.2-1b-instruct": None}
    pts = []
    for hg, ag in (("H19", "C19"), ("B", "C")):
        for gen, pairs in by_gen(a_rows, hg, {g: ag for g in gens_a}).items():
            pts.append((f"A {hg}/{ag} {gen}", stats(pairs)))
    for hg, pre in (("H19", "N19"), ("B", "NB")):
        for gen, pairs in by_gen(b_rows, hg, {g: f"{pre}.{g}" for g in dx.GENS12}).items():
            pts.append((f"B {hg}/{pre} {gen}", stats(pairs)))
    print(f"{'pool':40} {'G22':>6} {'G23':>6} {'G24':>6} {'f_hat_gen':>9} {'f F':>6} {'f Bi':>6}  y22 F")
    keep = []
    for name, s in pts:
        skip = name.startswith("B B/NB gptoss120b")
        print(f"{name:40} {s['G']['2022']:6.3f} {s['G']['2023']:6.3f} {s['G']['2024']:6.3f} {s['f_hat_gen']:9.3f} "
              f"{s['F']['f']:6.3f} {s['Bi']['f']:6.3f}  {s['F']['y2022']:+.2f}" + ("  (skipped: y22 CI has 0)" if skip else ""))
        if not skip:
            keep.append(s)
    fh = np.array([s["f_hat_gen"] for s in keep])
    for det in dz.DETS:
        f = np.array([s[det]["f"] for s in keep])
        r = float(np.corrcoef(fh, f)[0, 1])
        slope, icpt = np.polyfit(fh, f, 1)
        print(f"{det}: across {len(keep)} pools, corr(f_hat_gen, f) = {r:.2f}; f = {icpt:+.2f} + {slope:.2f} f_hat_gen;"
              f" mean f / f_hat_gen = {float(np.mean(f / fh)):.2f}")


if __name__ == "__main__":
    main()
