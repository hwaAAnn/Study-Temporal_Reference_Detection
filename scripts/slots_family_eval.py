"""Experiments 10 and 11 analysis: AUROC per cell, the pre-registered contrasts and verdicts
(docs/experiments/10-tic-slots.md, 11-falcon-slots.md).

Part A (08's abstracts): H19/C19 (main), B/C (replication), H24/C24 (exploratory); bootstrap over pairs (08).
Part B (09's main pools): P1 = RAID chat sampling vs RAID.H, P2 = APT strong vs APT.H; bootstrap over sources
(det04.read_meta clusters, as det09).
Scores, higher = more AI: F:<cell> = fdgpt, Bi:<cell> = -bino, ll:<model> = mean log p under that model alone.
Described next to them (not re-scored): DatedGPT cells from 08/09 (dg:F:2022_2024, ...) and, in part B, 04's
ll24, s075, s09, bino1b, fdneo. 2,000 draws (seed 0); every score uses the same draws, so differences are paired.

Examples:
  python scripts/slots_family_eval.py --exp 10 --grid outputs/s10_A.jsonl outputs/s10_B.jsonl \
      --b-input data/04_new.jsonl --dg-grid outputs/s08_grid.jsonl outputs/s09_main.jsonl \
      --dated outputs/yc04_2022.jsonl outputs/yc04_2024.jsonl --binoculars outputs/b04_binoculars.jsonl \
      --fdgpt outputs/b04_fdgpt.jsonl --json outputs/s10_eval.json > outputs/s10_eval.txt
  python scripts/slots_family_eval.py --exp 11 --grid outputs/s11_grid.jsonl ... (same comparison files)
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

from slots_grid import auroc, tpr_at  # noqa: E402

DETS = ["F", "Bi"]
A_SETS = [("H19", "C19", "main"), ("B", "C", "replication"), ("H24", "C24", "exploratory")]
P1, P2 = "P1 RAID chat sampling", "P2 APT strong"
CHAT = ["chatgpt", "gpt4", "cohere-chat", "llama-chat", "mistral-chat", "mpt-chat"]
CEIL, RATIO = 0.998, 1.3
BASE = ["ll24", "s075", "s09", "bino1b", "fdneo"]

# per experiment: key cell, cells to show, contrasts {name: (terms, rule, parts)}; a term is (sign, key) with
# "{d}" standing for the detector; rule lower>0 / upper<0 / upper>0; parts "AB", "A", "B" or "P2" (that pool only)
EXP = {
    "10": {
        "key": "202301_202407",
        "show": ["202301_202301", "202407_202407", "202301_202407", "202407_202301", "201305_202407"],
        "contrasts": {
            "H1": ([(1, "{d}:202301_202407"), (-1, "{d}:202301_202301")], "lower>0", "AB"),
            "H2": ([(1, "{d}:202301_202407"), (-1, "{d}:202407_202407")], "lower>0", "AB"),
            "H3": ([(1, "{d}:202301_202407"), (-1, "{d}:202407_202407"), (-1, "{d}:202101_202301"),
                    (1, "{d}:202301_202301")], "lower>0", "AB"),
            "H4": ([(1, "{d}:202407_202407"), (-1, "{d}:202301_202301")], "upper<0", "AB"),
            "H5": ([(1, "{d}:202301_202407"), (-1, "{d}:201305_202407")], "upper>0", "A"),
        },
        "ceiling": {"H2"},
        "ratio": {"H2": ("{d}:202407_202407", "{d}:202301_202407")},   # E(new single) / E(key), part A only
    },
    "11": {
        "key": "base_inst",
        "show": ["base_base", "base_inst", "inst_base", "inst_inst"],
        "contrasts": {
            "H1": ([(1, "{d}:base_inst"), (-1, "{d}:base_base")], "lower>0", "AB"),
            "H2": ([(1, "{d}:base_inst"), (-1, "{d}:inst_inst")], "lower>0", "AB"),
            "H3": ([(1, "{d}:inst_inst"), (-1, "{d}:base_base")], "upper<0", "AB"),
            "H4a": ([(1, "{d}:inst_base"), (-1, "{d}:base_base")], "upper<0", "AB"),
            "H4b": ([(1, "{d}:inst_base"), (-1, "{d}:inst_inst")], "upper<0", "AB"),
            "H5": ([(1, "s09"), (-1, "{d}:base_inst")], "lower>0", "P2"),   # P1: described (vs_s09)
        },
        "ceiling": {"H1", "H2", "H3", "H4a", "H4b"},
        "ratio": {},
    },
}
# described, not judged
EXTRA = {
    "vs_dg_key": [(1, "{d}:KEY"), (-1, "dg:{d}:2022_2024")],
    "vs_ll24": [(1, "{d}:KEY"), (-1, "ll24")],
    "vs_s09": [(1, "{d}:KEY"), (-1, "s09")],
    "vs_bino1b": [(1, "{d}:KEY"), (-1, "bino1b")],
}


# ---------------- loading ----------------
def read_jsonl(path: str) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def grid_scores(paths: list[str], prefix: str = "") -> tuple[dict, dict]:
    """id -> {key: score}, id -> row header (group, label, pair_id, generator)."""
    sc, head = {}, {}
    for p in paths:
        for r in read_jsonl(p):
            if r["id"] in sc:
                raise SystemExit(f"{r['id']} appears twice in {paths}")
            s = {}
            for cell, c in r["cells"].items():
                s[f"{prefix}F:{cell}"] = float(c["fdgpt"])
                s[f"{prefix}Bi:{cell}"] = -float(c["bino"])
                a, b = cell.split("_")
                if a == b:
                    s[f"{prefix}ll:{a}"] = float(c["ll"])
            sc[r["id"]] = s
            head[r["id"]] = {k: r[k] for k in ("group", "label", "pair_id", "generator")}
    return sc, head


def baselines04(b_input: str, dated: list[str], bino: str, fd: str, ids: set[str]) -> dict:
    """04's comparison scores for part B texts (same formulas as det09.text_scores)."""
    from det04 import load_dated
    lp = load_dated(dated, ids)
    rb = {r["id"]: r for r in read_jsonl(bino)}
    rf = {r["id"]: r for r in read_jsonl(fd)}
    out = {}
    for i in ids:
        if i in lp and {"2022", "2024"} <= set(lp[i]) and i in rb and i in rf:
            m24, m22 = lp[i]["2024"].mean(), lp[i]["2022"].mean()
            out[i] = {"ll24": float(m24), "s075": float(m24 - 0.75 * m22), "s09": float(m24 - 0.9 * m22),
                      "bino1b": -float(rb[i]["binoculars"]), "fdneo": float(rf[i]["fdgpt"])}
    return out


# ---------------- comparisons ----------------
class Comp:
    def __init__(self, name: str, part: str, h_ids: list[str], a_ids: list[str], sc: dict, cluster: dict):
        if not h_ids or not a_ids:
            raise SystemExit(f"{name}: empty side ({len(h_ids)} human, {len(a_ids)} AI)")
        ids = h_ids + a_ids
        self.name, self.part, self.n_h, self.n_a = name, part, len(h_ids), len(a_ids)
        self.ids = ids
        self.y = np.r_[np.zeros(len(h_ids)), np.ones(len(a_ids))].astype(bool)
        keys = set(sc[ids[0]])
        for i in ids[1:]:
            keys &= set(sc[i])
        self.keys = sorted(keys)
        self.S = {k: np.array([sc[i][k] for i in ids]) for k in self.keys}
        _, cl = np.unique([cluster[i] for i in ids], return_inverse=True)
        self.members = [np.flatnonzero(cl == k) for k in range(int(cl.max()) + 1)]

    def aurocs(self, idx=None) -> dict[str, float]:
        idx = np.arange(len(self.y)) if idx is None else idx
        y = self.y[idx]
        return {k: auroc(v[idx][y], v[idx][~y]) for k, v in self.S.items()}

    def tprs(self) -> dict[str, float]:
        return {k: tpr_at(v[self.y], v[~self.y]) for k, v in self.S.items()}

    def draw(self, rng) -> np.ndarray:
        pick = rng.integers(0, len(self.members), len(self.members))
        return np.concatenate([self.members[k] for k in pick])


def build(sc: dict, head: dict, meta_b: dict) -> list[Comp]:
    comps = []
    by: dict[str, list[str]] = {}
    for i, h in head.items():
        by.setdefault(h["group"], []).append(i)
    pair_cluster = {i: h["pair_id"] for i, h in head.items()}
    for hg, ag, role in A_SETS:
        hp = {head[i]["pair_id"]: i for i in by.get(hg, [])}
        ap = {head[i]["pair_id"]: i for i in by.get(ag, [])}
        common = sorted(set(hp) & set(ap))
        if common:
            c = Comp(f"{hg}/{ag}", "A", [hp[p] for p in common], [ap[p] for p in common], sc, pair_cluster)
            c.role = role
            comps.append(c)
    if meta_b:
        G = lambda *gs: [i for g in gs for i in by.get(g, [])]  # noqa: E731
        cl = {i: meta_b[i]["cluster"] for i in head if i in meta_b}
        apt_ai = [i for g in by if g.startswith("APT.") and g != "APT.H" for i in by[g]]
        strong = [i for i in apt_ai if meta_b[i].get("level") in ("slight_major", "major")]
        for name, h, a in ((P1, G("RAID.H"), G(*[f"RAID.{m}.sampling" for m in CHAT])), (P2, G("APT.H"), strong)):
            if h and a:
                c = Comp(name, "B", h, a, sc, cl)
                c.role = "main"
                comps.append(c)
    return comps


def combo(a: dict, terms, d: str, key: str) -> float | None:
    ks = [k.format(d=d).replace("KEY", key) for _, k in terms]
    if not all(k in a for k in ks):
        return None
    return float(sum(sign * a[k] for (sign, _), k in zip(terms, ks)))


def status(rule: str, lo: float, hi: float) -> str:
    ok = {"lower>0": lo > 0, "upper<0": hi < 0, "upper>0": hi > 0}[rule]
    return "holds" if ok else "fails"


def evaluate(exp: str, comps: list[Comp], n_boot: int, seed: int) -> dict:
    cfg = EXP[exp]
    key = cfg["key"]
    rng = np.random.default_rng(seed)
    res = {"exp": exp, "key": key, "sets": {}}
    for c in comps:
        point = c.aurocs()
        boots = [c.aurocs(c.draw(rng)) for _ in range(n_boot)]
        out = {"part": c.part, "role": c.role, "n_h": c.n_h, "n_ai": c.n_a, "auroc": point, "tpr5": c.tprs(),
               "contrasts": {}}
        for d in DETS:
            con = {}
            items = [(n, t, r) for n, (t, r, parts) in cfg["contrasts"].items()
                     if (parts == "P2" and c.name == P2) or (parts != "P2" and c.part in parts)]
            items += [(n, t, None) for n, t in EXTRA.items()]
            for n, terms, rule in items:
                v = combo(point, terms, d, key)
                if v is None:
                    continue
                vals = [combo(b, terms, d, key) for b in boots]
                lo, hi = np.percentile(vals, [2.5, 97.5]).tolist()
                cells = [k.format(d=d).replace("KEY", key) for _, k in terms]
                entry = {"terms": [[s, k] for (s, _), k in zip(terms, cells)], "a": [point[k] for k in cells],
                         "diff": v, "ci": [lo, hi], "status": status(rule, lo, hi) if rule else "-"}
                if rule and n in cfg["ceiling"] and len(cells) == 2 and min(entry["a"]) >= CEIL:
                    entry["status"] = "ceiling"
                if rule and n in cfg["ratio"] and c.part == "A" and entry["status"] == "holds":
                    num, den = (x.format(d=d) for x in cfg["ratio"][n])
                    e_num, e_den = 1 - point[num], 1 - point[den]
                    entry["error_ratio"] = float(e_num / e_den) if e_den > 0 else float("inf")
                    if entry["error_ratio"] < RATIO:
                        entry["status"] = "fails (error ratio)"
                con[n] = entry
            out["contrasts"][d] = con
        gens = {}
        if c.part == "B" or exp == "11":
            for g in sorted({HEAD[i]["generator"] for i in c.ids[c.n_h:]}):
                idx = np.r_[np.arange(c.n_h), c.n_h + np.flatnonzero(
                    np.array([HEAD[i]["generator"] == g for i in c.ids[c.n_h:]]))]
                gens[g] = {k: v for k, v in c.aurocs(idx).items() if k.split(":")[-1] in cfg["show"] or k in BASE}
        out["by_generator"] = gens
        res["sets"][c.name] = out
    res["verdict"] = JUDGE[exp](res)
    return res


def st(res, s, d, h):
    return res["sets"].get(s, {}).get("contrasts", {}).get(d, {}).get(h, {}).get("status", "missing")


def judge10(res: dict) -> dict:
    v = {"per_set": {s: {d: {h: st(res, s, d, h) for h in EXP["10"]["contrasts"]} for d in DETS} for s in res["sets"]}}
    rep = {}
    for d in DETS:
        x = [st(res, s, d, "H2") for s in ("H19/C19", "B/C")]
        x = [y for y in x if y != "ceiling"]
        rep[d] = "undecidable" if not x else ("holds" if all(y == "holds" for y in x) else "fails")
    n = sum(rep[d] == "holds" for d in DETS)
    v["H2_part_A"] = rep
    v["replication"] = ("undecidable" if all(rep[d] == "undecidable" for d in DETS)
                        else "replicates" if n == 2 else "partial" if n == 1 else "fails")
    v["part_B_generalises"] = all(st(res, p, d, h) == "holds" for p in (P1, P2) for d in DETS for h in ("H1", "H2"))
    v["H4_all_four"] = all(st(res, s, d, "H4") == "holds" for s in ("H19/C19", "B/C", P1, P2) for d in DETS)
    return v


def judge11(res: dict) -> dict:
    sets = ("H19/C19", "B/C", P1, P2)
    v = {"per_set": {s: {d: {h: st(res, s, d, h) for h in EXP["11"]["contrasts"]} for d in DETS} for s in res["sets"]}}
    for s in v["per_set"]:
        for d in DETS:
            a, b = v["per_set"][s][d].get("H4a"), v["per_set"][s][d].get("H4b")
            v["per_set"][s][d]["H4"] = "holds" if a == b == "holds" else ("ceiling" if "ceiling" in (a, b) and
                                                                            "fails" not in (a, b) else "fails")
    h2 = {d: [st(res, s, d, "H2") for s in sets] for d in DETS}
    det_ok = {d: bool([x for x in h2[d] if x != "ceiling"]) and all(x in ("holds", "ceiling") for x in h2[d])
              for d in DETS}
    any_hold = any(x == "holds" for d in DETS for x in h2[d])
    v["H2_by_det"] = {d: dict(zip(sets, h2[d])) for d in DETS}
    v["two_slot"] = "holds" if all(det_ok.values()) else ("partial" if any_hold else "fails")
    h5 = {d: st(res, P2, d, "H5") for d in DETS}
    v["H5_P2"] = h5
    n5 = sum(x == "holds" for x in h5.values())
    v["practical"] = "s09 beats both original detectors" if n5 == 2 else (
        "s09 beats one" if n5 == 1 else "s09 beats neither")
    return v


JUDGE = {"10": judge10, "11": judge11}
HEAD: dict = {}


def print_report(res: dict) -> None:
    show = EXP[res["exp"]]["show"]
    for name, s in res["sets"].items():
        print(f"\n=== {name} (part {s['part']}, {s['role']}): {s['n_h']} human, {s['n_ai']} AI ===")
        print("cell            " + "".join(f"{d:>10}" for d in DETS) + "   (TPR@5%: " + " ".join(DETS) + ")")
        for cell in show:
            a = [s["auroc"].get(f"{d}:{cell}") for d in DETS]
            t = [s["tpr5"].get(f"{d}:{cell}") for d in DETS]
            if a[0] is None:
                continue
            print(f"{cell:<16}" + "".join(f"{x:>10.4f}" for x in a) + "    " + " ".join(f"{x:.2f}" for x in t))
        ll = {k: v for k, v in s["auroc"].items() if k.startswith("ll:")}
        print("ll (single model): " + ", ".join(f"{k[3:]} {v:.4f}" for k, v in sorted(ll.items())))
        ref = {k: v for k, v in s["auroc"].items() if k.startswith("dg:") or k in BASE}
        if ref:
            print("reference: " + ", ".join(f"{k} {v:.4f}" for k, v in sorted(ref.items())))
        for d in DETS:
            print(f"{d} contrasts:")
            for n, c in s["contrasts"][d].items():
                terms = " ".join(f"{'+' if sg > 0 else '-'}{k}" for sg, k in c["terms"])
                extra = f"  E ratio {c['error_ratio']:.2f}" if "error_ratio" in c else ""
                print(f"  {n:<10} {terms}: {c['diff']:+.4f} [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]{extra}  {c['status']}")
        if s["by_generator"]:
            print("by generator (AI side; human side whole):")
            for g, a in s["by_generator"].items():
                print(f"  {g:<16}" + ", ".join(f"{k} {a[k]:.3f}" for k in sorted(a) if k.startswith(("F:", "Bi:"))))
    print("\n=== verdict ===")
    print(json.dumps(res["verdict"], indent=1))


def main() -> None:
    global HEAD
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=sorted(EXP))
    ap.add_argument("--grid", nargs="+", required=True)
    ap.add_argument("--b-input", default=None, help="data/04_new.jsonl (part B meta and source clusters)")
    ap.add_argument("--dg-grid", nargs="*", default=[], help="08/09 DatedGPT grids, described next to the cells")
    ap.add_argument("--dated", nargs="*", default=[], help="04 DatedGPT score files yc04_2022, yc04_2024")
    ap.add_argument("--binoculars", default=None)
    ap.add_argument("--fdgpt", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    sc, HEAD = grid_scores(args.grid)
    if args.dg_grid:
        dg, _ = grid_scores(args.dg_grid, prefix="dg:")
        for i in sc:
            if i in dg:
                sc[i].update({k: v for k, v in dg[i].items() if k.split(":")[-1] in ("2022_2024", "2022_2022",
                                                                                    "2024_2024")})
    meta_b = {}
    if args.b_input:
        from det04 import read_meta
        meta_b = {i: m for i, m in read_meta(args.b_input).items() if i in sc}
        if args.dated and args.binoculars and args.fdgpt:
            base = baselines04(args.b_input, args.dated, args.binoculars, args.fdgpt, set(meta_b))
            for i, b in base.items():
                sc[i].update(b)
    print(f"{len(sc)} texts with grid scores ({len(meta_b)} in part B)")
    res = evaluate(args.exp, build(sc, HEAD, meta_b), args.n_boot, args.seed)
    print_report(res)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
