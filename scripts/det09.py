"""Experiment 09: the 08 detector slots (reference r, scorer s; DatedGPT vintages) on the 04 public benchmarks
(RAID abstracts, APT-Eval).

Per text, from score_slots.py: F[r_s] = Fast-DetectGPT and Bi[r_s] = -Binoculars for every cell in the grid file.
The main groups (RAID.H, RAID chat sampling, APT.H, APT strong) carry vintages 2018, 2021, 2022, 2024 (16 cells);
the other groups carry 2022 and 2024 (4 cells). Higher = more AI throughout.
From 04, unchanged, for comparison: ll24, s075, s09 (DatedGPT score files), bino1b (Binoculars, Llama-3.2-1B
observer / -Instruct performer) and fdneo (Fast-DetectGPT, GPT-Neo-2.7B).

Comparisons and pools are those of det04 (AI groups vs the human group; attacks vs humans with the same attack).
AUROC and TPR at 5% FPR (threshold from the comparison's human texts). Pools carry 95% CIs from 2000 bootstrap
draws over sources (RAID source abstract, APT original), seed 0, the same draws for every detector.

Hypotheses, for F and Bi on the two main pools, P1 = RAID chat sampling and P2 = APT strong:
  H1 scorer slot      A(2022_2024) - A(2022_2022)                                   holds if CI lower > 0
  H2 reference slot   A(2022_2024) - A(2024_2024)                                   holds if CI lower > 0
  H3 boundary         [A(2022_2024) - A(2024_2024)] - [A(2018_2021) - A(2021_2021)] holds if CI lower > 0
  H4 newest single    A(2024_2024) - A(2022_2022)                                   holds if CI upper < 0
  H5 vs 04 detectors  on P2 only: (a) A(2022_2024) - A(bino1b), (b) A(2022_2024) - A(s09); holds if CI lower > 0
Verdict "generalises" if H1 and H2 hold for both detectors on both main pools.

Example:
  python scripts/det09.py --input data/04_new.jsonl --grid outputs/s09_main.jsonl outputs/s09_rest.jsonl \
      --dated outputs/yc04_2018.jsonl outputs/yc04_2021.jsonl outputs/yc04_2022.jsonl outputs/yc04_2024.jsonl \
      --binoculars outputs/b04_binoculars.jsonl --fdgpt outputs/b04_fdgpt.jsonl \
      --json outputs/det09.json > outputs/det09.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from det04 import ATTACK_MODELS, CHAT, LEVELS, PRECHAT, auroc, load_dated, read_meta, tpr5  # noqa: E402

DETS = ["F", "Bi"]
BASE = ["ll24", "s075", "s09", "bino1b", "fdneo"]
KEY = "2022_2024"
P1, P2 = "RAID chat, sampling", "APT strong (slight_major+major)"
SHOW_CELLS = ["2022_2022", "2024_2024", "2022_2024", "2024_2022"]
MAIN_CELLS = ["2018_2018", "2021_2021", "2018_2021", "2021_2018", "2018_2024", "2021_2024"]

# name -> (terms [(sign, key)], rule, pools); a key is "det:cell" with det filled in, or a 04 baseline
CONTRASTS = {
    "H1": ([(1, "{d}:2022_2024"), (-1, "{d}:2022_2022")], "lower>0", (P1, P2)),
    "H2": ([(1, "{d}:2022_2024"), (-1, "{d}:2024_2024")], "lower>0", (P1, P2)),
    "H3": ([(1, "{d}:2022_2024"), (-1, "{d}:2024_2024"), (-1, "{d}:2018_2021"), (1, "{d}:2021_2021")],
           "lower>0", (P1, P2)),
    "H4": ([(1, "{d}:2024_2024"), (-1, "{d}:2022_2022")], "upper<0", (P1, P2)),
    "H5a": ([(1, "{d}:2022_2024"), (-1, "bino1b")], "lower>0", (P2,)),
    "H5b": ([(1, "{d}:2022_2024"), (-1, "s09")], "lower>0", (P2,)),
}
# described, not judged
EXTRA = {
    "vs_ll24": [(1, "{d}:2022_2024"), (-1, "ll24")],
    "vs_bino1b": [(1, "{d}:2022_2024"), (-1, "bino1b")],
    "vs_s09": [(1, "{d}:2022_2024"), (-1, "s09")],
    "vs_fdneo": [(1, "{d}:2022_2024"), (-1, "fdneo")],
    "pre_ref_slot": [(1, "{d}:2018_2021"), (-1, "{d}:2021_2021")],
    "older_ref": [(1, "{d}:2018_2024"), (-1, "{d}:2022_2024")],
}


# ---------------- loading ----------------
def load_grid(paths: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["id"] in out:
                raise SystemExit(f"{r['id']} appears in more than one grid row")
            s = {}
            for cell, c in r["cells"].items():
                s[f"F:{cell}"] = float(c["fdgpt"])
                s[f"Bi:{cell}"] = -float(c["bino"])
            out[r["id"]] = s
    return out


def text_scores(meta, grid, lp, bino_path, fd_path) -> dict[str, dict[str, float]]:
    rd = lambda p: {json.loads(x)["id"]: json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x}  # noqa: E731
    bino, fd = rd(bino_path), rd(fd_path)
    out, missing = {}, []
    unknown = [i for i in grid if i not in meta]
    if unknown:
        raise SystemExit(f"{len(unknown)} grid texts are not in the input (e.g. {unknown[:3]})")
    print(f"{len(grid)} of {len(meta)} input texts have grid scores")
    for i in grid:
        if i not in lp or not {"2022", "2024"} <= set(lp[i]) or i not in bino or i not in fd:
            missing.append(i)
            continue
        m24, m22 = lp[i]["2024"].mean(), lp[i]["2022"].mean()
        out[i] = dict(grid[i], ll24=float(m24), s075=float(m24 - 0.75 * m22), s09=float(m24 - 0.9 * m22),
                      bino1b=-float(bino[i]["binoculars"]), fdneo=float(fd[i]["fdgpt"]))
    if missing:
        raise SystemExit(f"{len(missing)} texts lack a score (e.g. {missing[:3]})")
    return out


# ---------------- comparisons ----------------
class Comp:
    def __init__(self, name, h_ids, a_ids, sc, meta):
        if not h_ids or not a_ids:
            raise SystemExit(f"{name}: empty side ({len(h_ids)} human, {len(a_ids)} AI)")
        ids = h_ids + a_ids
        self.name, self.n_h, self.n_a = name, len(h_ids), len(a_ids)
        self.y = np.r_[np.zeros(len(h_ids)), np.ones(len(a_ids))].astype(bool)
        keys = set(sc[ids[0]])
        for i in ids[1:]:
            keys &= set(sc[i])
        self.keys = sorted(keys)
        self.S = {k: np.array([sc[i][k] for i in ids]) for k in self.keys}
        _, cl = np.unique([meta[i]["cluster"] for i in ids], return_inverse=True)
        self.members = [np.flatnonzero(cl == k) for k in range(int(cl.max()) + 1)]

    def aurocs(self, idx=None) -> dict[str, float]:
        idx = np.arange(len(self.y)) if idx is None else idx
        y = self.y[idx]
        return {k: auroc(v[idx][~y], v[idx][y]) for k, v in self.S.items()}

    def tprs(self) -> dict[str, float]:
        return {k: tpr5(v[~self.y], v[self.y]) for k, v in self.S.items()}

    def draw(self, rng) -> np.ndarray:
        pick = rng.integers(0, len(self.members), len(self.members))
        return np.concatenate([self.members[k] for k in pick])


def combo(a: dict[str, float], terms, d: str) -> float:
    return float(sum(sign * a[key.format(d=d)] for sign, key in terms))


def status(rule: str, lo: float, hi: float) -> str:
    ok = lo > 0 if rule == "lower>0" else hi < 0
    return "holds" if ok else "fails"


def comparisons(meta: dict, sc: dict):
    by: dict[str, list[str]] = {}
    for i, m in meta.items():
        if i in sc:
            by.setdefault(m["group"], []).append(i)
    G = lambda *gs: [i for g in gs for i in by.get(g, [])]  # noqa: E731
    apt_ai = G(*[g for g in by if g.startswith("APT.") and g != "APT.H"])
    strong = lambda ids: [i for i in ids if meta[i].get("level") in ("slight_major", "major")]  # noqa: E731
    single, pools = {}, {}
    for m in CHAT:
        single[f"RAID {m} sampling"] = (G("RAID.H"), G(f"RAID.{m}.sampling"))
    for m in CHAT + PRECHAT:
        single[f"RAID {m} greedy"] = (G("RAID.H"), G(f"RAID.{m}.greedy"))
    for p in sorted({meta[i]["polisher"] for i in apt_ai}):
        single[f"APT strong, {p}"] = (G("APT.H"), strong([i for i in apt_ai if meta[i]["polisher"] == p]))
    for dom in sorted({meta[i]["domain"] for i in G("APT.H")}):
        hd = [i for i in G("APT.H") if meta[i]["domain"] == dom]
        single[f"APT strong, {dom}"] = (hd, strong([i for i in apt_ai if meta[i]["domain"] == dom]))
    pools[P1] = (G("RAID.H"), G(*[f"RAID.{m}.sampling" for m in CHAT]))
    pools[P2] = (G("APT.H"), strong(apt_ai))
    pools["RAID chat, greedy"] = (G("RAID.H"), G(*[f"RAID.{m}.greedy" for m in CHAT]))
    pools["RAID pre-chat, greedy"] = (G("RAID.H"), G(*[f"RAID.{m}.greedy" for m in PRECHAT]))
    pools["RAID pre-chat, sampling"] = (G("RAID.H"), G(*[f"RAID.{m}.sampling" for m in PRECHAT]))
    pools["RAID 4 chat, sampling (no attack)"] = (G("RAID.H"), G(*[f"RAID.{m}.sampling" for m in ATTACK_MODELS]))
    for att in ("paraphrase", "synonym"):
        pools[f"RAID 4 chat, sampling +{att}"] = (G(f"RAID.H.{att}"),
                                                   G(*[f"RAID.{m}.sampling.{att}" for m in ATTACK_MODELS]))
    pools["APT weak (extreme_minor+minor)"] = (G("APT.H"), [i for i in apt_ai if meta[i]["level"] in LEVELS[:2]])
    return single, pools


def evaluate(meta: dict, sc: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    single, pools = comparisons(meta, sc)
    res = {"single": {}, "pools": {}}
    for name, (h, a) in single.items():
        if not h or not a:
            continue
        c = Comp(name, h, a, sc, meta)
        res["single"][name] = {"n_h": c.n_h, "n_ai": c.n_a, "auroc": c.aurocs()}
    for name, (h, a) in pools.items():
        if not h or not a:
            continue
        c = Comp(name, h, a, sc, meta)
        point = c.aurocs()
        boots = [c.aurocs(c.draw(rng)) for _ in range(n_boot)]
        ci = {k: np.percentile([b[k] for b in boots], [2.5, 97.5]).tolist() for k in c.keys}
        out = {"n_h": c.n_h, "n_ai": c.n_a, "auroc": point, "auroc_ci": ci, "tpr5": c.tprs(), "contrasts": {}}
        for d in DETS:
            con = {}
            items = [(n, t, r) for n, (t, r, ps) in CONTRASTS.items() if name in ps]
            items += [(n, t, None) for n, t in EXTRA.items()]
            for n, terms, rule in items:
                if not all(k.format(d=d) in point for _, k in terms):
                    continue
                v = combo(point, terms, d)
                lo, hi = np.percentile([combo(b, terms, d) for b in boots], [2.5, 97.5]).tolist()
                con[n] = {"diff": v, "ci": [lo, hi], "status": status(rule, lo, hi) if rule else "-"}
            out["contrasts"][d] = con
        res["pools"][name] = out
    res["verdict"] = judge(res)
    return res


def judge(res: dict) -> dict:
    v = {"per_pool": {}}
    for p in (P1, P2):
        if p not in res["pools"]:
            continue
        v["per_pool"][p] = {d: {n: c["status"] for n, c in res["pools"][p]["contrasts"][d].items() if c["status"] != "-"}
                            for d in DETS}
    have = all(p in v["per_pool"] for p in (P1, P2))
    v["generalises"] = bool(have and all(v["per_pool"][p][d].get(h) == "holds"
                                         for p in (P1, P2) for d in DETS for h in ("H1", "H2")))
    return v


# ---------------- report ----------------
def print_report(res: dict) -> None:
    def row(name, n_h, n_a, a, keys):
        return f"{name:<38}{n_h:>5}{n_a:>6}" + "".join(f"{a[k]:>11.3f}" if k in a else f"{'':>11}" for k in keys)

    keys = [f"{d}:{c}" for d in DETS for c in SHOW_CELLS] + BASE
    hdr = f"{'comparison':<38}{'n_h':>5}{'n_ai':>6}" + "".join(f"{k.replace('_', '/'):>11}" for k in keys)
    print("=== pools: AUROC (cells r_s = reference_scorer) ===")
    print(hdr)
    for name, p in res["pools"].items():
        print(row(name, p["n_h"], p["n_ai"], p["auroc"], keys))
    print("\n=== main pools: every cell, AUROC [95% CI] and TPR@5% ===")
    for name in (P1, P2):
        p = res["pools"].get(name)
        if not p:
            continue
        print(f"\n{name}  ({p['n_h']} human, {p['n_ai']} AI)")
        for k in [f"{d}:{c}" for d in DETS for c in SHOW_CELLS + MAIN_CELLS] + BASE:
            if k in p["auroc"]:
                lo, hi = p["auroc_ci"][k]
                print(f"  {k:<14}{p['auroc'][k]:.4f} [{lo:.4f}, {hi:.4f}]  TPR {p['tpr5'][k]:.2f}")
    print("\n=== contrasts ===")
    for name, p in res["pools"].items():
        for d in DETS:
            con = p["contrasts"].get(d, {})
            if not con:
                continue
            print(f"{name} | {d}: " + "; ".join(f"{n} {c['diff']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]"
                                                + (f" {c['status']}" if c["status"] != "-" else "") for n, c in con.items()))
    print("\n=== single comparisons (exploratory), AUROC ===")
    print(hdr)
    for name, s in res["single"].items():
        print(row(name, s["n_h"], s["n_ai"], s["auroc"], keys))
    v = res["verdict"]
    print("\n=== verdict ===")
    for p, dd in v["per_pool"].items():
        print(f"{p}: " + "; ".join(f"{d} " + ", ".join(f"{h} {x}" for h, x in hs.items()) for d, hs in dd.items()))
    print(f"generalises (H1 and H2, both detectors, both main pools): {v['generalises']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="data/04_new.jsonl")
    ap.add_argument("--grid", nargs="+", required=True, help="score_slots.py outputs")
    ap.add_argument("--dated", nargs="+", required=True, help="04 DatedGPT score files (2022 and 2024 at least)")
    ap.add_argument("--binoculars", required=True)
    ap.add_argument("--fdgpt", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json")
    args = ap.parse_args()
    meta = read_meta(args.input)
    sc = text_scores(meta, load_grid(args.grid), load_dated(args.dated, set(meta)), args.binoculars, args.fdgpt)
    print(f"texts: {len(sc)}")
    res = evaluate(meta, sc, args.n_boot, args.seed)
    print_report(res)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
