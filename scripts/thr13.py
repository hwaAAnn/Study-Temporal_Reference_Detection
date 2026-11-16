"""Experiment 13: threshold transfer. Each detector's threshold is set once, at 5% FPR on 08's H19 human abstracts
(arXiv 2019), and applied unchanged to other human/AI sets. Reads existing score files only (CPU).

Per detector and target set: FPR on the target's human texts, TPR on its AI texts, J = TPR - FPR at the transferred
threshold, and the TPR at the target's own 5% threshold (what per-set calibration would give). CIs come from a
source-cluster bootstrap (pair_id; 2,000 draws, seed 0). Each draw resamples the calibration set too, so threshold
uncertainty is included, and every detector uses the same draws, so differences are paired.

Families (scores with the same text ids are joined):
  local    DatedGPT (s08_grid + s09_main, local GTX 1060), Falcon-7B pair (s11_grid), TiC-LM (s10_A + s10_B)
  a100     DatedGPT (s12_dg) and Falcon (s12_fa), both scored on the A100 pod for experiment 12

usage: python scripts/thr13.py --outputs <main checkout>/outputs [--n-boot 2000] [--json out.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from det04 import CHAT, threshold5  # noqa: E402

CAL = "H19"
DG_CELLS = ["2022_2022", "2024_2024", "2022_2024"]
TIC_CELLS = ["202301_202301", "202407_202407", "202301_202407"]
P1, P2 = "RAID chat sampling (P1)", "APT strong (P2)"


def is_p1(g: str) -> bool:
    p = g.split(".")
    return p[0] == "RAID" and len(p) == 3 and p[1] in CHAT and p[2] == "sampling"


def is_p2(g: str) -> bool:
    return g.startswith("APT.") and g.split(".")[-1] in ("slight_major", "major")


# name -> (human group, AI-group predicate, role)
TARGETS = {
    "local": {
        "H19/C19 (calibration set, no transfer)": ("H19", lambda g: g == "C19", "in-set"),
        P1: ("RAID.H", is_p1, "same genre, other source"),
        P2: ("APT.H", is_p2, "other domains"),
        "H24/C24": ("H24", lambda g: g == "C24", "later human writing (2024)"),
        "B/C": ("B", lambda g: g == "C", "later human writing (2025-07~2026-10)"),
    },
    "a100": {
        "H19/N19 (calibration set, no transfer)": ("H19", lambda g: g.startswith("N19."), "in-set"),
        "B/NB": ("B", lambda g: g.startswith("NB."), "later human writing, 2025-26 generators"),
    },
}


def read(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r
    return out


def dg_scores(c: dict, prefix: str = "") -> dict[str, float]:
    s = {}
    for cell in DG_CELLS:
        s[f"{prefix}F:{cell}"] = c[cell]["fdgpt"]
        s[f"{prefix}Bi:{cell}"] = -c[cell]["bino"]
    s[f"{prefix}ll24"] = c["2024_2024"]["ll"]
    s[f"{prefix}s09"] = c["2024_2024"]["ll"] - 0.9 * c["2022_2022"]["ll"]
    return s


def fa_scores(c: dict) -> dict[str, float]:
    return {"Falcon F": c["base_inst"]["fdgpt"], "Falcon Bi": -c["base_inst"]["bino"]}


def tic_scores(c: dict) -> dict[str, float]:
    s = {}
    for cell in TIC_CELLS:
        s[f"TiC F:{cell}"] = c[cell]["fdgpt"]
        s[f"TiC Bi:{cell}"] = -c[cell]["bino"]
    s["TiC ll407"] = c["202407_202407"]["ll"]
    s["TiC s09"] = c["202407_202407"]["ll"] - 0.9 * c["202301_202301"]["ll"]
    return s


def load_family(out: Path, family: str):
    if family == "local":
        parts = [({**read(out / "s08_grid.jsonl"), **read(out / "s09_main.jsonl")}, dg_scores),
                 (read(out / "s11_grid.jsonl"), fa_scores),
                 ({**read(out / "s10_A.jsonl"), **read(out / "s10_B.jsonl")}, tic_scores)]
    else:
        parts = [(read(out / "s12_dg.jsonl"), dg_scores), (read(out / "s12_fa.jsonl"), fa_scores)]
    ids = set(parts[0][0])
    for recs, _ in parts[1:]:
        ids &= set(recs)
    meta, sc = {}, {}
    for i in sorted(ids):
        r = parts[0][0][i]
        meta[i] = (r["group"], int(r["label"]), str(r["pair_id"]))
        sc[i] = {}
        for recs, fn in parts:
            sc[i].update(fn(recs[i]["cells"]))
    dropped = {len(recs) - len(ids) for recs, _ in parts}
    return meta, sc, dropped


class Block:
    """Texts of one set (calibration humans, or a target's humans + AI) with their clusters."""

    def __init__(self, ids, meta, sc, keys):
        self.y = np.array([meta[i][1] for i in ids], dtype=bool)
        self.S = np.array([[sc[i][k] for k in keys] for i in ids])  # texts x detectors
        _, cl = np.unique([meta[i][2] for i in ids], return_inverse=True)
        self.members = [np.flatnonzero(cl == k) for k in range(int(cl.max()) + 1)]

    def draw(self, rng) -> np.ndarray:
        pick = rng.integers(0, len(self.members), len(self.members))
        return np.concatenate([self.members[k] for k in pick])


def thresholds(S: np.ndarray) -> np.ndarray:
    return np.array([threshold5(S[:, j]) for j in range(S.shape[1])])


def rates(blk: Block, idx: np.ndarray, tau: np.ndarray):
    y, S = blk.y[idx], blk.S[idx]
    fpr = (S[~y] > tau).mean(axis=0)
    tpr = (S[y] > tau).mean(axis=0)
    return fpr, tpr


def evaluate(meta, sc, targets, n_boot=2000, seed=0) -> dict:
    keys = sorted(next(iter(sc.values())))
    k = {name: j for j, name in enumerate(keys)}
    cal = Block([i for i, m in meta.items() if m[0] == CAL and m[1] == 0], meta, sc, keys)
    blocks = {}
    for name, (hg, pred, _) in targets.items():
        ids = [i for i, m in meta.items() if (m[1] == 0 and m[0] == hg) or (m[1] == 1 and pred(m[0]))]
        blocks[name] = Block(ids, meta, sc, keys)
    rng = np.random.default_rng(seed)
    tau0 = thresholds(cal.S)
    res = {"keys": keys, "n_cal": int((~cal.y).sum()), "tau": dict(zip(keys, tau0.tolist())), "targets": {}}
    point, boots = {}, {name: [] for name in targets}
    for name, blk in blocks.items():
        all_idx = np.arange(len(blk.y))
        fpr, tpr = rates(blk, all_idx, tau0)
        own = np.array([np.mean(blk.S[blk.y, j] > threshold5(blk.S[~blk.y, j])) for j in range(len(keys))])
        point[name] = (fpr, tpr)
        res["targets"][name] = {"n_h": int((~blk.y).sum()), "n_ai": int(blk.y.sum()), "role": targets[name][2],
                                "fpr": dict(zip(keys, fpr.tolist())), "tpr": dict(zip(keys, tpr.tolist())),
                                "J": dict(zip(keys, (tpr - fpr).tolist())), "tpr_own": dict(zip(keys, own.tolist()))}
    for _ in range(n_boot):
        tau = thresholds(cal.S[cal.draw(rng)])
        for name, blk in blocks.items():
            boots[name].append(rates(blk, blk.draw(rng), tau))
    for name in targets:
        F = np.array([b[0] for b in boots[name]])
        T = np.array([b[1] for b in boots[name]])
        t = res["targets"][name]
        t["fpr_ci"] = {kk: np.percentile(F[:, j], [2.5, 97.5]).tolist() for kk, j in k.items()}
        t["tpr_ci"] = {kk: np.percentile(T[:, j], [2.5, 97.5]).tolist() for kk, j in k.items()}
        t["_F"], t["_T"] = F, T
    return res


def contrast(res: dict, target: str, a: str, b: str, what: str = "J") -> dict:
    """a - b for J (= TPR - FPR) or FPR at the transferred threshold, with the bootstrap CI."""
    t, keys = res["targets"][target], res["keys"]
    ia, ib = keys.index(a), keys.index(b)
    if what == "J":
        pt = t["J"][a] - t["J"][b]
        bs = (t["_T"][:, ia] - t["_F"][:, ia]) - (t["_T"][:, ib] - t["_F"][:, ib])
    else:
        pt = t["fpr"][a] - t["fpr"][b]
        bs = t["_F"][:, ia] - t["_F"][:, ib]
    lo, hi = np.percentile(bs, [2.5, 97.5]).tolist()
    return {"diff": pt, "ci": [lo, hi]}


def verdict(res: dict) -> dict:
    v = {}
    for d in ("F", "Bi"):
        mix, s24, s22 = f"{d}:2022_2024", f"{d}:2024_2024", f"{d}:2022_2022"
        c1 = contrast(res, P1, mix, s24)
        c2 = contrast(res, P1, mix, s22)
        v[f"T1 {d}"] = {"vs_single2024": c1, "vs_single2022": c2,
                        "status": "holds" if c1["ci"][0] > 0 and c2["ci"][0] > 0 else "fails"}
        # point estimate: with 400 humans on each side the threshold's own uncertainty makes the CI about +-0.04
        # even when the true FPR is exactly 5%, so a CI-upper rule would fail a perfectly calibrated detector
        f = res["targets"][P1]["fpr"][mix]
        v[f"T2 {d}"] = {"fpr": f, "ci": res["targets"][P1]["fpr_ci"][mix],
                        "status": "holds" if f <= 0.10 else "fails"}
        t3 = {tg: contrast(res, tg, mix, s24, "FPR") for tg in ("H24/C24", "B/C")}
        v[f"T3 {d}"] = {**t3, "status": "holds" if all(c["ci"][0] > 0 for c in t3.values()) else "fails"}
    v["rule survives transfer (T1 for F and Bi)"] = all(v[f"T1 {d}"]["status"] == "holds" for d in ("F", "Bi"))
    return v


SHOW = {
    "local": ["F:2022_2022", "F:2024_2024", "F:2022_2024", "Bi:2022_2022", "Bi:2024_2024", "Bi:2022_2024",
              "ll24", "s09", "Falcon F", "Falcon Bi", "TiC F:202301_202301", "TiC F:202407_202407",
              "TiC F:202301_202407", "TiC ll407", "TiC s09"],
    "a100": ["F:2022_2022", "F:2024_2024", "F:2022_2024", "Bi:2022_2022", "Bi:2024_2024", "Bi:2022_2024",
             "ll24", "s09", "Falcon F", "Falcon Bi"],
}
EXTRA = {  # described, not judged: (target, a, b, what)
    "local": [(P1, "Bi:2022_2024", "Falcon Bi", "J"), (P1, "F:2022_2024", "ll24", "J"),
              (P1, "F:2022_2024", "s09", "J"), (P2, "s09", "F:2022_2024", "J"), (P2, "s09", "Falcon Bi", "J"),
              (P1, "TiC F:202301_202407", "TiC F:202407_202407", "J"),
              (P1, "TiC F:202301_202407", "TiC F:202301_202301", "J")],
    "a100": [("B/NB", "F:2022_2024", "F:2024_2024", "J"), ("B/NB", "F:2022_2024", "F:2022_2022", "J"),
             ("B/NB", "Bi:2022_2024", "Falcon Bi", "J"), ("B/NB", "s09", "F:2022_2024", "J"),
             ("B/NB", "F:2022_2024", "F:2024_2024", "FPR")],
}


def report(fam: str, res: dict) -> None:
    print(f"\n######## family: {fam}  (threshold: 5% FPR on {res['n_cal']} {CAL} human abstracts)")
    for name, t in res["targets"].items():
        print(f"\n== {name}  [{t['role']}]  {t['n_h']} human, {t['n_ai']} AI")
        print(f"{'detector':22s} {'FPR':>6s} {'[95% CI]':>15s} {'TPR':>6s} {'[95% CI]':>15s} {'J':>6s} | own-threshold TPR")
        for kk in SHOW[fam]:
            f, (fl, fh) = t["fpr"][kk], t["fpr_ci"][kk]
            tp, (tl, th) = t["tpr"][kk], t["tpr_ci"][kk]
            print(f"{kk:22s} {f:6.3f} [{fl:.3f}, {fh:.3f}] {tp:6.3f} [{tl:.3f}, {th:.3f}] {tp - f:6.3f} | {t['tpr_own'][kk]:.3f}")
    print("\n== contrasts (described)")
    for tg, a, b, what in EXTRA[fam]:
        c = contrast(res, tg, a, b, what)
        print(f"{tg} | {what}: {a} - {b}: {c['diff']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]")


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if not str(k).startswith("_")}
    return o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", required=True, help="the main checkout's outputs/ directory")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json")
    args = ap.parse_args()
    out, allres = Path(args.outputs), {}
    for fam in ("local", "a100"):
        meta, sc, dropped = load_family(out, fam)
        print(f"{fam}: {len(meta)} texts with every score (texts missing from some file: {sorted(dropped)})")
        res = evaluate(meta, sc, TARGETS[fam], args.n_boot, args.seed)
        report(fam, res)
        if fam == "local":
            res["verdict"] = verdict(res)
        allres[fam] = res
    v = allres["local"]["verdict"]
    print("\n=== verdict (local family; T1-T3 as pre-registered)")
    for kk, x in v.items():
        print(f"{kk}: {x if isinstance(x, bool) else x['status']}")
    for d in ("F", "Bi"):
        a, b = v[f"T1 {d}"]["vs_single2024"], v[f"T1 {d}"]["vs_single2022"]
        print(f"  T1 {d}: J mixed - single2024 {a['diff']:+.3f} [{a['ci'][0]:+.3f}, {a['ci'][1]:+.3f}];"
              f" mixed - single2022 {b['diff']:+.3f} [{b['ci'][0]:+.3f}, {b['ci'][1]:+.3f}]")
        t2 = v[f"T2 {d}"]
        print(f"  T2 {d}: FPR on RAID.H {t2['fpr']:.3f} [{t2['ci'][0]:.3f}, {t2['ci'][1]:.3f}]")
        for tg in ("H24/C24", "B/C"):
            c = v[f"T3 {d}"][tg]
            print(f"  T3 {d} {tg}: FPR mixed - single2024 {c['diff']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]")
    if args.json:
        Path(args.json).write_text(json.dumps(strip(allres), indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
