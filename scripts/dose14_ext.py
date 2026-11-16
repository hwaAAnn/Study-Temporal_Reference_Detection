"""Experiment 14, parts B and C: the part A tests on other generators, with the dose G held fixed at its part A
value (B/C pairs of 08, small Qwen/Llama generators). Every outcome here is out of sample for G.

Part B  12's texts (outputs/s12_dg.jsonl, full DatedGPT 5x5 grid), same-title design:
        pools N19 = H19 humans vs all four N19.<gen> (main), NB = B humans vs all four NB.<gen> (replication);
        each generator also alone (described). Generators: gpt-oss-120b, GLM-4.7-Flash, Qwen3-30B-A3B
        Instruct / Thinking (2025-2026 releases).
Part C  09's main pools rescored with the five 08 vintages (outputs/s14_C.jsonl):
        P1 = RAID chat sampling (ChatGPT, GPT-4, Llama-2-chat, Mistral-chat, Cohere-chat, MPT-chat) vs RAID.H,
        P2 = APT strong (slight_major + major; GPT-4o, DeepSeek-V3, Llama3.1-70B, Llama3-8B, Llama2-7B) vs APT.H;
        each generator / polisher also alone (described). D4 on P1/P2 adds TiC (s10_B: 202301, 202407) and
        Falcon (s11_grid: base, inst).
Tests and statuses are those of dose14.py (D1 status rules, D2 flat band, D3 g, D4 rho band); clusters for the
bootstrap are pair_id (titles) for 12 and RAID, and the APT source text (det04 meta "cluster") for APT.

Example:
  python scripts/dose14_ext.py --part B --outputs <main>/outputs --json <main>/outputs/s14_B.json > <main>/outputs/s14_B.txt
  python scripts/dose14_ext.py --part C --outputs <main>/outputs --input <main>/data/04_new.jsonl \
      --json <main>/outputs/s14_Cev.json > <main>/outputs/s14_Cev.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dose14 as dz  # noqa: E402

DG = dz.FAMILIES["DG"]["models"]
D1_CELLS = ["2022_2024", "2023_2024", "2024_2024", "2013_2024", "2019_2024", "2022_2022", "2022_2023"]
GENS12 = ["gptoss120b", "glm47flash", "qwen3i", "qwen3t"]
CHAT = ["chatgpt", "gpt4", "cohere-chat", "llama-chat", "mistral-chat", "mpt-chat"]
OTHER = {"TiC": ("s10_B.jsonl", "202301", "202407"), "Falcon": ("s11_grid.jsonl", "base", "inst")}


def read_jsonl(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r
    return out


class Pool:
    """Human and AI ids with clusters; scores per (family, det, cell) as arrays aligned to the ids."""

    def __init__(self, name: str, h_ids: list[str], a_ids: list[str], cluster: dict[str, str]):
        self.name, self.ids = name, h_ids + a_ids
        self.y = np.r_[np.zeros(len(h_ids)), np.ones(len(a_ids))].astype(bool)
        _, cl = np.unique([cluster[i] for i in self.ids], return_inverse=True)
        self.members = [np.flatnonzero(cl == k) for k in range(int(cl.max()) + 1)]
        self.n_h, self.n_a, self.n_cl = len(h_ids), len(a_ids), len(self.members)

    def arr(self, rows: dict[str, dict], det: str, cell: str) -> np.ndarray:
        return np.array([dz.score(rows[i], det, cell) for i in self.ids])

    def draws(self, rng, n_boot: int) -> list[np.ndarray]:
        return [np.concatenate([self.members[k] for k in rng.integers(0, self.n_cl, self.n_cl)])
                for _ in range(n_boot)]


def d_of(x: np.ndarray, y: np.ndarray, idx: np.ndarray | None = None) -> float:
    if idx is not None:
        x, y = x[idx], y[idx]
    return float(dz.cohens_d_rows(x[y][None, :], x[~y][None, :])[0])


def pool_tests(pool: Pool, fams: dict[str, dict], G: dict, draws: list[np.ndarray] | None) -> dict:
    """D1-D3 (DatedGPT) and D4 (other families present) for one pool; bootstrap if draws is given."""
    out: dict = {"n_h": pool.n_h, "n_ai": pool.n_a, "n_clusters": pool.n_cl}
    gd = G["DG"]
    x = {r: gd["2024"] - gd[r] for r in gd}
    f_hat = x["2023"] / x["2022"]
    g_hat = (gd["2023"] - gd["2022"]) / (gd["2024"] - gd["2022"])
    for det in dz.DETS:
        S = {c: pool.arr(fams["DG"], det, c) for c in D1_CELLS}
        P = {c: d_of(S[c], pool.y) for c in S}
        B = {c: np.array([d_of(S[c], pool.y, i) for i in draws]) for c in S} if draws else None
        A = {c: dz.auroc(S[c][pool.y], S[c][~pool.y]) for c in ("2022_2024", "2023_2024", "2024_2024")}
        t: dict = {"d": P, "auroc": A}
        y22, y23 = P["2022_2024"] - P["2024_2024"], P["2023_2024"] - P["2024_2024"]
        f = y23 / y22
        d1 = {"y2022": y22, "y2023": y23, "f": f, "f_hat": f_hat,
              "f_auroc": (A["2023_2024"] - A["2024_2024"]) / (A["2022_2024"] - A["2024_2024"])}
        if B:
            y22b, y23b = B["2022_2024"] - B["2024_2024"], B["2023_2024"] - B["2024_2024"]
            d1.update(y2022_ci=dz.ci(y22b), y2023_ci=dz.ci(y23b), f_ci=dz.ci(y23b / y22b))
            d1["status"] = dz.d1_status(d1["f_ci"], f, f_hat, d1["y2022_ci"])
        t["D1"] = d1
        t["D2"] = {}
        for r in ("2013", "2019"):
            ratio = (P[f"{r}_2024"] - P["2024_2024"]) / y22
            t["D2"][r] = {"ratio": ratio, "pred": x[r] / x["2022"],
                          "status": "flat" if dz.FLAT[0] <= ratio <= dz.FLAT[1] else "not flat"}
            if B:
                t["D2"][r]["ci"] = dz.ci((B[f"{r}_2024"] - B["2024_2024"]) / (B["2022_2024"] - B["2024_2024"]))
        g = (P["2022_2023"] - P["2022_2022"]) / (P["2022_2024"] - P["2022_2022"])
        t["D3"] = {"g": g, "g_hat": g_hat}
        if B:
            t["D3"]["ci"] = dz.ci((B["2022_2023"] - B["2022_2022"]) / (B["2022_2024"] - B["2022_2022"]))
        others = [f for f in OTHER if f in fams]
        if others:
            k_dg = y22 / x["2022"]
            kb_dg = (B["2022_2024"] - B["2024_2024"]) / x["2022"] if B else None
            t["D4"] = {}
            for fam in others:
                _, pre, new = OTHER[fam]
                xf = G[fam][new] - G[fam][pre]
                a_pre, a_new = pool.arr(fams[fam], det, f"{pre}_{new}"), pool.arr(fams[fam], det, f"{new}_{new}")
                yf = d_of(a_pre, pool.y) - d_of(a_new, pool.y)
                rho = (yf / xf) / k_dg
                e = {"x": xf, "y": yf, "rho": rho,
                     "status": "same scale" if dz.SCALE[0] <= rho <= dz.SCALE[1] else "different scale"}
                if B:
                    yfb = np.array([d_of(a_pre, pool.y, i) - d_of(a_new, pool.y, i) for i in draws])
                    e["rho_ci"] = dz.ci((yfb / xf) / kb_dg)
                t["D4"][fam] = e
        out[det] = t
    return out


def part_b(out: Path, G: dict, n_boot: int, seed: int) -> dict:
    rows = read_jsonl(out / "s12_dg.jsonl")
    cluster = {i: r["pair_id"] for i, r in rows.items()}
    by: dict[str, list[str]] = {}
    for i, r in rows.items():
        by.setdefault(r["group"], []).append(i)
    pools = {"N19 (H19 humans, main)": Pool("N19", by["H19"], [i for g in GENS12 for i in by[f"N19.{g}"]], cluster),
             "NB (B humans, replication)": Pool("NB", by["B"], [i for g in GENS12 for i in by[f"NB.{g}"]], cluster)}
    for g in GENS12:
        pools[f"N19 {g}"] = Pool(g, by["H19"], by[f"N19.{g}"], cluster)
        pools[f"NB {g}"] = Pool(g, by["B"], by[f"NB.{g}"], cluster)
    return run_pools(pools, {"DG": rows}, G, n_boot, seed, main=("N19 (H19 humans, main)", "NB (B humans, replication)"))


def part_c(out: Path, input_path: Path, G: dict, n_boot: int, seed: int) -> dict:
    from det04 import read_meta  # noqa: E402
    meta = read_meta(str(input_path))
    fams = {"DG": read_jsonl(out / "s14_C.jsonl")}
    for fam, (fname, _, _) in OTHER.items():
        fams[fam] = read_jsonl(out / fname)
    have = set(fams["DG"]) & set(fams["TiC"]) & set(fams["Falcon"])
    cluster = {i: str(meta[i]["cluster"]) for i in have}
    by: dict[str, list[str]] = {}
    for i in sorted(have):
        by.setdefault(meta[i]["group"], []).append(i)
    strong = [i for g, ids in by.items() if g.startswith("APT.") and g != "APT.H" for i in ids
              if meta[i].get("level") in ("slight_major", "major")]
    pools = {"P1 RAID chat, sampling": Pool("P1", by["RAID.H"], [i for m in CHAT for i in by[f"RAID.{m}.sampling"]], cluster),
             "P2 APT strong": Pool("P2", by["APT.H"], strong, cluster)}
    for m in CHAT:
        pools[f"RAID {m}"] = Pool(m, by["RAID.H"], by[f"RAID.{m}.sampling"], cluster)
    for p in sorted({meta[i]["polisher"] for i in strong}):
        pools[f"APT strong {p}"] = Pool(p, by["APT.H"], [i for i in strong if meta[i]["polisher"] == p], cluster)
    return run_pools(pools, fams, G, n_boot, seed, main=("P1 RAID chat, sampling", "P2 APT strong"))


def run_pools(pools: dict[str, Pool], fams: dict, G: dict, n_boot: int, seed: int, main: tuple) -> dict:
    rng = np.random.default_rng(seed)
    res: dict = {"pools": {}}
    for name, pool in pools.items():
        res["pools"][name] = pool_tests(pool, fams, G, pool.draws(rng, n_boot))
    res["verdict"] = {det: {p: res["pools"][p][det]["D1"]["status"] for p in main} for det in dz.DETS}
    return res


def print_report(res: dict) -> None:
    for name, p in res["pools"].items():
        print(f"\n=== {name}: {p['n_h']} human, {p['n_ai']} AI, {p['n_clusters']} clusters ===")
        for det in dz.DETS:
            t = p[det]
            d1 = t["D1"]
            print(f"{det}: D1 y(2022) {d1['y2022']:+.3f} {dz.fmt(d1['y2022_ci'])}  y(2023) {d1['y2023']:+.3f} "
                  f"{dz.fmt(d1['y2023_ci'])}  f {d1['f']:.3f} {dz.fmt(d1['f_ci'])}  f_hat {d1['f_hat']:.3f}  -> {d1['status']}")
            a = t["auroc"]
            print(f"    AUROC 2022_2024 {a['2022_2024']:.4f}  2023_2024 {a['2023_2024']:.4f}  2024_2024 {a['2024_2024']:.4f}"
                  f"  f(AUROC) {d1['f_auroc']:.3f}")
            print("    D2 " + "  ".join(f"{r}: {v['ratio']:.3f} {dz.fmt(v['ci'])} ({v['status']})" for r, v in t["D2"].items())
                  + f"   D3 g {t['D3']['g']:.3f} {dz.fmt(t['D3']['ci'])} (g_hat {t['D3']['g_hat']:.3f})")
            if "D4" in t:
                print("    D4 " + "  ".join(f"{f}: x {v['x']:.3f} y {v['y']:+.3f} rho {v['rho']:.2f} {dz.fmt(v['rho_ci'])} ({v['status']})"
                                         for f, v in t["D4"].items()))
    print("\n=== verdict (D1 status on the main pools) ===")
    for det, v in res["verdict"].items():
        print(f"{det}: " + "; ".join(f"{p}: {s}" for p, s in v.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--part", choices=["B", "C"], required=True)
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--input", help="data/04_new.jsonl (part C: clusters, groups, polisher, level)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--json")
    args = ap.parse_args()
    out = Path(args.outputs)
    G = json.loads((out / "s14_eval.json").read_text(encoding="utf-8"))["dose"]  # part A dose, fixed
    res = part_b(out, G, args.n_boot, 0) if args.part == "B" else part_c(out, Path(args.input), G, args.n_boot, 0)
    res["dose"] = G
    print_report(res)
    if args.json:
        Path(args.json).write_text(json.dumps(dz.strip(res), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
