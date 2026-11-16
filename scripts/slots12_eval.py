"""Experiment 12 analysis: the pre-registered contrasts H1-H6 and the verdict (docs/experiments/12-recent-generators.md).

Sets: H19 vs N19.* pooled over generators (main), B vs NB.* pooled (replication), and each generator alone
(described). Scores, higher = more AI: DatedGPT F:<cell>, Bi:<cell> (cells r_s over 2013..2024), ll:<v>,
s09 = ll:2024 - 0.9 ll:2022; Falcon fa:F:<cell>, fa:Bi:<cell> (base/inst). Bootstrap: 2,000 draws over pair_id
(a pooled set keeps each human abstract with its generations), seed 0, the same draws for every score.
  H1 A(2022_2024) - A(2022_2022)                          CI lower > 0
  H2 A(2022_2024) - A(2024_2024)                          CI lower > 0 and E(2024_2024)/E(2022_2024) >= 1.3;
                                                          both cells >= 0.998 -> ceiling, skipped
  H3 A(2023_2024) - A(2022_2024)                          CI upper < 0
  H4 A(2024_2024) - A(2022_2022)                          CI upper < 0
  H5 A_qwen3t(2022_2024) - A_qwen3i(2022_2024), H19 only  CI upper < 0 (paired over the same human abstracts)
  H6 A(Bi:2022_2024) - A(fa:Bi:base_inst)                 CI lower > 0 (F described)

Example:
  python scripts/slots12_eval.py --dg outputs/s12_dg.jsonl --fa outputs/s12_fa.jsonl \
      --json outputs/s12_eval.json > outputs/s12_eval.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from slots_family_eval import Comp, read_jsonl, status  # noqa: E402

DETS = ["F", "Bi"]
TAGS = ["gptoss120b", "glm47flash", "qwen3i", "qwen3t"]
CEIL, RATIO = 0.998, 1.3
CONTRASTS = {
    "H1": ([(1, "{d}:2022_2024"), (-1, "{d}:2022_2022")], "lower>0"),
    "H2": ([(1, "{d}:2022_2024"), (-1, "{d}:2024_2024")], "lower>0"),
    "H3": ([(1, "{d}:2023_2024"), (-1, "{d}:2022_2024")], "upper<0"),
    "H4": ([(1, "{d}:2024_2024"), (-1, "{d}:2022_2022")], "upper<0"),
    "H6": ([(1, "{d}:2022_2024"), (-1, "fa:{d}:base_inst")], "lower>0"),
}
JUDGED = {"H1": DETS, "H2": DETS, "H3": DETS, "H4": DETS, "H6": ["Bi"]}
EXTRA = {
    "vs_s09": [(1, "{d}:2022_2024"), (-1, "s09")],
    "s09_vs_falcon": [(1, "s09"), (-1, "fa:{d}:base_inst")],
    "falcon_H2": [(1, "fa:{d}:base_inst"), (-1, "fa:{d}:inst_inst")],
}
SHOW = ["2022_2022", "2024_2024", "2022_2024", "2023_2024", "2024_2022"]


def load(dg_path: str, fa_path: str | None) -> tuple[dict, dict]:
    sc, head = {}, {}
    for r in read_jsonl(dg_path):
        s = {}
        for cell, c in r["cells"].items():
            s[f"F:{cell}"], s[f"Bi:{cell}"] = float(c["fdgpt"]), -float(c["bino"])
            a, b = cell.split("_")
            if a == b:
                s[f"ll:{a}"] = float(c["ll"])
        s["s09"] = s["ll:2024"] - 0.9 * s["ll:2022"]
        sc[r["id"]] = s
        head[r["id"]] = {k: r[k] for k in ("group", "label", "pair_id", "generator")}
    if fa_path:
        for r in read_jsonl(fa_path):
            if r["id"] not in sc:
                raise SystemExit(f"{r['id']} in the Falcon grid but not in the DatedGPT grid")
            for cell, c in r["cells"].items():
                sc[r["id"]][f"fa:F:{cell}"], sc[r["id"]][f"fa:Bi:{cell}"] = float(c["fdgpt"]), -float(c["bino"])
    return sc, head


def sets(head: dict) -> dict[str, tuple[str, list[str], list[str]]]:
    by: dict[str, list[str]] = {}
    for i, h in head.items():
        by.setdefault(h["group"], []).append(i)
    out = {}
    for hg, prefix, role in (("H19", "N19.", "main"), ("B", "NB.", "replication")):
        ai = [i for g in sorted(by) if g.startswith(prefix) for i in by[g]]
        if by.get(hg) and ai:
            out[f"{hg} pooled"] = (role, by[hg], ai)
        for t in TAGS:
            if by.get(prefix + t):
                out[f"{hg} {t}"] = ("generator", by[hg], by[prefix + t])
    return out


def combo(a: dict, terms, d: str) -> float | None:
    ks = [k.format(d=d) for _, k in terms]
    return float(sum(s * a[k] for (s, _), k in zip(terms, ks))) if all(k in a for k in ks) else None


def evaluate(sc: dict, head: dict, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    cluster = {i: h["pair_id"] for i, h in head.items()}
    res = {"sets": {}}
    for name, (role, h_ids, a_ids) in sets(head).items():
        c = Comp(name, "A", h_ids, a_ids, sc, cluster)
        point = c.aurocs()
        boots = [c.aurocs(c.draw(rng)) for _ in range(n_boot)]
        out = {"role": role, "n_h": c.n_h, "n_ai": c.n_a, "auroc": point, "tpr5": c.tprs(), "contrasts": {}}
        for d in DETS:
            con = {}
            for n, (terms, rule) in list(CONTRASTS.items()) + [(k, (t, None)) for k, t in EXTRA.items()]:
                v = combo(point, terms, d)
                if v is None:
                    continue
                lo, hi = np.percentile([combo(b, terms, d) for b in boots], [2.5, 97.5]).tolist()
                keys = [k.format(d=d) for _, k in terms]
                judged = rule is not None and d in JUDGED.get(n, [])
                e = {"terms": [[s, k] for (s, _), k in zip(terms, keys)], "a": [point[k] for k in keys], "diff": v,
                     "ci": [lo, hi], "status": status(rule, lo, hi) if judged else "-"}
                if n == "H2" and judged:
                    if min(e["a"]) >= CEIL:
                        e["status"] = "ceiling"
                    else:
                        e_new, e_key = 1 - point[keys[1]], 1 - point[keys[0]]
                        e["error_ratio"] = float(e_new / e_key) if e_key > 0 else float("inf")
                        if e["status"] == "holds" and e["error_ratio"] < RATIO:
                            e["status"] = "fails (error ratio)"
                con[n] = e
            out["contrasts"][d] = con
        res["sets"][name] = out
    res["H5"] = h5(sc, head, n_boot, seed)
    res["verdict"] = judge(res)
    return res


def h5(sc: dict, head: dict, n_boot: int, seed: int) -> dict:
    """A_qwen3t - A_qwen3i on H19, bootstrap over the human abstracts' pair_ids (both AI sets drawn together)."""
    from slots_grid import auroc
    hum = {head[i]["pair_id"]: i for i in head if head[i]["group"] == "H19"}
    gen = {t: {head[i]["pair_id"]: i for i in head if head[i]["group"] == f"N19.{t}"} for t in ("qwen3t", "qwen3i")}
    if not all(gen.values()):
        return {"status": "missing"}
    pids = sorted(hum)
    rng = np.random.default_rng(seed + 5)
    out = {}
    for d in DETS:
        k = f"{d}:2022_2024"

        def diff(sample):
            h = np.array([sc[hum[p]][k] for p in sample])
            a = {t: np.array([sc[gen[t][p]][k] for p in sample if p in gen[t]]) for t in gen}
            return auroc(a["qwen3t"], h) - auroc(a["qwen3i"], h), auroc(a["qwen3t"], h), auroc(a["qwen3i"], h)

        v, at, ai = diff(pids)
        bs = [diff([pids[j] for j in rng.integers(0, len(pids), len(pids))])[0] for _ in range(n_boot)]
        lo, hi = np.percentile(bs, [2.5, 97.5]).tolist()
        out[d] = {"a_qwen3t": at, "a_qwen3i": ai, "diff": v, "ci": [lo, hi], "status": status("upper<0", lo, hi)}
    return out


def judge(res: dict) -> dict:
    v = {"per_set": {s: {d: {h: c["status"] for h, c in res["sets"][s]["contrasts"][d].items() if c["status"] != "-"}
                         for d in DETS} for s in res["sets"]}}
    deciding = ["H19 pooled", "B pooled"]
    h2 = {}
    for d in DETS:
        x = [v["per_set"].get(s, {}).get(d, {}).get("H2", "missing") for s in deciding]
        x = [y for y in x if y != "ceiling"]
        h2[d] = "undecidable" if not x else ("holds" if all(y == "holds" for y in x) else "fails")
    n = sum(h2[d] == "holds" for d in DETS)
    v["H2"] = h2
    v["verdict"] = ("undecidable" if all(h2[d] == "undecidable" for d in DETS)
                    else "extends to newer generators" if n == 2 else "partial" if n == 1 else "fails")
    v["H4_all"] = all(v["per_set"].get(s, {}).get(d, {}).get("H4") == "holds" for s in deciding for d in DETS)
    v["H5"] = {d: res["H5"].get(d, {}).get("status", "missing") for d in DETS}
    v["H6"] = {s: v["per_set"].get(s, {}).get("Bi", {}).get("H6", "missing") for s in deciding}
    return v


def print_report(res: dict) -> None:
    for name, s in res["sets"].items():
        a = s["auroc"]
        print(f"\n=== {name} ({s['role']}): {s['n_h']} human, {s['n_ai']} AI ===")
        print("cell          " + "".join(f"{d:>9}" for d in DETS) + "   TPR@5%: " + " ".join(DETS))
        for cell in SHOW:
            print(f"{cell:<14}" + "".join(f"{a[f'{d}:{cell}']:>9.4f}" for d in DETS) + "   "
                  + " ".join(f"{s['tpr5'][f'{d}:{cell}']:.2f}" for d in DETS))
        fa = [k for k in sorted(a) if k.startswith("fa:")]
        print("Falcon: " + ", ".join(f"{k[3:]} {a[k]:.4f}" for k in fa))
        print("single-model ll: " + ", ".join(f"{k[3:]} {a[k]:.4f}" for k in sorted(a) if k.startswith("ll:"))
              + f"; s09 {a['s09']:.4f}")
        for d in DETS:
            for n, c in s["contrasts"][d].items():
                t = " ".join(f"{'+' if sg > 0 else '-'}{k}" for sg, k in c["terms"])
                er = f"  E ratio {c['error_ratio']:.2f}" if "error_ratio" in c else ""
                print(f"  {d:<3}{n:<14}{t}: {c['diff']:+.4f} [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]{er}  {c['status']}")
    print("\n=== H5 (H19, qwen3t - qwen3i at 2022_2024) ===")
    for d, c in res["H5"].items():
        if isinstance(c, dict):
            print(f"  {d}: {c['a_qwen3t']:.4f} - {c['a_qwen3i']:.4f} = {c['diff']:+.4f} "
                  f"[{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]  {c['status']}")
    print("\n=== verdict ===")
    print(json.dumps(res["verdict"], indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dg", required=True)
    ap.add_argument("--fa", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    sc, head = load(args.dg, args.fa)
    res = evaluate(sc, head, args.n_boot, args.seed)
    print_report(res)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
