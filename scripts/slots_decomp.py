"""Post-hoc decomposition for 09 (same split as 08's section 3), the extra predictions E1-E3 of the 09 doc, and the
slot contrasts on every 04 pool.

Per cell r_s, ll = mean log s(x_t) and X = xppl = mean cross-entropy of the scorer s under the reference r's
next-token distribution (on the diagonal, the scorer's own entropy). Both detectors compare -ll with X; the more
negative (-ll) - X, the more AI-like.

Part 1 (no CI): human and AI means of ll, X and (-ll) - X, and the AUROC of ll alone, -X alone, F and Bi, for
  P1 (RAID chat sampling), P2 (APT strong), APT weak and RAID chat greedy.
Part 2 (det09's pools and source-cluster bootstrap, 2,000 draws, seed 0), for F and Bi on every pool:
  single-model cell minus the same model's likelihood (r_r - ll r, r = 2022, 2024), 2022_2024 - ll 2024,
  and the slot contrasts H1 (2022_2024 - 2022_2022), H2 (2022_2024 - 2024_2024), H4 (2024_2024 - 2022_2022).
  E1 = "2022_2024 - ll 2024" on APT weak (CI upper < 0 for F and Bi), E2 = "r_r - ll r" on APT weak (all four
  below -0.03), E3 = F "2022_2022 - ll 2022" on RAID chat greedy (> 0). Only H1-H5 on P1/P2 are judged (det09).

usage: python scripts/slots_decomp.py --input data/04_new.jsonl --grid outputs/s09_main.jsonl outputs/s09_rest.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import det09  # noqa: E402
from det04 import auroc, read_meta  # noqa: E402

WEAK, GREEDY = "APT weak (extreme_minor+minor)", "RAID chat, greedy"
DECOMP_POOLS = [det09.P1, det09.P2, WEAK, GREEDY]
DECOMP_CELLS = ["2022_2022", "2024_2024", "2022_2024", "2021_2024"]
BOOT_CELLS = ["2022_2022", "2024_2024", "2022_2024"]
CONTRASTS = {  # name -> [(sign, key)], {d} = F or Bi
    "22_22-ll22": [(1, "{d}:2022_2022"), (-1, "ll:2022_2022")],
    "24_24-ll24": [(1, "{d}:2024_2024"), (-1, "ll:2024_2024")],
    "22_24-ll24": [(1, "{d}:2022_2024"), (-1, "ll:2024_2024")],
    "H1": [(1, "{d}:2022_2024"), (-1, "{d}:2022_2022")],
    "H2": [(1, "{d}:2022_2024"), (-1, "{d}:2024_2024")],
    "H4": [(1, "{d}:2024_2024"), (-1, "{d}:2022_2022")],
}


def load(paths: list[str], cells: list[str] | None) -> dict[str, dict[str, float]]:
    sc = {}
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            s = {}
            for c, v in r["cells"].items():
                if cells is None or c in cells:
                    s.update({f"F:{c}": v["fdgpt"], f"Bi:{c}": -v["bino"], f"ll:{c}": v["ll"], f"X:{c}": v["xppl"]})
            sc[r["id"]] = s
    return sc


def part1(meta, sc) -> None:
    _, pools = det09.comparisons(meta, sc)
    for name in DECOMP_POOLS:
        c = det09.Comp(name, *pools[name], sc, meta)
        y = c.y
        au = lambda v: auroc(v[~y], v[y])  # noqa: E731
        print(f"== {name}: {c.n_h} human, {c.n_a} AI")
        print(f"{'cell':10s} {'h ll':>7s} {'h X':>6s} {'h -ll-X':>8s} {'a ll':>7s} {'a X':>6s} {'a -ll-X':>8s}"
              f" | AUROC ll    -X     F     Bi")
        for cell in DECOMP_CELLS:
            if f"ll:{cell}" not in c.S:
                continue
            ll, x = c.S[f"ll:{cell}"], c.S[f"X:{cell}"]
            d = -ll - x
            print(f"{cell:10s} {ll[~y].mean():7.3f} {x[~y].mean():6.3f} {d[~y].mean():+8.3f} {ll[y].mean():7.3f}"
                  f" {x[y].mean():6.3f} {d[y].mean():+8.3f} | {au(ll):.3f} {au(-x):.3f}"
                  f" {au(c.S[f'F:{cell}']):.3f} {au(c.S[f'Bi:{cell}']):.3f}")
        print()


def part2(meta, sc, n_boot: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    _, pools = det09.comparisons(meta, sc)
    print(f"== contrasts, AUROC difference [95% CI], source-cluster bootstrap {n_boot}, seed {seed}")
    for name, (h, a) in pools.items():
        if not h or not a:
            continue
        c = det09.Comp(name, h, a, sc, meta)
        point = c.aurocs()
        boots = [c.aurocs(c.draw(rng)) for _ in range(n_boot)]
        for d in det09.DETS:
            parts = []
            for n, terms in CONTRASTS.items():
                keys = [k.format(d=d) for _, k in terms]
                if not all(k in point for k in keys):
                    continue
                f = lambda a_: sum(s * a_[k] for (s, _), k in zip(terms, keys))  # noqa: E731
                lo, hi = np.percentile([f(b) for b in boots], [2.5, 97.5])
                parts.append(f"{n} {f(point):+.3f} [{lo:+.3f}, {hi:+.3f}]")
            print(f"{name} | {d}: " + "; ".join(parts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="data/04_new.jsonl")
    ap.add_argument("--grid", nargs="+", required=True, help="score_slots.py outputs (s09_main, s09_rest)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    meta = read_meta(args.input)
    part1(meta, load(args.grid, DECOMP_CELLS))
    part2(meta, load(args.grid, BOOT_CELLS), args.n_boot, args.seed)


if __name__ == "__main__":
    main()
