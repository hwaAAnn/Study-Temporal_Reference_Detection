"""Post-hoc decomposition for 09 (same split as 08's section 3): why Fast-DetectGPT and Binoculars lose to plain
likelihood on LLM-polished text.

Per cell r_s, ll = mean log s(x_t) and X = xppl = mean cross-entropy of the scorer s under the reference r's
next-token distribution (on the diagonal, the scorer's own entropy). Both detectors compare -ll with X; the more
negative (-ll) - X, the more AI-like. Prints the human and AI means and the AUROC of ll alone, -X alone, F and Bi
for the two main pools (P1 RAID chat sampling vs RAID.H, P2 APT strong vs APT.H).

usage: python scripts/slots_decomp.py outputs/s09_main.jsonl
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from det04 import CHAT  # noqa: E402

CELLS = ["2022_2022", "2024_2024", "2022_2024", "2021_2024"]


def pool(group: str):
    if group == "RAID.H":
        return "P1", 0
    if group == "APT.H":
        return "P2", 0
    p = group.split(".")
    if p[0] == "RAID" and p[1] in CHAT and p[-1] == "sampling":
        return "P1", 1
    if p[0] == "APT" and p[-1] in ("slight_major", "major"):
        return "P2", 1
    return None


def main() -> None:
    rows = {"P1": [], "P2": []}
    for line in open(sys.argv[1], encoding="utf-8"):
        r = json.loads(line)
        k = pool(r["group"])
        if k:
            rows[k[0]].append((k[1], r["cells"]))
    for name, rs in rows.items():
        y = np.array([lab for lab, _ in rs])
        h, a = y == 0, y == 1
        print(f"== {name}: {h.sum()} human, {a.sum()} AI")
        print(f"{'cell':10s} {'h ll':>7s} {'h X':>6s} {'h -ll-X':>8s} {'a ll':>7s} {'a X':>6s} {'a -ll-X':>8s}"
              f" | AUROC ll    -X     F     Bi")
        for c in CELLS:
            ll = np.array([d[c]["ll"] for _, d in rs])
            x = np.array([d[c]["xppl"] for _, d in rs])
            f = np.array([d[c]["fdgpt"] for _, d in rs])
            bi = -np.array([d[c]["bino"] for _, d in rs])
            d = -ll - x
            print(f"{c:10s} {ll[h].mean():7.3f} {x[h].mean():6.3f} {d[h].mean():+8.3f} {ll[a].mean():7.3f}"
                  f" {x[a].mean():6.3f} {d[a].mean():+8.3f} | {roc_auc_score(y, ll):.3f} {roc_auc_score(y, -x):.3f}"
                  f" {roc_auc_score(y, f):.3f} {roc_auc_score(y, bi):.3f}")


if __name__ == "__main__":
    main()
