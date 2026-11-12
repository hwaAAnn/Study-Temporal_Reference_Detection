"""Is Delta more than 'the newer model rewards predictable text'?

AI text is more predictable than human text, so if the 2024 model were simply sharper on predictable
text, Delta would rise with ll_old and AI text would get a higher Delta for that reason alone. Here we
fit Delta = a + b * ll_old on the human texts of each group, then score every text by its residual.
If the AUROC of the residual stays high, AI text has a higher Delta than human text *of the same
predictability*.

Example:
  python scripts/diag_residual.py --scores outputs/pilot_k0.jsonl --by generator
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from trd.metrics import auroc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--by", nargs="*", default=["generator"])
    args = ap.parse_args()

    recs = [json.loads(l) for l in Path(args.scores).read_text(encoding="utf-8").splitlines() if l.strip()]
    groups = defaultdict(list)
    for r in recs:
        groups[" / ".join(str(r.get(f, "")) for f in args.by)].append(r)
    groups["(pooled)"] = recs

    print(f"{'group':<36}{'slope_h':>8}{'r_h':>7}{'AUROC d':>9}{'AUROC resid':>12}{'AUROC ll_old':>13}")
    for g, rs in groups.items():
        y = np.array([r["label"] for r in rs])
        d = np.array([np.mean(np.array(r["lp_new"]) - np.array(r["lp_old"])) for r in rs])
        lo = np.array([np.mean(r["lp_old"]) for r in rs])
        h = y == 0
        b, a = np.polyfit(lo[h], d[h], 1)
        resid = d - (a + b * lo)
        r_h = np.corrcoef(lo[h], d[h])[0, 1]
        print(f"{g:<36}{b:>+8.3f}{r_h:>+7.2f}{auroc(y, d):>9.3f}{auroc(y, resid):>12.3f}{auroc(y, lo):>13.3f}")


if __name__ == "__main__":
    main()
