"""Pipeline check for score_slots_family.py (experiments 10 and 11), on a few texts before the grid. It never
computes an AUROC.

(1) Direct vs chunked: each model in turn scores every check text alone (batch 1). Its logits are kept in fp32,
    with no fp16 storage and no chunk bookkeeping, and the --cross cells are computed from them. fdgpt and
    bino must agree with the chunked output within --tol-f and --tol-b.
(2) tic only: the single-model ll on the diagonal (cell c_c) against 07's score files
    (<--tic-dir>/tic1b_<c>.jsonl, the same tokenization): per-text mean log p within --tol-ll.
(3) hf only: the first model's mean token loss (-ll on its diagonal) over the human texts must lie in
    --loss-range. A wrong model class, dtype or tokenization gives a loss far outside it.

Example:
  python scripts/score_slots_family.py --family tic --models 202301 202407 --input data/01_all.jsonl \
      --groups H19 C19 H24 C24 B C --out outputs/s10_check.jsonl --limit 12
  python scripts/slots_check_family.py --family tic --models 202301 202407 --input data/01_all.jsonl \
      --groups H19 C19 H24 C24 B C --check outputs/s10_check.jsonl --tic-dir outputs
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from score_slots_family import DTYPES, load_items, make_family  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["tic", "hf"])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--weights-dir", default="models/tic")
    ap.add_argument("--dtype", default="float16", choices=DTYPES)
    ap.add_argument("--input", required=True)
    ap.add_argument("--groups", nargs="+", default=None)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--check", required=True, help="score_slots_family.py output for the same --limit texts")
    ap.add_argument("--cross", nargs="+", default=None, help="cells to recompute (default: every off-diagonal)")
    ap.add_argument("--tol-f", type=float, default=0.02)
    ap.add_argument("--tol-b", type=float, default=0.02)
    ap.add_argument("--tic-dir", default=None, help="tic: directory with 07's tic1b_<cutoff>.jsonl")
    ap.add_argument("--tol-ll", type=float, default=0.01)
    ap.add_argument("--loss-range", type=float, nargs=2, default=[2.0, 4.0])
    ap.add_argument("--out", default=None, help="write the numbers as JSON")
    args = ap.parse_args()

    import torch
    from score_slots import cell_terms

    rows = [json.loads(line) for line in Path(args.check).read_text(encoding="utf-8").splitlines() if line.strip()]
    fam = make_family(args)
    labels = fam.labels
    cross = args.cross or [f"{r}_{s}" for r in labels for s in labels if r != s]
    need = sorted({x for c in cross for x in c.split("_")}, key=labels.index)
    by_id = {it["id"]: it for it in load_items(args.input, args.groups, 0)}
    ids_all = fam.encode([by_id[r["id"]]["text"] for r in rows], args.max_tokens)
    report: dict = {"n_texts": len(rows), "cells": {}}
    ok = True

    # (1) direct, one model at a time, fp32 logits, batch 1
    lg32: dict[str, list] = {}
    for lab in need:
        model = fam.load(lab)
        with torch.no_grad():
            lg32[lab] = []
            for seq in ids_all:
                ids = torch.tensor(seq).unsqueeze(0).cuda()
                lg32[lab].append(fam.logits(model, ids, torch.ones_like(ids)).float().cpu())
        del model
        gc.collect()
        torch.cuda.empty_cache()
    with torch.no_grad():
        for cell in cross:
            r, s = cell.split("_")
            df, db = [], []
            for k, (row, seq) in enumerate(zip(rows, ids_all)):
                if len(seq) != row["n_tokens"]:
                    raise SystemExit(f"{row['id']}: {len(seq)} tokens here, {row['n_tokens']} in --check")
                ids = torch.tensor(seq).unsqueeze(0).cuda()
                got = cell_terms(lg32[r][k].cuda(), lg32[s][k].cuda(), ids, torch.ones_like(ids))
                df.append(abs(got["fdgpt"] - row["cells"][cell]["fdgpt"]))
                db.append(abs(got["bino"] - row["cells"][cell]["bino"]))
            report["cells"][cell] = {"max_diff_fdgpt": max(df), "max_diff_bino": max(db)}
            print(f"(1) cell {cell}, direct fp32 vs chunked: max |diff| fdgpt {max(df):.5f}, bino {max(db):.6f}")
            ok &= max(df) <= args.tol_f and max(db) <= args.tol_b

    # (2) tic: diagonal ll against 07
    if args.family == "tic" and args.tic_dir:
        worst = 0.0
        for c in labels:
            ref = {}
            for line in (Path(args.tic_dir) / f"tic1b_{c}.jsonl").read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                ref[r["id"]] = float(np.mean(r["lp"][c]))
            d = [abs(row["cells"][f"{c}_{c}"]["ll"] - ref[row["id"]]) for row in rows]
            worst = max(worst, max(d))
            print(f"(2) ll {c}_{c} vs 07 tic1b_{c}: max |diff| {max(d):.5f} over {len(d)} texts")
        report["max_diff_ll_vs_07"] = worst
        ok &= worst <= args.tol_ll

    # (3) hf: loss range of the first model on human texts
    if args.family == "hf":
        first = labels[0]
        hum = [-row["cells"][f"{first}_{first}"]["ll"] for row in rows if row["label"] == 0]
        if hum:
            loss = float(np.mean(hum))
            report["human_loss_first_model"] = loss
            lo, hi = args.loss_range
            print(f"(3) {first} mean token loss on {len(hum)} human texts: {loss:.3f} nats (range {lo}-{hi})")
            ok &= lo <= loss <= hi
        else:
            print("(3) no human text among the check texts; loss range not checked")

    report["pass"] = bool(ok)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
