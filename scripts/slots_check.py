"""Experiment 08 pipeline check, run on a few texts before the grid (no AUROC is computed).

(1) Single-model ll on the diagonal (cell v_v) against the 01 score files (outputs/yc_<v>.jsonl, the same
    tokenization): per-text mean log p must agree within --tol-ll.
(2) Two cross cells computed directly, both models loaded together at batch size 1 as in baseline_scores.py,
    against the chunked pipeline: fdgpt and bino must agree within --tol-det.

Example:
  python scripts/score_slots.py --input data/01_all.jsonl --out outputs/s08_check.jsonl --limit 12
  python scripts/slots_check.py --input data/01_all.jsonl --check outputs/s08_check.jsonl --yc-dir outputs
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

from score_slots import cell_terms, model_id  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--check", required=True, help="score_slots.py output for a few texts")
    ap.add_argument("--yc-dir", required=True, help="directory with the 01 files yc_<vintage>.jsonl")
    ap.add_argument("--cross", nargs="+", default=["2022_2024", "2024_2022"])
    ap.add_argument("--tol-ll", type=float, default=0.01)
    ap.add_argument("--tol-det", type=float, default=0.02)
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.check).read_text(encoding="utf-8").splitlines() if line.strip()]
    vint = sorted({k.split("_")[0] for k in rows[0]["cells"]})
    ok = True

    worst = 0.0
    for v in vint:
        yc = {}
        for line in (Path(args.yc_dir) / f"yc_{v}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            (lp,) = r["lp"].values()
            yc[r["id"]] = float(np.mean(lp))
        d = [abs(r["cells"][f"{v}_{v}"]["ll"] - yc[r["id"]]) for r in rows]
        worst = max(worst, max(d))
        print(f"(1) ll {v}_{v} vs yc_{v}: max |diff| {max(d):.5f} over {len(d)} texts")
    ok &= worst <= args.tol_ll

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from trd.data import normalise_rows

    items = {it["id"]: it for it in normalise_rows(
        [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()])}
    tok = AutoTokenizer.from_pretrained(model_id(vint[-1]))
    for cell in args.cross:
        r_v, s_v = cell.split("_")
        load = lambda n: AutoModelForCausalLM.from_pretrained(n, dtype=torch.float16).to("cuda").eval()  # noqa: E731
        mr, ms = load(model_id(r_v)), load(model_id(s_v))
        df, db = [], []
        with torch.no_grad():
            for row in rows:
                ids = torch.tensor(tok(items[row["id"]]["text"], add_special_tokens=True)["input_ids"][:200]).unsqueeze(0).cuda()
                mask = torch.ones_like(ids)
                got = cell_terms(mr(input_ids=ids, attention_mask=mask).logits, ms(input_ids=ids, attention_mask=mask).logits,
                                 ids, mask)
                df.append(abs(got["fdgpt"] - row["cells"][cell]["fdgpt"]))
                db.append(abs(got["bino"] - row["cells"][cell]["bino"]))
        del mr, ms
        torch.cuda.empty_cache()
        print(f"(2) cell {cell}, direct vs chunked: max |diff| fdgpt {max(df):.5f}, bino {max(db):.6f}")
        ok &= max(df) <= args.tol_det and max(db) <= args.tol_det
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
