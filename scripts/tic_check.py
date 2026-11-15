"""TiC-LM checkpoint handling for 07: convert to fp16 safetensors, CPU fp32 reference, fp16 check.

  convert  --pt X.pt --out X.safetensors          strict key check, fp16 copy, SHA-256 of the .pt
  ref      --pt X.pt --input data/01_all.jsonl --out ref.json
           CPU fp32 log-probs of the check texts (first B, C and D' text in input order), one layer in
           memory at a time (weights stay memory-mapped); also the largest residual-stream magnitude
  compare  --ref ref.json --scores outputs/tic1b_X.jsonl
           pre-registered fp16 rule: |mean log p (scored) - mean log p (fp32 ref)| < 0.005 for every check
           text and all values finite; exit status 1 if the rule fails

Run with python -I (the .pt files are downloaded data); the repo's src/ is put on sys.path here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from trd.data import normalise_rows  # noqa: E402
from trd import tic  # noqa: E402

CHECK_GROUPS = ("B", "C", "D")
TOL = 0.005


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def check_items(input_path: str) -> list[dict]:
    rows = [json.loads(l) for l in Path(input_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    items, picked = normalise_rows(rows), {}
    for it in items:
        if it["group"] in CHECK_GROUPS and it["group"] not in picked:
            picked[it["group"]] = it
    return [picked[g] for g in CHECK_GROUPS]


def reference(pt: str, items: list[dict], max_tokens: int, threads: int) -> dict:
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(threads)
    olm = tic.import_open_lm()
    sd, info = tic.load_pt(pt)
    size = "3b" if sd["tok_embeddings.weight"].shape[1] == tic.SIZES["3b"][1] else "1b"
    _, dim, n_layers, n_heads = tic.SIZES[size]
    p = tic.params(size)
    enc = tic.encode([it["text"] for it in items], max_tokens)
    emb = sd["tok_embeddings.weight"]
    xs = [emb[torch.tensor(ids)].float().unsqueeze(0) for ids in enc]
    head_dim = dim // n_heads
    formula = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    peak = 0.0
    with torch.no_grad():
        for i in range(n_layers):
            with torch.device("meta"):
                blk = olm.Block(i, p)
            pre = f"layers.{i}."
            blk.load_state_dict({k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}, strict=True, assign=True)
            inv = sd[f"{pre}attention.pos_embed.inv_freq"].float()  # trained value: the formula rounded to bf16
            if not torch.allclose(inv, formula, rtol=1e-2, atol=0):
                raise SystemExit(f"{pt}: layer {i} rotary inv_freq is not the open_lm formula")
            cos, sin = tic.rotary_tables(inv, tic.SEQ_LEN, torch.float32, "cpu")
            pe = blk.attention.pos_embed
            pe._cos_cached, pe._sin_cached, pe._seq_len_cached = cos, sin, tic.SEQ_LEN
            pe._update_cos_sin_tables = lambda *a, **k: None
            blk.eval()
            for j in range(len(xs)):
                xs[j] = blk(xs[j])[0]
                peak = max(peak, float(xs[j].abs().max()))
            del blk
        out = []
        for it, ids, x in zip(items, enc, xs):
            h = F.layer_norm(x[0], (dim,), sd["norm.weight"].float(), None, 1e-5)
            logits = h @ sd["output.weight"].float().T
            lp = torch.log_softmax(logits[:-1], dim=-1).gather(-1, torch.tensor(ids[1:]).unsqueeze(-1)).squeeze(-1)
            out.append({"id": it["id"], "group": it["group"], "ids": ids[1:], "lp": lp.double().numpy().round(6).tolist()})
    return {"pt": Path(pt).name, "size": size, "info": {k: str(v) for k, v in info.items()},
            "max_residual_abs": peak, "fp16_max": 65504.0, "texts": out}


def compare(ref: dict, scores_path: str) -> dict:
    want = {t["id"]: t for t in ref["texts"]}
    got = {}
    for line in Path(scores_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r["id"] in want:
                got[r["id"]] = r
    rows, ok = [], True
    for i, t in want.items():
        if i not in got:
            raise SystemExit(f"check text {i} missing from {scores_path}")
        r = got[i]
        (key, lp), = r["lp"].items()
        n = min(len(lp), len(t["lp"]))
        if r["ids"][:n] != t["ids"][:n]:
            raise SystemExit(f"{i}: token ids differ between the reference and the scores")
        a, b = np.asarray(lp[:n]), np.asarray(t["lp"][:n])
        d = float(a.mean() - b.mean())
        fin = bool(np.isfinite(a).all())
        ok &= fin and abs(d) < TOL
        rows.append({"id": i, "group": t["group"], "n": n, "text_mean_diff": d,
                     "token_absdiff_max": float(np.abs(a - b).max()), "finite": fin})
    return {"scores": Path(scores_path).name, "rule": f"|text mean diff| < {TOL}, finite", "pass": bool(ok), "texts": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--pt", required=True)
    c.add_argument("--out", required=True)
    r = sub.add_parser("ref")
    r.add_argument("--pt", required=True)
    r.add_argument("--input", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--max-tokens", type=int, default=200)
    r.add_argument("--threads", type=int, default=4)
    m = sub.add_parser("compare")
    m.add_argument("--ref", required=True)
    m.add_argument("--scores", required=True)
    m.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.cmd == "convert":
        digest = sha256(args.pt)
        meta = tic.convert(args.pt, args.out)
        print(json.dumps({"pt": Path(args.pt).name, "sha256": digest, **meta}))
    elif args.cmd == "ref":
        res = reference(args.pt, check_items(args.input), args.max_tokens, args.threads)
        Path(args.out).write_text(json.dumps(res), encoding="utf-8")
        print(f"reference for {len(res['texts'])} texts, max |residual| {res['max_residual_abs']:.1f}")
    else:
        res = compare(json.loads(Path(args.ref).read_text(encoding="utf-8")), args.scores)
        if args.out:
            Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
        for t in res["texts"]:
            print(f"  {t['group']:<3}{t['id']:<40} n {t['n']:>4}  mean diff {t['text_mean_diff']:+.5f}  "
                  f"max token diff {t['token_absdiff_max']:.4f}  finite {t['finite']}")
        print("fp16 rule:", "pass" if res["pass"] else "FAIL")
        sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
