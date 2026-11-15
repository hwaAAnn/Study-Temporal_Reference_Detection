"""TiC-LM Oracle models (Li et al., ACL 2025): open_lm checkpoints trained from scratch at seven cutoffs.

They are not Hugging Face models. The architecture comes from the vendored open_lm model code
(third_party/open_lm, commit OPENLM_COMMIT, MIT). The released checkpoints are pickled .pt files; they are
opened only with torch.load(weights_only=True) and converted once to fp16 safetensors for scoring.

Scoring matches trd.scoring.token_logprobs: per-text arrays of log p(x_t | x_<t) for t >= 1 plus the token
ids, where id 0 (GPT-NeoX <|endoftext|>, the document separator in the training data) is prepended as BOS
and --max-tokens counts it.
"""

from __future__ import annotations

import gc
import sys
import types
from pathlib import Path
from typing import Iterable

import numpy as np

OPENLM_COMMIT = "9bb92ef1689333534b7057942a20d18a46d1fa52"
CUTOFFS = ["201305", "201501", "201701", "201901", "202101", "202301", "202407"]
PRE = CUTOFFS[:6]   # cutoffs before ChatGPT (2023-01 may hold up to two months of crawl after it)
POST = CUTOFFS[6:]
# name, hidden dim, layers, heads (third_party/open_lm/open_lm/model_configs/*.json)
SIZES = {"1b": ("open_lm_1b", 2048, 24, 16), "3b": ("open_lm_3b", 2560, 32, 32)}
URL = "https://ml-site.cdn-apple.com/models/tic-lm/oracles_{size}_220b/{cutoff}/checkpoints/epoch_{epoch}.pt"
TOKENIZER = "EleutherAI/gpt-neox-20b"
EOT = 0
VOCAB = 50432
SEQ_LEN = 2048
_OPENLM_DIR = Path(__file__).resolve().parents[2] / "third_party" / "open_lm"


def checkpoint_url(size: str, cutoff: str) -> str:
    if size not in SIZES or cutoff not in CUTOFFS:
        raise ValueError(f"unknown size/cutoff {size}/{cutoff}")
    return URL.format(size=size, cutoff=cutoff, epoch=12 if cutoff == "201305" else 21)


def cutoff_time(cutoff: str) -> float:
    """Decimal year at the end of the cutoff month: '201305' -> 2013 + 5/12."""
    return int(cutoff[:4]) + int(cutoff[4:]) / 12


def import_open_lm():
    """Import the vendored open_lm model module. xformers is imported there but unused by this config."""
    if "xformers" not in sys.modules:
        xf, xops = types.ModuleType("xformers"), types.ModuleType("xformers.ops")
        xf.ops = xops
        sys.modules["xformers"], sys.modules["xformers.ops"] = xf, xops
    if str(_OPENLM_DIR) not in sys.path:
        sys.path.insert(0, str(_OPENLM_DIR))
    import open_lm.model as olm
    return olm


def params(size: str):
    olm = import_open_lm()
    from open_lm.attention import torch_attn
    from open_lm.norms import get_norm_class

    _, dim, n_layers, n_heads = SIZES[size]
    return olm.Params(dim=dim, n_layers=n_layers, n_heads=n_heads, seq_len=SEQ_LEN, vocab_size=VOCAB,
                      post_embed_norm=False, weight_tying=False, norm_type=get_norm_class("gain_only_lp_layer_norm"),
                      attn_func=torch_attn, apply_qk_norm=True, positional_embedding_type="rotary",
                      ffn_type="swiglu_torch")


def build_meta(size: str):
    """The open_lm Transformer on the meta device (no memory); load weights with assign=True."""
    import torch

    olm = import_open_lm()
    with torch.device("meta"):
        return olm.Transformer(params(size))


def clean_state_dict(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        for pre in ("module.", "_orig_mod."):
            k = k.removeprefix(pre)
        out[k] = v
    return out


def load_pt(path: str) -> tuple[dict, dict]:
    """State dict (memory-mapped, no code execution) and the checkpoint's bookkeeping fields."""
    import torch

    ck = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    info = {k: ck[k] for k in ("epoch", "step", "samples_seen", "name") if isinstance(ck, dict) and k in ck}
    return clean_state_dict(sd), info


def rotary_tables(inv_freq, seq_len: int, dtype, device):
    """cos/sin tables computed in fp32 and then cast (open_lm builds them in the activation dtype)."""
    import torch

    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    fr = t[:, None] * inv_freq.float().to(device)[None, :]
    emb = torch.cat((fr, fr), dim=-1)
    return emb.cos()[None, :, None, :].to(dtype), emb.sin()[None, :, None, :].to(dtype)


def fix_rotary(model, inv_freqs: list, dtype, device) -> None:
    """Install fp32-computed tables; inv_freqs are the fp32 buffers captured before any dtype cast."""
    for layer, inv in zip(model.layers, inv_freqs):
        pe = layer.attention.pos_embed
        pe._cos_cached, pe._sin_cached = rotary_tables(inv, SEQ_LEN, dtype, device)
        pe._seq_len_cached = SEQ_LEN
        pe._update_cos_sin_tables = lambda *a, **k: None  # tables cover every position we score


def load_model(weights: str, size: str = "1b", dtype: str = "float16", device: str | None = None):
    """open_lm model from an fp16 safetensors copy (or a .pt checkpoint), strict key check."""
    import torch
    from safetensors.torch import load_file

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tdt = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    sd = load_file(weights) if str(weights).endswith(".safetensors") else load_pt(weights)[0]
    model = build_meta(size)
    model.load_state_dict(sd, strict=True, assign=True)
    invs = stored_inv_freqs(model, size, str(weights))
    model = model.to(device=device, dtype=tdt).eval()
    fix_rotary(model, invs, tdt, device)
    return model


def stored_inv_freqs(model, size: str, where: str = "") -> list:
    """The rotary inv_freq buffers as trained, in fp32.

    The released checkpoints store the open_lm formula 1 / 10000^(2i/d) rounded to bf16 (up to 0.37% off,
    checked on the 2023-01 1B model, 11/15): that is what the model was trained with, so it is used as is.
    bf16 values are exact in fp16, so the fp16 safetensors copy keeps them. A buffer more than 1% off the
    formula means a broken file.
    """
    import torch

    head_dim = SIZES[size][1] // SIZES[size][3]
    formula = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    out = []
    for i, layer in enumerate(model.layers):
        inv = layer.attention.pos_embed.inv_freq.detach().float().cpu().clone()
        if inv.shape != formula.shape or not torch.allclose(inv, formula, rtol=1e-2, atol=0):
            raise SystemExit(f"{where}: layer {i} rotary inv_freq is not the open_lm formula (bf16-rounded)")
        out.append(inv)
    return out


def convert(pt_path: str, out_path: str, dtype: str = "float16") -> dict:
    """Write an fp16 safetensors copy of a .pt checkpoint after a strict key check against the model."""
    import torch
    from safetensors.torch import save_file

    tdt = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    sd, info = load_pt(pt_path)
    size = "3b" if sd["tok_embeddings.weight"].shape[1] == SIZES["3b"][1] else "1b"
    want = set(build_meta(size).state_dict())
    if set(sd) != want:
        raise SystemExit(f"{pt_path}: keys differ from open_lm {size}: missing {sorted(want - set(sd))[:5]}, "
                         f"unexpected {sorted(set(sd) - want)[:5]}")
    half = {k: v.to(tdt).contiguous() for k, v in sd.items()}
    meta = {"source": Path(pt_path).name, "size": size, "openlm_commit": OPENLM_COMMIT,
            **{k: str(v) for k, v in info.items()}}
    save_file(half, out_path, metadata=meta)
    return meta


def encode(texts: Iterable[str], max_tokens: int = 200) -> list[list[int]]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    return [[EOT] + tok(t, add_special_tokens=False)["input_ids"][: max_tokens - 1] for t in texts]


def token_logprobs(texts: Iterable[str], weights: str, size: str = "1b", max_tokens: int = 200,
                   dtype: str = "float16", device: str | None = None, batch_size: int = 8,
                   desc: str = "tic") -> tuple[list[np.ndarray], list[list[int]]]:
    """Per-text log p(x_t | x_<t) for t >= 1 and the token ids used (EOT first).

    Batches are right-padded. With causal attention a real token never attends to a later pad, and rotary
    positions start at 0 for every row, so no attention mask is needed and the result equals unbatched scoring.
    """
    import torch
    from tqdm import tqdm

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ids_all = encode(texts, max_tokens)
    model = load_model(weights, size, dtype, device)
    out: list[np.ndarray | None] = [None] * len(ids_all)
    order = sorted(range(len(ids_all)), key=lambda i: len(ids_all[i]))
    with torch.no_grad():
        for b in tqdm(range(0, len(order), batch_size), desc=desc, leave=False):
            idx = order[b:b + batch_size]
            chunk = [ids_all[i] for i in idx]
            L = max(len(c) for c in chunk)
            inp = torch.full((len(chunk), L), EOT, dtype=torch.long)
            for j, c in enumerate(chunk):
                inp[j, :len(c)] = torch.tensor(c)
            logits = model(input_ids=inp.to(device))[0]
            real = torch.zeros((len(chunk), L), dtype=torch.bool)
            for j, c in enumerate(chunk):
                real[j, :len(c)] = True
            if not torch.isfinite(logits[real.to(device)]).all():
                raise FloatingPointError(f"non-finite logits from {weights} in {dtype}")
            lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
            got = lp.gather(-1, inp[:, 1:].to(device).unsqueeze(-1)).squeeze(-1).cpu().numpy()
            for j, (i, c) in enumerate(zip(idx, chunk)):
                out[i] = got[j, :len(c) - 1].astype(np.float64)
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return out, ids_all
