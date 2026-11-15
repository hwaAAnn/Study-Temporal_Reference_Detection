"""Model-free checks for the TiC-LM Oracle loader and the 07 curve statistics (no checkpoint needed)."""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from trd import tic  # noqa: E402


def test_urls_and_times():
    assert tic.checkpoint_url("1b", "201305").endswith("oracles_1b_220b/201305/checkpoints/epoch_12.pt")
    assert tic.checkpoint_url("3b", "202407").endswith("oracles_3b_220b/202407/checkpoints/epoch_21.pt")
    assert tic.cutoff_time("201305") == pytest.approx(2013 + 5 / 12)
    assert tic.PRE[-1] == "202301" and tic.POST == ["202407"]


def test_meta_model_keys_match_checkpoint_layout():
    sd = tic.build_meta("1b").state_dict()
    assert len(sd) == 219  # 1 embedding + 24 layers x 9 + final norm + output (as in the 2023-01 checkpoint)
    assert sd["layers.0.attention.in_proj.weight"].shape == (3 * 2048, 2048)
    assert sd["layers.0.feed_forward.w12.weight"].shape == (2 * 5632, 2048)
    assert sd["layers.23.feed_forward.w3.weight"].shape == (2048, 5632)
    assert "layers.0.attention.pos_embed.inv_freq" in sd and "output.weight" in sd


def test_rotary_tables_match_open_lm():
    olm = tic.import_open_lm()
    from open_lm.positional_embedding.rotary import RotaryEmbedding

    rot = RotaryEmbedding(128, 256)
    rot._update_cos_sin_tables(256, device=torch.device("cpu"), dtype=torch.float32)
    cos, sin = tic.rotary_tables(rot.inv_freq, 256, torch.float32, "cpu")
    assert torch.allclose(cos, rot._cos_cached, atol=1e-6) and torch.allclose(sin, rot._sin_cached, atol=1e-6)
    assert olm is not None


def test_right_padding_needs_no_mask():
    """Batched right-padded scoring equals one-by-one scoring on a tiny random open_lm model."""
    olm = tic.import_open_lm()
    from open_lm.attention import torch_attn
    from open_lm.norms import get_norm_class

    torch.manual_seed(0)
    p = olm.Params(dim=64, n_layers=2, n_heads=4, seq_len=64, vocab_size=97, norm_type=get_norm_class("gain_only_lp_layer_norm"),
                   attn_func=torch_attn, apply_qk_norm=True, positional_embedding_type="rotary", ffn_type="swiglu_torch")
    model = olm.Transformer(p).eval()
    seqs = [[0, 5, 9, 3, 7, 2], [0, 11, 4], [0, 8, 8, 1, 6]]
    L = max(map(len, seqs))
    batch = torch.zeros((len(seqs), L), dtype=torch.long)
    for j, s in enumerate(seqs):
        batch[j, :len(s)] = torch.tensor(s)
    with torch.no_grad():
        lb = torch.log_softmax(model(input_ids=batch)[0], -1)
        for j, s in enumerate(seqs):
            ls = torch.log_softmax(model(input_ids=torch.tensor([s]))[0], -1)[0]
            assert torch.allclose(lb[j, :len(s)], ls, atol=1e-5)


def test_curve_stats_recovers_planted_departure():
    import tic_curve

    t = tic_curve.T
    c = 0.30 + 0.004 * (t - 2013) + np.array([0.003, -0.002, 0.001, -0.003, 0.002, -0.001, 0.0])
    c[-1] += 0.10
    s = tic_curve.curve_stats(c)
    assert s["r"] == pytest.approx(0.10, abs=0.01)
    assert s["z"] > tic_curve.T975[4] and s["zJ"] > tic_curve.T975[4]
    assert s["J_post"] == pytest.approx(c[-1] - c[-2])
    flat = c.copy()
    flat[-1] -= 0.10
    assert tic_curve.curve_stats(flat)["z"] < tic_curve.T975[4]
