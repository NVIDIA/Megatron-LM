# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""P0 acceptance tests for ``msa_ref``.

Run with ``python -m pytest experimental/lite/ref/minimax_m3/test_msa_ref.py -v``
or directly with ``python test_msa_ref.py`` (prints a summary).

1. Degenerate case: when the number of eligible blocks <= top-k, MSA output must
   equal full causal attention (fp64, 1e-10).
2. Block-index invariants (local block present, causal, unique, left-packed).
3. Bitwise index agreement and fp32 output agreement against the Hugging Face
   ``MiniMaxM3VLAttention`` sparse layer on identical random weights (skipped
   when transformers lacks ``minimax_m3_vl``).
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from msa_ref import (  # noqa: E402
    MSARefAttention,
    MSARefConfig,
    eligible_blocks_leq_topk,
)

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _cfg(**kw) -> MSARefConfig:
    base = dict(
        hidden_size=256,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        rotary_dim=32,
        index_n_heads=2,
        index_head_dim=64,
        index_block_size=128,
        index_topk_blocks=4,
    )
    base.update(kw)
    return MSARefConfig(**base)


def _init(module: torch.nn.Module, std: float = 0.05) -> None:
    for name, p in module.named_parameters():
        if name.endswith("norm"):
            torch.nn.init.normal_(p, std=0.1)  # non-trivial (1 + w) scale
        else:
            torch.nn.init.normal_(p, std=std)


# --------------------------------------------------------------------------- #
def test_degenerate_equals_dense_fp64():
    cfg = _cfg()
    layer = MSARefAttention(cfg, dtype=torch.float64).to(DEVICE)
    _init(layer)
    for seq in (128, 500, 512):  # multiple / non-multiple / exactly topk*block
        assert eligible_blocks_leq_topk(seq, cfg.index_block_size, cfg.index_topk_blocks)
        x = torch.randn(2, seq, cfg.hidden_size, dtype=torch.float64, device=DEVICE)
        sparse, idx = layer(x)
        dense = layer.dense_reference(x)
        err = (sparse - dense).abs().max().item()
        assert err < 1e-10, f"seq={seq}: sparse vs dense max_abs={err:.3e}"
        assert idx is not None and idx.shape[-1] == -(-seq // cfg.index_block_size)


def test_sparse_differs_from_dense_when_truly_sparse():
    cfg = _cfg()
    layer = MSARefAttention(cfg, dtype=torch.float64).to(DEVICE)
    _init(layer)
    seq = 1024  # 8 blocks > top-4
    x = torch.randn(1, seq, cfg.hidden_size, dtype=torch.float64, device=DEVICE)
    sparse, idx = layer(x)
    dense = layer.dense_reference(x)
    assert idx.shape[-1] == cfg.index_topk_blocks
    # Queries in the first topk blocks see everything -> identical; later queries must differ.
    early = cfg.index_topk_blocks * cfg.index_block_size
    assert (sparse[:, :early] - dense[:, :early]).abs().max().item() < 1e-10
    assert (sparse[:, early:] - dense[:, early:]).abs().max().item() > 1e-6


def test_block_index_invariants():
    cfg = _cfg()
    layer = MSARefAttention(cfg, dtype=torch.float64).to(DEVICE)
    _init(layer)
    for seq in (1000, 1024):
        x = torch.randn(1, seq, cfg.hidden_size, dtype=torch.float64, device=DEVICE)
        _, idx = layer(x)  # [B, H_idx, S, k]
        pos = torch.arange(seq, device=DEVICE)
        q_block = pos // cfg.index_block_size
        valid = idx >= 0
        # causal: no selected block is in the future
        assert bool((idx[valid] <= q_block[None, None, :, None].expand_as(idx)[valid]).all())
        # local block always selected
        assert bool((idx == q_block[None, None, :, None]).any(-1).all())
        # unique within a row (ignoring -1)
        srt = idx.sort(-1).values
        dup = (srt[..., 1:] == srt[..., :-1]) & (srt[..., 1:] >= 0)
        assert not bool(dup.any())
        # left-packed: once -1 appears, everything after is -1
        first_neg = (~valid).int().cumsum(-1)
        assert bool(((first_neg > 0) == ~valid).all())
        # number of valid entries == min(topk, q_block + 1)
        n_valid = valid.sum(-1)
        expect = torch.clamp(q_block + 1, max=cfg.index_topk_blocks)
        assert bool((n_valid == expect[None, None, :]).all())


# --------------------------------------------------------------------------- #
def _hf_available() -> bool:
    try:
        import transformers.models.minimax_m3_vl.modeling_minimax_m3_vl  # noqa: F401
        return True
    except Exception:
        return False


def test_against_hf_fp32():
    if not _hf_available():
        print("SKIP: transformers without minimax_m3_vl")
        return
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLTextConfig
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
        MiniMaxM3VLAttention,
        MiniMaxM3VLRotaryEmbedding,
    )

    cfg = _cfg(hidden_size=256, num_attention_heads=8, num_key_value_heads=2, head_dim=64, rotary_dim=32,
               index_n_heads=2, index_head_dim=64, index_topk_blocks=4)
    hf_cfg = MiniMaxM3VLTextConfig(
        vocab_size=1024, hidden_size=cfg.hidden_size, num_hidden_layers=1,
        num_attention_heads=cfg.num_attention_heads, num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta, rotary_dim=cfg.rotary_dim, partial_rotary_factor=cfg.rotary_dim / cfg.head_dim,
        index_n_heads=cfg.index_n_heads, index_head_dim=cfg.index_head_dim,
        index_block_size=cfg.index_block_size, index_topk_blocks=cfg.index_topk_blocks, index_local_blocks=1,
        layer_types=["minimax_m3_sparse"], mlp_layer_types=["dense"], attn_implementation="eager",
    )
    assert hf_cfg.rope_parameters.get("partial_rotary_factor", 1.0) == cfg.rotary_dim / cfg.head_dim, hf_cfg.rope_parameters
    hf_attn = MiniMaxM3VLAttention(hf_cfg, layer_idx=0).to(DEVICE).float()
    hf_rope = MiniMaxM3VLRotaryEmbedding(hf_cfg).to(DEVICE)
    ref = MSARefAttention(cfg, dtype=torch.float32).to(DEVICE)
    _init(ref)
    # copy weights ref -> hf
    with torch.no_grad():
        hf_attn.q_proj.weight.copy_(ref.q_proj.weight); hf_attn.k_proj.weight.copy_(ref.k_proj.weight)
        hf_attn.v_proj.weight.copy_(ref.v_proj.weight); hf_attn.o_proj.weight.copy_(ref.o_proj.weight)
        hf_attn.q_norm.weight.copy_(ref.q_norm); hf_attn.k_norm.weight.copy_(ref.k_norm)
        hf_attn.indexer.q_proj.weight.copy_(ref.index_q_proj.weight)
        hf_attn.indexer.k_proj.weight.copy_(ref.index_k_proj.weight)
        hf_attn.indexer.q_norm.weight.copy_(ref.index_q_norm); hf_attn.indexer.k_norm.weight.copy_(ref.index_k_norm)

    for seq in (1000, 1024):
        x = torch.randn(2, seq, cfg.hidden_size, dtype=torch.float32, device=DEVICE)
        pos = torch.arange(seq, device=DEVICE).unsqueeze(0).expand(2, -1)
        cos, sin = hf_rope(x, pos)
        with torch.no_grad():
            hf_out, _ = hf_attn(x, (cos, sin), attention_mask=None, position_ids=pos)
            hf_idx = hf_attn.indexer(x, (cos, sin), None, pos)
            ref_out, ref_idx = ref(x, pos)
        assert hf_idx.shape == ref_idx.shape, (hf_idx.shape, ref_idx.shape)
        # compare as sets per row (top-k order may differ on ties); require identical valid sets
        assert torch.equal(hf_idx.sort(-1).values, ref_idx.sort(-1).values), "block indices differ from HF"
        err = (hf_out - ref_out).abs().max().item()
        rel = err / hf_out.abs().max().item()
        print(f"  seq={seq}: HF vs ref fp32 max_abs={err:.3e} rel={rel:.3e}")
        assert rel < 1e-5, f"seq={seq}: rel={rel:.3e}"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
