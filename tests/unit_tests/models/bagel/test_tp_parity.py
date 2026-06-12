# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""TP parity test for SelfAttentionMoT.

Verifies that ``SelfAttentionMoT._forward_train`` produces numerically
equivalent output at TP=1 vs TP=2. Catches regressions in the
``linear_qkv_out_dim_per_partition`` view (the bug fixed in Phase B) and
in the column/row-parallel sharding of ``linear_qkv`` / ``linear_proj``
for both the und and gen MoT branches.

Both ranks build two attention modules:
  * ``attn_tp1`` — TP=1 (singleton group), full unsharded weights.
  * ``attn_tp2`` — TP=2 (full pair group), per-rank-sharded weights.

We copy ``attn_tp1``'s full weights into ``attn_tp2`` with the proper
column-/row-parallel slicing per rank, then forward identical inputs
through both. ``RowParallelLinear`` all-reduces inside ``attn_tp2``, so
both ranks' final output should equal ``attn_tp1``'s full output.

Run::

    PYTHONPATH=/workspace/megatron-lm-bage_m4:/workspace/megatron-lm-bage_m4/bagel-package:\
/workspace/megatron-lm-bage_m4/bagel-package/bagel:/workspace/megatron-lm-bage_m4/examples/mimo_bagel \
        torchrun --nproc_per_node=2 examples/mimo_bagel/unit_test/test_tp_parity.py
"""

import math
import os
import sys

import torch
import torch.distributed as dist

_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_BAGEL_PKG = os.path.join(_ROOT, "bagel-package")
_BAGEL_SRC = os.path.join(_BAGEL_PKG, "bagel")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _BAGEL_PKG)
sys.path.insert(0, _BAGEL_SRC)

from megatron.core.models.bagel.attention_mot import (
    SelfAttentionMoT,
    SelfAttentionMoTSubmodules,
)
from megatron.core.models.bagel.flex_attention import FlexAttention
from megatron.core.models.bagel.mot_packed_seq_params import MoTPackedSeqParams

from torch.nn.attention.flex_attention import create_block_mask

from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


class _PGC:
    """Minimal ProcessGroupCollection-compatible container."""

    def __init__(self, tp, cp=None):
        self.tp = tp
        if cp is not None:
            self.cp = cp


def _block_mask(seq_len: int, device: str):
    return create_block_mask(
        lambda b, h, q, kv: q >= 0,
        B=1, H=1, Q_LEN=seq_len, KV_LEN=seq_len,
        device=device,
    )


def _make_config(nh: int, ng: int, hd: int) -> TransformerConfig:
    """Small bf16 TransformerConfig with GQA (num_query_groups < num_attention_heads)."""
    return TransformerConfig(
        num_layers=1,
        hidden_size=nh * hd,
        num_attention_heads=nh,
        num_query_groups=ng,
        kv_channels=hd,
        add_bias_linear=False,
        add_qkv_bias=False,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        attention_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
    )


def _make_attention(config: TransformerConfig, tp_group, seed: int) -> SelfAttentionMoT:
    """Build a SelfAttentionMoT (no qk layernorm, FlexAttention core)."""
    pgc = _PGC(tp=tp_group, cp=None)
    submodules = SelfAttentionMoTSubmodules(
        linear_qkv=ColumnParallelLinear,
        core_attention=FlexAttention,
        linear_proj=RowParallelLinear,
        q_layernorm=None,
        k_layernorm=None,
        linear_qkv_gen=ColumnParallelLinear,
        linear_proj_gen=RowParallelLinear,
        q_layernorm_gen=None,
        k_layernorm_gen=None,
    )
    torch.manual_seed(seed)
    attn = SelfAttentionMoT(
        config=config,
        submodules=submodules,
        layer_number=1,
        attn_mask_type=AttnMaskType.padding,
        pg_collection=pgc,
    )
    return attn.cuda().eval()


def _shard_copy_weights(src_attn: SelfAttentionMoT, dst_attn: SelfAttentionMoT,
                        tp_size: int, rank: int) -> None:
    """Copy ``src_attn`` (TP=1, full weights) into ``dst_attn`` (TP=tp_size, sharded)."""
    def _col_shard(full, tp_size, rank):
        # ColumnParallelLinear partitions along dim=0 (output_dim).
        per = full.shape[0] // tp_size
        return full[rank * per : (rank + 1) * per].clone()

    def _row_shard(full, tp_size, rank):
        # RowParallelLinear partitions along dim=1 (input_dim).
        per = full.shape[1] // tp_size
        return full[:, rank * per : (rank + 1) * per].clone()

    pairs_col = [
        (src_attn.linear_qkv,     dst_attn.linear_qkv),
        (src_attn.linear_qkv_gen, dst_attn.linear_qkv_gen),
    ]
    pairs_row = [
        (src_attn.linear_proj,     dst_attn.linear_proj),
        (src_attn.linear_proj_gen, dst_attn.linear_proj_gen),
    ]

    with torch.no_grad():
        for src_lin, dst_lin in pairs_col:
            if src_lin is None or dst_lin is None:
                continue
            dst_lin.weight.data.copy_(_col_shard(src_lin.weight.data, tp_size, rank))
            if getattr(src_lin, "bias", None) is not None and getattr(dst_lin, "bias", None) is not None:
                # Column-parallel bias is also partitioned along dim=0.
                per = src_lin.bias.shape[0] // tp_size
                dst_lin.bias.data.copy_(src_lin.bias.data[rank * per : (rank + 1) * per].clone())

        for src_lin, dst_lin in pairs_row:
            if src_lin is None or dst_lin is None:
                continue
            dst_lin.weight.data.copy_(_row_shard(src_lin.weight.data, tp_size, rank))
            # RowParallelLinear bias is full (added after all-reduce); identical on every rank.
            if getattr(src_lin, "bias", None) is not None and getattr(dst_lin, "bias", None) is not None:
                dst_lin.bias.data.copy_(src_lin.bias.data.clone())


# ─────────────────────────────────────────────────────────────────────────────
# Test driver
# ─────────────────────────────────────────────────────────────────────────────


def run_tp_parity_test(u: int, g: int, nh: int, ng: int, hd: int, seed: int = 42):
    """Compare SelfAttentionMoT TP=1 vs TP=2 forward output."""
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2, f"this test requires world_size=2, got {world}"

    device = "cuda"
    hidden = nh * hd
    config = _make_config(nh=nh, ng=ng, hd=hd)
    bm = _block_mask(u + g, device)

    # Build process groups.
    tp1_group = dist.new_group(ranks=[rank])           # singleton — TP=1 view
    tp2_group = dist.new_group(ranks=[0, 1])           # full pair  — TP=2 view

    # Build both attention modules with the same seed so that TP=1 weights
    # are identical on both ranks.
    attn_tp1 = _make_attention(config, tp1_group, seed=seed)
    attn_tp2 = _make_attention(config, tp2_group, seed=seed + 1)  # any seed; we overwrite below

    # Sync attn_tp2's weights to be the proper TP-shard of attn_tp1's full weights.
    _shard_copy_weights(attn_tp1, attn_tp2, tp_size=2, rank=rank)

    # Same input across ranks (identical seed).
    torch.manual_seed(seed + 1000)
    hs = torch.randn(u + g, 1, hidden, dtype=torch.bfloat16, device=device)

    psp = MoTPackedSeqParams(
        qkv_format="thd",
        packed_und_token_indexes=torch.arange(u, device=device),
        packed_gen_token_indexes=torch.arange(u, u + g, device=device),
        local_und_token_indexes=torch.arange(u, device=device),
        local_gen_token_indexes=torch.arange(u, u + g, device=device),
        padded_und_seqlen=u,
        padded_gen_seqlen=g,
    )

    with torch.no_grad():
        out_tp1, _ = attn_tp1._forward_train(hs, attention_mask=bm, packed_seq_params=psp)
        out_tp2, _ = attn_tp2._forward_train(hs, attention_mask=bm, packed_seq_params=psp)

    dist.barrier()

    assert out_tp1.shape == out_tp2.shape, (
        f"shape mismatch: tp1={out_tp1.shape}  tp2={out_tp2.shape}"
    )

    # Both ranks should see equal output (RowParallelLinear's all-reduce
    # produces the same final hidden state on every TP rank).
    err = (out_tp1 - out_tp2).abs().max().item()
    rel = err / (out_tp1.abs().max().item() + 1e-9)

    atol = rtol = 1e-2
    torch.testing.assert_close(
        out_tp2, out_tp1, atol=atol, rtol=rtol,
        msg=lambda m: f"[TP-parity rank={rank}]: {m}\nmax_abs_err={err:.6f}",
    )

    print(f"  [TP=1 vs TP=2 rank={rank}] PASS  "
          f"u={u} g={g} nh={nh} ng={ng} hd={hd}  "
          f"max_abs_err={err:.6f}  max_rel_err={rel:.6f}")


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)

    if world != 2:
        print(f"[TP parity] this test requires nproc_per_node=2 (got {world})")
        sys.exit(1)

    # Both branches active.
    run_tp_parity_test(u=8,  g=4,  nh=4, ng=2, hd=32)   # H=128
    run_tp_parity_test(u=16, g=8,  nh=4, ng=4, hd=32)   # H=128, MHA (ng=nh)
    run_tp_parity_test(u=8,  g=4,  nh=8, ng=2, hd=64)   # H=512, larger GQA ratio

    # Single-branch (und-only).
    run_tp_parity_test(u=12, g=0,  nh=4, ng=2, hd=32)
    # Single-branch (gen-only).
    run_tp_parity_test(u=0,  g=12, nh=4, ng=2, hd=32)

    if rank == 0:
        print("\nAll TP=2 SelfAttentionMoT parity tests passed.")


if __name__ == "__main__":
    main()
