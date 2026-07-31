# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fully-fused MoE decode kernel: FC1 -> SwiGLU -> FC2 -> topk-weighted reduce.

Single Triton kernel per (expert, M-block). Each program:
  1. computes the FC1 grouped GEMM for its BLOCK_M token-rows against expert e's
     gate|up weights (two accumulators),
  2. applies SwiGLU (SiLU(gate) * up) in the fp32 epilogue -> bf16 activated
     intermediate held in registers,
  3. computes the FC2 grouped GEMM (looping over the H output in BLOCK_N2 tiles),
  4. scales each row by its routing probability and atomically accumulates into
     the output buffer at the original token index.

This eliminates the standalone `bounded_silu_mul` and `_moe_sum` kernels and the
two intermediate HBM round-trips of the 4-kernel `vllm_fused_moe` path.

Reuses the CUDA-graph-safe indirection tables from `vllm_fused_moe`
(`_moe_align_block_size_cuda_graphable`) so it is drop-in behind the same
inference call contract. Output is atomic-accumulated, so the valid output rows
must be zeroed before launch (done here with a bounded, CG-safe kernel).
"""

from typing import Optional
from unittest.mock import MagicMock

import torch

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()

from megatron.core.inference.moe.fused_moe import ActivationType
from megatron.core.inference.moe.permute import _get_num_sms
from megatron.core.inference.moe.vllm_fused_moe import (
    _ceil_div,
    _moe_align_block_size_cuda_graphable,
)


@triton.jit
def _zero_valid_rows_kernel(out_ptr, valid_tokens_ptr, H, max_rows, BLOCK_N: tl.constexpr,
                            NUM_BLOCKS: tl.constexpr):
    """Zero out[0:valid_tokens, :H] with a fixed grid (CUDA-graph safe)."""
    pid = tl.program_id(0)
    n_used = tl.load(valid_tokens_ptr)
    for row in tl.range(pid, max_rows, NUM_BLOCKS):
        if row < n_used:
            for n in tl.range(0, H, BLOCK_N):
                o = n + tl.arange(0, BLOCK_N)
                tl.store(out_ptr + row * H + o, tl.zeros([BLOCK_N], dtype=out_ptr.dtype.element_ty),
                         mask=o < H)


@triton.jit
def _fused_moe_decode_kernel(
    a_ptr,            # hidden states [max_tokens, K1] bf16
    w1_ptr,           # fc1 weight [E, 2*Nf, K1] bf16   (gate|up stacked on dim 1)
    w2_ptr,           # fc2 weight [E, H, Nf]   bf16
    out_ptr,          # output [max_tokens, H]  (fp32 or bf16); atomic-add target
    topk_weights_ptr,  # [max_tokens*topk] fp32 flattened routing probs
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # dims
    H,
    Nf,
    K1,
    num_valid_tokens,
    # strides
    stride_am, stride_ak,
    stride_w1e, stride_w1n, stride_w1k,
    stride_w2e, stride_w2n, stride_w2k,
    stride_om, stride_on,
    top_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K1: tl.constexpr,
    BLOCK_NF: tl.constexpr,
    NUM_NF: tl.constexpr,
    BLOCK_N2: tl.constexpr,
):
    """One program per (expert-block). Loops over all blocks via grid-stride.

    Nf (FC1 out / FC2 contraction) is tiled in ``NUM_NF`` power-of-2 ``BLOCK_NF``
    chunks (768 = 3x256). Phase 1 computes the SwiGLU activation for every Nf tile
    and keeps the bf16 result resident (small: [BLOCK_M, BLOCK_NF] x NUM_NF). Phase 2
    contracts those resident tiles against W2 over H tiles. This avoids both a
    non-pow2 ``arange(Nf)`` and a full ``[BLOCK_M, H]`` accumulator.
    """
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_M)

    pid_init = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    for pid_m in tl.range(pid_init, num_pid_m, grid_size):
        off_e = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
        if off_e != -1:
            offs_token_id = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
            offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
            token_mask = offs_token < num_valid_tokens
            row = offs_token // top_k  # input/output token row
            probs = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)

            # FC1+SwiGLU per Nf tile, then immediately contract that tile through FC2
            # and atomic-accumulate its partial into out. Because
            #   out[row] = prob * sum_j (act_j @ W2_j),
            # each Nf tile's FC2 contribution can be added independently, so we need
            # neither a resident activation list nor a full [BLOCK_M, H] accumulator.
            offs_k = tl.arange(0, BLOCK_K1)
            offs_nf_base = tl.arange(0, BLOCK_NF)
            for j in tl.static_range(NUM_NF):
                offs_nf = j * BLOCK_NF + offs_nf_base
                nf_mask = offs_nf < Nf
                a_ptrs = a_ptr + (row[:, None] * stride_am + offs_k[None, :] * stride_ak)
                # W1 tiles in [K, NF] orientation so dot(a[M,K], w[K,NF]) needs no trans.
                wg_ptrs = (w1_ptr + off_e * stride_w1e
                           + offs_k[:, None] * stride_w1k + offs_nf[None, :] * stride_w1n)
                wu_ptrs = (w1_ptr + off_e * stride_w1e
                           + offs_k[:, None] * stride_w1k + (offs_nf[None, :] + Nf) * stride_w1n)
                gate_acc = tl.zeros((BLOCK_M, BLOCK_NF), dtype=tl.float32)
                up_acc = tl.zeros((BLOCK_M, BLOCK_NF), dtype=tl.float32)
                for k in range(0, tl.cdiv(K1, BLOCK_K1)):
                    k_mask = offs_k < K1 - k * BLOCK_K1
                    a = tl.load(a_ptrs, mask=token_mask[:, None] & k_mask[None, :], other=0.0)
                    wg = tl.load(wg_ptrs, mask=k_mask[:, None] & nf_mask[None, :], other=0.0)
                    wu = tl.load(wu_ptrs, mask=k_mask[:, None] & nf_mask[None, :], other=0.0)
                    gate_acc += tl.dot(a, wg)
                    up_acc += tl.dot(a, wu)
                    a_ptrs += BLOCK_K1 * stride_ak
                    wg_ptrs += BLOCK_K1 * stride_w1k
                    wu_ptrs += BLOCK_K1 * stride_w1k
                # SwiGLU: SiLU(gate) * up in fp32, cast to bf16 for FC2 (matches ref)
                silu = gate_acc * tl.sigmoid(gate_acc)
                act = (silu * up_acc).to(tl.bfloat16)  # [BLOCK_M, BLOCK_NF]

                # ---- FC2 for this Nf tile: out[row] += prob * (act @ W2_j), tiled over H
                for n2 in range(0, tl.cdiv(H, BLOCK_N2)):
                    offs_n2 = n2 * BLOCK_N2 + tl.arange(0, BLOCK_N2)
                    n2_mask = offs_n2 < H
                    # W2 tile [BLOCK_NF, BLOCK_N2]: contraction=Nf (stride_w2k), free=H (stride_w2n)
                    w2_ptrs = (w2_ptr + off_e * stride_w2e
                               + offs_nf[:, None] * stride_w2k + offs_n2[None, :] * stride_w2n)
                    w2 = tl.load(w2_ptrs, mask=nf_mask[:, None] & n2_mask[None, :], other=0.0)
                    y = tl.dot(act, w2) * probs[:, None]  # [BLOCK_M, BLOCK_N2]
                    out_ptrs = out_ptr + row[:, None] * stride_om + offs_n2[None, :] * stride_on
                    tl.atomic_add(out_ptrs, y, mask=token_mask[:, None] & n2_mask[None, :])


def fused_moe_decode(
    hidden_states: torch.Tensor,
    probs: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    activation_type: ActivationType,
    num_local_experts: int,
    local_expert_start: int,
    valid_tokens: torch.Tensor,
    routing_map: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    num_tokens_hint: Optional[int] = None,
    block_m: int = 16,
    block_k1: int = 64,
    block_nf: int = 256,
    block_n2: int = 128,
    num_warps: int = 8,
    num_stages: int = 3,
) -> torch.Tensor:
    """Fully-fused MoE (FC1+SwiGLU+FC2+topk reduce). Signature mirrors vllm_fused_moe.

    Only SwiGLU is supported (Qwen3 MoE). fp32 accumulation matches the reference.
    """
    assert activation_type == ActivationType.SWIGLU, "fused_moe_decode supports SwiGLU only"
    assert hidden_states.dtype == torch.bfloat16

    max_tokens = hidden_states.size(0)
    topk = routing_map.shape[1]
    effective_tokens = num_tokens_hint if num_tokens_hint is not None else max_tokens

    two_nf = fc1_weight.size(1)
    Nf = two_nf // 2
    K1 = fc1_weight.size(2)
    H = fc2_weight.size(1)
    assert fc2_weight.size(2) == Nf

    # Indirection table (groups token-pair indices by local expert into BLOCK_M blocks).
    sorted_token_ids, expert_ids, num_post_padded = _moe_align_block_size_cuda_graphable(
        routing_map, block_m, num_local_experts, local_expert_start, valid_tokens
    )
    num_valid = max_tokens * topk

    if out is None:
        out = torch.empty(max_tokens, H, dtype=torch.float32, device=hidden_states.device)

    # Zero the valid output rows (atomic-add accumulation target), CG-safe bounded grid.
    zero_blocks = min(max_tokens, 1024)
    _zero_valid_rows_kernel[(zero_blocks,)](
        out, valid_tokens, H, max_tokens, BLOCK_N=min(triton.next_power_of_2(H), 1024),
        NUM_BLOCKS=zero_blocks,
    )

    topk_weights_flat = probs.reshape(-1).contiguous()

    grid_size = _get_num_sms(hidden_states.device)
    _fused_moe_decode_kernel[(grid_size,)](
        hidden_states,
        fc1_weight,
        fc2_weight,
        out,
        topk_weights_flat,
        sorted_token_ids,
        expert_ids,
        num_post_padded,
        H,
        Nf,
        K1,
        num_valid,
        hidden_states.stride(0), hidden_states.stride(1),
        fc1_weight.stride(0), fc1_weight.stride(1), fc1_weight.stride(2),
        fc2_weight.stride(0), fc2_weight.stride(1), fc2_weight.stride(2),
        out.stride(0), out.stride(1),
        top_k=topk,
        BLOCK_M=block_m,
        BLOCK_K1=block_k1,
        BLOCK_NF=block_nf,
        NUM_NF=_ceil_div(Nf, block_nf),
        BLOCK_N2=block_n2,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
