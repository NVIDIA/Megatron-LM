# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Batch-invariant inference MoE helpers."""

from typing import Optional
from unittest.mock import MagicMock

import torch

from megatron.core.inference.communication.torch_symm_triton.barrier import symm_mem_sync
from megatron.core.inference.communication.torch_symm_triton.utils import (
    is_device_nvls_capable,
    sync_threads,
)
from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
    grouped_gemm_batch_invariant,
    grouped_gemm_batch_invariant_alignment,
    is_batch_invariant_mode_enabled,
)
from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()

try:
    from torch._C._distributed_c10d import _SymmetricMemory
except ImportError:
    _SymmetricMemory = MagicMock()


def enabled() -> bool:
    """Return whether global batch-invariant mode is active."""
    return is_batch_invariant_mode_enabled()


def grouped_mm(x_bf16: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """Batch-invariant BF16 grouped GEMM used by inference fused MoE."""
    return grouped_gemm_batch_invariant(
        x_bf16, weight, offs=offs.to(torch.int32), m_total=x_bf16.shape[0]
    )


def grouped_mm_alignment() -> int:
    """Per-expert row alignment required by the batch-invariant grouped GEMM."""
    return grouped_gemm_batch_invariant_alignment()


@triton.jit
def _squared_relu_with_probs_kernel(
    input_ptr,
    output_ptr,
    permutation_map_ptr,
    n_used_ptr,
    probs_ptr,
    hidden_size,
    max_rows,
    clamp_scale,
    CLAMP: tl.constexpr,
    ZERO_PADDING: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    """Apply squared ReLU and router probabilities in training order.

    With CLAMP set this reproduces training's fused ``weighted_clamped_squared_relu`` bit
    for bit: the soft-clamped pre-activation and the square both stay in FP32, and the only
    BF16 round is the final one after the FP32 routing probability is applied.

    Without CLAMP the square is materialized in BF16 first, matching the unclamped
    ``weighted_squared_relu``, which squares a BF16 ReLU output.
    """
    pid = tl.program_id(0)
    n_used = tl.load(n_used_ptr)
    if pid >= n_used:
        return

    for row in tl.range(pid, max_rows, NUM_BLOCKS):
        if row < n_used:
            if tl.load(permutation_map_ptr + row) >= 0:
                prob = tl.load(probs_ptr + row)
                for offset in tl.range(0, hidden_size, BLOCK_SIZE):
                    cols = offset + tl.arange(0, BLOCK_SIZE)
                    mask = cols < hidden_size
                    value = tl.load(input_ptr + row * hidden_size + cols, mask=mask).to(tl.float32)
                    value = tl.maximum(value, 0.0)
                    if CLAMP:
                        value = clamp_scale * libdevice.tanh(value / clamp_scale)
                    value = value * value
                    if not CLAMP:
                        # Unclamped training (weighted_squared_relu) squares a BF16 ReLU
                        # output, so the BF16 materialization is part of matching it. The
                        # clamped path stays in FP32 to the single final round instead.
                        value = value.to(tl.bfloat16).to(tl.float32)
                    value = (value * prob).to(tl.bfloat16)
                    tl.store(output_ptr + row * hidden_size + cols, value, mask=mask)
            elif ZERO_PADDING:
                for offset in tl.range(0, hidden_size, BLOCK_SIZE):
                    cols = offset + tl.arange(0, BLOCK_SIZE)
                    tl.store(output_ptr + row * hidden_size + cols, 0.0, mask=cols < hidden_size)


@triton.jit
def _swiglu_with_probs_kernel(
    input_ptr,
    output_ptr,
    permutation_map_ptr,
    n_used_ptr,
    probs_ptr,
    ffn_size,  # output width; input row width is 2*ffn_size (gate | up)
    max_rows,
    ZERO_PADDING: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    """Apply gated SiLU (SwiGLU) and router probabilities in training order.

    Matches the training fused weighted-swiglu rounding: SiLU(gate)*up*prob is
    computed in FP32 with a single BF16 round at the end. Input row width is
    2*ffn_size: gate = first half, up = second half (megatron chunk
    convention). Fixed NUM_BLOCKS CTAs iterating rows -> CUDA-graph safe.
    """
    pid = tl.program_id(0)
    n_used = tl.load(n_used_ptr)
    if pid >= n_used:
        return
    two_n = 2 * ffn_size

    for row in tl.range(pid, max_rows, NUM_BLOCKS):
        if row < n_used:
            if tl.load(permutation_map_ptr + row) >= 0:
                prob = tl.load(probs_ptr + row)
                for offset in tl.range(0, ffn_size, BLOCK_SIZE):
                    cols = offset + tl.arange(0, BLOCK_SIZE)
                    mask = cols < ffn_size
                    gate = tl.load(input_ptr + row * two_n + cols, mask=mask).to(tl.float32)
                    up = tl.load(input_ptr + row * two_n + ffn_size + cols, mask=mask).to(
                        tl.float32
                    )
                    value = gate * tl.sigmoid(gate) * up * prob
                    tl.store(output_ptr + row * ffn_size + cols, value.to(tl.bfloat16), mask=mask)
            elif ZERO_PADDING:
                for offset in tl.range(0, ffn_size, BLOCK_SIZE):
                    cols = offset + tl.arange(0, BLOCK_SIZE)
                    tl.store(output_ptr + row * ffn_size + cols, 0.0, mask=cols < ffn_size)


def swiglu_with_probs(
    x: torch.Tensor,
    permutation_map: torch.Tensor,
    n_used: torch.Tensor,
    probs: torch.Tensor,
    zero_padding: bool = False,
) -> torch.Tensor:
    """Gated-SiLU counterpart of squared_relu_with_probs (SwiGLU models)."""
    num_rows, two_ffn = x.shape
    ffn_size = two_ffn // 2
    out = torch.empty(num_rows, ffn_size, dtype=x.dtype, device=x.device)
    block_size = min(triton.next_power_of_2(ffn_size), 1024)
    num_blocks = min(num_rows, 512)
    _swiglu_with_probs_kernel[(num_blocks,)](
        x,
        out,
        permutation_map,
        n_used,
        probs,
        ffn_size,
        num_rows,
        ZERO_PADDING=zero_padding,
        BLOCK_SIZE=block_size,
        NUM_BLOCKS=num_blocks,
    )
    return out


@triton.jit
def _weighted_silu_mul_bounded_kernel(
    in_ptr0, in_ptr1, out_ptr0, bound_ptr, xnumel, HALF_N: tl.constexpr, XBLOCK: tl.constexpr
):
    """Device-bounded weighted SwiGLU with training-parity rounding.

    The per-element instruction sequence is copied VERBATIM from Inductor's
    emitted Triton for the training fused weighted-swiglu
    (bf16 -> fp32 silu(gate) * up * prob -> bf16, single final rounding), so a
    token's activation bits match the training forward exactly. Elementwise
    kernels have no cross-element reduction, so only the per-element sequence
    determines bits; the schedule below is a persistent 1D grid (static launch,
    CUDA-graph-safe) striding while xoffset < a DEVICE element bound
    (= valid_tokens * topk * HALF_N — the live prefix of the flat token-major
    layout). Rows beyond the bound are neither read nor written.
    """
    xbound = tl.load(bound_ptr)
    num_progs = tl.num_programs(0)
    xoffset = tl.program_id(0) * XBLOCK
    while xoffset < xbound:
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = (xindex < xbound) & (xindex < xnumel)
        x0 = xindex % HALF_N
        x1 = xindex // HALF_N
        tmp0 = tl.load(in_ptr0 + (x0 + 2 * HALF_N * x1), xmask).to(tl.float32)
        tmp8 = tl.load(in_ptr0 + (HALF_N + x0 + 2 * HALF_N * x1), xmask).to(tl.float32)
        tmp11 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
        tmp1 = tmp0.to(tl.float32)
        tmp2 = -tmp1
        tmp3 = libdevice.exp(tmp2)
        tmp4 = tl.full([1], 1.0, tl.float32)
        tmp5 = tmp3 + tmp4
        tmp6 = tmp1 / tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp10 * tmp11
        tmp13 = tmp12.to(tl.float32)
        tl.store(out_ptr0 + xindex, tmp13, xmask)
        xoffset += num_progs * XBLOCK


def weighted_silu_mul_bounded(
    y: torch.Tensor,
    weights_flat: torch.Tensor,
    bound_elems: torch.Tensor,
    num_programs: Optional[int] = None,
    xblock: int = 1024,
) -> torch.Tensor:
    """SwiGLU with routing weights applied at the activation (training parity).

    y: [rows, 2*half_n] bf16 (gate | up); weights_flat: [rows] fp32 routing
    probabilities; bound_elems: device scalar = live_rows * half_n.
    Returns [rows, half_n] bf16; rows beyond the live bound are untouched.

    num_programs defaults to SMs * 8 waves (Inductor's persistent-grid sizing;
    1184 on the B200 this was captured from). Grid size cannot affect bits:
    the kernel is elementwise with each program owning a disjoint strided
    index range, so it is an occupancy knob only.
    """
    rows, two_half_n = y.shape
    half_n = two_half_n // 2
    if num_programs is None:
        # Lazy import: permute.py imports this module at its top level.
        from megatron.core.inference.moe.permute import _get_num_sms

        num_programs = _get_num_sms(y.device) * 8
    out = torch.empty(rows, half_n, dtype=y.dtype, device=y.device)
    _weighted_silu_mul_bounded_kernel[(num_programs,)](
        y, weights_flat, out, bound_elems, rows * half_n, HALF_N=half_n, XBLOCK=xblock
    )
    return out


def squared_relu_with_probs(
    x: torch.Tensor,
    permutation_map: torch.Tensor,
    n_used: torch.Tensor,
    probs: torch.Tensor,
    clamp_scale: Optional[float] = None,
    zero_padding: bool = False,
) -> torch.Tensor:
    """Match training's BF16 squared-ReLU rounding before the FP32 probability multiply.

    Args:
        clamp_scale: config.activation_func_tanh_clamp_scale. If set, precondition the
            input with the tanh soft clamp ``s * tanh(x / s)``.
    """
    num_rows, hidden_size = x.shape
    out = torch.empty_like(x)
    block_size = min(triton.next_power_of_2(hidden_size), 1024)
    num_blocks = min(num_rows, 512)
    _squared_relu_with_probs_kernel[(num_blocks,)](
        x,
        out,
        permutation_map,
        n_used,
        probs,
        hidden_size,
        num_rows,
        clamp_scale if clamp_scale is not None else 0.0,
        CLAMP=clamp_scale is not None,
        ZERO_PADDING=zero_padding,
        BLOCK_SIZE=block_size,
        NUM_BLOCKS=num_blocks,
    )
    return out


@triton.jit
def _ordered_reduce_scatter_v_kernel(
    local_ptr,
    buffer_ptrs_dev,
    signal_pad_ptrs,
    local_tokens,
    rank_token_offset_ptr,
    ep_max_tokens_ptr,
    input_byte_offset,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """Reduce peer rows with an explicit rank-order FP32 sum."""
    pid = tl.program_id(axis=0)

    ep_max_tokens = tl.load(ep_max_tokens_ptr)
    if pid >= ep_max_tokens:
        return

    symm_mem_sync(
        signal_pad_ptrs,
        None,
        RANK,
        WORLD_SIZE,
        hasPreviousMemAccess=False,
        hasSubsequentMemAccess=True,
    )
    sync_threads()

    tid = tl.arange(0, BLOCK_SIZE)
    rank_token_offset = tl.load(rank_token_offset_ptr)
    buffer_ptrs = buffer_ptrs_dev.to(tl.pointer_type(tl.uint64))

    for token_offset in range(pid, local_tokens, tl.num_programs(axis=0)):
        global_token = rank_token_offset + token_offset

        for channel_offset in range(0, HIDDEN_SIZE, BLOCK_SIZE):
            offsets = channel_offset + tid
            mask = offsets < HIDDEN_SIZE
            acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

            for src_rank in tl.range(0, WORLD_SIZE):
                peer_base = tl.load(buffer_ptrs + src_rank).to(tl.pointer_type(tl.uint8))
                peer_ptr = (peer_base + input_byte_offset).to(tl.pointer_type(tl.float32))
                values = tl.load(
                    peer_ptr + global_token * HIDDEN_SIZE + offsets, mask=mask, other=0.0
                )
                acc += values

            tl.store(local_ptr + token_offset * HIDDEN_SIZE + offsets, acc, mask=mask)


def ordered_reduce_scatter_v(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    symm_mem_hdl: _SymmetricMemory,
    rank_token_offset: torch.Tensor,
    ep_max_tokens: torch.Tensor,
    per_rank_max_tokens: int,
    input_byte_offset: int = 0,
    **kwargs,
) -> torch.Tensor:
    """Reduce-scatter variable token rows with a fixed FP32 rank order."""
    assert HAVE_TRITON, "Triton is required for ordered_reduce_scatter_v."
    assert (
        output_tensor.ndim == 2 and input_tensor.ndim == 2
    ), "output_tensor and input_tensor must be 2-D [tokens, hidden_size]."
    assert is_device_nvls_capable(
        output_tensor.device
    ), "ordered_reduce_scatter_v requires a Hopper+ GPU with NVLink (SM >= 9)."
    assert (
        output_tensor.dtype == input_tensor.dtype == torch.float32
    ), "ordered_reduce_scatter_v requires fp32 input and output tensors."
    assert (
        rank_token_offset.numel() == 1
        and rank_token_offset.dtype == torch.int32
        and rank_token_offset.is_cuda
    ), "rank_token_offset must be a scalar int32 CUDA tensor."

    hidden_size = output_tensor.shape[1]
    assert (
        input_tensor.shape[1] == hidden_size
    ), f"input and output hidden_size mismatch: {input_tensor.shape[1]} vs {hidden_size}"

    max_num_blocks = kwargs.get("max_num_blocks", 128)
    block_size = min(triton.next_power_of_2(hidden_size), 1024)
    num_warps = max(1, block_size // 32)
    num_blocks = min(per_rank_max_tokens, max_num_blocks)

    _ordered_reduce_scatter_v_kernel[(num_blocks, 1, 1)](
        output_tensor,
        symm_mem_hdl.buffer_ptrs_dev,
        symm_mem_hdl.signal_pad_ptrs_dev,
        local_tokens=output_tensor.shape[0],
        rank_token_offset_ptr=rank_token_offset,
        ep_max_tokens_ptr=ep_max_tokens,
        input_byte_offset=input_byte_offset,
        HIDDEN_SIZE=hidden_size,
        BLOCK_SIZE=block_size,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=symm_mem_hdl.world_size,
        num_warps=num_warps,
    )
    return output_tensor


@triton.jit
def _unpermute_tokens_in_expert_order_kernel(
    expert_out_ptr,  # [output_size, hidden_dim] bf16 expert outputs
    inverse_map_ptr,  # [num_tokens, num_local_experts] permuted row or -1
    valid_tokens_ptr,  # scalar int32 CUDA tensor: number of valid tokens
    output_ptr,  # [num_tokens, hidden_dim] fp32 output buffer
    hidden_dim,
    num_local_experts: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Token-local batch-invariant unpermute.

    Each program owns one output token and one hidden tile. Contributions are
    accumulated in fp32 by increasing local expert id, avoiding atomic-add order.
    """
    tok = tl.program_id(0)
    block_h = tl.program_id(1)
    valid_tokens = tl.load(valid_tokens_ptr)
    offsets = block_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offsets < hidden_dim

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    if tok < valid_tokens:
        for lid in tl.range(0, num_local_experts):
            pos = tl.load(inverse_map_ptr + tok * num_local_experts + lid)
            if pos >= 0:
                vals = tl.load(expert_out_ptr + pos * hidden_dim + offsets, mask=mask_h).to(
                    tl.float32
                )
                acc += vals
        tl.store(output_ptr + tok * hidden_dim + offsets, acc, mask=mask_h)


def unpermute_tokens_in_expert_order(
    expert_output: torch.Tensor,
    inverse_map: torch.Tensor,
    valid_tokens: torch.Tensor,
    out: Optional[torch.Tensor],
) -> torch.Tensor:
    """Reduce local expert contributions token-by-token in fixed expert order."""
    _, hidden_dim = expert_output.shape
    num_tokens, num_local_experts = inverse_map.shape
    if out is None:
        out = torch.empty(num_tokens, hidden_dim, dtype=torch.float32, device=expert_output.device)

    BLOCK_H = min(triton.next_power_of_2(hidden_dim), 1024)
    grid = (num_tokens, triton.cdiv(hidden_dim, BLOCK_H))
    _unpermute_tokens_in_expert_order_kernel[grid](
        expert_output,
        inverse_map,
        valid_tokens,
        out,
        hidden_dim,
        num_local_experts,
        BLOCK_H=BLOCK_H,
    )
    return out
