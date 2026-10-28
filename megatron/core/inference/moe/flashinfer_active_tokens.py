# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""TEMPORARY: FlashInfer's MoE kernels with a device-side `num_active_tokens` argument.

FlashInfer is adding the matching `num_active_tokens` argument to the two kernels Megatron calls:

- `cutlass_fused_moe` (bf16 experts): flashinfer-ai/flashinfer#6783
- `trtllm_fp8_block_scale_routed_moe` (MXFP8 experts): flashinfer-ai/flashinfer#6784

Until a FlashInfer release ships them, this module exports both kernels under their own names
and signatures. When the installed FlashInfer accepts `num_active_tokens` the call is
forwarded unchanged. Otherwise the kernel runs without its own finalize and a persistent Triton
kernel finalizes the live rows into `output`; rows at or beyond the count are left untouched,
as the real argument guarantees.

UNWIND, once `pyproject.toml` requires a FlashInfer release that ships both changes:

1. `megatron/core/transformer/moe/experts.py`:
    import `cutlass_fused_moe` from `flashinfer.fused_moe` instead of this module.
2. `megatron/core/inference/moe/flashinfer_mxfp8.py`:
    import `trtllm_fp8_block_scale_routed_moe` from `flashinfer.fused_moe` instead of this module.
3. Delete this file, `tests/unit_tests/determinism/kernels/test_flashinfer_active_tokens.py` and
   the `inference_flashinfer_active_tokens` entry of
   `tests/unit_tests/determinism/kernels/manifest.py`.
"""

from __future__ import annotations

import inspect

import torch

from megatron.core.utils import null_decorator

try:
    from flashinfer.fused_moe import cutlass_fused_moe as _cutlass_fused_moe
    from flashinfer.fused_moe import (
        trtllm_fp8_block_scale_routed_moe as _trtllm_fp8_block_scale_routed_moe,
    )
except ImportError:
    _cutlass_fused_moe = None
    _trtllm_fp8_block_scale_routed_moe = None

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    from unittest.mock import MagicMock

    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()
    HAVE_TRITON = False

# Persistent grid: each CTA strides over the live rows, so idle CTAs cost nothing at decode loads.
# GB300, 65,536 live rows: 4096 CTAs x 4 warps 240 us (512 x 4: 438 us).
_FINALIZE_CTAS = 4096


def _accepts_num_active_tokens(kernel) -> bool:
    """Whether the installed kernel takes `num_active_tokens`."""
    return kernel is not None and "num_active_tokens" in inspect.signature(kernel).parameters


CUTLASS_MOE_HAS_ACTIVE_TOKENS = _accepts_num_active_tokens(_cutlass_fused_moe)
ROUTED_MOE_HAS_ACTIVE_TOKENS = _accepts_num_active_tokens(_trtllm_fp8_block_scale_routed_moe)


def _require_triton() -> None:
    if not HAVE_TRITON:
        raise RuntimeError("Emulating FlashInfer's num_active_tokens needs triton.")


@triton.jit
def _finalize_routed_rows_kernel(
    gemm2_ptr,  # [permuted_rows, gemm2_stride] bf16 FC2 rows in permuted order
    idx_ptr,  # [rows * TOPK] int32: (token, k) -> permuted row, -1 when the expert is not local
    topk_ids_ptr,  # [rows, TOPK] int32 packed (expert_id << 16) | bf16 weight bits
    out_ptr,  # [rows, hidden] fp32 output; rows [0, num_active_tokens) are fully written
    num_active_tokens_ptr,  # scalar int32 CUDA tensor
    hidden,  # logical hidden size
    gemm2_stride,  # row stride of gemm2_ptr (padded hidden size)
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,  # persistent grid size
):
    """Gather each live token's local-expert FC2 rows, weighted by its bf16 routing weights."""
    pid = tl.program_id(0)
    num_active_tokens = tl.load(num_active_tokens_ptr)
    for row in tl.range(pid, num_active_tokens, NUM_BLOCKS):
        row64 = tl.cast(row, tl.int64)
        for h in tl.range(0, hidden, BLOCK_H):
            offs = h + tl.arange(0, BLOCK_H)
            col_mask = offs < hidden
            acc = tl.zeros([BLOCK_H], dtype=tl.float32)
            for k in tl.static_range(TOPK):
                src = tl.load(idx_ptr + row * TOPK + k)
                is_local = src >= 0
                packed = tl.load(topk_ids_ptr + row * TOPK + k).to(tl.uint32)
                # The low 16 bits hold the bf16 routing weight; bf16 is the top half of fp32.
                weight = ((packed & 0xFFFF) << 16).to(tl.float32, bitcast=True)
                weight = tl.where(is_local, weight, 0.0)
                src64 = tl.where(is_local, src, 0).to(tl.int64)
                values = tl.load(
                    gemm2_ptr + src64 * gemm2_stride + offs, mask=col_mask & is_local, other=0.0
                ).to(tl.float32)
                acc += values * weight
            tl.store(out_ptr + row64 * hidden + offs, acc, mask=col_mask)


@triton.jit
def _combine_valid_rows_kernel(
    src_ptr,  # [rows, hidden] bf16 kernel output; only tokens with a local expert were written
    routing_ptr,  # [rows, TOPK] int32 global expert ids
    out_ptr,  # [rows, hidden] fp32 output; rows [0, num_active_tokens) are fully written
    num_active_tokens_ptr,  # scalar int32 CUDA tensor
    hidden,
    local_expert_offset,
    num_local_experts,
    TOPK: tl.constexpr,
    TOPK_PAD: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,  # persistent grid size
):
    """Copy each live token's row to fp32, or zero it when none of its experts is local."""
    pid = tl.program_id(0)
    num_active_tokens = tl.load(num_active_tokens_ptr)
    k = tl.arange(0, TOPK_PAD)
    for row in tl.range(pid, num_active_tokens, NUM_BLOCKS):
        row64 = tl.cast(row, tl.int64)
        experts = tl.load(routing_ptr + row64 * TOPK + k, mask=k < TOPK, other=-1)
        experts = experts - local_expert_offset
        is_local = (experts >= 0) & (experts < num_local_experts)
        has_local = tl.sum(is_local.to(tl.int32), axis=0) > 0
        for h in tl.range(0, hidden, BLOCK_H):
            offs = h + tl.arange(0, BLOCK_H)
            col_mask = offs < hidden
            values = tl.load(src_ptr + row64 * hidden + offs, mask=col_mask & has_local, other=0.0)
            tl.store(out_ptr + row64 * hidden + offs, values.to(tl.float32), mask=col_mask)


def finalize_routed_rows(
    gemm2_output: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    topk_ids: torch.Tensor,
    num_active_tokens: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Combine the unfinalized FC2 rows of rows [0, num_active_tokens) into the fp32 `output`,
    weighting each route with the bf16 weight packed in `topk_ids`; rows beyond the count are not
    written. Stands in for the kernel's own finalize with `num_active_tokens`. Returns `output`."""
    _require_triton()
    rows, topk = topk_ids.shape
    hidden = output.shape[1]
    assert topk_ids.dtype == torch.int32 and output.dtype == torch.float32
    assert output.shape[0] == rows and expanded_idx_to_permuted_idx.numel() == rows * topk
    assert gemm2_output.ndim == 2 and gemm2_output.shape[1] >= hidden
    num_blocks = min(rows, _FINALIZE_CTAS)
    _finalize_routed_rows_kernel[(num_blocks,)](
        gemm2_output,
        expanded_idx_to_permuted_idx,
        topk_ids,
        output,
        num_active_tokens,
        hidden,
        gemm2_output.shape[1],
        TOPK=topk,
        BLOCK_H=min(triton.next_power_of_2(hidden), 1024),
        NUM_BLOCKS=num_blocks,
    )
    return output


def combine_valid_rows(
    expert_output: torch.Tensor,
    routing_ids: torch.Tensor,
    num_active_tokens: torch.Tensor,
    output: torch.Tensor,
    *,
    local_expert_offset: int,
    num_local_experts: int,
) -> torch.Tensor:
    """Write rows [0, num_active_tokens) of the bf16 `expert_output` into the fp32 `output`; a
    token with no expert in [local_expert_offset, local_expert_offset + num_local_experts) gets a
    zero row, as the kernel's filling finalize would have produced. Returns `output`."""
    _require_triton()
    rows, hidden = expert_output.shape
    assert routing_ids.dtype == torch.int32 and routing_ids.shape[0] == rows
    assert output.shape == (rows, hidden) and output.dtype == torch.float32
    topk = routing_ids.shape[1]
    num_blocks = min(rows, _FINALIZE_CTAS)
    _combine_valid_rows_kernel[(num_blocks,)](
        expert_output,
        routing_ids,
        output,
        num_active_tokens,
        hidden,
        local_expert_offset,
        num_local_experts,
        TOPK=topk,
        TOPK_PAD=triton.next_power_of_2(topk),
        BLOCK_H=min(triton.next_power_of_2(hidden), 1024),
        NUM_BLOCKS=num_blocks,
    )
    return output


def cutlass_fused_moe(
    input: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    fc1_expert_weights: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    output_dtype: torch.dtype,
    *,
    output: torch.Tensor | None = None,
    num_active_tokens: torch.Tensor | None = None,
    ep_size: int = 1,
    ep_rank: int = 0,
    **kwargs,
):
    """`flashinfer.fused_moe.cutlass_fused_moe` with `num_active_tokens`.

    Forwarded unchanged when FlashInfer accepts the argument, or when it is None. Otherwise the
    kernel runs with its non-filling finalize (`enable_alltoall=True`), which writes a bf16 row
    only for tokens that route to a local expert, and `combine_valid_rows` copies the live rows
    into the fp32 `output`. Returns `[output]`, as the kernel does.
    """
    if _cutlass_fused_moe is None:
        raise RuntimeError("flashinfer-python is required for cutlass_fused_moe.")
    operands = (
        input,
        token_selected_experts,
        token_final_scales,
        fc1_expert_weights,
        fc2_expert_weights,
    )
    common = dict(kwargs, ep_size=ep_size, ep_rank=ep_rank)
    if num_active_tokens is None:
        return _cutlass_fused_moe(*operands, output_dtype, output=output, **common)
    if CUTLASS_MOE_HAS_ACTIVE_TOKENS:
        return _cutlass_fused_moe(
            *operands, output_dtype, output=output, num_active_tokens=num_active_tokens, **common
        )
    assert output is not None, "the emulated num_active_tokens needs an fp32 output buffer"
    kernel_rows = _cutlass_fused_moe(*operands, input.dtype, enable_alltoall=True, **common)[0]
    num_local_experts = fc1_expert_weights.shape[0]
    combine_valid_rows(
        kernel_rows,
        token_selected_experts,
        num_active_tokens,
        output,
        local_expert_offset=ep_rank * num_local_experts,
        num_local_experts=num_local_experts,
    )
    return [output]


def trtllm_fp8_block_scale_routed_moe(
    *,
    topk_ids: torch.Tensor,
    do_finalize: bool = True,
    output: torch.Tensor | None = None,
    num_active_tokens: torch.Tensor | None = None,
    **kwargs,
):
    """`flashinfer.fused_moe.trtllm_fp8_block_scale_routed_moe` with `num_active_tokens`.

    Keyword-only. Forwarded unchanged when FlashInfer accepts the argument, or when it is None.
    Otherwise the kernel runs with `do_finalize=False` and `finalize_routed_rows` combines the
    FC2 rows of the live tokens into the fp32 `output`. Returns `output`, as the finalized kernel
    does.
    """
    if _trtllm_fp8_block_scale_routed_moe is None:
        raise RuntimeError("flashinfer-python is required for trtllm_fp8_block_scale_routed_moe.")
    forwarded = dict(kwargs, topk_ids=topk_ids, do_finalize=do_finalize)
    if output is not None:
        forwarded["output"] = output
    if num_active_tokens is None:
        return _trtllm_fp8_block_scale_routed_moe(**forwarded)
    if ROUTED_MOE_HAS_ACTIVE_TOKENS:
        return _trtllm_fp8_block_scale_routed_moe(num_active_tokens=num_active_tokens, **forwarded)
    assert (
        do_finalize and output is not None
    ), "the emulated num_active_tokens needs do_finalize=True and an fp32 output buffer"
    # do_finalize=False returns [gemm2_output, expert_weights, expanded_idx_to_permuted_idx].
    gemm2_output, _, expanded_idx = _trtllm_fp8_block_scale_routed_moe(
        **dict(kwargs, topk_ids=topk_ids, do_finalize=False)
    )
    finalize_routed_rows(gemm2_output, expanded_idx, topk_ids, num_active_tokens, output)
    return output
