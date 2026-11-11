# Derived from csathe_trtllm_finalize.py (nvcsathe/Megatron-LM@trtllm-fused-moe, a5a2f299ea).
# Changes: the per-token loop runs to valid_tokens instead of max_tokens, and a bf16 output
# (bf16 combine buffer) is allowed; the fp32 accumulator is cast on store.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Padding-aware finalization for the FlashInfer TRT-LLM routed MoE kernel."""

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


@triton.jit
def _padding_aware_finalize_valid_loop_kernel(
    gemm2_ptr,
    weights_ptr,
    inverse_map_ptr,
    output_ptr,
    valid_tokens_ptr,
    hidden_size,
    max_tokens,
    gemm2_rows,
    TOP_K: tl.constexpr,
    BLOCK_TOP_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_TOKEN_BLOCKS: tl.constexpr,
    NUM_H_BLOCKS: tl.constexpr,
    ROUND_TO_BF16: tl.constexpr,
):
    """Finalize local expert contributions while skipping invalid routed pairs."""
    token_pid = tl.program_id(0)
    hidden_pid = tl.program_id(1)
    valid_tokens = tl.load(valid_tokens_ptr).to(tl.int32)

    # The launch grid is fixed for CUDA graph replay. The device scalar gates the
    # live prefix without requiring a host synchronization or a dynamic launch.
    if token_pid >= valid_tokens:
        return

    top_k_offsets = tl.arange(0, BLOCK_TOP_K)
    top_k_mask = top_k_offsets < TOP_K

    # Loop only over the valid prefix (the original walks all max_tokens rows and tests each).
    for token in tl.range(token_pid, valid_tokens, NUM_TOKEN_BLOCKS):
        if True:
            mapped_rows = tl.load(
                inverse_map_ptr + token * TOP_K + top_k_offsets, mask=top_k_mask, other=-1
            ).to(tl.int32)
            local_mask = top_k_mask & (mapped_rows >= 0) & (mapped_rows < gemm2_rows)
            has_local_expert = tl.sum(local_mask.to(tl.int32), axis=0) > 0

            hidden_start = hidden_pid * BLOCK_H
            if has_local_expert:
                for hidden_offset in tl.range(hidden_start, hidden_size, NUM_H_BLOCKS * BLOCK_H):
                    offsets = hidden_offset + tl.arange(0, BLOCK_H)
                    offsets_mask = offsets < hidden_size
                    accumulator = tl.zeros([BLOCK_H], dtype=tl.float32)

                    # Accumulate in deterministic top-k order. Masked loads avoid touching
                    # GEMM2's expert-tile padding for non-local or invalid routed pairs.
                    for top_k_idx in range(TOP_K):
                        mapped_row = tl.load(inverse_map_ptr + token * TOP_K + top_k_idx).to(
                            tl.int32
                        )
                        is_local = (mapped_row >= 0) & (mapped_row < gemm2_rows)
                        safe_row = tl.maximum(mapped_row, 0).to(tl.int64)
                        weight = tl.load(
                            weights_ptr + token * TOP_K + top_k_idx, mask=is_local, other=0.0
                        ).to(tl.float32)
                        value = tl.load(
                            gemm2_ptr + safe_row * hidden_size + offsets,
                            mask=is_local & offsets_mask,
                            other=0.0,
                        ).to(tl.float32)
                        accumulator += value * weight

                    if ROUND_TO_BF16:
                        accumulator = accumulator.to(tl.bfloat16).to(tl.float32)
                    tl.store(
                        output_ptr + token * hidden_size + offsets, accumulator, mask=offsets_mask
                    )
            else:
                # RSV buffers persist across CUDA graph replays. Explicitly overwrite a
                # token that no longer has a local expert contribution instead of leaving
                # a stale value from an earlier replay.
                for hidden_offset in tl.range(hidden_start, hidden_size, NUM_H_BLOCKS * BLOCK_H):
                    offsets = hidden_offset + tl.arange(0, BLOCK_H)
                    offsets_mask = offsets < hidden_size
                    tl.store(
                        output_ptr + token * hidden_size + offsets,
                        tl.zeros([BLOCK_H], dtype=tl.float32),
                        mask=offsets_mask,
                    )


def padding_aware_finalize_valid_loop(
    gemm2_output: torch.Tensor,
    expert_weights: torch.Tensor,
    inverse_map: torch.Tensor,
    output: torch.Tensor,
    valid_tokens: torch.Tensor,
    top_k: int,
    *,
    round_to_bf16: bool = False,
) -> torch.Tensor:
    """Finalize a non-finalized FlashInfer TRT-LLM routed MoE result into FP32 RSV.

    Args:
        gemm2_output: Expert-grouped, tile-padded GEMM2 output with shape
            ``[padded_permuted_rows, hidden_size]``.
        expert_weights: Routing weights with ``max_tokens * top_k`` elements.
        inverse_map: Map from ``[token, top_k]`` to GEMM2 rows. Negative entries
            represent padding, a non-local expert, or another invalid routed pair.
        output: FP32 ``[max_tokens, hidden_size]`` output, normally the symmetric
            ReduceScatter-V input buffer.
        valid_tokens: Device ``int32[1]`` containing the live gathered-token prefix.
        top_k: Number of routed experts per token.
        round_to_bf16: Round the local result through BF16 before the FP32 store.
            This is useful when comparing against FlashInfer's BF16 finalizer.

    Returns:
        The supplied ``output`` tensor.
    """
    assert HAVE_TRITON, "Triton is required for padding-aware TRT-LLM MoE finalization."
    assert gemm2_output.is_cuda and gemm2_output.ndim == 2
    assert gemm2_output.is_contiguous(), "gemm2_output must be contiguous."
    assert expert_weights.is_cuda and expert_weights.is_contiguous()
    assert inverse_map.is_cuda and inverse_map.is_contiguous()
    assert output.is_cuda and output.ndim == 2 and output.dtype in (torch.float32, torch.bfloat16)
    assert output.is_contiguous(), "output must be contiguous."
    assert valid_tokens.is_cuda and valid_tokens.dtype == torch.int32
    assert valid_tokens.numel() == 1
    assert top_k > 0

    max_tokens, hidden_size = output.shape
    gemm2_rows, gemm2_hidden_size = gemm2_output.shape
    assert gemm2_hidden_size == hidden_size
    assert expert_weights.numel() >= max_tokens * top_k
    assert inverse_map.numel() >= max_tokens * top_k

    if max_tokens == 0 or hidden_size == 0:
        return output

    block_h = 256
    num_token_blocks = min(max_tokens, 128)
    num_h_blocks = min(triton.cdiv(hidden_size, block_h), 8)
    _padding_aware_finalize_valid_loop_kernel[(num_token_blocks, num_h_blocks)](
        gemm2_output,
        expert_weights,
        inverse_map,
        output,
        valid_tokens,
        hidden_size,
        max_tokens,
        gemm2_rows,
        TOP_K=top_k,
        BLOCK_TOP_K=triton.next_power_of_2(top_k),
        BLOCK_H=block_h,
        NUM_TOKEN_BLOCKS=num_token_blocks,
        NUM_H_BLOCKS=num_h_blocks,
        ROUND_TO_BF16=round_to_bf16,
        num_warps=4,
    )
    return output
