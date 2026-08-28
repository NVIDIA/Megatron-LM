# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Memory-efficient Triton kernels for the dense SBHD CSA teacher denominator."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _csa_window_lse_kernel(
        query,
        full_kv,
        window_indices,
        attn_sink,
        output,
        stride_q_row: tl.constexpr,
        stride_q_head: tl.constexpr,
        stride_q_dim: tl.constexpr,
        stride_kv_row: tl.constexpr,
        stride_kv_dim: tl.constexpr,
        stride_idx_row: tl.constexpr,
        stride_idx_col: tl.constexpr,
        stride_sink: tl.constexpr,
        stride_out_row: tl.constexpr,
        stride_out_head: tl.constexpr,
        softmax_scale,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        total_kv: tl.constexpr,
        window_width: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Compute ``log(exp(sink) + sum_window(exp(q @ k * scale)))``."""
        query_row = tl.program_id(0)
        head_block = tl.program_id(1)

        head_offsets = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
        dim_offsets = tl.arange(0, BLOCK_D)
        head_mask = head_offsets < num_heads
        dim_mask = dim_offsets < head_dim

        q_offsets = (
            query_row * stride_q_row
            + head_offsets[:, None] * stride_q_head
            + dim_offsets[None, :] * stride_q_dim
        )
        q = tl.load(query + q_offsets, mask=head_mask[:, None] & dim_mask[None, :], other=0.0)

        running_max = tl.load(
            attn_sink + head_offsets * stride_sink, mask=head_mask, other=-float("inf")
        ).to(tl.float32)
        running_sum = tl.where(head_mask & (running_max > -float("inf")), 1.0, 0.0)

        for key_start in range(0, window_width, BLOCK_K):
            key_offsets = key_start + tl.arange(0, BLOCK_K)
            index_mask = key_offsets < window_width
            global_indices = tl.load(
                window_indices + query_row * stride_idx_row + key_offsets * stride_idx_col,
                mask=index_mask,
                other=-1,
            )
            valid_keys = index_mask & (global_indices >= 0) & (global_indices < total_kv)
            safe_indices = tl.where(valid_keys, global_indices, 0)

            k_offsets = dim_offsets[:, None] * stride_kv_dim + safe_indices[None, :] * stride_kv_row
            k = tl.load(
                full_kv + k_offsets, mask=dim_mask[:, None] & valid_keys[None, :], other=0.0
            )
            logits = tl.dot(q, k, out_dtype=tl.float32) * softmax_scale
            score_mask = head_mask[:, None] & valid_keys[None, :]
            logits = tl.where(score_mask, logits, -float("inf"))

            tile_max = tl.max(logits, axis=1)
            new_max = tl.maximum(running_max, tile_max)
            old_scale = tl.where(running_max > -float("inf"), tl.exp(running_max - new_max), 0.0)
            tile_sum = tl.sum(tl.where(score_mask, tl.exp(logits - new_max[:, None]), 0.0), axis=1)
            running_sum = running_sum * old_scale + tile_sum
            running_max = new_max

        lse = tl.where(running_sum > 0.0, running_max + tl.log(running_sum), -float("inf"))
        tl.store(
            output + query_row * stride_out_row + head_offsets * stride_out_head,
            lse,
            mask=head_mask,
        )

    @triton.jit
    def _csa_compressed_lse_sbhd_kernel(
        query,
        compressed_kv,
        non_compressed_lse,
        output,
        stride_q_row: tl.constexpr,
        stride_q_head: tl.constexpr,
        stride_q_dim: tl.constexpr,
        stride_k_batch: tl.constexpr,
        stride_k_row: tl.constexpr,
        stride_k_dim: tl.constexpr,
        stride_noncomp_row: tl.constexpr,
        stride_noncomp_head: tl.constexpr,
        stride_out_batch: tl.constexpr,
        stride_out_row: tl.constexpr,
        stride_out_head: tl.constexpr,
        softmax_scale,
        batch_size: tl.constexpr,
        seqlen_q: tl.constexpr,
        seqlen_k: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        ratio: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Add every causal compressed-key contribution for an SBHD input."""
        query_blocks = tl.cdiv(seqlen_q, BLOCK_Q)
        batch = tl.program_id(0) // query_blocks
        query_block = tl.program_id(0) % query_blocks
        head_block = tl.program_id(1)

        row_offsets = tl.arange(0, BLOCK_Q * BLOCK_H)
        query_offsets = query_block * BLOCK_Q + row_offsets // BLOCK_H
        head_offsets = head_block * BLOCK_H + row_offsets % BLOCK_H
        flat_query_rows = query_offsets * batch_size + batch
        row_mask = (query_offsets < seqlen_q) & (head_offsets < num_heads)
        dim_offsets = tl.arange(0, BLOCK_D)
        dim_mask = dim_offsets < head_dim

        q_offsets = (
            flat_query_rows[:, None] * stride_q_row
            + head_offsets[:, None] * stride_q_head
            + dim_offsets[None, :] * stride_q_dim
        )
        q = tl.load(query + q_offsets, mask=row_mask[:, None] & dim_mask[None, :], other=0.0)

        running_max = tl.load(
            non_compressed_lse
            + flat_query_rows * stride_noncomp_row
            + head_offsets * stride_noncomp_head,
            mask=row_mask,
            other=-float("inf"),
        ).to(tl.float32)
        running_sum = tl.where(row_mask & (running_max > -float("inf")), 1.0, 0.0)
        visible_keys = (query_offsets + 1) // ratio

        for key_start in range(0, seqlen_k, BLOCK_K):
            key_offsets = key_start + tl.arange(0, BLOCK_K)
            key_mask = key_offsets < seqlen_k
            k_offsets = (
                batch * stride_k_batch
                + dim_offsets[:, None] * stride_k_dim
                + key_offsets[None, :] * stride_k_row
            )
            k = tl.load(
                compressed_kv + k_offsets, mask=dim_mask[:, None] & key_mask[None, :], other=0.0
            )
            logits = tl.dot(q, k, out_dtype=tl.float32) * softmax_scale
            score_mask = (
                row_mask[:, None]
                & key_mask[None, :]
                & (key_offsets[None, :] < visible_keys[:, None])
            )
            logits = tl.where(score_mask, logits, -float("inf"))

            tile_max = tl.max(logits, axis=1)
            new_max = tl.maximum(running_max, tile_max)
            old_scale = tl.where(running_max > -float("inf"), tl.exp(running_max - new_max), 0.0)
            tile_sum = tl.sum(tl.where(score_mask, tl.exp(logits - new_max[:, None]), 0.0), axis=1)
            running_sum = running_sum * old_scale + tile_sum
            running_max = new_max

        lse = tl.where(running_sum > 0.0, running_max + tl.log(running_sum), -float("inf"))
        output_offsets = (
            batch * stride_out_batch
            + query_offsets * stride_out_row
            + head_offsets * stride_out_head
        )
        tl.store(output + output_offsets, lse, mask=row_mask)


def csa_teacher_lse_unsupported_reason(
    query: Tensor, full_kv: Tensor, compressed_kv: Tensor, attn_sink: Tensor, window_indices: Tensor
) -> Optional[str]:
    """Return why the Triton SBHD teacher-LSE kernels cannot run, or None if supported."""
    tensors = (query, full_kv, compressed_kv, attn_sink, window_indices)
    if not _TRITON_AVAILABLE:
        return "Triton is not available"
    if not all(tensor.is_cuda for tensor in tensors):
        return "query, full_kv, compressed_kv, attn_sink, and window_indices must be CUDA tensors"
    if any(tensor.device != query.device for tensor in tensors[1:]):
        return "query, full_kv, compressed_kv, attn_sink, and window_indices must share a device"
    if query.dtype not in (torch.bfloat16, torch.float16):
        return f"query dtype must be bfloat16 or float16, got {query.dtype}"
    if full_kv.dtype != query.dtype or compressed_kv.dtype != query.dtype:
        return "query, full_kv, and compressed_kv must have the same dtype"
    if query.ndim != 3 or full_kv.ndim != 2 or compressed_kv.ndim != 3:
        return (
            "expected flat query [total_q, heads, dim], flat full_kv [total_kv, dim], "
            "and SBHD compressed_kv [batch, seqlen_k, dim]"
        )
    if query.shape[-1] != full_kv.shape[-1] or query.shape[-1] != compressed_kv.shape[-1]:
        return "query, full_kv, and compressed_kv head dimensions must match"
    if query.shape[-1] < 16 or query.shape[-1] > 512:
        return f"head dimension must be in [16, 512], got {query.shape[-1]}"
    if query.stride(-1) != 1 or full_kv.stride(-1) != 1 or compressed_kv.stride(-1) != 1:
        return "query, full_kv, and compressed_kv must be contiguous in the head dimension"
    if attn_sink.ndim != 1 or attn_sink.numel() != query.shape[1]:
        return f"attn_sink must have shape [{query.shape[1]}], got {tuple(attn_sink.shape)}"
    if window_indices.ndim != 2 or window_indices.shape[0] != query.shape[0]:
        return (
            f"window_indices must have shape [{query.shape[0]}, window], "
            f"got {tuple(window_indices.shape)}"
        )
    if window_indices.dtype not in (torch.int32, torch.int64):
        return f"window_indices dtype must be int32 or int64, got {window_indices.dtype}"
    return None


def can_use_fused_csa_teacher_lse(
    query: Tensor, full_kv: Tensor, compressed_kv: Tensor, attn_sink: Tensor, window_indices: Tensor
) -> bool:
    """Return whether the Triton SBHD teacher-LSE kernels support these tensors."""
    return (
        csa_teacher_lse_unsupported_reason(query, full_kv, compressed_kv, attn_sink, window_indices)
        is None
    )


@torch.no_grad()
def fused_csa_teacher_lse(
    query: Tensor,
    full_kv: Tensor,
    compressed_kv: Tensor,
    attn_sink: Tensor,
    window_indices: Tensor,
    softmax_scale: float,
    ratio: int,
    *,
    batch_size: int,
    seqlen_q: int,
) -> Tensor:
    """Compute the full SBHD CSA teacher LSE without a score-matrix temporary.

    ``query`` and ``full_kv`` use FlashMLA's flat-global layout. Sliding-window
    indices address ``full_kv`` directly. ``compressed_kv`` has B/K/D layout.
    The result has B/S/H layout.
    """
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    unsupported_reason = csa_teacher_lse_unsupported_reason(
        query, full_kv, compressed_kv, attn_sink, window_indices
    )
    if unsupported_reason is not None:
        raise RuntimeError(f"fused SBHD CSA teacher LSE is unavailable: {unsupported_reason}")

    total_q, num_heads, head_dim = query.shape
    if compressed_kv.shape[0] != batch_size:
        raise ValueError("SBHD compressed_kv must have shape [batch, seqlen_k, dim]")
    if total_q != batch_size * seqlen_q:
        raise ValueError("flat query length must equal batch_size * seqlen_q")

    block_d = max(16, triton.next_power_of_2(head_dim))
    window_block_h = min(128, max(16, triton.next_power_of_2(num_heads)))
    window_block_k = min(64, max(16, triton.next_power_of_2(max(1, window_indices.shape[1]))))
    window_num_stages = 1 if window_block_h == 128 and block_d == 512 else 2
    compressed_block_h = 16
    compressed_block_k = 32
    compressed_block_q = 8

    non_compressed_lse = torch.empty((total_q, num_heads), device=query.device, dtype=torch.float32)
    window_grid = (total_q, triton.cdiv(num_heads, window_block_h))
    output = torch.empty(
        (batch_size, seqlen_q, num_heads), device=query.device, dtype=torch.float32
    )
    compressed_grid = (
        batch_size * triton.cdiv(seqlen_q, compressed_block_q),
        triton.cdiv(num_heads, compressed_block_h),
    )

    with torch.cuda.device(query.device):
        _csa_window_lse_kernel[window_grid](
            query,
            full_kv,
            window_indices,
            attn_sink,
            non_compressed_lse,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            full_kv.stride(0),
            full_kv.stride(1),
            window_indices.stride(0),
            window_indices.stride(1),
            attn_sink.stride(0),
            non_compressed_lse.stride(0),
            non_compressed_lse.stride(1),
            softmax_scale,
            num_heads,
            head_dim,
            full_kv.shape[0],
            window_indices.shape[1],
            BLOCK_H=window_block_h,
            BLOCK_D=block_d,
            BLOCK_K=window_block_k,
            num_warps=8,
            num_stages=window_num_stages,
        )
        _csa_compressed_lse_sbhd_kernel[compressed_grid](
            query,
            compressed_kv,
            non_compressed_lse,
            output,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            compressed_kv.stride(0),
            compressed_kv.stride(1),
            compressed_kv.stride(2),
            non_compressed_lse.stride(0),
            non_compressed_lse.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            softmax_scale,
            batch_size,
            seqlen_q,
            compressed_kv.shape[1],
            num_heads,
            head_dim,
            ratio,
            BLOCK_Q=compressed_block_q,
            BLOCK_H=compressed_block_h,
            BLOCK_D=block_d,
            BLOCK_K=compressed_block_k,
            num_warps=8,
            num_stages=2,
        )
    return output


__all__ = [
    "can_use_fused_csa_teacher_lse",
    "csa_teacher_lse_unsupported_reason",
    "fused_csa_teacher_lse",
]
