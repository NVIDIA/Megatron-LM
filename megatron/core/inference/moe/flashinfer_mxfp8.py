# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""FlashInfer routed-MoE support for MCore MXFP8 expert weights."""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass

import torch

try:
    from flashinfer import shuffle_matrix_a
    from flashinfer.fused_moe import (
        Fp8QuantizationType,
        WeightLayout,
        trtllm_fp8_block_scale_routed_moe,
    )
    from flashinfer.utils import get_shuffle_matrix_sf_a_row_indices

    HAVE_FLASHINFER_ROUTED_MXFP8 = True
    _FLASHINFER_ROUTED_MXFP8_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:
    HAVE_FLASHINFER_ROUTED_MXFP8 = False
    _FLASHINFER_ROUTED_MXFP8_IMPORT_ERROR = exc

# Patch until TRTLLM kernels are updated.
# Track flashinfer-ai/flashinfer#6783 flashinfer-ai/flashinfer#6784 flashinfer-ai/flashinfer#6785
HAVE_FLASHINFER_ACTIVE_TOKENS = (
    HAVE_FLASHINFER_ROUTED_MXFP8
    and os.environ.get("MCORE_FLASHINFER_ACTIVE_TOKENS", "1") != "0"
    and "num_active_tokens" in inspect.signature(trtllm_fp8_block_scale_routed_moe).parameters
)

from megatron.core.inference.quantization.mxfp8_quantize import (
    MXFP8_BLOCK_SIZE,
    _mxfp8_quantize_groups,
)
from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
from megatron.core.utils import null_decorator

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

# Persistent grid sizes; each CTA strides over the live rows, so idle CTAs cost nothing at decode.
# GB300, 65,536 live rows: quantize 2048 CTAs x 8 warps 94 us (512 x 4: 196 us);
# combine 4096 CTAs x 4 warps 240 us (512 x 4: 438 us).
_QUANTIZE_CTAS = 2048
# Only used before FlashInfer's update; see `HAVE_FLASHINFER_ACTIVE_TOKENS`.
_COMBINE_CTAS = 4096


def require_flashinfer_routed_mxfp8() -> None:
    """Raise an error when routed MXFP8 APIs are unavailable."""
    if not HAVE_FLASHINFER_ROUTED_MXFP8:
        raise RuntimeError(
            "FlashInfer routed MXFP8 MoE requires FlashInfer >= 0.6.4 which provides "
            "mxfp8_quantize, shuffled Major-K weights, and "
            "trtllm_fp8_block_scale_routed_moe. Upgrade flashinfer-python or select a "
            "different inference_grouped_gemm_backend."
        ) from _FLASHINFER_ROUTED_MXFP8_IMPORT_ERROR


@dataclass(frozen=True)
class FlashInferRoutedMXFP8Weight:
    """An expert-weight stack in TRT-LLM Major-K MXFP8 layout."""

    data: torch.Tensor
    scale: torch.Tensor
    logical_rows: int
    logical_cols: int
    padded_rows: int
    padded_cols: int


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _unshuffle_cublas_scale(
    scale: torch.Tensor, logical_rows: int, logical_cols: int
) -> torch.Tensor:
    """Undo MCore's 128x4 cuBLAS scale swizzle into a logical scale matrix."""
    if logical_cols % 32:
        raise ValueError(f"MXFP8 K dimension must be divisible by 32; got {logical_cols}")
    padded_rows = _round_up(logical_rows, 128)
    logical_scale_cols = logical_cols // 32
    padded_scale_cols = _round_up(logical_scale_cols, 4)
    expected = padded_rows * padded_scale_cols
    scale_u8 = scale.contiguous().view(torch.uint8).reshape(-1)
    if scale_u8.numel() != expected:
        raise ValueError(
            "unexpected MXFP8 scale size: "
            f"got {scale_u8.numel()}, expected {expected} for "
            f"weight shape=({logical_rows}, {logical_cols})"
        )
    logical = (
        scale_u8.reshape(padded_rows // 128, padded_scale_cols // 4, 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .reshape(padded_rows, padded_scale_cols)
    )
    return logical[:logical_rows, :logical_scale_cols]


def _block_scale_interleave(scale: torch.Tensor) -> torch.Tensor:
    """Apply TRT-LLM's architecture-independent 128x4 scale interleave."""
    rows, columns = scale.shape
    if rows % 128 or columns % 4:
        raise ValueError(
            "TRT-LLM scale interleave requires rows divisible by 128 and columns "
            f"divisible by 4; got shape={tuple(scale.shape)}"
        )
    return (
        scale.reshape(rows // 128, 4, 32, columns // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .reshape(-1)
    )


def _shuffle_routed_scale(scale: torch.Tensor) -> torch.Tensor:
    """Convert a logical UE8M0 scale matrix to TRT-LLM Major-K byte layout."""
    scale_u8 = scale.contiguous().view(torch.uint8)
    # Avoid FlashInfer's architecture-dispatched convenience wrapper: this
    # permutation and interleave are architecture-independent.
    row_indices = get_shuffle_matrix_sf_a_row_indices(scale_u8, 128).to(scale.device)
    return _block_scale_interleave(scale_u8[row_indices]).contiguous().view(torch.uint8)


def prepare_routed_mxfp8_weights(
    weight: MXFP8Tensor, out: FlashInferRoutedMXFP8Weight | None = None
) -> FlashInferRoutedMXFP8Weight:
    """Repack a stacked MCore MXFP8 expert weight without requantizing it.

    When `out` is provided, its data and scale tensors are refreshed in place.
    This preserves addresses captured by CUDA graphs across model refits.
    """
    require_flashinfer_routed_mxfp8()
    if weight.backend != "triton":
        raise ValueError(
            "FlashInfer routed MXFP8 weights must be repacked from the Triton/cublas "
            f"layout; got backend={weight.backend!r}."
        )
    if weight.data.ndim != 3:
        raise ValueError(f"expected [experts, M, K] MXFP8 data, got {weight.data.shape}")
    experts, logical_rows, logical_cols = weight.data.shape
    padded_rows = _round_up(logical_rows, 128)
    padded_cols = _round_up(logical_cols, 128)
    scale_cols = padded_cols // 32

    expected_data_shape = (experts, padded_rows, padded_cols)
    expected_scale_shape = (experts, padded_rows * scale_cols)
    if out is None:
        routed_data = torch.empty(
            expected_data_shape, dtype=weight.data.dtype, device=weight.data.device
        )
        routed_scale = torch.empty(
            expected_scale_shape, dtype=torch.uint8, device=weight.data.device
        )
    else:
        expected_metadata = (logical_rows, logical_cols, padded_rows, padded_cols)
        actual_metadata = (out.logical_rows, out.logical_cols, out.padded_rows, out.padded_cols)
        if actual_metadata != expected_metadata:
            raise ValueError(
                "existing FlashInfer routed MXFP8 weight metadata changed across refit: "
                f"got {actual_metadata}, expected {expected_metadata}"
            )
        if (
            tuple(out.data.shape) != expected_data_shape
            or tuple(out.scale.shape) != expected_scale_shape
        ):
            raise ValueError(
                "existing FlashInfer routed MXFP8 storage shape changed across refit: "
                f"data={tuple(out.data.shape)}, scale={tuple(out.scale.shape)}, "
                f"expected data={expected_data_shape}, scale={expected_scale_shape}"
            )
        routed_data = out.data
        routed_scale = out.scale

    # MCore and FlashInfer use different kernel-specific layouts for the same MXFP8
    # weights. MCore stores row-major MXFP8 data with cuBLAS-swizzled UE8M0 scales but TRT-LLM
    # kernels require 128-aligned Major-K data and its interleaved scale layout. Repack the data
    # and scales into TRT-LLM's 128-aligned Major-K layout without requantizing.
    for expert in range(experts):
        padded_data = torch.zeros(
            padded_rows, padded_cols, dtype=weight.data.dtype, device=weight.data.device
        )
        padded_data[:logical_rows, :logical_cols].copy_(weight.data[expert])
        routed_data[expert].copy_(
            shuffle_matrix_a(padded_data.view(torch.uint8), 128).view(weight.data.dtype)
        )

        logical_scale = torch.zeros(
            padded_rows, scale_cols, dtype=torch.uint8, device=weight.data.device
        )
        logical_scale[:logical_rows, : logical_cols // 32].copy_(
            _unshuffle_cublas_scale(weight.scale[expert], logical_rows, logical_cols)
        )
        routed_scale[expert].copy_(_shuffle_routed_scale(logical_scale))

    if out is not None:
        return out
    return FlashInferRoutedMXFP8Weight(
        data=routed_data,
        scale=routed_scale,
        logical_rows=logical_rows,
        logical_cols=logical_cols,
        padded_rows=padded_rows,
        padded_cols=padded_cols,
    )


@triton.jit
def _quantize_valid_rows_kernel(
    src_ptr,  # [rows, K] bf16/fp16/fp32 input
    out_ptr,  # [rows, padded_K] e4m3 output; columns >= K are zeroed
    scale_ptr,  # [rows, padded_K // 32] e8m0 scales in linear layout; groups >= K // 32 are zeroed
    valid_tokens_ptr,  # scalar int32 CUDA tensor: only rows [0, valid_tokens) are written
    K,  # logical hidden size (multiple of 32)
    padded_K,  # row stride of out_ptr (multiple of 128)
    BLOCK_K: tl.constexpr,  # next_power_of_2(padded_K)
    BLOCK_GROUPS: tl.constexpr,  # BLOCK_K // 32
    NUM_BLOCKS: tl.constexpr,  # persistent grid size
):
    """Quantize the valid rows to MXFP8 in the row-major layout the TRT-LLM routed MoE reads."""
    pid = tl.program_id(0)
    valid_tokens = tl.load(valid_tokens_ptr)
    padded_groups = padded_K // 32
    offs = tl.arange(0, BLOCK_K)
    load_mask = offs < K
    store_mask = offs < padded_K
    group_offs = tl.arange(0, BLOCK_GROUPS)
    group_mask = group_offs < padded_groups
    for row in tl.range(pid, valid_tokens, NUM_BLOCKS):
        row64 = tl.cast(row, tl.int64)
        # Padding columns load as 0, so they quantize to 0 with a 0 scale exponent.
        x = tl.load(src_ptr + row64 * K + offs, mask=load_mask, other=0.0).to(tl.float32)
        out_fp8, scale_exp = _mxfp8_quantize_groups(x, BLOCK_GROUPS)
        tl.store(out_ptr + row64 * padded_K + offs, out_fp8, mask=store_mask)
        tl.store(scale_ptr + row64 * padded_groups + group_offs, scale_exp, mask=group_mask)


@triton.jit
def _pack_routing_kernel(
    routing_ptr,  # [rows, TOPK] int expert ids; -1 marks an unrouted slot
    probs_ptr,  # [rows, TOPK] fp32 routing weights
    packed_ptr,  # [rows, TOPK] int32 output: (expert_id << 16) | bf16 weight bits
    valid_tokens_ptr,  # scalar int32 CUDA tensor: rows beyond it are packed as expert -1
    num_entries,  # rows * TOPK
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Pack routing into TRT-LLM's format, forcing every row beyond valid_tokens to expert -1."""
    pid = tl.program_id(0)
    valid_entries = tl.load(valid_tokens_ptr) * TOPK
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    in_bounds = offs < num_entries
    live = in_bounds & (offs < valid_entries)
    expert_ids = tl.load(routing_ptr + offs, mask=live, other=-1).to(tl.int32)
    probs = tl.load(probs_ptr + offs, mask=live, other=0.0).to(tl.float32)
    # Round-to-nearest-even fp32 -> bf16 on the bit pattern (probabilities are finite).
    prob_bits = probs.to(tl.int32, bitcast=True)
    weight_bits = ((prob_bits + 0x7FFF + ((prob_bits >> 16) & 1)) >> 16) & 0xFFFF
    tl.store(packed_ptr + offs, (expert_ids << 16) | weight_bits, mask=in_bounds)


@triton.jit
def _finalize_valid_rows_kernel(
    gemm2_ptr,  # [padded_rows, gemm2_stride] bf16 FC2 rows in permuted order
    idx_ptr,  # [rows * TOPK] int32: (token, k) -> permuted row, -1 when the expert is not local
    probs_ptr,  # [rows, TOPK] fp32 routing weights
    out_ptr,  # [rows, hidden] output; rows [0, valid_tokens) are fully written
    valid_tokens_ptr,  # scalar int32 CUDA tensor
    hidden,  # logical hidden size
    gemm2_stride,  # row stride of gemm2_ptr (padded hidden size)
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,  # persistent grid size
):
    """Gather and weight each valid token's local-expert rows into its output row."""
    pid = tl.program_id(0)
    valid_tokens = tl.load(valid_tokens_ptr)
    for row in tl.range(pid, valid_tokens, NUM_BLOCKS):
        row64 = tl.cast(row, tl.int64)
        for h in tl.range(0, hidden, BLOCK_H):
            offs = h + tl.arange(0, BLOCK_H)
            col_mask = offs < hidden
            acc = tl.zeros([BLOCK_H], dtype=tl.float32)
            for k in tl.static_range(TOPK):
                src = tl.load(idx_ptr + row * TOPK + k)
                is_local = src >= 0
                prob = tl.load(probs_ptr + row * TOPK + k)
                prob = tl.where(is_local, prob, 0.0)
                src64 = tl.where(is_local, src, 0).to(tl.int64)
                values = tl.load(
                    gemm2_ptr + src64 * gemm2_stride + offs, mask=col_mask & is_local, other=0.0
                ).to(tl.float32)
                acc += values * prob
            tl.store(out_ptr + row64 * hidden + offs, acc, mask=col_mask)


def _require_triton() -> None:
    if not HAVE_TRITON:
        raise RuntimeError("FlashInfer routed MXFP8 MoE requires triton for its row kernels.")


def quantize_routed_mxfp8_rows(
    hidden_states: torch.Tensor, valid_tokens: torch.Tensor, padded_hidden_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize rows [0, valid_tokens) of `hidden_states` for TRT-LLM routed MXFP8 MoE.

    Returns row-major e4m3 data of shape [rows, padded_hidden_size] and uint8 e8m0 scales of
    shape [rows, padded_hidden_size // 32] in the kernel's linear layout.
    Padding columns of the valid rows are zero; rows at or beyond valid_tokens are unspecified.
    """
    _require_triton()
    rows, hidden_size = hidden_states.shape
    if hidden_size % MXFP8_BLOCK_SIZE:
        raise ValueError(f"hidden size must be divisible by {MXFP8_BLOCK_SIZE}; got {hidden_size}")
    if padded_hidden_size < hidden_size or padded_hidden_size % 128:
        raise ValueError(
            f"padded hidden size {padded_hidden_size} must be a multiple of 128 that is at least "
            f"the hidden size {hidden_size}"
        )
    if not hidden_states.is_contiguous():
        raise ValueError("hidden_states must be contiguous")
    data = torch.empty(
        rows, padded_hidden_size, dtype=torch.float8_e4m3fn, device=hidden_states.device
    )
    scale = torch.empty(
        rows, padded_hidden_size // MXFP8_BLOCK_SIZE, dtype=torch.uint8, device=hidden_states.device
    )
    block_k = triton.next_power_of_2(padded_hidden_size)
    num_blocks = min(rows, _QUANTIZE_CTAS)
    _quantize_valid_rows_kernel[(num_blocks,)](
        hidden_states,
        data,
        scale,
        valid_tokens,
        hidden_size,
        padded_hidden_size,
        BLOCK_K=block_k,
        BLOCK_GROUPS=block_k // MXFP8_BLOCK_SIZE,
        NUM_BLOCKS=num_blocks,
        num_warps=8,
    )
    return data, scale


def pack_routed_mxfp8_routing(
    routing_map: torch.Tensor, probs: torch.Tensor, valid_tokens: torch.Tensor
) -> torch.Tensor:
    """Pack expert ids and BF16 routing weights as `(expert_id << 16) | weight_bits` int32.

    Every slot of rows at or beyond valid_tokens is packed as expert -1,
    so the TRT-LLM routing kernel skips them regardless of the stale contents of the buffers.
    """
    _require_triton()
    rows, topk = routing_map.shape
    if probs.shape != routing_map.shape:
        raise ValueError(
            f"probs shape {tuple(probs.shape)} must match routing_map shape "
            f"{tuple(routing_map.shape)}"
        )
    if not (routing_map.is_contiguous() and probs.is_contiguous()):
        raise ValueError("routing_map and probs must be contiguous")
    packed = torch.empty(rows, topk, dtype=torch.int32, device=routing_map.device)
    num_entries = rows * topk
    block = 1024
    _pack_routing_kernel[(triton.cdiv(num_entries, block),)](
        routing_map, probs, packed, valid_tokens, num_entries, TOPK=topk, BLOCK=block
    )
    return packed


def finalize_routed_mxfp8_rows(
    gemm2_output: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    probs: torch.Tensor,
    valid_tokens: torch.Tensor,
    hidden_size: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combine the unfinalized FC2 rows of rows [0, valid_tokens) into `out`.

    `out` defaults to a new fp32 [rows, hidden_size] tensor;
    a symmetric reduce-scatter buffer with at least `rows` can be passed to skip the extra copy.
    Rows at or beyond valid_tokens are not written.
    """
    _require_triton()
    rows, topk = probs.shape
    if expanded_idx_to_permuted_idx.numel() != rows * topk:
        raise ValueError(
            f"expanded_idx_to_permuted_idx has {expanded_idx_to_permuted_idx.numel()} entries; "
            f"expected rows x top_k = {rows * topk}"
        )
    if gemm2_output.ndim != 2 or gemm2_output.shape[1] < hidden_size:
        raise ValueError(
            f"gemm2_output shape {tuple(gemm2_output.shape)} cannot provide {hidden_size} columns"
        )
    if out is None:
        out = torch.empty(rows, hidden_size, dtype=torch.float32, device=probs.device)
    if out.ndim != 2 or out.shape[0] < rows or out.shape[1] != hidden_size:
        raise ValueError(
            f"output buffer shape {tuple(out.shape)} cannot hold {rows} rows of {hidden_size}"
        )
    for name, tensor in (
        ("gemm2_output", gemm2_output),
        ("expanded_idx_to_permuted_idx", expanded_idx_to_permuted_idx),
        ("probs", probs),
        ("out", out),
    ):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    num_blocks = min(rows, _COMBINE_CTAS)
    _finalize_valid_rows_kernel[(num_blocks,)](
        gemm2_output,
        expanded_idx_to_permuted_idx,
        probs,
        out,
        valid_tokens,
        hidden_size,
        gemm2_output.shape[1],
        TOPK=topk,
        BLOCK_H=min(triton.next_power_of_2(hidden_size), 1024),
        NUM_BLOCKS=num_blocks,
    )
    return out


def flashinfer_routed_mxfp8_moe_prequantized(
    quantized_hidden: torch.Tensor,
    hidden_scale: torch.Tensor,
    packed_routing: torch.Tensor,
    fc1_weight: FlashInferRoutedMXFP8Weight,
    fc2_weight: FlashInferRoutedMXFP8Weight,
    *,
    num_experts: int,
    local_expert_offset: int,
    top_k: int,
    activation_type: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch routed MXFP8 MoE with prequantized input and packed routing."""
    require_flashinfer_routed_mxfp8()
    result = trtllm_fp8_block_scale_routed_moe(
        topk_ids=packed_routing,
        routing_bias=None,
        hidden_states=quantized_hidden,
        hidden_states_scale=hidden_scale,
        gemm1_weights=fc1_weight.data,
        gemm1_weights_scale=fc1_weight.scale,
        gemm2_weights=fc2_weight.data,
        gemm2_weights_scale=fc2_weight.scale,
        num_experts=num_experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=fc1_weight.padded_rows,
        local_expert_offset=local_expert_offset,
        local_num_experts=fc1_weight.data.shape[0],
        routed_scaling_factor=None,
        # Packed precomputed routing consumes these weights as-is; no routing
        # softmax or top-k renormalization is applied.
        routing_method_type=0,
        use_shuffled_weight=True,
        weight_layout=WeightLayout.MajorK.value,
        do_finalize=False,
        tune_max_num_tokens=max(quantized_hidden.shape[0], 128),
        fp8_quantization_type=Fp8QuantizationType.MxFp8,
        activation_type=activation_type,
    )
    # do_finalize=False returns [gemm2_output, expert_weights, expanded_idx_to_permuted_idx].
    if not isinstance(result, (list, tuple)) or len(result) < 3:
        raise RuntimeError(
            "trtllm_fp8_block_scale_routed_moe(do_finalize=False) must return "
            "[gemm2_output, expert_weights, expanded_idx_to_permuted_idx]"
        )
    return result[0], result[2]


def flashinfer_routed_mxfp8_moe(
    hidden_states: torch.Tensor,
    routing_map: torch.Tensor,
    probabilities: torch.Tensor,
    fc1_weight: FlashInferRoutedMXFP8Weight,
    fc2_weight: FlashInferRoutedMXFP8Weight,
    *,
    num_experts: int,
    local_expert_offset: int,
    activation_type: int,
    valid_tokens: torch.Tensor | None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the FlashInfer TRT-LLM routed MXFP8 MoE kernel on a fixed-size token buffer.

    Every per-row stage outside the kernel's grouped GEMMs (input quantization, routing
    packing, and the weighted combine of FC2 rows) is a Triton kernel gated by `valid_tokens`,
    the device-side count the inference dispatcher updates each step. The host-shaped kernel
    call itself is then sized by the buffer, which its routing stage scans with -1 experts for
    every row beyond valid_tokens, while its GEMMs exit early on the device-side row counts.
    This keeps CUDA graph replay at the cost of the live tokens without a host-side bound.

    Args:
        hidden_states: [rows, hidden_size] bf16 gathered tokens; only [0, valid_tokens) are live.
        routing_map: [rows, top_k] expert ids; -1 marks an unrouted slot.
        probabilities: [rows, top_k] fp32 routing weights.
        fc1_weight, fc2_weight: prepared TRT-LLM Major-K expert weights.
        num_experts: global expert count.
        local_expert_offset: first global expert id owned by this rank.
        activation_type: FlashInfer ActivationType value.
        valid_tokens: scalar int32 CUDA tensor with the live row count; None -> every row is live.
        out: optional [>= rows, hidden_size] buffer, e.g. the fp32 reduce-scatter symmetric buffer.

    Returns:
        `out` when given, else a new fp32 [rows, hidden_size] tensor. Rows at or beyond
        valid_tokens hold unspecified values.
    """
    if hidden_states.shape[-1] != fc1_weight.logical_cols:
        raise ValueError(
            f"hidden size {hidden_states.shape[-1]} does not match FC1 K "
            f"{fc1_weight.logical_cols}"
        )
    if fc2_weight.logical_rows != fc1_weight.logical_cols:
        raise ValueError("FC2 output size must match the model hidden size")
    if fc2_weight.logical_cols != fc1_weight.logical_rows:
        raise ValueError("FC2 K size must match the FC1 output size")
    if probabilities.dtype != torch.float32:
        raise TypeError(
            f"FlashInfer routed MoE requires FP32 probabilities; got {probabilities.dtype}"
        )
    rows = hidden_states.shape[0]
    if valid_tokens is None:
        valid_tokens = torch.full((1,), rows, dtype=torch.int32, device=hidden_states.device)

    quantized_hidden, hidden_scale = quantize_routed_mxfp8_rows(
        hidden_states, valid_tokens, fc1_weight.padded_cols
    )
    packed_routing = pack_routed_mxfp8_routing(routing_map, probabilities, valid_tokens)
    if HAVE_FLASHINFER_ACTIVE_TOKENS and out is not None and out.shape[0] == rows:
        # The kernel finalizes in place over the live rows and writes fp32;
        # routing weights are the bf16 ones carried in the packed ids,
        # so this differs from the Triton combine's fp32 weights within bf16 rounding.
        trtllm_fp8_block_scale_routed_moe(
            topk_ids=packed_routing,
            routing_bias=None,
            hidden_states=quantized_hidden,
            hidden_states_scale=hidden_scale,
            gemm1_weights=fc1_weight.data,
            gemm1_weights_scale=fc1_weight.scale,
            gemm2_weights=fc2_weight.data,
            gemm2_weights_scale=fc2_weight.scale,
            num_experts=num_experts,
            top_k=routing_map.shape[1],
            n_group=None,
            topk_group=None,
            intermediate_size=fc1_weight.padded_rows,
            local_expert_offset=local_expert_offset,
            local_num_experts=fc1_weight.data.shape[0],
            routed_scaling_factor=None,
            routing_method_type=0,
            use_shuffled_weight=True,
            weight_layout=WeightLayout.MajorK.value,
            do_finalize=True,
            output=out,
            num_active_tokens=valid_tokens,
            tune_max_num_tokens=max(rows, 128),
            fp8_quantization_type=Fp8QuantizationType.MxFp8,
            activation_type=activation_type,
        )
        return out
    gemm2_output, expanded_idx_to_permuted_idx = flashinfer_routed_mxfp8_moe_prequantized(
        quantized_hidden,
        hidden_scale,
        packed_routing,
        fc1_weight,
        fc2_weight,
        num_experts=num_experts,
        local_expert_offset=local_expert_offset,
        top_k=routing_map.shape[1],
        activation_type=activation_type,
    )
    return finalize_routed_mxfp8_rows(
        gemm2_output,
        expanded_idx_to_permuted_idx,
        probabilities,
        valid_tokens,
        fc2_weight.logical_rows,
        out=out,
    )
