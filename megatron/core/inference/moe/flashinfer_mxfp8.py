# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""FlashInfer routed-MoE support for MCore MXFP8 expert weights."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

try:
    from flashinfer import mxfp8_quantize, shuffle_matrix_a
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

from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

logger = logging.getLogger(__name__)
_LOGGED_TOKEN_POLICIES: set[tuple[str, int, int | None, int]] = set()


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


def pack_routed_mxfp8_routing(
    expert_ids: torch.Tensor, probabilities: torch.Tensor
) -> torch.Tensor:
    """Pack global expert IDs and BF16 routing weights for TRT-LLM MoE."""
    probability_bits = (
        probabilities.to(torch.bfloat16).contiguous().view(torch.int16).to(torch.int32)
    )
    return (expert_ids.to(torch.int32) << 16) | (probability_bits & 0xFFFF)


def _unwrap_output(output) -> torch.Tensor:
    """Normalize FlashInfer's 0.6.x tensor/list return variants."""
    if isinstance(output, (list, tuple)):
        return output[0]
    return output


def quantize_routed_mxfp8_input(
    hidden_states: torch.Tensor, padded_hidden_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad and quantize routed-MoE input, returning a 2D uint8 scale matrix."""
    require_flashinfer_routed_mxfp8()
    hidden_padding = padded_hidden_size - hidden_states.shape[-1]
    if hidden_padding < 0:
        raise ValueError(
            f"padded hidden size {padded_hidden_size} is smaller than input hidden size "
            f"{hidden_states.shape[-1]}"
        )
    padded_hidden = F.pad(hidden_states, (0, hidden_padding)) if hidden_padding else hidden_states
    quantized_hidden, hidden_scale = mxfp8_quantize(padded_hidden, False)
    hidden_scale = (
        hidden_scale.contiguous()
        .view(torch.uint8)
        .reshape(hidden_states.shape[0], padded_hidden.shape[1] // 32)
    )
    return quantized_hidden, hidden_scale


def select_routed_mxfp8_active_rows(
    full_rows: int,
    *,
    token_capacity: int | None,
    decode_only: bool,
    decode_token_upper_bound: int | None,
) -> tuple[int, str]:
    """Select the graph-stable row count for one routed-MoE invocation.

    A bounded prefix is safe only for a decode-only graph with a host-known
    upper bound no larger than the configured capacity. Mixed and prefill
    graphs retain the full dispatcher buffer so prompt tokens are never
    truncated.
    """
    if full_rows <= 0:
        raise ValueError(f"full_rows must be positive; got {full_rows}")
    if token_capacity is None:
        return full_rows, "full"
    if token_capacity <= 0:
        raise ValueError(f"token_capacity must be positive; got {token_capacity}")
    if decode_token_upper_bound is not None and decode_token_upper_bound <= 0:
        raise ValueError(
            "decode_token_upper_bound must be positive; " f"got {decode_token_upper_bound}"
        )
    if (
        decode_only
        and decode_token_upper_bound is not None
        and decode_token_upper_bound <= token_capacity
    ):
        return min(token_capacity, full_rows), "bounded-decode"
    if decode_only:
        return full_rows, "full-decode-over-capacity"
    return full_rows, "full-mixed"


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
) -> torch.Tensor:
    """Launch routed MXFP8 MoE with prequantized input and packed routing."""
    require_flashinfer_routed_mxfp8()
    output = trtllm_fp8_block_scale_routed_moe(
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
        routing_method_type=0,
        use_shuffled_weight=True,
        weight_layout=WeightLayout.MajorK.value,
        do_finalize=True,
        tune_max_num_tokens=max(quantized_hidden.shape[0], 128),
        fp8_quantization_type=Fp8QuantizationType.MxFp8,
        activation_type=activation_type,
    )
    return _unwrap_output(output)[:, : fc2_weight.logical_rows]


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
    out: torch.Tensor | None = None,
    token_capacity: int | None = None,
    decode_only: bool = False,
    decode_token_upper_bound: int | None = None,
) -> torch.Tensor:
    """Run the FlashInfer TRT-LLM routed MXFP8 MoE kernel.

    When ``token_capacity`` is set, only that fixed prefix is processed during
    decode-only steps whose EP-wide token upper bound fits within the capacity.
    Prefill, mixed steps, and unsafe decode configurations process the full input.
    Invalid rows in the bounded prefix must already have expert ID -1.

    The row choice is made while each CUDA graph is built, so graph replay sees
    fixed shapes and addresses.
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

    full_rows = hidden_states.shape[0]
    active_rows, policy = select_routed_mxfp8_active_rows(
        full_rows,
        token_capacity=token_capacity,
        decode_only=decode_only,
        decode_token_upper_bound=decode_token_upper_bound,
    )
    if token_capacity is not None:
        policy_key = (policy, token_capacity, decode_token_upper_bound, full_rows)
        if policy_key not in _LOGGED_TOKEN_POLICIES:
            _LOGGED_TOKEN_POLICIES.add(policy_key)
            logger.warning(
                "FlashInfer MXFP8 token policy: %s active_rows=%d full_rows=%d "
                "configured_capacity=%d decode_upper_bound=%s",
                policy,
                active_rows,
                full_rows,
                token_capacity,
                decode_token_upper_bound,
            )

    selected_hidden_states = hidden_states[:active_rows]
    selected_routing_map = routing_map[:active_rows]
    selected_probabilities = probabilities[:active_rows]
    quantized_hidden, hidden_scale = quantize_routed_mxfp8_input(
        selected_hidden_states, fc1_weight.padded_cols
    )
    packed_routing = pack_routed_mxfp8_routing(selected_routing_map, selected_probabilities)
    output = flashinfer_routed_mxfp8_moe_prequantized(
        quantized_hidden,
        hidden_scale,
        packed_routing,
        fc1_weight,
        fc2_weight,
        num_experts=num_experts,
        local_expert_offset=local_expert_offset,
        top_k=selected_routing_map.shape[1],
        activation_type=activation_type,
    )

    if out is not None:
        if out.shape[0] < output.shape[0] or out.shape[1:] != output.shape[1:]:
            raise ValueError(
                f"output buffer shape {tuple(out.shape)} cannot hold {tuple(output.shape)}"
            )
        out[: output.shape[0]].copy_(output)
        return out
    return output
