# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fused MoE: permute -> FC1 -> activation -> FC2 -> unpermute.

Supports BF16, MCore MXFP8, and native Transformer Engine MXFP8 weights.
All permutation logic is handled internally — callers invoke a single function.
"""

from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, Optional

import torch

from megatron.core.inference.moe.activations import (
    padded_squared_relu,
    padded_swiglu,
    squared_relu_and_quantize_mxfp8,
)
from megatron.core.inference.moe.permute import (
    permute_and_quantize_mxfp8,
    permute_tokens,
    unpermute_tokens,
)
from megatron.core.inference.quantization.mxfp8_quantize import MXFP8_SCALE_ROW_BLOCK
from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

from . import batch_invariant

try:
    from torch.nn.functional import grouped_mm

    HAVE_GROUPED_MM = True
except ImportError:
    # Fallback to the private symbol for torch versions < 2.10.
    grouped_mm = getattr(torch, "_grouped_mm", None)
    HAVE_GROUPED_MM = grouped_mm is not None

try:
    from torch.nn.functional import ScalingType, SwizzleType, scaled_grouped_mm

    HAVE_SCALED_GMM = True
except ImportError:
    HAVE_SCALED_GMM = False

try:
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.cpp_extensions.gemm import (
        general_grouped_gemm_for_grouped_tensor,
    )
    from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor as TEGroupedTensor
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer as TEMXFP8Quantizer
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor as TEMXFP8Tensor

    HAVE_TE_GROUPED_MXFP8 = all(
        hasattr(tex, name)
        for name in (
            "group_quantize",
            "te_general_grouped_gemm_for_discrete_in",
            "te_general_grouped_gemm_for_grouped_tensor",
        )
    )
except (ImportError, AttributeError):
    tex = None
    TEGroupedTensor = ()
    TEMXFP8Quantizer = ()
    TEMXFP8Tensor = ()
    general_grouped_gemm_for_grouped_tensor = None
    HAVE_TE_GROUPED_MXFP8 = False


_TE_MXFP8_ACTIVATION_QUANTIZER = None


class ActivationType(Enum):
    """Activation functions supported by mcore_fused_moe."""

    SQUARED_RELU = "squared_relu"
    SWIGLU = "swiglu"


def _bf16_grouped_mm(
    x_bf16: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    """BF16 grouped GEMM using torch.nn.functional.grouped_mm."""
    assert x_bf16.dtype == torch.bfloat16, f"Expected bf16 input, got {x_bf16.dtype}"
    return grouped_mm(x_bf16, weight.transpose(1, 2), offs=offs)


def _mxfp8_grouped_mm(act: MXFP8Tensor, weight: MXFP8Tensor, offs: torch.Tensor) -> torch.Tensor:
    """MXFP8 scaled_grouped_mm with pre-quantized activations and weights."""
    return scaled_grouped_mm(
        act.data,
        weight.data.transpose(1, 2),
        act.scale_2d(),
        ScalingType.BlockWise1x32,
        weight.scale,
        ScalingType.BlockWise1x32,
        swizzle_a=SwizzleType.SWIZZLE_32_4_4,
        swizzle_b=SwizzleType.SWIZZLE_32_4_4,
        offs=offs,
        output_dtype=torch.bfloat16,
    )


def _unwrap_te_mxfp8_weight(weight):
    """Return a native TE MXFP8/GroupedTensor payload, if present."""
    if isinstance(weight, TEMXFP8Tensor):
        return weight
    if isinstance(weight, TEGroupedTensor) and isinstance(weight.quantizer, TEMXFP8Quantizer):
        return weight
    data = getattr(weight, "data", None)
    if isinstance(data, TEMXFP8Tensor):
        return data
    if isinstance(data, TEGroupedTensor) and isinstance(data.quantizer, TEMXFP8Quantizer):
        return data
    return None


def is_te_mxfp8_weight(weight: object) -> bool:
    """Whether weight uses TE's native MXFP8 representation.

    Both TE's discrete per-expert weights and its single GroupedTensor parameter
    representation are accepted by the device-metadata grouped GEMM API.
    """
    if isinstance(weight, (list, tuple)):
        return bool(weight) and all(_unwrap_te_mxfp8_weight(item) is not None for item in weight)
    return _unwrap_te_mxfp8_weight(weight) is not None


def _get_te_mxfp8_activation_quantizer():
    """Create the row-wise TE MXFP8 quantizer used by grouped activations."""
    global _TE_MXFP8_ACTIVATION_QUANTIZER
    if _TE_MXFP8_ACTIVATION_QUANTIZER is None:
        quantizer = TEMXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)
        quantizer.optimize_for_gemm = True
        _TE_MXFP8_ACTIVATION_QUANTIZER = quantizer
    return _TE_MXFP8_ACTIVATION_QUANTIZER


def _normalize_te_mxfp8_weight(weight):
    """Strip Parameter wrappers while preserving discrete/grouped TE layout."""
    if isinstance(weight, (list, tuple)):
        normalized = [_unwrap_te_mxfp8_weight(item) for item in weight]
        if any(item is None for item in normalized):
            raise TypeError("All discrete expert weights must be native TE MXFP8 tensors.")
        return normalized
    normalized = _unwrap_te_mxfp8_weight(weight)
    if normalized is None:
        raise TypeError(f"Expected native TE MXFP8 weight, got {type(weight).__name__}.")
    if isinstance(normalized, TEMXFP8Tensor):
        # A lone discrete tensor is the one-expert form of the discrete API.
        return [normalized]
    return normalized


def _te_weight_out_features(normalized) -> int:
    """Get the common GEMM output width of a normalized TE weight without device metadata."""
    if isinstance(normalized, list):
        out_features = normalized[0].shape[-2]
        if any(item.shape[-2] != out_features for item in normalized[1:]):
            raise ValueError("All expert weights must have the same output width.")
        return out_features
    if normalized.ndim >= 3:
        return normalized.shape[-2]
    shapes = getattr(normalized, "tensor_shapes", None)
    if shapes:
        return shapes[0][-2]
    raise ValueError("Unable to infer the output width of the TE grouped expert weight.")


def _te_mxfp8_grouped_mm(x_bf16: torch.Tensor, weight, first_dims: torch.Tensor) -> torch.Tensor:
    """TE MXFP8 grouped GEMM initialized entirely from CUDA split metadata."""
    assert HAVE_TE_GROUPED_MXFP8, (
        "Native TE MXFP8 grouped GEMM requires Transformer Engine's device-metadata "
        "group_quantize and grouped-GEMM APIs."
    )
    assert x_bf16.dtype == torch.bfloat16, f"Expected bf16 input, got {x_bf16.dtype}"
    assert x_bf16.is_cuda and first_dims.is_cuda, "TE grouped MXFP8 requires CUDA tensors."
    assert first_dims.dtype == torch.int64, f"Expected int64 expert splits, got {first_dims.dtype}"

    normalized_weight = _normalize_te_mxfp8_weight(weight)
    num_experts = first_dims.numel()
    weight_experts = (
        len(normalized_weight)
        if isinstance(normalized_weight, list)
        else normalized_weight.num_tensors
    )
    if weight_experts != num_experts:
        raise ValueError(
            f"Expert split count ({num_experts}) does not match weight count ({weight_experts})."
        )

    grouped_input = tex.group_quantize(
        x_bf16, _get_te_mxfp8_activation_quantizer(), num_experts, first_dims
    )
    out_features = _te_weight_out_features(normalized_weight)
    # tensor_offsets holds each expert's output start in the flat storage. Build it
    # with CUDA ops so changing routing counts does not synchronize or invalidate a graph.
    tensor_offsets = torch.cat(
        (first_dims.new_zeros(1), torch.cumsum(first_dims[:-1] * out_features, dim=0))
    )
    output_data = torch.empty(
        x_bf16.shape[0] * out_features, dtype=torch.bfloat16, device=x_bf16.device
    )
    grouped_output = TEGroupedTensor(
        shape=(x_bf16.shape[0], out_features),
        dtype=torch.bfloat16,
        num_tensors=num_experts,
        shapes=None,
        quantizer=None,
        data=output_data,
        first_dims=first_dims,
        tensor_offsets=tensor_offsets,
        requires_grad=False,
    )
    general_grouped_gemm_for_grouped_tensor(
        normalized_weight, grouped_input, grouped_output, layout="TN"
    )
    return grouped_output.rowwise_data.view(x_bf16.shape[0], out_features)


def _get_activation_func(
    activation_type: ActivationType,
    fused_quant: bool = False,
    activation_kwargs: Optional[Dict[str, Any]] = None,
) -> Callable:
    """Resolve ActivationType enum to a concrete kernel.

    Args:
        activation_type: which activation the kernel should implement.
        fused_quant: if True, return the fused activation + MXFP8 quantize kernel.
        activation_kwargs: activation-specific options, extracted per activation below.
            ``clamp_scale`` (squared ReLU): soft-clamp the pre-activation first.

    Returns:
        The kernel, partially applied with whichever options the activation consumes.
    """
    activation_kwargs = activation_kwargs or {}
    if activation_type == ActivationType.SQUARED_RELU:
        clamp_scale = activation_kwargs.get("clamp_scale")
        func = squared_relu_and_quantize_mxfp8 if fused_quant else padded_squared_relu
        return func if clamp_scale is None else partial(func, clamp_scale=clamp_scale)
    elif activation_type == ActivationType.SWIGLU:
        if fused_quant:
            raise NotImplementedError("SWIGLU + MXFP8 fused-quant not implemented (bf16 only)")
        if activation_kwargs.get("clamp_scale") is not None:
            raise NotImplementedError(
                "activation_func_tanh_clamp_scale is only implemented for squared ReLU here; "
                "the gated form (SiTU-GLU) has no inference kernel yet."
            )
        return padded_swiglu
    else:
        raise ValueError(f"Unsupported activation type: {activation_type}")


def mcore_fused_moe(
    hidden_states: torch.Tensor,
    probs: torch.Tensor,
    fc1_weight,
    fc2_weight,
    activation_type: ActivationType,
    num_local_experts: int,
    local_expert_start: int,
    valid_tokens: torch.Tensor,
    routing_map: torch.Tensor,
    disable_fused_quant_kernels: bool = False,
    out: torch.Tensor = None,
    activation_clamp_scale: Optional[float] = None,
) -> torch.Tensor:
    """Fused MoE: permute -> pad -> FC1 -> activation -> FC2 -> unpad -> unpermute.

    MCore MXFP8 weights use fused Triton permute/activation + quantization kernels
    unless disable_fused_quant_kernels=True. Native TE MXFP8 weights use TE's
    device-metadata grouped quantization and grouped GEMM kernels; their expert
    segments are zero-padded to 256 rows.

    Args:
        hidden_states: [max_tokens, hidden_size] BF16 input. max_tokens =
            max_local_tokens * ep_size; only the first valid_tokens rows are valid.
        probs: [max_tokens, topk] routing probabilities.
        fc1_weight: stacked BF16/MCore MXFP8 weight, a list of native TE MXFP8
            expert weights, or a native TE MXFP8 GroupedTensor.
        fc2_weight: weight for FC2 (same representation as fc1_weight).
        activation_type: ActivationType enum (SQUARED_RELU).
        num_local_experts: number of experts on this rank.
        local_expert_start: first global expert index on this rank.
        valid_tokens: scalar int32 CUDA tensor holding the number of valid tokens this
            iteration. Kernels use this to ignore rows beyond the valid prefix — required
            for CUDA graph compatibility since hidden_states is always max-sized.
        routing_map: [max_tokens, topk] int expert assignments.
        disable_fused_quant_kernels: if True, disable fused permute+quantize and
            activation+quantize kernels for MCore MXFP8, using separate launches
            instead. Useful for debugging. Ignored for BF16 and native TE MXFP8.
        out: optional pre-allocated output buffer. If provided, unpermute writes
            directly into this tensor (e.g. the RSV symmetric buffer), avoiding a
            separate copy before reduce-scatter.
        activation_clamp_scale: config.activation_func_tanh_clamp_scale. When set, the
            squared-ReLU pre-activation is soft-clamped with ``s * tanh(x / s)`` before the
            square, bounding the activation output by ``s ** 2``.

    Returns:
        [max_tokens, hidden_size] BF16 output. Only the first valid_tokens rows are
        meaningful; rows beyond that are undefined.
    """
    assert (
        hidden_states.dtype == torch.bfloat16
    ), f"mcore_fused_moe requires bf16 input, got {hidden_states.dtype}"

    max_tokens = hidden_states.shape[0]
    use_mcore_mxfp8 = isinstance(fc1_weight, MXFP8Tensor)
    use_te_mxfp8 = is_te_mxfp8_weight(fc1_weight)
    use_mxfp8 = use_mcore_mxfp8 or use_te_mxfp8
    if use_te_mxfp8 != is_te_mxfp8_weight(fc2_weight):
        raise TypeError("FC1 and FC2 must either both use native TE MXFP8 weights or neither.")
    # Fused Triton quant kernels only apply to the MCore MXFP8 path.
    use_fused_quant = use_mcore_mxfp8 and not disable_fused_quant_kernels
    batch_invariant_mode = batch_invariant.enabled()

    if batch_invariant_mode:
        # The MXFP8 path uses scaled_grouped_mm and is not batch invariant.
        assert not use_mxfp8, (
            "batch_invariant_mode requires the bf16 grouped GEMM path; got "
            "MXFP8 weights. Disable mxfp8 or batch_invariant_mode."
        )
        mm_fn = batch_invariant.grouped_mm
        expert_alignment = batch_invariant.grouped_mm_alignment()
    elif use_te_mxfp8:
        assert HAVE_TE_GROUPED_MXFP8, (
            "Native TE MXFP8 grouped GEMM requires Transformer Engine's device-metadata "
            "group_quantize and grouped-GEMM APIs."
        )
        mm_fn = _te_mxfp8_grouped_mm
        # TE's device-initialized MXFP8 grouped kernels currently require every
        # non-empty expert segment to be a multiple of 256 rows.
        expert_alignment = 256
    elif use_mcore_mxfp8:
        assert (
            HAVE_SCALED_GMM
        ), "torch.nn.functional.scaled_grouped_mm not available. Install PyTorch 2.10+."
        mm_fn = _mxfp8_grouped_mm
        # scaled_grouped_mm requires each expert's token count aligned to 32,
        # but swizzled MXFP8 scales require alignment to 128. Use 128 to
        # satisfy both constraints.
        expert_alignment = 128
    else:
        assert (
            HAVE_GROUPED_MM
        ), "torch.nn.functional.grouped_mm not available. Install PyTorch 2.10+."
        mm_fn = _bf16_grouped_mm
        expert_alignment = 16

    activation_func = _get_activation_func(
        activation_type,
        fused_quant=use_fused_quant,
        activation_kwargs={"clamp_scale": activation_clamp_scale},
    )

    # --- Pre-processing: permute ---
    if use_fused_quant:
        # Fused permute + MXFP8 quantize: single kernel produces MXFP8Tensor
        batch_invariant_inverse_map = None
        hidden_states, permuted_probs, permutation_map, offs = permute_and_quantize_mxfp8(
            hidden_states,
            probs,
            routing_map,
            local_expert_start,
            num_local_experts,
            valid_tokens,
            alignment=expert_alignment,
        )
    else:
        permuted = permute_tokens(
            hidden_states,
            probs,
            routing_map,
            local_expert_start,
            num_local_experts,
            valid_tokens,
            alignment=expert_alignment,
            row_alignment=MXFP8_SCALE_ROW_BLOCK if use_mxfp8 else 1,
            zero_padding=use_te_mxfp8,
            return_batch_invariant_inverse_map=batch_invariant_mode,
        )
        hidden_states, permuted_probs, permutation_map, offs = permuted[:4]
        # Maps each (token, local expert) pair to its row in the expert-grouped buffer,
        # allowing batch-invariant unpermute to read contributions in fixed expert order.
        batch_invariant_inverse_map = permuted[4] if batch_invariant_mode else None

    # --- FC1 -> activation -> FC2 ---
    # Quantize if MXFP8 path and hidden_states not already quantized (fused permute+quant
    # produces MXFP8Tensor directly).
    if use_mcore_mxfp8 and not isinstance(hidden_states, MXFP8Tensor):
        hidden_states = MXFP8Tensor.from_bf16(hidden_states, backend="triton")
    if use_te_mxfp8:
        first_dims = torch.cat((offs[:1], offs[1:] - offs[:-1])).to(torch.int64)
        fc1_output = mm_fn(hidden_states, fc1_weight, first_dims)
    else:
        fc1_output = mm_fn(hidden_states, fc1_weight, offs)

    # offs[-1:] is a 1-element view pointing to inclusive_expert_offsets[-1] — the total
    # number of rows actually used by experts this iteration (valid tokens + alignment
    # padding within expert blocks). Passed to activation and unpermute to skip unused rows.
    n_used = offs[-1:]
    if batch_invariant_mode:
        # Match training: BF16 activation, FP32 probability multiply, then BF16 before FC2.
        if activation_type == ActivationType.SWIGLU:
            assert activation_clamp_scale is None, (
                "activation_func_tanh_clamp_scale is only implemented for squared ReLU here; "
                "the gated form (SiTU-GLU) has no inference kernel yet."
            )
            activation_out = batch_invariant.swiglu_with_probs(
                fc1_output, permutation_map, n_used, permuted_probs
            )
        else:
            activation_out = batch_invariant.squared_relu_with_probs(
                fc1_output, permutation_map, n_used, permuted_probs, activation_clamp_scale
            )
    else:
        if use_te_mxfp8:
            activation_out = activation_func(fc1_output, permutation_map, n_used, zero_padding=True)
        else:
            activation_out = activation_func(fc1_output, permutation_map, n_used)
    # Fused activation+quant returns MXFP8Tensor; otherwise quantize separately.
    if use_mcore_mxfp8 and not isinstance(activation_out, MXFP8Tensor):
        activation_out = MXFP8Tensor.from_bf16(activation_out, backend="triton")
    if use_te_mxfp8:
        fc2_output = mm_fn(activation_out, fc2_weight, first_dims)
    else:
        fc2_output = mm_fn(activation_out, fc2_weight, offs)

    # --- Post-processing: unpermute ---
    return unpermute_tokens(
        fc2_output,
        None if batch_invariant_mode else permuted_probs,
        permutation_map,
        max_tokens,
        n_used,
        valid_tokens,
        out=out,
        batch_invariant_inverse_map=batch_invariant_inverse_map,
    )
