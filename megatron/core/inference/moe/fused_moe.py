# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fused MoE: permute -> FC1 -> activation -> FC2 -> unpermute.

Supports BF16, MCore MXFP8, and native Transformer Engine MXFP8 weights.
All permutation logic is handled internally — callers invoke a single function.
"""

import os
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
from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
    get_unrestricted_te_workspace_size_bytes,
)

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
        get_grouped_gemm_setup_workspace_size,
    )
    from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor as TEGroupedTensor
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer as TEMXFP8Quantizer
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor as TEMXFP8Tensor
    from transformer_engine.pytorch.utils import get_sm_count as _get_te_sm_count

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
    get_grouped_gemm_setup_workspace_size = None
    _get_te_sm_count = None
    HAVE_TE_GROUPED_MXFP8 = False


_TE_MXFP8_ACTIVATION_QUANTIZER = None
_TE_MXFP8_BATCH_INVARIANT_CHUNK_SIZE = 256
_TE_MXFP8_BATCH_INVARIANT_WEIGHT_CACHE = "_mcore_batch_invariant_gemm_weight"


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


def _te_mxfp8_batch_invariant_grouped_gemm(
    weight,
    grouped_input,
    grouped_output,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    workspace_setup: torch.Tensor,
    workspace_cublas: torch.Tensor,
) -> None:
    """Launch grouped MXFP8 GEMM without inheriting te_native's starved workspace.

    The te_native batch-invariant backend restricts ordinary TE GEMMs to a 1 KiB
    workspace to disqualify split-K algorithms. TE's device-metadata grouped
    MXFP8 kernel requires its normal cuBLASLt workspace even when every GEMM has
    a fixed M. Recover the unrestricted size and call the device-metadata API
    directly for this fixed-shape path only.
    """
    sm_count = _get_te_sm_count()
    sm_count -= int(os.getenv("NVTE_EXT_MARGIN_SM", str(sm_count)))
    grouped_gemm_impl = (
        tex.te_general_grouped_gemm_for_discrete_in
        if isinstance(weight, list)
        else tex.te_general_grouped_gemm_for_grouped_tensor
    )
    grouped_gemm_impl(
        weight,
        True,
        grouped_input,
        False,
        grouped_output,
        None,
        alpha,
        beta,
        workspace_setup,
        workspace_cublas,
        False,
        sm_count,
    )


def _get_te_mxfp8_batch_invariant_weight(normalized_weight):
    """Return graph-stable, GEMM-swizzled expert views for the fixed-M path."""
    if isinstance(normalized_weight, list):
        return normalized_weight

    cached = getattr(normalized_weight, _TE_MXFP8_BATCH_INVARIANT_WEIGHT_CACHE, None)
    if cached is not None:
        return cached[1]
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "Batch-invariant TE MXFP8 requires GroupedTensor GEMM views to be "
            "materialized before CUDA graph capture."
        )

    original_members = normalized_weight.quantized_tensors
    if original_members is None:
        original_members = normalized_weight.split_into_quantized_tensors()
        normalized_weight.quantized_tensors = original_members
    grouped_for_gemm = normalized_weight.copy()
    tex.grouped_swizzle_for_gemm(grouped_for_gemm, rowwise=True, columnwise=False)
    swizzled_members = grouped_for_gemm.split_into_quantized_tensors()
    gemm_members = [
        TEMXFP8Tensor(
            shape=original.shape,
            dtype=original.dtype,
            rowwise_data=original._rowwise_data,
            rowwise_scale_inv=swizzled._rowwise_scale_inv,
            columnwise_data=None,
            columnwise_scale_inv=None,
            fp8_dtype=original._fp8_dtype,
            quantizer=original._quantizer,
            requires_grad=False,
            with_gemm_swizzled_scales=True,
        )
        for original, swizzled in zip(original_members, swizzled_members)
    ]
    cache = (grouped_for_gemm, gemm_members)
    setattr(normalized_weight, _TE_MXFP8_BATCH_INVARIANT_WEIGHT_CACHE, cache)
    return gemm_members


def prepare_te_mxfp8_batch_invariant_weight(weight) -> None:
    """Materialize graph-stable GEMM views for a native TE MXFP8 weight."""
    _get_te_mxfp8_batch_invariant_weight(_normalize_te_mxfp8_weight(weight))


@torch.no_grad()
def refresh_te_mxfp8_batch_invariant_weight(weight) -> bool:
    """Refresh a cached grouped-weight scale layout in place after model refit."""
    normalized_weight = _normalize_te_mxfp8_weight(weight)
    if isinstance(normalized_weight, list):
        return False
    cached = getattr(normalized_weight, _TE_MXFP8_BATCH_INVARIANT_WEIGHT_CACHE, None)
    if cached is None:
        return False

    grouped_for_gemm = normalized_weight.copy()
    tex.grouped_swizzle_for_gemm(grouped_for_gemm, rowwise=True, columnwise=False)
    cached[0].scale_inv.copy_(grouped_for_gemm.scale_inv)
    return True


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


def _te_mxfp8_batch_invariant_grouped_mm(
    x_bf16: torch.Tensor, weight, first_dims: torch.Tensor, *, num_chunks: int
) -> torch.Tensor:
    """Run identical fixed-M grouped GEMMs for each expert-token chunk.

    ``x_bf16`` is chunk-major with ``num_experts * 256`` rows per chunk.
    Launching chunks separately keeps an expert at the same grouped-GEMM index;
    TE can otherwise select a different FC2 reduction recipe for the same shape
    when that expert's data moves to another group index.
    """
    assert x_bf16.dtype == torch.bfloat16, f"Expected bf16 input, got {x_bf16.dtype}"
    normalized_weight = _normalize_te_mxfp8_weight(weight)
    normalized_weight = _get_te_mxfp8_batch_invariant_weight(normalized_weight)
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
    rows_per_chunk = num_experts * _TE_MXFP8_BATCH_INVARIANT_CHUNK_SIZE
    if x_bf16.shape[0] != num_chunks * rows_per_chunk:
        raise ValueError(
            "Batch-invariant TE MXFP8 input has an invalid chunked row count: "
            f"got {x_bf16.shape[0]}, expected {num_chunks * rows_per_chunk}."
        )

    out_features = _te_weight_out_features(normalized_weight)
    output_data = torch.empty(
        x_bf16.shape[0] * out_features, dtype=torch.bfloat16, device=x_bf16.device
    )
    tensor_offsets = torch.arange(num_experts, dtype=first_dims.dtype, device=first_dims.device) * (
        _TE_MXFP8_BATCH_INVARIANT_CHUNK_SIZE * out_features
    )
    alpha = torch.ones(num_experts, dtype=torch.float32, device=x_bf16.device)
    beta = torch.zeros(num_experts, dtype=torch.float32, device=x_bf16.device)
    output_rows_per_chunk = rows_per_chunk * out_features
    launch_state = []
    for chunk in range(num_chunks):
        row_start = chunk * rows_per_chunk
        row_end = row_start + rows_per_chunk
        grouped_input = tex.group_quantize(
            x_bf16[row_start:row_end], _get_te_mxfp8_activation_quantizer(), num_experts, first_dims
        )
        output_start = chunk * output_rows_per_chunk
        grouped_output = TEGroupedTensor(
            shape=(rows_per_chunk, out_features),
            dtype=torch.bfloat16,
            num_tensors=num_experts,
            shapes=None,
            quantizer=None,
            data=output_data[output_start : output_start + output_rows_per_chunk],
            first_dims=first_dims,
            tensor_offsets=tensor_offsets,
            requires_grad=False,
        )
        # TE may execute grouped GEMMs on auxiliary streams. Keep workspaces distinct
        # across chunk launches just like its public wrapper does; reusing them here
        # introduces a cross-stream race between consecutive chunks.
        workspace_setup = torch.empty(
            get_grouped_gemm_setup_workspace_size(num_experts),
            dtype=torch.uint8,
            device=x_bf16.device,
        )
        workspace_cublas = torch.empty(
            get_unrestricted_te_workspace_size_bytes(), dtype=torch.uint8, device=x_bf16.device
        )
        # The implementation may consume its input and workspaces on auxiliary
        # streams. Retain each chunk's state until every launch has been queued;
        # otherwise the caching allocator can recycle an earlier chunk while a
        # later one is being prepared.
        launch_state.append((grouped_input, workspace_setup, workspace_cublas))
        _te_mxfp8_batch_invariant_grouped_gemm(
            normalized_weight,
            grouped_input,
            grouped_output,
            alpha,
            beta,
            workspace_setup,
            workspace_cublas,
        )
    return output_data.view(x_bf16.shape[0], out_features)


def _te_mxfp8_batch_invariant_reorder(
    hidden_states: torch.Tensor,
    permuted_probs: torch.Tensor,
    inverse_map: torch.Tensor,
    num_chunks: int,
):
    """Place each token/expert pair in a deterministic chunk-major row.

    The regular permutation compacts rows with atomics. Although its inverse map
    restores token order, a token can land at a different row inside the FC2
    matrix when the co-batch changes, and the MXFP8 kernel's reduction can then
    change bits. Assign row ``token % 256`` in chunk ``token // 256`` instead.
    """
    num_tokens, num_experts = inverse_map.shape
    chunk_size = _TE_MXFP8_BATCH_INVARIANT_CHUNK_SIZE
    rows = torch.arange(chunk_size, dtype=torch.int64, device=inverse_map.device)
    hidden_chunks = []
    prob_chunks = []
    map_chunks = []
    for chunk in range(num_chunks):
        token_rows = chunk * chunk_size + rows
        in_token_buffer = token_rows < num_tokens
        safe_token_rows = torch.where(in_token_buffer, token_rows, 0)
        source_rows = inverse_map[safe_token_rows].transpose(0, 1)
        valid_rows = in_token_buffer[None, :] & (source_rows >= 0)
        safe_source_rows = torch.where(valid_rows, source_rows, 0).to(torch.int64)
        hidden_chunks.append(
            torch.where(valid_rows[..., None], hidden_states[safe_source_rows], 0.0).flatten(0, 1)
        )
        prob_chunks.append(torch.where(valid_rows, permuted_probs[safe_source_rows], 0.0).flatten())
        map_chunks.append(
            torch.where(valid_rows, safe_token_rows.to(inverse_map.dtype)[None, :], -1).flatten()
        )

    expert_ids = torch.arange(num_experts, dtype=inverse_map.dtype, device=inverse_map.device)
    token_ids = torch.arange(num_tokens, dtype=inverse_map.dtype, device=inverse_map.device)[
        :, None
    ]
    chunked_inverse_map = (
        torch.div(token_ids, chunk_size, rounding_mode="floor") * num_experts + expert_ids
    ) * chunk_size + torch.remainder(token_ids, chunk_size)
    chunked_inverse_map = torch.where(inverse_map >= 0, chunked_inverse_map, -1)
    return (
        torch.cat(hidden_chunks),
        torch.cat(prob_chunks),
        torch.cat(map_chunks),
        chunked_inverse_map,
    )


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

    if batch_invariant_mode and use_te_mxfp8:
        # Launch each 256-row token chunk separately so an expert always has the same
        # grouped-GEMM index and M, independent of routing counts and graph bucket size.
        num_te_chunks = max(
            1,
            (max_tokens + _TE_MXFP8_BATCH_INVARIANT_CHUNK_SIZE - 1)
            // _TE_MXFP8_BATCH_INVARIANT_CHUNK_SIZE,
        )
        mm_fn = partial(_te_mxfp8_batch_invariant_grouped_mm, num_chunks=num_te_chunks)
        expert_alignment = _TE_MXFP8_BATCH_INVARIANT_CHUNK_SIZE
    elif batch_invariant_mode:
        assert not use_mcore_mxfp8, (
            "batch_invariant_mode does not support MCore MXFP8 weights. Use native "
            "TE MXFP8 weights or disable mxfp8."
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
    # offs[-1:] normally points to the dynamic used prefix. The fixed-chunk TE path
    # replaces it below with the complete chunk-major row count.
    n_used = offs[-1:]
    if use_te_mxfp8:
        first_dims = torch.cat((offs[:1], offs[1:] - offs[:-1])).to(torch.int64)
        if batch_invariant_mode:
            hidden_states, permuted_probs, permutation_map, batch_invariant_inverse_map = (
                _te_mxfp8_batch_invariant_reorder(
                    hidden_states, permuted_probs, batch_invariant_inverse_map, num_te_chunks
                )
            )
            first_dims = first_dims.new_full(
                (num_local_experts,), _TE_MXFP8_BATCH_INVARIANT_CHUNK_SIZE
            )
            n_used = offs.new_full((1,), hidden_states.shape[0])
        fc1_output = mm_fn(hidden_states, fc1_weight, first_dims)
    else:
        fc1_output = mm_fn(hidden_states, fc1_weight, offs)

    if batch_invariant_mode:
        # Match training: BF16 activation, FP32 probability multiply, then BF16 before FC2.
        if activation_type == ActivationType.SWIGLU:
            assert activation_clamp_scale is None, (
                "activation_func_tanh_clamp_scale is only implemented for squared ReLU here; "
                "the gated form (SiTU-GLU) has no inference kernel yet."
            )
            activation_out = batch_invariant.swiglu_with_probs(
                fc1_output, permutation_map, n_used, permuted_probs, zero_padding=use_te_mxfp8
            )
        else:
            activation_out = batch_invariant.squared_relu_with_probs(
                fc1_output,
                permutation_map,
                n_used,
                permuted_probs,
                activation_clamp_scale,
                zero_padding=use_te_mxfp8,
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
