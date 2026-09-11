# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch

from megatron.core.inference.quantization.mxfp8_tensor import (
    MXFP8Backend,
    MXFP8Tensor,
    validate_mxfp8_tensor,
)

if TYPE_CHECKING:
    from megatron.core.inference.moe import InferenceGroupedGemmBackend

try:
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor as TEMXFP8Tensor

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    from flashinfer import mm_mxfp8 as flashinfer_mm_mxfp8

    HAVE_FLASHINFER = True
except ImportError:
    HAVE_FLASHINFER = False

try:
    from torch.nn.functional import ScalingType, SwizzleType
    from torch.nn.functional import scaled_mm as torch_scaled_mm

    HAVE_TORCH_SCALED_MM = True
except ImportError:
    HAVE_TORCH_SCALED_MM = False


def _verify_te_to_mcore_mxfp8_conversion(te_dequantized, fi_quantized: MXFP8Tensor) -> None:
    # Sanity check: compare the first logical block (32 values)
    # Slice logical dimensions first to naturally handle any data swizzling/strides
    te_block = te_dequantized[0, :32].float()

    # Safely extract bytes from the first logical block, then view as e4m3
    fi_data_bytes = fi_quantized.data[0, :32].contiguous().view(torch.uint8)
    fi_data_e4m3 = fi_data_bytes.view(torch.float8_e4m3fn).float()

    # Extract the scale. Logical block (0, 0) is always at physical index 0,
    # bypassing any scale swizzling layout complexity (like SWIZZLED_128x4)
    fi_scale_byte = fi_quantized.scale.contiguous().flatten()[0:1].view(torch.uint8).to(torch.int32)
    fi_scale_f32 = (fi_scale_byte << 23).view(torch.float32)

    fi_block = fi_data_e4m3 * fi_scale_f32

    if not torch.allclose(te_block, fi_block):
        diff_norm = torch.norm(te_block - fi_block)
        raise ValueError(f"MXFP8 sanity check failed. Diff norm: {diff_norm}")


def resolve_mxfp8_backend(
    inference_grouped_gemm_backend: str | InferenceGroupedGemmBackend,
) -> MXFP8Backend:
    """Resolve the canonical MXFP8 storage required by a grouped-MoE backend.

    Args:
        inference_grouped_gemm_backend: The configured backend, either as its raw
            string value or as the enum produced by ``TransformerConfig``.

    Returns:
        The MXFP8 quantization and storage backend to use. FlashInfer routed MoE
        derives its TRT-LLM Major-K weights from the canonical Triton/cuBLAS layout.
        With the TE backend, grouped-MoE expert weights stay in native TE format while
        other inference-optimized linear layers use the canonical Triton layout.

    Raises:
        ValueError: If the grouped-GEMM backend does not support MXFP8.
    """
    grouped_gemm_backend = getattr(
        inference_grouped_gemm_backend, "value", inference_grouped_gemm_backend
    )
    # All grouped-MoE backends consume MCore's canonical Triton/cuBLAS layout for the
    # dense inference-optimized linear layers. FlashInfer repacks expert weights into
    # TRT-LLM Major-K layout separately; TE keeps expert weights in native TE format.
    if grouped_gemm_backend in ("te", "torch", "flashinfer", "vllm"):
        return "triton"
    raise ValueError(
        "MXFP8 inference does not support "
        f"inference_grouped_gemm_backend={grouped_gemm_backend!r}."
    )


def get_te_grouped_moe_parameter_ids(model: torch.nn.Module) -> set[int]:
    """Collect expert parameters preserved for the native TE grouped-MoE backend."""
    parameter_ids: set[int] = set()
    for module in model.modules():
        grouped_gemm_backend = getattr(module, "inference_grouped_gemm_backend", None)
        grouped_gemm_backend = getattr(grouped_gemm_backend, "value", grouped_gemm_backend)
        if grouped_gemm_backend != "te":
            continue
        for linear_name in ("linear_fc1", "linear_fc2"):
            linear = getattr(module, linear_name, None)
            if isinstance(linear, torch.nn.Module):
                parameter_ids.update(id(param) for param in linear.parameters())
    return parameter_ids


def matches_mxfp8_parameter_filter(
    parameter_name: str, include_pattern: str | None = None, exclude_pattern: str | None = None
) -> bool:
    """Return whether a fully qualified parameter name should remain in MXFP8.

    Inclusion defaults to all parameters. Exclusion takes precedence when both
    regular expressions match.
    """
    included = include_pattern is None or re.search(include_pattern, parameter_name) is not None
    excluded = (
        exclude_pattern is not None and re.search(exclude_pattern, parameter_name) is not None
    )
    return included and not excluded


def _validate_mxfp8_parameter_filters(
    include_pattern: str | None, exclude_pattern: str | None
) -> None:
    """Validate parameter filters before any model parameters are mutated."""
    for pattern in (include_pattern, exclude_pattern):
        if pattern is None:
            continue
        try:
            re.compile(pattern)
        except re.error as error:
            raise ValueError(f"Invalid MXFP8 parameter regex {pattern!r}: {error}") from error


def _materialize_mxfp8_parameter_as_bf16(
    module: torch.nn.Module, parameter_name: str, parameter: torch.Tensor
) -> None:
    """Replace a TE MXFP8 parameter with a BF16 parameter while preserving sharding metadata."""
    bf16_parameter = torch.nn.Parameter(
        parameter.dequantize().to(torch.bfloat16), requires_grad=parameter.requires_grad
    )
    for attribute in (
        "allreduce",
        "expert_parallel",
        "expert_tp",
        "group",
        "is_embedding_or_output_parameter",
        "is_gtp_weight_remat",
        "is_qkv",
        "pad_length",
        "partition_dim",
        "partition_sizes",
        "partition_stride",
        "qkv_split_shapes",
        "sequence_parallel",
        "tensor_model_parallel",
    ):
        if hasattr(parameter, attribute):
            setattr(bf16_parameter, attribute, getattr(parameter, attribute))
    del module._parameters[parameter_name]
    setattr(module, parameter_name, bf16_parameter)


def quantize_model_to_mxfp8(
    model: torch.nn.Module,
    backend: MXFP8Backend = "flashinfer",
    excluded_parameter_ids: set[int] | None = None,
    include_pattern: str | None = None,
    exclude_pattern: str | None = None,
    _prefix: str = "",
) -> None:
    """Convert TE MXFP8 weights to mcore MXFP8Tensor format.

    Recursively walks the model and applies the configured precision policy to
    each TEMXFP8Tensor parameter. Selected parameters are re-quantized into an
    MCore MXFP8Tensor and unselected parameters are materialized in BF16.

    Args:
        model: The model whose TE MXFP8 parameters should be converted.
        backend: 'flashinfer' or 'triton' quantization backend.
        excluded_parameter_ids: Parameters to preserve in their native TE format.
            The TE grouped-MoE backend uses this for its expert weights while dense
            inference layers are still converted to the requested MCore layout.
        include_pattern: Regex selecting fully qualified parameter names to keep in MXFP8.
            If unset, all MXFP8 parameters are included.
        exclude_pattern: Regex selecting fully qualified parameter names to materialize
            in BF16. Exclusion takes precedence over inclusion.
        _prefix: Internal recursion prefix; callers should not set this.
    """
    assert HAVE_TE
    if backend == "flashinfer":
        assert HAVE_FLASHINFER, "FlashInfer not available for MXFP8 quantization"

    if _prefix == "":
        _validate_mxfp8_parameter_filters(include_pattern, exclude_pattern)

    if _prefix == "" and excluded_parameter_ids is not None:
        for parameter_name, parameter in model.named_parameters():
            if id(parameter) not in excluded_parameter_ids:
                continue
            is_te_mxfp8 = isinstance(parameter, TEMXFP8Tensor) or (
                hasattr(parameter, 'data') and isinstance(parameter.data, TEMXFP8Tensor)
            )
            if is_te_mxfp8 and not matches_mxfp8_parameter_filter(
                parameter_name, include_pattern=include_pattern, exclude_pattern=exclude_pattern
            ):
                raise ValueError(
                    f"MXFP8 parameter filters select {parameter_name!r} for BF16, but the "
                    "configured grouped-MoE backend requires it in native TE MXFP8."
                )

    for child_name, child in model.named_children():
        child_prefix = f"{_prefix}{child_name}."
        quantize_model_to_mxfp8(
            child,
            backend=backend,
            excluded_parameter_ids=excluded_parameter_ids,
            include_pattern=include_pattern,
            exclude_pattern=exclude_pattern,
            _prefix=child_prefix,
        )

    def replace_in_dict(attr_dict):
        """Helper function to replace TE MXFP8 weights."""
        keys = list(attr_dict.keys())
        for key in keys:
            val = attr_dict[key]
            is_te_mxfp8 = isinstance(val, TEMXFP8Tensor) or (
                hasattr(val, 'data') and isinstance(val.data, TEMXFP8Tensor)
            )
            if is_te_mxfp8:
                full_name = f"{_prefix}{key}"
                keep_mxfp8 = matches_mxfp8_parameter_filter(
                    full_name, include_pattern=include_pattern, exclude_pattern=exclude_pattern
                )
                if excluded_parameter_ids is not None and id(val) in excluded_parameter_ids:
                    if not keep_mxfp8:
                        raise ValueError(
                            f"MXFP8 parameter filters select {full_name!r} for BF16, but the "
                            "configured grouped-MoE backend requires it in native TE MXFP8."
                        )
                    continue
                if not keep_mxfp8:
                    _materialize_mxfp8_parameter_as_bf16(model, key, val)
                    continue
                # Undo the TE quantization and re-quantize
                # Note that this introduces a one-time overhead but avoids any
                # numerical differences between TE and mcore MXFP8 formats
                te_dequantized = val.dequantize()
                mcore_quantized = MXFP8Tensor.from_bf16(te_dequantized, backend=backend)
                validate_mxfp8_tensor(
                    mcore_quantized,
                    expected_backend=backend,
                    tensor_name=f"quantized MXFP8 parameter {key!r}",
                )
                _verify_te_to_mcore_mxfp8_conversion(te_dequantized, mcore_quantized)
                del model._parameters[key]
                setattr(model, key, mcore_quantized)

    if hasattr(model, '_parameters') and model._parameters:
        replace_in_dict(model._parameters)

    return model


def _should_quantize_param(val: torch.Tensor) -> bool:
    """Return True if a parameter should be converted to an MCore MXFP8 tensor."""
    if not val.is_cuda:
        return False
    if HAVE_TE and isinstance(val, TEMXFP8Tensor):
        return True
    if HAVE_TE and hasattr(val, 'data') and isinstance(val.data, TEMXFP8Tensor):
        return True
    if isinstance(val, MXFP8Tensor):
        return True
    if hasattr(val, 'data') and isinstance(val.data, MXFP8Tensor):
        return True
    return False


def _to_bf16(val: torch.Tensor) -> torch.Tensor:
    """Convert a parameter value to BF16 for quantization."""
    if HAVE_TE and isinstance(val, TEMXFP8Tensor):
        return val.dequantize()
    if HAVE_TE and hasattr(val, 'data') and isinstance(val.data, TEMXFP8Tensor):
        return val.data.dequantize()
    return val.data.to(torch.bfloat16)


def collect_mxfp8_param_metadata(
    model: torch.nn.Module,
) -> Dict[str, Tuple[torch.Size, torch.dtype, torch.device]]:
    """Record shape/dtype/device for each parameter that will be quantized.

    Called once before the first quantization to record the original parameter
    metadata (shape, dtype, device) before any format conversion.
    """
    metadata: Dict[str, Tuple[torch.Size, torch.dtype, torch.device]] = {}
    for name, param in model.named_parameters():
        if _should_quantize_param(param):
            if HAVE_TE and isinstance(param, TEMXFP8Tensor):
                bf16 = param.dequantize()
                metadata[name] = (bf16.shape, bf16.dtype, bf16.device)
            else:
                metadata[name] = (param.shape, param.dtype, param.device)
    return metadata


@torch.inference_mode(False)
@torch.no_grad()
def quantize_params_to_mxfp8(
    model: torch.nn.Module,
    persistent_buffers: Optional[Dict[str, MXFP8Tensor]] = None,
    _prefix: str = "",
    backend: MXFP8Backend = "flashinfer",
    excluded_parameter_ids: set[int] | None = None,
    include_pattern: str | None = None,
    exclude_pattern: str | None = None,
    _filter_prefix: str = "",
) -> Dict[str, MXFP8Tensor]:
    """Quantize model parameters to mutable MXFP8Tensor storage.

    Handles both TEMXFP8Tensor (fp8_param=True) and BF16/FP16 nn.Parameter
    inputs.  When *persistent_buffers* is provided, new quantized values are
    ``copy_()``'d into the existing MXFP8Tensor objects so that CUDA-graph
    device-pointer captures remain valid.  Persistent buffers are deliberately
    created outside inference mode so later refits can update them regardless
    of the caller's execution mode.

    Args:
        model: The model whose parameters should be quantized.
        persistent_buffers: If not ``None``, a dict mapping fully-qualified
            parameter names to previously-created ``MXFP8Tensor`` objects.
            Updated in-place and returned.
        _prefix: Internal recursion prefix; callers should not set this.
        backend: 'flashinfer' or 'triton' quantization backend.
        excluded_parameter_ids: Parameters that should remain in native TE format.
        include_pattern: Regex selecting fully qualified parameter names to keep in MXFP8.
            If unset, all MXFP8 parameters are included.
        exclude_pattern: Regex selecting fully qualified parameter names to materialize
            in BF16. Exclusion takes precedence over inclusion.
        _filter_prefix: Internal prefix used to match names from a model subtree.

    Returns:
        The ``persistent_buffers`` dict (created on first call if ``None``).
    """
    if backend == "flashinfer":
        assert HAVE_FLASHINFER, "FlashInfer not available for MXFP8 quantization"

    if _prefix == "":
        _validate_mxfp8_parameter_filters(include_pattern, exclude_pattern)

        if excluded_parameter_ids is not None:
            for parameter_name, parameter in model.named_parameters():
                if id(parameter) not in excluded_parameter_ids:
                    continue
                full_name = f"{_filter_prefix}{parameter_name}"
                if _should_quantize_param(parameter) and not matches_mxfp8_parameter_filter(
                    full_name, include_pattern=include_pattern, exclude_pattern=exclude_pattern
                ):
                    raise ValueError(
                        f"MXFP8 parameter filters select {full_name!r} for BF16, but the "
                        "configured grouped-MoE backend requires it in native TE MXFP8."
                    )

    if persistent_buffers is None:
        persistent_buffers = {}

    # Recurse through child modules
    for child_name, child_module in model.named_children():
        child_prefix = f"{_prefix}{child_name}." if _prefix else f"{child_name}."
        quantize_params_to_mxfp8(
            child_module,
            persistent_buffers,
            _prefix=child_prefix,
            backend=backend,
            excluded_parameter_ids=excluded_parameter_ids,
            include_pattern=include_pattern,
            exclude_pattern=exclude_pattern,
            _filter_prefix=_filter_prefix,
        )

    # Process parameters owned directly by this module
    if hasattr(model, '_parameters') and model._parameters:
        keys = list(model._parameters.keys())
        for key in keys:
            val = model._parameters[key]
            if val is None:
                continue
            if not _should_quantize_param(val):
                continue

            fqn = f"{_prefix}{key}"
            filter_fqn = f"{_filter_prefix}{fqn}"
            keep_mxfp8 = matches_mxfp8_parameter_filter(
                filter_fqn, include_pattern=include_pattern, exclude_pattern=exclude_pattern
            )
            if excluded_parameter_ids is not None and id(val) in excluded_parameter_ids:
                if not keep_mxfp8:
                    raise ValueError(
                        f"MXFP8 parameter filters select {filter_fqn!r} for BF16, but the "
                        "configured grouped-MoE backend requires it in native TE MXFP8."
                    )
                continue
            if not keep_mxfp8:
                _materialize_mxfp8_parameter_as_bf16(model, key, val)
                continue
            bf16_data = _to_bf16(val)

            if fqn in persistent_buffers:
                # Subsequent call: copy into existing tensors to preserve addresses
                persistent_tensor = persistent_buffers[fqn]
                validate_mxfp8_tensor(
                    persistent_tensor,
                    expected_backend=backend,
                    tensor_name=f"persistent MXFP8 parameter {fqn!r}",
                )
                persistent_tensor.copy_(bf16_data)
                mcore_tensor = persistent_tensor
            else:
                # First call: create new MXFP8Tensor
                mcore_tensor = MXFP8Tensor.from_bf16(bf16_data, backend=backend)

                # Verify correctness for TEMXFP8Tensor inputs
                if HAVE_TE and isinstance(val, TEMXFP8Tensor):
                    _verify_te_to_mcore_mxfp8_conversion(bf16_data, mcore_tensor)

                persistent_buffers[fqn] = mcore_tensor

            validate_mxfp8_tensor(
                mcore_tensor,
                expected_backend=backend,
                tensor_name=f"quantized MXFP8 parameter {fqn!r}",
            )

            # Replace nn.Parameter with MXFP8Tensor attribute
            del model._parameters[key]
            setattr(model, key, mcore_tensor)

    return persistent_buffers


def _mm_mxfp8_flashinfer(x_mxfp8: MXFP8Tensor, weight: MXFP8Tensor, out=None):
    """MXFP8 matmul via FlashInfer."""
    return flashinfer_mm_mxfp8(
        x_mxfp8.data, weight.data.T, x_mxfp8.scale, weight.scale, out_dtype=torch.bfloat16, out=out
    )


def _mm_mxfp8_torch(x_mxfp8: MXFP8Tensor, weight: MXFP8Tensor, out=None):
    """MXFP8 matmul via torch.nn.functional.scaled_mm."""
    result = torch_scaled_mm(
        x_mxfp8.data,
        weight.data.t(),
        x_mxfp8.scale_2d(),
        ScalingType.BlockWise1x32,
        weight.scale,
        ScalingType.BlockWise1x32,
        swizzle_a=SwizzleType.SWIZZLE_32_4_4,
        swizzle_b=SwizzleType.SWIZZLE_32_4_4,
        output_dtype=torch.bfloat16,
    )
    if out is not None:
        out.copy_(result)
        return out
    return result


def mm_mxfp8(x: torch.Tensor, weight: MXFP8Tensor, out: torch.Tensor = None):
    """Compute a matmul in MXFP8.

    Quantizes the bf16 input activation tensor on the fly. Weight must be
    pre-quantized. Dispatches to FlashInfer or torch based on weight.backend.
    """
    backend = weight.backend
    assert (
        backend is not None
    ), "weight.backend is None — was the weight created via MXFP8Tensor.from_bf16?"

    x_squeezed = x.squeeze(1)
    x_mxfp8 = MXFP8Tensor.from_bf16(x_squeezed, backend=backend)

    if backend == "flashinfer":
        assert HAVE_FLASHINFER, "FlashInfer not available for MXFP8 matmul"
        result = _mm_mxfp8_flashinfer(x_mxfp8, weight, out=out)
    elif backend == "triton":
        assert (
            HAVE_TORCH_SCALED_MM
        ), "torch.nn.functional.scaled_mm with ScalingType/SwizzleType not available"
        result = _mm_mxfp8_torch(x_mxfp8, weight, out=out)
    else:
        raise ValueError(f"Unknown MXFP8 backend: '{backend}'")

    return result.unsqueeze(1)
