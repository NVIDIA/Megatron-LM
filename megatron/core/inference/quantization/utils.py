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
from megatron.core.utils import copy_parameter_metadata

if TYPE_CHECKING:
    from megatron.core.inference.moe import InferenceGroupedGemmBackend
    from megatron.core.transformer.transformer_config import TransformerConfig

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

    Raises:
        ValueError: If the grouped-GEMM backend does not support MXFP8.
    """
    grouped_gemm_backend = getattr(
        inference_grouped_gemm_backend, "value", inference_grouped_gemm_backend
    )
    # All supported grouped-MoE backends consume MCore's canonical Triton/cuBLAS
    # layout. FlashInfer repacks expert weights into TRT-LLM Major-K layout separately.
    if grouped_gemm_backend in ("torch", "flashinfer", "vllm"):
        return "triton"
    raise ValueError(
        "MXFP8 inference does not support "
        f"inference_grouped_gemm_backend={grouped_gemm_backend!r}."
    )


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


def _has_mxfp8_storage(parameter: object) -> bool:
    """Return whether a parameter or its data uses TE or MCore MXFP8 storage."""
    if isinstance(parameter, MXFP8Tensor) or (HAVE_TE and isinstance(parameter, TEMXFP8Tensor)):
        return True
    data = getattr(parameter, "data", None)
    return isinstance(data, MXFP8Tensor) or (HAVE_TE and isinstance(data, TEMXFP8Tensor))


def _validate_mxfp8_expert_precision_policy(
    model: torch.nn.Module,
    include_pattern: str | None,
    exclude_pattern: str | None,
    filter_prefix: str = "",
) -> None:
    """Reject a selective policy that splits an MoE layer across precisions."""
    for module_name, module in model.named_modules():
        if not (
            hasattr(module, "num_local_experts")
            and hasattr(module, "linear_fc1")
            and hasattr(module, "linear_fc2")
        ):
            continue

        weight_formats = []
        for linear_name in ("linear_fc1", "linear_fc2"):
            linear = getattr(module, linear_name)
            if hasattr(linear, "weight0"):
                weight_names = (
                    f"weight{expert_index}" for expert_index in range(module.num_local_experts)
                )
            elif hasattr(linear, "weight"):
                weight_names = ("weight",)
            else:
                continue

            for weight_name in weight_names:
                if not hasattr(linear, weight_name):
                    continue
                relative_name = ".".join(
                    part for part in (module_name, linear_name, weight_name) if part
                )
                filter_name = f"{filter_prefix}{relative_name}"
                keep_mxfp8 = _has_mxfp8_storage(getattr(linear, weight_name)) and (
                    matches_mxfp8_parameter_filter(
                        filter_name,
                        include_pattern=include_pattern,
                        exclude_pattern=exclude_pattern,
                    )
                )
                weight_formats.append((filter_name, keep_mxfp8))

        format_flags = [keep_mxfp8 for _, keep_mxfp8 in weight_formats]
        if format_flags and any(format_flags) != all(format_flags):
            mxfp8_name = next(name for name, keep_mxfp8 in weight_formats if keep_mxfp8)
            bf16_name = next(name for name, keep_mxfp8 in weight_formats if not keep_mxfp8)
            layer_name = f"{filter_prefix}{module_name}" or "<root>"
            raise ValueError(
                "MXFP8 inference requires every FC1 and FC2 expert weight in an MoE layer "
                f"to use one precision, but the policy mixes formats in {layer_name!r}: "
                f"{mxfp8_name!r} would remain MXFP8 while {bf16_name!r} would use BF16. "
                "Adjust inference_mxfp8_include_parameters and "
                "inference_mxfp8_exclude_parameters to select both expert projections and "
                "all local experts together."
            )


def _materialize_mxfp8_parameter_as_bf16(
    module: torch.nn.Module, parameter_name: str, parameter: torch.Tensor
) -> None:
    """Replace a filtered-out TE MXFP8 weight with an ordinary BF16 parameter.

    ``fp8_param=True`` wraps eligible weights in TE MXFP8 storage during model
    construction, before the inference include/exclude filters are applied. A
    filtered-out weight must therefore be dequantized so the BF16 grouped-GEMM
    and refit paths do not receive TE's quantized tensor subclass. Rebuilding
    the parameter also requires copying Megatron's public sharding and refit
    metadata so checkpoint loading and reshard planning retain the same layout.
    """
    bf16_parameter = torch.nn.Parameter(
        parameter.dequantize().to(torch.bfloat16), requires_grad=parameter.requires_grad
    )
    copy_parameter_metadata(bf16_parameter, parameter)
    del module._parameters[parameter_name]
    setattr(module, parameter_name, bf16_parameter)


def materialize_unselected_mxfp8_parameters_as_bf16(
    model: torch.nn.Module,
    include_pattern: str | None = None,
    exclude_pattern: str | None = None,
    _prefix: str = "",
) -> None:
    """Materialize filtered-out TE MXFP8 parameters as ordinary BF16 parameters.

    Core GPT and Hybrid model constructors call this before returning the model.
    Any checkpoint loader can then copy directly into the replacement parameters
    instead of quantizing into TE MXFP8 storage only to dequantize them afterward.

    Args:
        model: Model whose unselected TE MXFP8 parameters should be replaced.
        include_pattern: Regex selecting fully qualified parameter names to keep in MXFP8.
            If unset, all MXFP8 parameters are included.
        exclude_pattern: Regex selecting fully qualified parameter names to materialize
            in BF16. Exclusion takes precedence over inclusion.
        _prefix: Internal recursion prefix; callers should not set this.
    """
    assert HAVE_TE
    if not _prefix:
        _validate_mxfp8_expert_precision_policy(model, include_pattern, exclude_pattern)

    for child_name, child in model.named_children():
        materialize_unselected_mxfp8_parameters_as_bf16(
            child,
            include_pattern=include_pattern,
            exclude_pattern=exclude_pattern,
            _prefix=f"{_prefix}{child_name}.",
        )

    for parameter_name, parameter in list(model._parameters.items()):
        if parameter is None:
            continue
        is_te_mxfp8 = isinstance(parameter, TEMXFP8Tensor) or isinstance(
            getattr(parameter, "data", None), TEMXFP8Tensor
        )
        if is_te_mxfp8 and not matches_mxfp8_parameter_filter(
            f"{_prefix}{parameter_name}",
            include_pattern=include_pattern,
            exclude_pattern=exclude_pattern,
        ):
            _materialize_mxfp8_parameter_as_bf16(model, parameter_name, parameter)


def apply_mxfp8_parameter_filter(model: torch.nn.Module, config: "TransformerConfig") -> None:
    """Apply the configured MXFP8 storage filter before checkpoint loading.

    Core GPT and Hybrid model constructors call this after all submodules have
    initialized. Checkpoint and refit integrations therefore receive the final
    BF16/MXFP8 parameter layout without coordinating a separate callback.

    Args:
        model: Fully initialized model whose parameter names are available.
        config: Model configuration containing the optional MXFP8 filters.
    """
    include_pattern = getattr(config, "inference_mxfp8_include_parameters", None)
    exclude_pattern = getattr(config, "inference_mxfp8_exclude_parameters", None)
    if include_pattern is None and exclude_pattern is None:
        return

    materialize_unselected_mxfp8_parameters_as_bf16(
        model, include_pattern=include_pattern, exclude_pattern=exclude_pattern
    )


def quantize_model_to_mxfp8(
    model: torch.nn.Module,
    backend: MXFP8Backend = "flashinfer",
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
        include_pattern: Regex selecting fully qualified parameter names to keep in MXFP8.
            If unset, all MXFP8 parameters are included.
        exclude_pattern: Regex selecting fully qualified parameter names to materialize
            in BF16. Exclusion takes precedence over inclusion.
        _prefix: Internal recursion prefix; callers should not set this.
    """
    assert HAVE_TE
    if backend == "flashinfer":
        assert HAVE_FLASHINFER, "FlashInfer not available for MXFP8 quantization"
    if not _prefix:
        # Keep this pre-pass for direct callers and custom model types that do
        # not run the Core GPT/Hybrid model-construction hook.
        materialize_unselected_mxfp8_parameters_as_bf16(
            model, include_pattern=include_pattern, exclude_pattern=exclude_pattern
        )

    for child_name, child in model.named_children():
        child_prefix = f"{_prefix}{child_name}."
        quantize_model_to_mxfp8(
            child,
            backend=backend,
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
    return val.is_cuda and _has_mxfp8_storage(val)


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
    if not _prefix:
        _validate_mxfp8_expert_precision_policy(
            model, include_pattern, exclude_pattern, filter_prefix=_filter_prefix
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
