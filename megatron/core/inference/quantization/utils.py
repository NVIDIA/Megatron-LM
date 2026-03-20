# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import gc
from typing import Dict, Optional, Tuple

import torch

from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

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


def _free_te_state(model: torch.nn.Module) -> None:
    """Remove TE-specific state that is no longer needed after MXFP8 conversion.

    Frees fp8_meta (CUDA tensors for scales/amax history), _extra_state
    (serialized FP8 checkpoint data), and any TE-registered buffers.
    """
    if hasattr(model, 'fp8_meta'):
        del model.fp8_meta
    if hasattr(model, '_extra_state'):
        del model._extra_state
    if hasattr(model, '_buffers') and model._buffers:
        model._buffers.clear()


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


def quantize_model_to_mxfp8(model: torch.nn.Module, backend: str = "flashinfer") -> None:
    """Convert TE MXFP8 weights to mcore MXFP8Tensor format.

    Recursively walks the model and replaces each TEMXFP8Tensor parameter
    with an MXFP8Tensor re-quantized via the specified backend.

    Args:
        model: The model whose TE MXFP8 parameters should be converted.
        backend: 'flashinfer' or 'triton' quantization backend.
    """
    assert HAVE_TE
    import logging

    rank = torch.distributed.get_rank()
    if backend == "flashinfer":
        assert HAVE_FLASHINFER, "FlashInfer not available for MXFP8 quantization"

    for child in model.children():
        quantize_model_to_mxfp8(child, backend=backend)

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
                _verify_te_to_mcore_mxfp8_conversion(te_dequantized, mcore_quantized)
                del model._parameters[key]
                setattr(model, key, mcore_quantized)
                # Free TE weight and intermediate BF16 tensor immediately
                del val, te_dequantized

    if hasattr(model, '_parameters') and model._parameters:
        replace_in_dict(model._parameters)

    _free_te_state(model)

    return model


def _should_quantize_param(val: torch.Tensor) -> bool:
    """Return True if a parameter should be quantized to FlashInfer MXFP8."""
    if not val.is_cuda:
        return False
    if HAVE_TE and isinstance(val, TEMXFP8Tensor):
        return True
    if HAVE_TE and hasattr(val, 'data') and isinstance(val.data, TEMXFP8Tensor):
        return True
    if (
        isinstance(val, torch.nn.Parameter)
        and val.dim() == 2
        and val.dtype in (torch.bfloat16, torch.float16)
    ):
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


def quantize_params_to_mxfp8(
    model: torch.nn.Module,
    persistent_buffers: Optional[Dict[str, MXFP8Tensor]] = None,
    _prefix: str = "",
    backend: str = "flashinfer",
) -> Dict[str, MXFP8Tensor]:
    """Quantize model parameters to MXFP8Tensor format.

    Handles both TEMXFP8Tensor (fp8_param=True) and BF16/FP16 nn.Parameter
    inputs.  When *persistent_buffers* is provided, new quantized values are
    ``copy_()``'d into the existing MXFP8Tensor objects so that CUDA-graph
    device-pointer captures remain valid.

    Args:
        model: The model whose parameters should be quantized.
        persistent_buffers: If not ``None``, a dict mapping fully-qualified
            parameter names to previously-created ``MXFP8Tensor`` objects.
            Updated in-place and returned.
        _prefix: Internal recursion prefix – callers should not set this.
        backend: 'flashinfer' or 'triton' quantization backend.

    Returns:
        The ``persistent_buffers`` dict (created on first call if ``None``).
    """
    if backend == "flashinfer":
        assert HAVE_FLASHINFER, "FlashInfer not available for MXFP8 quantization"

    if persistent_buffers is None:
        persistent_buffers = {}

    # Recurse through child modules
    for child_name, child_module in model.named_children():
        child_prefix = f"{_prefix}{child_name}." if _prefix else f"{child_name}."
        quantize_params_to_mxfp8(
            child_module, persistent_buffers, _prefix=child_prefix, backend=backend
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
            bf16_data = _to_bf16(val)

            if fqn in persistent_buffers:
                # Subsequent call: copy into existing tensors to preserve addresses
                new_tensor = MXFP8Tensor.from_bf16(bf16_data, backend=backend)
                persistent_buffers[fqn].data.copy_(new_tensor.data)
                persistent_buffers[fqn].scale.copy_(new_tensor.scale)
                mcore_tensor = persistent_buffers[fqn]
            else:
                # First call: create new MXFP8Tensor
                mcore_tensor = MXFP8Tensor.from_bf16(bf16_data, backend=backend)

                # Verify correctness for TEMXFP8Tensor inputs
                if HAVE_TE and isinstance(val, TEMXFP8Tensor):
                    _verify_te_to_mcore_mxfp8_conversion(bf16_data, mcore_tensor)

                persistent_buffers[fqn] = mcore_tensor

            # Replace nn.Parameter with MXFP8Tensor attribute
            del model._parameters[key]
            setattr(model, key, mcore_tensor)
            # Free TE weight and intermediate BF16 tensor immediately
            del val, bf16_data

    _free_te_state(model)

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
