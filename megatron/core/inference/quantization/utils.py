# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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
    from flashinfer import mxfp8_quantize

    HAVE_FLASHINFER = True
except ImportError:
    HAVE_FLASHINFER = False


def _verify_te_to_flashinfer_mxfp8_conversion(te_dequantized, fi_quantized: MXFP8Tensor) -> None:
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


def quantize_model_to_mxfp8(model: torch.nn.Module) -> None:
    """
    Converts a TE MXFP8 model to a FlashInfer MXFP8 model by
    recursively translating each layer's weights.
    """
    assert HAVE_TE
    assert HAVE_FLASHINFER

    # Recurse through child modules
    for child in model.children():
        quantize_model_to_mxfp8(child)

    def replace_in_dict(attr_dict):
        """Helper function to replace TE MXFP8 weights."""
        keys = list(attr_dict.keys())
        for key in keys:
            val = attr_dict[key]
            if isinstance(val, TEMXFP8Tensor):
                # Undo the TE quantization and re-quantize with FlashInfer
                # Note that this introduces a one-time overhead but avoids any
                # numerical differences between TE and FlashInfer MXFP8 formats
                te_dequantized = val.dequantize()
                fi_quantized = MXFP8Tensor.from_bf16(te_dequantized)

                # Sanity check the numerical correctness of the TE -> FlashInfer conversion
                _verify_te_to_flashinfer_mxfp8_conversion(te_dequantized, fi_quantized)

                # Remove the existing TE parameter and then replace the
                # attribute with the re-quantized tensor
                del model._parameters[key]
                setattr(model, key, fi_quantized)

    if hasattr(model, '_parameters') and model._parameters:
        replace_in_dict(model._parameters)

    return model


def _should_quantize_param(val: torch.Tensor) -> bool:
    """Return True if a parameter should be quantized to FlashInfer MXFP8."""
    if not val.is_cuda:
        return False
    if HAVE_TE and isinstance(val, TEMXFP8Tensor):
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
) -> Dict[str, MXFP8Tensor]:
    """Quantize model parameters to FlashInfer MXFP8Tensor format.

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

    Returns:
        The ``persistent_buffers`` dict (created on first call if ``None``).
    """
    assert HAVE_FLASHINFER

    if persistent_buffers is None:
        persistent_buffers = {}

    # Recurse through child modules
    for child_name, child_module in model.named_children():
        child_prefix = f"{_prefix}{child_name}." if _prefix else f"{child_name}."
        quantize_params_to_mxfp8(child_module, persistent_buffers, _prefix=child_prefix)

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
                new_data, new_scale = mxfp8_quantize(bf16_data)
                persistent_buffers[fqn].data.copy_(new_data)
                persistent_buffers[fqn].scale.copy_(new_scale)
                fi_tensor = persistent_buffers[fqn]
            else:
                # First call: create new MXFP8Tensor
                fi_tensor = MXFP8Tensor.from_bf16(bf16_data)

                # Verify correctness for TEMXFP8Tensor inputs
                if HAVE_TE and isinstance(val, TEMXFP8Tensor):
                    _verify_te_to_flashinfer_mxfp8_conversion(bf16_data, fi_tensor)

                persistent_buffers[fqn] = fi_tensor

            # Replace nn.Parameter with MXFP8Tensor attribute
            del model._parameters[key]
            setattr(model, key, fi_tensor)

    return persistent_buffers


def mm_mxfp8(x: torch.Tensor, weight: MXFP8Tensor, out: torch.Tensor = None):
    """
    Computes a matmul in MXFP8 using FlashInfer.

    Quantizes the bf16 input activation tensor. Weight must be pre-quantized.
    """
    assert HAVE_FLASHINFER

    x = MXFP8Tensor.from_bf16(x.squeeze(1))
    return flashinfer_mm_mxfp8(
        x.data, weight.data.T, x.scale, weight.scale, out_dtype=torch.bfloat16, out=out
    ).unsqueeze(1)
