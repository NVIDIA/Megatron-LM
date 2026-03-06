# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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
    from torch.nn.functional import scaled_mm, ScalingType, SwizzleType

    HAVE_TORCH_SCALED_MM = True
except ImportError:
    HAVE_TORCH_SCALED_MM = False


def _verify_te_to_mxfp8_conversion(te_dequantized, fi_quantized: MXFP8Tensor) -> None:
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
    """
    Converts a TE MXFP8 model to an inference MXFP8 model by
    recursively translating each layer's weights.

    Args:
        model: The model to convert.
        backend: "flashinfer" or "torch" — which quantization backend to use.
    """
    assert HAVE_TE
    assert backend in ("flashinfer", "torch"), f"Unknown mxfp8 backend: {backend}"

    if backend == "flashinfer":
        assert HAVE_FLASHINFER, "FlashInfer not available for mxfp8 quantization"
        quantize_fn = MXFP8Tensor.from_bf16_flashinfer
    else:
        quantize_fn = MXFP8Tensor.from_bf16_torch

    # Recurse through child modules
    for child in model.children():
        quantize_model_to_mxfp8(child, backend=backend)

    def replace_in_dict(attr_dict):
        """Helper function to replace TE MXFP8 weights."""
        keys = list(attr_dict.keys())
        for key in keys:
            val = attr_dict[key]
            # Check both direct TEMXFP8Tensor and Parameter wrapping one.
            # Also use dequantize() as a duck-type check since isinstance may
            # fail when TE's tensor subclass is wrapped in a Parameter.
            te_tensor = None
            if isinstance(val, TEMXFP8Tensor):
                te_tensor = val
            elif hasattr(val, 'data') and isinstance(val.data, TEMXFP8Tensor):
                te_tensor = val.data
    
            import logging 
            logging.log(logging.INFO, f"Quantizing {key}, te_tensor={te_tensor is not None}")

            if te_tensor is not None:
                te_dequantized = te_tensor.dequantize()
                quantized = quantize_fn(te_dequantized)

                # Sanity check the numerical correctness
                _verify_te_to_mxfp8_conversion(te_dequantized, quantized)

                # Remove the existing TE parameter and replace
                del model._parameters[key]
                setattr(model, key, quantized)

    if hasattr(model, '_parameters') and model._parameters:
        replace_in_dict(model._parameters)

    return model


def mm_mxfp8(x: torch.Tensor, weight: MXFP8Tensor, out: torch.Tensor = None):
    """
    Computes a matmul in MXFP8 using FlashInfer.

    Quantizes the bf16 input activation tensor. Weight must be pre-quantized.
    """
    assert HAVE_FLASHINFER

    x = MXFP8Tensor.from_bf16_flashinfer(x.squeeze(1))
    return flashinfer_mm_mxfp8(
        x.data, weight.data.T, x.scale, weight.scale, out_dtype=torch.bfloat16, out=out
    ).unsqueeze(1)


def mm_mxfp8_torch(x: torch.Tensor, weight: MXFP8Tensor, out: torch.Tensor = None):
    """
    Computes a matmul in MXFP8 using torch.nn.functional.scaled_mm.

    Quantizes the bf16 input activation tensor on the fly using Triton.
    Weight must be pre-quantized with swizzled scales.
    """
    assert HAVE_TORCH_SCALED_MM, (
        "torch.nn.functional.scaled_mm with ScalingType/SwizzleType not available. "
        "Upgrade PyTorch."
    )

    x_mxfp8 = MXFP8Tensor.from_bf16_torch(x.squeeze(1))

    # scaled_mm expects scale_a as 2D matching xq ndim,
    # and scale_b as 2D matching wq.t() ndim (after transpose).
    xs = x_mxfp8.scale.reshape(-1, x_mxfp8.data.shape[-1] // 32)
    ws = weight.scale  # already 1D swizzled from pre-quantization

    result = scaled_mm(
        x_mxfp8.data,
        weight.data.t(),
        xs, ScalingType.BlockWise1x32,
        ws, ScalingType.BlockWise1x32,
        swizzle_a=SwizzleType.SWIZZLE_32_4_4,
        swizzle_b=SwizzleType.SWIZZLE_32_4_4,
        output_dtype=torch.bfloat16,
    )

    if out is not None:
        out.copy_(result.unsqueeze(1))
        return out
    return result.unsqueeze(1)