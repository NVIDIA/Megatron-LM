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
