# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

try:
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor as TEMXFP8Tensor

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    from flashinfer import bmm_mxfp8

    HAVE_FLASHINFER = True
except ImportError:
    HAVE_FLASHINFER = False


def quantize_model_to_mxfp8(model: torch.nn.Module) -> None:
    """
    Converts a TE MXFP8 model to a FlashInfer MXFP8 model by
    recursively translating each layer's weights.
    """
    assert HAVE_TE

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
                new_val = MXFP8Tensor.from_bf16(val.dequantize())

                # Remove the existing TE parameter and then replace the
                # attribute with the re-quantized tensor
                del model._parameters[key]
                setattr(model, key, new_val)

    if hasattr(model, '_parameters') and model._parameters:
        replace_in_dict(model._parameters)

    return model

def mm_mxfp8(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor = None):
    """Computes a matmul in MXFP8 using FlashInfer."""
    assert HAVE_FLASHINFER

    # TODO(ksanthanam): Use the FlashInfer mm_mxfp8 API once 0.6.4 is released

    x = MXFP8Tensor.from_bf16(x.squeeze(1))
    return bmm_mxfp8(
        x.data.unsqueeze(0),
        self.weight.data.T.unsqueeze(0),
        x.scale,
        self.weight.scale,
        dtype=torch.bfloat16,
        out=out,
    ).transpose(0, 1)

