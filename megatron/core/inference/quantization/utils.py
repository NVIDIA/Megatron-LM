# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

try:
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor as TEMXFP8Tensor

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


def quantize_model_to_mxfp8(model: torch.nn.Module) -> None:
    """Quantizes a model to mxfp8."""

    assert HAVE_TE

    # 1. Recurse on all child modules (layers)
    for child in model.children():
        quantize_model_to_mxfp8(child)

    # 2. Inspect and replace Parameters, Buffers, and Attributes
    # We iterate over the instance dictionary to ensure we catch attributes
    # that might not be registered as Parameters or Buffers yet.

    # Helper to process a specific dictionary (like _parameters, _buffers, or __dict__)
    def replace_in_dict(attr_dict):
        # Create a list of keys to avoid runtime errors while modifying the dict
        keys = list(attr_dict.keys())
        for key in keys:
            val = attr_dict[key]
            if isinstance(val, TEMXFP8Tensor):
                # Perform the quantization
                new_val = MXFP8Tensor.from_bf16(val.dequantize())

                # Replace the value.
                # using setattr ensures PyTorch internal logic (like moving
                # from _parameters to normal attributes) is handled correctly.
                del model._parameters[key]
                setattr(model, key, new_val)

    # Check Parameters (e.g., weights, biases)
    # We check _parameters directly to be safe, though setattr handles updates.
    if hasattr(model, '_parameters') and model._parameters:
        replace_in_dict(model._parameters)

    """
    # Check Buffers (e.g., running stats in BatchNorm)
    if hasattr(module, '_buffers') and module._buffers:
        replace_in_dict(module._buffers)

    # Check standard attributes (in case the tensor is stored loosely)
    # We filter out methods and system attributes
    for key in list(module.__dict__.keys()):
        val = module.__dict__[key]
        if isinstance(val, TEMXFP8Tensor):
             new_val = MXFP8Tensor.from_bf16(val.dequantize())
             setattr(module, key, new_val)
    """

    return model
