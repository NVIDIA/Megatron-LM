# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Utility functions related to FP8 that are used throughout Megatron core"""
from typing import Tuple

import torch

from megatron.core.utils import is_te_min_version

# Check if Transformer Engine is installed
HAVE_TE = False
try:
    import transformer_engine  # pylint: disable=W0611

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    # Transformer Engine not found
    pass

# Check if Transformer Engine has Float8Tensor class
HAVE_TE_FLOAT8TENSOR = False
try:
    from transformer_engine.pytorch.float8_tensor import Float8Tensor

    HAVE_TE_FLOAT8TENSOR = True
except (ImportError, ModuleNotFoundError):
    # Float8Tensor not found
    pass

# Check if Transformer Engine has MXFP8Tensor class
HAVE_TE_MXFP8TENSOR = False
try:
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

    HAVE_TE_MXFP8TENSOR = True
except (ImportError, ModuleNotFoundError):
    # MXFP8Tensor not found
    pass

# utils for transformer engine fp8 and mxfp8 tensor

if HAVE_TE and is_te_min_version("2.0"):
    # TE quantization logic using quantizer API
    # Supported TE versions: 2.0+
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor

    def _quantize_param_fragment_impl(
        input_: torch.Tensor, *, out: torch.Tensor, param: torch.nn.Parameter
    ) -> None:
        quantizer = param._quantizer
        out = Float8Tensor(
            shape=input_.size(),
            dtype=param.dtype,
            requires_grad=False,
            data=out,
            fp8_scale_inv=param._scale_inv,
            fp8_dtype=param._fp8_dtype,
            quantizer=quantizer,
        )
        quantizer.update_quantized(input_, out)

    def _get_fp8_scale_and_amax_impl(tensor: Float8Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        quantizer = tensor._quantizer
        return quantizer.scale, quantizer.amax

elif HAVE_TE and is_te_min_version("1.0"):
    # TE quantization logic with fp8_meta dicts
    # Supported TE versions: 1.0 - 1.14
    from transformer_engine.pytorch.cpp_extensions import cast_to_fp8

    def _quantize_param_fragment_impl(
        input_: torch.Tensor, *, out: torch.Tensor, param: torch.nn.Parameter
    ) -> None:
        cast_to_fp8(
            input_.view(1, -1),
            param._fp8_meta["scaling_fwd"],
            param._fp8_meta_index,
            param._fp8_dtype,
            out=out.view(1, -1),
        )

    def _get_fp8_scale_and_amax_impl(tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fp8_meta = tensor._fp8_meta["scaling_fwd"]
        fp8_meta_index = tensor._fp8_meta_index
        return fp8_meta.scale[fp8_meta_index], fp8_meta.amax_history[0][fp8_meta_index]

else:
    # Fallback impl if TE version is invalid
    def _quantize_param_fragment_impl(*args, **kwargs) -> None:
        raise RuntimeError("Invalid Transformer Engine version for FP8 distributed optimizer")

    def _get_fp8_scale_and_amax_impl(*args, **kwargs):
        raise RuntimeError("Invalid Transformer Engine version for FP8 distributed optimizer")


def quantize_param_fragment(
    input_: torch.Tensor, *, out: torch.Tensor, param: torch.nn.Parameter
) -> None:
    """Cast values in parameter fragment to FP8
    Arguments:
      input_ (torch.Tensor): Values to quantize.
      out (torch.Tensor): Raw UINT8 buffer to fill with FP8 values.
          Dimensions should match input_.
      param (torch.nn.Parameter): Parameter containing this parameter
          fragment. Must be a Float8Tensor.
    """
    _quantize_param_fragment_impl(input_, out=out, param=param)


def get_fp8_scale_and_amax(tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get FP8 scale and amax from Float8Tensor"""
    return _get_fp8_scale_and_amax_impl(tensor)


def is_float8tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a Transformer Engine Float8Tensor"""
    return HAVE_TE_FLOAT8TENSOR and isinstance(tensor, Float8Tensor)


def is_mxfp8tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a Transformer Engine MXFP8Tensor"""
    return HAVE_TE_MXFP8TENSOR and isinstance(tensor, MXFP8Tensor)
