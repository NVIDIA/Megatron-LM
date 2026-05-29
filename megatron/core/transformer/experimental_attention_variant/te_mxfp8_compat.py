# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DSA-local TransformerEngine MXFP8 compatibility helpers."""

import os


def _is_mxfp8_config(config) -> bool:
    fp8_recipe = getattr(config, "fp8_recipe", None)
    return fp8_recipe == "mxfp8"


def patch_te_mxfp8_view_backward_if_needed(config) -> bool:
    """Patch TE MXFP8 view backward only for DSA MXFP8 runs.

    This is a DSA-local compatibility path for environments where
    ``_ViewFunc.backward`` calls ``view`` on non-contiguous gradients. It is
    intentionally not installed through ``sitecustomize`` or global Megatron
    initialization.

    Returns:
        True if the patch is active after this call, False otherwise.
    """
    if not _is_mxfp8_config(config):
        return False
    if os.environ.get("MEGATRON_TE_MXFP8_VIEW_RESHAPE") != "1":
        return False

    try:
        from transformer_engine.pytorch.tensor import mxfp8_tensor
    except Exception:
        return False

    view_func = getattr(mxfp8_tensor, "_ViewFunc", None)
    mxfp8_cls = getattr(mxfp8_tensor, "MXFP8Tensor", None)
    if view_func is None or mxfp8_cls is None:
        return False
    if getattr(view_func, "_megatron_view_backward_reshape_patch", False):
        return True

    @staticmethod
    def backward(ctx, grad):
        if isinstance(grad, mxfp8_cls):
            new_data = (
                grad._rowwise_data.reshape(*ctx.shape) if grad._rowwise_data is not None else None
            )
            new_columnwise_data = (
                grad._columnwise_data.reshape(*ctx.shape)
                if grad._columnwise_data is not None
                else None
            )
            dgrad = mxfp8_cls(
                ctx.shape,
                grad.dtype,
                rowwise_data=new_data,
                rowwise_scale_inv=grad._rowwise_scale_inv,
                columnwise_data=new_columnwise_data,
                columnwise_scale_inv=grad._columnwise_scale_inv,
                fp8_dtype=grad._fp8_dtype,
                quantizer=grad._quantizer,
                with_gemm_swizzled_scales=grad._with_gemm_swizzled_scales,
            )
            return dgrad, None
        return grad.reshape(ctx.shape), None

    view_func.backward = backward
    view_func._megatron_view_backward_reshape_patch = True
    return True
