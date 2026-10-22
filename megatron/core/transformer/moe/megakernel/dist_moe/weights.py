# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Borrow native TE weights and adapt only MXFP8 scale-factor layout."""

from __future__ import annotations

import torch

from megatron.core.fp8_utils import get_grouped_quantized_members, is_grouped_mxfp8tensor
from megatron.core.transformer.moe.megakernel.mok.weights import (
    _single_grouped_mxfp8_scale_view,
    _storage_view,
    _swizzle_mxfp8_scale,
)


def bf16_view(weight):
    """Expose GroupedTensor payload without materializing its logical wrapper."""
    storage = getattr(weight, "rowwise_data", weight)
    return _storage_view(
        storage, tuple(weight.shape), dtype=torch.bfloat16, name="dist_moe BF16 weight"
    ).detach()


@torch.no_grad()
def prepare_weight(weight, *, use_mxfp8, cached=None):
    """Refresh TE's independent 1D operands and stable Dist scale buffers.

    FP8 primary payloads alias TE storage. BF16 primary parameters use one TE
    grouped quantization workspace, updated in place once per optimizer step.
    Neither mode creates a BF16 shadow of a quantized primary parameter.
    """
    if not use_mxfp8:
        return bf16_view(weight), None
    import dist_moe

    cache = {} if cached is None else cached
    if is_grouped_mxfp8tensor(weight):
        operands = weight
    else:
        import transformer_engine_torch as tex
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

        if "quantizer" not in cache:
            quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
            quantizer.optimize_for_gemm = False
            quantizer.internal = False
            cache["quantizer"] = quantizer
        operands = tex.group_quantize(
            bf16_view(weight).flatten(0, 1),
            cache["quantizer"],
            weight.shape[0],
            None,
            output=cache.get("workspace"),
        )
        if getattr(operands, "quantized_tensors", None) is None:
            operands.quantized_tensors = operands.split_into_quantized_tensors()
        cache["workspace"] = operands
    members = get_grouped_quantized_members(operands, create_if_missing=True)
    if any(getattr(member, "_with_gemm_swizzled_scales", False) for member in members):
        raise ValueError("dist_moe requires unswizzled TE 1D MXFP8 scales")
    if any(getattr(member._quantizer, "with_2d_quantization", False) for member in members):
        raise ValueError("dist_moe requires independent TE 1D MXFP8 operands")
    experts, rows, columns = weight.shape
    row = _storage_view(
        operands.rowwise_data,
        tuple(weight.shape),
        dtype=torch.float8_e4m3fn,
        name="dist_moe rowwise",
    )
    column = _storage_view(
        operands.columnwise_data,
        tuple(weight.shape),
        dtype=torch.float8_e4m3fn,
        name="dist_moe columnwise",
    )
    scales = (
        _single_grouped_mxfp8_scale_view(
            operands, "_rowwise_scale_inv", (experts, rows, columns // 32), name="dist_moe rowwise"
        ),
        _single_grouped_mxfp8_scale_view(
            operands,
            "_columnwise_scale_inv",
            (experts, rows // 32, columns),
            name="dist_moe columnwise",
        ).transpose(-2, -1),
    )
    cache["scales"] = tuple(
        _swizzle_mxfp8_scale(
            scale, rows=n, columns=k, out=None if "scales" not in cache else cache["scales"][axis]
        )
        for axis, (scale, n, k) in enumerate(
            ((scales[0], rows, columns), (scales[1], columns, rows))
        )
    )
    prepared = dist_moe.import_block_scaled_weight(
        source=weight.detach(),
        format=dist_moe.BlockScaledFormat.MXFP8_E4M3,
        fprop_data=row,
        fprop_scale=cache["scales"][0]
        .view(torch.float8_e8m0fnu)
        .view(experts * rows, columns // 32),
        dgrad_data=column,
        dgrad_scale=cache["scales"][1]
        .view(torch.float8_e8m0fnu)
        .view(experts * columns, rows // 32),
    )
    return prepared, cache
