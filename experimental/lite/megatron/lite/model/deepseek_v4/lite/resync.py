# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepSeek-V4 checkpoint-format resync adapter."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

import torch

from megatron.lite.model.deepseek_v4.quantization import (
    is_release_fp32_control,
    is_release_unquantized_weight,
    requantize_block_fp8_weight,
)
from megatron.lite.primitive.quantization.block_fp8 import quantize_block_fp8
from megatron.lite.primitive.quantization.mxfp4 import quantize_mxfp4

_EXPERT_DTYPES = {"fp4", "fp8"}


def _scale_name(weight_name: str) -> str:
    return f"{weight_name[:-7]}.scale"


def _matches_prefix(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(f"{prefix}.")


def is_routed_expert(name: str) -> bool:
    return ".ffn.experts." in name and ".shared_experts." not in name


def _quantization_contract(
    config: Any, resync_config: Mapping[str, Any] | None = None
) -> tuple[str, tuple[int, int], tuple[str, ...]]:
    quant_config = getattr(config, "quantization_config", None)
    if not isinstance(quant_config, dict) or not quant_config:
        raise ValueError("DeepSeek-V4 checkpoint resync requires quantization_config")
    options = dict(resync_config or {})
    unsupported = sorted(options.keys() - {"expert_dtype"})
    if unsupported:
        raise ValueError(f"unsupported DeepSeek-V4 resync_config keys: {unsupported}")
    expert_dtype = (
        options.get("expert_dtype")
        or getattr(config, "expert_dtype", None)
        or quant_config.get("expert_dtype", "fp4")
    )
    if expert_dtype not in _EXPERT_DTYPES:
        raise ValueError(
            f"unsupported DeepSeek-V4 expert_dtype={expert_dtype!r}; "
            f"expected one of {sorted(_EXPERT_DTYPES)}"
        )
    raw_block_shape = quant_config.get("weight_block_size", (128, 128))
    if len(raw_block_shape) != 2:
        raise ValueError(
            f"weight_block_size must have two dimensions, got {raw_block_shape}"
        )
    block_shape = tuple(int(value) for value in raw_block_shape)
    ignored = tuple(
        quant_config.get("ignored_layers")
        or quant_config.get("modules_to_not_convert")
        or ()
    )
    return expert_dtype, block_shape, ignored


def export_resync_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    config: Any,
    *,
    resync_config: Mapping[str, Any] | None = None,
    source_scales: Mapping[str, torch.Tensor] | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Convert gathered DS4 BF16 weights to original checkpoint representation."""
    expert_dtype, block_shape, ignored = _quantization_contract(config, resync_config)
    # Match the serialized block-FP8 checkpoint contract on the reload wire:
    # scales are FP32 even when scale_fmt=ue8m0 asks vLLM to cast and pack them
    # as E8M0 during post-load processing.
    fp8_scale_format = "e8m0" if expert_dtype == "fp4" else "float32"

    for name, tensor in weights:
        if is_release_fp32_control(name) and tensor.dtype != torch.float32:
            raise TypeError(
                f"DeepSeek-V4 FP32 control {name} was exported as {tensor.dtype}"
            )
        if (
            not name.endswith(".weight")
            or tensor.ndim < 2
            or not tensor.dtype.is_floating_point
            or is_release_unquantized_weight(name)
            or any(_matches_prefix(name, prefix) for prefix in ignored)
        ):
            yield name, tensor
            continue

        # Quantized deployment weights are computed from the BF16 value used by
        # the actor forward. FSDP2 may expose an FP32 storage shard here, while
        # dist-opt ordinarily already exposes BF16.
        tensor = tensor.to(torch.bfloat16)

        fixed_scale = None if source_scales is None else source_scales.get(name)
        if (
            fixed_scale is not None
            and is_routed_expert(name)
            and expert_dtype == "fp4"
        ):
            raise RuntimeError("routed MXFP4 weights cannot carry block-FP8 source scales")
        if fixed_scale is not None:
            canonical = requantize_block_fp8_weight(
                tensor,
                fixed_scale.to(tensor.device, dtype=torch.float32).contiguous(),
            )
            quantized, scale = canonical.qweight, canonical.scales
            if fp8_scale_format == "e8m0":
                scale = (
                    (scale.view(torch.int32) >> 23)
                    .to(torch.uint8)
                    .view(torch.float8_e8m0fnu)
                )
        elif is_routed_expert(name) and expert_dtype == "fp4":
            quantized, scale = quantize_mxfp4(tensor)
        else:
            quantized, scale = quantize_block_fp8(
                tensor, block_shape, scale_format=fp8_scale_format
            )
        yield name, quantized
        yield _scale_name(name), scale


__all__ = [
    "export_resync_weights",
    "is_release_unquantized_weight",
    "is_routed_expert",
]
