# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepSeek-V4 checkpoint-format resync adapter."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

import torch

from megatron.lite.primitive.quantization.block_fp8 import quantize_block_fp8
from megatron.lite.primitive.quantization.mxfp4 import quantize_mxfp4

_EXPERT_DTYPES = {"fp4", "fp8"}


def _scale_name(weight_name: str) -> str:
    return f"{weight_name[:-7]}.scale"


def _matches_prefix(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(f"{prefix}.")


def is_routed_expert(name: str) -> bool:
    return ".ffn.experts." in name and ".shared_experts." not in name


def is_release_unquantized_weight(name: str) -> bool:
    """Match the unscaled families in the official V4 Flash checkpoint index."""
    if name in {"embed.weight", "head.weight", "norm.weight"}:
        return True
    if name.endswith("norm.weight") or name.endswith(".ffn.gate.weight"):
        return True
    if ".attn.compressor." in name:
        return True
    if ".attn.indexer." in name and not name.endswith(".attn.indexer.wq_b.weight"):
        return True
    return False


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
) -> Iterator[tuple[str, torch.Tensor]]:
    """Convert gathered DS4 BF16 weights to original checkpoint representation."""
    expert_dtype, block_shape, ignored = _quantization_contract(config, resync_config)
    # vLLM's FP8 expert online-reload path materializes block scales as
    # float32 ``weight_scale_inv`` tensors.  Keep W8 resync scales in that
    # representation; serializing them as UE8M0 causes a large online-reload
    # regression even though W4/MXFP4 legitimately uses UE8M0 scales.
    fp8_scale_format = "e8m0" if expert_dtype == "fp4" else "float32"

    for name, tensor in weights:
        if (
            not name.endswith(".weight")
            or tensor.ndim < 2
            or not tensor.dtype.is_floating_point
            or is_release_unquantized_weight(name)
            or any(_matches_prefix(name, prefix) for prefix in ignored)
        ):
            yield name, tensor
            continue

        if is_routed_expert(name) and expert_dtype == "fp4":
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
