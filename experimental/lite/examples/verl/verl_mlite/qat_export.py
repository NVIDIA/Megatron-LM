# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MLite-owned QAT weight export for rollout synchronization.

This module intentionally depends only on MLite primitives.  VERL's ModelOpt
helpers are a private integration surface and have moved between releases.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from fnmatch import fnmatch
from typing import Any

import torch
from megatron.lite.primitive.quantization.mxfp4 import MXFP4_BLOCK_SIZE, quantize_mxfp4

# These are HF checkpoint paths, deliberately distinct from QATSpec's
# Megatron-module component defaults.  An enabled rollout exporter that omits
# ``ignore_patterns`` must still leave fragile output, embedding, and router
# tensors in BF16; callers can explicitly provide a different list when their
# checkpoint naming requires it.
_DEFAULT_IGNORE_PATTERNS = (
    "lm_head",
    "embed_tokens",
    "re:.*mlp.gate$",
)


def _get_field(config: Any, name: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _is_ignored(name: str, patterns: list[str]) -> bool:
    module_name = name.removesuffix(".weight")
    for pattern in patterns:
        if pattern.startswith("re:"):
            if re.search(pattern[3:], module_name):
                return True
        elif pattern in module_name or fnmatch(module_name, pattern):
            return True
    return False


def _process_weights(
    weights: Iterator[tuple[str, torch.Tensor]], ignore_patterns: list[str]
) -> Iterator[tuple[str, torch.Tensor]]:
    for name, weight in weights:
        if "_quantizer." in name:
            continue
        if (
            not name.endswith(".weight")
            or "norm" in name
            or _is_ignored(name, ignore_patterns)
        ):
            yield name, weight
            continue

        packed, scale = quantize_mxfp4(weight)
        yield name, packed.view(torch.uint8)
        yield name.removesuffix(".weight") + ".weight_scale", scale.view(torch.uint8)


def export_qat_weights(
    weights: Iterator[tuple[str, torch.Tensor]], qat_config: Any
) -> Iterator[tuple[str, torch.Tensor]]:
    """Wrap an HF-named BF16 stream with MLite's MXFP4 serializer.

    MLite owns fake quantization during training, so ModelOpt module metadata is
    neither available nor needed at this boundary.
    """
    if _get_field(qat_config, "apply_modelopt_fake_quant", True):
        raise ValueError(
            "MLite QAT export requires apply_modelopt_fake_quant=False; "
            "MLite owns training fake quantization"
        )
    mode = str(_get_field(qat_config, "mode", "w4a16")).lower()
    if mode != "mxfp4":
        raise ValueError(f"MLite QAT export only supports mode='mxfp4', got {mode!r}")
    group_size = int(_get_field(qat_config, "group_size", MXFP4_BLOCK_SIZE))
    if group_size != MXFP4_BLOCK_SIZE:
        raise ValueError(
            f"MXFP4 QAT export requires group_size={MXFP4_BLOCK_SIZE}, got {group_size}"
        )
    patterns = [
        str(pattern)
        for pattern in _get_field(
            qat_config, "ignore_patterns", _DEFAULT_IGNORE_PATTERNS
        )
    ]
    return _process_weights(weights, patterns)


__all__ = ["export_qat_weights"]
