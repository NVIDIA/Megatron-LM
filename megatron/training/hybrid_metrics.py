# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Helpers for reporting metrics from a resolved hybrid architecture."""

from dataclasses import dataclass
from typing import Any, Iterator

_LAYER_TYPE_ALIASES = {
    "M": "mamba",
    "G": "gdn",
    "*": "attention",
    "D": "dsa",
    "+": "mla",
    "-": "mlp",
    "E": "moe",
}


def get_resolved_hybrid_architecture(args: Any) -> Any | None:
    """Return a model-provided direct architecture, if one was installed."""

    architecture = getattr(args, "resolved_hybrid_architecture", None)
    if architecture is None or getattr(architecture, "source", None) != "direct":
        return None
    return architecture


def get_hybrid_layer_type(layer: Any) -> str:
    """Return the stable semantic type name for a resolved hybrid layer."""

    layer_type = layer.layer_type
    if not isinstance(layer_type, str):
        layer_type = getattr(layer_type, "value", layer_type)
    layer_type = str(layer_type)
    return _LAYER_TYPE_ALIASES.get(layer_type, layer_type.lower())


def iter_resolved_hybrid_layers(architecture: Any) -> Iterator[Any]:
    """Iterate main layers followed by each repeated MTP-depth template."""

    yield from architecture.main_layers
    for _ in range(architecture.mtp_num_layers):
        yield from architecture.mtp_layers


@dataclass(frozen=True)
class HybridMoEMetricMetadata:
    """Arguments needed to size and normalize global MoE metric tensors."""

    num_layers: int
    moe_layer_freq: list[int]
    mtp_num_layers: int
    num_moe_layers: int


def get_hybrid_moe_metric_metadata(architecture: Any) -> HybridMoEMetricMetadata:
    """Derive MoE logging metadata from per-occurrence semantic layer types.

    Direct hybrid models assign metric slots over the fully expanded global
    architecture, including every layer in every MTP depth. Consequently MTP
    has already been incorporated into ``num_layers`` and ``moe_layer_freq``;
    ``mtp_num_layers`` is zero to disable the tracker's legacy implicit MTP
    expansion.
    """

    layer_types = [
        get_hybrid_layer_type(layer) for layer in iter_resolved_hybrid_layers(architecture)
    ]
    moe_layer_freq = [int(layer_type == "moe") for layer_type in layer_types]
    return HybridMoEMetricMetadata(
        num_layers=len(layer_types),
        moe_layer_freq=moe_layer_freq,
        mtp_num_layers=0,
        num_moe_layers=sum(moe_layer_freq),
    )
