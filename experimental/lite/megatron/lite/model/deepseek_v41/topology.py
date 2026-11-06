# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DS4.1 source ownership; Core owns layout, communication and scheduling."""
from dataclasses import dataclass

from megatron.lite.model.deepseek_v41.config import DEFAULT_RATIOS, DeepseekV41Config


@dataclass(frozen=True)
class LayerSources:
    kv: int | None
    index: int | None
    candidate: bool


def layer_sources(config: DeepseekV41Config) -> tuple[LayerSources, ...]:
    n = config.num_hidden_layers
    ratios = config.compress_ratios or list(DEFAULT_RATIOS)
    if type(n) is int and not config.compress_ratios and n > 40:
        raise ValueError("DS4.1 default ratios support at most 40 text layers; MTP is unsupported")
    if type(n) is not int or n <= 0 or len(ratios) < n:
        raise ValueError("DS4.1 needs positive layer count and a ratio for every layer")
    sources = (config.kv_source_layer_ids, config.index_source_layer_ids)
    for ids in sources:
        if any(type(i) is not int or i < 0 for i in ids) or list(ids) != sorted(set(ids)):
            raise ValueError("DS4.1 owner layer ids must be sorted unique nonnegative integers")
    candidate = config.candidate_source_layer_id
    if 0 <= candidate and (
        candidate not in config.index_source_layer_ids
        or candidate >= len(ratios)
        or ratios[candidate] != 1
    ):
        # Release (layer 20): the publisher is a ratio-one index owner. A publisher
        raise ValueError("DS4.1 candidate publisher must be a ratio-one index source layer")
    result = []
    for i, ratio in enumerate(ratios[:n]):
        if type(ratio) is not int or ratio not in (0, 1, 2):
            raise ValueError("DS4.1 compression ratios are literal 0, 1 or 2")
        owners = tuple(max((s for s in ids if s <= i), default=None) for ids in sources)
        if ratio and any(s is None or ratios[s] != ratio for s in owners):
            raise ValueError(f"DS4.1 layer {i} has no compatible KV/index owner")
        uses_candidate = candidate >= 0 and i >= candidate
        if ratio == 1 and not uses_candidate:
            raise ValueError("DS4.1 ratio-1 layers require a candidate publisher")
        result.append(LayerSources(*(owners if ratio else (None, None)), uses_candidate))
    return tuple(result)


def validate_build(
    config: DeepseekV41Config, *, context_parallel_size: int
) -> tuple[LayerSources, ...]:
    """Call before process-group initialization or allocating model parameters."""
    if type(context_parallel_size) is not int or context_parallel_size < 1:
        raise ValueError("Context parallel size must be a positive integer")
    policies = layer_sources(config)
    ratios = config.compress_ratios or list(DEFAULT_RATIOS)
    if any(p.candidate and ratios[i] != 1 for i, p in enumerate(policies)):
        raise ValueError("DS4.1 candidate policy requires the ratio-one model path")
    return policies
