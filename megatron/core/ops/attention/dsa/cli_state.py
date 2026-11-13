# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Per-forward state for sparse DSA cross-layer indexer distillation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

    from megatron.core.ops.attention.dsa.cli_target import GroupedCLITargetAccumulator

_HOLDER_ATTR = "_dsa_cross_layer_indexer_state_holder"


@dataclass
class DSACrossLayerIndexerState:
    """Original owner selection and no-grad teacher statistics for one microbatch."""

    deferred_cli_target: GroupedCLITargetAccumulator | None = None
    topk_indices: Tensor | None = None
    topk_length: Tensor | None = None

    def release_forward_values(self) -> None:
        """Drop temporary tensors after the owner has consumed all teacher targets."""
        self.deferred_cli_target = None
        self.topk_indices = None
        self.topk_length = None


def get_dsa_cross_layer_state_holder(carrier: object) -> dict[int, list[DSACrossLayerIndexerState]]:
    """Return CLI state on the caller's existing per-forward IndexShare carrier."""
    holder = getattr(carrier, _HOLDER_ATTR, None)
    if holder is None:
        holder = {}
        setattr(carrier, _HOLDER_ATTR, holder)
    return holder


def cli_group_size(layer_number: int, num_layers: int, offset: int, frequency: int) -> int:
    """Count the owner and reuse layers, including partial groups and prefix owners."""
    if not 1 <= layer_number <= num_layers or frequency < 1 or offset < 0:
        raise ValueError(
            "CLI requires valid layer numbers, positive frequency and nonnegative offset"
        )
    offset = max(1, offset)
    if layer_number < offset:
        return 1
    owner = layer_number - (layer_number - offset) % frequency
    return min(frequency, num_layers - owner + 1)
