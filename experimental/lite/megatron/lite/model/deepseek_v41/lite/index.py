# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Relay owner index projections; Core remains responsible for scoring/top-k."""
from typing import Any

import torch
from torch import nn

from megatron.lite.model.deepseek_v41.lite.state import KVSourceState


class IndexProjection(nn.Module):
    """Preserve owner q/weights without registering reader projection parameters.

    Core dev 7b150acf CSAIndexer uses linear_wq_b.forward and
    linear_weights_proj.forward before calling cp_utils.compute_cp_indexer_topk.
    Their tuple-return contract is tested with the native scorer. Balanced
    indexer projection and quantized projection modes need separate validation.
    """

    def __init__(
        self, linear: nn.Module | None, state: KVSourceState, layer: int, owner: int, channel: str
    ) -> None:
        super().__init__()
        if not 0 <= owner <= layer or channel not in {"index_q", "index_weights"}:
            raise ValueError("Invalid DS4.1 index source or projection channel")
        if (linear is not None) != (layer == owner):
            raise ValueError("Only the index owner may register a projection")
        self.linear = linear
        self.state, self.layer, self.owner, self.channel = state, layer, owner, channel

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        geometry = (self.channel, tuple(x.shape), x.dtype, x.device, torch.is_grad_enabled())
        return self.state.resolve(
            self.layer, self.owner, geometry, lambda: self.linear(x, *args, **kwargs)
        )

    def backward_dw(self) -> None:
        """Leave deferred owner weight gradients to the registered owner only."""
        if self.linear is not None and hasattr(self.linear, "backward_dw"):
            self.linear.backward_dw()
