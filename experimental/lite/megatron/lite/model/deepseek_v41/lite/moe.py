# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DS4.1 routing selection over the native Core router."""
from collections.abc import Callable

import torch

from megatron.core.transformer.moe.router import TopKRouter


class SortedTopKPolicy:
    """Select in score order; Core still normalizes and dispatches.

    Core dev 7b150acf: moe_utils.compute_topk delegates through the instance
    get_replay_topk interface. No routes are recorded or replayed, and no
    global replay instance is registered. This preserves the 410b438240
    selection contract without changing Core or requiring deterministic math.
    """

    layer_number = None

    def get_replay_topk(
        self,
        scores: torch.Tensor,
        topk: int,
        num_groups: int | None = None,
        group_topk: int | None = None,
        default_compute_topk: Callable | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if group_topk:
            return default_compute_topk(scores, topk, num_groups, group_topk)
        return torch.topk(scores, k=topk, dim=1, sorted=True)


def bind_sorted_selection(router: TopKRouter) -> None:
    """Attach the policy only to a supported non-hash routing instance."""
    if router.is_hash_layer:
        return
    if router.config.moe_router_fusion or router.use_quantile_balancing:
        raise ValueError("DS4.1 sorted selection requires unfused, non-quantile routing")
    if router.router_replay is not None:
        raise ValueError("DS4.1 sorted selection cannot replace an active replay policy")
    router.router_replay = SortedTopKPolicy()
