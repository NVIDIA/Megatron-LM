# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Expose upstream MoonEP execution through the common scheduler ownership API."""

import weakref

from megatron.core.transformer.moe.moe_scheduler import (
    ExpertDispatch,
    MoELoadPlanner,
    MoEPlacementResult,
)


class UpstreamMoonEPPlanner(MoELoadPlanner):
    """Planning runs at the upstream compact-route preprocessing boundary."""

    planner_name = "moon_ep"

    def update_placement(self, probs, routing_map, context, *, tokens_per_expert=None):
        """The execution runtime retains the compact placement through backward."""
        return None, None, MoEPlacementResult()

    def reroute(self, probs, routing_map, placement_result, context):
        """Preserve compact scores until the upstream dispatcher remaps their IDs."""
        return routing_map, probs


class UpstreamMoonEPExpertDispatch(ExpertDispatch):
    """Borrow the HybridEP manager; it owns hooks, streams, weights, and gradients.

    Do not register the same expert parameters or attach a second set of hooks.
    This adapter keeps the legacy configuration on its accepted execution path.
    """

    dispatcher_name = "upstream_moon_ep"
    owns_token_preprocessing = True

    def __init__(self, runtime):
        super().__init__()
        self._runtime = weakref.ref(runtime)

    def bind_experts(self, experts):
        """Bind exactly one expert runtime to the layer."""
        self._runtime().bind_experts(experts)

    def supports(self, expert_placement, context):
        """The upstream runtime owns placement dispatch internally."""
        return False

    def dispatch(self, experts, expert_placement, context):
        """Reject accidental activation of the second replica execution path."""
        raise RuntimeError("Upstream MoonEP dispatch is owned by compact token preprocessing.")

    def finalize(self, context):
        """The upstream layer-output hook retains its own backward plan."""
