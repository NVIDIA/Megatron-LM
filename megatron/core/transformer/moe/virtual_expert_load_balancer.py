# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compatibility imports for the upstream MoonEP public entry points.

The aliases share one class and one cache: compatibility callers cannot create
an independent set of runtime arenas or teardown state.
"""

from megatron.core.transformer.moe.moe_scheduler_plan import (
    VirtualExpertPlan,
    plan_virtual_expert_routes,
)
from megatron.core.transformer.moe.moonep_execution_runtime import MoonEPExecutionRuntime

VirtualExpertLoadBalancer = MoonEPExecutionRuntime

__all__ = ["VirtualExpertLoadBalancer", "VirtualExpertPlan", "plan_virtual_expert_routes"]
