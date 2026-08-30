# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Dynamo-specific dynamic inference engine type."""

from megatron.core.inference.disaggregation.engine import StateHandoffDynamicInferenceEngine


class DynamoDynamicInferenceEngine(StateHandoffDynamicInferenceEngine):
    """Dynamic inference engine with Dynamo KV/state handoff support."""
