# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Dynamo-specific dynamic inference engine extensions."""

from megatron.core.inference.disaggregation.inference_state_handoff import (
    InferenceStateHandoffMixin,
)
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine


class DynamoDynamicInferenceEngine(InferenceStateHandoffMixin, DynamicInferenceEngine):
    """Dynamic inference engine with Dynamo KV/state handoff support."""
