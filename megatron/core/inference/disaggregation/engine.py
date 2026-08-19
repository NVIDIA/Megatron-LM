# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Dynamic inference engine composed with KV-cache handoff behavior."""

from megatron.core.inference.disaggregation.inference_state_handoff import (
    InferenceStateHandoffMixin,
)
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine


class DisaggDynamicInferenceEngine(InferenceStateHandoffMixin, DynamicInferenceEngine):
    """Dynamic inference engine with prefill/decode KV-cache handoff support.

    Used by both control planes: the Dynamo integration and the
    coordinator-native 2-hop mode.
    """
