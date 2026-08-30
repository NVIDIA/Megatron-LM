# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Engine metadata sent to the Dynamo parent at startup."""

from __future__ import annotations

from megatron.core.inference.engine_endpoint import InferenceEngineCapabilities
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine


def engine_metadata(engine: DynamicInferenceEngine, role: str) -> dict[str, int | bool | str]:
    """Return the capabilities Dynamo needs to configure this engine."""

    return {**InferenceEngineCapabilities.from_engine(engine).to_dict(), "role": role}
