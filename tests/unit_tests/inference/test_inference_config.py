# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import dataclasses

from megatron.core.inference.config import InferenceConfig
from megatron.core.transformer.transformer_config import TransformerConfig


class TestInferenceConfig:
    def test_mutual_exclusivity_with_transformer_config(self):
        """
        Ensure mutual exclusivity between fields in `InferenceConfig` and
        `TransformerConfig`.
        """
        dynamic_inference_config_fields = set(dataclasses.fields(InferenceConfig))
        transformer_config_fields = set(dataclasses.fields(TransformerConfig))
        assert len(dynamic_inference_config_fields.intersection(transformer_config_fields)) == 0

    def test_async_scheduling_config_default(self):
        """Async scheduling is opt-in."""
        config = InferenceConfig()
        assert config.enable_async_scheduling is False

    def test_async_scheduling_config_can_enable(self):
        """Async scheduling can be enabled explicitly."""
        config = InferenceConfig(enable_async_scheduling=True)
        assert config.enable_async_scheduling is True
