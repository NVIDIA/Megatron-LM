# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import dataclasses

import pytest

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

    def test_async_overlap_rollout_defaults_are_conservative(self):
        config = InferenceConfig()

        assert config.enable_async_overlap_architecture is False
        assert config.async_overlap_queue_depth == 1
        assert config.async_overlap_debug_checks is False

    def test_async_overlap_queue_depth_must_be_positive(self):
        with pytest.raises(ValueError, match="async_overlap_queue_depth"):
            InferenceConfig(async_overlap_queue_depth=0)
