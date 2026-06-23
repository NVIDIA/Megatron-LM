# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import dataclasses

from megatron.core.inference.config import InferenceConfig, RequestResolutionMode
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

    def test_request_resolution_mode_default_and_string_coercion(self):
        """Ensure request resolution mode defaults to legacy and accepts strings."""
        assert InferenceConfig().request_resolution_mode == RequestResolutionMode.LEGACY
        assert (
            InferenceConfig(request_resolution_mode="defer").request_resolution_mode
            == RequestResolutionMode.DEFER
        )
