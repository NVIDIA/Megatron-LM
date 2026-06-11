# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse
import dataclasses

from megatron.core.inference.config import InferenceConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import _add_inference_args


class TestInferenceConfig:
    def test_mutual_exclusivity_with_transformer_config(self):
        """
        Ensure mutual exclusivity between fields in `InferenceConfig` and
        `TransformerConfig`.
        """
        dynamic_inference_config_fields = set(dataclasses.fields(InferenceConfig))
        transformer_config_fields = set(dataclasses.fields(TransformerConfig))
        assert len(dynamic_inference_config_fields.intersection(transformer_config_fields)) == 0

    def test_async_scheduling_defaults_to_disabled(self):
        """Async scheduling is opt-in."""
        config = InferenceConfig()

        assert not config.enable_async_scheduling

    def test_async_scheduling_cli_flag(self):
        """The dynamic batching async scheduling CLI flag is parsed as an opt-in bool."""
        parser = _add_inference_args(argparse.ArgumentParser())

        default_args = parser.parse_args([])
        enabled_args = parser.parse_args(["--inference-dynamic-batching-async-scheduling"])

        assert not default_args.inference_dynamic_batching_async_scheduling
        assert enabled_args.inference_dynamic_batching_async_scheduling
