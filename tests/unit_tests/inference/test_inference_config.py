# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse
import dataclasses

import pytest

from megatron.core.inference.config import AsyncSchedulingMode, InferenceConfig
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

    def test_async_scheduling_defaults_to_legacy(self):
        """The preserved legacy request-update path is the default."""
        config = InferenceConfig()

        assert config.async_scheduling_mode == AsyncSchedulingMode.LEGACY

    def test_async_scheduling_cli_mode(self):
        """The dynamic batching request-update scheduling mode is parsed as a choice."""
        parser = _add_inference_args(argparse.ArgumentParser())

        default_args = parser.parse_args([])
        enabled_args = parser.parse_args(
            ["--inference-dynamic-batching-async-scheduling-mode", "async"]
        )

        assert default_args.inference_dynamic_batching_async_scheduling_mode == "legacy"
        assert enabled_args.inference_dynamic_batching_async_scheduling_mode == "async"

    def test_async_scheduling_bool_flag_is_rejected(self):
        """The old opt-in bool flag is intentionally not accepted."""
        parser = _add_inference_args(argparse.ArgumentParser())

        with pytest.raises(SystemExit):
            parser.parse_args(["--inference-dynamic-batching-async-scheduling"])
