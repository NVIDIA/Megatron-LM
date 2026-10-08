# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import dataclasses
from argparse import ArgumentParser
from types import SimpleNamespace

import pytest

from megatron.core.inference.config import AsyncScheduleMode, InferenceConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import _add_inference_args
from megatron.training.config.inference_config import InferenceSetupConfig


class TestInferenceConfig:
    def test_mutual_exclusivity_with_transformer_config(self):
        """
        Ensure mutual exclusivity between fields in `InferenceConfig` and
        `TransformerConfig`.
        """
        dynamic_inference_config_fields = set(dataclasses.fields(InferenceConfig))
        transformer_config_fields = set(dataclasses.fields(TransformerConfig))
        assert len(dynamic_inference_config_fields.intersection(transformer_config_fields)) == 0

    @pytest.mark.parametrize(
        "async_sched_mode, expected",
        [
            (None, AsyncScheduleMode.LEGACY),
            ("serial", AsyncScheduleMode.SERIAL),
            (AsyncScheduleMode.SERIAL, AsyncScheduleMode.SERIAL),
            ("overlap", AsyncScheduleMode.OVERLAP),
            (AsyncScheduleMode.OVERLAP, AsyncScheduleMode.OVERLAP),
        ],
    )
    def test_async_sched_mode_default_and_coercion(self, async_sched_mode, expected):
        """Ensure async scheduling mode defaults to legacy and accepts strings."""
        kwargs = {} if async_sched_mode is None else {"async_sched_mode": async_sched_mode}
        assert InferenceConfig(**kwargs).async_sched_mode == expected

    def test_async_sched_mode_rejects_invalid_value(self):
        """Ensure invalid async scheduling modes fail during config construction."""
        with pytest.raises(ValueError):
            InferenceConfig(async_sched_mode="invalid")

    def test_async_sched_argparse_plumbing(self):
        """Ensure the CLI exposes async scheduling mode."""
        parser = _add_inference_args(ArgumentParser())
        args = parser.parse_args(["--inference-dynamic-batching-async-sched-mode", "overlap"])
        assert args.inference_dynamic_batching_async_sched_mode == "overlap"

    def test_inference_setup_config_maps_async_sched_mode(self):
        """Ensure declarative inference config maps async scheduling mode to runtime config."""
        model = SimpleNamespace(
            position_embedding_type="rope",
            max_sequence_length=4096,
            pg_collection="pg",
            decoder=SimpleNamespace(layer_type_list=None),
        )
        setup_config = InferenceSetupConfig(inference_dynamic_batching_async_sched_mode="overlap")

        inference_config = setup_config.to_inference_config(
            model=model,
            kv_cache_management_mode="persist",
            static_kv_memory_pointers=False,
            enable_cuda_graphs=False,
            verbose=False,
        )

        assert inference_config.async_sched_mode == AsyncScheduleMode.OVERLAP
