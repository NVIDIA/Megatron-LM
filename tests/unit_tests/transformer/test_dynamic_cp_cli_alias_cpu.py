# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse

import pytest

from megatron.training.arguments import (
    _normalize_dynamic_context_parallel_args,
    add_megatron_arguments,
)


def test_deprecated_dynamic_cp_alias_reaches_runtime_args():
    parser = add_megatron_arguments(argparse.ArgumentParser())
    args = parser.parse_args(["--hybrid-context-parallel"])
    with pytest.warns(DeprecationWarning, match="dynamic-context-parallel"):
        _normalize_dynamic_context_parallel_args(args)
    assert args.dynamic_context_parallel is True
    assert args.hybrid_context_parallel is False
    _normalize_dynamic_context_parallel_args(args)
    assert args.dynamic_context_parallel is True


def test_duplicate_dynamic_cp_spellings_are_rejected():
    parser = add_megatron_arguments(argparse.ArgumentParser())
    args = parser.parse_args(["--hybrid-context-parallel", "--dynamic-context-parallel"])
    with pytest.raises(ValueError, match="Cannot set both"):
        _normalize_dynamic_context_parallel_args(args)


def test_model_parallel_config_accepts_registered_dynamic_scheduler():
    from megatron.core.model_parallel_config import ModelParallelConfig

    config = ModelParallelConfig(dynamic_context_parallel=True, max_seqlen_per_dp_cp_rank=128)
    assert config.sequence_packing_scheduler == "default_dynamic_cp"
    assert config.variable_seq_lengths is True


def test_model_parallel_config_dynamic_scheduler_requires_length_bound():
    from megatron.core.model_parallel_config import ModelParallelConfig

    with pytest.raises(ValueError, match="max_seqlen_per_dp_cp_rank"):
        ModelParallelConfig(dynamic_context_parallel=True)
