# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys

import pytest

from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.transformer.enums import CudaGraphModule
from megatron.training import arguments as training_arguments
from megatron.training.arguments import parse_args, validate_args
from megatron.training.global_vars import destroy_global_vars


def _validate_simulation_args(monkeypatch, tmp_path, cli_args=None, **overrides):
    destroy_global_vars()
    destroy_num_microbatches_calculator()

    monkeypatch.setattr(training_arguments, "warn_rank_0", lambda *args, **kwargs: None)
    monkeypatch.setattr(training_arguments, "print_rank_0", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test_vpp_simulation_args.py",
            "--simulate-global-step",
            "--simulate-result-dir",
            str(tmp_path),
            *(cli_args or []),
        ],
    )

    args = parse_args()
    args.num_layers = 2
    args.vocab_size = 256
    args.hidden_size = 64
    args.num_attention_heads = 4
    args.max_position_embeddings = 128
    args.seq_length = 128
    args.micro_batch_size = 1

    for key, value in overrides.items():
        setattr(args, key, value)

    return validate_args(args)


def test_simulation_rejects_cuda_graph_impl(monkeypatch, tmp_path):
    with pytest.raises(
        AssertionError,
        match="VPP simulation does not support CUDA graph capture/replay yet",
    ):
        _validate_simulation_args(monkeypatch, tmp_path, ["--cuda-graph-impl", "local"])


def test_simulation_rejects_cuda_graph_modules(monkeypatch, tmp_path):
    with pytest.raises(
        AssertionError,
        match="VPP simulation does not support --cuda-graph-modules yet",
    ):
        _validate_simulation_args(
            monkeypatch,
            tmp_path,
            cuda_graph_modules=[CudaGraphModule.attn],
        )


def test_simulation_rejects_optimizer_cuda_graph(monkeypatch, tmp_path):
    with pytest.raises(
        AssertionError,
        match="VPP simulation does not support --optimizer-cuda-graph",
    ):
        _validate_simulation_args(monkeypatch, tmp_path, ["--optimizer-cuda-graph"])
