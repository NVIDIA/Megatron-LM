# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Distributed tests for replicated streamwise wide-residual controls."""

import os
from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.distributed.finalize_model_grads import (
    _allreduce_non_tensor_model_parallel_grads,
)
from megatron.core.transformer.transformer_config import TransformerConfig, WideResidualConfig
from megatron.core.transformer.wide_residual_layer import StreamwiseSigmoidWideResidualConnection
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(scope="module", autouse=True)
def model_parallel():
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("Run this test with torchrun --nproc-per-node=2.")
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
    yield
    Utils.destroy_model_parallel()


def _config(*, sequence_parallel: bool) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        hidden_dropout=0.0,
        sequence_parallel=sequence_parallel,
        tensor_model_parallel_size=2,
        use_cpu_initialization=True,
        wide_residual=WideResidualConfig(
            num_streams=3,
            streamwise_sigmoid_init_scale=0.01,
            learned_retention=True,
            retention_init=0.999,
            retention_max_forget=0.10,
        ),
    )


@pytest.mark.parametrize(("sequence_parallel", "expected_value"), [(False, 1.5), (True, 3.0)])
def test_streamwise_control_gradients_use_correct_tp_reduction(sequence_parallel, expected_value):
    config = _config(sequence_parallel=sequence_parallel)
    connection = StreamwiseSigmoidWideResidualConnection(
        config=config, layer_number=1, branch_name="test", pg_collection=None
    ).cuda()
    connection.ddp_config = SimpleNamespace(use_megatron_fsdp=False)

    rank_value = float(torch.distributed.get_rank() + 1)
    for parameter in connection.parameters():
        parameter.main_grad = torch.full_like(parameter, rank_value)

    _allreduce_non_tensor_model_parallel_grads(
        [connection], config, tp_group=parallel_state.get_tensor_model_parallel_group()
    )

    for parameter in connection.parameters():
        assert torch.equal(
            parameter.main_grad, torch.full_like(parameter.main_grad, expected_value)
        )
