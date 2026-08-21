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
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from megatron.core.transformer.transformer_config import TransformerConfig, WideResidualConfig
from megatron.core.transformer.wide_residual_layer import StreamwiseSigmoidWideResidualConnection
from tests.unit_tests.test_utilities import Utils, clear_nvte_env_vars


@pytest.fixture(scope="module", autouse=True)
def model_parallel():
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("Run this test with torchrun --nproc-per-node=2.")
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
    yield
    MTPLossLoggingHelper.tracker = {}
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


def test_wide_residual_mtp_runs_with_tensor_parallelism_and_matching_controller_gradients():
    clear_nvte_env_vars()
    model_parallel_cuda_manual_seed(123)
    MTPLossLoggingHelper.tracker = {}
    config = TransformerConfig(
        num_layers=2,
        mtp_num_layers=1,
        mtp_loss_scaling_factor=0.1,
        hidden_size=64,
        num_attention_heads=4,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        tensor_model_parallel_size=2,
        use_cpu_initialization=True,
        recompute_granularity="selective",
        recompute_modules=["residual_stream"],
        residual_stream_recompute_num_layers=1,
        wide_residual=WideResidualConfig(num_streams=3, learned_retention=True),
    )
    layer_spec = get_gpt_layer_local_spec()
    model = GPTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        mtp_block_spec=get_gpt_mtp_block_spec(
            config=config, spec=layer_spec, use_transformer_engine=False
        ),
        vocab_size=128,
        max_sequence_length=4,
        position_embedding_type="none",
    ).cuda()
    input_ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], device="cuda")
    position_ids = torch.arange(4, device="cuda").unsqueeze(0).expand(2, -1)
    labels = input_ids.roll(-1, dims=1)
    loss_mask = torch.ones_like(labels, dtype=torch.float32)

    loss = model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=None,
        labels=labels,
        loss_mask=loss_mask,
    )
    loss.mean().backward()

    controller_grad = model.decoder.residual_stream_readout.exit_map.logit.grad
    gathered_grads = [torch.empty_like(controller_grad) for _ in range(2)]
    torch.distributed.all_gather(
        gathered_grads, controller_grad, group=parallel_state.get_tensor_model_parallel_group()
    )

    assert model.mtp.layers[0].mtp_model_layer.residual_connection_self_attn is None
    assert model.mtp.layers[0].eh_proj.weight.grad is not None
    assert all(torch.equal(grad, gathered_grads[0]) for grad in gathered_grads[1:])
