# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Distributed CP and TP x CP parity tests for wide-residual controllers."""

import copy
import os

import pytest
import torch
from torch import nn

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import (
    _allreduce_non_tensor_model_parallel_grads,
)
from megatron.core.transformer.residual_recompute import (
    build_residual_stream_recompute_plan,
    checkpoint_residual_read,
    checkpoint_residual_write,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.wide_residual_config import WideResidualConfig
from megatron.core.transformer.wide_residual_layer import StreamwiseSigmoidWideResidualConnection
from tests.unit_tests.test_utilities import Utils


class _WideResidualHarness(nn.Module):
    """Exercise read, retention, and write through one differentiable branch."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.connection = StreamwiseSigmoidWideResidualConnection(
            config=config, layer_number=1, branch_name="test", pg_collection=None
        )

    def forward(self, residual_stream: torch.Tensor) -> torch.Tensor:
        branch_input, state = self.connection(residual_stream, operation="read")
        branch_update = torch.tanh(branch_input) + 0.125 * branch_input
        return self.connection(
            branch_update, operation="write", state=state, dropout_probability=0.0, training=False
        )


class _WideResidualReplayHarness(nn.Module):
    """Exercise one complete two-layer residual-stream replay block."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.connections = nn.ModuleList(
            [
                StreamwiseSigmoidWideResidualConnection(
                    config=config, layer_number=layer_number, branch_name="test", pg_collection=None
                )
                for layer_number in range(1, 3)
            ]
        )

    def forward(self, residual_stream: torch.Tensor, *, replay: bool) -> torch.Tensor:
        contexts = build_residual_stream_recompute_plan(2, 2) if replay else [None, None]
        for connection, context in zip(self.connections, contexts):
            if context is None:
                branch_input, state = connection(residual_stream, operation="read")
            else:
                branch_input, state = checkpoint_residual_read(
                    connection, residual_stream, context, fp32_residual_connection=False
                )

            branch_update = torch.tanh(branch_input) + 0.125 * branch_input
            if context is not None and not context.is_block_end:
                residual_stream = checkpoint_residual_write(
                    connection,
                    branch_update,
                    state,
                    context,
                    dropout_probability=0.0,
                    training=False,
                )
            else:
                residual_stream = connection(
                    branch_update,
                    operation="write",
                    state=state,
                    dropout_probability=0.0,
                    training=False,
                )
            if context is not None:
                context.finalize(residual_stream)

        return residual_stream


def _config(
    *, tp_size: int, cp_size: int, sequence_parallel: bool, num_layers: int = 1
) -> TransformerConfig:
    return TransformerConfig(
        num_layers=num_layers,
        hidden_size=8,
        num_attention_heads=2,
        hidden_dropout=0.0,
        tensor_model_parallel_size=tp_size,
        context_parallel_size=cp_size,
        sequence_parallel=sequence_parallel,
        use_cpu_initialization=True,
        wide_residual=WideResidualConfig(
            num_streams=3,
            streamwise_sigmoid_init_scale=0.01,
            learned_retention=True,
            retention_init=0.999,
            retention_max_forget=0.10,
        ),
    )


def _local_residual(config: TransformerConfig) -> torch.Tensor:
    """Create the token shard owned by this CP and optional SP rank."""

    sequence_length = 8
    batch_size = 2
    wide_hidden_size = config.wide_residual.num_streams * config.hidden_size
    residual = torch.linspace(
        -0.75,
        1.25,
        steps=sequence_length * batch_size * wide_hidden_size,
        device=torch.cuda.current_device(),
        dtype=torch.float32,
    ).reshape(sequence_length, batch_size, wide_hidden_size)
    residual = residual.chunk(config.context_parallel_size, dim=0)[
        parallel_state.get_context_parallel_rank()
    ]
    if config.sequence_parallel:
        residual = residual.chunk(config.tensor_model_parallel_size, dim=0)[
            parallel_state.get_tensor_model_parallel_rank()
        ]
    return residual.contiguous()


def _reduce_reference_gradient(
    gradient: torch.Tensor, ddp_model: DistributedDataParallel, *, sequence_parallel: bool
) -> torch.Tensor:
    """Apply the expected CP average followed by the configured TP reduction."""

    expected = gradient.clone()
    torch.distributed.all_reduce(
        expected, op=torch.distributed.ReduceOp.AVG, group=ddp_model.dp_cp_group
    )
    if ddp_model.tp_group.size() > 1:
        tp_op = (
            torch.distributed.ReduceOp.SUM if sequence_parallel else torch.distributed.ReduceOp.AVG
        )
        torch.distributed.all_reduce(expected, op=tp_op, group=ddp_model.tp_group)
    return expected


def _run_parallel_parity(*, tp_size: int, cp_size: int, sequence_parallel: bool) -> None:
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=cp_size,
    )
    try:
        config = _config(tp_size=tp_size, cp_size=cp_size, sequence_parallel=sequence_parallel)
        module = _WideResidualHarness(config).cuda()
        reference = copy.deepcopy(module)
        ddp_model = DistributedDataParallel(
            config,
            DistributedDataParallelConfig(
                overlap_grad_reduce=False, use_distributed_optimizer=False
            ),
            module,
        )

        local_residual = _local_residual(config)
        reference_input = local_residual.detach().clone().requires_grad_(True)
        reference_output = reference(reference_input)
        reference_parameters = tuple(reference.parameters())
        reference_gradients = torch.autograd.grad(
            reference_output.float().square().mean(), (reference_input, *reference_parameters)
        )

        ddp_model.zero_grad_buffer()
        distributed_input = local_residual.detach().clone().requires_grad_(True)
        distributed_output = ddp_model(distributed_input)
        distributed_output.float().square().mean().backward()
        ddp_model.finish_grad_sync()
        _allreduce_non_tensor_model_parallel_grads([ddp_model], config, tp_group=ddp_model.tp_group)

        torch.testing.assert_close(distributed_output, reference_output)
        torch.testing.assert_close(distributed_input.grad, reference_gradients[0])
        for parameter, reference_gradient in zip(module.parameters(), reference_gradients[1:]):
            expected_gradient = _reduce_reference_gradient(
                reference_gradient, ddp_model, sequence_parallel=sequence_parallel
            )
            torch.testing.assert_close(parameter.main_grad, expected_gradient)
    finally:
        Utils.destroy_model_parallel()


def _run_replay_parallel_parity(*, sequence_parallel: bool) -> None:
    """Compare eager and replayed execution under real TP and CP reductions."""

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=2, pipeline_model_parallel_size=1, context_parallel_size=2
    )
    try:
        config = _config(tp_size=2, cp_size=2, sequence_parallel=sequence_parallel, num_layers=2)
        module = _WideResidualReplayHarness(config).cuda()
        ddp_model = DistributedDataParallel(
            config,
            DistributedDataParallelConfig(
                overlap_grad_reduce=False, use_distributed_optimizer=False
            ),
            module,
        )
        local_residual = _local_residual(config)

        def run_backward(*, replay: bool):
            ddp_model.zero_grad_buffer()
            residual = local_residual.detach().clone().requires_grad_(True)
            output = ddp_model(residual, replay=replay)
            output.float().square().mean().backward()
            ddp_model.finish_grad_sync()
            pre_tp_gradients = {
                name: parameter.main_grad.detach().clone()
                for name, parameter in ddp_model.module.named_parameters()
            }
            _allreduce_non_tensor_model_parallel_grads(
                [ddp_model], config, tp_group=ddp_model.tp_group
            )
            post_tp_gradients = {
                name: parameter.main_grad.detach().clone()
                for name, parameter in ddp_model.module.named_parameters()
            }
            return output, residual.grad, pre_tp_gradients, post_tp_gradients

        eager_output, eager_input_grad, eager_pre_tp_gradients, eager_post_tp_gradients = (
            run_backward(replay=False)
        )
        replay_output, replay_input_grad, replay_pre_tp_gradients, replay_post_tp_gradients = (
            run_backward(replay=True)
        )

        torch.testing.assert_close(replay_output, eager_output)
        torch.testing.assert_close(replay_input_grad, eager_input_grad)
        for name in eager_pre_tp_gradients:
            torch.testing.assert_close(
                replay_pre_tp_gradients[name],
                eager_pre_tp_gradients[name],
                msg=lambda details, name=name: f"{name} before TP reduction: {details}",
            )
            assert name in replay_post_tp_gradients
            torch.testing.assert_close(
                replay_post_tp_gradients[name],
                eager_post_tp_gradients[name],
                msg=lambda details, name=name: f"{name} after TP reduction: {details}",
            )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 2,
    reason="Run this test with torchrun --nproc-per-node=2.",
)
def test_streamwise_controller_gradients_match_cp_reference():
    _run_parallel_parity(tp_size=1, cp_size=2, sequence_parallel=False)


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 4,
    reason="Run this test with torchrun --nproc-per-node=4.",
)
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_streamwise_controller_gradients_match_tp_cp_reference(sequence_parallel):
    _run_parallel_parity(tp_size=2, cp_size=2, sequence_parallel=sequence_parallel)


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 4,
    reason="Run this test with torchrun --nproc-per-node=4.",
)
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_residual_stream_replay_matches_eager_with_tp_cp(sequence_parallel):
    _run_replay_parallel_parity(sequence_parallel=sequence_parallel)
