# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CUDA Graph tests for Mamba state-passing context parallelism.

Both graph backends are covered: Megatron's local CUDA Graph lifecycle
(``cuda_graph_impl="local"``) and the Transformer Engine helper
(``cuda_graph_impl="transformer_engine"``). Each test records a graph, replays
it, checks bitwise parity with eager execution on the first replay, and then
changes the input to confirm that the static input buffer is actually updated.
"""

import gc

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from megatron.core import parallel_state
from megatron.core.num_microbatches_calculator import (
    init_num_microbatches_calculator,
    unset_num_microbatches_calculator,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import (
    TECudaGraphHelper,
    create_cudagraphs,
    delete_cuda_graphs,
)
from tests.unit_tests.ssm.mamba_state_passing_cp_utils import (
    STATE_PASSING_CP_MODES,
    MambaModelShape,
    build_mamba_layer,
    select_balanced_cp_shard,
)
from tests.unit_tests.test_utilities import Utils

try:
    import causal_conv1d  # noqa: F401
    import mamba_ssm  # noqa: F401

    from megatron.core.ssm.mamba_mixer import HAVE_MAMBA_STATE_PASSING_CP
except ImportError:
    HAVE_MAMBA_STATE_PASSING_CP = False

pytestmark = [
    pytest.mark.internal,
    pytest.mark.skipif(
        not HAVE_MAMBA_STATE_PASSING_CP, reason="mamba_ssm and causal_conv1d are required"
    ),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]

CP_SIZE = 2
INPUT_GRAD_RTOL = 1e-3
INPUT_GRAD_ATOL = 3e-6


class _DecoderShell(nn.Module):
    """Minimal decoder interface used by ``TECudaGraphHelper`` discovery."""

    def __init__(self, layer):
        super().__init__()
        self.layers = nn.ModuleList([layer])


class _ModelChunkShell(nn.Module):
    """Minimal training model chunk required by ``TECudaGraphHelper``."""

    def __init__(self, layer, config):
        super().__init__()
        self.config = config
        self.decoder = _DecoderShell(layer)

    def zero_grad_buffer(self):
        """Match the training loop's interface for clearing gradient buffers."""
        self.zero_grad(set_to_none=True)


def _unwrap_output(output):
    """Normalize graph replay's optional one-tensor tuple to a tensor."""
    if isinstance(output, tuple):
        assert len(output) == 1
        return output[0]
    return output


def _assert_module_grads_finite(module):
    for name, parameter in module.named_parameters():
        assert (
            parameter.grad is None or torch.isfinite(parameter.grad).all()
        ), f"non-finite gradient for {name}"


def _setup(mode, cuda_graph_impl, warmup_steps):
    """Initialize CP, build a graphed MambaLayer, and return it with its local input."""
    Utils.initialize_model_parallel(context_parallel_size=CP_SIZE)
    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234, te_rng_tracker=True, force_reset_rng=True)
    cp_group = parallel_state.get_context_parallel_group()
    tp_group = parallel_state.get_tensor_model_parallel_group()
    device = torch.device(torch.cuda.current_device())

    shape = MambaModelShape()
    sequence_length = shape.chunk_size * 2 * CP_SIZE
    batch_size = 1
    layer, config = build_mamba_layer(
        shape,
        cp_group,
        tp_group,
        load_balancing=mode,
        cuda_graph_impl=cuda_graph_impl,
        cuda_graph_warmup_steps=warmup_steps,
    )
    generator = torch.Generator(device=device).manual_seed(5678)
    global_input = torch.randn(
        sequence_length,
        batch_size,
        shape.hidden_size,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    local_input = select_balanced_cp_shard(global_input, cp_group.rank(), CP_SIZE)

    # Cudagraph backward capture accumulates wgrads into ``main_grad``, which
    # DDP normally provides; this test drives a bare module, so create them here
    # (the same setup tests/unit_tests/transformer/test_cuda_graphs.py uses).
    for parameter in layer.parameters():
        parameter.main_grad = torch.zeros_like(parameter)

    return layer, config, cp_group, local_input, sequence_length, batch_size


def _replay_and_check(layer, local_input, eager_output, eager_input_grad):
    """Replay the graph twice, checking eager parity and static-buffer refresh."""
    replay_outputs = []
    for replay_index in range(2):
        layer.zero_grad(set_to_none=True)
        replay_input = (local_input * (1.0 + 0.01 * replay_index)).detach().requires_grad_(True)
        replay_output = _unwrap_output(layer(hidden_states=replay_input, attention_mask=None))
        replay_output.float().square().mean().backward()
        torch.cuda.synchronize()

        assert torch.isfinite(replay_output).all()
        assert torch.isfinite(replay_input.grad).all()
        _assert_module_grads_finite(layer)
        if replay_index == 0:
            torch.testing.assert_close(replay_output, eager_output, rtol=0, atol=0)
            torch.testing.assert_close(
                replay_input.grad, eager_input_grad, rtol=INPUT_GRAD_RTOL, atol=INPUT_GRAD_ATOL
            )
        replay_outputs.append(replay_output.detach().clone())

    assert not torch.equal(
        replay_outputs[0], replay_outputs[1]
    ), "CUDA Graph replay did not consume the changed input"


@pytest.mark.parametrize("mode", STATE_PASSING_CP_MODES)
def test_local_cuda_graph(mode):
    """Megatron's local CUDA Graph lifecycle must capture and replay the CP path."""
    if Utils.world_size % CP_SIZE != 0:
        pytest.skip(f"world size {Utils.world_size} is not a multiple of {CP_SIZE}")
    layer, _, cp_group, local_input, _, _ = _setup(mode, "local", warmup_steps=2)
    runner = None
    try:
        # A normal Megatron call records the runners and establishes eager references.
        recorded_input = local_input.detach().clone().requires_grad_(True)
        recorded_output = layer(hidden_states=recorded_input, attention_mask=None)
        recorded_output.float().square().mean().backward()
        eager_output = recorded_output.detach().clone()
        eager_input_grad = recorded_input.grad.detach().clone()
        torch.cuda.synchronize()
        dist.barrier(group=cp_group)

        assert len(layer.cudagraph_manager.cudagraph_runners) == 1
        runner = layer.cudagraph_manager.cudagraph_runners[0]
        assert runner.fwd_graph_recorded and runner.bwd_graph_recorded

        create_cudagraphs()
        torch.cuda.synchronize()
        dist.barrier(group=cp_group)
        assert runner.cudagraph_created
        assert runner.fwd_graph is not None and runner.bwd_graph is not None

        _replay_and_check(layer, local_input, eager_output, eager_input_grad)
        dist.barrier(group=cp_group)
    finally:
        # Release the graph execs and the global cudagraph record before
        # destroying the NCCL process groups. Leaving the global record in place
        # makes the next test see `cudagraph_created` with no matching runners.
        torch.cuda.synchronize()
        delete_cuda_graphs()
        del layer, runner
        gc.collect()
        torch.cuda.empty_cache()
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("mode", STATE_PASSING_CP_MODES)
def test_te_cuda_graph(mode):
    """The Transformer Engine graph helper must capture and replay the CP path."""
    if Utils.world_size % CP_SIZE != 0:
        pytest.skip(f"world size {Utils.world_size} is not a multiple of {CP_SIZE}")
    warmup_steps = 2
    layer, config, cp_group, local_input, sequence_length, batch_size = _setup(
        mode, "transformer_engine", warmup_steps
    )
    helper = None
    try:
        init_num_microbatches_calculator(
            rank=torch.distributed.get_rank(),
            rampup_batch_size=None,
            global_batch_size=batch_size,
            micro_batch_size=batch_size,
            data_parallel_size=1,
            decrease_batch_size_if_needed=False,
        )
        model_chunk = _ModelChunkShell(layer, config).to(torch.cuda.current_device())

        eager_output = None
        eager_input_grad = None
        for _ in range(warmup_steps):
            layer.zero_grad(set_to_none=True)
            eager_input = local_input.detach().clone().requires_grad_(True)
            eager_result = layer(hidden_states=eager_input, attention_mask=None)
            eager_result.float().square().mean().backward()
            torch.cuda.synchronize()
            eager_output = eager_result.detach().clone()
            eager_input_grad = eager_input.grad.detach().clone()

        # Training releases the previous iteration's autograd graph before TE
        # capture. Keeping these tensors alive makes an old AccumulateGrad node
        # retain its original stream, which can invalidate capture.
        del eager_result, eager_input
        layer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.synchronize()
        dist.barrier(group=cp_group)

        helper = TECudaGraphHelper(
            model=[model_chunk],
            config=config,
            seq_length=sequence_length,
            micro_batch_size=batch_size,
            optimizers=[],
        )
        assert helper.flattened_callables == [layer]
        helper.create_cudagraphs()
        torch.cuda.synchronize()
        dist.barrier(group=cp_group)
        assert helper.capture_finished() and helper.graphs_created()
        assert len(layer.cuda_graphs) == 1

        _replay_and_check(layer, local_input, eager_output, eager_input_grad)
        dist.barrier(group=cp_group)
    finally:
        if helper is not None:
            helper.delete_cuda_graphs()
        delete_cuda_graphs()
        del layer
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        unset_num_microbatches_calculator()
        Utils.destroy_model_parallel()
