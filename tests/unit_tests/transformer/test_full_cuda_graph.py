# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import warnings
from unittest.mock import Mock, patch

import pytest
import torch
from pytest_mock import mocker

import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core import ModelParallelConfig
from megatron.core.full_cuda_graph import (
    FullCudaGraphWrapper,
    clone_tensors_in_struct,
    get_shared_capture_stream,
)
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.tensor_parallel.random import (
    HAVE_TE,
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.utils import is_te_min_version
from megatron.training.models.dist_utils import _ddp_wrap
from tests.unit_tests.test_utilities import Utils

rank = Utils.rank


def test_ddp_grad_accumulators_share_full_cuda_graph_stream():
    """Retained DDP AccumulateGrad nodes must use the full-iteration capture stream."""

    class RetainingDataParallel(torch.nn.Module):
        """Minimal DDP wrapper that retains parameter AccumulateGrad nodes."""

        def __init__(self, *, module, **_):
            super().__init__()
            self.module = module
            self.grad_accumulators = []
            for param in module.parameters():
                expanded_param = param.expand_as(param)
                grad_accumulator = expanded_param.grad_fn.next_functions[0][0]
                grad_accumulator.register_hook(lambda *_: None)
                self.grad_accumulators.append(grad_accumulator)

        def forward(self, inputs):
            """Run the wrapped module."""
            return self.module(inputs)

    assert torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch is not None
    model = torch.nn.Linear(4, 4, device="cuda")
    model.config = Mock(cuda_graph_impl="full_iteration")
    ddp_config = Mock(
        num_buckets=None,
        bucket_size=1024,
        overlap_grad_reduce=True,
        use_distributed_optimizer=False,
    )
    process_groups = Mock()
    with patch(
        "megatron.training.models.dist_utils.DistributedDataParallel", RetainingDataParallel
    ):
        wrapped_model = _ddp_wrap(
            [model],
            data_parallel_random_init=False,
            ddp_config=ddp_config,
            overlap_param_gather_with_optimizer_step=False,
            pg_collection=process_groups,
        )[0]

    capture_stream = get_shared_capture_stream()
    current_stream = torch.cuda.current_stream()
    capture_stream.wait_stream(current_stream)
    static_input = torch.ones(2, 4, device="cuda")

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        with torch.cuda.stream(capture_stream):
            wrapped_model(static_input).sum().backward()
            wrapped_model.zero_grad(set_to_none=False)

            cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cuda_graph, stream=capture_stream):
                wrapped_model(static_input).sum().backward()

    cuda_graph.replay()
    torch.cuda.synchronize()

    stream_mismatch_warnings = [
        warning
        for warning in caught_warnings
        if "AccumulateGrad node's stream does not match" in str(warning.message)
    ]
    assert not stream_mismatch_warnings
    assert all(param.grad is not None for param in wrapped_model.parameters())


@pytest.mark.parametrize("container", [lambda value: {"mask": value}, lambda value: [value]])
@pytest.mark.parametrize("tensor_first", [False, True])
def test_clone_tensors_in_struct_rejects_optional_tensor_changes(container, tensor_first):
    """Replay cannot add or remove a tensor input recorded by the captured graph."""
    value = torch.arange(4, dtype=torch.float32)
    original = value if tensor_first else None
    target = container(original)
    source = container(None if tensor_first else value)

    with pytest.raises(ValueError, match="tensor inputs must keep the same structure"):
        clone_tensors_in_struct(target, source)

    assert (target["mask"] if isinstance(target, dict) else target[0]) is original


def test_clone_tensors_in_struct_preserves_static_tensor_storage():
    """Valid updates retain captured addresses and leave absent optional fields absent."""
    value = torch.arange(4, dtype=torch.float32)
    buffer = torch.zeros(4)
    target = {"nested": [buffer, None]}

    clone_tensors_in_struct(target, {"nested": [value, None]})

    assert target["nested"][0] is buffer
    assert torch.equal(buffer, value)
    assert target["nested"][1] is None


@pytest.mark.parametrize("capturing", [False, True])
@pytest.mark.parametrize("batch_p2p_sync", [False, True])
def test_batched_p2p_sync_respects_cuda_graph_capture(mocker, capturing, batch_p2p_sync):
    """Capture skips device synchronization but still waits for communication work."""
    communicator = P2PCommunicator.__new__(P2PCommunicator)
    communicator.config = ModelParallelConfig(batch_p2p_comm=True, batch_p2p_sync=batch_p2p_sync)
    communicator.pp_group = Mock()
    communicator.next_rank = 1
    communicator.prev_rank = 1
    request = Mock()
    mocker.patch(
        "megatron.core.pipeline_parallel.p2p_communication._batched_p2p_ops", return_value=[request]
    )
    mocker.patch("torch.cuda.is_current_stream_capturing", return_value=capturing)
    synchronize = mocker.patch("torch.cuda.synchronize")

    communicator._communicate(
        tensor_send_next=torch.ones(4),
        tensor_send_prev=None,
        recv_prev=False,
        recv_next=False,
        tensor_shape=None,
    )

    request.wait.assert_called_once_with()
    assert synchronize.call_count == int(batch_p2p_sync and not capturing)


@pytest.mark.skipif(
    not (HAVE_TE and is_te_min_version("1.5.0")),
    reason="use_te_rng_tracker requires TransformerEngine version >= 1.5",
)
def test_forward_backward_func_with_full_cuda_graph(mocker):
    from megatron.core.pipeline_parallel import get_forward_backward_func

    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)

    def forward_step_func(data_iterator, model):
        import os

        rank = int(os.environ['LOCAL_RANK'])
        dummy_data = torch.ones(1, 4)

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return model(dummy_data), loss_func

    model = torch.nn.Linear(4, 1)

    model.model_type = 'unit-test'

    def set_input_tensor(input_tensor):
        return None

    model.set_input_tensor = set_input_tensor

    forward_backward_func = get_forward_backward_func()
    assert schedule.get_forward_backward_func() == schedule.forward_backward_no_pipelining

    # Wrapping the forward_backward_func with FullCudaGraphWrapper enables full iteration CUDA graphs.
    forward_backward_func = FullCudaGraphWrapper(forward_backward_func)
    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)
    config = ModelParallelConfig(pipeline_model_parallel_size=1)
    model.config = config

    num_microbatches = 4

    # CUDA graph warmup
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[iter([{'input': torch.ones(1, 4)}] * num_microbatches)],
        model=[model],
        num_microbatches=num_microbatches,
        seq_length=None,
        micro_batch_size=None,
        forward_only=True,
    )
    # CUDA graph capture and replay
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[iter([{'input': torch.ones(1, 4)}] * num_microbatches)],
        model=[model],
        num_microbatches=num_microbatches,
        seq_length=None,
        micro_batch_size=None,
        forward_only=True,
    )
    loss_reduced_expected = [
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
    ]

    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(losses_reduced)
        assert i['loss_reduced'] == j['loss_reduced']
    Utils.destroy_model_parallel()
