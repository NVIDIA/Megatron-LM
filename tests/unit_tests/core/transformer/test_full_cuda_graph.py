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
    StaticBufferLoader,
    get_shared_capture_stream,
)
from megatron.core.rerun_state_machine import RerunMode, RerunStateMachine
from megatron.core.tensor_parallel.random import (
    HAVE_TE,
    model_parallel_cuda_manual_seed,
    prime_cuda_rng_states_for_graph_capture,
)
from megatron.core.utils import is_te_min_version, is_torch_min_version
from megatron.training.models.dist_utils import _ddp_wrap
from tests.unit_tests.test_utilities import Utils

rank = Utils.rank


@pytest.fixture(autouse=True)
def reset_full_cuda_graph_state():
    """The wrapper keeps its graph and static buffers on the class."""
    # Select each rank's device before a test creates the shared capture stream.
    Utils.initialize_distributed()
    yield
    FullCudaGraphWrapper.curr_iteration = {'training': 0, 'validation': 0}
    FullCudaGraphWrapper.cuda_graph = {'training': None, 'validation': None}
    FullCudaGraphWrapper.result = {'training': None, 'validation': None}
    StaticBufferLoader.static_buffers = {'training': [], 'validation': []}


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


def test_full_cuda_graph_warmup_with_retained_none_grad_leaf(mocker):
    """Warmup-created leaves returning None must retain the stream used for capture."""

    class FusedWeightGrad(torch.autograd.Function):
        """Model fused wgrad writing main_grad and returning no autograd gradient."""

        @staticmethod
        def forward(ctx, weight, inputs, main_grad):
            """Save the inputs needed to compute the weight gradient."""
            ctx.save_for_backward(inputs)
            ctx.main_grad = main_grad
            return weight * inputs

        @staticmethod
        def backward(ctx, grad_output):
            """Write the real gradient without returning a dummy weight gradient."""
            (inputs,) = ctx.saved_tensors
            ctx.main_grad.copy_(grad_output * inputs)
            return None, None, None

    mocker.patch("megatron.core.full_cuda_graph.get_all_rng_states", return_value={})
    mocker.patch(
        "megatron.core.full_cuda_graph.get_rerun_state_machine",
        return_value=RerunStateMachine(mode=RerunMode.DISABLED),
    )
    # Stream alignment must work without changing PyTorch's process-global override.
    mocker.patch("torch.autograd.graph.set_override_stale_capture_stream", create=True)
    Utils.initialize_model_parallel()
    weight = torch.ones(8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    inputs = torch.empty_like(weight)
    main_grad = torch.empty_like(weight)
    accumulators = []

    def forward_backward(**_):
        assert torch.cuda.current_stream() == get_shared_capture_stream()
        if not accumulators:
            # Retain a lazily created leaf, as MoonEP/TE do during eager warmup.
            accumulators.append(weight.expand_as(weight).grad_fn.next_functions[0][0])
        loss = FusedWeightGrad.apply(weight, inputs, main_grad).sum()
        loss.backward()
        return loss.detach()

    warmup_steps = 2
    wrapper = FullCudaGraphWrapper(forward_backward, cuda_graph_warmup_steps=warmup_steps)
    wrapper.reset_cuda_graph()
    try:
        for value in range(1, warmup_steps + 4):
            inputs.fill_(value)
            main_grad.zero_()
            loss = wrapper(
                model=[torch.nn.Identity()],
                data_iterator=None,
                num_microbatches=1,
                seq_length=8,
                forward_only=False,
            )
            torch.testing.assert_close(loss, inputs.sum(), rtol=0, atol=0)
            torch.testing.assert_close(main_grad, inputs, rtol=0, atol=0)
            assert weight.grad is None
        assert wrapper.cuda_graph["training"] is not None
    finally:
        wrapper.reset_cuda_graph()
        accumulators.clear()
        Utils.destroy_model_parallel()


@pytest.mark.skipif(
    not is_torch_min_version("2.14.0a0"),
    reason="Generator capture state is created lazily only on PyTorch >= 2.14",
)
def test_prime_cuda_rng_states_for_graph_capture():
    """The first RNG use in a capture must not poll an event on a newly captured stream."""
    side_stream = torch.cuda.Stream()
    pending = torch.empty(1024, device="cuda")
    # Recorded before capture, so the allocator still holds this use when the block is freed.
    pending.record_stream(side_stream)

    cuda_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(
        cuda_graph, stream=torch.cuda.Stream(), capture_error_mode="thread_local"
    ):
        prime_cuda_rng_states_for_graph_capture()
        # side_stream is not captured yet, so the allocator defers this free.
        del pending
        # side_stream joins the capture, as an NCCL stream does on a pipeline recv.
        side_stream.wait_stream(torch.cuda.current_stream())
        # Unprimed, this RNG use creates its capture state on the default stream; that allocation
        # records and queries an event on side_stream and fails with cudaErrorCapturedEvent.
        sample = torch.rand(8, device="cuda")
        torch.cuda.current_stream().wait_stream(side_stream)

    cuda_graph.replay()
    torch.cuda.synchronize()
    assert bool(((sample >= 0) & (sample < 1)).all())


@pytest.mark.skipif(
    not (HAVE_TE and is_te_min_version("1.5.0")),
    reason="use_te_rng_tracker requires TransformerEngine version >= 1.5",
)
def test_forward_backward_func_with_full_cuda_graph(mocker):
    from megatron.core.pipeline_parallel import get_forward_backward_func

    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123, te_rng_tracker=True, force_reset_rng=True)

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
