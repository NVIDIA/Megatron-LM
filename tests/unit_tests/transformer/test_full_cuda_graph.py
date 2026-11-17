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
    _reset_static_buffers()


def _reset_static_buffers():
    StaticBufferLoader.static_buffers = {'training': [], 'validation': []}
    StaticBufferLoader.stacked_buffers = {'training': {}, 'validation': {}}
    StaticBufferLoader.pinned_buffers = {'training': {}, 'validation': {}}
    StaticBufferLoader.staging_done_event = {'training': None, 'validation': None}
    StaticBufferLoader.consumed_event = {'training': None, 'validation': None}


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


class _CountingIterator:
    """Iterator that counts how many items were read from it."""

    def __init__(self, items):
        self.items = iter(items)
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.count += 1
        return next(self.items)


def _to_device(src, device):
    if isinstance(src, torch.Tensor):
        return src.to(device)
    if isinstance(src, dict):
        return {key: _to_device(value, device) for key, value in src.items()}
    return src


def _make_inputs(step, chunk, microbatch, device, nested):
    """Inputs of one microbatch; every chunk reads different values."""
    generator = torch.Generator().manual_seed(10000 * step + 100 * chunk + microbatch)
    inputs = {
        'tokens': torch.randint(0, 50000, (2, 16), generator=generator),
        'loss_mask': torch.rand(2, 16, generator=generator),
        'attention_bias': torch.randn(1, 4, 4, generator=generator).to(torch.bfloat16),
        # The shape differs between microbatches, so this key is not stacked.
        'packed_seq': torch.randint(0, 100, (microbatch + 1,), generator=generator),
        # Not contiguous, so this key is not stacked.
        'transposed': torch.randn(4, 3, generator=generator).t(),
        'num_tokens': 32 + step + chunk,
    }
    if chunk == 0:
        # Only the first chunk reads this key; the static buffers keep its value.
        inputs['first_chunk_only'] = torch.randn(3, generator=generator)
    if nested:
        inputs['nested'] = {'inner': torch.randn(2, generator=generator)}
    inputs = _to_device(inputs, device)
    # The loader also accepts a tuple whose first item is the input dict.
    return (inputs,) if microbatch % 2 else inputs


def _snapshot(src):
    """Copy the static buffers to host, keeping dtype, shape, strides and raw bytes."""
    if isinstance(src, torch.Tensor):
        assert src.is_cuda
        raw = src.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        return ('tensor', src.dtype, tuple(src.shape), src.stride(), raw)
    if isinstance(src, dict):
        return {key: _snapshot(value) for key, value in src.items()}
    if isinstance(src, list):
        return [_snapshot(value) for value in src]
    return src


def _assert_same_snapshot(actual, expected):
    if isinstance(expected, tuple) and expected and expected[0] == 'tensor':
        assert actual[:4] == expected[:4]
        assert torch.equal(actual[4], expected[4])
    elif isinstance(expected, dict):
        assert list(actual.keys()) == list(expected.keys())
        for key in expected:
            _assert_same_snapshot(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_same_snapshot(actual_item, expected_item)
    else:
        assert actual == expected


def _data_ptrs(buffers):
    return [
        {key: value.data_ptr() for key, value in static.items() if isinstance(value, torch.Tensor)}
        for static in buffers
    ]


def _stage_steps(batched, device, num_chunks, nested, stage, num_steps, num_microbatches):
    """Run data_read for num_steps steps and record what the static buffers hold."""
    _reset_static_buffers()
    wrapper = FullCudaGraphWrapper(lambda **_: None, batched_input_staging=batched)
    # With several chunks, chunk 1 has no data iterator, as on some pipeline stages.
    missing_chunk = 1 if num_chunks > 2 else None

    def chunk_iterator(chunk):
        # A function binds chunk per iterator; a nested generator would read it late.
        return _CountingIterator(
            _make_inputs(step, chunk, microbatch, device, nested)
            for step in range(num_steps + 1)
            for microbatch in range(num_microbatches)
        )

    iterators = [
        None if chunk == missing_chunk else chunk_iterator(chunk) for chunk in range(num_chunks)
    ]
    model = [torch.nn.Identity() for _ in range(num_chunks)]
    snapshots, data_ptrs, counts = [], [], []
    for _ in range(num_steps):
        data_list = wrapper.data_read(iterators, model, stage == 'training', num_microbatches)
        torch.cuda.synchronize()
        buffers = StaticBufferLoader.static_buffers[stage]
        assert len(data_list) == num_chunks
        for iterator, chunk_data in zip(iterators, data_list):
            if iterator is None:
                assert chunk_data is None
            else:
                chunk_data = list(chunk_data)
                assert len(chunk_data) == num_microbatches
                assert all(static is buffers[b] for b, static in enumerate(chunk_data))
        snapshots.append(_snapshot(buffers))
        data_ptrs.append(_data_ptrs(buffers))
        counts.append([None if iterator is None else iterator.count for iterator in iterators])
        if batched:
            # As after a graph replay: the next step's copies of host inputs wait on this event.
            consumed_event = torch.cuda.Event()
            consumed_event.record()
            StaticBufferLoader.consumed_event[stage] = consumed_event
    return snapshots, data_ptrs, counts


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("num_chunks", [1, 4])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("stage", ["training", "validation"])
def test_batched_input_staging_matches_per_microbatch_staging(device, num_chunks, nested, stage):
    """Batched staging leaves the same static buffers and reads the same data as the default."""
    num_steps, num_microbatches = 3, 4
    expected_snapshots, expected_ptrs, expected_counts = _stage_steps(
        False, device, num_chunks, nested, stage, num_steps, num_microbatches
    )
    snapshots, data_ptrs, counts = _stage_steps(
        True, device, num_chunks, nested, stage, num_steps, num_microbatches
    )

    # Every step leaves the same bytes, dtypes, shapes, strides and keys in the static buffers.
    for snapshot, expected in zip(snapshots, expected_snapshots):
        _assert_same_snapshot(snapshot, expected)
    # Every iterator is read once per microbatch and step, as by the default path.
    assert counts == expected_counts
    for step, step_counts in enumerate(counts):
        assert all(count is None or count == (step + 1) * num_microbatches for count in step_counts)
    # The static buffers are allocated once and reused in later steps.
    assert all(step_ptrs == data_ptrs[0] for step_ptrs in data_ptrs)
    assert all(step_ptrs == expected_ptrs[0] for step_ptrs in expected_ptrs)

    buffers = StaticBufferLoader.static_buffers[stage]
    stacked_keys = {'tokens', 'loss_mask', 'attention_bias', 'first_chunk_only'}
    if nested and num_chunks > 1:
        # Nested inputs of several chunks use the per-microbatch path.
        assert StaticBufferLoader.stacked_buffers[stage] == {}
        return
    assert set(StaticBufferLoader.stacked_buffers[stage]) == stacked_keys
    for key in stacked_keys:
        # The static buffers of a stacked key are the rows of one contiguous tensor.
        row_bytes = buffers[0][key].numel() * buffers[0][key].element_size()
        for b in range(num_microbatches):
            assert buffers[b][key].data_ptr() == buffers[0][key].data_ptr() + b * row_bytes
    if device == "cpu":
        assert set(StaticBufferLoader.pinned_buffers[stage]) == stacked_keys
        assert all(t.is_pinned() for t in StaticBufferLoader.pinned_buffers[stage].values())
    else:
        assert StaticBufferLoader.pinned_buffers[stage] == {}


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_full_cuda_graph_replay_with_batched_input_staging(mocker, device):
    """Replays of a captured graph read the inputs that batched staging copied for that step."""
    mocker.patch("megatron.core.full_cuda_graph.get_all_rng_states", return_value={})
    mocker.patch(
        "megatron.core.full_cuda_graph.get_rerun_state_machine",
        return_value=RerunStateMachine(mode=RerunMode.DISABLED),
    )
    mocker.patch("torch.autograd.graph.set_override_stale_capture_stream", create=True)
    Utils.initialize_model_parallel()
    num_steps, num_microbatches, warmup_steps = 5, 4, 1

    def forward_backward(*, data_iterator, num_microbatches, **_):
        batches = [next(data_iterator[0]) for _ in range(num_microbatches)]
        return torch.stack([batch['tokens'] for batch in batches]).sum()

    inputs = [
        [_make_inputs(step, 0, microbatch, device, False) for microbatch in range(num_microbatches)]
        for step in range(num_steps)
    ]
    iterator = _CountingIterator(batch for step_inputs in inputs for batch in step_inputs)
    wrapper = FullCudaGraphWrapper(
        forward_backward, cuda_graph_warmup_steps=warmup_steps, batched_input_staging=True
    )
    wrapper.reset_cuda_graph()
    try:
        for step in range(num_steps):
            result = wrapper(
                model=[torch.nn.Identity()],
                data_iterator=[iterator],
                num_microbatches=num_microbatches,
                seq_length=16,
                forward_only=False,
            )
            expected = sum(
                StaticBufferLoader.unwrap_inputs(batch)['tokens'].sum().item()
                for batch in inputs[step]
            )
            assert result.item() == expected
            assert iterator.count == (step + 1) * num_microbatches
            # From the capture step on, each replay records the event that the staging of the
            # next step waits on.
            consumed_event = StaticBufferLoader.consumed_event['training']
            assert (consumed_event is not None) == (step >= warmup_steps)
        assert wrapper.cuda_graph['training'] is not None
    finally:
        wrapper.reset_cuda_graph()
        Utils.destroy_model_parallel()
