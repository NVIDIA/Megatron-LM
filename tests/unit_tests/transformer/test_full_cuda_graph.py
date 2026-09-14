# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import warnings
from unittest.mock import Mock, patch

import pytest
import torch
from pytest_mock import mocker

import megatron.core.full_cuda_graph as full_cuda_graph_module
import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core import ModelParallelConfig, parallel_state
from megatron.core.full_cuda_graph import FullCudaGraphWrapper, get_shared_capture_stream
from megatron.core.tensor_parallel.random import (
    HAVE_TE,
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.experimental_attention_variant.dsa_logging import (
    DSAIndexerLossLoggingHelper,
)
from megatron.core.utils import is_te_min_version
from megatron.training.models.dist_utils import _ddp_wrap
from tests.unit_tests.test_utilities import Utils

rank = Utils.rank


def _reset_full_cuda_graph_state():
    """Drop process-global full-iteration graph state between focused tests."""
    FullCudaGraphWrapper.curr_iteration = {'training': 0, 'validation': 0}
    FullCudaGraphWrapper.cuda_graph = {'training': None, 'validation': None}
    FullCudaGraphWrapper.result = {'training': None, 'validation': None}


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph test requires a GPU")
def test_full_cuda_graph_capture_counts_dsa_metric_once(monkeypatch):
    """Capture records the DSA write; the immediate replay contributes it exactly once."""
    _reset_full_cuda_graph_state()
    initialize_rng_tracker(force_reset=True)
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    values = torch.full((1,), 7.0, device="cuda")
    tracker = {"values": values, "agreed_size": 1, "reduce_group": None, "avg_group": None}
    tracker["agreed_size_pp_group"] = parallel_state.get_pipeline_model_parallel_group()
    monkeypatch.setattr(DSAIndexerLossLoggingHelper, "tracker", tracker)
    reduce_group = object()
    avg_group = object()

    def forward_backward_func(**kwargs):
        del kwargs
        tracker["reduce_group"] = reduce_group
        tracker["avg_group"] = avg_group
        tracker["values"].add_(1.0)
        return [tracker["values"]]

    model = torch.nn.Module()
    model.logs_dsa_indexer_loss = True
    model.layer_number = 1
    wrapped = FullCudaGraphWrapper(forward_backward_func, cuda_graph_warmup_steps=0)
    wrapped.data_read = lambda *_args: []

    try:
        result = wrapped(
            data_iterator=[], model=[model], num_microbatches=1, seq_length=1, forward_only=True
        )
        torch.cuda.synchronize()

        assert tracker["values"] is values
        torch.testing.assert_close(values, torch.full_like(values, 8.0))
        torch.testing.assert_close(result[0], torch.full_like(result[0], 8.0))
        assert tracker["reduce_group"] is reduce_group
        assert tracker["avg_group"] is avg_group
    finally:
        wrapped.reset_cuda_graph()
        Utils.destroy_model_parallel()


def test_full_cuda_graph_initializes_dsa_tracker_before_eager_warmup(monkeypatch):
    """Custom full-iteration callers fix tracker storage before eager warmup writes."""
    _reset_full_cuda_graph_state()
    calls = []
    events = []
    storage = torch.zeros(1)

    def initialize_tracker(model, pg_collection):
        calls.append((model, pg_collection))
        events.append("initialize")

    def forward_backward_func(**_kwargs):
        events.append("forward")
        storage.add_(1)
        return [storage]

    monkeypatch.setattr(
        full_cuda_graph_module.dsa_logging, "initialize_dsa_metric_tracker", initialize_tracker
    )
    monkeypatch.setattr(
        full_cuda_graph_module.torch.autograd.graph,
        "set_override_stale_capture_stream",
        lambda _enabled: None,
        raising=False,
    )

    model = torch.nn.Module()
    pg_collection = object()
    wrapped = FullCudaGraphWrapper(forward_backward_func, cuda_graph_warmup_steps=1)
    wrapped.data_read = lambda *_args: []

    result = wrapped(
        data_iterator=[],
        model=[model],
        num_microbatches=1,
        seq_length=1,
        forward_only=True,
        pg_collection=pg_collection,
    )

    assert events == ["initialize", "forward"]
    assert calls == [([model], pg_collection)]
    assert result[0] is storage

    def stop_before_capture():
        raise RuntimeError("stop before capture")

    monkeypatch.setattr(full_cuda_graph_module.torch.distributed, "barrier", stop_before_capture)
    with pytest.raises(RuntimeError, match="stop before capture"):
        wrapped(
            data_iterator=[],
            model=[model],
            num_microbatches=1,
            seq_length=1,
            forward_only=True,
            pg_collection=pg_collection,
        )

    assert calls == [([model], pg_collection)]
    assert storage.item() == 1

    wrapped.reset_cuda_graph()
    assert not wrapped._dsa_tracker_initialized_stages
