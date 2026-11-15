# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Capture, reset and recapture of the full-iteration and optimizer CUDA graphs."""

from unittest.mock import patch

import pytest
import torch

import megatron.core.full_cuda_graph as full_cuda_graph
from megatron.core.full_cuda_graph import (
    FullCudaGraphWrapper,
    StaticBufferLoader,
    get_shared_capture_stream,
    release_shared_graph_pool_if_unused,
)
from megatron.core.optimizer.optimizer_cuda_graph import OptimizerCudaGraphWrapper
from megatron.core.rerun_state_machine import RerunMode, RerunStateMachine
from tests.unit_tests.test_utilities import Utils


def _reset_class_state():
    """Both wrappers keep their graph on the class, and the pool handle is a module global."""
    FullCudaGraphWrapper.curr_iteration = {'training': 0, 'validation': 0}
    FullCudaGraphWrapper.cuda_graph = {'training': None, 'validation': None}
    FullCudaGraphWrapper.result = {'training': None, 'validation': None}
    StaticBufferLoader.static_buffers = {'training': [], 'validation': []}
    OptimizerCudaGraphWrapper.cuda_graph = None
    OptimizerCudaGraphWrapper.result = None
    OptimizerCudaGraphWrapper.curr_iteration = 0
    full_cuda_graph._shared_graph_pool = None


@pytest.fixture
def graph_env():
    """A single-rank setup with the rerun state machine and RNG tracking stubbed out."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    _reset_class_state()
    machine = RerunStateMachine(mode=RerunMode.DISABLED)
    with (
        patch("megatron.core.full_cuda_graph.get_all_rng_states", return_value={}),
        patch("megatron.core.full_cuda_graph.get_rerun_state_machine", return_value=machine),
    ):
        yield
    _reset_class_state()
    torch.cuda.synchronize()
    Utils.destroy_model_parallel()


def _step_kwargs(*, forward_only: bool) -> dict:
    return dict(
        model=[torch.nn.Identity()],
        data_iterator=None,
        num_microbatches=1,
        seq_length=1,
        forward_only=forward_only,
    )


def _doubling_step(source: torch.Tensor):
    def forward_backward_func(**_):
        return [{"loss": source * 2}]

    return forward_backward_func


class TestTrainingGraphReset:
    """Resetting the training graph under a shared pool."""

    def _capture_training_and_optimizer(self, source, weight, *, use_single_mempool):
        def optimizer_step():
            weight.sub_(1)

        train = FullCudaGraphWrapper(
            _doubling_step(source), cuda_graph_warmup_steps=1, use_single_mempool=use_single_mempool
        )
        optimizer = OptimizerCudaGraphWrapper(
            optimizer_step, cuda_graph_warmup_steps=1, use_single_mempool=use_single_mempool
        )
        # One eager warmup step each, then the capturing step.
        for _ in range(2):
            train(**_step_kwargs(forward_only=False))
            optimizer()
        assert FullCudaGraphWrapper.cuda_graph['training'] is not None
        assert OptimizerCudaGraphWrapper.cuda_graph is not None
        return train, optimizer

    def test_reset_discards_optimizer_graph_and_pool_then_recaptures(self, graph_env):
        """Graphs sharing a pool are only safe in capture order, so both must be recaptured."""
        source = torch.ones(1024, device="cuda")
        weight = torch.zeros(1024, device="cuda")
        train, optimizer = self._capture_training_and_optimizer(
            source, weight, use_single_mempool=True
        )
        first_pool = full_cuda_graph._shared_graph_pool
        assert first_pool is not None

        train.reset_cuda_graph(stage='training')

        assert FullCudaGraphWrapper.cuda_graph['training'] is None
        assert OptimizerCudaGraphWrapper.cuda_graph is None
        assert OptimizerCudaGraphWrapper.curr_iteration == 0
        # The retired handle must not be captured into again.
        assert full_cuda_graph._shared_graph_pool is None

        source.fill_(3)
        weight.zero_()
        for _ in range(3):  # eager warmup, capture, replay
            result = train(**_step_kwargs(forward_only=False))
            optimizer()
        torch.cuda.synchronize()

        assert FullCudaGraphWrapper.cuda_graph['training'] is not None
        assert OptimizerCudaGraphWrapper.cuda_graph is not None
        assert full_cuda_graph._shared_graph_pool is not None
        assert full_cuda_graph._shared_graph_pool != first_pool
        torch.testing.assert_close(result[0]["loss"], torch.full_like(source, 6))
        # Every call ran the step once: eagerly, or by replaying the graph it had just captured.
        torch.testing.assert_close(weight, torch.full_like(weight, -3))

    def test_reset_keeps_optimizer_graph_with_separate_pools(self, graph_env):
        """Without a shared pool the optimizer graph does not depend on capture order."""
        source = torch.ones(1024, device="cuda")
        weight = torch.zeros(1024, device="cuda")
        train, _ = self._capture_training_and_optimizer(source, weight, use_single_mempool=False)
        optimizer_graph = OptimizerCudaGraphWrapper.cuda_graph

        train.reset_cuda_graph(stage='training')

        assert FullCudaGraphWrapper.cuda_graph['training'] is None
        assert OptimizerCudaGraphWrapper.cuda_graph is optimizer_graph

    def test_reset_empties_the_cache_only_after_discarding_a_graph(self, graph_env):
        """A discarded graph's pool memory stays cached where eager steps cannot use it."""
        source = torch.ones(1024, device="cuda")
        train = FullCudaGraphWrapper(_doubling_step(source), cuda_graph_warmup_steps=1)

        with patch("megatron.core.full_cuda_graph.torch.cuda.empty_cache") as empty_cache:
            train.reset_cuda_graph()
            empty_cache.assert_not_called()

            for _ in range(2):
                train(**_step_kwargs(forward_only=False))
            assert FullCudaGraphWrapper.cuda_graph['training'] is not None
            # Entering a capture empties the cache as well; count the reset's call only.
            empty_cache.reset_mock()
            train.reset_cuda_graph()
            empty_cache.assert_called_once()


class TestSharedGraphPoolRelease:
    """release_shared_graph_pool_if_unused forgets the handle only when no graph uses it."""

    @pytest.fixture(autouse=True)
    def isolated_state(self, monkeypatch):
        monkeypatch.setattr(
            FullCudaGraphWrapper, "cuda_graph", {'training': None, 'validation': None}
        )
        monkeypatch.setattr(OptimizerCudaGraphWrapper, "cuda_graph", None)
        monkeypatch.setattr(full_cuda_graph, "_shared_graph_pool", object())

    def test_released_when_no_graph_is_left(self):
        release_shared_graph_pool_if_unused()
        assert full_cuda_graph._shared_graph_pool is None

    @pytest.mark.parametrize("stage", ['training', 'validation'])
    def test_kept_while_a_full_iteration_graph_exists(self, stage):
        handle = full_cuda_graph._shared_graph_pool
        FullCudaGraphWrapper.cuda_graph[stage] = object()
        release_shared_graph_pool_if_unused()
        assert full_cuda_graph._shared_graph_pool is handle

    def test_kept_while_the_optimizer_graph_exists(self, monkeypatch):
        handle = full_cuda_graph._shared_graph_pool
        monkeypatch.setattr(OptimizerCudaGraphWrapper, "cuda_graph", object())
        release_shared_graph_pool_if_unused()
        assert full_cuda_graph._shared_graph_pool is handle

    def test_no_handle_is_a_no_op(self, monkeypatch):
        monkeypatch.setattr(full_cuda_graph, "_shared_graph_pool", None)
        release_shared_graph_pool_if_unused()
        assert full_cuda_graph._shared_graph_pool is None


class TestValidationGraphWarmup:
    """When the validation graph is captured relative to the training graph."""

    def test_validation_warms_up_when_no_training_graph_exists(self, graph_env):
        source = torch.ones(1024, device="cuda")
        wrapper = FullCudaGraphWrapper(_doubling_step(source), cuda_graph_warmup_steps=2)

        for _ in range(2):
            wrapper(**_step_kwargs(forward_only=True))
            assert FullCudaGraphWrapper.cuda_graph['validation'] is None
        wrapper(**_step_kwargs(forward_only=True))
        assert FullCudaGraphWrapper.cuda_graph['validation'] is not None

    def test_validation_is_captured_on_its_first_step_after_training(self, graph_env):
        """Training warmup has built the lazy state; eager validation steps would only reserve
        working memory next to the training graph's pool."""
        source = torch.ones(1024, device="cuda")
        wrapper = FullCudaGraphWrapper(_doubling_step(source), cuda_graph_warmup_steps=2)

        for _ in range(3):
            wrapper(**_step_kwargs(forward_only=False))
        assert FullCudaGraphWrapper.cuda_graph['training'] is not None

        result = wrapper(**_step_kwargs(forward_only=True))
        assert FullCudaGraphWrapper.cuda_graph['validation'] is not None

        source.fill_(4)
        result = wrapper(**_step_kwargs(forward_only=True))
        torch.cuda.synchronize()
        torch.testing.assert_close(result[0]["loss"], torch.full_like(source, 8))


class TestOptimizerGraphWarmup:
    def test_warmup_step_runs_on_the_shared_capture_stream(self, graph_env):
        """The allocator caches freed memory per stream, so warmup must share the stream that
        the forward/backward warmup and the captures use."""
        seen_streams = []
        weight = torch.zeros(1024, device="cuda")

        def optimizer_step():
            seen_streams.append(torch.cuda.current_stream())
            weight.add_(1)
            return "stepped"

        optimizer = OptimizerCudaGraphWrapper(optimizer_step, cuda_graph_warmup_steps=2)
        caller_stream = torch.cuda.current_stream()

        result = optimizer()

        assert result == "stepped"
        assert OptimizerCudaGraphWrapper.cuda_graph is None
        assert seen_streams == [get_shared_capture_stream()]
        assert seen_streams[0] != caller_stream
        assert torch.cuda.current_stream() == caller_stream
        # The caller's stream waits for the warmup step before reading its results.
        torch.testing.assert_close(weight, torch.ones_like(weight))

    def test_reset_cuda_graph_restarts_warmup(self, graph_env):
        weight = torch.zeros(1024, device="cuda")
        optimizer = OptimizerCudaGraphWrapper(lambda: weight.add_(1), cuda_graph_warmup_steps=1)
        for _ in range(2):
            optimizer()
        assert OptimizerCudaGraphWrapper.cuda_graph is not None

        OptimizerCudaGraphWrapper.reset_cuda_graph()

        assert OptimizerCudaGraphWrapper.cuda_graph is None
        assert OptimizerCudaGraphWrapper.result is None
        assert OptimizerCudaGraphWrapper.curr_iteration == 0
        optimizer()
        assert OptimizerCudaGraphWrapper.cuda_graph is None, "warmup must run again before capture"
