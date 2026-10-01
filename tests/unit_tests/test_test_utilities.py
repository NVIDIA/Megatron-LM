# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Regression tests for shared Megatron unit-test utilities."""

import os
from argparse import Namespace

import pytest
import torch.distributed

import megatron.core.full_cuda_graph as full_cuda_graph
import megatron.core.num_microbatches_calculator as num_microbatches
import megatron.core.parallel_state as ps
import megatron.core.transformer.cuda_graphs as cuda_graphs
import megatron.training.global_vars as global_vars
from megatron.core.full_cuda_graph import FullCudaGraphWrapper, StaticBufferLoader
from megatron.core.optimizer.optimizer_cuda_graph import OptimizerCudaGraphWrapper
from megatron.core.transformer.cuda_graphs import CudaGraphManager
from tests.unit_tests.test_utilities import _NVTE_ATTN_ENV_VARS, Utils, reset_megatron_test_state


def test_reset_megatron_test_state_clears_process_wide_test_state():
    """A reset leaves the next test without prior training or CUDA-graph state."""
    global_vars.set_args(Namespace(test_marker=True))
    num_microbatches.init_num_microbatches_calculator(
        rank=0, global_batch_size=1, micro_batch_size=1, data_parallel_size=1
    )
    ps.set_tensor_model_parallel_world_size(2)
    ps.set_tensor_model_parallel_rank(1)
    Utils.inited = True
    for name in _NVTE_ATTN_ENV_VARS:
        os.environ[name] = "1"
    CudaGraphManager.global_mempool = object()
    CudaGraphManager.fwd_mempools = object()
    CudaGraphManager.bwd_mempools = object()
    StaticBufferLoader.static_buffers = {"training": [object()], "validation": [object()]}
    FullCudaGraphWrapper.curr_iteration = {"training": 3, "validation": 4}
    FullCudaGraphWrapper.cuda_graph = {"training": object(), "validation": object()}
    FullCudaGraphWrapper.result = {"training": object(), "validation": object()}
    OptimizerCudaGraphWrapper.curr_iteration = 5
    OptimizerCudaGraphWrapper.cuda_graph = object()
    OptimizerCudaGraphWrapper.result = object()
    full_cuda_graph._shared_graph_pool = object()
    full_cuda_graph._shared_capture_stream = object()

    reset_megatron_test_state()
    reset_megatron_test_state()

    assert global_vars._GLOBAL_ARGS is None
    assert num_microbatches._GLOBAL_NUM_MICROBATCHES_CALCULATOR is None
    assert ps._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is None
    assert ps._MPU_TENSOR_MODEL_PARALLEL_RANK is None
    assert Utils.inited is False
    assert all(name not in os.environ for name in _NVTE_ATTN_ENV_VARS)
    assert CudaGraphManager.global_mempool is None
    assert CudaGraphManager.fwd_mempools is None
    assert CudaGraphManager.bwd_mempools is None
    assert StaticBufferLoader.static_buffers == {"training": [], "validation": []}
    assert FullCudaGraphWrapper.curr_iteration == {"training": 0, "validation": 0}
    assert FullCudaGraphWrapper.cuda_graph == {"training": None, "validation": None}
    assert FullCudaGraphWrapper.result == {"training": None, "validation": None}
    assert OptimizerCudaGraphWrapper.curr_iteration == 0
    assert OptimizerCudaGraphWrapper.cuda_graph is None
    assert OptimizerCudaGraphWrapper.result is None
    assert full_cuda_graph._shared_graph_pool is None
    assert full_cuda_graph._shared_capture_stream is None


def test_reset_megatron_test_state_releases_cuda_graphs_before_model_parallel(monkeypatch):
    """CUDA graph cleanup precedes model-parallel teardown on every host."""
    calls = []
    monkeypatch.setattr(cuda_graphs, "delete_cuda_graphs", lambda: calls.append("cuda_graphs"))
    monkeypatch.setattr(Utils, "destroy_model_parallel", lambda: calls.append("utils"))
    monkeypatch.setattr(ps, "destroy_model_parallel", lambda: calls.append("model_parallel"))

    reset_megatron_test_state()

    assert calls == ["cuda_graphs", "utils", "model_parallel"]


def test_reset_megatron_test_state_tears_down_model_parallel_after_graph_cleanup_error(monkeypatch):
    """A graph cleanup failure propagates only after mandatory teardown runs."""
    calls = []

    def fail_graph_cleanup():
        calls.append("cuda_graphs")
        raise RuntimeError("graph cleanup failed")

    Utils.inited = True
    CudaGraphManager.global_mempool = object()
    CudaGraphManager.fwd_mempools = object()
    CudaGraphManager.bwd_mempools = object()
    monkeypatch.setattr(cuda_graphs, "delete_cuda_graphs", fail_graph_cleanup)
    monkeypatch.setattr(Utils, "destroy_model_parallel", lambda: calls.append("utils"))
    monkeypatch.setattr(ps, "destroy_model_parallel", lambda: calls.append("model_parallel"))

    with pytest.raises(RuntimeError, match="graph cleanup failed"):
        reset_megatron_test_state()

    assert calls == ["cuda_graphs", "utils", "model_parallel"]
    assert Utils.inited is False
    assert CudaGraphManager.global_mempool is None
    assert CudaGraphManager.fwd_mempools is None
    assert CudaGraphManager.bwd_mempools is None


def test_reset_megatron_test_state_finishes_teardown_after_utils_error(monkeypatch):
    """Direct parallel-state cleanup runs even if the utility teardown fails."""
    calls = []

    def fail_utils_cleanup():
        calls.append("utils")
        raise RuntimeError("utility cleanup failed")

    Utils.inited = True
    monkeypatch.setattr(Utils, "destroy_model_parallel", fail_utils_cleanup)
    monkeypatch.setattr(ps, "destroy_model_parallel", lambda: calls.append("model_parallel"))

    with pytest.raises(RuntimeError, match="utility cleanup failed"):
        reset_megatron_test_state()

    assert calls == ["utils", "model_parallel"]
    assert Utils.inited is False


def test_reset_megatron_test_state_tears_down_model_parallel_but_keeps_default_group(tmp_path):
    """The shared reset owns Megatron groups, not the test launcher's group."""
    initialized_default_group = torch.distributed.is_initialized()
    reset_megatron_test_state()
    if not initialized_default_group:
        launcher_env = {"RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"}
        if launcher_env <= os.environ.keys():
            default_world_size = int(os.environ["WORLD_SIZE"])
            rank = int(os.environ["RANK"])
            torch.distributed.init_process_group(
                backend="gloo", rank=rank, world_size=default_world_size
            )
        else:
            default_world_size = 1
            init_file = tmp_path / "default-process-group"
            torch.distributed.init_process_group(
                backend="gloo", init_method=f"file://{init_file}", rank=0, world_size=1
            )
    else:
        default_world_size = torch.distributed.get_world_size()

    try:
        ps.initialize_model_parallel()
        assert ps.is_initialized()

        reset_megatron_test_state()

        assert not ps.is_initialized()
        assert torch.distributed.is_initialized()
        assert torch.distributed.get_world_size() == default_world_size
    finally:
        reset_megatron_test_state()
        if not initialized_default_group and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
