# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from functools import lru_cache
from types import SimpleNamespace

import pytest
import torch

import megatron.core.parallel_state as ps
from tests.unit_tests.test_utilities import Utils

pytestmark = [pytest.mark.internal, pytest.mark.launch_on_gb200]


@pytest.fixture
def distributed_without_model_parallel():
    Utils.initialize_distributed()
    ps.destroy_model_parallel()
    yield
    ps.destroy_model_parallel()


def _assert_world_usable():
    value = torch.ones(2, device="cuda")
    torch.distributed.all_reduce(value)
    assert torch.equal(value, torch.full_like(value, torch.distributed.get_world_size()))


@pytest.mark.skipif(Utils.world_size < 2, reason="Requires a rank outside the subgroup")
def test_destroy_subgroup_preserves_world(distributed_without_model_parallel, monkeypatch):
    # Another test/application may initialize c10d without populating Utils.store.
    monkeypatch.setattr(Utils, "store", None)
    baseline = set(torch.distributed.distributed_c10d._world.pg_map)
    group = ps.create_group(ranks=[0], backend="nccl")
    if torch.distributed.get_rank() == 0:
        value = torch.ones(1, device="cuda")
        torch.distributed.all_reduce(value, group=group)
        torch.cuda.synchronize()

    ps.destroy_model_parallel()

    # Nonmembers must not accidentally satisfy a WORLD barrier inside teardown.
    store = torch.distributed.PrefixStore(
        "test_destroy_subgroup_preserves_world",
        torch.distributed.distributed_c10d._get_default_store(),
    )
    if torch.distributed.get_rank() == 0:
        store.set("destroyed", "1")
    else:
        store.wait(["destroyed"], timedelta(seconds=90))

    assert set(torch.distributed.distributed_c10d._world.pg_map) == baseline
    _assert_world_usable()


@pytest.mark.parametrize("parallel_axis", ["tensor", "pipeline", "context"])
def test_repeated_lifetimes_release_process_groups(
    distributed_without_model_parallel, parallel_axis
):
    baseline = set(torch.distributed.distributed_c10d._world.pg_map)
    size_key = {
        "tensor": "tensor_model_parallel_size",
        "pipeline": "pipeline_model_parallel_size",
        "context": "context_parallel_size",
    }[parallel_axis]

    for _ in range(2):
        ps.initialize_model_parallel(**{size_key: torch.distributed.get_world_size()})
        created = set(torch.distributed.distributed_c10d._world.pg_map) - baseline
        assert created

        # Initialize actual communicators, including Gloo groups, before checking release.
        for group in ps._global_process_group_list:
            if group is None or torch.distributed.get_world_size(group) == 1:
                continue
            device = "cpu" if torch.distributed.get_backend(group) == "gloo" else "cuda"
            value = torch.ones(1, device=device)
            torch.distributed.all_reduce(value, group=group)
            assert value.item() == torch.distributed.get_world_size(group)

        ps.destroy_model_parallel()
        assert not ps.model_parallel_is_initialized()
        assert set(torch.distributed.distributed_c10d._world.pg_map) == baseline
        _assert_world_usable()


@pytest.mark.parametrize("direct_cache_clear", [False, True], ids=["torch_2_6_to_2_9", "current"])
def test_destroy_invalidates_dtensor_sharding_cache(
    distributed_without_model_parallel, monkeypatch, direct_cache_clear
):
    current_group = [object()]

    @lru_cache(maxsize=8)
    def cached_sharding(mesh_layout):
        return current_group[0]

    propagate = SimpleNamespace(cache=cached_sharding)
    if direct_cache_clear:
        propagate.cache_clear = cached_sharding.cache_clear
    dtensor = SimpleNamespace(
        _op_dispatcher=SimpleNamespace(
            sharding_propagator=SimpleNamespace(propagate_op_sharding=propagate)
        )
    )
    monkeypatch.setitem(sys.modules, "torch.distributed.tensor", SimpleNamespace(DTensor=dtensor))
    native_cache = {"same_mesh": current_group[0]}
    monkeypatch.setattr(
        torch._C, "_clear_DTensor_sharding_propagator_cache", native_cache.clear, raising=False
    )

    previous_group = cached_sharding("same_mesh")
    ps.destroy_model_parallel()
    current_group[0] = object()

    assert cached_sharding("same_mesh") is current_group[0]
    assert cached_sharding("same_mesh") is not previous_group
    assert not native_cache


@pytest.mark.skipif(Utils.world_size < 4, reason="Requires partial DP and hierarchical CP")
def test_destroy_clears_partial_and_context_groups(distributed_without_model_parallel):
    baseline = set(torch.distributed.distributed_c10d._world.pg_map)
    ps.initialize_model_parallel(
        context_parallel_size=2,
        hierarchical_context_parallel_sizes=[2],
        hybrid_context_parallel=True,
        num_distributed_optimizer_instances=2,
    )
    partial_dp = ps.get_data_parallel_group(
        with_context_parallel=True, with_gtp_remat=False, partial_data_parallel=True
    )
    partial_gloo = ps.get_data_parallel_group_gloo(
        with_context_parallel=True, partial_data_parallel=True
    )
    assert ps.get_hierarchical_context_parallel_groups()
    assert ps._HYBRID_DP_CP_GROUPS
    for group, device in ((partial_dp, "cuda"), (partial_gloo, "cpu")):
        value = torch.ones(1, device=device)
        torch.distributed.all_reduce(value, group=group)
        assert value.item() == torch.distributed.get_world_size(group)

    ps.destroy_model_parallel()
    with pytest.raises(AssertionError):
        ps.get_data_parallel_group(
            with_context_parallel=True, with_gtp_remat=False, partial_data_parallel=True
        )
    with pytest.raises(AssertionError):
        ps.get_data_parallel_group_gloo(with_context_parallel=True, partial_data_parallel=True)
    with pytest.raises(AssertionError):
        ps.get_hierarchical_context_parallel_groups()
    assert not ps._HYBRID_DP_CP_GROUPS
    assert set(torch.distributed.distributed_c10d._world.pg_map) == baseline

    # A lifetime without hybrid CP must not inherit the earlier size-to-group map.
    ps.initialize_model_parallel()
    assert not ps._HYBRID_DP_CP_GROUPS
    ps.destroy_model_parallel()
    _assert_world_usable()


def test_initialize_clears_dtensor_cache_after_threaded_teardown(
    distributed_without_model_parallel,
):
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor, Shard

    # Exercise the real LocalLRUCache with identical mesh layouts across lifetimes.
    for lifetime in range(2):
        ps.initialize_model_parallel(tensor_model_parallel_size=Utils.world_size)
        group = ps.get_tensor_model_parallel_group()
        mesh = DeviceMesh.from_group(group, "cuda")
        local = torch.full((2, 2), float(lifetime + 1), device="cuda")
        value = DTensor.from_local(local, mesh, [Shard(0)])
        result = value + 1
        assert result.device_mesh.get_group() is group
        assert torch.equal(
            result.full_tensor(),
            torch.full((2 * Utils.world_size, 2), lifetime + 2.0, device="cuda"),
        )
        del value, result, mesh
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(ps.destroy_model_parallel).result(timeout=90)


def test_initialize_clears_thread_local_dtensor_caches(
    distributed_without_model_parallel, monkeypatch
):
    class Cache(threading.local):
        def __init__(self):
            self.cache = {"old_mesh": object()}

        def cache_clear(self):
            self.cache.clear()

    python_cache, native_cache = Cache(), Cache()
    dtensor = SimpleNamespace(
        _op_dispatcher=SimpleNamespace(
            sharding_propagator=SimpleNamespace(propagate_op_sharding=python_cache)
        )
    )
    monkeypatch.setitem(sys.modules, "torch.distributed.tensor", SimpleNamespace(DTensor=dtensor))
    monkeypatch.setattr(
        torch._C,
        "_clear_DTensor_sharding_propagator_cache",
        native_cache.cache_clear,
        raising=False,
    )
    thread = threading.Thread(target=ps._clear_dtensor_sharding_cache)
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert python_cache.cache and native_cache.cache

    ps.initialize_model_parallel()
    assert not python_cache.cache and not native_cache.cache


def test_create_group_default_ranks(distributed_without_model_parallel):
    baseline = set(torch.distributed.distributed_c10d._world.pg_map)
    group = ps.create_group(backend="gloo")
    value = torch.ones(1)
    torch.distributed.all_reduce(value, group=group)
    assert value.item() == Utils.world_size
    ps.destroy_model_parallel()
    assert set(torch.distributed.distributed_c10d._world.pg_map) == baseline


@pytest.mark.parametrize("owner", ["runner", "full_iteration", "optimizer"])
def test_destroy_releases_captured_collective(distributed_without_model_parallel, owner):
    from megatron.core.full_cuda_graph import FullCudaGraphWrapper
    from megatron.core.optimizer.optimizer_cuda_graph import OptimizerCudaGraphWrapper
    from megatron.core.transformer.cuda_graphs import _CudaGraphRunner
    from megatron.core.transformer.module import MegatronModule

    baseline = set(torch.distributed.distributed_c10d._world.pg_map)
    group = ps.create_group(ranks=list(range(Utils.world_size)), backend="nccl")
    value = torch.ones(2, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.distributed.all_reduce(value, group=group)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        torch.distributed.all_reduce(value, group=group)
    graph.replay()
    torch.cuda.synchronize()

    if owner == "runner":
        runner = _CudaGraphRunner(
            MegatronModule(config=None), None, [], {}, func=None, need_backward=False
        )
        # Simulate a capture that never reached the record/flag updates.
        runner.fwd_graph = graph
    elif owner == "full_iteration":
        FullCudaGraphWrapper.cuda_graph['training'] = graph
        FullCudaGraphWrapper.curr_iteration['training'] = 3
    else:
        OptimizerCudaGraphWrapper.cuda_graph = graph
        OptimizerCudaGraphWrapper.curr_iteration = 3
    del graph

    # A live captured communicator prevents graceful NCCL finalization from completing.
    ps.destroy_model_parallel()
    assert set(torch.distributed.distributed_c10d._world.pg_map) == baseline
    if owner == "runner":
        assert runner.fwd_graph is None
    elif owner == "full_iteration":
        assert FullCudaGraphWrapper.cuda_graph['training'] is None
        assert FullCudaGraphWrapper.curr_iteration['training'] == 0
    else:
        assert OptimizerCudaGraphWrapper.cuda_graph is None
        assert OptimizerCudaGraphWrapper.curr_iteration == 0
    _assert_world_usable()


@pytest.mark.skipif(Utils.world_size < 2, reason="Requires an unmatched collective")
def test_abort_teardown_with_unmatched_collective(distributed_without_model_parallel, monkeypatch):
    # Isolate the injected failure to the new subgroup. Its watchdog must leave
    # recovery to the explicit abort rather than terminating the pytest worker.
    monkeypatch.setenv("TORCH_NCCL_ASYNC_ERROR_HANDLING", "0")
    baseline = set(torch.distributed.distributed_c10d._world.pg_map)
    group = ps.create_group(ranks=list(range(Utils.world_size)), backend="nccl")
    value = torch.ones(2, device="cuda")
    torch.distributed.all_reduce(value, group=group)
    torch.cuda.synchronize()
    store = torch.distributed.PrefixStore(
        "test_abort_teardown_with_unmatched_collective",
        torch.distributed.distributed_c10d._get_default_store(),
    )
    work = None
    if torch.distributed.get_rank() == 0:
        work = torch.distributed.all_reduce(value, group=group, async_op=True)
        store.set("enqueued", "1")
    else:
        store.wait(["enqueued"], timedelta(seconds=30))
    ps.destroy_model_parallel(abort=True)
    del work
    torch.cuda.synchronize()
    assert set(torch.distributed.distributed_c10d._world.pg_map) == baseline
    _assert_world_usable()
