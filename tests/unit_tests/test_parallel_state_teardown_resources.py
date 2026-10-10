# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core import parallel_state as ps
from megatron.core.full_cuda_graph import FullCudaGraphWrapper
from megatron.core.optimizer.optimizer_cuda_graph import OptimizerCudaGraphWrapper
from megatron.core.tensor_parallel import generalized_tensor_parallelism as gtp
from megatron.core.tensor_parallel import gtp_cuda_graphs, gtp_symmetric_memory
from megatron.core.transformer import cuda_graphs
from megatron.core.transformer.moe import fused_a2a, token_dispatcher
from tests.unit_tests.test_utilities import Utils

pytestmark = [pytest.mark.internal, pytest.mark.launch_on_gb200]


def test_destroy_attempts_remaining_groups_after_failure(monkeypatch):
    groups = [object(), object(), object()]
    pg_map = dict.fromkeys(groups)
    destroyed = []

    def destroy(group):
        destroyed.append(group)
        if group is groups[1]:
            raise RuntimeError("injected shutdown failure")
        del pg_map[group]

    monkeypatch.setattr(
        torch.distributed.distributed_c10d, "_world", SimpleNamespace(pg_map=pg_map)
    )
    monkeypatch.setattr(torch.distributed, "destroy_process_group", destroy)
    monkeypatch.setattr(ps, "_global_process_group_list", [None, *groups])
    with pytest.raises(RuntimeError, match="injected shutdown failure"):
        ps._destroy_created_process_groups()
    assert destroyed == groups[::-1]
    assert ps._global_process_group_list is None
    assert set(pg_map) == {groups[1]}

    # A subsequent lifetime must not append to the failed lifetime's registry.
    next_group = object()
    monkeypatch.setattr(torch.distributed, "new_group", lambda **kwargs: next_group)
    ps.create_group()
    assert ps._global_process_group_list == [None, next_group]


@pytest.fixture
def recorded_teardown(monkeypatch):
    """Replace the groups, callbacks and caches of teardown with recorders."""
    events = []
    backend = SimpleNamespace(abort=lambda: events.append("abort"))
    group = Mock()
    group._get_backend.return_value = backend
    monkeypatch.setattr(
        torch.distributed.distributed_c10d, "_world", SimpleNamespace(pg_map={group: None})
    )
    monkeypatch.setattr(torch.distributed, "get_backend", lambda group: "nccl")
    monkeypatch.setattr(
        torch.distributed, "destroy_process_group", lambda group: events.append("destroy")
    )
    monkeypatch.setattr(ps, "_global_process_group_list", [None, group])
    monkeypatch.setattr(ps.SymmetricMemoryManager, "destroy", lambda: events.append("memory"))
    monkeypatch.setattr(ps, "_clear_dtensor_sharding_cache", lambda: events.append("dtensor"))
    monkeypatch.setattr(
        ps, "_MODEL_PARALLEL_TEARDOWN_CALLBACKS", {stage: [] for stage in ps.TeardownStage}
    )
    monkeypatch.setattr(ps, "_MODEL_PARALLEL_TEARDOWN_ABORT_CALLBACKS", {})
    return events


def _recorder(events, name):
    return lambda: events.append(name)


@pytest.mark.parametrize("abort", [False, True])
def test_teardown_runs_callbacks_in_stage_order(recorded_teardown, abort):
    events = recorded_teardown
    stage = ps.TeardownStage
    # Register out of stage order. Within a stage the later registration runs first.
    ps.register_model_parallel_teardown(stage.RESET_STATE, _recorder(events, "te_state"))
    ps.register_model_parallel_teardown(stage.RELEASE_CUDA_GRAPHS, _recorder(events, "graphs"))
    ps.register_model_parallel_teardown(
        stage.RELEASE_COMMUNICATION, _recorder(events, "a2a_buffers")
    )
    release_symm_buffers = _recorder(events, "symm_buffers")
    for _ in range(2):
        ps.register_model_parallel_teardown(stage.RELEASE_COMMUNICATION, release_symm_buffers)
    ps.register_model_parallel_teardown(stage.VALIDATE, _recorder(events, "validate"))

    ps.destroy_model_parallel(abort=abort)
    assert events == [
        "validate",
        *(["abort"] if abort else []),
        "symm_buffers",
        "a2a_buffers",
        "graphs",
        "te_state",
        "dtensor",
        "memory",
        "destroy",
    ]
    assert ps._global_process_group_list is None


def test_resource_owners_register_teardown():
    callbacks = ps._MODEL_PARALLEL_TEARDOWN_CALLBACKS
    communication = callbacks[ps.TeardownStage.RELEASE_COMMUNICATION]
    # Teardown runs these in reverse: symm buffers, NCCL EP finalize, fused A2A buffers.
    positions = [
        communication.index(fused_a2a.reset_fused_a2a_buffers),
        communication.index(fused_a2a.nccl_ep_finalize),
        communication.index(token_dispatcher._release_nccl_ep_symm_buffers),
    ]
    assert positions == sorted(positions)
    graphs = callbacks[ps.TeardownStage.RELEASE_CUDA_GRAPHS]
    assert cuda_graphs.release_all_cuda_graphs in graphs
    assert FullCudaGraphWrapper.reset_cuda_graph in graphs
    assert OptimizerCudaGraphWrapper.reset_cuda_graph in graphs
    release_pools = gtp_symmetric_memory.deregister_and_clear_gtp_symm_pools
    assert release_pools in communication
    assert ps._MODEL_PARALLEL_TEARDOWN_ABORT_CALLBACKS[release_pools] is release_pools
    assert gtp._destroy_gtp_state in callbacks[ps.TeardownStage.RESET_STATE]


@pytest.mark.parametrize("abort", [False, True])
def test_gtp_pools_release_before_groups(recorded_teardown, monkeypatch, abort):
    events = recorded_teardown
    group = ps._global_process_group_list[1]
    # A caller-owned group is not aborted by model-parallel teardown. Its pool must
    # still be deregistered normally, even when the owned group was aborted.
    external_group = object()
    pools = {"owned": object(), "external": object()}
    monkeypatch.setattr(gtp_symmetric_memory, "_pools", pools.copy())
    monkeypatch.setattr(
        gtp_symmetric_memory, "_registered", {"owned": group, "external": external_group}
    )
    pool_cache = gtp_symmetric_memory.RegisteredLIFOPool()
    pool_cache._free["buffer"] = [object()]
    monkeypatch.setattr(gtp_symmetric_memory, "symmetric_wgrad_pool", pool_cache)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("synchronize"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)

    def deregister(pool, registered_group):
        assert pool_cache._free  # Keep allocations alive until deregistration.
        assert "destroy" not in events
        name = "owned" if registered_group is group else "external"
        assert pool is pools[name]
        events.append(name)

    monkeypatch.setattr(gtp_symmetric_memory.vmm_symm_allocator, "deregister_mem_pool", deregister)
    release = gtp_symmetric_memory.deregister_and_clear_gtp_symm_pools
    ps.register_model_parallel_teardown(
        ps.TeardownStage.RELEASE_COMMUNICATION, release, on_abort=release
    )
    ps.destroy_model_parallel(abort=abort)
    assert events == [
        *(["abort"] if abort else []),
        "synchronize",
        "external",
        *([] if abort else ["owned"]),
        "dtensor",
        "memory",
        "destroy",
    ]
    assert not gtp_symmetric_memory._registered
    assert not gtp_symmetric_memory._pools
    assert not pool_cache._free
    release()  # Explicit cleanup after teardown must also be harmless.
    assert events.count("synchronize") == 1


@pytest.mark.parametrize("abort", [False, True])
def test_gtp_teardown_discards_pending_gradients_and_cache(recorded_teardown, monkeypatch, abort):
    param = Mock()
    param._get_cache_key.return_value = ("old_group",)
    param.flush_accumulated_wgrad.side_effect = AssertionError("used the destroyed group")
    cache = gtp.GTPWeightCache()
    cache.reserve(param, torch.float32, fwd=True)
    monkeypatch.setattr(gtp, "_GTP_CACHE", cache)
    monkeypatch.setattr(gtp, "_GTP_PARAMS", [param])
    monkeypatch.setattr(gtp, "_GTP_PENDING_WGRAD_ACCUM", {param})
    monkeypatch.setattr(gtp, "_inflight_comm_params", {param})
    monkeypatch.setattr(gtp, "_AG_STREAMS", {"old_group": object()})
    monkeypatch.setattr(gtp, "_RS_STREAMS", {"old_group": object()})
    monkeypatch.setattr(gtp, "_wgrad_buf_pool", {"old_group": [object()]})
    monkeypatch.setattr(gtp.GTPShardedParam, "_chain_state", {"old_group": param})
    monkeypatch.setattr(gtp.GTPShardedParam, "_recompute_chain_state", {"old_group": param})
    monkeypatch.setattr(gtp_cuda_graphs, "_GRAPH_WGRAD_RINGS", {"old_group": [object()]})
    monkeypatch.setattr(gtp_cuda_graphs, "_CG_MEMPOOL", object())
    ps.register_model_parallel_teardown(ps.TeardownStage.RESET_STATE, gtp._destroy_gtp_state)

    ps.destroy_model_parallel(abort=abort)

    # A later non-GTP model can reach this fence without constructing a GTP model
    # (which would reset the cursors). It must not flush the previous model's gradient.
    gtp._close_wgrad_accumulation_windows()
    param.flush_accumulated_wgrad.assert_not_called()
    assert param._wgrad_accum_buf is None
    assert not gtp._GTP_PENDING_WGRAD_ACCUM
    assert not gtp._GTP_PARAMS and not gtp._inflight_comm_params
    assert not gtp.GTPShardedParam._chain_state
    assert not gtp.GTPShardedParam._recompute_chain_state
    assert not gtp._AG_STREAMS and not gtp._RS_STREAMS and not gtp._wgrad_buf_pool
    assert not gtp_cuda_graphs._GRAPH_WGRAD_RINGS
    assert gtp_cuda_graphs._CG_MEMPOOL is None
    assert gtp._GTP_CACHE is None
    assert gtp.get_global_GTP_cache() is not cache
    assert not gtp.get_global_GTP_cache()._slots


@pytest.mark.skipif(Utils.world_size < 2, reason="Requires a multi-rank registered pool")
@pytest.mark.parametrize("abort", [False, True])
def test_gtp_registered_pool_and_pending_gradient_across_lifetimes(abort):
    Utils.initialize_distributed()
    ps.destroy_model_parallel()
    baseline = set(torch.distributed.distributed_c10d._world.pg_map)
    try:
        for _ in range(2):
            group = ps.create_group(ranks=list(range(Utils.world_size)), backend="nccl")
            gtp_symmetric_memory.register_gtp_symm_pool(group)
            with gtp_symmetric_memory.gtp_symm_pool_ctx(group):
                value = torch.ones(4, device="cuda")
            torch.distributed.all_reduce(value, group=group)
            torch.cuda.synchronize()

            # A later model's gradient fence must not reach an old group's pending
            # accumulation, even if that later model does not construct GTP layers.
            pending = Mock()
            pending.flush_accumulated_wgrad.side_effect = AssertionError("used the old group")
            gtp._GTP_PENDING_WGRAD_ACCUM.add(pending)
            gtp._GTP_PARAMS.append(pending)
            ps.destroy_model_parallel(abort=abort)
            gtp.wait_for_gtp_grad_reduction_on_current_stream()
            pending.flush_accumulated_wgrad.assert_not_called()

            assert not gtp_symmetric_memory._registered
            assert not gtp_symmetric_memory._pools
            assert not gtp._GTP_PARAMS
            assert set(torch.distributed.distributed_c10d._world.pg_map) == baseline
            # Caller-owned tensors survive; WORLD still supports collectives.
            assert torch.equal(value, torch.full_like(value, Utils.world_size))
            world_value = torch.ones(1, device="cuda")
            torch.distributed.all_reduce(world_value)
            assert world_value.item() == Utils.world_size
            del value, group
    finally:
        ps.destroy_model_parallel(abort=abort)


def test_release_failure_keeps_groups(recorded_teardown, monkeypatch):
    events = recorded_teardown
    stage = ps.TeardownStage

    def finalize_nccl_ep():
        events.append("ep_finalize")
        raise RuntimeError("injected EP finalization failure")

    ps.register_model_parallel_teardown(stage.RELEASE_CUDA_GRAPHS, _recorder(events, "graphs"))
    ps.register_model_parallel_teardown(
        stage.RELEASE_COMMUNICATION, _recorder(events, "a2a_buffers")
    )
    ps.register_model_parallel_teardown(stage.RELEASE_COMMUNICATION, finalize_nccl_ep)
    group = object()
    monkeypatch.setattr(ps, "_MODEL_PARALLEL_GROUP", group)
    groups = ps._global_process_group_list

    with pytest.raises(RuntimeError, match="injected EP finalization failure"):
        ps.destroy_model_parallel()
    # The other releases still ran; the groups and module state were kept.
    assert events == ["ep_finalize", "a2a_buffers", "graphs"]
    assert ps._global_process_group_list is groups
    assert ps._MODEL_PARALLEL_GROUP is group


def test_inprocess_restart_raises_teardown_failure(monkeypatch):
    inprocess = pytest.importorskip("nvidia_resiliency_ext.inprocess")
    from megatron.training import inprocess_restart

    wrapper_kwargs = {}

    def wrapper(**kwargs):
        wrapper_kwargs.update(kwargs)
        return lambda train: train

    monkeypatch.setattr(inprocess, "Wrapper", wrapper)
    monkeypatch.setenv("MASTER_PORT", "29500")
    args = SimpleNamespace(
        inprocess_active_world_size=1,
        inprocess_granularity="rank",
        inprocess_empty_cuda_cache=True,
        async_strategy=None,
        inprocess_heartbeat_interval=30,
        inprocess_heartbeat_timeout=60,
        inprocess_barrier_timeout=120,
        inprocess_completion_timeout=120,
        inprocess_monitor_process_interval=1.0,
        inprocess_monitor_thread_interval=1.0,
        inprocess_last_call_wait=1.0,
        inprocess_soft_timeout=60,
        inprocess_hard_timeout=90,
        inprocess_termination_grace_time=1.0,
    )
    inprocess_restart.inprocess_restart(lambda: None, args)
    finalize = wrapper_kwargs["finalize"]
    torch.cuda.init()
    state = SimpleNamespace(rank=0)

    calls = []
    monkeypatch.setattr(inprocess_restart, "destroy_state", lambda: calls.append("destroy"))
    assert finalize(state) is state
    assert calls == ["destroy"]

    def destroy_state():
        raise RuntimeError("injected teardown failure")

    # ThreadedFinalize alone would drop this error on its thread.
    monkeypatch.setattr(inprocess_restart, "destroy_state", destroy_state)
    with pytest.raises(RuntimeError, match="injected teardown failure"):
        finalize(state)


def test_abort_failure_does_not_enter_graceful_destroy(monkeypatch):
    groups = [Mock(), Mock()]
    groups[1]._get_backend.return_value.abort.side_effect = RuntimeError("injected abort failure")
    monkeypatch.setattr(
        torch.distributed.distributed_c10d, "_world", SimpleNamespace(pg_map=dict.fromkeys(groups))
    )
    monkeypatch.setattr(torch.distributed, "get_backend", lambda group: "nccl")
    monkeypatch.setattr(ps, "_global_process_group_list", [None, *groups])
    with pytest.raises(RuntimeError, match="injected abort failure"):
        ps._abort_created_process_groups()
    for group in groups:
        group._get_backend.return_value.abort.assert_called_once()


def test_hybridep_rebuilds_buffer_for_new_group(monkeypatch):
    created = []

    class Buffer:
        def __init__(self, group, **kwargs):
            self.group = group
            created.append(self)

        def dispatch_with_permute(self, **kwargs):
            return None, None, None, None, object()

    monkeypatch.setattr(fused_a2a, "HybridEPBuffer", Buffer, raising=False)
    monkeypatch.setattr(fused_a2a, "_hybrid_ep_buffer", None)
    monkeypatch.setattr(fused_a2a, "_buffer", None)
    first_group, second_group = object(), object()
    routing_map = torch.ones(2, 1, dtype=torch.bool)
    for group in (first_group, first_group, second_group):
        fused_a2a.HybridEPDispatch.forward(
            SimpleNamespace(), torch.ones(2, 4), routing_map, None, group, 1
        )
        assert fused_a2a._hybrid_ep_buffer.group is group
    assert [buffer.group for buffer in created] == [first_group, second_group]
    fused_a2a.reset_fused_a2a_buffers()
    assert fused_a2a._hybrid_ep_buffer is None
