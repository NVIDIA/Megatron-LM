# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core import parallel_state as ps
from megatron.core.full_cuda_graph import FullCudaGraphWrapper
from megatron.core.optimizer.optimizer_cuda_graph import OptimizerCudaGraphWrapper
from megatron.core.transformer import cuda_graphs
from megatron.core.transformer.moe import fused_a2a, token_dispatcher

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
    for group in (first_group, first_group, second_group):
        fused_a2a.HybridEPDispatch.forward(
            SimpleNamespace(), torch.ones(2, 4), None, None, group, 1
        )
        assert fused_a2a._hybrid_ep_buffer.group is group
    assert [buffer.group for buffer in created] == [first_group, second_group]
    fused_a2a.reset_fused_a2a_buffers()
    assert fused_a2a._hybrid_ep_buffer is None
