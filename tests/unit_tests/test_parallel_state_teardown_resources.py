# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core import parallel_state as ps

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


@pytest.mark.parametrize("abort", [False, True])
@pytest.mark.parametrize("release_context_fails", [False, True])
def test_abort_precedes_resource_cleanup(monkeypatch, abort, release_context_fails):
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
    monkeypatch.setattr(ps, "is_torch_min_version", lambda version: True)
    monkeypatch.setattr(ps.SymmetricMemoryManager, "destroy", lambda: events.append("memory"))
    monkeypatch.setattr(ps, "_clear_dtensor_sharding_cache", lambda: None)
    monkeypatch.delitem(sys.modules, "transformer_engine.pytorch", raising=False)
    for name in (
        "megatron.core.transformer.cuda_graphs",
        "megatron.core.full_cuda_graph",
        "megatron.core.optimizer.optimizer_cuda_graph",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)

    def release_context():
        events.append("ep_context")
        if release_context_fails:
            raise RuntimeError("injected EP finalization failure")

    fused = SimpleNamespace(reset_fused_a2a_buffers=lambda: events.append("buffers"))
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.moe.fused_a2a", fused)
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.transformer.moe.token_dispatcher",
        SimpleNamespace(nccl_ep_release_context=release_context),
    )
    ps.destroy_model_parallel(abort=abort)
    assert events == (["abort"] if abort else []) + ["ep_context", "buffers", "memory", "destroy"]
    assert ps._global_process_group_list is None


def test_abort_failure_does_not_enter_graceful_destroy(monkeypatch):
    groups = [Mock(), Mock()]
    groups[1]._get_backend.return_value.abort.side_effect = RuntimeError("injected abort failure")
    monkeypatch.setattr(
        torch.distributed.distributed_c10d, "_world", SimpleNamespace(pg_map=dict.fromkeys(groups))
    )
    monkeypatch.setattr(torch.distributed, "get_backend", lambda group: "nccl")
    monkeypatch.setattr(ps, "_global_process_group_list", [None, *groups])
    monkeypatch.setattr(ps, "is_torch_min_version", lambda version: True)
    with pytest.raises(RuntimeError, match="injected abort failure"):
        ps._abort_created_process_groups()
    for group in groups:
        group._get_backend.return_value.abort.assert_called_once()


def test_hybridep_rebuilds_buffer_for_new_group(monkeypatch):
    from megatron.core.transformer.moe import fused_a2a

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
