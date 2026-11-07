# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""User-buffer registration with shared dense/expert groups and independent AG groups."""

from unittest.mock import Mock

import pytest
import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp import param_and_grad_buffer as buffers


@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_user_buffers_register_each_communicator_once(monkeypatch, manual, shared):
    dense = Mock(group_desc="dense")
    expert = dense if shared else Mock(group_desc="expert")
    dense_ag, expert_ag, outer = [
        Mock(group_desc=name) for name in ("dense_ag", "expert_ag", "outer")
    ]
    groups = {
        (False, False): dense,
        (True, False): expert,
        (False, True): dense_ag,
        (True, True): expert_ag,
    }
    index = Mock()
    index.get_fsdp_group.side_effect = (
        lambda is_expert_parallel=False, independent_all_gather=False: groups[
            is_expert_parallel, independent_all_gather
        ]
    )
    index.get_outer_fsdp_group.return_value = outer
    allocator = Mock()
    monkeypatch.setattr(buffers, "nccl_allocator", allocator)
    monkeypatch.setattr(buffers, "NCCL_ALLOCATOR", "MCORE")
    monkeypatch.setattr(buffers, "NCCL_MEMORY_POOL", None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "barrier", Mock())
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())
    monkeypatch.setattr(buffers, "log_single_rank", Mock())
    monkeypatch.setattr(buffers, "_get_parameter_groups", lambda *args: ([], {}, {}))
    for name in (
        "_init_each_parameter_group_buffers",
        "_init_distributed_params",
        "_init_optimizer_named_parameters",
        "_log_parameter_groups",
    ):
        monkeypatch.setattr(buffers.ParamAndGradBuffer, name, Mock())
    config = buffers.DistributedDataParallelConfig(
        nccl_ub=True, fsdp_manual_registration=manual, disable_symmetric_registration=True
    )
    buffer = buffers.ParamAndGradBuffer(
        config, torch.nn.Module(), buffers.BucketingPolicy(), index, device=torch.device("cpu")
    )
    expected = (
        [dense, dense_ag, expert_ag, outer]
        if shared
        else [dense, expert, dense_ag, expert_ag, outer]
    )
    assert buffer.ubr_groups == expected
    if manual:
        buffer.manual_buffer_registration()
        assert [call.args[1] for call in allocator.register_mem_pool.call_args_list] == expected
    else:
        buffer.mem_alloc_context()
        assert allocator.MultiGroupMemPoolAllocator.call_args.kwargs["groups"] == expected
