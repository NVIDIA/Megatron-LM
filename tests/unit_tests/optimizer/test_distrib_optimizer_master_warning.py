# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Checkpoint model/master disagreement regressions with Gloo or NCCL collectives."""

import logging
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from tests.unit_tests.optimizer.test_distrib_optimizer_fp32_load import _optimizer_for_parameter

pytestmark = pytest.mark.internal


@pytest.mark.parametrize("model_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mismatch", [False, True])
def test_master_mismatch_counts_fp32_shards_before_copy(model_dtype, mismatch):
    model_param = torch.nn.Parameter(torch.arange(8, dtype=model_dtype))
    optimizer, main_param, state = _optimizer_for_parameter(model_param)
    optimizer._loaded_master_mismatch = torch.zeros((), dtype=torch.long)
    master = main_param.detach().clone() + (100 if mismatch else 0)
    tensors = {"param": master, **{key: torch.full_like(value, 3) for key, value in state.items()}}
    DistributedOptimizer._set_main_param_and_optimizer_states(optimizer, model_param, tensors)
    expected_count = int(model_dtype == torch.float32 and mismatch)
    assert optimizer._loaded_master_mismatch.item() == expected_count
    torch.testing.assert_close(main_param, master)
    # A repeated identical load is not another disagreement.
    DistributedOptimizer._set_main_param_and_optimizer_states(optimizer, model_param, tensors)
    assert optimizer._loaded_master_mismatch.item() == expected_count


@pytest.mark.parametrize("mismatch", [False, True])
def test_master_warning_reduces_across_all_ranks(cpu_default_process_group, caplog, mismatch):
    """A mismatch present only on the last rank still reaches the rank-zero warning."""
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = (
        torch.device('cuda', torch.cuda.current_device())
        if dist.get_backend() == 'nccl'
        else torch.device('cpu')
    )
    optimizer = SimpleNamespace(
        _loaded_master_mismatch=torch.tensor(
            int(mismatch and rank == world_size - 1), device=device
        )
    )
    with caplog.at_level(logging.WARNING):
        DistributedOptimizer._warn_if_master_disagrees_with_model_weights(optimizer)
        # The counter is consumed; this second invocation must be an inert operation.
        DistributedOptimizer._warn_if_master_disagrees_with_model_weights(optimizer)
    assert optimizer._loaded_master_mismatch is None
    messages = [
        record.getMessage()
        for record in caplog.records
        if "optimizer master that disagrees" in record.getMessage()
    ]
    assert len(messages) == int(mismatch and rank == 0)
    if messages:
        assert "1 fp32 parameter shards" in messages[0]
