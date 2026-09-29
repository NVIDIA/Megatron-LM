# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Optimizer restoration preserves FP32 views and reports model/master disagreements."""

import logging
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

pytestmark = pytest.mark.internal


def _optimizer_for_parameter(model_param):
    """Supply the parameter/state tables consumed by the production load method."""
    main_param = model_param.view(-1)[2:6]
    if model_param.dtype != torch.float32:
        main_param = main_param.detach().float().clone()
    state = {"exp_avg": torch.zeros_like(main_param), "exp_avg_sq": torch.zeros_like(main_param)}
    optimizer = SimpleNamespace(
        config=SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=False),
        model_param_group_index_map={model_param: (0, 0)},
        optimizer=SimpleNamespace(
            param_groups=[{"params": [main_param]}], state={main_param: state}
        ),
    )
    return optimizer, main_param, state


@pytest.mark.parametrize("source_dtype", [torch.float32, torch.float64])
def test_load_fp32_optimizer_shard_preserves_autograd_and_other_model_rows(source_dtype):
    """Legacy load has no warning counter and copies into a live leaf-parameter view."""
    model_param = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
    optimizer, main_param, state = _optimizer_for_parameter(model_param)
    assert main_param.requires_grad and main_param._base is model_param
    tensors = {
        "param": torch.arange(4, dtype=source_dtype) + 100,
        "exp_avg": torch.arange(4, dtype=source_dtype) + 200,
        "exp_avg_sq": torch.arange(4, dtype=source_dtype) + 300,
    }
    DistributedOptimizer._set_main_param_and_optimizer_states(optimizer, model_param, tensors)
    torch.testing.assert_close(model_param[:2], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(model_param[6:], torch.tensor([6.0, 7.0]))
    torch.testing.assert_close(main_param, tensors["param"].float())
    for key in state:
        torch.testing.assert_close(state[key], tensors[key].float())
    model_param.sum().backward()
    torch.testing.assert_close(model_param.grad, torch.ones_like(model_param))
    assert not hasattr(optimizer, "_loaded_master_mismatch")


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
@pytest.mark.parametrize("subgroup", [False, True])
def test_master_warning_reduces_within_optimizer_group(
    cpu_default_process_group, caplog, mismatch, subgroup
):
    """Report on the optimizer's group leader even when global rank zero is absent."""
    rank, world_size = dist.get_rank(), dist.get_world_size()
    ranks = list(range(world_size // 2 if subgroup else 0, world_size))
    group = dist.new_group(ranks=ranks)
    if rank not in ranks:
        # These ranks model a pipeline stage without this optimizer.
        dist.barrier()
        return
    try:
        device = (
            torch.device('cuda', torch.cuda.current_device())
            if dist.get_backend(group) == 'nccl'
            else torch.device('cpu')
        )
        optimizer = SimpleNamespace(
            _loaded_master_mismatch=torch.tensor(
                int(mismatch and rank == world_size - 1), device=device
            ),
            data_parallel_group=group,
        )
        with caplog.at_level(logging.WARNING):
            DistributedOptimizer._warn_if_master_disagrees_with_model_weights(optimizer)
            # The counter is consumed; the second invocation must be inert.
            DistributedOptimizer._warn_if_master_disagrees_with_model_weights(optimizer)
        assert optimizer._loaded_master_mismatch is None
        messages = [
            record.getMessage()
            for record in caplog.records
            if "optimizer master that disagrees" in record.getMessage()
        ]
        assert len(messages) == int(mismatch and rank == ranks[0])
        if messages:
            assert "1 fp32 parameter shards" in messages[0]
    finally:
        dist.destroy_process_group(group)
        dist.barrier()
