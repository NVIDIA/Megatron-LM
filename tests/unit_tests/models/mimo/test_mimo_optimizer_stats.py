# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for heterogeneous MIMO optimizer statistics."""

from types import SimpleNamespace
from unittest import mock

import torch

from megatron.core.models.mimo.optimizer import (
    MimoOptimizer,
    ModuleOptimizerInfo,
    get_mimo_optimizer,
)


def test_num_zeros_uses_world_max_per_module_then_sums():
    encoder_optimizer = SimpleNamespace(count_zeros=mock.Mock(return_value=4))
    optimizer = MimoOptimizer(
        module_infos={
            "encoder": ModuleOptimizerInfo(encoder_optimizer, None, None, True),
            "language": ModuleOptimizerInfo(None, None, None, False),
        },
        config=SimpleNamespace(),
    )
    original_zeros = torch.zeros

    def cpu_zeros(*args, **kwargs):
        kwargs["device"] = "cpu"
        return original_zeros(*args, **kwargs)

    def world_max(values, op):
        assert op is torch.distributed.ReduceOp.MAX
        values.copy_(torch.tensor([4, 9], dtype=values.dtype))

    with (
        mock.patch.object(torch, "zeros", side_effect=cpu_zeros),
        mock.patch.object(torch.distributed, "all_reduce", side_effect=world_max) as all_reduce,
    ):
        assert optimizer.count_zeros() == 13
    all_reduce.assert_called_once()
    encoder_optimizer.count_zeros.assert_called_once_with()


def test_get_mimo_optimizer_reuses_active_wrapped_module_pg_collection():
    active_pg_collection = mock.sentinel.active_pg_collection
    active_module = SimpleNamespace(
        module=SimpleNamespace(pg_collection=active_pg_collection),
        ddp_config=SimpleNamespace(num_distributed_optimizer_instances=1),
    )
    active_grid = mock.Mock()
    active_grid.is_current_rank_in_grid.return_value = True
    active_grid.get_pg.side_effect = AssertionError("must not reconstruct process groups")
    inactive_grid = mock.Mock()
    inactive_grid.is_current_rank_in_grid.return_value = False
    inactive_grid.get_pg.side_effect = AssertionError("must not reconstruct process groups")
    mimo_model = SimpleNamespace(
        mimo_config=SimpleNamespace(
            module_to_grid_map={"encoder": active_grid, "language": inactive_grid}
        ),
        language_model=None,
        modality_submodules={"encoder": active_module},
    )
    config = SimpleNamespace()
    module_optimizer = mock.sentinel.module_optimizer

    with mock.patch(
        "megatron.core.optimizer.get_megatron_optimizer", return_value=module_optimizer
    ) as optimizer_factory:
        optimizer = get_mimo_optimizer(mimo_model, config)

    optimizer_factory.assert_called_once_with(
        config=config,
        model_chunks=[active_module],
        pg_collection=active_pg_collection,
        use_gloo_process_groups=False,
    )
    assert optimizer.module_infos["encoder"].optimizer is module_optimizer
    assert optimizer.module_infos["encoder"].pg_collection is active_pg_collection
    assert optimizer.module_infos["language"].optimizer is None
    assert optimizer.module_infos["language"].pg_collection is None
    active_grid.get_pg.assert_not_called()
    inactive_grid.get_pg.assert_not_called()
