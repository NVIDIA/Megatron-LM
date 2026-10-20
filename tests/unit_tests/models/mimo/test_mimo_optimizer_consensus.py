# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Distributed test for MimoOptimizer cross-grid step-success consensus."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.optimizer import MimoOptimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires >= 2 ranks.")
def test_step_success_is_world_min():
    """One rank's failed update must propagate to every rank via the MIN reduction."""
    Utils.initialize_distributed()
    try:
        opt = MimoOptimizer(module_infos={}, config=OptimizerConfig(log_num_zeros_in_grad=False))
        last_rank = torch.distributed.get_world_size() - 1
        opt.step_with_ready_grads = lambda: torch.distributed.get_rank() != last_rank
        success, _, _ = opt.step()
        assert success is False
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires >= 2 ranks.")
def test_param_norm_combines_disjoint_module_grids(monkeypatch):
    """Parameter logging sums module norms without reading global model-parallel state."""
    from examples.mimo.training.topology import ModuleGridSpec, create_topology
    from megatron.training.utils import common_utils

    Utils.initialize_distributed()
    topology = None
    try:
        world_size = torch.distributed.get_world_size()
        if world_size < 2 or world_size % 2:
            pytest.skip("Requires an even distributed world size >= 2")
        module_size = world_size // 2
        encoder_name = "encoder"
        topology = create_topology(
            [
                ModuleGridSpec(name=encoder_name, num_ranks=module_size, tp=module_size),
                ModuleGridSpec(
                    name=MIMO_LANGUAGE_MODULE_KEY,
                    num_ranks=module_size,
                    tp=module_size,
                    rank_offset=module_size,
                ),
            ]
        )

        model = torch.nn.Module()
        model.mimo_config = SimpleNamespace(module_to_grid_map=topology.grids)
        model.language_model = None
        model.modality_submodules = torch.nn.ModuleDict()

        active_names = list(topology.schedule_pg_collection.keys())
        assert len(active_names) == 1
        active_name = active_names[0]
        active_module = torch.nn.Linear(2, 2, bias=False, device="cuda")
        active_module.weight.data.fill_(1.0 if active_name == encoder_name else 2.0)
        if active_name == MIMO_LANGUAGE_MODULE_KEY:
            model.language_model = active_module
        else:
            model.modality_submodules[active_name] = active_module

        monkeypatch.setattr(
            common_utils, "get_args", lambda: SimpleNamespace(bf16=False, use_megatron_fsdp=False)
        )
        assert common_utils.calc_params_l2_norm(
            model, pg_collection=topology.schedule_pg_collection
        ) == pytest.approx(20**0.5)
    finally:
        if topology is not None:
            topology.destroy()
        Utils.destroy_model_parallel()
