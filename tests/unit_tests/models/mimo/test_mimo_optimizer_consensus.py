# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Distributed test for MimoOptimizer cross-grid step-success consensus."""

import pytest
import torch

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
