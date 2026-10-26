# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Keep zero-token ranks in fused MoE forward/backward and optimizer collectives."""

import os
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist

from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.moe import test_scheduler_bf16_layer_parity as parity

pytestmark = pytest.mark.launch_on_gb200


@pytest.fixture(scope="module")
def distributed_world():
    Utils.set_world_size(world_size=int(os.environ["WORLD_SIZE"]), rank=int(os.environ["RANK"]))
    Utils.initialize_distributed()
    return dist.group.WORLD


def test_fused_experts_with_empty_ranks(distributed_world, monkeypatch):
    assert dist.get_world_size() == 4
    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
    monkeypatch.setenv("NVTE_DISABLE_CUTEDSL_WGRAD_FUSED_GROUPED_MLP", "1")
    original = parity._config
    monkeypatch.setattr(
        parity, "_config", lambda *args: replace(original(*args), moe_router_mock_maxvio=4)
    )
    Utils.initialize_model_parallel(expert_model_parallel_size=4)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    x = torch.randn(256, 1, 256, device="cuda", dtype=torch.bfloat16)
    try:
        _, observed, dispatch = parity._run(None, pg, None, [x], [torch.ones_like(x) * 0.01])
        assert not dispatch[0]["overflow"]
        assert dispatch[0]["logical_assignments_per_rank"] == [2048, 0, 0, 0]
        assert observed["forward/0"].shape == x.shape
        assert torch.isfinite(observed["dgrad/0"]).all()
        if dist.get_rank() != 0:
            for name, value in observed.items():
                if name.startswith("wgrad/experts."):
                    assert torch.count_nonzero(value) == 0, name
    finally:
        Utils.destroy_model_parallel()
